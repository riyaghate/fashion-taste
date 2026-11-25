"""Step 5: zero-shot evaluation of a vision-language model (Gemini) with 4 prompting strategies.

  direct  - "rate how well these items go together, 1-10"
  reason  - explain the reasoning briefly first, then rate
  rubric  - the prompt names the four compatibility rules (color, pattern, formality, season)
  fewshot - three labelled example outfits from the TRAINING split are shown first

Every outfit is shown as one numbered collage image. Fill-in-the-blank shows the partial
outfit plus a second image with candidates A-D.

Free-tier friendly: requests are paced (--rpm), every answer is cached to
<work>/vlm_cache/<strategy>.jsonl, and re-running the same command resumes where it stopped
(e.g. the next day once the daily quota resets).

    export GEMINI_API_KEY=...       (on Colab: Secrets panel, name GEMINI_API_KEY)
    python -m fashion_taste.vlm --strategies direct --data $FT_DATA --work $FT_WORK
"""
import argparse
import json
import os
import random
import re
import time

import pandas as pd

from . import data, tasks as T
from .config import SEED, add_path_args, work_paths
from .viz import collage, to_jpeg_bytes

STRATEGIES = ["direct", "reason", "rubric", "fewshot"]

RATE_BASE = ("The image shows the items of one outfit (numbered). Rate how well these items go together "
             "as a single outfit someone would wear, on a scale from 1 (they clearly do not go together) "
             "to 10 (they go together very well).")
RUBRIC = ("Judge the outfit by checking four things: (1) COLOR - do the colors work together or clash? "
          "(2) PATTERN - are prints/patterns balanced, or do several busy patterns compete? "
          "(3) FORMALITY - are all items at a similar level of dressiness? "
          "(4) SEASON - do the items suit the same weather/season?")
FITB_BASE = ("The first image shows an incomplete outfit (numbered items). The second image shows four "
             "candidate items labelled A, B, C and D. Exactly one candidate completes the outfit best. "
             "Which one?")

JSON_SCORE = 'Respond only with JSON: {"score": <integer 1-10>}'
JSON_SCORE_REASON = 'Respond only with JSON: {"reasoning": "<at most 60 words>", "score": <integer 1-10>}'
JSON_CHOICE = 'Respond only with JSON: {"choice": "<A, B, C or D>"}'
JSON_CHOICE_REASON = 'Respond only with JSON: {"reasoning": "<at most 60 words>", "choice": "<A, B, C or D>"}'


def rating_prompt(strategy):
    if strategy == "direct":
        return f"{RATE_BASE}\n{JSON_SCORE}"
    if strategy == "reason":
        return f"{RATE_BASE}\nFirst think briefly about what works and what does not, then give the score.\n{JSON_SCORE_REASON}"
    if strategy == "rubric":
        return f"{RATE_BASE}\n{RUBRIC}\n{JSON_SCORE}"
    return f"Now the outfit to rate. {RATE_BASE}\n{JSON_SCORE}"


def fitb_prompt(strategy):
    if strategy == "direct":
        return f"{FITB_BASE}\n{JSON_CHOICE}"
    if strategy == "reason":
        return f"{FITB_BASE}\nFirst think briefly about each candidate, then choose.\n{JSON_CHOICE_REASON}"
    if strategy == "rubric":
        return f"{FITB_BASE}\n{RUBRIC.replace('Judge the outfit', 'Judge each candidate')}\n{JSON_CHOICE}"
    return f"Now the question to answer. {FITB_BASE}\n{JSON_CHOICE}"


def parse(text, want):
    text = text or ""
    try:
        obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        if want == "score":
            v = float(obj["score"])
            return v if 1 <= v <= 10 else None
        c = str(obj["choice"]).strip().upper()[:1]
        return "ABCD".index(c) if c in "ABCD" and c else None
    except Exception:
        pass
    if want == "score":
        m = re.findall(r"\b(10|[1-9])\b", text)
        return float(m[-1]) if m else None
    m = re.findall(r"\b([ABCD])\b", text.upper())
    return "ABCD".index(m[-1]) if m else None


class Gemini:
    def __init__(self, model, api_key, rpm, temperature=0.0):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model, self.gap, self.temp = model, 60.0 / rpm, temperature
        self.last = 0.0

    def __call__(self, parts):
        """parts: list of str or PIL images. Returns response text. Raises QuotaExhausted."""
        contents = [self.types.Part.from_bytes(data=to_jpeg_bytes(p), mime_type="image/jpeg")
                    if not isinstance(p, str) else p for p in parts]
        cfg = self.types.GenerateContentConfig(temperature=self.temp, response_mime_type="application/json")
        for attempt in range(6):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=cfg).text
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    if "per day" in msg.lower() or "PerDay" in msg or attempt >= 4:
                        raise QuotaExhausted(msg[:300])
                    time.sleep(30 * (attempt + 1))
                elif "500" in msg or "503" in msg or "UNAVAILABLE" in msg or "deadline" in msg.lower():
                    time.sleep(10 * (attempt + 1))
                else:
                    return f"ERROR: {msg[:300]}"
        return "ERROR: retries exhausted"


class QuotaExhausted(Exception):
    pass


def fewshot_examples(ds, seed=SEED, cell=200):
    """2 real + 1 random outfit from the TRAINING split (no test leakage)."""
    rng = random.Random(seed + 7)
    tr = ds.compatibility("train")
    good = rng.sample([t for t in tr if t[0] == 1 and 3 <= len(t[1]) <= 5], 2)
    bad = rng.sample([t for t in tr if t[0] == 0 and 3 <= len(t[1]) <= 5], 1)
    parts = ["Here are example outfits to calibrate your judgement."]
    for lab, items, _ in [good[0], bad[0], good[1]]:
        parts.append(collage([ds.image_path(i) for i in items], labels=list(range(1, len(items) + 1)), cell=cell))
        parts.append("Example of an outfit people consider well put together (high score)." if lab == 1 else
                     "Example of items that were randomly combined and do not form a coherent outfit (low score).")
    fq = ds.fitb("train")
    q = dict(rng.choice([x for x in fq if len(x["question"]) <= 5]))
    order = list(range(4))
    while order.index(q["correct"]) == 0:  # example answer not in slot A (avoid position bias)
        rng.shuffle(order)
    q["answers"], q["correct"] = [q["answers"][k] for k in order], order.index(q["correct"])
    fitb_parts = ["Here is an example question with its answer.",
                  collage([ds.image_path(i) for i in q["question"]], labels=list(range(1, len(q["question"]) + 1)), cell=cell),
                  collage([ds.image_path(i) for i in q["answers"]], labels=list("ABCD"), cell=cell),
                  f'Correct answer: {{"choice": "{"ABCD"[q["correct"]]}"}}']
    return parts, fitb_parts


def build_parts(ds, task, strategy, shots, cell=224):
    if task["task"] == "fitb":
        q = collage([ds.image_path(i) for i in task["question"]], labels=list(range(1, len(task["question"]) + 1)), cell=cell)
        a = collage([ds.image_path(i) for i in task["answers"]], labels=list("ABCD"), cell=cell)
        pre = shots[1] if strategy == "fewshot" else []
        return pre + [q, a, fitb_prompt(strategy)], "choice"
    im = collage([ds.image_path(i) for i in task["items"]], labels=list(range(1, len(task["items"]) + 1)), cell=cell)
    pre = shots[0] if strategy == "fewshot" else []
    return pre + [im, rating_prompt(strategy)], "score"


def load_cache(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["key"]] = r
            except json.JSONDecodeError:
                continue
    return done


def cache_to_csv(P, strategy):
    done = load_cache(P["vlm"] / f"{strategy}.jsonl")
    rows = [{"task": r["task"], "id": r["id"], "score": r.get("score"), "choice": r.get("choice"),
             "parsed": r.get("score") is not None or r.get("choice") is not None} for r in done.values()]
    df = pd.DataFrame(rows)
    df.to_csv(P["results"] / f"scores_gemini_{strategy}.csv", index=False)
    return df


def eval_tasks(ds, P, n_compat, n_fitb, n_hn):
    all_t = T.compat_tasks(ds) + T.fitb_tasks(ds)
    hn_path = P["results"] / "hard_negatives.json"
    if hn_path.exists():
        all_t += T.hn_tasks(json.loads(hn_path.read_text()))
    man = P["survey"] / "manifest.csv"
    if man.exists():
        all_t += T.survey_tasks(man)
    return T.subset(all_t, n_compat, n_fitb, n_hn)


def run(ds, P, strategy, client, tasks, max_calls=None):
    path = P["vlm"] / f"{strategy}.jsonl"
    done = load_cache(path)
    todo = [t for t in T.interleave(list(tasks)) if f"{t['task']}:{t['id']}" not in done]
    print(f"[{strategy}] {len(done)} cached, {len(todo)} to go")
    shots = fewshot_examples(ds) if strategy == "fewshot" else ([], [])
    from tqdm.auto import tqdm
    calls = 0
    for t in tqdm(todo, desc=strategy):
        if max_calls and calls >= max_calls:
            print(f"reached --max_calls {max_calls}; re-run to continue")
            break
        parts, want = build_parts(ds, t, strategy, shots)
        try:
            text = client(parts)
        except QuotaExhausted as e:
            print(f"\nQuota exhausted ({e}).\nProgress is saved - re-run this cell later (e.g. tomorrow) to continue.")
            break
        calls += 1
        val = parse(text, want)
        line = json.dumps({"key": f"{t['task']}:{t['id']}", "task": t["task"], "id": t["id"],
                           want: val, "raw": text}) + "\n"
        # open-write-fsync-close for EVERY answer: Colab's Drive mount only uploads reliably when a
        # file is closed, so a file kept open is lost if the session is cut off.
        with path.open("a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    df = cache_to_csv(P, strategy)
    if len(df):
        print(f"[{strategy}] answered {len(df)}; unparseable: {(~df['parsed']).sum()}")


def main():
    p = add_path_args(argparse.ArgumentParser())
    p.add_argument("--strategies", default="direct", help=f"comma list from {STRATEGIES}")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                   help="any vision-capable Gemini model name from Google AI Studio")
    p.add_argument("--rpm", type=float, default=8, help="requests per minute (stay under your quota)")
    p.add_argument("--max_calls", type=int, default=None, help="stop after this many calls this run")
    p.add_argument("--n_compat", type=int, default=300)
    p.add_argument("--n_fitb", type=int, default=200)
    p.add_argument("--n_hn", type=int, default=75, help="hard-negative pairs per rule (and controls)")
    p.add_argument("--dry_run", action="store_true", help="save example prompts as images, make no calls")
    a = p.parse_args()
    P = work_paths(a.work)
    ds = data.load(a.data, a.split)
    tasks = eval_tasks(ds, P, a.n_compat, a.n_fitb, a.n_hn)
    print(f"{len(tasks)} prompts per strategy: " + ", ".join(
        f"{k}={v}" for k, v in pd.Series([t['task'] for t in tasks]).value_counts().items()))
    strategies = [s.strip() for s in a.strategies.split(",")]
    if a.dry_run:
        for s in strategies:
            shots = fewshot_examples(ds) if s == "fewshot" else ([], [])
            for t in [next(x for x in tasks if x["task"] == k) for k in ("compat", "fitb")]:
                parts, _ = build_parts(ds, t, s, shots)
                d = P["vlm"] / "dry_run" / f"{s}_{t['task']}"
                d.mkdir(parents=True, exist_ok=True)
                for k, part in enumerate(parts):
                    (d / f"{k:02d}.txt").write_text(part) if isinstance(part, str) else part.save(d / f"{k:02d}.jpg")
        print(f"example prompts written to {P['vlm'] / 'dry_run'}")
        return
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY first")
    client = Gemini(a.model, key, a.rpm)
    for s in strategies:
        assert s in STRATEGIES, s
        run(ds, P, s, client, tasks, a.max_calls)


if __name__ == "__main__":
    main()
