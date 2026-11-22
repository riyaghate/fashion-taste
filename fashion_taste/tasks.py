"""The evaluation tasks, defined once so every method is scored on identical examples.

  compat  - real vs random outfit (label 1/0); metric: AUC
  fitb    - pick the missing item out of 4; metric: accuracy
  hn      - single-rule hard negatives: original vs modified outfit; metric: pairwise
            accuracy = P(score(original) > score(modified)); for controls 0.5 is ideal
  survey  - the outfits shown to human raters (from survey/manifest.csv)

`subset()` draws the fixed, seeded subset used for the (rate-limited) Gemini runs; local
CLIP models are reported on both the subset (apples-to-apples) and the full test set.
"""
import json
import random

import pandas as pd

from .config import SEED


def compat_tasks(ds):
    return [dict(task="compat", id=qid, items=items, label=label) for label, items, qid in ds.compatibility("test")]


def fitb_tasks(ds):
    """Answer order is shuffled with a fixed per-question seed: in the released files the correct
    answer is usually listed first, which would reward a VLM that always says "A"."""
    out = []
    for q in ds.fitb("test"):
        order = list(range(len(q["answers"])))
        random.Random(f"{SEED}-{q['qid']}").shuffle(order)
        out.append(dict(task="fitb", id=q["qid"], question=q["question"],
                        answers=[q["answers"][k] for k in order], correct=order.index(q["correct"])))
    return out


def hn_tasks(hn_records):
    out = []
    for r in hn_records:
        out.append(dict(task="hn", id=r["hn_id"] + "|orig", pair=r["hn_id"], rule=r["rule"], side="orig", items=r["original"]))
        out.append(dict(task="hn", id=r["hn_id"] + "|mod", pair=r["hn_id"], rule=r["rule"], side="mod", items=r["modified"]))
    return out


def survey_tasks(manifest_csv):
    m = pd.read_csv(manifest_csv)
    m = m[m["kind"] != "repeat"]
    return [dict(task="survey", id=r.survey_id, items=json.loads(r.items)) for r in m.itertuples()]


def subset(tasks, n_compat=300, n_fitb=200, n_hn_pairs_per_rule=75, seed=SEED):
    rng = random.Random(seed)
    by = {}
    for t in tasks:
        by.setdefault(t["task"], []).append(t)
    out = []
    if "compat" in by:  # balanced real/random
        pos = [t for t in by["compat"] if t["label"] == 1]
        neg = [t for t in by["compat"] if t["label"] == 0]
        out += rng.sample(pos, min(n_compat // 2, len(pos))) + rng.sample(neg, min(n_compat // 2, len(neg)))
    if "fitb" in by:
        out += rng.sample(by["fitb"], min(n_fitb, len(by["fitb"])))
    if "hn" in by:
        pairs = {}
        for t in by["hn"]:
            pairs.setdefault((t["rule"], t["pair"]), []).append(t)
        for rule in sorted({r for r, _ in pairs}):
            keys = sorted(k for k in pairs if k[0] == rule)
            for k in rng.sample(keys, min(n_hn_pairs_per_rule, len(keys))):
                out += pairs[k]
    out += by.get("survey", [])
    return out


def interleave(tasks):
    """Round-robin across task types, so a run cut short still covers every task."""
    by = {}
    for t in tasks:
        by.setdefault(t["task"], []).append(t)
    # keep hard-negative pairs adjacent so partial runs give complete pairs
    queues = list(by.values())
    out, i = [], 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            out.append(q.pop(0))
            if out[-1]["task"] == "hn" and q and q[0].get("pair") == out[-1]["pair"]:
                out.append(q.pop(0))
        i += 1
    return out
