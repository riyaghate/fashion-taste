"""Smoke-test helpers: run the Gemini step with a fake client, and fake survey/validation answers."""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fashion_taste import data, vlm  # noqa: E402
from fashion_taste.config import work_paths  # noqa: E402


class FakeGemini:
    def __call__(self, parts):
        assert any(not isinstance(p, str) for p in parts), "no image sent"
        prompt = parts[-1]
        if "choice" in prompt:
            return json.dumps({"reasoning": "ok", "choice": random.choice("ABCD")})
        return json.dumps({"reasoning": "ok", "score": random.randint(1, 10)})


def main(data_dir, work):
    random.seed(0)
    P = work_paths(work)
    ds = data.load(data_dir)
    tasks = vlm.eval_tasks(ds, P, 40, 30, 10)
    for s in vlm.STRATEGIES:
        vlm.run(ds, P, s, FakeGemini(), tasks)
    vlm.run(ds, P, "direct", FakeGemini(), tasks)  # resume: nothing left to do

    man = pd.read_csv(P["survey"] / "manifest.csv")
    nq = len(man)
    rows = []
    for r in range(25):
        row = {"Timestamp": "x", "How interested are you in fashion?": random.randint(1, 5)}
        for q in range(1, nq + 1):
            kind = man.loc[man["question"] == q, "kind"].item()
            base = {"real": 4, "random": 2, "hard_negative": 3, "repeat": 4}[kind]
            row[f"Outfit {q}: how well do these items go together?"] = min(5, max(1, base + random.randint(-1, 1)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(P["survey"] / "responses.csv", index=False)

    key = pd.read_csv(P["validation"] / "validation_key_DO_NOT_OPEN_UNTIL_DONE.csv")
    for name, noise in (("alex", 0.2), ("sam", 0.3)):
        labs = [("none" if r == "control" else r) if random.random() > noise else random.choice(["none", "multiple", "color"])
                for r in key["rule"]]
        pd.DataFrame({"case": key["case"], "label": labs, "notes": ""}).to_csv(
            P["validation"] / f"validation_labels_{name}.csv", index=False)
    print("fake Gemini answers, survey responses and validation labels written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
