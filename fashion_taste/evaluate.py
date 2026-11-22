"""Step 6: score every method on every task and build the results / ablation tables.

    python -m fashion_taste.evaluate --data $FT_DATA --work $FT_WORK

Writes to <work>/results/:
  main_results.csv      - all methods on the SAME subset Gemini sees (the ablation table)
  full_test_results.csv - CLIP variants on the full test set
  per_example_scores.csv
Each cell has a 95% bootstrap confidence interval (lo/hi columns).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data, tasks as T
from .clip_utils import EmbeddingTable
from .config import RULES, SEED, add_path_args, work_paths
from .scoring import score_tasks
from .vlm import STRATEGIES, eval_tasks

CLIP_METHODS = {"clip_zeroshot": "CLIP zero-shot", "clip_head": "CLIP + trained head",
                "clip_finetuned": "CLIP fine-tuned"}


def bootstrap(fn, n, reps=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def metrics(tasks, scores):
    """scores: DataFrame with task,id,score,choice for one method."""
    S = scores.set_index(["task", "id"])
    out = {}
    # compatibility AUC
    c = [(t["label"], S.loc[("compat", t["id"]), "score"]) for t in tasks
         if t["task"] == "compat" and ("compat", t["id"]) in S.index]
    c = [(l, s) for l, s in c if pd.notna(s)]
    if len(c) > 10 and len({l for l, _ in c}) == 2:
        y, s = np.array([l for l, _ in c]), np.array([s for _, s in c], float)
        out["compat_auc"] = roc_auc_score(y, s)
        out["compat_auc_lo"], out["compat_auc_hi"] = bootstrap(lambda i: roc_auc_score(y[i], s[i]), len(y))
        out["compat_n"] = len(y)
    # fill in the blank
    f = [(t["correct"], S.loc[("fitb", t["id"]), "choice"]) for t in tasks
         if t["task"] == "fitb" and ("fitb", t["id"]) in S.index]
    f = np.array([float(ch == cor) for cor, ch in f if pd.notna(ch)])
    if len(f):
        out["fitb_acc"] = f.mean()
        out["fitb_acc_lo"], out["fitb_acc_hi"] = bootstrap(lambda i: f[i].mean(), len(f))
        out["fitb_n"] = len(f)
    # hard negatives: pairwise accuracy per rule
    pairs = {}
    for t in tasks:
        if t["task"] == "hn" and ("hn", t["id"]) in S.index:
            v = S.loc[("hn", t["id"]), "score"]
            if pd.notna(v):
                pairs.setdefault((t["rule"], t["pair"]), {})[t["side"]] = float(v)
    for rule in RULES + ["control"]:
        w = np.array([1.0 if p["orig"] > p["mod"] else 0.5 if p["orig"] == p["mod"] else 0.0
                      for (r, _), p in pairs.items() if r == rule and len(p) == 2])
        if len(w):
            out[f"hn_{rule}"] = w.mean()
            out[f"hn_{rule}_lo"], out[f"hn_{rule}_hi"] = bootstrap(lambda i: w[i].mean(), len(w))
            out[f"hn_{rule}_n"] = len(w)
            out[f"hn_{rule}_ties"] = float((w == 0.5).mean())
    if all(f"hn_{r}" in out for r in RULES):
        out["hn_mean"] = float(np.mean([out[f"hn_{r}"] for r in RULES]))
    return out


def method_scores(P, ds, tasks):
    """method name -> DataFrame(task,id,score,choice)."""
    res = {}
    for key, name in CLIP_METHODS.items():
        if EmbeddingTable.exists(P["emb"], key):
            res[name] = pd.DataFrame(score_tasks(EmbeddingTable.load(P["emb"], key), tasks))
    for s in STRATEGIES:
        f = P["results"] / f"scores_gemini_{s}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["id"] = df["id"].astype(str)
            res[f"Gemini ({s})"] = df
    for df in res.values():
        for col in ("score", "choice"):
            if col not in df:
                df[col] = np.nan
    return res


def table(results):
    df = pd.DataFrame(results).T
    order = [c for c in ["compat_auc", "fitb_acc"] + [f"hn_{r}" for r in RULES] + ["hn_mean", "hn_control"] if c in df]
    return df, df[order]


def main():
    p = add_path_args(argparse.ArgumentParser())
    p.add_argument("--n_compat", type=int, default=300)
    p.add_argument("--n_fitb", type=int, default=200)
    p.add_argument("--n_hn", type=int, default=75)
    a = p.parse_args()
    P = work_paths(a.work)
    ds = data.load(a.data, a.split)
    sub = eval_tasks(ds, P, a.n_compat, a.n_fitb, a.n_hn)
    full = T.compat_tasks(ds) + T.fitb_tasks(ds)
    hn_path = P["results"] / "hard_negatives.json"
    if hn_path.exists():
        full += T.hn_tasks(json.loads(hn_path.read_text()))

    sub_scores = method_scores(P, ds, sub)
    all_rows = []
    for name, df in sub_scores.items():
        d = df.copy(); d["method"] = name; all_rows.append(d)
    if all_rows:
        pd.concat(all_rows).to_csv(P["results"] / "per_example_scores.csv", index=False)
    res = {name: metrics(sub, df) for name, df in sub_scores.items()}
    detail, short = table(res)
    detail.to_csv(P["results"] / "main_results.csv")
    print("=== Same subset for every method (main / ablation table) ===")
    print(short.round(3).to_string())

    full_res = {}
    for key, name in CLIP_METHODS.items():
        if EmbeddingTable.exists(P["emb"], key):
            full_res[name] = metrics(full, pd.DataFrame(score_tasks(EmbeddingTable.load(P["emb"], key), full)))
    if full_res:
        fd, fs = table(full_res)
        fd.to_csv(P["results"] / "full_test_results.csv")
        print("\n=== CLIP variants on the full test set ===")
        print(fs.round(3).to_string())
    print("\nPublished reference (Vasileva et al. 2018, type-aware, nondisjoint): AUC 0.86, FITB 0.553 "
          "(their released improved checkpoint: AUC 0.88, FITB 0.576)")


if __name__ == "__main__":
    main()
