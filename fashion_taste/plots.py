"""Step 8: paper/slide figures from the results CSVs.

    python -m fashion_taste.plots --work $FT_WORK      -> <work>/figures/*.png (+ .pdf for LaTeX)

Color: CLIP variants blue, Gemini variants orange (identity = model family; the method name
is always written on the axis, so color is never the only cue). Hard-negative heatmap uses a
diverging scale centred on 0.5 (= chance): blue above chance, orange below.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .config import RULES, add_path_args, work_paths

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d95926", "#f3b89c", "#f0efec", "#86b6ef", "#1c5cab"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.dpi": 200, "savefig.bbox": "tight"})


def family_color(name):
    return ORANGE if name.lower().startswith("gemini") else BLUE


def save(fig, P, name):
    fig.savefig(P["figures"] / f"{name}.png")
    fig.savefig(P["figures"] / f"{name}.pdf")
    plt.close(fig)
    print("wrote", P["figures"] / f"{name}.png")


def main_figure(P):
    f = P["results"] / "main_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, index_col=0)
    panels = [(c, t, ref, chance) for c, t, ref, chance in
              [("compat_auc", "Outfit compatibility (AUC)", 0.86, 0.5),
               ("fitb_acc", "Fill-in-the-blank (accuracy)", 0.553, 0.25)] if c in df]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.45 * len(df) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    names = list(df.index)[::-1]
    for ax, (col, title, ref, chance) in zip(axes, panels):
        for y, n in enumerate(names):
            v = df.loc[n, col]
            if pd.isna(v):
                continue
            lo, hi = df.loc[n, f"{col}_lo"], df.loc[n, f"{col}_hi"]
            ax.plot([lo, hi], [y, y], color=family_color(n), lw=2, solid_capstyle="round")
            ax.plot(v, y, "o", ms=8, color=family_color(n), mec=SURF, mew=2)
            ax.text(v, y + 0.22, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1))
        top = max(ref, np.nanmax(df[f"{col}_hi"].astype(float))) + 0.12
        ax.set_xlim(min(chance, np.nanmin(df[f"{col}_lo"].astype(float))) - 0.03, top)
        ax.axvline(chance, color=INK2, lw=1, ls=":")
        ax.axvline(ref, color=AQUA, lw=1.5, ls="--")
        ax.text(chance, len(names) - 0.2, "chance", fontsize=8, color=INK2, ha="center")
        ax.text(ref, len(names) - 0.2, "specialist\n(published)", fontsize=8, color=INK2, ha="center")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names))
        ax.grid(axis="y", visible=False)
    save(fig, P, "fig_main_results")


def rules_heatmap(P):
    f = P["results"] / "main_results.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, index_col=0)
    cols = [c for c in [f"hn_{r}" for r in RULES] + ["hn_control"] if c in df]
    if not cols:
        return
    M = df[cols].astype(float)
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 2.5, 0.5 * len(M) + 1.2))
    ax.imshow(M.values, cmap=DIVERGING, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                continue
            n = df.loc[M.index[i], f"{cols[j]}_n"] if f"{cols[j]}_n" in df else np.nan
            ax.text(j, i, f"{v:.2f}" + (f"\n(n={int(n)})" if pd.notna(n) else ""), ha="center", va="center",
                    fontsize=8, color=INK if abs(v - 0.5) < 0.3 else "white")
    ax.set_xticks(range(len(cols)), [c.replace("hn_", "").capitalize() + ("\n(ideal 0.50)" if c == "hn_control" else "") for c in cols])
    ax.set_yticks(range(len(M)), M.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(len(cols) - 1.5, color=SURF, lw=4) if "hn_control" in cols else None
    ax.set_title("Hard negatives: how often the original outfit beats its one-rule-broken version",
                 loc="left", fontsize=10, color=INK)
    save(fig, P, "fig_hard_negative_heatmap")


def human_agreement(P):
    hf, af, sf = (P["results"] / n for n in ("survey_human_loo.csv", "survey_ai_agreement.csv", "survey_summary.json"))
    if not (hf.exists() and af.exists()):
        return
    h = pd.read_csv(hf)["human_loo_rho"].dropna().values
    ai = pd.read_csv(af)
    summ = json.loads(sf.read_text()) if sf.exists() else {}
    rows = ["Humans (each rater)"] + list(ai["method"])
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.3))
    rng = np.random.default_rng(0)
    y0 = len(rows) - 1
    ax.plot(h, y0 + rng.uniform(-0.18, 0.18, len(h)), "o", ms=6, color=INK2, alpha=0.6, mec=SURF, mew=1)
    ax.plot([np.median(h)] * 2, [y0 - 0.3, y0 + 0.3], color=INK, lw=2)
    for k, r in enumerate(ai.itertuples()):
        y = y0 - 1 - k
        ax.plot(r.ai_vs_loo_mean_rho, y, "o", ms=9, color=family_color(r.method), mec=SURF, mew=2)
        ax.text(r.ai_vs_loo_mean_rho + 0.02, y, f"{r.ai_vs_loo_mean_rho:.2f}  ({r.ai_percentile_among_humans:.0f}th pct of humans)",
                va="center", fontsize=8, color=INK)
    if "split_half_reliability" in summ:
        ax.axvline(summ["split_half_reliability"], color=AQUA, ls="--", lw=1.5)
        ax.text(summ["split_half_reliability"], -0.9, " reliability of the\n human average\n (split-half)", fontsize=8, color=INK2, va="bottom")
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_yticks(range(len(rows)), rows[::-1])
    ax.set_ylim(-1, len(rows))
    ax.set_xlabel("Spearman correlation with the other raters' mean rating")
    ax.grid(axis="y", visible=False)
    ax.set_title("Does each AI agree with people as well as people agree with each other?", loc="left", fontsize=10)
    save(fig, P, "fig_human_agreement")


def attribute_figure(P):
    f = P["results"] / "attribute_agreement.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.barh(df["attribute"][::-1], df["keyword_clip_agreement"][::-1], color=BLUE, height=0.5)
    for y, v in enumerate(df["keyword_clip_agreement"][::-1]):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Keyword rule and CLIP agree (items with a keyword label)")
    ax.grid(axis="y", visible=False)
    save(fig, P, "fig_attribute_agreement")


def training_curves(P):
    series = []
    for name, f, key in [("CLIP + trained head", "train_log_head.json", "log"),
                         ("CLIP fine-tuned", "train_log_finetune.json", "log")]:
        p = P["results"] / f
        if p.exists():
            log = json.loads(p.read_text())
            series.append((name, [r["epoch"] for r in log[key]], [r["valid_auc"] for r in log[key]],
                           log.get("zero_shot_valid_auc")))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for (name, x, y, base), c in zip(series, [BLUE, "#6250d6"]):
        ax.plot(x, y, color=c, lw=2, marker="o", ms=4, label=name)
        ax.text(x[-1], y[-1], f"  {name}", color=INK, fontsize=8, va="center")
        if base:
            ax.axhline(base, color=INK2, ls=":", lw=1)
            ax.text(0, base, "zero-shot CLIP", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, P, "fig_training_curves")


def main():
    a = add_path_args(argparse.ArgumentParser()).parse_args()
    P = work_paths(a.work)
    for fn in (main_figure, rules_heatmap, human_agreement, attribute_figure, training_curves):
        try:
            fn(P)
        except Exception as e:  # one missing input should not block the other figures
            print(f"skipped {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
