"""Step 3: build single-rule hard negatives, controls, and the blind validation booklet.

For a real (compatible) test outfit we swap ONE garment for another item of the SAME
category that matches the original on three attributes and differs only on the target
attribute, in a way that breaks a rule given the rest of the outfit:

  color      - the rest of the outfit has a chromatic color C; the original item is neutral
               or harmonious with C (same/analogous/complementary hue); the replacement's hue
               sits 60-150 degrees from C (the "discordant" range in basic color theory).
  pattern    - the rest of the outfit already has a patterned item; the original is solid;
               the replacement is patterned (two competing prints).
  formality  - the rest of the outfit is consistently formal (or casual); the original is
               the same or neutral; the replacement is the opposite.
  season     - the rest of the outfit is consistently summer (or winter); the original is
               the same or neutral; the replacement is the opposite.
  control    - the replacement matches the original on ALL four attributes. A model that
               also penalises controls is reacting to "something changed", not to a rule.

Only *confident* attributes (keyword rules and CLIP agree) are used, and "matches" means
the same confident value. By construction each hard negative changes exactly one attribute.
Human validation (validation booklet + `validate`) then measures how often that holds.

    python -m fashion_taste.hard_negatives build   --data $FT_DATA --work $FT_WORK
    python -m fashion_taste.hard_negatives booklet --data $FT_DATA --work $FT_WORK
    python -m fashion_taste.hard_negatives validate --work $FT_WORK   (after labelling)
"""
import argparse
import json
import random

import pandas as pd

from . import data
from .attributes import HUE, NEUTRALS
from .config import RULES, SEED, SWAPPABLE, add_path_args, work_paths
from .viz import collage, to_data_uri

LABEL_CHOICES = ["color", "pattern", "formality", "season", "none", "multiple"]


def hue_diff(a, b):
    d = abs(HUE[a] - HUE[b]) % 360
    return min(d, 360 - d)


def harmonious(a, b):
    """Neutral, same, analogous (<=45 deg) or complementary (>=150 deg)."""
    if a in NEUTRALS or b in NEUTRALS:
        return True
    d = hue_diff(a, b)
    return d <= 45 or d >= 150


def clashes(a, b):
    return a not in NEUTRALS and b not in NEUTRALS and 60 <= hue_diff(a, b) <= 150


def _dominant(values):
    """Single consistent non-neutral value among the others, else None."""
    vals = {v for v in values if v and v != "neutral"}
    return vals.pop() if len(vals) == 1 else None


OPPOSITE = {"formal": "casual", "casual": "formal", "summer": "winter", "winter": "summer"}


def target_ok(rule, others, orig, repl):
    """Does swapping orig->repl break `rule` given the other items' attributes?"""
    if rule == "color":
        chroma = [o["color"] for o in others if o["color"] and o["color"] not in NEUTRALS]
        if not chroma or not orig["color"] or not repl["color"]:
            return False
        anchor = chroma[0]
        return all(harmonious(orig["color"], c) for c in chroma) and clashes(repl["color"], anchor)
    if rule == "pattern":
        return any(o["pattern"] == "patterned" for o in others) and orig["pattern"] == "solid" \
            and repl["pattern"] == "patterned"
    dom = _dominant([o[rule] for o in others])
    if dom is None or orig[rule] not in (dom, "neutral"):
        return False
    return repl[rule] == OPPOSITE[dom]


def matches(a, b, attrs):
    return all(a[x] is not None and a[x] == b[x] for x in attrs)


def build(ds, attrs_df, n_per_rule=300, seed=SEED, which="test", attr_mode="agree"):
    """attr_mode='agree': keyword AND CLIP must agree (default). 'keyword': keyword rules only
    (fallback if too few confident items; report it as a limitation)."""
    rng = random.Random(seed)
    col = (lambda x: x) if attr_mode == "agree" else (lambda x: f"{x}_kw")
    A = {r.item_id: {x: (None if pd.isna(getattr(r, col(x))) else getattr(r, col(x))) for x in RULES}
         for r in attrs_df.itertuples()}
    cat = {r.item_id: r.category for r in attrs_df.itertuples()}
    outfits = {s: items for s, items in ds.outfits[which].items() if all(i in A for i in items)}
    pool = sorted({i for items in outfits.values() for i in items if cat[i] in SWAPPABLE})

    def keyof(i, keep):
        return (cat[i],) + tuple(A[i][x] for x in keep)

    records, used = [], {r: set() for r in RULES + ["control"]}
    set_ids = sorted(outfits)
    for rule in RULES + ["control"]:
        keep = RULES if rule == "control" else [x for x in RULES if x != rule]
        index = {}  # items sharing category + the attributes that must match
        for c in pool:
            if all(A[c][x] is not None for x in keep):
                index.setdefault(keyof(c, keep), []).append(c)
        n_found = 0
        rng.shuffle(set_ids)
        for sid in set_ids:
            if n_found >= n_per_rule:
                break
            items = outfits[sid]
            positions = [k for k, i in enumerate(items) if cat[i] in SWAPPABLE]
            rng.shuffle(positions)
            for pos in positions:
                orig = items[pos]
                others = [A[i] for k, i in enumerate(items) if k != pos]
                o = A[orig]
                if not matches(o, o, keep) or (rule != "control" and o[rule] is None):
                    continue
                same = [c for c in index.get(keyof(orig, keep), []) if c != orig and c not in items]
                if rule == "control":
                    cands = same
                else:
                    cands = [c for c in same if A[c][rule] is not None and target_ok(rule, others, o, A[c])]
                if not cands:
                    continue
                n_found += 1
                repl = rng.choice(cands)
                mod = list(items)
                mod[pos] = repl
                records.append(dict(hn_id=f"{rule}_{len(used[rule])}", rule=rule, set_id=sid,
                                    original=items, modified=mod, position=pos,
                                    original_item=orig, replacement_item=repl,
                                    original_attrs=o, replacement_attrs=A[repl],
                                    category=cat[orig]))
                used[rule].add(sid)
                break
    return records


def booklet(ds, records, out_dir, n_per_rule=20, seed=SEED):
    """Blind HTML booklet: original vs modified outfit, swapped item outlined. True rule hidden."""
    rng = random.Random(seed + 1)
    by_rule = {}
    for r in records:
        by_rule.setdefault(r["rule"], []).append(r)
    sample = []
    for rule, rs in by_rule.items():
        sample += rng.sample(rs, min(n_per_rule, len(rs)))
    rng.shuffle(sample)
    key, cards = [], []
    for case, r in enumerate(sample, 1):
        key.append({"case": case, "hn_id": r["hn_id"], "rule": r["rule"]})
        a = collage([ds.image_path(i) for i in r["original"]], highlight=r["position"], cell=170)
        b = collage([ds.image_path(i) for i in r["modified"]], highlight=r["position"], cell=170)
        cards.append(f"""<div class=card><h2>Case {case}</h2>
<p><b>Original outfit</b></p><img src="{to_data_uri(a)}">
<p><b>After swapping the outlined item</b></p><img src="{to_data_uri(b)}"></div>""")
    opts = ", ".join(LABEL_CHOICES)
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Hard-negative validation</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}}
.card{{border:1px solid #ccc;border-radius:8px;padding:12px;margin:16px 0}} img{{max-width:100%}}</style></head><body>
<h1>Hard-negative validation booklet</h1>
<p>For each case, decide which compatibility rule the swap breaks. Label independently, without
discussing with the other labeler. Write one label per case in your copy of
<code>validation_labels_TEMPLATE.csv</code>: <b>{opts}</b>.</p>
<ul><li><b>color</b>: the new item's color clashes with the rest of the outfit</li>
<li><b>pattern</b>: the new item adds a competing print/pattern</li>
<li><b>formality</b>: the new item is much more formal or more casual than the rest</li>
<li><b>season</b>: the new item belongs to the opposite season (e.g. wool coat with sandals)</li>
<li><b>none</b>: the outfit still works / nothing is broken</li>
<li><b>multiple</b>: the swap breaks more than one of the rules above</li></ul>
{''.join(cards)}</body></html>"""
    (out_dir / "validation_booklet.html").write_text(html)
    pd.DataFrame(key).to_csv(out_dir / "validation_key_DO_NOT_OPEN_UNTIL_DONE.csv", index=False)
    pd.DataFrame({"case": range(1, len(sample) + 1), "label": "", "notes": ""}).to_csv(
        out_dir / "validation_labels_TEMPLATE.csv", index=False)
    return len(sample)


def validate(out_dir, results_dir, files=None, pass_threshold=0.7):
    """Compare two labelers with the hidden key: single-rule rate, Cohen's kappa, per-rule pass."""
    from sklearn.metrics import cohen_kappa_score
    key = pd.read_csv(out_dir / "validation_key_DO_NOT_OPEN_UNTIL_DONE.csv")
    files = files or sorted(p for p in out_dir.glob("validation_labels_*.csv") if "TEMPLATE" not in p.name)
    if len(files) < 2:
        raise SystemExit(f"need two filled label files in {out_dir} (validation_labels_<name>.csv)")
    labs = []
    for f in files[:2]:
        d = pd.read_csv(f)
        d["label"] = d["label"].astype(str).str.strip().str.lower()
        labs.append(d.set_index("case")["label"])
    df = key.set_index("case").copy()
    df["l1"], df["l2"] = labs[0], labs[1]
    df = df.dropna(subset=["l1", "l2"])
    expected = df["rule"].where(df["rule"] != "control", "none")
    rows = []
    for rule, g in df.groupby("rule"):
        e = "none" if rule == "control" else rule
        m1, m2 = (g["l1"] == e).mean(), (g["l2"] == e).mean()
        rows.append({"rule": rule, "n": len(g), "labeler1_matches_intended": m1,
                     "labeler2_matches_intended": m2, "both_match": ((g["l1"] == e) & (g["l2"] == e)).mean(),
                     "multiple_flagged": ((g["l1"] == "multiple") | (g["l2"] == "multiple")).mean(),
                     "kappa_within_rule": cohen_kappa_score(g["l1"], g["l2"]) if g["l1"].nunique() + g["l2"].nunique() > 2 else float("nan"),
                     "passes": (m1 + m2) / 2 >= pass_threshold})
    summ = pd.DataFrame(rows)
    overall = {"cohen_kappa_overall": cohen_kappa_score(df["l1"], df["l2"]),
               "raw_agreement": float((df["l1"] == df["l2"]).mean()),
               "single_rule_rate": float(((df["l1"] == expected) & (df["l2"] == expected) & (df["rule"] != "control")).sum()
                                         / max(1, (df["rule"] != "control").sum())),
               "labeler_files": [str(f) for f in files[:2]]}
    summ.to_csv(results_dir / "validation_summary.csv", index=False)
    (results_dir / "validation_overall.json").write_text(json.dumps(overall, indent=2))
    print(summ.to_string(index=False))
    print(json.dumps(overall, indent=2))
    failing = summ.loc[~summ["passes"], "rule"].tolist()
    if failing:
        print(f"\nRules below the {pass_threshold:.0%} threshold (report and consider dropping): {failing}")
    return summ, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "booklet", "validate"])
    add_path_args(p)
    p.add_argument("--n_per_rule", type=int, default=300)
    p.add_argument("--booklet_per_rule", type=int, default=20)
    p.add_argument("--attr_mode", default="agree", choices=["agree", "keyword"])
    a = p.parse_args()
    P = work_paths(a.work)
    hn_path = P["results"] / "hard_negatives.json"
    if a.cmd == "build":
        ds = data.load(a.data, a.split)
        attrs = pd.read_csv(P["results"] / "item_attributes.csv", dtype={"item_id": str})
        recs = build(ds, attrs, a.n_per_rule, attr_mode=a.attr_mode)
        hn_path.write_text(json.dumps(recs))
        counts = pd.Series([r["rule"] for r in recs]).value_counts()
        print(counts.to_string())
        low = [r for r in RULES + ["control"] if counts.get(r, 0) < min(50, a.n_per_rule)]
        if low:
            print(f"WARNING: few examples for {low}. Options: re-run attributes with a lower "
                  "--neutral_margin, or build with --attr_mode keyword (then human validation matters more).")
    elif a.cmd == "booklet":
        ds = data.load(a.data, a.split)
        n = booklet(ds, json.loads(hn_path.read_text()), P["validation"], a.booklet_per_rule)
        print(f"wrote {n} cases to {P['validation'] / 'validation_booklet.html'}")
    else:
        validate(P["validation"], P["results"])


if __name__ == "__main__":
    main()
