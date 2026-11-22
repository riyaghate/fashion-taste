"""Step 7: the human study.

  build    - picks the survey outfits, renders one image per outfit, and writes a Google Apps
             Script that creates the whole Google Form automatically.
  analyze  - reads the exported responses CSV and measures human-human and AI-human agreement.

    python -m fashion_taste.survey build   --data $FT_DATA --work $FT_WORK
    python -m fashion_taste.survey analyze --data $FT_DATA --work $FT_WORK

Agreement measures (addressing subjectivity in fashion taste):
  * Krippendorff's alpha (interval) across all raters - overall reliability.
  * Split-half reliability: raters split randomly in two, correlation of the two halves' mean
    ratings, Spearman-Brown corrected (1000 splits). This is the noise ceiling.
  * Per-rater z-scoring removes harsh/lenient rating styles before averaging.
  * Leave-one-out: each human is correlated with the mean of the OTHER humans; each AI is
    correlated with the same leave-one-out means. The AI is then placed within the human
    distribution ("as good as the median person?").
  * Consensus split: outfits with low vs high disagreement (SD of z-scores) analysed separately.
  * Fashion interest: do self-rated enthusiasts agree with the AI more than others?
  * Test-retest: one outfit is shown twice to measure within-person consistency.
"""
import argparse
import json
import random
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from . import data
from .config import RULES, SEED, add_path_args, work_paths
from .viz import collage

FOLDER_NAME = "fashion_taste_survey_images"


def build(ds, P, n_real=12, n_random=6, n_hn_per_rule=3, seed=SEED):
    rng = random.Random(seed + 11)
    hn = json.loads((P["results"] / "hard_negatives.json").read_text())
    comp = ds.compatibility("test")
    real = [it for l, it, _ in comp if l == 1 and 3 <= len(it) <= 5]
    rand = [it for l, it, _ in comp if l == 0 and 3 <= len(it) <= 5]
    rows = []
    for it in rng.sample(real, n_real):
        rows.append(dict(kind="real", rule="", items=it))
    for it in rng.sample(rand, n_random):
        rows.append(dict(kind="random", rule="", items=it))
    for rule in RULES:
        cands = [r for r in hn if r["rule"] == rule and 3 <= len(r["modified"]) <= 5]
        for r in rng.sample(cands, min(n_hn_per_rule, len(cands))):
            rows.append(dict(kind="hard_negative", rule=rule, items=r["modified"], hn_id=r["hn_id"]))
    rng.shuffle(rows)
    rows.append(dict(rows[0], kind="repeat"))  # test-retest: first outfit again at the end
    img_dir = P["survey"] / FOLDER_NAME
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in img_dir.glob("*.jpg"):
        f.unlink()
    for q, r in enumerate(rows, 1):
        r["question"] = q
        r["survey_id"] = f"S{q:02d}" if r["kind"] != "repeat" else rows[0]["survey_id"]
        collage([ds.image_path(i) for i in r["items"]], cell=300).save(img_dir / f"outfit_{q:02d}.jpg", quality=88)
    man = pd.DataFrame([{**r, "items": json.dumps(r["items"])} for r in rows])
    man.to_csv(P["survey"] / "manifest.csv", index=False)
    (P["survey"] / "make_form.gs").write_text(APPS_SCRIPT.replace("__FOLDER__", FOLDER_NAME).replace("__N__", str(len(rows))))
    print(f"{len(rows)} survey questions -> {img_dir}")
    print(man["kind"].value_counts().to_string())


APPS_SCRIPT = r"""/**
 * Builds the outfit-rating Google Form automatically.
 * 1. The folder "__FOLDER__" must be in your Google Drive (it is, if WORK_DIR is on Drive).
 * 2. Go to script.google.com -> New project, paste this file, click Run (buildSurvey), allow access.
 * 3. View -> Logs (or Execution log) shows the link to share and the link to edit.
 */
function buildSurvey() {
  var folders = DriveApp.getFoldersByName('__FOLDER__');
  if (!folders.hasNext()) throw new Error('Folder __FOLDER__ not found in Drive');
  var files = [], it = folders.next().getFiles();
  while (it.hasNext()) { var f = it.next(); if (/^outfit_\d+\.jpg$/.test(f.getName())) files.push(f); }
  files.sort(function (a, b) { return a.getName() < b.getName() ? -1 : 1; });
  if (files.length != __N__) Logger.log('Warning: expected __N__ images, found ' + files.length);

  var form = FormApp.create('Rate the Outfit! (UVA AI course project, ~5 minutes)');
  form.setDescription(
    'We are studying whether AI models judge outfits the way people do. You will see ' + files.length +
    ' outfits, each shown as separate clothing items. For each, rate how well the items go together as ' +
    'one outfit. There are no right answers - go with your gut. Participation is voluntary and anonymous; ' +
    'we do not collect names or emails, and responses are only used for a class project.');
  form.setCollectEmail(false);
  form.setProgressBar(true);
  form.addScaleItem().setTitle('How interested are you in fashion?')
      .setBounds(1, 5).setLabels('Not at all', 'Very interested').setRequired(true);
  files.forEach(function (f, i) {
    var n = i + 1;
    if (n % 8 == 1) form.addPageBreakItem().setTitle('Outfits ' + n + '-' + Math.min(n + 7, files.length));
    form.addImageItem().setImage(f.getBlob()).setTitle('Outfit ' + n);
    form.addScaleItem().setTitle('Outfit ' + n + ': how well do these items go together?')
        .setBounds(1, 5).setLabels("Don't go together", 'Go together very well').setRequired(true);
  });
  Logger.log('SHARE THIS LINK: ' + form.getPublishedUrl());
  Logger.log('Edit link: ' + form.getEditUrl());
}
"""


# ------------------------------------------------------------------ analysis
def load_responses(path, manifest):
    raw = pd.read_csv(path)
    interest_col = next((c for c in raw.columns if "interested in fashion" in c.lower()), None)
    qcols = {}
    for c in raw.columns:
        m = re.match(r"\s*Outfit (\d+):", c)
        if m:
            qcols[int(m.group(1))] = c
    R = pd.DataFrame({q: pd.to_numeric(raw[c], errors="coerce") for q, c in sorted(qcols.items())})
    interest = pd.to_numeric(raw[interest_col], errors="coerce") if interest_col else pd.Series(np.nan, index=raw.index)
    man = manifest.set_index("question")
    rep_q = int(man.index[man["kind"] == "repeat"][0]) if (man["kind"] == "repeat").any() else None
    first_q = int(man.index[(man["survey_id"] == man.loc[rep_q, "survey_id"]) & (man["kind"] != "repeat")][0]) if rep_q else None
    retest = (R[first_q], R[rep_q]) if rep_q and rep_q in R and first_q in R else None
    keep = [q for q in R.columns if man.loc[q, "kind"] != "repeat"]
    R = R[keep]
    R.columns = [man.loc[q, "survey_id"] for q in keep]
    ok = R.notna().mean(axis=1) >= 0.8  # drop very incomplete responses
    return R[ok].reset_index(drop=True), interest[ok].reset_index(drop=True), retest


def zscore_rows(R):
    mu, sd = R.mean(axis=1), R.std(axis=1).replace(0, np.nan)
    return R.sub(mu, axis=0).div(sd, axis=0)


def split_half(Z, reps=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(Z)
    rs = []
    for _ in range(reps):
        perm = rng.permutation(n)
        a, b = Z.iloc[perm[: n // 2]].mean(), Z.iloc[perm[n // 2:]].mean()
        r = spearmanr(a, b, nan_policy="omit").statistic
        rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs)), float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def loo_corrs(Z, ai=None):
    """Per rater: Spearman(rater, mean of others). If ai given: Spearman(ai, mean of others)."""
    out = []
    for i in range(len(Z)):
        others = Z.drop(index=Z.index[i]).mean()
        x = Z.iloc[i] if ai is None else ai
        out.append(spearmanr(x, others, nan_policy="omit").statistic)
    return np.array(out, float)


def ai_survey_scores(P, ds, survey_ids, manifest):
    """method -> Series indexed by survey_id."""
    from .evaluate import method_scores
    from . import tasks as T
    st = T.survey_tasks(P["survey"] / "manifest.csv")
    res = {}
    for name, df in method_scores(P, ds, st).items():
        d = df[df["task"] == "survey"].set_index("id")["score"]
        if len(d.dropna()) >= 0.8 * len(survey_ids):
            res[name] = d.reindex(survey_ids).astype(float)
    return res


def analyze(ds, P, responses_csv):
    import krippendorff
    manifest = pd.read_csv(P["survey"] / "manifest.csv")
    R, interest, retest = load_responses(responses_csv, manifest)
    Z = zscore_rows(R)
    summary = {"n_raters": int(len(R)), "n_outfits": int(R.shape[1]),
               "krippendorff_alpha_interval_raw": float(krippendorff.alpha(R.values, level_of_measurement="interval")),
               "krippendorff_alpha_interval_z": float(krippendorff.alpha(Z.values, level_of_measurement="interval"))}
    sh, lo, hi = split_half(Z)
    summary.update(split_half_reliability=sh, split_half_lo=lo, split_half_hi=hi)
    human = loo_corrs(Z)
    summary.update(human_loo_median=float(np.nanmedian(human)), human_loo_q25=float(np.nanpercentile(human, 25)),
                   human_loo_q75=float(np.nanpercentile(human, 75)))
    if retest is not None:
        a, b = retest
        summary["test_retest_exact"] = float((a == b).mean())
        summary["test_retest_mean_abs_diff"] = float((a - b).abs().mean())

    # mean rating by outfit kind (sanity check: real > hard negatives > random?)
    kinds = manifest[manifest["kind"] != "repeat"].set_index("survey_id")
    by_kind = R.mean().groupby(kinds["kind"].reindex(R.columns).values).mean()
    summary["mean_rating_by_kind"] = {k: float(v) for k, v in by_kind.items()}

    # consensus split
    sd = Z.std()
    high_cons = sd[sd <= sd.median()].index
    low_cons = sd[sd > sd.median()].index
    mean_h = Z.mean()

    rows = []
    for name, s in ai_survey_scores(P, ds, list(R.columns), manifest).items():
        ai_loo = loo_corrs(Z, s)
        per_rater = np.array([spearmanr(s, Z.iloc[i], nan_policy="omit").statistic for i in range(len(Z))])
        row = {"method": name,
               "ai_vs_loo_mean_rho": float(np.nanmean(ai_loo)),
               "ai_percentile_among_humans": float((human < np.nanmean(ai_loo)).mean() * 100),
               "rho_all_outfits": float(spearmanr(s, mean_h, nan_policy="omit").statistic),
               "rho_high_consensus": float(spearmanr(s[high_cons], mean_h[high_cons], nan_policy="omit").statistic),
               "rho_low_consensus": float(spearmanr(s[low_cons], mean_h[low_cons], nan_policy="omit").statistic)}
        hi_i, lo_i = interest >= 4, interest <= 3
        if hi_i.sum() >= 3 and lo_i.sum() >= 3:
            row["rho_with_enthusiasts"] = float(np.nanmean(per_rater[hi_i.values]))
            row["rho_with_others"] = float(np.nanmean(per_rater[lo_i.values]))
            row["enthusiast_diff_p"] = float(mannwhitneyu(per_rater[hi_i.values], per_rater[lo_i.values]).pvalue)
        rows.append(row)
    ai = pd.DataFrame(rows)
    (P["results"] / "survey_summary.json").write_text(json.dumps(summary, indent=2))
    ai.to_csv(P["results"] / "survey_ai_agreement.csv", index=False)
    pd.DataFrame({"human_loo_rho": human}).to_csv(P["results"] / "survey_human_loo.csv", index=False)
    pd.DataFrame({"mean_z": mean_h, "sd_z": sd, "mean_raw": R.mean(),
                  "kind": kinds["kind"].reindex(R.columns)}).to_csv(P["results"] / "survey_outfit_stats.csv")
    print(json.dumps(summary, indent=2))
    if len(ai):
        print(ai.round(3).to_string(index=False))
    return summary, ai


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "analyze"])
    add_path_args(p)
    p.add_argument("--responses", default=None, help="CSV exported from the Google Form (default: <work>/survey/responses.csv)")
    a = p.parse_args()
    P = work_paths(a.work)
    ds = data.load(a.data, a.split)
    if a.cmd == "build":
        build(ds, P)
    else:
        analyze(ds, P, a.responses or P["survey"] / "responses.csv")


if __name__ == "__main__":
    main()
