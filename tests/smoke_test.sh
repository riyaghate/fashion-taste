#!/usr/bin/env bash
# End-to-end test of every step on a tiny fake dataset with an untrained tiny CLIP (no downloads).
# Usage: bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
export FT_DATA=$T/polyvore_outfits FT_WORK=$T/work
python tests/make_fake_data.py "$FT_DATA"
python -m fashion_taste.data
python -m fashion_taste.embed --clip tiny-random --workers 0
python -m fashion_taste.attributes --clip tiny-random --neutral_margin 0.5
python -m fashion_taste.hard_negatives build --n_per_rule 40 --attr_mode keyword
python -m fashion_taste.hard_negatives booklet --booklet_per_rule 5
python -m fashion_taste.train head --epochs 3 --batch 64
python -m fashion_taste.train finetune --clip tiny-random --epochs 2 --batch 32 --workers 0
python -m fashion_taste.survey build
python tests/fake_gemini_and_survey.py "$FT_DATA" "$FT_WORK"
python -m fashion_taste.vlm --dry_run --strategies direct,fewshot --n_compat 40 --n_fitb 30 --n_hn 10
python -m fashion_taste.hard_negatives validate
python -m fashion_taste.evaluate --n_compat 40 --n_fitb 30 --n_hn 10
python -m fashion_taste.survey analyze
python -m fashion_taste.plots
# demo on two fake "photo" outfits
mkdir -p $T/photos/fit_a $T/photos/fit_b
cp $(ls $FT_DATA/images/*.jpg | head -3) $T/photos/fit_a/
cp $(ls $FT_DATA/images/*.jpg | tail -3) $T/photos/fit_b/
python -m fashion_taste.demo rate --photos $T/photos --clip tiny-random --keep_bg
mkdir -p $T/closet/tops $T/closet/bottoms
cp $(ls $FT_DATA/images/*.jpg | head -2) $T/closet/tops/
cp $(ls $FT_DATA/images/*.jpg | tail -2) $T/closet/bottoms/
python -m fashion_taste.demo closet --photos $T/closet --clip tiny-random --keep_bg
echo "SMOKE TEST PASSED  (outputs in $FT_WORK)"
