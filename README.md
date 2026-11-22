# Does AI Have Taste?
**Benchmarking vision-language models, CLIP and human judgment on outfit compatibility.**
UVA AI course final project, Fall 2026.

We test whether AI models judge outfits the way people do, and *where* they fail:
1. **Standard benchmarks** on Polyvore Outfits: real vs. random outfits (AUC) and fill-in-the-blank (accuracy).
2. **Single-rule trick outfits ("hard negatives")**: one garment is swapped so that exactly one rule
   (color, pattern, formality or season) breaks. Two teammates validate them blind, and controls check
   whether a model reacts to the rule or just to "something changed".
3. **A human survey** comparing each AI with how much people agree with *each other*.
4. **Ablations**: CLIP zero-shot vs. trained head vs. full fine-tune, and four Gemini prompting strategies
   (direct, reason-first, rubric, few-shot), broken down by rule.

## How to run
Open **`notebooks/run_on_colab.ipynb`** in Google Colab (File → Open notebook → GitHub → `riyaghate/fashion-taste`),
switch to a T4 GPU, and run the cells in order. Everything saves to Google Drive and resumes after disconnects.

## Repository layout
| File | What it does |
|---|---|
| `fashion_taste/data.py` | Loads Polyvore Outfits and its benchmark tasks |
| `fashion_taste/embed.py` | Zero-shot CLIP embeddings for every item |
| `fashion_taste/attributes.py` | Color/pattern/formality/season tags (keyword rules ∩ CLIP zero-shot) |
| `fashion_taste/hard_negatives.py` | Builds single-rule trick outfits and controls, the blind validation booklet, and the agreement check |
| `fashion_taste/train.py` | Trained head on frozen CLIP, and full CLIP fine-tuning (contrastive, resumable) |
| `fashion_taste/vlm.py` | Gemini evaluation with 4 prompting strategies (rate-limited, cached, resumable) |
| `fashion_taste/survey.py` | Builds the Google Form automatically; human agreement analysis |
| `fashion_taste/evaluate.py` | All metrics with 95% bootstrap CIs → the main and ablation tables |
| `fashion_taste/plots.py` | Figures for slides (PNG) and the report (PDF) |
| `fashion_taste/demo.py` | "Rate My Fit" live demo and "pick my best outfit from my closet" |
| `tests/smoke_test.sh` | Runs every step end to end on a fake mini dataset (no GPU or downloads needed) |

## How we address the professor's feedback
| Feedback | What we do | Where |
|---|---|---|
| Each hard negative should break **one** rule | Swap in a same-category item that matches the original on the other 3 attributes; attributes count only when keyword rules **and** CLIP agree; controls change nothing; 2 blind human labelers + Cohen's κ; rules below 70% are reported and dropped | `hard_negatives.py` |
| Human agreement & subjectivity | Krippendorff's α; split-half reliability; per-rater z-scores; leave-one-out comparison placing each AI within the human distribution; high- vs. low-consensus outfits; fashion-interest split; test-retest item | `survey.py` |
| Ablations | Zero-shot → trained head → full fine-tune; direct / reason / rubric / few-shot prompting; every method on the same examples, per rule | `evaluate.py`, `fig_hard_negative_heatmap` |

## Team roles
| Person | Owns | Manual work |
|---|---|---|
| Riya Ghate (nmy4vs) | Colab runs: data, embeddings, training (cells 1–10, 15–16) | Get the dataset into Drive; keep the fine-tuning tab alive |
| Member 2 | Gemini runs (cell 12) and prompting ablation | Get the API key, re-run daily until done; validation labeler #1 |
| Member 3 | Trick outfits and validation (cells 8, 13) | Validation labeler #2; spot-check examples |
| Member 4 | Survey and demo (cells 11, 14, 17) | Build the form, **send it out by 11/24**, get 20+ responses; take demo photos |

Everyone: slides (presentation **Dec 8**) and report (8 pages, IEEE format, due **Dec 10**).

## Timeline
| Dates | Milestone |
|---|---|
| Sun 11/22 – Tue 11/24 | Data in Drive; embeddings, attributes, trick outfits; train head; **survey sent out**; Gemini `direct` starts |
| Wed 11/25 – Sun 11/29 | Fine-tuning runs; Gemini strategies keep running daily; validation labeling |
| Mon 11/30 – Thu 12/3 | Survey closes; evaluate + figures; demo photos; draft slides; email the TA for a practice run |
| Fri 12/4 – Mon 12/7 | Slides final, rehearse demo and Q&A |
| **Tue 12/8** | **Presentation** |
| Wed 12/9 – Thu 12/10 | Final numbers into the report; **submit by 12/10 11:59pm** |

## Data and references
- Polyvore Outfits: M. I. Vasileva et al., *Learning Type-Aware Embeddings for Fashion Compatibility*, ECCV 2018.
  Dataset: https://github.com/mvasil/fashion-compatibility (not redistributed here).
- CLIP: A. Radford et al., ICML 2021 (`openai/clip-vit-base-patch32` via Hugging Face).
- Survey responses are anonymous and used only for this course project; raw responses and demo photos are git-ignored.
