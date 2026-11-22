"""Shared paths and constants.

Two directories matter:
  DATA_DIR  - the extracted Polyvore Outfits folder (contains images/, nondisjoint/, ...)
  WORK_DIR  - where every output goes (embeddings, checkpoints, results, figures).
              On Colab, point this at Google Drive so nothing is lost on disconnect.

Set them with environment variables FT_DATA and FT_WORK, or pass --data / --work.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FT_DATA", "data/polyvore_outfits"))
WORK_DIR = Path(os.environ.get("FT_WORK", "work"))
SPLIT = os.environ.get("FT_SPLIT", "nondisjoint")  # or "disjoint" (harder: no item overlap)

CLIP_MODEL = os.environ.get("FT_CLIP", "openai/clip-vit-base-patch32")
SEED = 0

# Categories we swap when building hard negatives (garments, not jewellery etc.)
SWAPPABLE = ["tops", "bottoms", "shoes", "outerwear", "all-body"]
RULES = ["color", "pattern", "formality", "season"]


def add_path_args(parser):
    parser.add_argument("--data", default=str(DATA_DIR), help="Polyvore Outfits folder")
    parser.add_argument("--work", default=str(WORK_DIR), help="output folder")
    parser.add_argument("--split", default=SPLIT, choices=["nondisjoint", "disjoint"])
    return parser


def work_paths(work):
    """All output locations, created on demand."""
    w = Path(work)
    paths = {
        "root": w,
        "emb": w / "embeddings",
        "ckpt": w / "checkpoints",
        "results": w / "results",
        "figures": w / "figures",
        "vlm": w / "vlm_cache",
        "survey": w / "survey",
        "validation": w / "validation",
        "demo": w / "demo_out",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
