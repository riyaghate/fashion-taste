"""Step 1: embed every outfit item with pretrained (zero-shot) CLIP.

    python -m fashion_taste.embed --data $FT_DATA --work $FT_WORK

Output: <work>/embeddings/clip_zeroshot.npy (+ _ids.json). Skips if already done.
"""
import argparse

from . import data
from .clip_utils import EmbeddingTable, clip_image_features, device, embed_paths, load_clip
from .config import CLIP_MODEL, add_path_args, work_paths


def all_item_ids(ds):
    return sorted(set(ds.key2item.values()))


def main():
    p = add_path_args(argparse.ArgumentParser())
    p.add_argument("--clip", default=CLIP_MODEL)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    P = work_paths(a.work)
    if EmbeddingTable.exists(P["emb"], "clip_zeroshot") and not a.force:
        print("clip_zeroshot embeddings already exist; use --force to redo")
        return
    ds = data.load(a.data, a.split)
    ids = all_item_ids(ds)
    model, _ = load_clip(a.clip)
    model = model.to(device()).eval()
    size = model.config.vision_config.image_size
    vecs, ok = embed_paths(lambda x: clip_image_features(model, x), [ds.image_path(i) for i in ids],
                           size, a.batch, a.workers, desc="zero-shot CLIP")
    EmbeddingTable(ids, vecs, ok).save(P["emb"], "clip_zeroshot")
    print(f"embedded {len(ids):,} items ({(~ok).sum()} missing images)")


if __name__ == "__main__":
    main()
