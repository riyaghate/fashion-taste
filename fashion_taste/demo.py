""""Rate My Fit" live demo on real photos.

Photograph each item of an outfit separately, flat on a plain background:
    demo_photos/
      riya_outfit/   top.jpg  jeans.jpg  shoes.jpg
      ...
Optional closet mode: demo_closet/{tops,bottoms,shoes}/*.jpg -> picks the best combinations.

    python -m fashion_taste.demo rate   --photos demo_photos --work $FT_WORK [--gemini]
    python -m fashion_taste.demo closet --photos demo_closet --work $FT_WORK

Backgrounds are removed with `rembg` so photos look like Polyvore product images.
CLIP scores are shown as a 1-5 rating: the outfit's percentile among Polyvore test outfits.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .clip_utils import EmbeddingTable, clip_image_features, device, embed_paths, load_clip
from .config import CLIP_MODEL, add_path_args, work_paths
from .viz import collage

EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # iPhone: export HEIC as JPEG first


def clean_photos(src_dir, out_dir, remove_bg=True):
    """Background-removed, white-background copies of every photo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from rembg import remove
    except ImportError:
        remove = None
        if remove_bg:
            print("rembg not installed: using photos as-is (pip install rembg[gpu] onnxruntime-gpu)")
    from PIL import Image, ImageOps
    out = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        dst = out_dir / (p.stem + ".png")
        if not dst.exists():
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((1024, 1024))
            if remove and remove_bg:
                im = remove(im)
                bg = Image.new("RGB", im.size, "white")
                bg.paste(im, mask=im.split()[-1])
                bbox = im.split()[-1].getbbox()
                im = bg.crop(bbox) if bbox else bg
            im.save(dst)
        out.append(dst)
    return out


class Scorers:
    def __init__(self, P, clip_name):
        self.P = P
        dev = device()
        model, _ = load_clip(clip_name)
        self.zs = model.to(dev).eval()
        self.size = model.config.vision_config.image_size
        self.head = None
        if (P["ckpt"] / "head_best.pt").exists():
            from .train import Head
            self.head = Head(model.config.projection_dim).to(dev)
            self.head.load_state_dict(torch.load(P["ckpt"] / "head_best.pt", map_location=dev)["head"])
            self.head.eval()
        self.ft = None
        if (P["ckpt"] / "finetune_best.pt").exists():
            from .train import load_finetuned
            self.ft = load_finetuned(P)
        self.ref = self._reference_distributions()

    def _reference_distributions(self):
        """Scores of real Polyvore test outfits, to turn raw scores into percentiles."""
        f = self.P["results"] / "per_example_scores.csv"
        if not f.exists():
            return {}
        df = pd.read_csv(f)
        df = df[df["task"] == "compat"]
        return {m: np.sort(g["score"].dropna().values) for m, g in df.groupby("method")}

    @torch.no_grad()
    def embeddings(self, paths):
        out = {}
        zv, _ = embed_paths(lambda x: clip_image_features(self.zs, x), paths, self.size, 64, 0, "CLIP")
        out["CLIP zero-shot"] = zv
        if self.head is not None:
            out["CLIP + trained head"] = self.head(torch.tensor(zv, device=device())).cpu().numpy()
        if self.ft is not None:
            out["CLIP fine-tuned"], _ = embed_paths(self.ft, paths, self.size, 64, 0, "fine-tuned")
        return out

    def rating(self, method, raw):
        ref = self.ref.get(method)
        if ref is None or not len(ref):
            return None
        pct = np.searchsorted(ref, raw) / len(ref)
        return round(float(1 + 4 * pct), 1)


def pair_mean(V):
    S = V @ V.T
    n = len(V)
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def gemini_rate(paths):
    from .vlm import Gemini, parse, rating_prompt
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, ""
    g = Gemini(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), key, rpm=8)
    im = collage(paths, labels=list(range(1, len(paths) + 1)))
    text = g([im, rating_prompt("reason")])
    s = parse(text, "score")
    try:
        why = json.loads(text).get("reasoning", "")
    except Exception:
        why = ""
    return (round(1 + 4 * (s - 1) / 9, 1) if s else None), why


def rate(P, photos, clip_name, use_gemini, remove_bg):
    sc = Scorers(P, clip_name)
    rows = []
    for d in sorted(p for p in Path(photos).iterdir() if p.is_dir()):
        paths = clean_photos(d, P["demo"] / "clean" / d.name, remove_bg)
        if len(paths) < 2:
            continue
        row = {"outfit": d.name}
        for m, V in sc.embeddings(paths).items():
            raw = pair_mean(V)
            row[f"{m} (raw)"] = raw
            row[f"{m} (1-5)"] = sc.rating(m, raw)
        if use_gemini:
            row["Gemini (1-5)"], row["Gemini says"] = gemini_rate(paths)
        rows.append(row)
        collage(paths, title=d.name.replace("_", " ")).save(P["demo"] / f"{d.name}.jpg")
        print(row)
    df = pd.DataFrame(rows)
    df.to_csv(P["demo"] / "ratings.csv", index=False)
    print(f"\nsaved ratings and outfit images to {P['demo']}")
    return df


def closet(P, photos, clip_name, remove_bg, top_k=3):
    sc = Scorers(P, clip_name)
    groups = {d.name: clean_photos(d, P["demo"] / "closet_clean" / d.name, remove_bg)
              for d in sorted(Path(photos).iterdir()) if d.is_dir()}
    groups = {k: v for k, v in groups.items() if v}
    allp = [p for v in groups.values() for p in v]
    embs = sc.embeddings(allp)
    method = "CLIP fine-tuned" if "CLIP fine-tuned" in embs else list(embs)[-1]
    V = dict(zip(map(str, allp), embs[method]))
    combos = []
    for combo in itertools.product(*groups.values()):
        combos.append((pair_mean(np.stack([V[str(p)] for p in combo])), combo))
    combos.sort(key=lambda x: -x[0])
    for rank, (s, combo) in enumerate(combos[:top_k], 1):
        collage(list(combo), title=f"#{rank} by {method} (rating {sc.rating(method, s)})").save(P["demo"] / f"closet_best_{rank}.jpg")
    collage(list(combos[-1][1]), title=f"Worst combination by {method}").save(P["demo"] / "closet_worst.jpg")
    print(f"{len(combos)} combinations scored with {method}; best/worst saved to {P['demo']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["rate", "closet"])
    add_path_args(p)
    p.add_argument("--photos", required=True)
    p.add_argument("--clip", default=CLIP_MODEL)
    p.add_argument("--gemini", action="store_true")
    p.add_argument("--keep_bg", action="store_true", help="skip background removal")
    a = p.parse_args()
    P = work_paths(a.work)
    if a.cmd == "rate":
        rate(P, a.photos, a.clip, a.gemini, not a.keep_bg)
    else:
        closet(P, a.photos, a.clip, not a.keep_bg)


if __name__ == "__main__":
    main()
