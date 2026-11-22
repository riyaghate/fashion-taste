"""Step 4: train compatibility models on Polyvore training outfits.

Objective (both variants): contrastive InfoNCE over item pairs. For each training outfit we
sample two items; they are a positive pair, and items from the other outfits in the batch are
negatives. Items that are worn together end up close in the learned space. An outfit's score
is the mean pairwise cosine similarity of its items (the same rule used for zero-shot CLIP).

  head      - a small MLP trained on top of FROZEN zero-shot CLIP features (minutes)
  finetune  - the CLIP image encoder AND the head trained end-to-end (hours on a T4)

    python -m fashion_taste.train head     --data $FT_DATA --work $FT_WORK
    python -m fashion_taste.train finetune --data $FT_DATA --work $FT_WORK

Checkpoints are written every epoch to <work>/checkpoints and runs resume automatically.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from . import data
from .clip_utils import (EmbeddingTable, clip_image_features, device, embed_paths, image_transform,
                         load_clip, load_rgb)
from .config import CLIP_MODEL, SEED, add_path_args, work_paths
from .scoring import outfit_scores


class Head(nn.Module):
    """Residual MLP: starts as (almost) identity, so training begins from zero-shot CLIP."""

    def __init__(self, dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = F.normalize(x.float(), dim=-1)
        return F.normalize(x + self.net(x), dim=-1)


def info_nce(a, b, logit_scale):
    logits = logit_scale.exp().clamp(max=100) * a @ b.T
    target = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def train_pairs(ds, ok, rng):
    """One random item pair per training outfit (only items with valid images)."""
    pairs = []
    for items in ds.outfits["train"].values():
        items = [i for i in items if ok(i)]
        if len(items) >= 2:
            pairs.append(tuple(rng.sample(items, 2)))
    rng.shuffle(pairs)
    return pairs


def valid_auc(ds, table, n=2000):
    qs = [(l, it) for l, it, _ in ds.compatibility("valid") if all(i in table for i in it)]
    random.Random(SEED).shuffle(qs)
    qs = qs[:n]
    s = outfit_scores(table, [it for _, it in qs])
    return roc_auc_score([l for l, _ in qs], s)


def _ckpt_path(P, name):
    return P["ckpt"] / f"{name}.pt"


def _save(P, name, state):
    tmp = _ckpt_path(P, name).with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(_ckpt_path(P, name))


# --------------------------------------------------------------------------- head
def train_head(ds, P, epochs=30, batch=512, lr=1e-3, seed=SEED):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    zs = EmbeddingTable.load(P["emb"], "clip_zeroshot")
    dev = device()
    X = torch.tensor(zs.vecs, device=dev)
    head = Head(X.shape[1]).to(dev)
    scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    opt = torch.optim.AdamW(list(head.parameters()) + [scale], lr=lr, weight_decay=1e-4)
    start, best, log = 0, -1.0, []
    if _ckpt_path(P, "head").exists():
        st = torch.load(_ckpt_path(P, "head"), map_location=dev)
        head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); scale.data.fill_(st["scale"])
        start, best, log = st["epoch"] + 1, st["best"], st["log"]
        print(f"resuming head training from epoch {start}")
    base = valid_auc(ds, zs)
    print(f"zero-shot CLIP validation AUC: {base:.4f}")
    for ep in range(start, epochs):
        head.train()
        pairs = train_pairs(ds, lambda i: i in zs, rng)
        losses = []
        for k in range(0, len(pairs) - batch + 1, batch):
            a_ids, b_ids = zip(*pairs[k: k + batch])
            a = head(X[[zs.index[i] for i in a_ids]])
            b = head(X[[zs.index[i] for i in b_ids]])
            loss = info_nce(a, b, scale)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        table = project_table(head, zs)
        auc = valid_auc(ds, table)
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "valid_auc": auc})
        print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  valid AUC {auc:.4f}")
        if auc > best:
            best = auc
            torch.save({"head": head.state_dict()}, P["ckpt"] / "head_best.pt")
            table.save(P["emb"], "clip_head")
        _save(P, "head", {"head": head.state_dict(), "opt": opt.state_dict(), "scale": scale.item(),
                          "epoch": ep, "best": best, "log": log})
    (P["results"] / "train_log_head.json").write_text(json.dumps({"zero_shot_valid_auc": base, "log": log}, indent=2))
    print(f"best validation AUC {best:.4f}; embeddings saved as clip_head")


@torch.no_grad()
def project_table(head, table):
    head.eval()
    dev = next(head.parameters()).device
    out = []
    for k in range(0, len(table.vecs), 8192):
        out.append(head(torch.tensor(table.vecs[k: k + 8192], device=dev)).cpu())
    return EmbeddingTable(table.ids, torch.cat(out).numpy(), table.ok)


# ----------------------------------------------------------------------- finetune
class PairImages(Dataset):
    def __init__(self, ds, pairs, size):
        self.ds, self.pairs, self.tf = ds, pairs, image_transform(size, train=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        a, b = self.pairs[k]
        return self.tf(load_rgb(self.ds.image_path(a))), self.tf(load_rgb(self.ds.image_path(b)))


class FineTuned(nn.Module):
    def __init__(self, clip, dim):
        super().__init__()
        self.clip, self.head = clip, Head(dim)

    def forward(self, pixels):
        return self.head(clip_image_features(self.clip, pixels))


def embed_all(model, ds, ids, size, batch, workers, desc):
    model.eval()
    vecs, ok = embed_paths(model, [ds.image_path(i) for i in ids], size, batch, workers, desc)
    return EmbeddingTable(ids, vecs, ok)


def train_finetune(ds, P, clip_name, epochs=5, batch=128, lr=1e-5, head_lr=1e-3, workers=2,
                   unfreeze_last=None, max_hours=None, seed=SEED):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = device()
    clip, _ = load_clip(clip_name)
    clip.text_model.requires_grad_(False)
    dim = clip.config.projection_dim
    size = clip.config.vision_config.image_size
    model = FineTuned(clip, dim).to(dev)
    if unfreeze_last is not None:  # optionally train only the last N vision blocks
        clip.vision_model.requires_grad_(False)
        layers = clip.vision_model.encoder.layers
        for l in layers[len(layers) - unfreeze_last:]:
            l.requires_grad_(True)
        clip.vision_model.post_layernorm.requires_grad_(True)
    scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=dev))
    backbone = [p for n, p in model.named_parameters() if n.startswith("clip.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr},
                             {"params": list(model.head.parameters()) + [scale], "lr": head_lr}], weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    zs = EmbeddingTable.load(P["emb"], "clip_zeroshot")
    ok = lambda i: i in zs
    start, best, log = 0, -1.0, []
    if _ckpt_path(P, "finetune").exists():
        st = torch.load(_ckpt_path(P, "finetune"), map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); scale.data.fill_(st["scale"])
        start, best, log = st["epoch"] + 1, st["best"], st["log"]
        print(f"resuming fine-tuning from epoch {start}")
    valid_ids = sorted({i for _, it, _ in ds.compatibility("valid") for i in it if ok(i)})
    t0 = time.time()
    from tqdm.auto import tqdm
    for ep in range(start, epochs):
        model.train()
        pairs = train_pairs(ds, ok, rng)
        dl = DataLoader(PairImages(ds, pairs, size), batch_size=batch, shuffle=False, num_workers=workers,
                        drop_last=True, pin_memory=dev == "cuda", persistent_workers=workers > 0)
        losses = []
        for a, b in tqdm(dl, desc=f"fine-tune epoch {ep}"):
            a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                ea, eb = model(a), model(b)
            loss = info_nce(ea.float(), eb.float(), scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        vt = embed_all(model, ds, valid_ids, size, 256, workers, "validation embeddings")
        auc = valid_auc(ds, vt)
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "valid_auc": auc})
        print(f"epoch {ep}  loss {np.mean(losses):.4f}  valid AUC {auc:.4f}")
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "clip_name": clip_name, "dim": dim}, P["ckpt"] / "finetune_best.pt")
        _save(P, "finetune", {"model": model.state_dict(), "opt": opt.state_dict(), "scale": scale.item(),
                              "epoch": ep, "best": best, "log": log})
        (P["results"] / "train_log_finetune.json").write_text(json.dumps({"log": log}, indent=2))
        if max_hours and (time.time() - t0) / 3600 > max_hours:
            print(f"stopping after {max_hours}h (re-run the same command to continue)")
            break
    print("embedding all items with the best fine-tuned model...")
    st = torch.load(P["ckpt"] / "finetune_best.pt", map_location=dev)
    model.load_state_dict(st["model"])
    ids = sorted(set(ds.key2item.values()))
    embed_all(model, ds, ids, size, 256, workers, "fine-tuned CLIP").save(P["emb"], "clip_finetuned")
    print(f"best validation AUC {best:.4f}; embeddings saved as clip_finetuned")


def load_finetuned(P, device_=None):
    """For the demo: rebuild the fine-tuned model from its best checkpoint."""
    dev = device_ or device()
    st = torch.load(P["ckpt"] / "finetune_best.pt", map_location=dev)
    clip, _ = load_clip(st["clip_name"])
    m = FineTuned(clip, st["dim"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["head", "finetune"])
    add_path_args(p)
    p.add_argument("--clip", default=CLIP_MODEL)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--unfreeze_last", type=int, default=None, help="fine-tune only the last N vision blocks")
    p.add_argument("--max_hours", type=float, default=None, help="stop after this long (resumable)")
    a = p.parse_args()
    P = work_paths(a.work)
    ds = data.load(a.data, a.split)
    if a.which == "head":
        train_head(ds, P, epochs=a.epochs or 30, batch=a.batch or 512, lr=a.lr or 1e-3)
    else:
        train_finetune(ds, P, a.clip, epochs=a.epochs or 5, batch=a.batch or 128, lr=a.lr or 1e-5,
                       workers=a.workers, unfreeze_last=a.unfreeze_last, max_hours=a.max_hours)


if __name__ == "__main__":
    main()
