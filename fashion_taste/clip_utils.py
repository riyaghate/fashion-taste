"""CLIP loading, image preprocessing, and batched embedding with on-disk caching."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(name):
    """Returns (model, tokenizer). name='tiny-random' builds a small untrained model (tests only)."""
    from transformers import CLIPConfig, CLIPModel
    if name == "tiny-random":
        torch.manual_seed(0)
        cfg = CLIPConfig(
            text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, vocab_size=1000, max_position_embeddings=32),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=2, image_size=32, patch_size=8),
            projection_dim=16)
        return CLIPModel(cfg), _HashTokenizer()
    from transformers import CLIPTokenizer
    return CLIPModel.from_pretrained(name), CLIPTokenizer.from_pretrained(name)


class _HashTokenizer:
    """Stand-in tokenizer for the tiny test model (no downloads)."""
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", max_length=16):
        rows = [[1] + [hash(w) % 997 + 2 for w in t.lower().split()][: max_length - 2] + [999] for t in texts]
        n = max(map(len, rows))
        ids = torch.zeros(len(rows), n, dtype=torch.long)
        mask = torch.zeros(len(rows), n, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def image_transform(size, train=False):
    aug = [transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
           transforms.RandomHorizontalFlip()] if train else \
          [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(size)]
    return transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])


def load_rgb(path):
    """Open an image; transparent backgrounds become white (Polyvore style)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


class ImageFiles(Dataset):
    def __init__(self, paths, tf, size):
        self.paths, self.tf, self.size = list(paths), tf, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.tf(load_rgb(self.paths[i])), 1
        except Exception:  # missing/corrupt image -> flagged, excluded downstream
            return torch.zeros(3, self.size, self.size), 0


@torch.no_grad()
def embed_paths(image_encoder, paths, size, batch_size=256, workers=2, desc="embedding"):
    """image_encoder: callable(pixel_values)->(B,D) tensor. Returns L2-normalised float32 array + ok mask."""
    from tqdm.auto import tqdm
    dl = DataLoader(ImageFiles(paths, image_transform(size), size), batch_size=batch_size,
                    num_workers=workers, pin_memory=torch.cuda.is_available())
    dev = device()
    outs, oks = [], []
    for x, ok in tqdm(dl, desc=desc):
        with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
            e = image_encoder(x.to(dev, non_blocking=True)).float()
        outs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        oks.append(ok)
    return torch.cat(outs).numpy(), torch.cat(oks).numpy().astype(bool)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, batch_size=256):
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(texts[i: i + batch_size], padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(dev) for k, v in tok.items()}
        e = model.get_text_features(**tok)
        if not torch.is_tensor(e):  # newer transformers may return an output object
            e = e.pooler_output if hasattr(e, "pooler_output") else e[0]
        outs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(outs).numpy()


def clip_image_features(model, pixel_values):
    e = model.get_image_features(pixel_values=pixel_values)
    if not torch.is_tensor(e):
        e = e.pooler_output if hasattr(e, "pooler_output") else e[0]
    return e


class EmbeddingTable:
    """item_id -> unit vector, saved as <name>.npy + <name>_ids.json."""

    def __init__(self, ids, vecs, ok=None):
        self.ids = list(ids)
        self.vecs = vecs.astype(np.float32)
        self.ok = np.ones(len(ids), bool) if ok is None else ok
        self.index = {i: k for k, i in enumerate(self.ids)}

    def __contains__(self, item_id):
        k = self.index.get(item_id)
        return k is not None and self.ok[k]

    def __getitem__(self, item_ids):
        if isinstance(item_ids, str):
            return self.vecs[self.index[item_ids]]
        return self.vecs[[self.index[i] for i in item_ids]]

    def save(self, folder, name):
        folder = Path(folder)
        np.save(folder / f"{name}.npy", self.vecs.astype(np.float16))
        np.save(folder / f"{name}_ok.npy", self.ok)
        (folder / f"{name}_ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, folder, name):
        folder = Path(folder)
        ok_path = folder / f"{name}_ok.npy"
        return cls(json.loads((folder / f"{name}_ids.json").read_text()),
                   np.load(folder / f"{name}.npy"),
                   np.load(ok_path) if ok_path.exists() else None)

    @staticmethod
    def exists(folder, name):
        return (Path(folder) / f"{name}.npy").exists()
