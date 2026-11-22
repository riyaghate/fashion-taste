"""Loading the Polyvore Outfits dataset (Vasileva et al., ECCV 2018).

Expected layout (the official download, extracted):
  polyvore_outfits/
    images/<item_id>.jpg
    polyvore_item_metadata.json
    nondisjoint/{train,valid,test}.json
    nondisjoint/compatibility_{train,valid,test}.txt
    nondisjoint/fill_in_blank_{train,valid,test}.json
    disjoint/...   (same files)

Run `python -m fashion_taste.data --data <folder>` to check the folder is complete.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import add_path_args


@dataclass
class Polyvore:
    root: Path
    split: str
    meta: dict                                   # item_id -> metadata dict
    outfits: dict = field(default_factory=dict)  # split name -> {set_id: [item_id, ...]}
    key2item: dict = field(default_factory=dict) # "setid_index" -> item_id (all splits)

    # ---- item helpers -------------------------------------------------
    def category(self, item_id):
        return self.meta.get(item_id, {}).get("semantic_category", "unknown")

    def text(self, item_id):
        m = self.meta.get(item_id, {})
        return " ".join(str(m.get(k, "") or "") for k in ("title", "url_name", "description")).lower()

    def title(self, item_id):
        m = self.meta.get(item_id, {})
        return (m.get("title") or m.get("url_name") or item_id).strip()

    def image_path(self, item_id):
        return self.root / "images" / f"{item_id}.jpg"

    # ---- benchmark tasks ---------------------------------------------
    def compatibility(self, which="test"):
        """List of (label, [item_ids], question_id). label 1 = real outfit, 0 = random."""
        out = []
        path = self.root / self.split / f"compatibility_{which}.txt"
        for i, line in enumerate(path.read_text().splitlines()):
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            items = [self.key2item[k] for k in parts[1:] if k in self.key2item]
            if len(items) >= 2:
                out.append((label, items, f"compat_{which}_{i}"))
        return out

    def fitb(self, which="test"):
        """Fill-in-the-blank questions: dict(question=[ids], answers=[4 ids], correct=int, qid)."""
        data = json.loads((self.root / self.split / f"fill_in_blank_{which}.json").read_text())
        out = []
        for i, q in enumerate(data):
            q_keys, a_keys = q["question"], q["answers"]
            q_set = q_keys[0].split("_")[0]
            correct = [k.split("_")[0] == q_set for k in a_keys]
            if sum(correct) != 1:
                continue
            try:
                question = [self.key2item[k] for k in q_keys]
                answers = [self.key2item[k] for k in a_keys]
            except KeyError:
                continue
            out.append(dict(question=question, answers=answers, correct=correct.index(True), qid=f"fitb_{which}_{i}"))
        return out


def load(root, split="nondisjoint"):
    root = Path(root)
    meta = json.loads((root / "polyvore_item_metadata.json").read_text())
    ds = Polyvore(root=root, split=split, meta=meta)
    for which in ("train", "valid", "test"):
        f = root / split / f"{which}.json"
        if not f.exists():
            continue
        outfits = {}
        for o in json.loads(f.read_text()):
            sid = str(o["set_id"])
            items = []
            for it in o["items"]:
                iid = str(it["item_id"])
                ds.key2item[f"{sid}_{it['index']}"] = iid
                items.append(iid)
            outfits[sid] = items
        ds.outfits[which] = outfits
    return ds


def check(root, split):
    root = Path(root)
    need = [root / "polyvore_item_metadata.json", root / "images"]
    need += [root / split / n for n in ("train.json", "valid.json", "test.json",
                                         "compatibility_test.txt", "fill_in_blank_test.json")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print("MISSING:\n  " + "\n  ".join(missing))
        return False
    ds = load(root, split)
    n_img = sum(1 for _ in (root / "images").glob("*.jpg"))
    print(f"OK  items={len(ds.meta):,}  images={n_img:,}")
    for k, v in ds.outfits.items():
        print(f"    {k}: {len(v):,} outfits")
    print(f"    compatibility test questions: {len(ds.compatibility('test')):,}")
    print(f"    fill-in-the-blank test questions: {len(ds.fitb('test')):,}")
    return True


if __name__ == "__main__":
    p = add_path_args(argparse.ArgumentParser())
    a = p.parse_args()
    raise SystemExit(0 if check(a.data, a.split) else 1)
