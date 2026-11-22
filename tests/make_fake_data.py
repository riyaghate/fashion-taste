"""Generate a tiny fake dataset in the exact Polyvore Outfits format (for offline smoke tests)."""
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"black": (20, 20, 20), "white": (245, 245, 245), "navy": (20, 30, 90), "red": (200, 30, 30),
          "green": (30, 150, 50), "purple": (120, 40, 160), "blue": (40, 90, 220), "beige": (220, 200, 160),
          "pink": (240, 130, 170), "orange": (240, 130, 30)}
CATS = {"tops": ["blouse", "t-shirt", "sweater", "tank"], "bottoms": ["trousers", "jeans", "shorts", "pencil skirt"],
        "shoes": ["sneakers", "heels", "boots", "sandals"], "outerwear": ["coat", "blazer", "hoodie"],
        "bags": ["clutch", "tote"]}


def main(root, n_items=1500, n_outfits=(400, 80, 200), seed=0):
    rng = random.Random(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    meta, by_cat = {}, {}
    for k in range(n_items):
        cat = rng.choice(list(CATS))
        color = rng.choice(list(COLORS))
        noun = rng.choice(CATS[cat])
        pat = rng.random() < 0.3
        iid = str(100000 + k)
        title = f"{color} {'striped ' if pat else ''}{noun}"
        meta[iid] = {"title": title, "url_name": title, "description": "", "semantic_category": cat}
        im = Image.new("RGB", (64, 64), COLORS[color])
        if pat:
            d = ImageDraw.Draw(im)
            for x in range(0, 64, 8):
                d.rectangle([x, 0, x + 3, 64], fill=(255, 255, 255))
        im.save(root / "images" / f"{iid}.jpg")
        by_cat.setdefault(cat, []).append(iid)
    (root / "polyvore_item_metadata.json").write_text(json.dumps(meta))
    split = root / "nondisjoint"
    split.mkdir(exist_ok=True)
    sid = 1000
    for which, n in zip(("train", "valid", "test"), n_outfits):
        outfits, compat, fitb = [], [], []
        for _ in range(n):
            sid += 1
            cats = ["tops", "bottoms", "shoes"] + rng.sample(["outerwear", "bags"], rng.randint(0, 2))
            items = [rng.choice(by_cat[c]) for c in cats]
            outfits.append({"set_id": str(sid), "items": [{"item_id": i, "index": j + 1} for j, i in enumerate(items)]})
        keys_by_cat = {}
        for o in outfits:
            for it in o["items"]:
                keys_by_cat.setdefault(meta[it["item_id"]]["semantic_category"], []).append(f"{o['set_id']}_{it['index']}")
        all_keys = [k for v in keys_by_cat.values() for k in v]
        for o in outfits:
            keys = [f"{o['set_id']}_{it['index']}" for it in o["items"]]
            compat.append("1 " + " ".join(keys))
            compat.append("0 " + " ".join(rng.sample(all_keys, len(keys))))
            blank = rng.randrange(len(keys))
            cat = meta[o["items"][blank]["item_id"]]["semantic_category"]
            wrong = [k for k in keys_by_cat[cat] if not k.startswith(o["set_id"] + "_")]
            answers = [keys[blank]] + rng.sample(wrong, 3)
            rng.shuffle(answers)
            fitb.append({"question": [k for j, k in enumerate(keys) if j != blank], "answers": answers,
                         "blank_position": blank + 1})
        (split / f"{which}.json").write_text(json.dumps(outfits))
        (split / f"compatibility_{which}.txt").write_text("\n".join(compat))
        (split / f"fill_in_blank_{which}.json").write_text(json.dumps(fitb))
    print("fake dataset at", root)


if __name__ == "__main__":
    main(sys.argv[1])
