"""Step 2: tag every item with 4 attributes (color, pattern, formality, season).

Each attribute is labelled twice, independently:
  (1) keyword rules on the item's title/description
  (2) CLIP zero-shot classification of the item image against text prompts
An attribute counts as *confident* only when both methods agree. Hard negatives are
built only from confident attributes, which filters out most metadata noise.

    python -m fashion_taste.attributes --data $FT_DATA --work $FT_WORK
Output: <work>/results/item_attributes.csv and attribute_agreement.csv
"""
import argparse
import re

import numpy as np
import pandas as pd

from . import data
from .clip_utils import EmbeddingTable, device, embed_texts, load_clip
from .config import CLIP_MODEL, RULES, add_path_args, work_paths

# ---------------------------------------------------------------- vocabulary
COLOR_WORDS = {
    "black": ["black", "jet", "onyx", "ebony"],
    "white": ["white", "ivory", "optic white", "off-white", "off white"],
    "grey": ["grey", "gray", "charcoal", "silver", "heather"],
    "beige": ["beige", "tan", "camel", "nude", "taupe", "sand", "cream", "khaki", "stone", "oatmeal"],
    "brown": ["brown", "chocolate", "cognac", "coffee", "mocha", "chestnut"],
    "navy": ["navy", "midnight"],
    "blue": ["blue", "cobalt", "sky", "azure", "denim blue", "royal"],
    "teal": ["teal", "turquoise", "aqua"],
    "red": ["red", "burgundy", "maroon", "wine", "crimson", "scarlet", "oxblood", "cherry"],
    "pink": ["pink", "blush", "rose", "fuchsia", "magenta", "hot pink"],
    "orange": ["orange", "coral", "rust", "tangerine", "terracotta", "peach"],
    "yellow": ["yellow", "mustard", "lemon", "canary"],
    "green": ["green", "olive", "emerald", "mint", "sage", "forest", "lime", "khaki green"],
    "purple": ["purple", "lilac", "lavender", "violet", "plum", "mauve", "grape"],
}
NEUTRALS = {"black", "white", "grey", "beige", "brown", "navy"}
HUE = {"red": 0, "orange": 30, "yellow": 55, "green": 120, "teal": 175, "blue": 220, "purple": 280, "pink": 330}

PATTERN_WORDS = ["floral", "flower", "print", "printed", "stripe", "striped", "stripes", "plaid", "check",
                 "checked", "gingham", "polka", "dot", "dots", "leopard", "animal print", "camo", "camouflage",
                 "paisley", "geometric", "tartan", "houndstooth", "graphic", "tie-dye", "tie dye", "zebra",
                 "snake", "snakeskin", "abstract", "aztec", "ikat", "chevron", "argyle", "logo", "slogan",
                 "patterned", "jacquard", "embroidered", "sequin", "sequined", "colorblock", "color block"]
FORMAL_WORDS = ["blazer", "suit", "tuxedo", "gown", "cocktail", "evening", "pump", "pumps", "stiletto",
                "stilettos", "heel", "heels", "oxford", "loafer", "loafers", "trousers", "pencil skirt",
                "tailored", "dress shirt", "satin", "silk", "sequin", "sequined", "formal", "office",
                "pearl", "clutch", "brogue", "brogues", "slingback", "court shoe", "waistcoat", "vest"]
CASUAL_WORDS = ["t-shirt", "tee", "hoodie", "hooded", "sweatshirt", "jogger", "joggers", "sweatpants",
                "sneaker", "sneakers", "trainer", "trainers", "legging", "leggings", "flip flop",
                "flip-flop", "slide", "slides", "track", "athletic", "sport", "sports", "running", "gym",
                "cargo", "distressed", "ripped", "denim shorts", "graphic", "crocs", "workout", "yoga",
                "baseball", "trucker", "boyfriend jeans"]
SUMMER_WORDS = ["sandal", "sandals", "shorts", "tank", "sleeveless", "linen", "swim", "swimsuit", "bikini",
                "flip flop", "flip-flop", "espadrille", "espadrilles", "crop top", "straw", "sundress",
                "cami", "camisole", "beach", "romper", "short sleeve", "short-sleeve", "slides"]
WINTER_WORDS = ["wool", "woolen", "cashmere", "knit", "knitted", "sweater", "jumper", "coat", "parka",
                "puffer", "down jacket", "fleece", "boot", "boots", "shearling", "fur", "faux fur",
                "turtleneck", "roll neck", "thermal", "beanie", "corduroy", "velvet", "cardigan", "mohair",
                "quilted", "padded", "teddy", "sherpa", "snow", "uggs"]

CLIP_PROMPTS = {
    "color": {c: [f"a photo of a {c} piece of clothing", f"a product photo of a {c} garment"] for c in COLOR_WORDS},
    "pattern": {"solid": ["a photo of a plain solid-colored piece of clothing", "a product photo of a garment with no pattern"],
                "patterned": ["a photo of a piece of clothing with a busy print or pattern", "a product photo of a striped, floral or printed garment"]},
    "formality": {"formal": ["a photo of formal, dressy clothing", "a product photo of elegant office or evening wear"],
                  "casual": ["a photo of casual, sporty clothing", "a product photo of relaxed streetwear or athleisure"]},
    "season": {"summer": ["a photo of lightweight summer clothing", "a product photo of warm-weather beachwear"],
               "winter": ["a photo of warm winter clothing", "a product photo of heavy cold-weather wear"]},
}


def _has(text, words):
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text) for w in words)


def keyword_labels(title, full_text):
    """Keyword-rule label for each attribute (None = undecided)."""
    lab = {}
    found = [c for c, ws in COLOR_WORDS.items() if _has(title, ws)]
    if not found:
        found = [c for c, ws in COLOR_WORDS.items() if _has(full_text, ws)]
    lab["color"] = found[0] if len(found) == 1 else None
    lab["pattern"] = "patterned" if _has(full_text, PATTERN_WORDS) else "solid"
    f, c = _has(full_text, FORMAL_WORDS), _has(full_text, CASUAL_WORDS)
    lab["formality"] = "formal" if f and not c else "casual" if c and not f else "neutral" if not (f or c) else None
    s, w = _has(full_text, SUMMER_WORDS), _has(full_text, WINTER_WORDS)
    lab["season"] = "summer" if s and not w else "winter" if w and not s else "neutral" if not (s or w) else None
    return lab


def clip_labels(img_vecs, text_vecs_by_attr, neutral_margin=0.65):
    """Zero-shot labels. For 2-way attributes a low-confidence prediction becomes 'neutral'
    (formality/season), mirroring the keyword rule's 'no words found' case."""
    out = {}
    for attr, (classes, T) in text_vecs_by_attr.items():
        logits = 100.0 * img_vecs @ T.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        top = p.argmax(1)
        labels = np.array(classes, dtype=object)[top]
        if attr in ("formality", "season"):
            labels = np.where(p.max(1) >= neutral_margin, labels, "neutral")
        out[attr] = labels
    return out


def prompt_matrix(model, tok):
    res = {}
    for attr, cls in CLIP_PROMPTS.items():
        classes = list(cls)
        T = []
        for c in classes:
            v = embed_texts(model, tok, cls[c]).mean(0)
            T.append(v / np.linalg.norm(v))
        res[attr] = (classes, np.stack(T))
    return res


def tag_items(ds, emb, model, tok, neutral_margin=0.65):
    ids = [i for i in emb.ids if i in emb]
    kw = [keyword_labels(ds.title(i).lower(), ds.text(i)) for i in ids]
    cl = clip_labels(emb[ids], prompt_matrix(model, tok), neutral_margin)
    rows = []
    for k, iid in enumerate(ids):
        r = {"item_id": iid, "category": ds.category(iid), "title": ds.title(iid)}
        for a in RULES:
            r[f"{a}_kw"] = kw[k][a]
            r[f"{a}_clip"] = cl[a][k]
            r[a] = kw[k][a] if kw[k][a] is not None and kw[k][a] == cl[a][k] else None
        rows.append(r)
    return pd.DataFrame(rows)


def agreement_table(df):
    rows = []
    for a in RULES:
        has_kw = df[f"{a}_kw"].notna()
        rows.append({"attribute": a, "items_with_keyword_label": int(has_kw.sum()),
                     "keyword_clip_agreement": float((df.loc[has_kw, f"{a}_kw"] == df.loc[has_kw, f"{a}_clip"]).mean()),
                     "confident_items": int(df[a].notna().sum()),
                     "confident_fraction": float(df[a].notna().mean())})
    return pd.DataFrame(rows)


def main():
    p = add_path_args(argparse.ArgumentParser())
    p.add_argument("--clip", default=CLIP_MODEL)
    p.add_argument("--neutral_margin", type=float, default=0.65)
    a = p.parse_args()
    P = work_paths(a.work)
    ds = data.load(a.data, a.split)
    emb = EmbeddingTable.load(P["emb"], "clip_zeroshot")
    model, tok = load_clip(a.clip)
    model = model.to(device()).eval()
    df = tag_items(ds, emb, model, tok, a.neutral_margin)
    df.to_csv(P["results"] / "item_attributes.csv", index=False)
    ag = agreement_table(df)
    ag.to_csv(P["results"] / "attribute_agreement.csv", index=False)
    print(ag.to_string(index=False))


if __name__ == "__main__":
    main()
