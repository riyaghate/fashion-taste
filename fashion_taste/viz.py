"""Outfit collages (used for Gemini prompts, the survey, validation booklet and demo)."""
import base64
import io
import math

from PIL import Image, ImageDraw, ImageFont

from .clip_utils import load_rgb


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(im, cell):
    im = im.copy()
    im.thumbnail((cell, cell), Image.LANCZOS)
    canvas = Image.new("RGB", (cell, cell), "white")
    canvas.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2))
    return canvas


def collage(paths, labels=None, highlight=None, cell=256, cols=None, pad=12, title=None):
    """Grid of item images on white. labels: text drawn under each cell; highlight: index to outline red."""
    n = len(paths)
    cols = cols or min(n, 4 if n > 4 else n)
    rows = math.ceil(n / cols)
    lab_h = 30 if labels else 0
    top = 40 if title else 0
    W = cols * (cell + pad) + pad
    H = rows * (cell + pad + lab_h) + pad + top
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    f = _font(max(14, cell // 11))
    if title:
        d.text((pad, 8), title, fill="black", font=_font(22))
    for k, p in enumerate(paths):
        r, c = divmod(k, cols)
        x, y = pad + c * (cell + pad), top + pad + r * (cell + pad + lab_h)
        try:
            im = _fit(load_rgb(p), cell)
        except Exception:
            im = Image.new("RGB", (cell, cell), (235, 235, 235))
        out.paste(im, (x, y))
        d.rectangle([x - 1, y - 1, x + cell, y + cell], outline=(200, 200, 200), width=1)
        if highlight is not None and k == highlight:
            d.rectangle([x - 5, y - 5, x + cell + 4, y + cell + 4], outline=(220, 30, 30), width=5)
        if labels:
            d.text((x + 4, y + cell + 4), str(labels[k]), fill="black", font=f)
    return out


def to_jpeg_bytes(im, quality=85):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def to_data_uri(im, quality=80):
    return "data:image/jpeg;base64," + base64.b64encode(to_jpeg_bytes(im, quality)).decode()
