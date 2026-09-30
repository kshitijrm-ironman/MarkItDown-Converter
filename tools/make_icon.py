"""
Draw the MarkItDown app logo and write assets/logo.png + assets/markitdown.ico.

Pure Pillow, no SVG toolchain: run it whenever you want to tweak the mark.

    venv\\Scripts\\python.exe tools\\make_icon.py

Design: indigo -> violet -> sky gradient tile, a white document sheet with a
folded corner, and a cyan badge holding the "down" arrow of MarkItDown.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter

S = 1024          # design canvas
SS = 2            # supersampling factor for smooth edges
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

INDIGO = (79, 70, 229)
VIOLET = (124, 58, 237)
SKY = (14, 165, 233)
CYAN = (34, 211, 238)
SHEET = (255, 255, 255)
FOLD = (199, 210, 254)
INK = (100, 116, 139)
INK_SOFT = (148, 163, 184)


def _lerp(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def _gradient(size: int) -> Image.Image:
    """Diagonal 3-stop gradient, built row-major from a 1-D ramp."""
    ramp = []
    for i in range(2 * size - 1):
        t = i / (2 * size - 2)
        ramp.append(_lerp(INDIGO, VIOLET, t / 0.55) if t <= 0.55
                    else _lerp(VIOLET, SKY, (t - 0.55) / 0.45))
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = ramp[x + y]
    return img


def build(size: int = S * SS) -> Image.Image:
    k = size / S                      # design units -> pixels
    def u(*vals):                     # scale helper
        return [v * k for v in vals] if len(vals) > 1 else vals[0] * k

    # ---- background tile ------------------------------------------------
    tile = _gradient(size).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=u(232), fill=255)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(tile, (0, 0), mask)

    # soft top-left sheen
    sheen = Image.new("L", (size, size), 0)
    ImageDraw.Draw(sheen).ellipse(u(-260, -420, 780, 300), fill=48)
    sheen = sheen.filter(ImageFilter.GaussianBlur(u(40)))
    sheen = Image.composite(sheen, Image.new("L", (size, size), 0), mask)
    icon.paste(Image.new("RGBA", (size, size), (255, 255, 255, 255)), (0, 0), sheen)

    d = ImageDraw.Draw(icon)

    # ---- document sheet -------------------------------------------------
    d.rounded_rectangle(u(250, 180, 700, 830), radius=u(44), fill=SHEET)
    # cut the top-right corner away, then lay the fold over the cut
    d.polygon(u(556, 180, 716, 180, 716, 346), fill=(0, 0, 0, 0))
    d.polygon(u(556, 180, 700, 324, 556, 324), fill=FOLD)

    # text lines
    d.rounded_rectangle(u(318, 404, 566, 442), radius=u(19), fill=INK)
    d.rounded_rectangle(u(318, 486, 636, 518), radius=u(16), fill=INK_SOFT)
    d.rounded_rectangle(u(318, 556, 578, 588), radius=u(16), fill=INK_SOFT)

    # ---- badge + down arrow --------------------------------------------
    d.ellipse(u(470, 470, 930, 930), fill=CYAN, outline=SHEET, width=int(u(34)))
    d.rounded_rectangle(u(672, 574, 728, 734), radius=u(12), fill=SHEET)
    d.polygon(u(586, 690, 814, 690, 700, 838), fill=SHEET)

    return icon.resize((S, S), Image.LANCZOS) if size != S else icon


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    icon = build()
    png = os.path.join(ASSETS, "logo.png")
    ico = os.path.join(ASSETS, "markitdown.ico")
    icon.resize((512, 512), Image.LANCZOS).save(png)
    icon.save(ico, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)])
    print("wrote", png)
    print("wrote", ico)


if __name__ == "__main__":
    main()
