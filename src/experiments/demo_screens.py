"""Demo paper, Figure 2: the views of the demo, five panels in two rows.

Input: the captures of src/experiments/demo_screens_capture.mjs (experiments/figures/demo_screens/*.png,
1600 CSS px wide at device scale 2). Each panel is cropped to a box in CSS pixels, the panels of a row are
scaled to one height so that the row fills the figure width, and a bold caption goes under each panel.

Usage: conda run -n autovus python src/experiments/demo_screens.py
Writes experiments/figures/demo_screens.jpg and prints the size of every panel.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "experiments", "figures", "demo_screens")
OUT = os.path.join(ROOT, "experiments", "figures", "demo_screens.jpg")
SCALE = 2                                   # device pixels per CSS pixel in the captures
WIDTH, GAP, CAP, FONT = 3000, 36, 64, 44    # figure width, gap between panels, caption band, caption size (px)
BORDER = (160, 160, 160)

# (capture, crop box in CSS px (left, top, right, bottom), caption), one list per row
ROWS = [
    [("play", (310, 98, 1290, 585), "(a) Play"),
     ("explore", (24, 190, 1576, 942), "(b) Explore, with the buffer panel")],
    [("cases", (364, 162, 1496, 600), "(c) Cases"),
     ("rankings", (364, 118, 1492, 797), "(d) Rankings, per series and per anomaly"),
     ("profiles", (104, 264, 1496, 1230), "(e) Profiles")],
]


def font():
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, FONT)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    rows = []
    for row in ROWS:
        crops = [(Image.open(os.path.join(SRC, f"{name}.png")).convert("RGB").crop(tuple(v * SCALE for v in box)), cap)
                 for name, box, cap in row]
        # one height h for the row: sum of widths at height h, plus the gaps, equals WIDTH
        h = round((WIDTH - GAP * (len(crops) - 1)) / sum(im.width / im.height for im, _ in crops))
        rows.append([(im.resize((round(im.width * h / im.height), h), Image.LANCZOS), cap) for im, cap in crops])
    H = sum(r[0][0].height + CAP for r in rows) + GAP * (len(rows) - 1)
    fig = Image.new("RGB", (WIDTH, H), "white")
    draw, f = ImageDraw.Draw(fig), font()
    y = 0
    for r in rows:
        x = 0
        for im, cap in r:
            fig.paste(im, (x, y))
            draw.rectangle([x, y, x + im.width - 1, y + im.height - 1], outline=BORDER, width=2)
            tw = draw.textlength(cap, font=f)
            draw.text((x + (im.width - tw) / 2, y + im.height + 8), cap, fill="black", font=f)
            print(f"{cap}: {im.width} x {im.height}")
            x += im.width + GAP
        y += r[0][0].height + CAP + GAP
    fig.save(OUT, quality=92)
    print(f"saved: {OUT} ({WIDTH} x {H})")


if __name__ == "__main__":
    main()
