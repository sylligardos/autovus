"""Sec V-A figure variant at L=16: two real TSB-AD cases, two detectors each.

(a) 798_YAHOO_id_248 (15 point/short anomalies): Chronos vs Sub_KNN
(b) 282_NEK_id_6 (7 anomalies): OmniAnomaly vs Sub_PCA

Scores are computed here with the same calls as the demo export (app/export.py one_series,
paper AutoVUS), at L=16. Pink = label.
Pink = label, faded orange = the buffers AutoVUS selects per anomaly.
Usage: conda run -n autovus python src/experiments/plot_case_study_buffers.py [TAG]
Writes experiments/figures/case_study_tsb_L{L}[_TAG].pdf and .png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "app", "api"))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
from export import one_series  # noqa: E402

L = 20
CASES = [
    ("798_YAHOO_id_248", "798_YAHOO_id_248_WebService_tr_500_1st_105.csv", ["Chronos", "Sub_KNN"]),
    ("282_NEK_id_6", "282_NEK_id_6_WebService_tr_500_1st_64.csv", ["OmniAnomaly", "Sub_PCA"]),
]
PINK = "#d62728"  # red (was hot pink #ff69b4)
ORANGE = "#ff8c00"


def span(ax, a, b, color, alpha, minw):
    """Shade [a, b] (inclusive indices), widened to minw so point anomalies stay visible."""
    a, b = a - 0.5, b + 0.5
    if b - a < minw:
        c = (a + b) / 2; a, b = c - minw / 2, c + minw / 2
    ax.axvspan(a, b, facecolor=color, alpha=alpha, edgecolor="none")


def main(out_pdf, out_png=None):
    """Drawn in the style and at the scale of Fig 1(b) (src/experiments/plot_fig1b.py): same width, font sizes,
    frame edges and colours (experiments/figures/fig1a_layout.json), scores in the left margin."""
    import json
    from matplotlib.transforms import Bbox, blended_transform_factory
    lay = json.load(open(os.path.join(ROOT, "experiments", "figures", "fig1a_layout.json")))
    FS = lay["font_size"]
    W = lay["width"]                    # one column, same print scale as Fig 1
    LEFT = 0.54                         # smallest left margin (in) that fits "AutoVUS" (0.49 in) + gap
    BOX_FS, NAME_FS = 8, 9
    rows_in = [0.3, 0.45, 0.45, 0.1, 0.3, 0.45, 0.45]; gap_in = 0.04
    H = sum(rows_in) + (len(rows_in) - 1) * gap_in + 0.05
    p = TSADParser()
    plt.rcParams.update({"font.size": FS, "text.color": "0.15", "axes.labelcolor": "0.15", "xtick.color": "0.15"})
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(rows_in), 1, height_ratios=rows_in, hspace=gap_in / np.mean(rows_in),
                          left=LEFT / W, right=1 - lay["ax_right"] / W, top=1 - 0.025 / H, bottom=0.025 / H)
    rows = [0, 1, 2, 4, 5, 6]
    for ci, (name_ts, file, dets) in enumerate(CASES):
        values, label = load_series(p, file)
        _, _, _, res, anoms = one_series((file, L, dets))
        lo, hi = 0, len(label); x = np.arange(hi); minw = 0.004 * hi
        panels = [(name_ts, values, None)] + [(d, load_score(p, file, d), res[d]) for d in dets]
        for ri, (name, y, r) in enumerate(panels):
            ax = fig.add_subplot(gs[rows[3 * ci + ri]])
            own = [a for a in anoms if a[1] == (name if r is not None else dets[0])]
            for _, _, _, s, e, bl, br, _ in own:
                if r is not None and (bl > 0 or br > 0):
                    span(ax, s - bl, e + br, ORANGE, 0.25, minw)
                span(ax, s, e, PINK, 0.5, minw)
            ax.plot(x, y, linewidth=0.8, color="#1f5fbf", zorder=3)
            for sp in ax.spines.values():  # same frame as Fig 1
                sp.set_edgecolor(lay["edge_color"]); sp.set_linewidth(lay["edge_width"])
            ax.set_xlim(lo, hi - 1); ax.set_yticks([]); ax.margins(x=0)
            ax.text(0.99, 0.94, name, transform=ax.transAxes, fontsize=NAME_FS, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))
            if r is not None:
                print(f"{name_ts} {name}: VUS@{L} {r['vus_pr']:.3f} AutoVUS {r['autovus']:.3f} b* {[int(a[5]) for a in own]} {[a[7] for a in own]}")
                tr = blended_transform_factory(fig.transFigure, ax.transAxes)
                for k, (m, v) in enumerate([(f"VUS@{L}", r["vus_pr"]), ("AutoVUS", r["autovus"])]):
                    yk = 0.97 - 0.5 * k
                    ax.text(0.0, yk, m, transform=tr, fontsize=BOX_FS, color="0.45", ha="left", va="top")
                    ax.text(0.0, yk - 0.23, f"{v:.3f}", transform=tr, fontsize=NAME_FS, color="black", ha="left", va="top")
                for yb in (0.0, 1.0):  # dashed separators in the frame colour, as in Fig 1(b)
                    ax.plot([0.0, LEFT / W], [yb, yb], transform=tr, clip_on=False, linestyle="--",
                            color=lay["edge_color"], linewidth=lay["edge_width"])
            ax.set_xticks([])   # no x ticks or label (space; the axis is time)
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox([[0, tb.y0], [W, tb.y1]])  # full width, as Fig 1, so both print at the same scale
    fig.savefig(out_pdf, bbox_inches=box)
    if out_png:
        fig.savefig(out_png, bbox_inches=box, dpi=250)


if __name__ == "__main__":
    tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"case_study_tsb_L{L}{tag}")
    main(stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf")
