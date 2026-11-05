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
    p = TSADParser()
    plt.rcParams.update({"font.size": 26})
    fig = plt.figure(figsize=(24, 10.0))
    gs = fig.add_gridspec(7, 1, height_ratios=[1.0, 0.75, 0.75, 0.18, 1.0, 0.75, 0.75], hspace=0.08)
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
            ax.plot(x, y, linewidth=1.6, color="#1f5fbf")
            ax.set_xlim(lo, hi - 1); ax.set_yticks([]); ax.margins(x=0)
            ax.text(0.985, 0.90, name, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"))
            if r is not None:
                print(f"{name_ts} {name}: VUS@{L} {r['vus_pr']:.3f} AutoVUS {r['autovus']:.3f} b* {[int(a[5]) for a in own]} {[a[7] for a in own]}")
                ax.text(0.01, 0.92, f"VUS@{L} = {r['vus_pr']:.3f}\nAutoVUS = {r['autovus']:.3f}",
                        transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.93, edgecolor="gray"))
            ax.set_xticks([])   # no x ticks or label (space; the axis is time)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=60)


if __name__ == "__main__":
    tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"case_study_tsb_L{L}{tag}")
    main(stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf")
