"""Figure 1(b): 704_YAHOO_id_154, LOF vs SAND at L=20, in the template of Fig 12
(src/experiments/plot_case_study_buffers.py).

Pink = label, dashed grey lines = edges of the fixed VUS buffer (+-L), faded orange = the buffers AutoVUS selects per anomaly. Boxes: AUC-PR, VUS-PR at L,
AutoVUS (paper rule, symmetric), all on the full series.
Usage: conda run -n autovus python src/experiments/plot_fig1b.py
Writes experiments/figures/fig1b_L20.pdf and .png.
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
NAME, FILE, DETS = "704_YAHOO_id_154", "704_YAHOO_id_154_WebService_tr_500_1st_541.csv", ["LOF", "SAND"]
PINK = "#ff69b4"
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
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.9, 0.9], hspace=0.08)
    values, label = load_series(p, FILE)
    _, _, _, res, anoms = one_series((FILE, L, DETS))
    lo, hi = 0, len(label); x = np.arange(hi); minw = 0.004 * hi
    panels = [(NAME, values, None)] + [(d, load_score(p, FILE, d), res[d]) for d in DETS]
    for ri, (name, y, r) in enumerate(panels):
        ax = fig.add_subplot(gs[ri])
        own = [a for a in anoms if a[1] == (name if r is not None else DETS[0])]
        for _, _, _, s, e, bl, br, _ in own:
            if r is not None:
                for xb in (s - L - 0.5, e + L + 0.5):
                    ax.axvline(xb, color="grey", linestyle="--", linewidth=1.6)
            if r is not None and (bl > 0 or br > 0):
                span(ax, s - bl, e + br, ORANGE, 0.25, minw)
            span(ax, s, e, PINK, 0.35, minw)
        ax.plot(x, y, linewidth=1.6, color="#1f5fbf")
        ax.set_xlim(lo, hi - 1); ax.set_yticks([]); ax.margins(x=0)
        ax.text(0.985, 0.90, name, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"))
        if r is not None:
            print(f"{NAME} {name}: AUC-PR {r['auc_pr']:.3f} VUS@{L} {r['vus_pr']:.3f} AutoVUS {r['autovus']:.3f} "
                  f"b* {[int(a[5]) for a in own]} {[a[7] for a in own]}")
            ax.text(0.01, 0.92, f"AUC-PR = {r['auc_pr']:.3f}\nVUS@{L} = {r['vus_pr']:.3f}\nAutoVUS = {r['autovus']:.3f}",
                    transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.93, edgecolor="gray"))
        if ri == 2:
            ax.set_xlabel("Time index", fontsize=28)
        else:
            ax.tick_params(labelbottom=False)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=60)


if __name__ == "__main__":
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"fig1b_L{L}")
    main(stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf")
