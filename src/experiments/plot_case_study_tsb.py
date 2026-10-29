"""Sec V-A figure: two real TSB-AD cases, two detectors each.

(a) 798_YAHOO_id_248 (15 point/short anomalies): Chronos (exact) vs Sub_KNN (wide)
(b) 058_WSD_id_30 (one anomaly at 1236-1243, window 0-1680 so both x axes match): Lag_Llama vs TimesNet

Scores (VUS-PR, AutoVUS) are read from the demo export
app/data/tsb-ad/scores.json (L=128), the raw series and detector scores from TSB-AD.
Usage: conda run -n autovus python src/experiments/plot_case_study_tsb.py OUT.pdf [OUT.png]
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app", "api"))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402

CASES = [
    ("798_YAHOO_id_248", "798_YAHOO_id_248_WebService_tr_500_1st_105.csv", ["Chronos", "Sub_KNN"], (0, 1680)),
    ("058_WSD_id_30", "058_WSD_id_30_WebService_tr_1136_1st_1236.csv", ["Lag_Llama", "TimesNet"], (0, 1680)),
]
PINK = "#ff69b4"


def intervals(label):
    d = np.diff(np.r_[0, label, 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def shade(ax, label, x0, xlen):
    minw = 0.004 * xlen  # keep point anomalies visible
    for s, e in zip(*intervals(label)):
        a, b = x0 + s - 0.5, x0 + e - 0.5
        if b - a < minw:
            c = (a + b) / 2; a, b = c - minw / 2, c + minw / 2
        ax.axvspan(a, b, facecolor=PINK, alpha=0.35, edgecolor="none")


def main(out_pdf, out_png=None):
    scores = json.load(open(os.path.join(ROOT, "app", "data", "tsb-ad", "scores.json")))["files"]
    p = TSADParser()
    plt.rcParams.update({"font.size": 26})
    fig = plt.figure(figsize=(24, 17))
    gs = fig.add_gridspec(7, 1, height_ratios=[1.0, 0.75, 0.75, 0.06, 1.0, 0.75, 0.75], hspace=0.08)
    rows = [0, 1, 2, 4, 5, 6]
    for ci, (name_ts, file, dets, zoom) in enumerate(CASES):
        values, label = load_series(p, file)
        lo, hi = zoom
        x = np.arange(lo, hi)
        panels = [(name_ts, values[lo:hi], None)] + [(d, load_score(p, file, d)[lo:hi], scores[file][d]) for d in dets]
        for ri, (name, y, r) in enumerate(panels):
            ax = fig.add_subplot(gs[rows[3 * ci + ri]])
            shade(ax, label[lo:hi], lo, hi - lo)
            ax.plot(x, y, linewidth=1.6, color="#1f5fbf")
            ax.set_xlim(lo, hi - 1); ax.set_yticks([]); ax.margins(x=0)
            ax.text(0.985, 0.90, name, transform=ax.transAxes, fontsize=26, fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"))
            if r is not None:
                ax.text(0.01, 0.92, f"VUS-PR = {r['vus_pr']:.3f}\nAutoVUS = {r['autovus']:.3f}",
                        transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.93, edgecolor="gray"))
            if ri == 2 and ci == len(CASES) - 1:
                ax.set_xlabel("Time index", fontsize=28)
            else:
                ax.tick_params(labelbottom=False)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=60)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
