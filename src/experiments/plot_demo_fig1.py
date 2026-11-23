"""Demo paper, Figure 1: 556_YAHOO_id_6 with AutoEncoder and Chronos at L=20.

One row for the series and one per detector. Red = labelled anomalies (A1..A4).
The number above each anomaly in a detector row is its AutoVUS Local there. Left
margin of a detector row: its name, AUC-PR, VUS-PR at L, AutoVUS on the whole
series (global), and Local, the mean of the local scores.

Values: local and global AutoVUS from experiments/local_benchmark/series/<file>.npz
(chance-adjusted as in app/export_local.py), AUC-PR and VUS-PR from
app/data/tsb-ad/scores20.json. Same visual language as Fig 1(b) of the research
paper (src/experiments/plot_fig1b.py).

Usage: conda run -n autovus python src/experiments/plot_demo_fig1.py
Writes experiments/figures/demo_fig1_L20.pdf and .png, and prints the table it draws.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "app", "api"))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402

L = 20
NAME, FILE, DETS = "556_YAHOO_id_6", "556_YAHOO_id_6_WebService_tr_500_1st_359.csv", ["AutoEncoder", "Chronos"]
RED, BLUE = "#d62728", "#1f5fbf"
W, FS, NAME_FS, BOX_FS = 3.5, 7, 8, 6.5      # one IEEE column


def adjusted(area, random):
    v, r = area[..., :L + 1].mean(-1), random[..., :L + 1].mean(-1)
    return (v - r) / np.maximum(1 - r, 1e-10)


def span(ax, a, b, minw):
    """Shade [a, b] (inclusive indices), widened to minw so point anomalies stay visible."""
    a, b = a - 0.5, b + 0.5
    if b - a < minw:
        c = (a + b) / 2; a, b = c - minw / 2, c + minw / 2
    ax.axvspan(a, b, facecolor=RED, alpha=0.45, edgecolor="none")


def main(out_pdf, out_png):
    p = TSADParser()
    values, label = load_series(p, FILE)
    z = np.load(os.path.join(ROOT, "experiments", "local_benchmark", "series", FILE + ".npz"))
    dets = [str(d) for d in z["detectors"]]
    local = adjusted(z["auto20_area"], z["auto20_random"])           # (D, K)
    glob = z["auto20_adjusted"]
    start, end = z["start"], z["end"]
    table = json.load(open(os.path.join(ROOT, "app", "data", "tsb-ad", "scores20.json")))["files"][FILE]

    n = len(label); x = np.arange(n); minw = 0.006 * n
    rows_in = [0.5, 0.72, 0.72]; top_in, bot_in, gap_in, left_in, right_in = 0.16, 0.3, 0.05, 0.74, 0.03
    H = top_in + sum(rows_in) + 2 * gap_in + bot_in
    plt.rcParams.update({"font.size": FS, "text.color": "0.15", "axes.labelcolor": "0.15", "xtick.color": "0.15"})
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 1, height_ratios=rows_in, hspace=gap_in / np.mean(rows_in),
                          left=left_in / W, right=1 - right_in / W, top=1 - top_in / H, bottom=bot_in / H)
    print(f"{NAME}: n = {n}, anomalies {[(int(s), int(e)) for s, e in zip(start, end)]}, lengths {[int(e - s + 1) for s, e in zip(start, end)]}")
    for ri, name in enumerate([NAME] + DETS):
        ax = fig.add_subplot(gs[ri])
        y = values if ri == 0 else load_score(p, FILE, name)
        for s, e in zip(start, end):
            span(ax, s, e, minw)
        ax.plot(x, y, linewidth=0.6, color=BLUE, zorder=3)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.6"); sp.set_linewidth(0.6)
        ax.set_xlim(0, n - 1); ax.set_yticks([]); ax.margins(x=0)
        ax.set_xticks(np.arange(0, n - 100, 200))
        ax.tick_params(length=2, pad=1.5)
        tra = blended_transform_factory(ax.transData, ax.transAxes)
        if ri == 0:
            ax.text(0.005, 0.93, NAME, transform=ax.transAxes, fontsize=NAME_FS, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))
            for k, (s, e) in enumerate(zip(start, end)):
                ax.text((s + e) / 2, 1.04, f"A{k + 1}", transform=tra, fontsize=BOX_FS, color=RED, ha="center", va="bottom", fontweight="bold")
            ax.tick_params(bottom=False, labelbottom=False)
            continue
        j = dets.index(name)
        ylo, yhi = float(np.min(y)), float(np.max(y))
        ax.set_ylim(ylo, yhi + 0.42 * (yhi - ylo))                     # head room for the local scores
        for k, (s, e) in enumerate(zip(start, end)):
            ax.text((s + e) / 2, 0.955, f"{local[j, k]:.2f}", transform=tra, fontsize=BOX_FS, ha="center", va="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=RED, linewidth=0.5), zorder=5)
        r = table[name]
        cells = [("AUC-PR", r["auc_pr"]), (f"VUS@{L}", r["vus_pr"]), ("AutoVUS", float(glob[j])), ("Local", float(local[j].mean()))]
        trf = blended_transform_factory(fig.transFigure, ax.transAxes)
        ax.text(0.012, 1.0, name, transform=trf, fontsize=FS - 0.5, color="black", ha="left", va="top", fontweight="bold")
        for k, (m, v) in enumerate(cells):
            yk = 0.78 - 0.2 * k
            ax.text(0.012, yk, m, transform=trf, fontsize=BOX_FS - 0.5, color="0.4", ha="left", va="top")
            ax.text(left_in / W - 0.014, yk, f"{v:.2f}", transform=trf, fontsize=BOX_FS, color="black", ha="right", va="top")
        print(f"  {name}: " + ", ".join(f"{m} {v:.3f}" for m, v in cells) + f" | local {[round(float(v), 3) for v in local[j]]}"
              f" | rank by AutoVUS {int((glob > glob[j]).sum()) + 1}, by mean local {int((local.mean(1) > local[j].mean()).sum()) + 1} of {len(dets)}")
        if ri == 2:
            ax.set_xlabel("Time index", labelpad=1.5)
        else:
            ax.tick_params(bottom=False, labelbottom=False)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)


if __name__ == "__main__":
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"demo_fig1_L{L}")
    main(stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf")
