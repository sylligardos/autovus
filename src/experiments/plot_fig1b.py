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
PINK = "#d62728"  # red (was hot pink #ff69b4)
ORANGE = "#ff8c00"


def span(ax, a, b, color, alpha, minw):
    """Shade [a, b] (inclusive indices), widened to minw so point anomalies stay visible."""
    a, b = a - 0.5, b + 0.5
    if b - a < minw:
        c = (a + b) / 2; a, b = c - minw / 2, c + minw / 2
    ax.axvspan(a, b, facecolor=color, alpha=alpha, edgecolor="none")


def main(out_pdf, out_png=None):
    """Drawn at the size of Fig 1(a) (same width, font size, frame left/right edges and title gap, read from
    experiments/figures/fig1a_layout.json), so both panels scale by the same factor at \\linewidth."""
    import json
    from matplotlib.transforms import Bbox, blended_transform_factory
    lay = json.load(open(os.path.join(ROOT, "experiments", "figures", "fig1a_layout.json")))
    W, FS = lay["width"], lay["font_size"]
    TITLE_FS, BOX_FS, NAME_FS = 7.2, 8, 9
    rows_in = [0.55, 0.85, 0.85]; top_in, bot_in, gap_in = 0.25, 0.4, 0.05
    H = top_in + sum(rows_in) + 2 * gap_in + bot_in
    p = TSADParser()
    plt.rcParams.update({"font.size": FS})
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 1, height_ratios=rows_in, hspace=gap_in / np.mean(rows_in),
                          left=lay["ax_left"] / W, right=1 - lay["ax_right"] / W, top=1 - top_in / H, bottom=bot_in / H)
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
                    ax.axvline(xb, color="grey", linestyle="--", linewidth=0.6)
            if r is not None and (bl > 0 or br > 0):
                span(ax, s - bl, e + br, ORANGE, 0.25, minw)
            span(ax, s, e, PINK, 0.5, minw)
        ax.plot(x, y, linewidth=0.8, color="#1f5fbf", zorder=3)
        for sp in ax.spines.values():  # same frame as Fig 1(a)
            sp.set_edgecolor(lay["edge_color"]); sp.set_linewidth(lay["edge_width"])
        if name == "SAND":
            # FP after the first anomaly: highest score between the first and second fixed buffers,
            # above the score SAND gives inside the last anomaly (dotted line from that detection).
            a0, a1, aL = own[0], own[1], own[-1]
            lo_fp, hi_fp = int(a0[4] + L + 1), int(a1[3] - L)
            fp = lo_fp + int(np.argmax(y[lo_fp:hi_fp]))
            level = float(np.max(y[int(aL[3]):int(aL[4]) + 1]))
            ax.hlines(level, fp, int(aL[3]), color="black", linestyle=(0, (1.5, 1.5)), linewidth=0.7)
            r0, r1 = fp, fp  # contiguous run around the FP that lies above the last detection
            while r0 > lo_fp and y[r0 - 1] > level: r0 -= 1
            while r1 < hi_fp - 1 and y[r1 + 1] > level: r1 += 1
            span(ax, r0, r1, PINK, 0.3, minw)
            ax.annotate("FP", (r1, y[fp]), xytext=(3, 5), textcoords="offset points", fontsize=NAME_FS,
                        fontweight="bold", color=PINK, ha="left", va="bottom")
            print(f"FP run above the last detection: {r0}..{r1}; FP at {fp}: score {y[fp]:.3f} > last-anomaly peak {level:.3f}")
        ax.set_xlim(lo, hi - 1); ax.set_yticks([]); ax.margins(x=0)
        ax.set_xticks(np.arange(0, hi - 100, 200))  # last tick (1400) would stick out past the right edge
        ax.text(0.99, 0.94, name, transform=ax.transAxes, fontsize=NAME_FS, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))
        if r is not None:
            print(f"{NAME} {name}: AUC-PR {r['auc_pr']:.3f} VUS@{L} {r['vus_pr']:.3f} AutoVUS {r['autovus']:.3f} "
                  f"b* {[int(a[5]) for a in own]} {[a[7] for a in own]}")
            # score box in the left margin (the y-label space of Fig 1(a))
            tr = blended_transform_factory(fig.transFigure, ax.transAxes)
            for k, (m, v) in enumerate([("AUC-PR", r["auc_pr"]), (f"VUS@{L}", r["vus_pr"]), ("AutoVUS", r["autovus"])]):
                yk = 0.96 - 0.32 * k
                ax.text(0.0, yk, m, transform=tr, fontsize=BOX_FS, color="0.45", ha="left", va="top")
                ax.text(0.0, yk - 0.13, f"{v:.3f}", transform=tr, fontsize=NAME_FS, color="black", ha="left", va="top")
            for yb in (0.0, 1.0):  # dashed lines in the frame colour above and below the scores, per row
                ax.plot([0.0, lay["ax_left"] / W], [yb, yb], transform=tr, clip_on=False, linestyle="--",
                        color=lay["edge_color"], linewidth=lay["edge_width"])
        if ri == 0:
            ax.set_title("L2: VUS applies the same buffer lengths to every anomaly and detector", fontsize=TITLE_FS,
                         fontweight="bold")
            ax.text(0.01, 0.94, "(b)", transform=ax.transAxes, fontsize="medium", fontweight="bold", ha="left", va="top")
        if ri == 2:
            ax.set_xlabel("Time index")
        else:
            ax.tick_params(bottom=False, labelbottom=False)
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox([[0, tb.y0], [W, tb.y1]])  # full width kept so the frame edges match Fig 1(a)
    fig.savefig(out_pdf, bbox_inches=box)
    if out_png:
        fig.savefig(out_png, bbox_inches=box, dpi=250)


if __name__ == "__main__":
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"fig1b_L{L}")
    main(stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf")
