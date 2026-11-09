"""Sec V-C figure (event-level interpretability), rebuilt on the current engine.

Three toy multi-anomaly cases, two scores each, scored with FastVUS / AutoVUS (src/autovus,
paper rule, symmetric buffers) at maximum buffer length L. Prints the table and draws the
figure: label row, each score with its selected per-anomaly buffers b* below it, table on the right.
Usage: conda run -n autovus python src/experiments/plot_multi_case.py [L]
Writes experiments/figures/multi_case_study_L{L}.pdf and .png (default L = 20).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from autovus import FastVUS, AdjustedVUS, AutoVUS  # noqa: E402

T = 180
x = np.arange(T)


# Drawn at Fig 1's nominal width (4.8 in, printed at \\linewidth = 0.73x) instead of 12 in (0.30x):
KW = 4.8 / 12.0     # width
KF = 0.72           # fonts (about 7 pt in print)
KL = 0.41           # line widths and markers (same printed thickness as before)
KT = 0.62           # table cells (about 6 pt in print)


def g(c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def label(intervals):
    y = np.zeros(T, dtype=int)
    for s, e in intervals:
        y[s:e] = 1
    return y


def nz(y):
    """Normalize to [0, 1] and zero the Gaussian dust (< 1e-6), as a real detector score has no such tails."""
    y = y / y.max()
    y[y < 1e-6] = 0
    return y


def plateau(s, e, edge):
    """Flat 1 on [s, e) with a soft Gaussian edge of `edge` points on each side."""
    y = np.zeros(T)
    y[s:e] = 1
    for k in range(1, edge + 1):
        v = np.exp(-0.5 * (k / (edge / 2)) ** 2)
        if s - k >= 0:
            y[s - k] = max(y[s - k], v)
        if e - 1 + k < T:
            y[e - 1 + k] = max(y[e - 1 + k], v)
    return y


# (title, intervals [start, end), S1, S2)
CASES = [
    ("Dense anomalies", [(35, 42), (48, 55), (62, 69), (76, 83)],
     nz(g(39.5, .95, 2.2) + g(52.5, 1, 2.5) + g(66.5, .92, 2.4) + g(80.5, .9, 2.4)),  # S1: each event, one step late
     nz(g(60, 1, 15.0))),                                                      # S2: one broad bridge
    ("Boundary-only firing", [(30, 40), (85, 95), (140, 150)],
     nz(g(37, 1, 2.5) + g(92, 1, 2.5) + g(147, 1, 2.5)),                        # S1: inside, near the end edge
     nz(sum(g(c, .95, 2.0) for c in (24, 46, 79, 101, 134, 156)))),             # S2: outside both sides
    ("Long anomaly vs. spikes", [(20, 28), (70, 110), (150, 158)],
     nz(g(26, 1, 2.5) + plateau(82, 122, 4) + g(156, 1, 2.5)),                  # S1: lag grows with the anomaly length (2, 12, 2)
     nz(g(36, 1, 2.5) + plateau(82, 122, 4) + g(166, 1, 2.5))),                 # S2: the same lag of 12 everywhere
]

# Palette and layout of the published figure (sampled from experiments/figures/multi_case_study_published.png).
C_LABEL, C_BUFFER, C_BUFFER_CORE = "#c8102e", "#f8c8d0", "#e06a74"
C_S1, C_S2, C_DOT, C_LINE, C_BOX = "#1f6fb4", "#f25c19", "#333333", "#cfcfcf", "#d9d9d9"
COLS = ["Score", "AUC", "VUS", "AutoVUS", "$b^*$"]
# y positions (data units) of the rows inside each case panel, shared by the panel and its table.
Y_LABEL, Y_S1, Y_B1, Y_S2, Y_B2, AMP = 4.62, 3.45, 2.80, 1.38, 0.73, 0.62
BOX1, BOX2 = (2.45, 4.22), (0.38, 2.15)   # (bottom, top) of the S1 and S2 boxes
Y_MIN = 0.25                              # bottom of each panel
X0 = -27                                  # left edge of the boxes; row names sit right-aligned at x = -6


def score_all(L):
    eng = FastVUS(L=L, cumulative=True)
    adj, auto = AdjustedVUS(eng), AutoVUS(eng, rule="paper", sides="symmetric")
    out = []
    for title, iv, s1, s2 in CASES:
        y = label(iv)
        rows = []
        for name, s in (("S1", s1), ("S2", s2)):
            a, r = adj.compute(y, s), auto.compute(y, s)
            b = tuple(int(q["left"]["buffer_val"]) for q in r["anomalies"])
            rows.append((name, average_precision_score(y, s), a["vus"], r["adjusted"], b))
        out.append((title, iv, s1, s2, rows))
    return out


class BufferHandler(HandlerBase):
    """Legend key for the selected buffer: pale buffer with the darker anomaly core inside."""
    def create_artists(self, legend, orig, x0, y0, w, h, fontsize, trans):
        return [Rectangle((x0, y0 - 0.15 * h), w, 1.3 * h, facecolor=C_BUFFER, edgecolor="none", transform=trans),
                Rectangle((x0 + 0.15 * w, y0 + 0.2 * h), 0.7 * w, 0.6 * h, facecolor=C_BUFFER_CORE, edgecolor="none", transform=trans)]


def draw_buffer_row(ax, intervals, bvec, y):
    """Pale pink = selected expanded buffer, darker pink = original anomaly."""
    ax.hlines(y, 0, T - 1, color=C_LINE, linewidth=0.8*KL, zorder=0)
    for (s, e), b in zip(intervals, bvec):
        ax.add_patch(Rectangle((max(0, s - b), y - 0.17), min(T, e + b) - max(0, s - b), 0.34, facecolor=C_BUFFER, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((s, y - 0.09), e - s, 0.18, facecolor=C_BUFFER_CORE, edgecolor="none", zorder=3))


def draw_panel(ax, iv, s1, s2, rows, last):
    ax.set_xlim(X0 - 1, T + 2); ax.set_ylim(Y_MIN, 5.0)
    bb = ax.get_position(); W, H = ax.figure.get_size_inches()
    aspect = ((5.0 - Y_MIN) / (bb.height * H)) / ((T + 3 - X0) / (bb.width * W))   # data-y per data-x at equal inches
    for lo, hi in (BOX1, BOX2):
        ax.add_patch(FancyBboxPatch((X0, lo), T + 1 - X0, hi - lo, boxstyle="round,pad=0,rounding_size=3", mutation_aspect=aspect,
                                    facecolor="none", edgecolor=C_BOX, linewidth=1.0*KL, zorder=-5))
    ax.plot(x[::3], np.full(len(x[::3]), Y_LABEL), linestyle="none", marker="o", markersize=2.6*KL, color=C_DOT, zorder=1)
    for s, e in iv:
        ax.add_patch(Rectangle((s, Y_LABEL - 0.14), e - s, 0.28, facecolor=C_LABEL, edgecolor="none", zorder=3))
    for ys, yb, sc, col, name, b in ((Y_S1, Y_B1, s1, C_S1, "S1", rows[0][4]), (Y_S2, Y_B2, s2, C_S2, "S2", rows[1][4])):
        ax.plot(x, ys + AMP * sc, color=col, linewidth=1.8*KL, zorder=5)
        draw_buffer_row(ax, iv, b, yb)
        ax.text(-6, ys + 0.1, name, ha="right", va="center", fontsize=15*KF, fontweight="bold", color=col)
        ax.text(-6, yb, name + r" $b^*$", ha="right", va="center", fontsize=12*KF)
    ax.text(-6, Y_LABEL, "Label", ha="right", va="center", fontsize=12.5*KF, fontweight="bold")
    ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    if last:
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_bounds(0, T)
        ax.set_xticks(range(0, T + 1, 30)); ax.set_xlabel("Time", fontsize=13*KF, labelpad=2)
        ax.tick_params(axis="x", labelsize=11.5*KF, length=4*KL)
    else:
        ax.set_xticks([])


def draw_table(ax, title, rows):
    """Booktabs-style table on the same y scale as the panel: title at the label row, rules around the boxes."""
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(Y_MIN, 5.0)
    xs = [0.07, 0.25, 0.43, 0.645, 0.885]
    ax.text(0.5, Y_LABEL, title, ha="center", va="center", fontsize=13*KF, fontweight="bold")
    for y, lw in ((4.15, 1.3), (3.05, 0.9), (0.75, 1.3)):
        ax.hlines(y, 0, 1, color="black", linewidth=lw)
    for xc, h in zip(xs, COLS):
        ax.text(xc, 3.6, h, ha="center", va="center", fontsize=12.5*KT, fontweight="normal" if h == "$b^*$" else "bold")
    best = [None] + [0 if rows[0][j] >= rows[1][j] else 1 for j in (1, 2, 3)]
    for i, (name, auc, vus, autov, b) in enumerate(rows):
        y = (2.45, 1.4)[i]
        ax.text(xs[0], y, name, ha="center", va="center", fontsize=15*KT, fontweight="bold", color=(C_S1, C_S2)[i])
        for j, v in enumerate((auc, vus, autov), start=1):
            ax.text(xs[j], y, f"{v:.3f}", ha="center", va="center", fontsize=13*KT, fontweight="bold" if best[j] == i else "normal")
        ax.text(xs[4], y, "(" + ",".join(map(str, b)) + ")", ha="center", va="center", fontsize=13*KT)


def main(L, out_pdf, out_png=None):
    res = score_all(L)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(12.0 * KW, 6.0 * KF * 0.82))
    gs = GridSpec(3, 2, width_ratios=[1.0, 1.1], hspace=0.07, wspace=0.06, figure=fig, left=0.02, right=0.98, top=0.93, bottom=0.07)
    for row, (title, iv, s1, s2, rows) in enumerate(res):
        ax = fig.add_subplot(gs[row, 0]); ax_t = fig.add_subplot(gs[row, 1])
        draw_panel(ax, iv, s1, s2, rows, last=row == 2)
        draw_table(ax_t, title, rows)
    handles = [Patch(facecolor=C_LABEL, edgecolor="none", label="label"), Patch(label="selected buffer $b^*$"),
               plt.Line2D([0], [0], color=C_S1, linewidth=1.8*KL, label="score")]
    fig.legend(handles=handles, handler_map={handles[1]: BufferHandler()}, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.08, 0.995), fontsize=13*KF, handlelength=2.2, columnspacing=2.2)
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png:
        fig.savefig(out_png, bbox_inches="tight", dpi=110)
    for title, *_, rows in res:
        print(title)
        for n, auc, v, a, b in rows:
            print(f"   {n} AUC-PR {auc:.3f} VUS@{L} {v:.3f} AutoVUS {a:.3f} b* {b}")


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"multi_case_study_L{L}")
    main(L, stem + ".pdf", stem + ".png")
    print("saved:", stem + ".pdf", stem + ".png")
