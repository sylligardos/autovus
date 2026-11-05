"""Sec V-B figure (controlled case study), rebuilt on the current engine.

One anomaly at [70, 90) in T = 160 points and 24 binary detector scores (groups A, B, C,
D1, D2, E), the same cases as the published figure (src/notebooks/case_studies.ipynb,
except D2, see CASES).
Every table value is computed here, none is typed in:
    F        point-wise F1 of the 0/1 score     (TSB-AD metric_PointF1, preds = score)
    RF       Range-F1                           (TSB-AD metric_RF1, preds = score)
    Aff-F    Affiliation F1                     (TSB-AD metric_Affiliation, preds = score)
    AUC      AUC-PR                             (sklearn average_precision_score)
    VUS      VUS-PR at maximum buffer length L  (FastVUS)
    AutoVUS  AutoVUS at L                       (src/autovus, paper rule, symmetric buffers)
L defaults to 20, the anomaly length. Prints the computed table next to the published one.
Usage: conda run -n autovus python src/experiments/plot_controlled_case.py [L]
Writes experiments/figures/controlled_case_study_L{L}.pdf/.png and _vs_paper.csv.
"""
import os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from autovus import FastVUS, AdjustedVUS, AutoVUS  # noqa: E402
from baselines.tsb_ad.basic_metrics import basic_metricor  # noqa: E402

T = 160
x = np.arange(T)
GT_START, GT_END = 70, 90
GT_CENTER = (GT_START + GT_END) / 2
gt = np.zeros(T, dtype=int)
gt[GT_START:GT_END] = 1


def block(start, length):
    y = np.zeros(T, dtype=int)
    a, b = max(0, int(round(start))), min(T, int(round(start + length)))
    if b > a:
        y[a:b] = 1
    return y


def centered(center, length):
    return block(center - length / 2, length)


def union(*arrays):
    return np.maximum.reduce(arrays)


# (group, name, score), as built in the notebook for the published figure.
CASES = []
for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
    CASES.append(("A", f"A{i}", centered(GT_CENTER + shift, 20)))    # same length, shifted
for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
    CASES.append(("B", f"B{i}", centered(GT_CENTER + shift, 10)))    # shorter, shifted
for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
    CASES.append(("C", f"C{i}", centered(GT_CENTER + shift, 30)))    # longer, shifted
inside = block(75, 10)
for i, s in enumerate([86, 90, 96, 102], start=1):
    CASES.append(("D1", f"D1-{i}", union(inside, block(s, 8))))      # inside + extra segment moving away
# D2 lengths 4..16: the notebook builds 6..18, but the published F and AUC-PR only match 4..16.
for i, n in enumerate([4, 8, 12, 16], start=1):
    CASES.append(("D2", f"D2-{i}", union(inside, block(96, n))))     # inside + growing outside segment
CASES.append(("E", "E1", union(block(50, 20), block(90, 20))))       # outside only, both sides

# The published table (paper Fig. controlled_cases, typed in the notebook).
PAPER = {
    "A1": (0.00, 0.00, 0.81, 0.12, 0.33, 0.13), "A2": (0.50, 0.55, 0.94, 0.31, 0.65, 0.54),
    "A3": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00), "A4": (0.50, 0.55, 0.94, 0.31, 0.65, 0.54),
    "A5": (0.00, 0.00, 0.81, 0.12, 0.33, 0.13),
    "B1": (0.00, 0.00, 0.78, 0.12, 0.30, 0.09), "B2": (0.33, 0.44, 0.93, 0.22, 0.50, 0.34),
    "B3": (0.67, 0.75, 0.99, 0.56, 0.61, 0.50), "B4": (0.33, 0.44, 0.93, 0.22, 0.50, 0.34),
    "B5": (0.00, 0.00, 0.78, 0.12, 0.30, 0.09),
    "C1": (0.20, 0.24, 0.84, 0.14, 0.36, 0.17), "C2": (0.60, 0.62, 0.94, 0.41, 0.73, 0.65),
    "C3": (0.80, 0.80, 0.97, 0.67, 0.92, 0.88), "C4": (0.60, 0.62, 0.94, 0.41, 0.73, 0.65),
    "C5": (0.20, 0.24, 0.84, 0.14, 0.36, 0.17),
    "D1-1": (0.74, 0.59, 0.98, 0.58, 0.81, 0.73), "D1-2": (0.53, 0.55, 0.95, 0.34, 0.67, 0.55),
    "D1-3": (0.53, 0.55, 0.93, 0.34, 0.54, 0.40), "D1-4": (0.53, 0.55, 0.92, 0.34, 0.45, 0.28),
    "D2-1": (0.59, 0.55, 0.96, 0.42, 0.59, 0.45), "D2-2": (0.53, 0.55, 0.93, 0.34, 0.54, 0.40),
    "D2-3": (0.48, 0.55, 0.91, 0.29, 0.49, 0.34), "D2-4": (0.43, 0.55, 0.89, 0.25, 0.43, 0.26),
    "E1": (0.00, 0.00, 0.83, 0.12, 0.39, 0.20),
}
COLS = ["F", "RF", "Aff-F", "AUC", "VUS", "AutoVUS"]

GROUP_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D1": "#d62728", "D2": "#9467bd", "E": "#8c564b"}
BLUE_CASES = {"A1", "A5", "B1", "B5", "E1"}                                     # F = RF = 0, Aff-F high
PINK_GREEN_CASES = {"A1", "A5", "B1", "B5", "C1", "C5", "D1-4", "D2-4", "E1"}   # VUS vs AutoVUS


def score_all(L):
    eng = FastVUS(L=L, cumulative=True)
    adj, auto = AdjustedVUS(eng), AutoVUS(eng, rule="paper", sides="symmetric")
    grader = basic_metricor()
    rows = []
    for group, name, s in CASES:
        r = auto.compute(gt, s)
        rows.append({
            "Case": name, "group": group,
            "F": grader.metric_PointF1(gt, s, preds=s),
            "RF": grader.metric_RF1(gt, s, preds=s),
            "Aff-F": grader.metric_Affiliation(gt, s, preds=s),
            "AUC": average_precision_score(gt, s),
            "VUS": adj.compute(gt, s)["vus"],
            "AutoVUS": r["adjusted"],
            "b*": int(r["anomalies"][0]["left"]["buffer_val"]),
        })
    return pd.DataFrame(rows)


def max_buffer_label(L):
    """The GT label with the full buffer L, as the engine builds it (square-root ramp to zita)."""
    eng = FastVUS(L=L, global_mask=False)
    _, (start, end) = eng.anomaly_coordinates(gt)
    dl, dr = eng.distance_from_anomaly(gt, start, end)
    return eng.buffered_labels(gt, dl, dr)[-1]


def draw_scores(ax, soft):
    rows = [("GT", "GT", gt)] + CASES
    n, amp, gap = len(rows), 0.64, 0.92
    for idx, (group, name, s) in enumerate(rows):
        yb = (n - 1 - idx) * gap
        if name == "GT":
            ax.plot(x, yb + amp * soft, color="red", linewidth=1.8, zorder=2)
            ax.step(x, yb + amp * s, where="post", color="black", linewidth=2.2, zorder=3)
            color = "black"
        else:
            color = GROUP_COLORS[group]
            ax.step(x, yb + amp * s, where="post", color=color, linewidth=1.55)
        ax.hlines(yb, 0, T - 1, color=color, linewidth=0.55, alpha=0.25)
        ax.text(-5.8, yb + amp * 0.32, name, ha="right", va="center", fontsize=8.7, color=color, fontweight="bold")
    ax.set_xlim(-20, T - 1)
    top, y_min = (n - 1) * gap, -0.16
    ax.set_ylim(y_min, (top - y_min) / 0.886 + y_min)   # GT row level with the table's header rule
    for bnd in (GT_START, GT_END):
        ax.vlines(bnd, y_min, top + amp * 1.08, color="0.55", linestyle="--", linewidth=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Time", labelpad=4)
    ax.text(0.50, 1.018, "\n(a) Controlled\ndetector scores", transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", linespacing=0.95)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def draw_table(ax, df):
    ax.axis("off")
    ax.text(0.5, 1.012, "(b) Metric comparison", ha="center", va="top", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.986, "Pink/green: VUS vs. AutoVUS. Blue: F=RF=0 but Aff-F remains high.", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.7, style="italic")
    headers = ["Case"] + COLS
    left, right, top, bottom = 0.012, 0.997, 0.962, 0.0005
    width, n_header = right - left, 2
    row_h = (top - bottom) / (n_header + len(df))
    cw = np.array([0.115, 0.095, 0.105, 0.150, 0.110, 0.110, 0.160])
    xe = left + width * np.r_[0, np.cumsum(cw / cw.sum())]
    xc = (xe[:-1] + xe[1:]) / 2
    ye = top - np.arange(n_header + len(df) + 1) * row_h
    yc = (ye[:-1] + ye[1:]) / 2
    tr = ax.transAxes
    for y, lw in ((ye[0], 1.55), (ye[2], 1.0), (ye[-1], 1.55)):
        ax.hlines(y, left, right, colors="black", linewidth=lw, transform=tr)
    ax.hlines(ye[1], xe[1], xe[4], colors="0.40", linewidth=0.65, transform=tr)
    ax.hlines(ye[1], xe[4], xe[-1], colors="0.40", linewidth=0.65, transform=tr)
    for a, b, txt in ((1, 4, "Threshold-\ndependent"), (4, 7, "Threshold-\nindependent")):
        ax.text((xe[a] + xe[b]) / 2, yc[0], txt, ha="center", va="center", transform=tr, fontsize=9.3,
                fontweight="bold", linespacing=0.84)
    for j, h in enumerate(headers):
        ax.text(xc[j], yc[1], h, ha="center", va="center", transform=tr, fontsize=9.9, fontweight="bold")

    def cell(i, j, color, z):
        ax.add_patch(Rectangle((xe[j], ye[i + n_header + 1]), xe[j + 1] - xe[j], ye[i + n_header] - ye[i + n_header + 1],
                               transform=tr, facecolor=color, edgecolor="none", zorder=z))

    prev = None
    for i, row in df.reset_index(drop=True).iterrows():
        case, group = row["Case"], row["group"]
        if i % 2 == 1:
            ax.add_patch(Rectangle((left, ye[i + n_header + 1]), width, row_h, transform=tr,
                                   facecolor="0.972", edgecolor="none", zorder=-5))
        if prev is not None and group != prev:
            ax.hlines(ye[i + n_header], left, right, colors="0.82", linewidth=0.55, transform=tr)
        prev = group
        if case in PINK_GREEN_CASES:
            cell(i, headers.index("VUS"), "#f8d7da", -3)
            cell(i, headers.index("AutoVUS"), "#d9f0d3", -3)
        if case in BLUE_CASES:
            for h, c in (("F", "#dceeff"), ("RF", "#dceeff"), ("Aff-F", "#cfe9ff")):
                cell(i, headers.index(h), c, -2)
        for j, h in enumerate(headers):
            kw = {"fontsize": 9.85, "ha": "center", "va": "center", "transform": tr}
            if h == "Case":
                kw.update(fontsize=10.0, fontweight="bold", color=GROUP_COLORS[group])
                txt = case
            else:
                txt = f"{float(row[h]):.2f}"
            if (case in BLUE_CASES and h in {"F", "RF", "Aff-F"}) or (case in PINK_GREEN_CASES and h in {"VUS", "AutoVUS"}):
                kw["fontweight"] = "bold"
            ax.text(xc[j], yc[i + n_header], txt, **kw)


def main(L, stem):
    df = score_all(L)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig = plt.figure(figsize=(8.0, 7.3))
    gs = GridSpec(1, 2, width_ratios=[0.77, 1.23], wspace=0.045, figure=fig)
    draw_scores(fig.add_subplot(gs[0, 0]), max_buffer_label(L))
    draw_table(fig.add_subplot(gs[0, 1]), df)
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(stem + ".png", bbox_inches="tight", pad_inches=0, dpi=110)
    # Text boxes are taller than their glyphs: crop the PDF to the ink, no white margin.
    subprocess.run(["pdfcrop", "--margins", "0", stem + ".pdf", stem + ".pdf"], check=True, stdout=subprocess.DEVNULL)

    # Computed vs published, per cell.
    cmp = df[["Case", "b*"]].copy()
    for j, c in enumerate(COLS):
        cmp[c + " paper"] = [PAPER[k][j] for k in df["Case"]]
        cmp[c] = df[c].round(3)
        cmp[c + " diff"] = (df[c] - cmp[c + " paper"]).round(3)
    cmp.to_csv(stem + "_vs_paper.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"L = {L}, computed (paper) per cell")
    show = df[["Case"]].copy()
    for j, c in enumerate(COLS):
        show[c] = [f"{v:.2f} ({PAPER[k][j]:.2f})" for k, v in zip(df["Case"], df[c])]
    show["b*"] = df["b*"]
    print(show.to_string(index=False))
    print("max |diff| per column:", {c: float(cmp[c + " diff"].abs().max()) for c in COLS})


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    outdir = os.path.join(ROOT, "experiments", "figures")
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, f"controlled_case_study_L{L}")
    main(L, stem)
    print("saved:", stem + ".pdf", stem + ".png", stem + "_vs_paper.csv")
