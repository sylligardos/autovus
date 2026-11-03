"""Global detector ranking on TSB-AD-U under AUC-PR, VUS at maximum buffer 2..128 and AutoVUS@128 (rank heatmap).

Reads experiments/autovus_benchmark/pairs.csv and anomalies.csv (autovus_benchmark.py). Rank = rank of the mean score over the series each detector scores. Rows = the TOP detectors under AutoVUS,
names underlined for the sharp group (b* = 0 on most anomalies), bold for the broad group (a buffer on most anomalies), then the
mean rank of each group. Last column = mean selected buffer b* over the detector's (anomaly, detector) pairs at L=128.
Style follows the paper figures (plot_multi_case.py): DejaVu Sans, booktabs rules, S1 blue / S2 orange.
Drawn at print size (single column, about 3.7 in before LaTeX scales it to 3.5 in): green cells, one hue
(dark = good rank), ranks outside the top 9 faded, a group bar left of each row, AutoVUS column set apart by a gap.
Usage: conda run -n autovus python src/experiments/plot_rank_heatmap.py
Writes experiments/figures/rank_heatmap.pdf/.png and prints the group mean ranks.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN = os.path.join(ROOT, "experiments", "autovus_benchmark")
OUT = os.path.join(ROOT, "experiments", "figures", "rank_heatmap")
LS = [2, 4, 8, 16, 32, 64, 128]
COLS = ['auc_pr'] + [f'vus_{b}' for b in LS] + ['autovus_128']
XLAB = ['AUC-PR'] + [f'VUS@{b}' for b in LS] + ['AutoVUS']
TOP = 9                                                    # detectors shown (top under AutoVUS)
ZSPLIT = 0.5                                               # share of b* = 0 at or above = sharp, below = broad
C_SHARP, C_BROAD, DARK, FADE = "#1f6fb4", "#f25c19", "#222222", "#8a8a8a"   # paper S1 blue, S2 orange
CMAP = LinearSegmentedColormap.from_list('rk', plt.get_cmap('Greens')(np.linspace(0.95, 0.12, 256)))   # one hue, dark = good rank
RMAX = 12                                                  # ranks at or above RMAX share the lightest shade
# Layout in inches at print size (single IEEE column, 3.5 in): cell size, gaps, font size (pt).
X0, CW, CH, PAD, GAP, BW = 0.88, 0.26, 0.2, 0.012, 0.06, 0.25
HEAD, SEP, FS, ROT = 0.54, 0.09, 7.5, 60                   # header height, gap before the group rows, font size, header angle


def main():
    d = pd.read_csv(os.path.join(IN, 'pairs.csv'))
    d = d[~d.detector.str.startswith('Random_')]
    d = d[d['error'].isna()]
    a = pd.read_csv(os.path.join(IN, 'anomalies.csv'))
    a = a[a.series.isin(d.series.unique()) & ~a.detector.str.startswith('Random_')]
    zero = a.groupby('detector').b_star_128.apply(lambda b: (b == 0).mean())
    bmean = a.groupby('detector').b_star_128.mean()
    rk = d.groupby('detector')[COLS].mean().rank(ascending=False).astype(int)
    top = list(rk.autovus_128.sort_values().index[:TOP])
    sharp = [t for t in top if zero[t] >= ZSPLIT]
    broad = [t for t in top if zero[t] < ZSPLIT]
    M = rk.loc[top, COLS].values.astype(float)
    G = np.vstack([rk.loc[sharp, COLS].mean().values, rk.loc[broad, COLS].mean().values])
    print(f'{d.series.nunique()} series, {len(d)} pairs')
    print('share of b*=0 per detector:', {t: round(float(zero[t]), 3) for t in top}, 'sharp', round(float((a.b_star_128[a.detector.isin(sharp)] == 0).mean()), 3), 'broad', round(float((a.b_star_128[a.detector.isin(broad)] == 0).mean()), 3))
    print(pd.DataFrame(G, index=['sharp', 'broad'], columns=XLAB).round(1).to_string())

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavuserif"})
    nc = len(COLS)
    xs = [X0 + j * CW + (GAP if j == nc - 1 else 0) for j in range(nc)]     # cell left edges, AutoVUS set apart
    xb = xs[-1] + CW + GAP + BW / 2                                          # mean b* column centre
    right = xb + BW / 2
    ys = [HEAD + (i + 0.5) * CH for i in range(TOP)] + [HEAD + (TOP + k + 0.5) * CH + SEP for k in (0, 1)]
    H = ys[-1] + CH / 2 + 0.03
    fig = plt.figure(figsize=(right + 0.02, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, right + 0.02); ax.set_ylim(H, 0)                           # data units = inches

    def cell(x, y, v, bold, fmt):
        t = (min(v, RMAX) - 1) / (RMAX - 1)
        ax.add_patch(plt.Rectangle((x + PAD, y - CH / 2 + PAD), CW - 2 * PAD, CH - 2 * PAD, facecolor=CMAP(t), edgecolor='none'))
        ax.text(x + CW / 2, y, fmt.format(v), ha='center', va='center', fontsize=FS,
                color='white' if t < 0.45 else (FADE if v > TOP else DARK), fontweight='bold' if bold else 'normal')

    for x, h in zip(xs + [xb - CW / 2], XLAB + ['mean $b^*$']):
        ax.text(x + CW / 2 - 0.03, HEAD - 0.05, h, ha='left', va='bottom', rotation=ROT, rotation_mode='anchor', fontsize=FS, fontweight='bold')
    ax.hlines(HEAD - 0.02, 0, right, color='black', lw=0.8)
    names = [(t, 'under' if t in sharp else 'bold', 'normal') for t in top] + [('sharp (mean)', 'under', 'italic'), ('broad (mean)', 'bold', 'italic')]   # sharp underlined, broad bold
    vals = [(M[i], '{:.0f}', f'{bmean[t]:.1f}') for i, t in enumerate(top)] + \
           [(G[k], '{:.1f}', f'{a.b_star_128[a.detector.isin(g)].mean():.1f}') for k, g in enumerate((sharp, broad))]
    for y, (name, c, st), (row, fmt, bs) in zip(ys, names, vals):
        txt = ax.text(X0 - 0.04, y, name, ha='right', va='center', fontsize=FS, fontweight='bold' if c == 'bold' else 'normal', color=DARK, style=st)
        if c == 'under':                                                           # matplotlib has no underline: draw it
            bb = txt.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
            ax.plot([bb.x0, bb.x1], [max(bb.y0, bb.y1) + 0.015] * 2, color=DARK, lw=0.6, solid_capstyle='butt')
        for j in range(nc):
            cell(xs[j], y, row[j], j == nc - 1, fmt)
        ax.text(xb, y, bs, ha='center', va='center', fontsize=FS, color=DARK, style=st)
    ax.hlines(HEAD + TOP * CH + SEP / 2, 0, right, color='black', lw=0.5)
    ax.hlines(H - 0.02, 0, right, color='black', lw=0.8)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + '.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT + '.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    print('wrote', OUT + '.pdf')


if __name__ == '__main__':
    main()
