"""Global detector ranking on TSB-AD-U under VUS at maximum buffer 2..128 and AutoVUS@128 (bump chart).

Reads experiments/autovus_benchmark/pairs.csv and anomalies.csv (autovus_benchmark.py). Uses the series
that have all detectors. Rank = rank of the mean score over series. Shows the TOP detectors under AutoVUS.
Lines are coloured by the detector's mean selected buffer b* over its (anomaly, detector) pairs at L=128;
next to each name the distribution of those b* over 0..128 (share per value, log height, symlog x; or binned histograms).
Usage: conda run -n autovus python src/experiments/plot_rank_autovus_review.py [linear|log|nonzero] [TOP]
Writes experiments/figures/rank_autovus_review.pdf/.png and prints the rank table.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN = os.path.join(ROOT, "experiments", "autovus_benchmark")
OUT = os.path.join(ROOT, "experiments", "figures", "rank_autovus_review")
LS = [2, 4, 8, 16, 32, 64, 128]
COLS = ['auc_pr'] + [f'vus_{b}' for b in LS] + ['autovus_128']
XLAB = ['AUC-PR'] + [f'VUS@{b}' for b in LS] + ['AutoVUS']
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 9     # detectors shown (top under AutoVUS)
ZSPLIT = 0.5                                           # share of b* = 0 above = 'sharp' group, below = 'broad' group
C1, C2 = '#2e8b57', '#c0392b'                           # pale green (sharp, 'good') and red (broad, 'bad') for the names
LINE = '#666666'
EDGES = [0, 1, 3, 5, 9, 17, 33, 65, 129]          # bins 0 | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128
BLAB = ['0', '1-2', '3-4', '5-8', '9-16', '17-32', '33-64', '65-128']
GREY, DARK, LIGHT = '#555555', '#222222', '#e6e6e6'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'line'   # 'line' = share per b* value (log height, symlog x); histogram: linear | log; 'nonzero' = share of b* > 0


def main():
    d = pd.read_csv(os.path.join(IN, 'pairs.csv'))
    d = d[~d.detector.str.startswith('Random_')]
    if 'error' in d:
        d = d[d['error'].isna()]
    cnt = d.groupby('series').detector.count()
    d = d[d.series.isin(cnt[cnt == cnt.max()].index)]
    a = pd.read_csv(os.path.join(IN, 'anomalies.csv'))
    a = a[a.series.isin(d.series.unique()) & ~a.detector.str.startswith('Random_')]
    bmean = a.groupby('detector').b_star_128.mean()
    zero = a.groupby('detector').b_star_128.apply(lambda b: (b == 0).mean())
    g = d.groupby('detector')[COLS].mean()
    rk = g.rank(ascending=False).astype(int)
    tab = rk.copy(); tab.columns = XLAB; tab['mean b*'] = bmean.round(1); tab['b*=0'] = zero.round(2); tab['mean VUS@128'] = g.vus_128.round(3); tab['mean AutoVUS'] = g.autovus_128.round(3)
    print(f'{d.series.nunique()} series, {len(d)} pairs'); print(tab.sort_values('AutoVUS').to_string())

    top = rk.autovus_128.sort_values().index[:TOP]
    edges = np.arange(130) - 0.5 if MODE == 'line' else EDGES
    hist = {det: np.histogram(a.b_star_128[a.detector == det], bins=edges)[0] for det in top}
    hist = {det: h / h.sum() for det, h in hist.items()}
    if MODE in ('log', 'line'):
        hist = {det: np.log10(1 + 99 * h) / 2 for det, h in hist.items()}   # 1% -> 0.15, 10% -> 0.52, 100% -> 1
    if MODE == 'nonzero':
        hist = {det: np.array([1 - h[0]]) for det, h in hist.items()}
    hmax = max(h.max() for h in hist.values())

    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif', 'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2})
    W, H = 3.5, 1.6 + 0.15 * TOP
    fig = plt.figure(figsize=(W, H))
    L, R, B, T = 0.48, 0.05, 0.58, 0.40                     # margins in inches
    NAMES, HIST = (1.02 if TOP > 16 else 0.90), 0.62                              # width of the name column and of the histogram column
    ax = fig.add_axes([L / W, B / H, (W - L - R - NAMES - HIST) / W, (H - B - T) / H])
    axh = fig.add_axes([(W - R - HIST) / W, B / H, HIST / W, (H - B - T) / H], sharey=ax)

    cmap = LinearSegmentedColormap.from_list('blues', plt.get_cmap('Blues')(np.linspace(0.4, 1.0, 256)))
    norm = plt.Normalize(bmean[top].min(), bmean[top].max())
    x = np.arange(len(COLS))
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color='#f2f2f2', lw=0, zorder=0)
    rest = TOP < len(rk)                                  # is there a '>TOP' row
    if rest:
        ax.axhline(TOP + 0.5, color=GREY, lw=0.5, zorder=0)
    for det in top:
        c = C1 if zero[det] >= ZSPLIT else C2
        ax.plot(x, rk.loc[det, COLS].clip(upper=TOP + 1), '-', color=LINE, lw=1.0, solid_capstyle='round', zorder=3)
        r = rk.loc[det, 'autovus_128']
        ax.annotate(det if len(det) <= 14 else det[:13] + '.', (1.0, r), xycoords=('axes fraction', 'data'), xytext=(3, 0), textcoords='offset points', fontsize=6, va='center', ha='left', color=c, fontweight='bold', annotation_clip=False)
        axh.annotate(f'{bmean[det]:.1f}', (0.0, r), xycoords=('axes fraction', 'data'), xytext=(-4, 0), textcoords='offset points', fontsize=6, va='center', ha='right', color=DARK, annotation_clip=False)
        if MODE == 'line':
            xb = np.arange(129)
            axh.fill_between(xb, r + 0.39, r + 0.39 - 0.78 * hist[det] / hmax, color='#bbbbbb', lw=0, zorder=2)
            axh.plot(xb, r + 0.39 - 0.78 * hist[det] / hmax, color=GREY, lw=0.6, zorder=3)
            axh.hlines(r + 0.39, 0, 128, color='#cccccc', lw=0.4, zorder=1)
        elif MODE == 'nonzero':
            axh.barh(r, hist[det][0] * len(BLAB), height=0.55, left=-0.5, color=GREY, lw=0, zorder=3)
            axh.annotate(f'{hist[det][0]:.0%}', (-0.5 + hist[det][0] * len(BLAB), r), xytext=(2, 0), textcoords='offset points', fontsize=5.5, va='center', ha='left', color=DARK)
        else:
            axh.bar(np.arange(len(BLAB)), -0.78 * hist[det] / hmax, width=0.8, bottom=r + 0.39, color=GREY, lw=0, zorder=3)
            axh.hlines(r + 0.39, -0.5, len(BLAB) - 0.5, color='#cccccc', lw=0.4, zorder=2)

    ax.set_xticks(x); ax.set_xticklabels(XLAB, rotation=60, ha='right', rotation_mode='anchor', fontsize=7)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ticks = list(range(1, TOP + 1)) if TOP <= 10 else [t for t in [1, 5, 10, 15, 20, 25, 30] if t <= TOP]
    ax.set_ylim(TOP + (1.6 if rest else 0.6), 0.4); ax.set_yticks(ticks + ([TOP + 1] if rest else [])); ax.set_yticklabels([str(t) for t in ticks] + ([f'>{TOP}'] if rest else []))
    ax.set_ylabel('Global rank', fontsize=8)
    ax.grid(axis='both', color=LIGHT, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(colors=DARK, labelsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GREY)

    axh.set_xlim(-0.6, len(BLAB) - 0.4)
    if MODE == 'line':
        axh.set_xscale('symlog', linthresh=2, linscale=0.6); axh.set_xlim(0, 128)
        axh.set_xticks([0, 8, 32, 128]); axh.set_xticklabels(['0', '8', '32', '128'], fontsize=6, color=DARK); axh.minorticks_off()
    elif MODE == 'nonzero':
        axh.set_xticks([-0.5, len(BLAB) - 0.5]); axh.set_xticklabels(['0', '1'], fontsize=6, color=DARK)
    else:
        axh.set_xticks(np.arange(len(BLAB))); axh.set_xticklabels(BLAB, rotation=90, fontsize=6, color=DARK)
    axh.tick_params(axis='y', left=False, labelleft=False); axh.tick_params(axis='x', length=1.5, pad=2)
    for s in ('top', 'right', 'left', 'bottom'):
        axh.spines[s].set_visible(False)
    axh.set_xlabel('share of $b^* > 0$' if MODE == 'nonzero' else 'selected $b^*$', fontsize=7, labelpad=1)
    hdr = dict(xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=6.5, va='bottom', color=GREY, annotation_clip=False)
    axh.annotate({'linear': 'share of pairs', 'log': 'share (log)', 'line': 'share (log)', 'nonzero': 'share of $b^* > 0$'}[MODE], (0.5, 0.4), xytext=(0, 3), ha='center', **hdr)
    axh.annotate('mean $b^*$', (0.0, 0.4), xytext=(-4, 3), ha='right', **hdr)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=C1, label='sharp: no buffer ($b^*=0$) on most anomalies'),
               Patch(color=C2, label='broad: a buffer on most anomalies')]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=6, frameon=False, handlelength=1.0, handleheight=0.7, borderaxespad=0.1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = OUT + ('' if MODE == 'line' else '_' + MODE) + ('' if TOP == 9 else f'_top{TOP}')
    fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=300)
    print('wrote', out + '.pdf')


if __name__ == '__main__':
    main()
