"""Five flagged anomalies: left = the flagged one, right = the best-caught one in the same series; series + 3 detectors that catch the others."""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/home/sylli/Documents/autovus'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/app/api')
from tsad_parser import TSADParser
from bench import load_series, load_score
EX = [('188_SMD', 10), ('139_WSD', 7), ('265_IOPS', 10), ('207_SMD', 2), ('187_SMD', 3)]
a = pd.read_csv(f'{ROOT}/experiments/local_benchmark/anomalies.csv', usecols=['series', 'detector', 'k', 'auto20'])
p = TSADParser()
fig, axes = plt.subplots(len(EX) * 2, 2, figsize=(14, 3.0 * len(EX)), gridspec_kw={'height_ratios': [1, 1.3] * len(EX)})
for row, (prefix, kf) in enumerate(EX):
    F = [s for s in a.series.unique() if s.startswith(prefix)][0]
    M = a[a.series == F].pivot(index='detector', columns='k', values='auto20')
    values, label = load_series(p, F)
    edges = np.flatnonzero(np.diff(np.r_[0, label, 0]))
    st, en = edges[::2], edges[1::2] - 1
    others = M.drop(columns=kf)
    kg = int(M.median().drop(kf).idxmax())                       # best-caught other anomaly
    dets = others.mean(1).sort_values(ascending=False).index[:3]  # detectors that catch the others best
    for col, k in enumerate([kf, kg]):
        L = en[k] - st[k] + 1
        w = max(3 * L, 150)
        lo, hi = max(0, st[k] - w), min(len(values), en[k] + w + 1)
        x = np.arange(lo, hi)
        ax = axes[2 * row, col]
        ax.plot(x, values[lo:hi], color='#152033', lw=1)
        ax.axvspan(st[k], en[k] + 1, color='#F4B942', alpha=0.5)
        tag = 'FLAGGED' if col == 0 else 'best caught'
        ax.set_title(f"{F[:-4]}  A{k+1} ({tag}, len {L})", fontsize=9, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        ax2 = axes[2 * row + 1, col]
        for i, d in enumerate(dets):
            s = load_score(p, F, d)
            s = (s - s.min()) / max(s.max() - s.min(), 1e-12)
            ax2.plot(x, s[lo:hi] + i * 1.1, lw=1, label=f"{d} ({M.loc[d, k]:.2f})")
        ax2.axvspan(st[k], en[k] + 1, color='#F4B942', alpha=0.35)
        ax2.set_yticks([]); ax2.legend(fontsize=7, loc='upper left', frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=90)
print('saved')
