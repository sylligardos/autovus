"""Summaries for the benchmark-wide AutoVUS paragraph (Sec V), from autovus_benchmark.py output.

Prints: global detector ranking under VUS@2..128, AutoVUS@128/20 and AUC-PR; rank agreement
(global and per series); top-1 flips with the no-buffer reference; random-detector control;
selected-buffer distribution.
Usage: conda run -n autovus python src/experiments/analyze_autovus_benchmark.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau as _kt, spearmanr


def kendalltau(x, y):
    m = ~(np.isnan(np.asarray(x, float)) | np.isnan(np.asarray(y, float)))
    return _kt(np.asarray(x, float)[m], np.asarray(y, float)[m])

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "experiments", "autovus_benchmark")
LS = [2, 4, 8, 16, 20, 32, 64, 128]
COLS = [f'vus_{b}' for b in LS] + ['autovus_128', 'autovus_20', 'auc_pr']
NAMES = {**{f'vus_{b}': f'VUS@{b}' for b in LS}, 'autovus_128': 'AutoVUS@128', 'autovus_20': 'AutoVUS@20',
         'auc_pr': 'AUC-PR', 'vus_0': 'VUS@0'}


def main():
    d = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    if 'error' in d:
        print('errors:', d['error'].notna().sum()); d = d[d['error'].isna()]
    rnd = d[d['detector'].str.startswith('Random_')]
    d = d[~d['detector'].str.startswith('Random_')]
    print(f'pairs {len(d)}, series {d.series.nunique()}, detectors {d.detector.nunique()} (mean over the series each detector scores)')

    # 1. global ranking (mean over series)
    g = d.groupby('detector')[COLS].mean()
    ranks = g.rank(ascending=False).astype(int)
    tab = ranks.copy(); tab.columns = [NAMES[c] for c in tab.columns]
    tab['mean VUS@128'] = g['vus_128'].round(3); tab['mean AutoVUS@128'] = g['autovus_128'].round(3)
    print('\n== Global ranking (rank of the mean score over series)')
    print(tab.sort_values('VUS@128').to_string())
    print('\n== Global rank agreement with AutoVUS@128 (Kendall tau over the 32 detector means)')
    for c in COLS:
        if c != 'autovus_128':
            print(f'  {NAMES[c]:12s} tau={kendalltau(g[c], g["autovus_128"]).statistic:.3f}  '
                  f'spearman={spearmanr(g[c], g["autovus_128"]).statistic:.3f}')
    print('== Global rank agreement with VUS@128')
    for c in COLS:
        if c != 'vus_128':
            print(f'  {NAMES[c]:12s} tau={kendalltau(g[c], g["vus_128"]).statistic:.3f}')

    # 2. per-series agreement and top-1 flips
    print('\n== Per-series Kendall tau and top-1 agreement (median tau over series; share of series with the same top-1)')
    piv = {c: d.pivot(index='series', columns='detector', values=c) for c in COLS}
    ref = 'autovus_128'
    for c in COLS:
        if c == ref:
            continue
        taus = [kendalltau(piv[c].loc[s], piv[ref].loc[s]).statistic for s in piv[c].index]
        same = np.mean([piv[c].loc[s].idxmax() == piv[ref].loc[s].idxmax() for s in piv[c].index])
        print(f'  AutoVUS@128 vs {NAMES[c]:12s} tau median={np.nanmedian(taus):.3f} mean={np.nanmean(taus):.3f}  same top-1 {same:.1%}')
    print('  (reference rows: VUS@128 vs other VUS)')
    for c in ['vus_16', 'vus_20', 'vus_32', 'vus_64', 'auc_pr']:
        taus = [kendalltau(piv[c].loc[s], piv['vus_128'].loc[s]).statistic for s in piv[c].index]
        same = np.mean([piv[c].loc[s].idxmax() == piv['vus_128'].loc[s].idxmax() for s in piv[c].index])
        print(f'  VUS@128     vs {NAMES[c]:12s} tau median={np.nanmedian(taus):.3f} mean={np.nanmean(taus):.3f}  same top-1 {same:.1%}')

    # 3. flips: is the AutoVUS top-1 still good?
    print('\n== Top-1 flips VUS@128 -> AutoVUS@128')
    rows = []
    for s in piv['vus_128'].index:
        tv, ta = piv['vus_128'].loc[s].idxmax(), piv['autovus_128'].loc[s].idxmax()
        if tv == ta:
            continue
        rv = d[(d.series == s) & (d.detector == tv)].iloc[0]; ra = d[(d.series == s) & (d.detector == ta)].iloc[0]
        rows.append(dict(series=s, vus_top=tv, auto_top=ta, vus_gap=rv.vus_128 - ra.vus_128, auto_gap=ra.autovus_128 - rv.autovus_128,
                         auc_vus_top=rv.auc_pr, auc_auto_top=ra.auc_pr, b_vus_top=rv.b_mean_128, b_auto_top=ra.b_mean_128,
                         rank_auto_top_under_vus=int(piv['vus_128'].loc[s].rank(ascending=False)[ta]),
                         rank_vus_top_under_auto=int(piv['autovus_128'].loc[s].rank(ascending=False)[tv])))
    f = pd.DataFrame(rows)
    n = len(piv['vus_128'])
    print(f'  flips: {len(f)}/{n} series ({len(f) / n:.1%})')
    if len(f):
        print(f'  VUS gap between the two (median) {f.vus_gap.median():.3f}; AutoVUS gap {f.auto_gap.median():.3f}')
        print(f'  AutoVUS top has higher AUC-PR than VUS top in {(f.auc_auto_top > f.auc_vus_top).mean():.1%} of flips')
        print(f'  mean b* (L=128): VUS top {f.b_vus_top.median():.1f}, AutoVUS top {f.b_auto_top.median():.1f} (medians)')
        print(f'  AutoVUS top ranked under VUS: median {f.rank_auto_top_under_vus.median():.0f}, '
              f'<=3 in {(f.rank_auto_top_under_vus <= 3).mean():.1%}; VUS top under AutoVUS: median {f.rank_vus_top_under_auto.median():.0f}')
        f.to_csv(os.path.join(OUT, 'flips.csv'), index=False)

    # 4. random control
    print('\n== Random detector control (uniform scores, mean +- std over series x seeds)')
    for c in ['vus_0'] + COLS:
        print(f'  {NAMES[c]:12s} {rnd[c].mean():.3f} +- {rnd[c].std():.3f}   p95 {rnd[c].quantile(.95):.3f}')
    print('  Real detectors, same columns:')
    for c in ['vus_0', 'vus_16', 'vus_128', 'autovus_128', 'autovus_20']:
        print(f'  {NAMES[c]:12s} {d[c].mean():.3f} +- {d[c].std():.3f}')

    # 5. selected buffers
    a = pd.read_csv(os.path.join(OUT, 'anomalies.csv'))
    a = a[~a.detector.str.startswith('Random_')]
    b = a.b_star_128
    print(f'\n== Selected buffers at L=128 over {len(a)} (anomaly, detector) pairs: b*=0 {np.mean(b == 0):.1%}, '
          f'0<b*<128 {np.mean((b > 0) & (b < 128)):.1%}, b*=128 {np.mean(b == 128):.1%}; median {b.median():.0f}, '
          f'p90 {b.quantile(.9):.0f}')
    print('  detection labels:', a.detection.value_counts(normalize=True).round(3).to_dict())
    print(f'  pairs with every b* = 128 (AutoVUS raw = VUS): {np.mean(d.b_max_128 == 128):.1%} have max b*=128; '
          f'share of pairs with vus_adapt_128 == vus_128: {np.mean(np.isclose(d.vus_adapt_128, d.vus_128)):.1%}')


if __name__ == '__main__':
    main()
