"""Quick check: anomalies that most informative detectors score worst within their series (local AutoVUS@20)."""
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
a = pd.read_csv('experiments/local_benchmark/anomalies.csv', usecols=['series', 'detector', 'k', 'len', 'auto20'])
out = []
for s, g in a.groupby('series'):
    M = g.pivot(index='detector', columns='k', values='auto20').dropna()
    K = M.shape[1]
    if K < 3:
        continue
    X = M.values
    inf = X.std(1) >= 0.05                      # informative: scores vary across anomalies
    X = X[inf]
    D = len(X)
    if D < 5:
        continue
    R = np.vstack([rankdata(x) for x in X])     # 1 = worst, ties get the average rank
    u = (R - 0.5) / K                           # null: mean 0.5, var (K^2-1)/(12 K^2) (ties only lower it)
    mu = u.mean(0)
    z = (0.5 - mu) / np.sqrt((K * K - 1) / (12 * K * K) / D)
    p = norm.sf(z)                              # one-sided: ranked low by the detectors
    m = (R == R.min(1, keepdims=True)).sum(0)   # detectors with it (tied) worst, for display
    lens = g.drop_duplicates('k').set_index('k')['len'].reindex(M.columns).values
    others = (X.sum(1, keepdims=True) - X) / (K - 1)
    for k in range(K):
        out.append((s, s.split('_')[1], K, D, k, int(lens[k]), m[k], p[k], float(np.median(X[:, k])), float(np.median(others[:, k]))))
r = pd.DataFrame(out, columns=['series', 'dataset', 'K', 'D', 'k', 'len', 'votes', 'p', 'med_this', 'med_others'])
# Benjamini-Hochberg over all tested anomalies
r = r.sort_values('p').reset_index(drop=True)
n = len(r); r['q'] = np.minimum.accumulate((r.p * n / (np.arange(n) + 1))[::-1])[::-1].clip(upper=1)
sig = r[r.q < 0.05]
print(f'series tested {r.series.nunique()}, anomalies tested {n}, significant (BH q<0.05): {len(sig)} in {sig.series.nunique()} series')
print('by dataset (significant / tested series):')
print(pd.concat([sig.groupby("dataset").series.nunique().rename("sig_series"), r.groupby("dataset").series.nunique().rename("series")], axis=1).fillna(0).astype(int).sort_values("sig_series", ascending=False).to_string())
print('\nK of significant series:', sig.drop_duplicates('series').K.describe()[['min', '50%', 'max']].to_dict())
print('\ntop examples (K <= 20):')
ex = sig[sig.K <= 20].sort_values(['q', 'votes'], ascending=[True, False]).head(10)
for _, e in ex.iterrows():
    print(f"  {e.series[:45]:45s} K={e.K:3d} A{e.k+1:<3d} len={e.len:4d} worst for {e.votes:4.0f}/{e.D:2d} q={e.q:.1e}  median this {e.med_this:.2f} vs others {e.med_others:.2f}")
r.to_csv('experiments/local_eval/odd_check.csv', index=False)
