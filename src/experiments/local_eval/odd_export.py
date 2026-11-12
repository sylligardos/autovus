"""Suspicious-anomalies page data: flagged = q < 0.05 (odd_check.py) and median score >= 0.2 below the other anomalies.

    conda run -n autovus python src/experiments/local_eval/odd_export.py <odd_check.csv> <out.json>

Per flagged anomaly: a window around it (anomaly +- max(3 len, 100), at most 300 points)
with the series (min-max) and the median over detectors of the min-max scores.
"""
import json, sys
import numpy as np, pandas as pd
ROOT = '/home/sylli/Documents/autovus'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/app/api')
from tsad_parser import TSADParser
from bench import load_series, load_score

r = pd.read_csv(sys.argv[1])
r['gap'] = r.med_others - r.med_this
flag = r[(r.q < 0.05) & (r.gap >= 0.2)].copy()
tested = r.groupby('dataset').agg(series=('series', 'nunique'), anomalies=('k', 'size'))
p = TSADParser()
mm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-12)
R = lambda a, d=3: [round(float(v), d) for v in a]
items = []
for F, g in flag.groupby('series'):
    values, label = load_series(p, F)
    edges = np.flatnonzero(np.diff(np.r_[0, label, 0])); st, en = edges[::2], edges[1::2] - 1
    S = [load_score(p, F, d) for d in p.get_detectors()]
    S = np.vstack([mm(s) for s in S if s is not None and len(s) == len(values)])
    v = mm(values)
    for _, e in g.iterrows():
        k = int(e.k); L = en[k] - st[k] + 1; w = max(3 * L, 100)
        lo, hi = max(0, st[k] - w), min(len(v), en[k] + w + 1)
        step = max(1, (hi - lo) // 300)
        idx = np.arange(lo, hi, step)
        items.append({'series': F[:-4], 'dataset': e.dataset, 'K': int(e.K), 'D': int(e.D), 'k': k, 'len': int(L),
                      'worst': int(e.votes), 'this': round(e.med_this, 2), 'others': round(e.med_others, 2), 'q': float(f'{e.q:.2e}'),
                      'lo': int(lo), 'hi': int(hi), 'a': [int(st[k]), int(en[k])],
                      'values': R(v[idx]), 'score': R(np.median(S[:, idx], 0))})
out = {'tested': {d: {'series': int(t.series), 'anomalies': int(t.anomalies)} for d, t in tested.iterrows()}, 'items': items}
json.dump(out, open(sys.argv[2], 'w'), separators=(',', ':'))
print(len(items), 'flagged in', flag.series.nunique(), 'series;', (flag.dataset == 'Stock').sum(), 'Stock')
