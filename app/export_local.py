"""
The tables of the local-evaluation pages (Local, Finder, Coverage, Buffers,
Suspicious), written to app/data/<benchmark>/local/. Reads the finished
benchmark exports; no metric is recomputed here.

    nice -n 19 conda run -n autovus python app/export_local.py

Inputs (experiments/, not in git):
    local_benchmark/series/<series>.npz   per-anomaly local scores (variant C), AutoVUS@20
    local_benchmark/global_adjusted.npz   global VUS-PR at every L (field 'vus')
    buffers/anomalies_ps20.csv            per-side AutoVUS buffers and types, L = 20
app/data/<benchmark>/anomalies.csv        per-side buffers and types, L = 128 (type_ps)

Outputs (app/data/<benchmark>/local/):
    series/<series>.json  anomalies, per-anomaly local AutoVUS@20 (detector x anomaly),
                          global AutoVUS, global VUS-PR at L = 0..128
    pairs.json            per (series, detector): K, global AutoVUS, mean and min local
    cover.json            per series: the detector x anomaly matrix (x100, -999 = no run)
    buffers_20.json, buffers_128.json
                          per (series, detector): counts of exact/early/late/wide/missed,
                          mean buffer before and after
    odd.json              suspicious anomalies with a window around each
    game.json             the game's rounds (series, detector, anomaly); edit by hand
    stitched.json         per series: AutoVUS@20 of the stitched score (from scores20.json,
                          app/export_rankings20.py)
    speed.json            the Intro speed box: per measure (original VUS, FastVUS, FastVUS-GPU,
                          AutoVUS) the box of the times per (series, detector), their total, and
                          the mean per-pair ratio to the original VUS (the paper's runtime runs, L = 128)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'app' / 'api'))
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402

EXP = ROOT / 'experiments'
TYPES = ['exact', 'early', 'late', 'wide', 'missed']
ODD_Q = 0.05        # suspicious: Benjamini-Hochberg q below this
ODD_GAP = 0.2       # and the median local AutoVUS at least this much below the other anomalies
GAME = [            # first draft; the user picks the final 6-10 rounds
    ('682_YAHOO_id_132_WebService_tr_500_1st_354.csv', 'Chronos', 0),
    ('682_YAHOO_id_132_WebService_tr_500_1st_354.csv', 'TimesFM', 4),
    ('282_NEK_id_6_WebService_tr_500_1st_64.csv', 'AnomalyTransformer', 4),
    ('282_NEK_id_6_WebService_tr_500_1st_64.csv', 'Chronos', 4),
    ('285_NEK_id_9_WebService_tr_500_1st_780.csv', 'POLY', 0),
]


def adjusted(area, random, L):
    """Chance-adjusted local VUS-PR at L: running means over the buffers 0..L."""
    v = area[..., :L + 1].mean(-1)
    r = random[..., :L + 1].mean(-1)
    return (v - r) / np.maximum(1 - r, 1e-10)


def r2(x, d=2):
    return [round(float(v), d) for v in np.asarray(x).ravel()]


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def local_tables(out, detectors):
    """series/*.json, pairs.json and cover.json in one pass over the local export."""
    g = np.load(EXP / 'local_benchmark' / 'global_adjusted.npz')
    gi = {(s, d): i for i, (s, d) in enumerate(zip(g['series'], g['detector']))}
    di = {d: i for i, d in enumerate(detectors)}
    pairs = {'detectors': detectors, 'series': [], 's': [], 'd': [], 'K': [], 'a': [], 'm': [], 'n': []}
    cover = {'detectors': detectors, 'series': [], 'K': [], 'M': []}
    files = sorted((EXP / 'local_benchmark' / 'series').glob('*.npz'))
    for i, f in enumerate(files):
        z = np.load(f)
        name = f.name[:-4]                                   # '<series>.csv'
        dets = [str(d) for d in z['detectors']]
        loc = adjusted(z['auto20_area'], z['auto20_random'], 20)         # (D, K) buffers capped at b*
        K = loc.shape[1]
        vus = np.stack([g['vus'][gi[(name, d)]] for d in dets])          # (D, 129) plain VUS-PR
        dump({'file': name, 'anomalies': [[int(a), int(b)] for a, b in zip(z['start'], z['end'])], 'detectors': dets,
              'auto': [r2(row) for row in loc], 'autoG': r2(z['auto20_adjusted']), 'vusG': [r2(row, 3) for row in vus]},
             out / 'series' / (name + '.json'))
        si = len(pairs['series'])
        pairs['series'].append(name)
        for j, d in enumerate(dets):
            pairs['s'].append(si); pairs['d'].append(di[d]); pairs['K'].append(K)
            pairs['a'].append(round(float(z['auto20_adjusted'][j]), 3))
            pairs['m'].append(round(float(loc[j].mean()), 3)); pairs['n'].append(round(float(loc[j].min()), 3))
        M = np.full((len(detectors), K), -999, dtype=int)
        M[[di[d] for d in dets]] = np.round(loc * 100).astype(int)
        cover['series'].append(name); cover['K'].append(K); cover['M'].append(M.ravel().tolist())
        if (i + 1) % 100 == 0:
            print(f'  local {i + 1}/{len(files)}', flush=True)
    dump(pairs, out / 'pairs.json')
    dump(cover, out / 'cover.json')
    print(f'local: {len(files)} series, {len(pairs["s"])} pairs')


def buffer_table(df, out_path, L):
    """Per (series, detector): counts per detection type and the mean buffers."""
    series = sorted(df.series.unique())
    dets = sorted(df.detector.unique())
    si = {s: i for i, s in enumerate(series)}
    di = {d: i for i, d in enumerate(dets)}
    c = df.groupby(['series', 'detector', 'type']).size().unstack(fill_value=0).reindex(columns=TYPES, fill_value=0)
    b = df.groupby(['series', 'detector'])[['b_left', 'b_right']].mean()
    rows = [[si[f], di[d]] + [int(x) for x in r.values] + [round(float(b.loc[(f, d), 'b_left']), 1), round(float(b.loc[(f, d), 'b_right']), 1)]
            for (f, d), r in c.iterrows()]
    dump({'L': L, 'types': TYPES, 'series': series, 'detectors': dets, 'rows': rows}, out_path)
    print(f'buffers L={L}: {len(rows)} pairs')


def odd_table(out, p):
    """Suspicious anomalies: ranked low by most informative detectors (mean-rank z-test, BH), and clearly lower."""
    a = pd.read_csv(EXP / 'local_benchmark' / 'anomalies.csv', usecols=['series', 'detector', 'k', 'len', 'auto20'])
    rows = []
    for s, g in a.groupby('series'):
        M = g.pivot(index='detector', columns='k', values='auto20').dropna()
        K = M.shape[1]
        if K < 3:
            continue
        X = M.values[M.values.std(1) >= 0.05]           # informative detectors: their scores vary over the anomalies
        D = len(X)
        if D < 5:
            continue
        R = np.vstack([rankdata(x) for x in X])          # 1 = worst, ties get the average rank
        mu = ((R - 0.5) / K).mean(0)
        z = (0.5 - mu) / np.sqrt((K * K - 1) / (12 * K * K) / D)
        worst = (R == R.min(1, keepdims=True)).sum(0)
        others = (X.sum(1, keepdims=True) - X) / (K - 1)   # per detector: mean over the other anomalies
        for k in range(K):
            rows.append((s, s.split('_')[1], K, D, k, int(worst[k]), float(norm.sf(z[k])),
                         float(np.median(X[:, k])), float(np.median(others[:, k]))))
    r = pd.DataFrame(rows, columns=['series', 'dataset', 'K', 'D', 'k', 'worst', 'p', 'this', 'others'])
    r = r.sort_values('p').reset_index(drop=True)
    n = len(r)
    r['q'] = np.minimum.accumulate((r.p * n / (np.arange(n) + 1))[::-1])[::-1].clip(upper=1)
    tested = r.groupby('dataset').agg(series=('series', 'nunique'), anomalies=('k', 'size'))
    flag = r[(r.q < ODD_Q) & (r.others - r.this >= ODD_GAP)]
    mm = lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-12)  # noqa: E731
    items = []
    for F, g in flag.groupby('series'):
        values, label = load_series(p, F)
        edges = np.flatnonzero(np.diff(np.r_[0, label, 0]))
        st, en = edges[::2], edges[1::2] - 1
        S = [load_score(p, F, d) for d in p.get_detectors()]
        S = np.vstack([mm(x) for x in S if x is not None and len(x) == len(values)])
        v = mm(np.asarray(values, dtype=float))
        for _, e in g.iterrows():
            k = int(e.k)
            L = int(en[k] - st[k] + 1)
            w = max(3 * L, 100)
            lo, hi = max(0, st[k] - w), min(len(v), en[k] + w + 1)
            idx = np.arange(lo, hi, max(1, (hi - lo) // 300))
            items.append({'file': F, 'dataset': e.dataset, 'K': int(e.K), 'D': int(e.D), 'k': k, 'len': L,
                          'worst': int(e.worst), 'this': round(e.this, 2), 'others': round(e.others, 2), 'q': float(f'{e.q:.2e}'),
                          'lo': int(lo), 'hi': int(hi), 'start': int(st[k]), 'end': int(en[k]),
                          'values': r2(v[idx], 3), 'score': r2(np.median(S[:, idx], 0), 3)})
    dump({'q': ODD_Q, 'gap': ODD_GAP,
          'tested': {d: {'series': int(t.series), 'anomalies': int(t.anomalies)} for d, t in tested.iterrows()},
          'items': items}, out / 'odd.json')
    print(f'odd: {n} anomalies tested, {len(items)} suspicious in {flag.series.nunique()} series')


def speed_table(out):
    """speed.json from the paper's runtime experiments (summed 'Metric time')."""
    res = EXP / 'all_metrics_tsb_def_params' / 'results'
    opt = '_128_1_dynamic_plus_globalmask_function_optimized_subsampled.csv'
    runs = [('vus', 'VUS', 'original', res / 'tsb_VUS_128.csv'),
            ('fastvus', 'FastVUS', 'CPU, exact', res / ('tsb_FF-VUS' + opt)),
            ('fastvus_gpu', 'FastVUS-GPU', 'approximate', res / ('tsb_FF-VUS-GPU' + opt)),
            ('autovus', 'AutoVUS', 'on FastVUS, CPU', EXP / 'autovus_runtime' / 'results' / 'tsb_autovus_L128_step1.csv')]
    base, rows = None, []
    for key, label, tag, path in runs:
        t = pd.read_csv(path).set_index(['Time series', 'Detector'])['Metric time']
        if base is None:
            base = t
        ratio = None if key == 'vus' else round(float((base / t.reindex(base.index)).mean()), 2)
        # The box of the per-pair times: quartiles, whiskers at the last value within 1.5 IQR.
        v = t.to_numpy(dtype=float)
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        lo, hi = float(v[v >= q1 - 1.5 * (q3 - q1)].min()), float(v[v <= q3 + 1.5 * (q3 - q1)].max())
        rows.append({'key': key, 'label': label, 'tag': tag, 'seconds': round(float(t.sum()), 1), 'ratio': ratio,
                     'box': [float(f'{x:.4g}') for x in (lo, q1, med, q3, hi)]})
    dump({'pairs': int(len(base)), 'series': int(base.index.get_level_values(0).nunique()), 'L': 128, 'rows': rows}, out / 'speed.json')
    print('speed:', ', '.join(f"{r['label']} {r['seconds']} s x{r['ratio']}" for r in rows))


def stitched_table(out, benchmark):
    """stitched.json: AutoVUS@20 of the stitched score per series, copied from the Rankings table."""
    src = ROOT / 'app' / 'data' / benchmark / 'scores20.json'
    if not src.exists():
        print('stitched: no scores20.json, run app/export_rankings20.py')
        return
    with open(src) as f:
        files = json.load(f)['files']
    score = {name: round(float(rows['Stitched score']['autovus']), 3) for name, rows in files.items()
             if 'autovus' in rows.get('Stitched score', {})}
    dump({'L': 20, 'score': score}, out / 'stitched.json')
    print(f'stitched: {len(score)} series')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='tsb-ad')
    ap.add_argument('--only', nargs='*', default=None, help='local buffers odd game speed stitched (default: all)')
    a = ap.parse_args()
    out = ROOT / 'app' / 'data' / a.benchmark / 'local'
    parts = set(a.only or ['local', 'buffers', 'odd', 'game', 'speed', 'stitched'])
    p = TSADParser()
    detectors = sorted(str(d) for d in p.get_detectors())
    if 'local' in parts:
        local_tables(out, detectors)
    if 'buffers' in parts:
        b20 = pd.read_csv(EXP / 'buffers' / 'anomalies_ps20.csv', usecols=['series', 'detector', 'b_left', 'b_right', 'type'])
        buffer_table(b20, out / 'buffers_20.json', 20)
        b128 = pd.read_csv(ROOT / 'app' / 'data' / a.benchmark / 'anomalies.csv', usecols=['file', 'detector', 'b_left_ps', 'b_right_ps', 'type_ps'])
        b128 = b128[b128.detector.isin(detectors)].rename(columns={'file': 'series', 'b_left_ps': 'b_left', 'b_right_ps': 'b_right', 'type_ps': 'type'})
        buffer_table(b128, out / 'buffers_128.json', 128)
    if 'odd' in parts:
        odd_table(out, p)
    if 'speed' in parts:
        speed_table(out)
    if 'stitched' in parts:
        stitched_table(out, a.benchmark)
    if 'game' in parts and not (out / 'game.json').exists():
        dump({'L': 20, 'sides': 'per_side', 'rounds': [{'file': f, 'detector': d, 'k': k} for f, d, k in GAME]}, out / 'game.json')
        print('game: first draft written')


if __name__ == '__main__':
    main()
