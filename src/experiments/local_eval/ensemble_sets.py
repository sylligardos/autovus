"""
Experiment: do a few complementary detectors, averaged, beat the best single one?
Kept as the motivation for combining detectors over time (future work); the
demo does not read it.

The score of a set of detectors = the mean over the anomalies of the best
local AutoVUS@20 any detector of the set reaches on the anomaly. The sets are
greedy: the best detector alone, then twice the detector that raises the
score of the set the most.

Per series, the average ensemble of the set (the mean of the min-max scaled
scores of its detectors) for 2 and 3 detectors is scored like every detector
(local AutoVUS@20 per anomaly, AutoVUS). The single detector is rescored too,
as a check against cover.json.

    nice -n 19 conda run -n autovus python src/experiments/local_eval/ensemble_sets.py     # 49 s

Output: experiments/local_eval/ensemble_sets.json, per dataset of the Profiles page:
    all         the score of all detectors together (the ceiling of any set)
    sets        for 1, 2, 3 detectors: the detectors, the score of the set ('best'),
                the mean local AutoVUS of its average ensemble ('ensemble'), the mean
                AutoVUS of the ensemble per series ('series')
    members     per detector of the 3-set: mean seconds per series, and its best substitute
    local       per anomaly: local AutoVUS x100 of the ensemble of 1, 2 and 3 detectors
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'app' / 'api'))
from autovus import FastVUS, AutoVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
from check_profiles import load  # noqa: E402

L = 20


def score(X, S):
    return float(X[:, S].max(1).mean())


def greedy(X, k):
    S = []
    for _ in range(k):
        S.append(max((d for d in range(X.shape[1]) if d not in S), key=lambda d: score(X, S + [d])))
    return S


def scaled(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def evaluate(label, s):
    """Per anomaly the local AutoVUS@20 (as in app/export_local.py), and the AutoVUS of the series."""
    r = AutoVUS(FastVUS(L=L, cumulative=True, per_anomaly=True)).compute(label, s)
    v, rnd = r['local']['area'][..., :L + 1].mean(-1), r['local']['random'][..., :L + 1].mean(-1)
    return (v - rnd) / np.maximum(1 - rnd, 1e-10), float(r['adjusted'])


def dataset(p, benchmark, name, runtime):
    X, series, k, names, detectors, _, _ = load(benchmark, name)
    S = greedy(X, 3)
    exact = max(combinations(range(X.shape[1]), 3), key=lambda c: score(X, list(c)))
    files = [names[i] for i in sorted(set(series.tolist()))]
    sec = lambda d: round(float(runtime.loc[runtime['File Name'].isin(files) & (runtime['Model Name'] == d), 'Time Cost'].mean()), 2)

    local = np.full((3, len(X)), np.nan)
    glob = [[], [], []]
    for i in sorted(set(series.tolist())):
        rows = np.where(series == i)[0]
        _, label = load_series(p, names[i])
        scores = [load_score(p, names[i], detectors[d]) for d in S]
        if any(x is None for x in scores):
            continue
        Z = np.array([scaled(x) for x in scores])
        for n in range(3):
            loc, g = evaluate(label, Z[:n + 1].mean(0))
            local[n, rows] = np.asarray(loc)[k[rows]]
            glob[n].append(g)
    ok = ~np.isnan(local).any(0)
    check = float(np.abs(local[0, ok] - X[ok, S[0]]).max())          # the single detector against cover.json

    members = []
    for m in S:
        rest = [d for d in S if d != m]
        sub = max((d for d in range(X.shape[1]) if d not in S), key=lambda d: score(X, rest + [d]))
        members.append({'detector': detectors[m], 'seconds': sec(detectors[m]),
                        'substitute': detectors[sub], 'substitute_seconds': sec(detectors[sub]), 'substitute_best': round(score(X, rest + [sub]), 3)})
    out = {
        'dataset': name, 'anomalies': len(X), 'scored': int(ok.sum()), 'all': round(score(X, list(range(X.shape[1]))), 3),
        'sets': [{'detectors': [detectors[d] for d in S[:n + 1]], 'best': round(score(X, S[:n + 1]), 3),
                  'ensemble': round(float(local[n, ok].mean()), 3), 'series': round(float(np.mean(glob[n])), 3)} for n in range(3)],
        'members': members,
        'local': [[int(round(v * 100)) if np.isfinite(v) else -999 for v in row] for row in local],
    }
    e = out['sets']
    print(f"{name:<9} {' + '.join(e[2]['detectors']):<40} best {e[0]['best']:.2f} {e[1]['best']:.2f} {e[2]['best']:.2f} (all {out['all']:.2f})   "
          f"ensemble {e[0]['ensemble']:.2f} {e[1]['ensemble']:.2f} {e[2]['ensemble']:.2f}   series {e[0]['series']:.2f} {e[1]['series']:.2f} {e[2]['series']:.2f}   "
          f"exact 3-set {score(X, list(exact)):.3f} {'same' if set(exact) == set(S) else [detectors[d] for d in exact]}   check {check:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='tsb-ad')
    a = ap.parse_args()
    folder = ROOT / 'app' / 'data' / a.benchmark / 'local' / 'profiles'
    out_path = ROOT / 'experiments' / 'local_eval' / 'ensemble_sets.json'
    p = TSADParser()
    runtime = pd.concat([pd.read_csv(f) for f in sorted(p.runtime_path.glob('*.csv'))], ignore_index=True)
    t0 = time.time()
    out = {r['dataset']: dataset(p, a.benchmark, r['dataset'], runtime) for r in json.load(open(folder / 'index.json'))}
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f"wrote {out_path} in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
