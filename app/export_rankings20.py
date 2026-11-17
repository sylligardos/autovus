"""
The Rankings table at L=20 (app/data/<benchmark>/scores20.json), reusing what exists.

Reused, every (series, detector):
    AutoVUS@20       experiments/local_benchmark/series/<series>.npz  auto20_adjusted
    VUS-PR@20        same file, fast128_vus[:, 20]
    Adjusted VUS@20  experiments/local_benchmark/global_adjusted.npz  adjusted[:, 20]
    AUC-PR           app/data/<benchmark>/scores.json                 auc_pr
Derived, per series:
    Oracle           the detector with the best AUC-PR, all its values
Computed, per series (the only new work):
    Stitched min-max, Stitched median   app/api/oracle.py stitch with the winners
                     of the local AutoVUS@20 (coverage tie rule), the same
                     four metrics computed as above

    nice -n 19 conda run -n autovus python app/export_rankings20.py --workers 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.metrics import average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'app', 'api'))
from autovus import FastVUS, AdjustedVUS, AutoVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
import oracle as orc  # noqa: E402

L = 20
EXP = os.path.join(_ROOT, 'experiments', 'local_benchmark')
OUT = os.path.join(_ROOT, 'app', 'data', 'tsb-ad')
STITCHED = {'Stitched min-max': 'minmax', 'Stitched median': 'median'}


def metrics(label, score):
    """AUC-PR, VUS-PR@20, Adjusted VUS@20 (L=128 cumulative curve read at 20, as the reused
    numbers) and AutoVUS@20."""
    eng = FastVUS(L=128, cumulative=True)
    vus, rnd = AdjustedVUS(eng).curves(label, score)
    auto = AutoVUS(FastVUS(L=L, cumulative=True, per_anomaly=True)).compute(label, score)
    return {'auc_pr': float(average_precision_score(label, score)), 'vus_pr': float(vus[L]),
            'skill_full': float(AdjustedVUS.adjust(vus[L], rnd[L])), 'autovus': float(auto['adjusted'])}


def stitched(file):
    """The two stitched rows of one series, or {} when there is nothing to stitch."""
    z = np.load(os.path.join(EXP, 'series', file + '.npz'))
    if len(z['detectors']) == 0 or len(z['start']) == 0:
        return file, {}, []
    p = TSADParser()
    _, label = load_series(p, file)
    M = orc._local_adjusted(z['auto20_area'], z['auto20_random'], L)
    winners = [str(z['detectors'][i]) for i in orc.winners_from_local(M)]
    cache = {}

    def score_of(d):
        if d not in cache:
            cache[d] = load_score(p, file, d)
        return cache[d]

    rows = {}
    for name, norm in STITCHED.items():
        try:
            rows[name] = metrics(label, orc.stitch(label, score_of, winners, norm))
        except Exception as e:  # noqa: BLE001
            rows[name] = {'error': str(e)[:120]}
    return file, rows, winners


def main(workers):
    t0 = time.time()
    with open(os.path.join(OUT, 'scores.json')) as f:
        old = json.load(f)
    g = np.load(os.path.join(EXP, 'global_adjusted.npz'))
    adj = {(str(s), str(d)): float(a) for s, d, a in zip(g['series'], g['detector'], g['adjusted'][:, L])}
    files, out = [], {}
    for file in sorted(old['files']):
        path = os.path.join(EXP, 'series', file + '.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path)
        rows = {}
        for j, d in enumerate(map(str, z['detectors'])):
            ap = old['files'][file].get(d, {}).get('auc_pr')
            a = adj.get((file, d))
            if ap is None or a is None or not np.isfinite(z['auto20_adjusted'][j]):
                continue
            rows[d] = {'auc_pr': ap, 'vus_pr': float(z['fast128_vus'][j, L]), 'skill_full': a,
                       'autovus': float(z['auto20_adjusted'][j])}
        if not rows:
            continue
        best = max(rows, key=lambda d: rows[d]['auc_pr'])
        rows['Oracle'] = dict(rows[best], pick=best)
        out[file] = rows
        files.append(file)
    print(f'reused: {len(files)} series, {time.time() - t0:.0f} s', flush=True)

    with Pool(workers) as pool:
        for i, (file, rows, winners) in enumerate(pool.imap_unordered(stitched, files), 1):
            for name, r in rows.items():
                out[file][name] = dict(r, winners=winners)
            if i % 50 == 0 or i == len(files):
                print(f'stitched {i}/{len(files)}, {time.time() - t0:.0f} s', flush=True)

    with open(os.path.join(OUT, 'scores20.json'), 'w') as f:
        json.dump({'L': L, 'variants': {'': ['paper', 'symmetric']}, 'detectors': old['detectors'],
                   'references': ['Oracle', *STITCHED], 'files': out}, f)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    main(ap.parse_args().workers)
