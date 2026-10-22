"""
Precompute the tables the demo reads (app/data/<benchmark>/).

Phase 1, every series x every detector plus the two reference detectors:
    AUC-PR, AUC-ROC (sklearn), VUS-PR, VUS-ROC (FastVUS, L), AutoVUS (per side,
    paper rule), per-anomaly buffers and types, FastVUS and AutoVUS times.
Phase 2, timings of the slow baselines on a subset of short series:
    upstream VUS, Range-F1, Affiliation-F1, PATE.
Phase 3, FastVUS GPU time on every series, one detector.

Writes:
    summary.json   counts
    series.json    file -> {dataset, n, anomalies}
    scores.json    {L, rule, sides, detectors, files: {file: {detector: {...}}}}
    anomalies.csv  file, detector, k, start, end, b_left, b_right, type
    times.json     the timings of phases 1 to 3

    conda run -n autovus python app/export.py --L 256 --workers 6
    conda run -n autovus python app/export.py --phase 2 --subset 2
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")             # one thread per worker, no oversubscription

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from tsad_parser import TSADParser  # noqa: E402
from autovus import FastVUS, AutoVUS  # noqa: E402
sys.path.insert(0, os.path.join(_ROOT, 'app', 'api'))
from bench import load_series, load_score  # noqa: E402

ORACLE_DELAY = 64
REFERENCES = ('Delayed oracle', 'Random')
OUT = os.path.join(_ROOT, 'app', 'data', 'tsb-ad')


def reference_score(name, label, seed=0):
    label = np.asarray(label).astype(float)
    if name == 'Delayed oracle':
        out = np.zeros_like(label)
        out[ORACLE_DELAY:] = label[:len(label) - ORACLE_DELAY]
        return out
    return np.random.default_rng(seed).random(len(label))


def runs(label):
    d = np.diff(np.concatenate([[0], label, [0]]))
    return list(zip(np.where(d > 0)[0], np.where(d < 0)[0] - 1))


def _single_thread():
    """One thread per worker: BLAS, torch and numexpr all try to use every core."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # noqa: BLE001
        pass


def one_series(args):
    """Phase 1 for one series: every detector and the references."""
    file, L, rule, sides, detectors = args
    p = TSADParser()
    _, label = load_series(p, file)
    T = len(label)
    rows, anoms = {}, []
    if label.sum() == 0:
        return file, T, 0, rows, anoms
    scores = {}
    for det in detectors:
        s = load_score(p, file, det)
        if s is not None:
            scores[det] = s
    for det in REFERENCES:
        scores[det] = reference_score(det, label)
    eng = FastVUS(L=L, cumulative=True)
    auto = AutoVUS(eng, rule=rule, sides=sides)
    for det, score in scores.items():
        n = min(T, len(score))
        lab, sco = label[:n], score[:n]
        if np.any(np.isnan(sco)) or lab.sum() == 0:
            continue
        row = {}
        try:
            t0 = time.perf_counter()
            vus = np.asarray(eng.compute(lab, sco))
            row['t_vus'] = time.perf_counter() - t0
            row['vus_pr'] = float(vus[-1])
            roc = np.asarray(eng.compute(lab, sco, metric='roc'))
            row['vus_roc'] = float(roc[-1])
            row['auc_pr'] = float(average_precision_score(lab, sco))
            row['auc_roc'] = float(roc_auc_score(lab, sco)) if lab.sum() < n else None
            t0 = time.perf_counter()
            r = auto.compute(lab, sco)
            row['t_autovus'] = time.perf_counter() - t0
            row['autovus'] = float(r['skill_adapt'])
            row['vus_adapt'] = float(r['vus_adapt'])
            row['skill_full'] = float(r['skill_full'])
            row['random_full'] = float(r['random_full'])
            types = [a['detection'] for a in r['anomalies']]
            row['n_anomalies'] = len(types)
            for t in ('exact', 'early', 'late', 'missed', 'both'):
                row[f'n_{t}'] = types.count(t)
            row['b_left'] = float(np.mean([a['left']['buffer_val'] for a in r['anomalies']]))
            row['b_right'] = float(np.mean([a['right']['buffer_val'] for a in r['anomalies']]))
            for k, a in enumerate(r['anomalies']):
                anoms.append((file, det, k, a['start'], a['end'], a['left']['buffer_val'], a['right']['buffer_val'], a['detection']))
        except Exception as e:  # noqa: BLE001
            row['error'] = str(e)[:120]
        rows[det] = row
    return file, T, len(runs(label)), rows, anoms


def phase1(L, rule, sides, workers, limit):
    p = TSADParser()
    meta = p.meta
    detectors = list(p.get_detectors())
    files = meta['filename'].tolist()
    if limit:
        files = files[:limit]
    files.sort(key=lambda f: os.path.getsize(p.raw_path / f))     # short series first: the tables fill up early
    os.makedirs(OUT, exist_ok=True)
    scores, series, anom_rows = {}, {}, []
    if os.path.exists(os.path.join(OUT, 'scores.json')):                 # resume: keep what an earlier run saved
        with open(os.path.join(OUT, 'scores.json')) as f:
            scores = json.load(f)['files']
        with open(os.path.join(OUT, 'series.json')) as f:
            series = json.load(f)
        with open(os.path.join(OUT, 'anomalies.csv'), newline='') as f:
            anom_rows = [tuple(r) for r in list(csv.reader(f))[1:]]
        files = [f for f in files if f not in scores]
        print(f'resuming: {len(series)} series done, {len(files)} to go', flush=True)
    t0 = time.time()

    def save(done):
        with open(os.path.join(OUT, 'series.json'), 'w') as f:
            json.dump(series, f)
        with open(os.path.join(OUT, 'scores.json'), 'w') as f:
            json.dump({'L': L, 'rule': rule, 'sides': sides, 'detectors': detectors, 'references': list(REFERENCES), 'files': scores}, f)
        with open(os.path.join(OUT, 'anomalies.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'detector', 'k', 'start', 'end', 'b_left', 'b_right', 'type'])
            w.writerows(anom_rows)
        with open(os.path.join(OUT, 'summary.json'), 'w') as f:
            json.dump({'n_series': len(series), 'n_series_total': len(files), 'done': done,
                       'n_datasets': int(meta['dataset'].nunique()), 'n_detectors': len(detectors),
                       'n_anomalies': sum(s['anomalies'] for s in series.values()), 'n_points': int(sum(s['n'] for s in series.values())),
                       'L': L, 'rule': rule, 'sides': sides, 'oracle_delay': ORACLE_DELAY,
                       'phase1_seconds': time.time() - t0, 'workers': workers}, f, indent=1)

    jobs = [(f, L, rule, sides, detectors) for f in files]
    with Pool(workers, initializer=_single_thread) as pool:
        for i, (file, T, n_anom, rows, anoms) in enumerate(pool.imap_unordered(one_series, jobs), 1):
            dataset = meta.loc[meta['filename'] == file, 'dataset'].iloc[0]
            series[file] = {'dataset': dataset, 'n': int(T), 'anomalies': int(n_anom)}
            scores[file] = rows
            anom_rows.extend(anoms)
            if i % 10 == 0 or i == len(files):
                print(f'{i}/{len(files)} series, {time.time() - t0:.0f} s, last {file} ({T} points, {n_anom} anomalies)', flush=True)
                save(i == len(files))
    save(True)
    print(f'phase 1 done in {time.time() - t0:.0f} s', flush=True)


def phase2(L, per_dataset, max_T):
    """Slow baselines on a subset: up to per_dataset series per dataset, at most max_T points."""
    from baselines import compute as baseline
    p = TSADParser()
    meta = p.meta
    with open(os.path.join(OUT, 'series.json')) as f:
        series = json.load(f)
    chosen = []
    for ds, rows in meta.groupby('dataset'):
        files = [f for f in rows['filename'] if f in series and series[f]['n'] <= max_T and series[f]['anomalies'] > 0]
        files.sort(key=lambda f: series[f]['n'])
        chosen += files[:per_dataset]
    detectors = list(p.get_detectors())
    out = []
    t_all = time.time()
    for i, file in enumerate(chosen, 1):
        _, label = load_series(p, file)
        for det in detectors:
            score = load_score(p, file, det)
            if score is None:
                continue
            n = min(len(label), len(score))
            lab, sco = label[:n], score[:n]
            row = {'file': file, 'detector': det, 'n': int(n)}
            for m in ('vus', 'rf', 'affiliation'):
                try:
                    _, sec = baseline(m, lab, sco, L=L)
                    row[f't_{m}'] = sec
                except Exception as e:  # noqa: BLE001
                    row[f't_{m}'] = None
                    row[f'err_{m}'] = str(e)[:80]
            if n <= 10_000 and det in detectors[:4]:
                try:
                    _, sec = baseline('pate', lab, sco, L=L)
                    row['t_pate'] = sec
                except Exception as e:  # noqa: BLE001
                    row['t_pate'] = None
                    row['err_pate'] = str(e)[:80]
            out.append(row)
        print(f'phase 2: {i}/{len(chosen)} {file} ({n} points), {time.time() - t_all:.0f} s', flush=True)
        with open(os.path.join(OUT, 'times_baselines.json'), 'w') as f:
            json.dump({'L': L, 'per_dataset': per_dataset, 'max_T': max_T, 'rows': out}, f)
    print(f'phase 2 done in {time.time() - t_all:.0f} s', flush=True)


def phase3(L):
    """FastVUS GPU time on every series, the first detector, after one warm-up."""
    from autovus import FastVUSGPU
    p = TSADParser()
    det = p.get_detectors()[0]
    with open(os.path.join(OUT, 'series.json')) as f:
        series = json.load(f)
    eng = FastVUSGPU(L=L)
    out = {}
    t_all = time.time()
    for i, file in enumerate(sorted(series, key=lambda f: series[f]['n']), 1):
        if series[file]['anomalies'] == 0:
            continue
        _, label = load_series(p, file)
        score = load_score(p, file, det)
        if score is None:
            continue
        n = min(len(label), len(score))
        try:
            eng.compute(label[:n], score[:n])
            t0 = time.perf_counter()
            eng.compute(label[:n], score[:n])
            out[file] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            out[file] = None
            print('gpu failed on', file, str(e)[:80], flush=True)
        if i % 100 == 0:
            print(f'phase 3: {i}/{len(series)}, {time.time() - t_all:.0f} s', flush=True)
    with open(os.path.join(OUT, 'times_gpu.json'), 'w') as f:
        json.dump({'L': L, 'detector': det, 'seconds': out}, f)
    print(f'phase 3 done in {time.time() - t_all:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=256)
    ap.add_argument('--rule', default='paper')
    ap.add_argument('--sides', default='per_side')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--phase', type=int, default=0, help='0 = all, else one phase')
    ap.add_argument('--limit', type=int, default=0, help='phase 1: only the first N series (smoke test)')
    ap.add_argument('--subset', type=int, default=2, help='phase 2: series per dataset')
    ap.add_argument('--max-T', type=int, default=60_000, help='phase 2: longest series timed')
    a = ap.parse_args()
    if a.phase in (0, 1):
        phase1(a.L, a.rule, a.sides, a.workers, a.limit)
    if a.phase in (0, 2):
        phase2(a.L, a.subset, a.max_T)
    if a.phase in (0, 3):
        phase3(a.L)
