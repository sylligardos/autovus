"""Real-data check of the new engines and AutoVUS on a few short TSB-AD series.

Compares, per (series, detector): upstream VUS-PR (src/baselines, 250 index
thresholds), FastVUS exact (all thresholds), FastVUS default (2000 thresholds),
FastVUSGPU default, and runs AutoVUS end to end on both engines.
Small by design: 8 short series x 4 detectors. Ask before enlarging.

    conda run -n autovus python src/experiments/check_real.py
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch
from tsad_parser import TSADParser

from baselines import compute as baseline
from vus.fastvus import FastVUS
from vus.fastvus_gpu import FastVUSGPU
from vus.autovus import AutoVUS

L = 128
SERIES = [
    '557_YAHOO_id_7_WebService_tr_500_1st_770.csv',
]
DATASETS = ['UCR', 'WSD', 'SMD', 'NAB', 'Stock', 'TODS', 'MSL']       # one short series each
DETECTORS = ['IForest', 'LOF', 'Sub_PCA', 'AnomalyTransformer']


def pick_series(p):
    m = p.meta.copy()
    m['bytes'] = [os.path.getsize(p.raw_path / f) for f in m.filename]
    names = list(SERIES)
    for d in DATASETS:
        rows = m[(m.dataset == d) & (m.bytes < 400_000)].sort_values('bytes')
        if len(rows):
            names.append(rows.iloc[len(rows) // 2].filename)       # a median-short one, not the tiniest
    return names


def main():
    p = TSADParser()
    names = pick_series(p)
    fnames, series, labels = p.load_timeseries(filename=names)
    scores, _ = p.load_scores_and_delete(fnames, series, labels, detectors=DETECTORS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    exact = FastVUS(L=L, max_thresholds=None)
    fast = FastVUS(L=L)
    gpu = FastVUSGPU(L=L, device=device)
    auto_cpu = AutoVUS(FastVUS(L=L, cumulative=True))
    auto_gpu = AutoVUS(FastVUSGPU(L=L, cumulative=True, device=device))

    rows = []
    print(f"{'series':44s} {'det':19s} {'T':>7s} {'#an':>4s} {'upstream':>9s} {'exact':>9s} {'fast':>9s} {'gpu':>9s} "
          f"{'ex-up':>8s} {'fast-ex':>8s} {'gpu-fast':>9s} {'t_up':>6s} {'t_fast':>7s} {'t_gpu':>6s}")
    for fname, label, det_scores in zip(fnames, labels, scores):
        label = np.asarray(label).astype(int)
        n_anom = int(np.sum(np.diff(np.r_[0, label, 0]) == 1))
        for det, score in zip(DETECTORS, det_scores):
            score = np.asarray(score, dtype=np.float64)
            if len(score) != len(label) or not np.isfinite(score).all():
                print(f"{fname[:44]:44s} {det:19s} skipped (bad score)"); continue
            v_up, t_up = baseline('vus', label, score, L=L)
            t0 = time.perf_counter(); v_ex = exact.compute(label, score); t_ex = time.perf_counter() - t0
            t0 = time.perf_counter(); v_fast = fast.compute(label, score); t_fast = time.perf_counter() - t0
            t0 = time.perf_counter(); v_gpu = gpu.compute(label, score); torch.cuda.synchronize() if device == 'cuda' else None; t_gpu = time.perf_counter() - t0
            rows.append((fname, det, v_up, v_ex, v_fast, v_gpu))
            print(f"{fname[:44]:44s} {det:19s} {len(label):7d} {n_anom:4d} {v_up:9.5f} {v_ex:9.5f} {v_fast:9.5f} {v_gpu:9.5f} "
                  f"{v_ex - v_up:+8.4f} {v_fast - v_ex:+8.1e} {v_gpu - v_fast:+9.1e} {t_up:6.1f} {t_fast * 1000:6.0f}ms {t_gpu * 1000:5.0f}ms")
    a = np.array([r[2:] for r in rows])
    print(f"\n{len(rows)} pairs. exact - upstream: mean {np.mean(a[:, 1] - a[:, 0]):+.4f}, max |.| {np.abs(a[:, 1] - a[:, 0]).max():.4f}; "
          f"fast - exact: max |.| {np.abs(a[:, 2] - a[:, 1]).max():.1e}; gpu - fast: max |.| {np.abs(a[:, 3] - a[:, 2]).max():.1e}")

    print('\nAutoVUS end to end (CPU engine; GPU engine in brackets where it differs):')
    print(f"{'series':44s} {'det':19s} {'vus_full':>8s} {'vus_adapt':>9s} {'skill_adapt':>11s} {'t_cpu':>6s} {'t_gpu':>6s}  per-anomaly (b*_left, b*_right, type)")
    for fname, label, det_scores in zip(fnames, labels, scores):
        label = np.asarray(label).astype(int)
        for det, score in zip(DETECTORS, det_scores):
            score = np.asarray(score, dtype=np.float64)
            if len(score) != len(label) or not np.isfinite(score).all():
                continue
            t0 = time.perf_counter(); r = auto_cpu.compute(label, score); t_c = time.perf_counter() - t0
            t0 = time.perf_counter(); g = auto_gpu.compute(label, score); t_g = time.perf_counter() - t0
            per = []
            for a_c, a_g in zip(r['anomalies'], g['anomalies']):
                c = (a_c['left']['buffer_val'], a_c['right']['buffer_val'], a_c['detection'][0])
                gg = (a_g['left']['buffer_val'], a_g['right']['buffer_val'], a_g['detection'][0])
                per.append(f"{c[0]},{c[1]}{c[2]}" + ('' if c == gg else f"[{gg[0]},{gg[1]}{gg[2]}]"))
            summary = ' '.join(per[:12]) + (' ...' if len(per) > 12 else '')
            diff = '' if abs(r['vus_adapt'] - g['vus_adapt']) < 1e-4 else f" [gpu vus_adapt {g['vus_adapt']:.5f}]"
            print(f"{fname[:44]:44s} {det:19s} {r['vus_full']:8.4f} {r['vus_adapt']:9.4f} {r['skill_adapt']:11.4f} {t_c:5.2f}s {t_g:5.2f}s  {summary}{diff}")


if __name__ == '__main__':
    main()
