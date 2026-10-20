"""Brute-force oracle check of the engines on long real series (D1: MITDB).

The oracle reuses the engines' buffered labels and threshold set (both
validated against upstream) and recomputes everything else from the
definitions: dense true/false positives over the whole series (no global
mask, no masked matmul), existence from the merged buffered intervals of
the anomalies (no skipping), stepwise PR areas. FastVUS must match it to
float precision; FastVUSGPU within its existence approximation.
Cost: a few minutes per pair. Ask before enlarging.

    conda run -n autovus python src/experiments/check_oracle.py
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch
from tsad_parser import TSADParser

from autovus import FastVUS, FastVUSGPU

L = 128
SERIES = ['170_MITDB_id_1_Medical_tr_17675_1st_17775.csv', '171_MITDB_id_2_Medical_tr_50000_1st_88864.csv']
DETECTORS = ['IForest', 'Sub_PCA']


def oracle(label, score, thresholds, buffered, start, end, buffers):
    """Mean stepwise AUC-PR over buffers, from the definitions."""
    T = len(label)
    inside = label.astype(bool)
    n_inside = inside.sum()
    n_b, n_t = len(buffers), len(thresholds)
    tp = np.zeros((n_b, n_t)); n_pred = np.zeros(n_t); fn = np.zeros(n_t)
    for lo in range(0, n_t, 200):                                   # dense, in threshold chunks
        sm = (score[None, :] >= thresholds[lo:lo + 200, None]).astype(np.float64)
        tp[:, lo:lo + 200] = buffered @ sm.T
        n_pred[lo:lo + 200] = sm.sum(axis=1)
        fn[lo:lo + 200] = (1.0 - sm[:, inside]).sum(axis=1)
    fp = n_pred[None, :] - tp
    positives = (tp + fn[None, :] + n_inside) / 2
    # existence: merged buffered intervals, one max per component
    existence = np.zeros((n_b, n_t))
    for i, b in enumerate(buffers):
        lo_, hi_ = np.maximum(start - b, 0), np.minimum(end + b, T - 1)
        comp_max = []
        cur_lo, cur_hi = lo_[0], hi_[0]
        for s, e in zip(lo_[1:], hi_[1:]):
            if s <= cur_hi:                                         # overlap or touch: one component
                cur_hi = max(cur_hi, e)
            else:
                comp_max.append(score[cur_lo:cur_hi + 1].max()); cur_lo, cur_hi = s, e
        comp_max.append(score[cur_lo:cur_hi + 1].max())
        comp_max = np.sort(comp_max)
        existence[i] = (len(comp_max) - np.searchsorted(comp_max, thresholds, side='left')) / len(comp_max)
    precision = np.hstack((np.ones((n_b, 1)), tp / (tp + fp)))
    recall = np.hstack((np.zeros((n_b, 1)), np.minimum(tp / positives, 1) * existence))
    return np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1).mean()


def main():
    p = TSADParser()
    fnames, series, labels = p.load_timeseries(filename=SERIES)
    scores, _ = p.load_scores_and_delete(fnames, series, labels, detectors=DETECTORS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cpu = FastVUS(L=L)                                   # default: global mask on, 2000 thresholds
    plain = FastVUS(L=L, global_mask=False)              # gives the oracle its labels and thresholds
    gpu = FastVUSGPU(L=L, device=device)
    ok = True
    for fname, label, det_scores in zip(fnames, labels, scores):
        label = np.asarray(label).astype(int)
        _, (start, end) = plain.anomaly_coordinates(label)
        print(f"{fname}  T={len(label):,}  anomalies={len(start)}")
        for det, score in zip(DETECTORS, det_scores):
            score = np.asarray(score, dtype=np.float64)
            t0 = time.perf_counter(); v_cpu = cpu.compute(label, score); t_cpu = time.perf_counter() - t0
            t0 = time.perf_counter(); v_gpu = gpu.compute(label, score); t_gpu = time.perf_counter() - t0
            dl, dr = plain.distance_from_anomaly(label, start, end)
            buffered = plain.buffered_labels(label, dl, dr)
            thresholds = plain.thresholds(buffered, score)
            t0 = time.perf_counter(); v_or = oracle(label, score, thresholds, buffered, start, end, plain.buffers); t_or = time.perf_counter() - t0
            flag = '' if abs(v_cpu - v_or) < 1e-9 else '  <-- CPU differs from the oracle'
            ok &= abs(v_cpu - v_or) < 1e-9
            print(f"  {det:9s} thresholds={len(thresholds):5d}  oracle {v_or:.8f} ({t_or:5.1f}s)  FastVUS {v_cpu:.8f} ({t_cpu * 1000:5.0f}ms, diff {v_cpu - v_or:+.1e})  "
                  f"FastVUSGPU {v_gpu:.8f} ({t_gpu * 1000:4.0f}ms, diff {v_gpu - v_or:+.1e}){flag}")
    print('ALL OK' if ok else 'FAILURES')


if __name__ == '__main__':
    main()
