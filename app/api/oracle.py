"""The stitched score: region k (midpoint split, FastVUS.regions) takes the
score of anomaly k's winner (best local AutoVUS@20, tie to the detector best
on the most anomalies), each piece min-max scaled to [0, 1], then scored.
Used by the API (/local/oracle).
"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
from autovus import FastVUS, AutoVUS  # noqa: E402

NORMS = {'minmax': 'region: piece scaled to [0, 1]',
         'median': 'region: piece minus its median, clipped at 0, divided by its max'}


def _scaled(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _median_scaled(x):
    y = np.clip(x - float(np.median(x)), 0.0, None)
    hi = float(np.max(y))
    return y / hi if hi > 0 else np.zeros_like(x)


def runs(label):
    d = np.diff(np.concatenate([[0], np.asarray(label, dtype=int), [0]]))
    return np.where(d > 0)[0], np.where(d < 0)[0] - 1


def winners_from_local(M):
    """M (D, K) local AutoVUS; returns the winner index per anomaly (coverage tie rule)."""
    M = np.where(np.isfinite(M), M, 0.0)
    best = M.max(0)
    isbest = M == best[None]
    reach = isbest.sum(1)
    return np.array([max(np.where(isbest[:, k])[0], key=lambda d: reach[d]) for k in range(M.shape[1])])


def stitch(label, score_of, winners, norm='minmax'):
    """score_of(detector) -> full score; winners: one detector name per anomaly."""
    if norm not in NORMS:
        raise ValueError(f'unknown norm {norm}')
    start, end = runs(label)
    if len(winners) != len(start):
        raise ValueError(f'{len(winners)} winners for {len(start)} anomalies')
    T = len(label)
    lo, hi = FastVUS.regions(start, end, T)
    out = np.empty(T)
    for k, d in enumerate(winners):
        piece = np.asarray(score_of(d), dtype=float)[lo[k]:hi[k]]
        out[lo[k]:hi[k]] = _scaled(piece) if norm == 'minmax' else _median_scaled(piece)
    return out


def _local_adjusted(area, random, L):
    v, r = area[..., :L + 1].mean(-1), random[..., :L + 1].mean(-1)
    return (v - r) / np.maximum(1 - r, 1e-10)


def evaluate(label, score):
    """AUC-PR, VUS-PR at every L = 0..128, AutoVUS@20, and per anomaly the
    local AutoVUS@20 (buffers capped at b*)."""
    e = FastVUS(L=128, cumulative=True)
    vusG = np.asarray(e.compute(label, score), dtype=float)
    r = AutoVUS(FastVUS(L=20, cumulative=True, per_anomaly=True)).compute(label, score)
    loc = _local_adjusted(r['local']['area'], r['local']['random'], 20)
    return {'aucpr': float(average_precision_score(label, score)), 'vusG': vusG,
            'autoG': float(r['adjusted']), 'auto': loc}
