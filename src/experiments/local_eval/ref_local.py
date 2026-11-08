"""Brute-force reference of FastVUS(per_anomaly=True) (variant C, agreed rules).

Every distinct score is a threshold, every (threshold, point) is a dense mask.
Anomaly k's region is halfway to its neighbours; its buffer credit is its own
ramp inside its region; its existence window is clipped to its region.
Only for small series.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autovus import FastVUS  # noqa: E402


def reference(y, s, L=20, caps=None, side="both"):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    T = len(y)
    eng = FastVUS(L=L, global_mask=False, max_thresholds=None, side=side)
    _, (start, end) = eng.anomaly_coordinates(y)
    K = len(start)
    dl, dr = eng.distance_from_anomaly(y, start, end)
    pc = (None, None) if caps is None else eng.per_point_caps(y, start, end, caps[0], caps[1])
    buf = eng.buffered_labels(y, dl, dr, *pc)
    left = eng._ramp(dl, pc[0]) if side in ("both", "left") else np.zeros((L, T))
    right = eng._ramp(dr, pc[1]) if side in ("both", "right") else np.zeros((L, T))

    cuts = [0] + [int((end[k] + start[k + 1]) // 2) for k in range(K - 1)] + [T]
    own = np.zeros((K, L + 1, T))
    for k in range(K):
        lo, hi = cuts[k], cuts[k + 1]
        own[k, 1:, lo:start[k]] = left[:, lo:start[k]]
        own[k, 1:, end[k] + 1:hi] = right[:, end[k] + 1:hi]
        own[k, :, start[k]:end[k] + 1] = 1
    a = (end - start + 1).astype(float)
    capl = np.full(K, L) if caps is None else np.asarray(caps[0])
    capr = np.full(K, L) if caps is None else np.asarray(caps[1])

    t = np.unique(s)[::-1]
    M = (s[None, :] >= t[:, None]).astype(float)
    hit = np.array([M[:, start[k]:end[k] + 1].sum(1) for k in range(K)])
    area = np.zeros((K, L + 1))
    rnd = np.zeros((K, L + 1))
    cost = np.zeros(T)
    for l in range(L + 1):
        lab = buf[l]
        FP = M.sum(1) - M @ lab
        for k in range(K):
            rl = min(l, capl[k]) if side in ("both", "left") else 0
            rr = min(l, capr[k]) if side in ("both", "right") else 0
            lo, hi = max(start[k] - rl, cuts[k]), min(end[k] + rr, cuts[k + 1] - 1)
            E = s[lo:hi + 1].max() >= t
            tp = M @ own[k, l]
            pos = (tp + (a[k] - hit[k]) + a[k]) / 2
            R = np.minimum(tp / pos, 1) * E
            den = tp + FP / K
            P = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
            dR = np.diff(np.r_[0, R])
            area[k, l] = np.sum(dR * P)
            g = np.divide(dR, den, out=np.zeros_like(dR), where=den > 0) / K
            cost += (g @ M) * (1 - lab) / K / (L + 1)
            mass = own[k, l].sum()
            rnd[k, l] = mass / (mass + (T - lab.sum()) / K)
    return {"start": start, "end": end, "area": area, "random": rnd, "cost": cost}
