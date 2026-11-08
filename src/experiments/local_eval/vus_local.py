"""Per-anomaly VUS-PR, B style (reference implementation, slow, for toys).

For anomaly k and buffer l (0..L):
  TP_k   soft TP of anomaly k: its points plus its own buffer ramp (the points
         of its region, i.e. nearest to it).
  FP     false positives of the WHOLE series (same as global VUS).
  recall_k = min(TP_k / pos_k, 1) * E_k, pos_k = (TP_k + FN_k + a_k) / 2 (the
         FastVUS/VUS positives formula restricted to k), E_k = 1 when the max
         score in [start_k - l, end_k + l] reaches the threshold.
  precision_k = TP_k / (TP_k + FP).   (C variant: TP_k / (TP_k + FP / K))
local VUS_k = mean over l of the stepwise area. Random baseline per buffer:
  mass_k(l) / (mass_k(l) + T - mass(l)) (own soft positives over own +
  every negative mass), averaged over l. *_adj = (x - random) / (1 - random).
cost[t]: how much of the mean over anomalies of (1 - VUS_k) the false alarm
  at point t causes (exact: sum_t cost[t] = mean_k (1 - VUS_k)).

caps (optional): per-anomaly buffer caps (AutoVUS b*), buffer l becomes
min(l, cap_k) for anomaly k, exactly like FastVUS.compute(cap_left, cap_right).
The same pass also computes the global VUS-PR as a check against FastVUS.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autovus import FastVUS  # noqa: E402


def local_vus(y, s, L=20, caps=None):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    T = len(y)
    eng = FastVUS(L=L, global_mask=False, max_thresholds=None)
    _, (start, end) = eng.anomaly_coordinates(y)            # end inclusive
    K = len(start)
    dl, dr = eng.distance_from_anomaly(y, start, end)
    pc = (None, None) if caps is None else eng.per_point_caps(y, start, end, caps, caps)
    buf = eng.buffered_labels(y, dl, dr, *pc)                # (L+1, T)

    cuts = [0] + [int((end[k] + start[k + 1]) // 2) for k in range(K - 1)] + [T]
    region = np.zeros((K, T), bool)
    for k in range(K):
        region[k, cuts[k]:cuts[k + 1]] = True
    a = np.array([end[k] - start[k] + 1 for k in range(K)], float)

    t = np.unique(s)[::-1]
    M = s[None, :] >= t[:, None]                             # (n_t, T)
    inside = [np.arange(start[k], end[k] + 1) for k in range(K)]
    fn_k = np.array([(~M[:, i]).sum(1) for i in inside])     # (K, n_t)
    capv = np.full(K, L) if caps is None else np.asarray(caps)

    cost = np.zeros(T)    # per-point false-alarm cost, mean over anomalies and buffers
    costC = np.zeros(T)   # same for the C variant
    locC = np.zeros((K, L + 1))
    rndC = np.zeros((K, L + 1))
    loc = np.zeros((K, L + 1))
    rnd = np.zeros((K, L + 1))
    glob = np.zeros(L + 1)
    for l in range(L + 1):
        lab = buf[l]
        reach = np.minimum(l, capv)
        E = np.zeros((K, len(t)))
        for k in range(K):
            lo, hi = max(start[k] - reach[k], 0), min(end[k] + reach[k], T - 1)
            E[k] = s[lo:hi + 1].max() >= t
        tp_k = M @ (lab[None, :] * region).T                 # (n_t, K)
        tp_k = tp_k.T
        TP = tp_k.sum(0)
        FP = M.sum(1) - TP
        # global (check against FastVUS)
        pos = (TP + fn_k.sum(0) + a.sum()) / 2
        R = np.minimum(TP / pos, 1) * E.mean(0)
        P = TP / np.maximum(TP + FP, 1e-12)
        glob[l] = np.sum(np.diff(np.r_[0, R]) * P)
        # local
        pos_k = (tp_k + fn_k + a[:, None]) / 2
        Rk = np.minimum(tp_k / pos_k, 1) * E
        Pk = tp_k / np.maximum(tp_k + FP, 1e-12)
        loc[:, l] = (np.diff(np.c_[np.zeros(K), Rk], axis=1) * Pk).sum(1)
        PkC = tp_k / np.maximum(tp_k + FP / K, 1e-12)
        locC[:, l] = (np.diff(np.c_[np.zeros(K), Rk], axis=1) * PkC).sum(1)
        G = np.diff(np.c_[np.zeros(K), Rk], axis=1) / np.maximum(tp_k + FP, 1e-12)
        cost += (G.mean(0) @ M) * (1 - lab) / (L + 1)
        GC = np.diff(np.c_[np.zeros(K), Rk], axis=1) / np.maximum(tp_k + FP / K, 1e-12) / K
        costC += (GC.mean(0) @ M) * (1 - lab) / (L + 1)
        mass_k = (lab[None, :] * region).sum(1)
        rnd[:, l] = mass_k / (mass_k + T - lab.sum())
        rndC[:, l] = mass_k / (mass_k + (T - lab.sum()) / K)

    v, r = loc.mean(1), rnd.mean(1)
    vC, rC = locC.mean(1), rndC.mean(1)
    return {"C_k": vC, "C_adj": (vC - rC) / (1 - rC),"vus_k": v, "random_k": r, "adj_k": (v - r) / (1 - r),
            "vus_global": glob.mean(), "curve_k": loc, "cost": cost,
            "check_cost": abs(cost.sum() - (1 - v).mean()),
            "costC": costC, "check_costC": abs(costC.sum() - (1 - vC).mean())}
