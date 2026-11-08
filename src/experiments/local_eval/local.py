"""Per-anomaly (local) scores, AUC-PR / AUC-ROC stage (no buffers yet).

Every anomaly k has a region: halfway to the previous and next anomaly.
Thresholds are always the distinct score values of the WHOLE series.

  naive   AUC-PR of the region alone (the cut).                  blocker
  A       sum_n dR_k(n) * P(n): own recall, precision of the whole series.
          Exact: AUC-PR = sum_k w_k A_k, w_k = |A_k| / |all anomaly points|.
  B       AUC-PR of anomaly k against ALL normal points of the series
          (own recall, own TP, every FP of the series). No coupling.
  C       like B but anomaly k pays a fair share FP / K of the false alarms.
  roc     AUC-ROC of anomaly k against all normal points.
          Exact: AUC-ROC = sum_k w_k roc_k.
  loss_r  sum_n dR(n) * FP_r(n) / (TP(n) + FP(n)): the AUC-PR lost to the
          false alarms of region r. Exact: 1 - AUC-PR = sum_r loss_r.
  *_adj   (x - random) / (1 - random), random = the value a random score gets
          in expectation: A -> prevalence pi, B -> a_k / (a_k + N), roc -> 0.5.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def anomalies(y):
    """List of (start, end_exclusive) of the anomalies of a 0/1 label."""
    d = np.diff(np.r_[0, y, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def regions(A, T):
    """Region of each anomaly: halfway to its neighbours, [lo, hi)."""
    cuts = [0] + [(A[i][1] + A[i + 1][0]) // 2 for i in range(len(A) - 1)] + [T]
    return list(zip(cuts[:-1], cuts[1:]))


def counts(y, s, A, Z):
    """Per distinct threshold (descending): TP per anomaly (K, n), FP per region (K, n)."""
    t = np.unique(s)[::-1]
    def above(idx):  # number of points of idx with score >= t, per t
        v = np.sort(s[idx])
        return len(v) - np.searchsorted(v, t, side="left")
    tp = np.array([above(np.arange(a, b)) for a, b in A])
    idx_all = np.arange(len(y))
    fp = np.array([above(idx_all[lo:hi][y[lo:hi] == 0]) for lo, hi in Z])
    return t, tp, fp


def local_scores(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    A = anomalies(y)
    Z = regions(A, len(y))
    a = np.array([e - b for b, e in A], dtype=float)          # anomaly lengths
    N = int((y == 0).sum())
    _, tp, fp = counts(y, s, A, Z)
    TP, FP = tp.sum(0), fp.sum(0)
    P = TP / np.maximum(TP + FP, 1)                            # precision of the whole series
    Rk = tp / a[:, None]                                       # own recall per anomaly
    dRk = np.diff(np.c_[np.zeros(len(A)), Rk], axis=1)
    R = TP / a.sum()
    dR = np.diff(np.r_[0.0, R])
    w = a / a.sum()

    A_k = (dRk * P).sum(1)
    PB = tp / np.maximum(tp + FP, 1)
    B_k = (dRk * PB).sum(1)
    Kn = len(A)
    PC = tp / np.maximum(tp + FP / Kn, 1e-12)
    C_k = (dRk * PC).sum(1)
    loss = (dR * fp / np.maximum(TP + FP, 1)).sum(1)
    # M[k, r]: the part of 1 - B_k caused by false alarms in region r (exact: 1 - B_k = sum_r M[k, r]).
    M = np.einsum("kn,rn->kr", dRk / np.maximum(tp + FP, 1), fp)

    normals = np.where(y == 0)[0]
    roc_k, naive = [], []
    for (b, e), (lo, hi) in zip(A, Z):
        idx = np.r_[np.arange(b, e), normals]
        roc_k.append(roc_auc_score(y[idx], s[idx]))
        naive.append(average_precision_score(y[lo:hi], s[lo:hi]))
    roc_k = np.array(roc_k)

    pi = a.sum() / len(y)
    rB = a / (a + N)
    out = {
        "A_regions": A, "regions": Z, "w": w,
        "naive": np.array(naive),
        "A": A_k, "A_adj": (A_k - pi) / (1 - pi),
        "B": B_k, "B_adj": (B_k - rB) / (1 - rB),
        "C": C_k, "C_adj": (C_k - a / (a + N / Kn)) / (1 - a / (a + N / Kn)),
        "roc": roc_k, "roc_adj": (roc_k - 0.5) / 0.5,
        "loss": loss, "M": M,
        "ap": average_precision_score(y, s),
        "auc": roc_auc_score(y, s),
    }
    # exactness checks
    out["check_A"] = abs((w * A_k).sum() - out["ap"])
    out["check_roc"] = abs((w * roc_k).sum() - out["auc"])
    out["check_loss"] = abs(loss.sum() - (1 - out["ap"]))
    out["check_M"] = np.abs(M.sum(1) - (1 - B_k)).max()
    return out
