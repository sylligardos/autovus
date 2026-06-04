"""
Compute the buffer-free VUS variant (vus_v2) on the YAHOO dataset for
every detector and write a single CSV.

Run from the project root, e.g.:
    conda run -n ffvus python src/compute_vus_v2_yahoo.py --smoke 5
    conda run -n ffvus python src/compute_vus_v2_yahoo.py --full
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader import Dataloader
from utils.scoreloader import Scoreloader
from utils.utils import analyze_label


def expand_labels_fast(labels: np.ndarray) -> np.ndarray:
    """
    Buffer-expansion label matrix, vectorized via distance transform.

    Row k contains 1 wherever the distance to the nearest original anomaly
    is <= k. Row 0 is the original labels; the last row is all ones.
    Equivalent to the iterative one-step expansion in vus_v2.expand_labels.
    """
    labels = np.asarray(labels).astype(np.int8)
    n = len(labels)
    if labels.sum() == 0:
        return labels.reshape(1, -1)
    if labels.sum() == n:
        return labels.reshape(1, -1)
    d = distance_transform_edt(1 - labels).astype(np.int32)
    max_d = int(d.max())
    levels = np.arange(max_d + 1, dtype=np.int32)[:, None]
    return (d[None, :] <= levels).astype(np.int8)


def compute_vus_v2_slow(label: np.ndarray, score: np.ndarray):
    """Reference implementation: one sklearn call per buffer level."""
    matrix = expand_labels_fast(label)
    n = matrix.shape[1]
    aps, rocs = [], []
    for row in matrix:
        n_pos = int(row.sum())
        if n_pos == 0 or n_pos == n:
            continue
        aps.append(average_precision_score(row, score))
        rocs.append(roc_auc_score(row, score))
    if not aps:
        return float("nan"), float("nan"), matrix.shape[0]
    return float(np.mean(aps)), float(np.mean(rocs)), matrix.shape[0]


def compute_vus_v2(label: np.ndarray, score: np.ndarray):
    """
    Vectorized over buffer levels.

    Strategy:
      * Sort samples once by descending score (with a small tie-break on
        distance to keep parity with sklearn's argsort behaviour on ties).
      * For every buffer level k, the positives are {i : d_i <= k}. Build
        an indicator matrix (n_buffers, n) over score-sorted positions, do
        a single cumsum to get TP counts at every position × every k,
        derive precision and the AP sum.
      * ROC is computed via Mann-Whitney: U_k = sum of ranks of positives
        minus n_pos*(n_pos+1)/2.

    Returns (vus_pr_uniform, vus_roc_uniform, n_buffers).
    """
    label = np.asarray(label).astype(np.int8)
    score = np.asarray(score, dtype=np.float64)
    n = label.shape[0]
    if label.sum() == 0 or label.sum() == n:
        return float("nan"), float("nan"), 1

    d = distance_transform_edt(1 - label).astype(np.int32)
    max_d = int(d.max())
    n_buffers = max_d + 1   # rows 0..max_d in the (un-built) expanded matrix

    # --- AUC-ROC, vectorized via Mann-Whitney over ranks ---
    R = rankdata(score, method="average")              # 1..n with tie averaging
    order_d = np.argsort(d, kind="stable")             # group samples by distance
    d_sorted = d[order_d]
    R_sorted = R[order_d]
    cumR = np.cumsum(R_sorted)                          # cumulative rank-sum
    ks = np.arange(n_buffers, dtype=np.int32)
    # For each k, n_pos_k = count of samples with d <= k
    n_pos = np.searchsorted(d_sorted, ks, side="right")
    n_neg = n - n_pos
    sumR_pos = np.where(n_pos > 0, cumR[np.clip(n_pos - 1, 0, n - 1)], 0.0)
    U = sumR_pos - n_pos * (n_pos + 1) / 2.0
    valid_roc = (n_pos > 0) & (n_neg > 0)
    roc = np.full(n_buffers, np.nan, dtype=np.float64)
    roc[valid_roc] = U[valid_roc] / (n_pos[valid_roc] * n_neg[valid_roc])

    # --- AP, vectorized over buffer levels, matching sklearn ---
    # sklearn's _binary_clf_curve groups all samples with the same score into a
    # single threshold. AP = Σ_t (R_t − R_{t−1}) · P_t with t indexing unique
    # score values in descending order.
    order_s = np.argsort(score, kind="mergesort")[::-1]
    d_perm = d[order_s]
    score_desc = score[order_s]
    ind = (d_perm[None, :] <= ks[:, None]).astype(np.float64)   # (n_buffers, n)
    tp_cum = np.cumsum(ind, axis=1)                              # TP at each position × k
    # Tie-group ends (positions in descending-sorted order where score changes,
    # plus the very last position).
    diff_idx = np.flatnonzero(np.diff(score_desc) != 0)
    group_ends = np.concatenate([diff_idx, [n - 1]])            # (n_groups,)
    j_size = (group_ends + 1).astype(np.float64)                # samples with score ≥ this threshold
    tp_at = tp_cum[:, group_ends]                                # (n_buffers, n_groups) TP at each threshold
    zeros_col = np.zeros((n_buffers, 1), dtype=np.float64)
    tp_diff = np.diff(np.concatenate([zeros_col, tp_at], axis=1), axis=1)  # ΔTP per group
    ap_sum = (tp_diff * tp_at / j_size[None, :]).sum(axis=1)
    valid_ap = n_pos > 0
    ap = np.full(n_buffers, np.nan, dtype=np.float64)
    ap[valid_ap] = ap_sum[valid_ap] / n_pos[valid_ap]

    # Drop degenerate rows (matches the slow version: all-zeros / all-ones rows skipped).
    keep = (n_pos > 0) & (n_pos < n)
    if not keep.any():
        return float("nan"), float("nan"), n_buffers
    return float(np.mean(ap[keep])), float(np.mean(roc[keep])), n_buffers


def sanity_check():
    """
    Two checks:
      1. expand_labels_fast matches the iterative vus_v2.expand_labels.
      2. compute_vus_v2 (fast) matches compute_vus_v2_slow within tolerance
         on random labels+scores.
    """
    from vus.vus_v2 import expand_labels as expand_labels_slow

    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(10, 40)
        labels = np.zeros(n, dtype=int)
        n_anom = rng.integers(1, 4)
        idx = rng.choice(n, size=n_anom, replace=False)
        labels[idx] = 1
        slow = expand_labels_slow(labels)
        fast = expand_labels_fast(labels)
        assert slow.shape == fast.shape, (slow.shape, fast.shape)
        assert np.array_equal(slow, fast), labels
    print("[sanity] expand_labels_fast matches vus_v2.expand_labels on 20 random cases")

    def _agreement(label_factory, score_factory, n_trials, tag):
        max_dap = 0.0
        max_droc = 0.0
        for _ in range(n_trials):
            labels = label_factory()
            score = score_factory(labels)
            ap_s, roc_s, _ = compute_vus_v2_slow(labels, score)
            ap_f, roc_f, _ = compute_vus_v2(labels, score)
            max_dap = max(max_dap, abs(ap_s - ap_f))
            max_droc = max(max_droc, abs(roc_s - roc_f))
        print(f"[sanity] fast vs slow ({tag}): max |ΔAP|={max_dap:.2e}  max |ΔROC|={max_droc:.2e}")
        return max_dap, max_droc

    # No ties: continuous random scores.
    def mk_lab():
        n = rng.integers(50, 200)
        lab = np.zeros(n, dtype=int)
        idx = rng.choice(n, size=rng.integers(2, 6), replace=False)
        lab[idx] = 1
        return lab
    dap1, droc1 = _agreement(mk_lab, lambda lab: rng.random(len(lab)), 20, "continuous")

    # Heavy ties: integer scores from a small alphabet.
    dap2, droc2 = _agreement(mk_lab, lambda lab: rng.integers(0, 5, size=len(lab)).astype(float), 20, "tied   ")

    assert max(dap1, dap2) < 1e-9 and max(droc1, droc2) < 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=5,
                        help="Number of YAHOO time series to use (smoke test).")
    parser.add_argument("--full", action="store_true",
                        help="Run on the whole YAHOO dataset.")
    parser.add_argument("--out", type=str,
                        default="experiments/vus_v2_yahoo/results.csv",
                        help="Path to the output CSV.")
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--scores", type=str, default="data/scores")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    if not args.skip_sanity:
        sanity_check()

    dl = Dataloader(raw_data_path=args.raw)
    _, labels, filenames = dl.load_raw_datasets(["YAHOO"])

    if not args.full:
        labels = labels[: args.smoke]
        filenames = filenames[: args.smoke]
        print(f"[smoke] using {len(filenames)} YAHOO time series")
    else:
        print(f"[full] using {len(filenames)} YAHOO time series")

    sl = Scoreloader(args.scores)
    detectors = sl.get_detector_names()
    print(f"[info] detectors ({len(detectors)}): {detectors}")

    scores_per_ts, idx_failed = sl.load_parallel(filenames, detectors=detectors)
    labels = sl.clean_failed_idx(labels, idx_failed)
    filenames = sl.clean_failed_idx(filenames, idx_failed)
    assert len(scores_per_ts) == len(labels) == len(filenames)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    rows = []
    for fname, label, score_mat in tqdm(
        list(zip(filenames, labels, scores_per_ts)),
        desc="YAHOO time series",
    ):
        length, n_anom, anom_avg = analyze_label(label)
        for det_idx, detector in enumerate(detectors):
            score = score_mat[:, det_idx]
            if label.shape[0] != score.shape[0]:
                m = min(label.shape[0], score.shape[0])
                lab = label[:m]
                scr = score[:m]
            else:
                lab = label
                scr = score
            t0 = time.perf_counter()
            ap_mean, roc_mean, n_buf = compute_vus_v2(lab, scr)
            dt = time.perf_counter() - t0
            rows.append({
                "filename": fname,
                "detector": detector,
                "length": int(length),
                "n_anomalies": int(n_anom),
                "anom_avg_length": float(anom_avg),
                "n_buffers": int(n_buf),
                "vus_pr_uniform": ap_mean,
                "vus_roc_uniform": roc_mean,
                "time_seconds": dt,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[done] wrote {len(df)} rows to {args.out}")
    print("\nPer-detector means:")
    print(df.groupby("detector")[["vus_pr_uniform", "vus_roc_uniform",
                                   "time_seconds", "n_buffers"]].mean()
          .round(4).to_string())


if __name__ == "__main__":
    main()
