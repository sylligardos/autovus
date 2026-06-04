"""
Compute the inverted-buffer VUS variant on the YAHOO dataset for every
detector and write a single CSV.

Inverted variant: instead of widening the LABEL by k (vus_v2), we widen
the PREDICTION by k. Equivalently, replace the score by its max over a
window of radius k, then compute standard AP/ROC against the ORIGINAL
label. A sharp prediction that lands within k of any anomaly gets full
credit; POLY-style wide predictions get no recall bonus from widening
and are penalized through precision instead.

Run from the project root, e.g.:
    conda run -n ffvus python src/compute_vus_inv.py --smoke 5
    conda run -n ffvus python src/compute_vus_inv.py --full
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, maximum_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader import Dataloader
from utils.scoreloader import Scoreloader
from utils.utils import analyze_label


def compute_vus_inv(label: np.ndarray, score: np.ndarray):
    """
    Inverted VUS: max-filter the score by radius k, compute AP/ROC vs
    the ORIGINAL label, average across k = 0 .. max_d.

    Buffer range matches compute_vus_v2: max_d is the largest distance
    from any point to the nearest anomaly, so the final iterations cover
    the entire signal with the global max of the score.

    Returns (vus_pr_inv, vus_roc_inv, n_buffers).
    """
    label = np.asarray(label).astype(np.int8)
    score = np.asarray(score, dtype=np.float64)
    n = label.shape[0]
    if label.sum() == 0 or label.sum() == n:
        return float("nan"), float("nan"), 1

    d = distance_transform_edt(1 - label).astype(np.int32)
    max_d = int(d.max())
    n_buffers = max_d + 1

    aps = np.empty(n_buffers, dtype=np.float64)
    rocs = np.empty(n_buffers, dtype=np.float64)
    for k in range(n_buffers):
        if k == 0:
            score_k = score
        else:
            score_k = maximum_filter1d(score, size=2 * k + 1, mode="nearest")
        aps[k] = average_precision_score(label, score_k)
        rocs[k] = roc_auc_score(label, score_k)

    return float(aps.mean()), float(rocs.mean()), n_buffers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=5,
                        help="Number of YAHOO time series to use (smoke test).")
    parser.add_argument("--full", action="store_true",
                        help="Run on the whole YAHOO dataset.")
    parser.add_argument("--out", type=str,
                        default="experiments/vus_inv_yahoo/results.csv",
                        help="Path to the output CSV.")
    parser.add_argument("--raw", type=str, default="data/raw")
    parser.add_argument("--scores", type=str, default="data/scores")
    args = parser.parse_args()

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
            ap_mean, roc_mean, n_buf = compute_vus_inv(lab, scr)
            dt = time.perf_counter() - t0
            rows.append({
                "filename": fname,
                "detector": detector,
                "length": int(length),
                "n_anomalies": int(n_anom),
                "anom_avg_length": float(anom_avg),
                "n_buffers": int(n_buf),
                "vus_pr_inv": ap_mean,
                "vus_roc_inv": roc_mean,
                "time_seconds": dt,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[done] wrote {len(df)} rows to {args.out}")
    print("\nPer-detector means:")
    print(df.groupby("detector")[["vus_pr_inv", "vus_roc_inv",
                                   "time_seconds", "n_buffers"]].mean()
          .round(4).to_string())


if __name__ == "__main__":
    main()
