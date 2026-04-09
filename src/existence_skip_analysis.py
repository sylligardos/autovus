"""
Analyze how many buffer iterations the optimized existence algorithm skips
compared to the naive approach (which would visit every buffer for every anomaly).
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root for data paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from vus.vus_numpy import VUSNumpy
from utils.utils import load_tsb


def existence_optimized_instrumented(vus, labels, score, thresholds, start_points, end_points):
    """
    Same as VUSNumpy.existence_optimized but counts:
    - total_possible: n_slopes * n_anomalies (naive would visit each buffer per anomaly)
    - actual_iters: number of while-loop iterations actually performed
    - middle_hits: how many times the 'case middle' fires (fills all remaining buffers at once)
    - left_hits / right_hits: how many times each buffer case fires
    """
    n_slopes, T = labels.shape
    n_thresholds = thresholds.shape[0]
    n_anomalies_base = start_points.shape[0]

    # Stats
    actual_iters = 0
    middle_hits = 0
    left_hits = 0
    right_hits = 0
    buffers_filled_at_once = []  # how many buffers each iteration fills

    existence = np.zeros((n_slopes, n_thresholds))
    n_anomalies = np.full(n_slopes, n_anomalies_base, dtype=int)
    thresholds_inv = thresholds[::-1]

    all_start_points = np.maximum((np.tile(start_points, (n_slopes, 1)) - vus.slope_values[:, None]), 0)
    all_end_points = np.minimum((np.tile(end_points, (n_slopes, 1)) + vus.slope_values[:, None]), T - 1)

    anomalies_overlaps = np.zeros(start_points.shape, dtype=int)
    for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
        distance = start - end
        overlap = (2 * vus.slope_size) - distance
        if overlap >= 0:
            index = math.ceil((vus.slope_size - (overlap // 2)) / vus.step)
            n_anomalies[index:] -= 1
            anomalies_overlaps[k] = index

    prev_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
    for k in range(n_anomalies[0]):
        overlap_index = anomalies_overlaps[k - 1]

        og_start = start_points[k]
        og_end = end_points[k]

        curr_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
        tmp_existence = np.zeros((n_slopes, n_thresholds), dtype=int)

        i = n_slopes - 1
        while i >= 0:
            actual_iters += 1
            start = all_start_points[i, k]
            end = all_end_points[i, k]

            argmax_t = [score[start:og_start + 1][::-1].argmax(), score[og_start:og_end + 1].argmax(), score[og_end:end + 1].argmax()]
            max_t = [score[og_start - argmax_t[0]], score[og_start + argmax_t[1]], score[og_end + argmax_t[2]]]

            if max_t[1] >= max_t[0] and max_t[1] >= max_t[2]:   # Case middle
                threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[1], side="right")
                curr_existence[:i+1, threshold_index:] += 1
                middle_hits += 1
                buffers_filled_at_once.append(i + 1)  # fills buffers 0..i
                break
            elif max_t[0] > max_t[2]:    # Case left
                threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[0], side="right")
                new_i = math.ceil(argmax_t[0] / vus.step) - 1
                filled = i + 1 - math.ceil(argmax_t[0] / vus.step)
                curr_existence[math.ceil(argmax_t[0] / vus.step):i+1, threshold_index:] += 1
                left_hits += 1
                buffers_filled_at_once.append(filled)
                i = new_i
            elif max_t[2] >= max_t[0]:   # Case right
                threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[2], side="right")
                new_i = math.ceil(argmax_t[2] / vus.step) - 1
                filled = i + 1 - math.ceil(argmax_t[2] / vus.step)
                curr_existence[math.ceil(argmax_t[2] / vus.step):i+1, threshold_index:] += 1
                right_hits += 1
                buffers_filled_at_once.append(filled)
                i = new_i

        if overlap_index > 0:
            tmp_existence[overlap_index:] = np.logical_or(curr_existence[overlap_index:], prev_existence[overlap_index:])
            curr_existence[overlap_index:] = np.logical_and(curr_existence[overlap_index:], np.logical_not(prev_existence[overlap_index:]))
        existence += curr_existence
        prev_existence = np.logical_or(tmp_existence, curr_existence)

    total_possible = n_slopes * n_anomalies_base
    skipped = total_possible - actual_iters

    return {
        'n_slopes': n_slopes,
        'n_anomalies': n_anomalies_base,
        'n_thresholds': n_thresholds,
        'T': T,
        'total_possible': total_possible,
        'actual_iters': actual_iters,
        'skipped': skipped,
        'skip_ratio': skipped / total_possible if total_possible > 0 else 0.0,
        'middle_hits': middle_hits,
        'left_hits': left_hits,
        'right_hits': right_hits,
        'mean_buffers_filled_per_iter': np.mean(buffers_filled_at_once) if buffers_filled_at_once else 0.0,
        'median_buffers_filled_per_iter': np.median(buffers_filled_at_once) if buffers_filled_at_once else 0.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run on small subset')
    parser.add_argument('--slope_size', type=int, default=128, help='Slope size to analyze (default: 128)')
    args = parser.parse_args()

    print("Loading TSB-UAD benchmark...")
    if args.test:
        filenames, labels, scores, detectors = load_tsb(testing=True, dataset='KDD21', n_timeseries=10)
    else:
        filenames, labels, scores, detectors = load_tsb(testing=False)

    vus = VUSNumpy(
        slope_size=args.slope_size,
        step=1,
        global_mask=True,
        slopes='precomputed',
        existence='optimized',
        conf_matrix='dynamic_plus',
    )

    results = []
    skipped_series = 0
    for idx, (fname, label, score) in enumerate(tqdm(zip(filenames, labels, scores), total=len(labels), desc="Analyzing existence skips")):
        # Guard: skip if label/score length mismatch
        min_len = min(len(label), len(score))
        label, score = label[:min_len], score[:min_len]

        # Replicate the compute pipeline up to existence
        ((_, _), (start_with_edges, end_with_edges)), _ = vus.get_anomalies_coordinates(label)
        safe_mask, _ = vus.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=vus.global_mask)
        thresholds, _ = vus.get_unique_thresholds(score)

        # Apply global mask (same as in compute)
        label_masked, score_masked = label[safe_mask], score[safe_mask]
        if len(label_masked) == 0 or label_masked.sum() == 0:
            skipped_series += 1
            continue
        ((start_no_edges, end_no_edges), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label_masked)
        if start_we.shape[0] == 0:
            skipped_series += 1
            continue
        labels_slopes, _ = vus.add_slopes(label_masked, start_no_edges, end_no_edges, None)

        # Run instrumented existence
        stats = existence_optimized_instrumented(vus, labels_slopes, score_masked, thresholds, start_we, end_we)
        stats['filename'] = fname
        stats['detector'] = detectors[idx]
        stats['T_original'] = len(label)
        results.append(stats)

    df = pd.DataFrame(results)

    # Save
    out_dir = 'experiments/existence_skip_analysis'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'skip_stats.csv')
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXISTENCE SKIP ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total time series analyzed: {len(df)} (skipped {skipped_series} with no anomalies/data issues)")
    print(f"Total buffer iterations (naive):     {df['total_possible'].sum():>12,}")
    print(f"Total buffer iterations (optimized): {df['actual_iters'].sum():>12,}")
    print(f"Total iterations skipped:            {df['skipped'].sum():>12,}")
    print(f"Overall skip ratio:                  {df['skipped'].sum() / df['total_possible'].sum():.4f} ({df['skipped'].sum() / df['total_possible'].sum() * 100:.2f}%)")
    print()
    print(f"Per time series (mean ± std):")
    print(f"  Skip ratio:                {df['skip_ratio'].mean():.4f} ± {df['skip_ratio'].std():.4f} (i.e., {df['skip_ratio'].mean()*100:.2f}% of buffers skipped on average)")
    print(f"  Buffers filled per iter:   {df['mean_buffers_filled_per_iter'].mean():.2f} ± {df['mean_buffers_filled_per_iter'].std():.2f}")
    print(f"  n_slopes (buffers):        {df['n_slopes'].mean():.1f}")
    print(f"  n_anomalies:               {df['n_anomalies'].mean():.1f} ± {df['n_anomalies'].std():.1f}")
    print()
    print(f"Case breakdown (totals across all series):")
    total_hits = df['middle_hits'].sum() + df['left_hits'].sum() + df['right_hits'].sum()
    print(f"  Middle hits (full skip):   {df['middle_hits'].sum():>10,} ({df['middle_hits'].sum()/total_hits*100:.1f}%)")
    print(f"  Left buffer hits:          {df['left_hits'].sum():>10,} ({df['left_hits'].sum()/total_hits*100:.1f}%)")
    print(f"  Right buffer hits:         {df['right_hits'].sum():>10,} ({df['right_hits'].sum()/total_hits*100:.1f}%)")
    print()
    print(f"Skip ratio distribution:")
    print(f"  Min:    {df['skip_ratio'].min():.4f}")
    print(f"  25th:   {df['skip_ratio'].quantile(0.25):.4f}")
    print(f"  Median: {df['skip_ratio'].median():.4f}")
    print(f"  75th:   {df['skip_ratio'].quantile(0.75):.4f}")
    print(f"  Max:    {df['skip_ratio'].max():.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
