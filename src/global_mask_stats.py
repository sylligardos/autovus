"""
Measure how much of each time series the global mask removes.

The global mask keeps only the anomalies and their buffers (the region returned by
create_safe_mask with extra_safe=True) and discards the rest of the label before the
expensive VUS computation. This script quantifies, across the full TSB-AD benchmark,
the fraction of each series removed and the resulting label shrink factor (T / kept).

Companion to src/existence_skip_analysis.py. Run with the ffvus conda env:
    conda run -n ffvus python src/global_mask_stats.py [--slope_size 128]
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root for data paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slope_size', type=int, default=128, help='Buffer (slope) size to analyze (default: 128)')
    args = parser.parse_args()

    print("Loading TSB-AD benchmark...")
    tsb = TSBADParser()
    _, _, labels = tsb.load_timeseries(load_all=True)

    vus = VUSNumpy(
        slope_size=args.slope_size,
        step=1,
        global_mask=True,
        slopes='precomputed',
        existence='optimized',
        conf_matrix='dynamic_plus',
    )

    removed_fracs, shrinks = [], []
    for label in labels:
        T = len(label)
        if label.sum() == 0:
            continue
        ((_, _), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
        safe_mask, _ = vus.create_safe_mask(label, start_we, end_we, extra_safe=vus.global_mask)
        kept = int(safe_mask.sum())
        if kept == 0:
            continue
        removed_fracs.append(1 - kept / T)
        shrinks.append(T / kept)

    removed_fracs = np.array(removed_fracs)
    shrinks = np.array(shrinks)

    print("\n" + "=" * 60)
    print("GLOBAL MASK REMOVED-LENGTH SUMMARY")
    print("=" * 60)
    print(f"Series analyzed:        {len(removed_fracs)}")
    print(f"Removed fraction:       mean {removed_fracs.mean()*100:.2f}%  median {np.median(removed_fracs)*100:.2f}%")
    print(f"Removed frac 25/75:     {np.quantile(removed_fracs, 0.25)*100:.2f}% / {np.quantile(removed_fracs, 0.75)*100:.2f}%")
    print(f"Shrink factor (T/kept): mean {shrinks.mean():.1f}x  median {np.median(shrinks):.1f}x  max {shrinks.max():.1f}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
