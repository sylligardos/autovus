"""
Compare add_slopes implementations: precomputed vs function.
Runs both on a loaded time series and reports timing and correctness.
"""

import os
import sys
import time
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy


def compare_add_slopes(vus, label, start_no_edges, end_no_edges, pos,
                       n_runs=10, atol=1e-8, rtol=1e-5):
    implementations = {
        'precomputed': lambda: vus.add_slopes_precomputed(label, start_no_edges, end_no_edges),
        'function':    lambda: vus.add_slopes_function(label, pos),
    }

    results = {}
    timings = {}

    for name, func in implementations.items():
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            result = func()
            times.append(time.time() - t0)
        results[name] = result
        timings[name] = {
            'mean': np.mean(times), 'std': np.std(times),
            'min': np.min(times),   'max': np.max(times),
        }

    ref = results['precomputed']
    matches = np.allclose(results['function'], ref, atol=atol, rtol=rtol)
    max_diff  = np.max(np.abs(results['function'] - ref))
    mean_diff = np.mean(np.abs(results['function'] - ref))

    print("=" * 70)
    print("ADD_SLOPES IMPLEMENTATIONS COMPARISON")
    print("=" * 70)
    print(f"T={len(label)}, n_slopes={vus.n_slopes}, slope_size={vus.slope_size}")
    print(f"n_anomalies={len(start_no_edges)}")
    print(f"n_runs={n_runs}")
    print()

    print("TIMING RESULTS (seconds):")
    print("-" * 70)
    print(f"{'Implementation':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    for name, t in timings.items():
        print(f"{name:<15} {t['mean']:>12.6f} {t['std']:>12.6f} {t['min']:>12.6f} {t['max']:>12.6f}")
    print()

    print("SPEEDUP (relative to precomputed):")
    print("-" * 70)
    ref_time = timings['precomputed']['mean']
    for name, t in timings.items():
        print(f"{name:<15} {ref_time / t['mean']:>10.2f}x")
    print()

    print("CORRECTNESS (compared to precomputed):")
    print("-" * 70)
    print(f"{'precomputed':<15} REFERENCE")
    status = "MATCH" if matches else "DIFFER"
    print(f"{'function':<15} {status:<10} max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    print()
    print("=" * 70)
    print(f"OVERALL: {'MATCH' if matches else 'DIFFER'}")
    print("=" * 70)

    return {'matches': matches, 'timings': timings, 'max_diff': max_diff, 'mean_diff': mean_diff}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--slope_size', type=int, default=128)
    parser.add_argument('--n_runs',     type=int, default=10)
    parser.add_argument('--dataset',    type=str, default='YAHOO')
    parser.add_argument('--n',          type=int, default=5)
    args = parser.parse_args()

    tsb = TSADParser()
    n = 1 if args.test else args.n
    filenames, ts_list, labels = tsb.load_timeseries(dataset=[args.dataset])
    filenames, ts_list, labels = filenames[:n], ts_list[:n], labels[:n]
    raw_scores, _ = tsb.load_scores_and_delete(filenames, ts_list, labels)
    scores = [s[0] for s in raw_scores]

    vus = VUSNumpy(slope_size=args.slope_size, step=1, global_mask=True,
                   slopes='precomputed', existence='optimized', conf_matrix='dynamic_plus')

    for fname, label, score in zip(filenames, labels, scores):
        min_len = min(len(label), len(score))
        label, score = label[:min_len], score[:min_len]
        if label.sum() == 0:
            continue

        ((start_no_edges, end_no_edges), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
        pos, _ = vus.distance_from_anomaly(label, start_we, end_we)

        print(f"\nFile: {fname}")
        compare_add_slopes(vus, label, start_no_edges, end_no_edges, pos, n_runs=args.n_runs)

        if args.test:
            break


if __name__ == '__main__':
    main()
