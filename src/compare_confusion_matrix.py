"""
Compare confusion matrix implementations: trivial, dynamic, dynamic_plus, sparse.
Runs all on a loaded time series and reports timing and correctness.
"""

import os
import sys
import time
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch


def compare_confusion(vus, labels, sm, n_runs=10, atol=1e-8, rtol=1e-5):
    implementations = {
        'trivial':      vus.conf_matrix_trivial,
        'dynamic':      vus.conf_matrix_dyn,
        'dynamic_plus': vus.conf_matrix_dyn_plus,
        'sparse':       vus.sparse_conf_matrix,
    }

    results = {}
    timings = {}

    for name, func in implementations.items():
        results[name] = func(labels, sm)
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            func(labels, sm)
            times.append(time.time() - t0)
        timings[name] = {
            'mean': np.mean(times), 'std': np.std(times),
            'min': np.min(times),   'max': np.max(times),
        }

    ref_fn, ref_tp, ref_pos = results['trivial']
    comparisons = {}
    all_match = True

    for name, (fn, tp, pos) in results.items():
        if name == 'trivial':
            continue
        fn_match  = np.allclose(fn,  ref_fn,  atol=atol, rtol=rtol)
        tp_match  = np.allclose(tp,  ref_tp,  atol=atol, rtol=rtol)
        pos_match = np.allclose(pos, ref_pos, atol=atol, rtol=rtol)
        matches = fn_match and tp_match and pos_match
        all_match = all_match and matches
        comparisons[name] = {
            'matches': matches,
            'fn_match': fn_match,   'fn_max_diff':  np.max(np.abs(fn  - ref_fn))  if not fn_match  else 0,
            'tp_match': tp_match,   'tp_max_diff':  np.max(np.abs(tp  - ref_tp))  if not tp_match  else 0,
            'pos_match': pos_match, 'pos_max_diff': np.max(np.abs(pos - ref_pos)) if not pos_match else 0,
        }

    print("=" * 70)
    print("CONFUSION MATRIX IMPLEMENTATIONS COMPARISON")
    print("=" * 70)
    print(f"Input shapes: labels={labels.shape}, sm={sm.shape}")
    print(f"Number of timing runs: {n_runs}")
    print()

    print("TIMING RESULTS (seconds):")
    print("-" * 70)
    print(f"{'Implementation':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    for name, t in timings.items():
        print(f"{name:<15} {t['mean']:>12.6f} {t['std']:>12.6f} {t['min']:>12.6f} {t['max']:>12.6f}")
    print()

    print("SPEEDUP (relative to trivial):")
    print("-" * 70)
    trivial_time = timings['trivial']['mean']
    for name, t in timings.items():
        print(f"{name:<15} {trivial_time / t['mean']:>8.2f}x")
    print()

    print("CORRECTNESS (compared to trivial):")
    print("-" * 70)
    for name, comp in comparisons.items():
        status = "MATCH" if comp['matches'] else "MISMATCH"
        print(f"{name:<15} {status}")
        if not comp['matches']:
            print(f"  fn_match={comp['fn_match']}  (max_diff: {comp['fn_max_diff']:.2e})")
            print(f"  tp_match={comp['tp_match']}  (max_diff: {comp['tp_max_diff']:.2e})")
            print(f"  pos_match={comp['pos_match']} (max_diff: {comp['pos_max_diff']:.2e})")
    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL IMPLEMENTATIONS MATCH' if all_match else 'SOME IMPLEMENTATIONS DIFFER'}")
    print("=" * 70)

    return {'all_match': all_match, 'comparisons': comparisons, 'timings': timings, 'results': results}


def compare_torch_confusion(vus, labels, sm, n_runs=10, atol=1e-5, rtol=1e-5):
    implementations = {
        'trivial':      vus.conf_matrix_trivial,
        'dynamic':      vus.conf_matrix_dyn,
        'dynamic_plus': vus.conf_matrix_dyn_plus,
        'cusparse':     vus.conf_matrix_cusparse,
    }

    results = {}
    timings = {}

    for name, func in implementations.items():
        try:
            raw = func(labels, sm)
            results[name] = tuple(t.float().cpu().numpy() for t in raw)
            times = []
            for _ in range(n_runs):
                t0 = time.time()
                func(labels, sm)
                times.append(time.time() - t0)
            timings[name] = {
                'mean': np.mean(times), 'std': np.std(times),
                'min': np.min(times),   'max': np.max(times),
            }
        except Exception as e:
            print(f"{name}: Error - {e}")

    ref_fn, ref_tp, ref_pos = results['trivial']
    comparisons = {}
    all_match = True

    for name, (fn, tp, pos) in results.items():
        if name == 'trivial':
            continue
        fn_match  = np.allclose(fn.squeeze(),  ref_fn.squeeze(),  atol=atol, rtol=rtol)
        tp_match  = np.allclose(tp,  ref_tp,  atol=atol, rtol=rtol)
        pos_match = np.allclose(pos, ref_pos, atol=atol, rtol=rtol)
        matches = fn_match and tp_match and pos_match
        all_match = all_match and matches
        comparisons[name] = {
            'matches': matches,
            'fn_match': fn_match,   'fn_max_diff':  np.max(np.abs(fn.squeeze()  - ref_fn.squeeze()))  if not fn_match  else 0,
            'tp_match': tp_match,   'tp_max_diff':  np.max(np.abs(tp  - ref_tp))  if not tp_match  else 0,
            'pos_match': pos_match, 'pos_max_diff': np.max(np.abs(pos - ref_pos)) if not pos_match else 0,
        }

    print("=" * 70)
    print("CONFUSION MATRIX IMPLEMENTATIONS COMPARISON (Torch)")
    print("=" * 70)
    print(f"Input shapes: labels={labels.shape}, sm={sm.shape}")
    print(f"Number of timing runs: {n_runs}")
    print()

    print("TIMING RESULTS (seconds):")
    print("-" * 70)
    print(f"{'Implementation':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    for name, t in timings.items():
        print(f"{name:<15} {t['mean']:>12.6f} {t['std']:>12.6f} {t['min']:>12.6f} {t['max']:>12.6f}")
    print()

    print("SPEEDUP (relative to trivial):")
    print("-" * 70)
    trivial_time = timings['trivial']['mean']
    for name, t in timings.items():
        print(f"{name:<15} {trivial_time / t['mean']:>8.2f}x")
    print()

    print("CORRECTNESS (compared to trivial):")
    print("-" * 70)
    for name, comp in comparisons.items():
        status = "MATCH" if comp['matches'] else "MISMATCH"
        print(f"{name:<15} {status}")
        if not comp['matches']:
            print(f"  fn_match={comp['fn_match']}  (max_diff: {comp['fn_max_diff']:.2e})")
            print(f"  tp_match={comp['tp_match']}  (max_diff: {comp['tp_max_diff']:.2e})")
            print(f"  pos_match={comp['pos_match']} (max_diff: {comp['pos_max_diff']:.2e})")
    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL IMPLEMENTATIONS MATCH' if all_match else 'SOME IMPLEMENTATIONS DIFFER'}")
    print("=" * 70)

    return {'all_match': all_match, 'comparisons': comparisons, 'timings': timings, 'results': results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--slope_size', type=int, default=128)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy')
    args = parser.parse_args()

    tsb = TSBADParser()
    n = 1 if args.test else 5
    filenames, ts_list, labels = tsb.load_timeseries(dataset=['YAHOO'])
    filenames, ts_list, labels = filenames[:n], ts_list[:n], labels[:n]
    raw_scores, _ = tsb.load_scores_and_delete(filenames, ts_list, labels)
    scores = [s[0] for s in raw_scores]

    if args.backend == 'torch':
        vus = VUSTorch(slope_size=args.slope_size, step=1, global_mask=True,
                       existence=True, conf_matrix='dynamic_plus')

        for fname, label, score in zip(filenames, labels, scores):
            min_len = min(len(label), len(score))
            label_t = torch.tensor(label[:min_len], dtype=torch.int8)
            score_t = torch.tensor(score[:min_len], dtype=torch.float32)

            ((_), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label_t)
            safe_mask, _ = vus.create_safe_mask(label_t, start_we, end_we, extra_safe=vus.global_mask)
            label_t, score_t = label_t[safe_mask], score_t[safe_mask]
            if label_t.sum() == 0:
                continue

            ((_), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label_t)
            pos, _ = vus.distance_from_anomaly(label_t, start_we, end_we)
            thresholds, _ = vus.get_unique_thresholds(score_t)
            sm, _ = vus.get_score_mask(score_t, thresholds)
            labels_slopes, _ = vus.add_slopes(label_t, pos)

            print(f"\nFile: {fname}")
            compare_torch_confusion(vus, labels_slopes, sm, n_runs=args.n_runs)
            break

    else:
        vus = VUSNumpy(slope_size=args.slope_size, step=1, global_mask=True,
                       slopes='precomputed', existence='optimized', conf_matrix='dynamic_plus')

        for fname, label, score in zip(filenames, labels, scores):
            min_len = min(len(label), len(score))
            label, score = label[:min_len], score[:min_len]

            ((start_no_edges, end_no_edges), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
            safe_mask, _ = vus.create_safe_mask(label, start_we, end_we, extra_safe=vus.global_mask)
            label, score = label[safe_mask], score[safe_mask]
            if label.sum() == 0:
                continue

            ((start_no_edges, end_no_edges), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
            thresholds, _ = vus.get_unique_thresholds(score)
            sm, _ = vus.get_score_mask(score, thresholds)
            labels_slopes, _ = vus.add_slopes(label, start_no_edges, end_no_edges, None)

            print(f"\nFile: {fname}")
            compare_confusion(vus, labels_slopes, sm, n_runs=args.n_runs)
            break


if __name__ == '__main__':
    main()
