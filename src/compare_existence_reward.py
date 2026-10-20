"""
Compare existence implementations: trivial vs optimized.
Runs both on a loaded time series and reports timing and correctness.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy


def compare_existence(vus, labels, sm, score, thresholds, start_points, end_points,
                      n_runs=10, atol=1e-8, rtol=1e-5, vus_gpu=None,
                      save_path='experiments/existence_comparison/existence_comparison_results.csv'):
    implementations = {
        'trivial': lambda: vus.existence_trivial(labels, sm, thresholds, start_points, end_points),
        'optimized': lambda: vus.existence_optimized(labels, score, thresholds, start_points, end_points),
    }

    if vus_gpu is not None:
        import torch
        # Feed the GPU implementation the exact same slope labels and score mask the
        # numpy versions get, so we compare the existence algorithm, not the inputs.
        vus_gpu.update_max_memory_tokens()
        # labels carries the float slope ramps (0..1); compute_existence does (labels > 0)
        # internally, so it must stay float — an int cast would truncate the ramps to 0.
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=vus_gpu.device)
        sm_t = torch.as_tensor(sm, dtype=torch.float32, device=vus_gpu.device)

        def run_gpu():
            # compute_existence is @time_it-decorated, so it returns (result, elapsed).
            res, _ = vus_gpu.compute_existence(labels_t, sm_t, safe_mask=None, normalize=True)
            if vus_gpu.device == 'cuda':
                torch.cuda.synchronize()
            return res.detach().cpu().numpy()

        implementations['gpu'] = run_gpu

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

    ref_result = results['trivial']
    comparisons = {}
    all_match = True

    for name, result in results.items():
        if name == 'trivial':
            comparisons[name] = {'is_reference': True}
            continue
        matches = np.allclose(result, ref_result, atol=atol, rtol=rtol)
        comparisons[name] = {
            'matches': matches,
            'max_diff': np.max(np.abs(result - ref_result)),
            'mean_diff': np.mean(np.abs(result - ref_result)),
        }
        if not matches:
            all_match = False

    print("=" * 70)
    print("EXISTENCE IMPLEMENTATIONS COMPARISON")
    print("=" * 70)
    print(f"Input shapes: labels={labels.shape}, sm={sm.shape}, thresholds={thresholds.shape}")
    print(f"Number of anomalies: {start_points.shape[0]}")
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
        print(f"{name:<15} {trivial_time / t['mean']:>10.2f}x")
    print()

    print("CORRECTNESS (compared to trivial):")
    print("-" * 70)
    for name, comp in comparisons.items():
        if comp.get('is_reference'):
            print(f"{name:<15} REFERENCE")
        else:
            status = "MATCH" if comp['matches'] else "DIFFER"
            print(f"{name:<15} {status:<10} max_diff={comp['max_diff']:.2e} mean_diff={comp['mean_diff']:.2e}")
    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL IMPLEMENTATIONS MATCH' if all_match else 'SOME IMPLEMENTATIONS DIFFER'}")
    print("=" * 70)

    row = {
        'n_slopes': labels.shape[0], 'T': labels.shape[1],
        'n_thresholds': thresholds.shape[0], 'n_anomalies': start_points.shape[0],
        'n_runs': n_runs,
        **{f'trivial_{k}': v for k, v in timings['trivial'].items()},
        **{f'optimized_{k}': v for k, v in timings['optimized'].items()},
        'speedup': trivial_time / timings['optimized']['mean'],
        'matches': comparisons['optimized']['matches'],
        'max_diff': comparisons['optimized']['max_diff'],
        'mean_diff': comparisons['optimized']['mean_diff'],
    }
    if 'gpu' in timings:
        row.update({f'gpu_{k}': v for k, v in timings['gpu'].items()})
        row['gpu_speedup'] = trivial_time / timings['gpu']['mean']
        row['gpu_matches'] = comparisons['gpu']['matches']
        row['gpu_max_diff'] = comparisons['gpu']['max_diff']
        row['gpu_mean_diff'] = comparisons['gpu']['mean_diff']
    new_row_df = pd.DataFrame([row])
    if os.path.exists(save_path):
        pd.concat([pd.read_csv(save_path), new_row_df], ignore_index=True).to_csv(save_path, index=False)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        new_row_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    return {'all_match': all_match, 'comparisons': comparisons, 'timings': timings, 'results': results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Quick smoke test: benchmark a single series (alias for --limit 1).')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help="Dataset name(s) to benchmark, e.g. --dataset YAHOO NAB. "
                             "Default: the whole TSB-AD benchmark.")
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap the number of series benchmarked (after filtering).')
    parser.add_argument('--slope_size', type=int, default=128)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=True,
                        help='Also benchmark the VUSTorch (GPU) existence implementation.')
    parser.add_argument('--device', default=None, help="Force torch device, e.g. 'cuda' or 'cpu'.")
    args = parser.parse_args()

    limit = 1 if (args.test and args.limit is None) else args.limit

    tsb = TSADParser()
    # No dataset filter -> load_all=True loads the whole TSB-AD benchmark.
    load_kwargs = {'dataset': args.dataset} if args.dataset else {}
    filenames, ts_list, labels = tsb.load_timeseries(**load_kwargs)
    if limit is not None:
        filenames, ts_list, labels = filenames[:limit], ts_list[:limit], labels[:limit]
    raw_scores, _ = tsb.load_scores_and_delete(filenames, ts_list, labels)
    scores = [s[0] for s in raw_scores]

    vus = VUSNumpy(slope_size=args.slope_size, step=1, global_mask=True,
                   slopes='precomputed', existence='optimized', conf_matrix='dynamic_plus')

    vus_gpu = None
    if args.gpu:
        from vus.vus_torch import VUSTorch
        vus_gpu = VUSTorch(slope_size=args.slope_size, step=1, global_mask=True,
                           existence=True, conf_matrix='dynamic_plus', device=args.device)

    n_total = len(filenames)
    for idx, (fname, label, score) in enumerate(zip(filenames, labels, scores), start=1):
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
        labels_slopes, _ = vus.add_slopes(label, start_no_edges, end_no_edges, None, None)

        print(f"\n[{idx}/{n_total}] File: {fname}")
        compare_existence(vus, labels_slopes, sm, score, thresholds, start_we, end_we,
                          n_runs=args.n_runs, vus_gpu=vus_gpu)


if __name__ == '__main__':
    main()
