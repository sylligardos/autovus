"""
Benchmark confusion matrix implementations over a whole dataset.

CPU (numpy): trivial, dynamic, dynamic_plus, sparse (scipy.sparse)
GPU (torch): trivial, dynamic, dynamic_plus, cusparse (torch sparse COO)

For every time series the inputs (masked labels with slopes, thresholded score
mask) are prepared once, then each implementation is timed n_runs times and
checked for correctness against the trivial implementation.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch


def time_implementation(func, args, n_runs, cuda=False):
    times = []
    for _ in range(n_runs):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func(*args)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def benchmark_series(implementations, labels_slopes, sm, n_runs, cuda=False, atol=1e-5, rtol=1e-5):
    """Time every implementation on prepared inputs and compare outputs to trivial."""
    row = {}
    results = {}

    for name, func in implementations.items():
        raw = func(labels_slopes, sm)
        if cuda:
            raw = tuple(t.float().cpu().numpy() for t in raw)
        results[name] = raw
        times = time_implementation(func, (labels_slopes, sm), n_runs, cuda=cuda)
        row[f'{name} mean'] = float(np.mean(times))
        row[f'{name} std'] = float(np.std(times))
        row[f'{name} min'] = float(np.min(times))

    ref_fn, ref_tp, ref_pos = results['trivial']
    for name, (fn, tp, pos) in results.items():
        if name == 'trivial':
            continue
        fn, ref_fn_s = np.asarray(fn).squeeze(), np.asarray(ref_fn).squeeze()
        matches = (np.allclose(fn, ref_fn_s, atol=atol, rtol=rtol)
                   and np.allclose(tp, ref_tp, atol=atol, rtol=rtol)
                   and np.allclose(pos, ref_pos, atol=atol, rtol=rtol))
        row[f'{name} match'] = bool(matches)
        row[f'{name} max diff'] = float(max(
            np.max(np.abs(fn - ref_fn_s)),
            np.max(np.abs(np.asarray(tp) - np.asarray(ref_tp))),
            np.max(np.abs(np.asarray(pos) - np.asarray(ref_pos))),
        ))

    return row


def prepare_numpy(vus, label, score):
    ((start_ne, end_ne), (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
    safe_mask, _ = vus.create_safe_mask(label, start_we, end_we, extra_safe=vus.global_mask)
    label, score = label[safe_mask], score[safe_mask]
    if label.sum() == 0:
        return None, None

    ((start_ne, end_ne), _), _ = vus.get_anomalies_coordinates(label)
    thresholds, _ = vus.get_unique_thresholds(score)
    sm, _ = vus.get_score_mask(score, thresholds)
    labels_slopes, _ = vus.add_slopes(label, start_ne, end_ne, None, None)
    return labels_slopes, sm


def prepare_torch(vus, label, score):
    label = torch.tensor(label, dtype=torch.int8, device=vus.device)
    score = torch.tensor(score, dtype=torch.float32, device=vus.device)

    (_, (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
    safe_mask, _ = vus.create_safe_mask(label, start_we, end_we, extra_safe=vus.global_mask)
    label, score = label[safe_mask], score[safe_mask]
    if label.sum() == 0:
        return None, None

    (_, (start_we, end_we)), _ = vus.get_anomalies_coordinates(label)
    pos, _ = vus.distance_from_anomaly(label, start_we, end_we)
    thresholds, _ = vus.get_unique_thresholds(score)
    sm, _ = vus.get_score_mask(score, thresholds)
    labels_slopes, _ = vus.add_slopes(label, pos)
    return labels_slopes, sm


def compare_over_dataset(
        dataset,
        backend,
        global_mask=True,
        slope_size=128,
        step=1,
        thresholds='pr_relevant_fuzzy',
        n_runs=10,
        testing=False,
        experiment_dir=None,
        detector=None,
    ):
    # Load dataset
    if dataset == 'tsb':
        tsb = TSBADParser()
        if testing:
            fnames, timeseries, labels = tsb.load_timeseries(dataset=['NAB'])
            fnames, timeseries, labels = fnames[:10], timeseries[:10], labels[:10]
        else:
            fnames, timeseries, labels = tsb.load_timeseries(load_all=True)

        detectors = tsb.get_detectors()
        if detector is None:
            selected = [detectors[0]]
        elif detector == 'all':
            selected = detectors
        elif detector in detectors:
            selected = [detector]
        else:
            raise ValueError(f"Unknown detector '{detector}'. Available: {detectors}")
        print(f"Using detector(s): {selected}")

        scores_grouped, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=selected)
        data = [
            (fname, label, score, det)
            for fname, label, score_stack in zip(fnames, labels, scores_grouped)
            for det, score in zip(selected, score_stack)
        ]
    else:
        raise ValueError(f"Wrong argument for dataset: {dataset}")

    backends = ['numpy', 'torch'] if backend == 'both' else [backend]

    if experiment_dir is not None:
        save_path    = os.path.join('experiments', experiment_dir)
        results_path = os.path.join(save_path, 'results')
        info_path    = os.path.join(save_path, 'info')
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(info_path,    exist_ok=True)

    all_dfs = {}
    for curr_backend in backends:
        if curr_backend == 'torch':
            vus = VUSTorch(slope_size=slope_size, step=step, global_mask=global_mask,
                           existence=True, conf_matrix='dynamic_plus', thresholds=thresholds)
            implementations = {
                'trivial':      vus.conf_matrix_trivial,
                'dynamic':      vus.conf_matrix_dyn,
                'dynamic_plus': vus.conf_matrix_dyn_plus,
                'cusparse':     vus.conf_matrix_cusparse,
            }
            cuda = vus.device == 'cuda'
        else:
            vus = VUSNumpy(slope_size=slope_size, step=step, global_mask=global_mask,
                           slopes='precomputed', existence='optimized',
                           conf_matrix='dynamic_plus', thresholds=thresholds)
            implementations = {
                'trivial':      vus.conf_matrix_trivial,
                'dynamic':      vus.conf_matrix_dyn,
                'dynamic_plus': vus.conf_matrix_dyn_plus,
                'sparse':       vus.sparse_conf_matrix,
            }
            cuda = False

        if experiment_dir is not None:
            filename = f"{dataset}_confmatrix_{curr_backend}_{slope_size}_{step}_{thresholds}_{n_runs}runs.csv"
            curr_save_path = os.path.join(results_path, filename)
        else:
            curr_save_path = None

        rows = []
        for fname, label, score, det in tqdm(data, desc=f'Benchmarking conf matrix ({curr_backend})'):
            min_len = min(len(label), len(score))
            label, score = label[:min_len], score[:min_len]

            try:
                with torch.no_grad():
                    if curr_backend == 'torch':
                        labels_slopes, sm = prepare_torch(vus, label, score)
                    else:
                        labels_slopes, sm = prepare_numpy(vus, label, score)
                    if labels_slopes is None:
                        continue

                    row = {
                        'Time series':   fname,
                        'Detector':      det,
                        'Length':        min_len,
                        'Masked length': int(labels_slopes.shape[1]),
                        'Buffers (L)':   int(labels_slopes.shape[0]),
                        'Thresholds (N)': int(sm.shape[0]),
                    }
                    row.update(benchmark_series(implementations, labels_slopes, sm,
                                                n_runs=n_runs, cuda=cuda))
            except Exception as e:
                print(f"\n{fname}: Error - {e}")
                continue

            if curr_save_path is not None:
                df_row = pd.DataFrame([row])
                write_header = not os.path.exists(curr_save_path)
                df_row.to_csv(curr_save_path, mode='a', header=write_header, index=False)
            rows.append(row)

        df = pd.DataFrame(rows)
        all_dfs[curr_backend] = df
        if df.empty:
            continue

        print(f"\n=== {curr_backend.upper()} summary over {len(df)} series ===")
        trivial_total = df['trivial mean'].sum()
        for name in implementations:
            total = df[f'{name} mean'].sum()
            line = f"{name:<15} total {total:>10.4f}s   speedup vs trivial {trivial_total / total:>8.2f}x"
            if name != 'trivial':
                line += f"   match {df[f'{name} match'].mean() * 100:.1f}%   max diff {df[f'{name} max diff'].max():.2e}"
            print(line)

        if experiment_dir is not None:
            info_df = pd.DataFrame([{
                "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "GPU Memory (GB)": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else None,
                "CPU": os.popen("lscpu | grep 'Model name' | awk -F ':' '{print $2}'").read().strip(),
                "Cores": os.cpu_count(),
                "RAM (GB)": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2) if hasattr(os, 'sysconf') else None,
                "Platform": os.uname().sysname if hasattr(os, 'uname') else None,
                "Platform-release": os.uname().release if hasattr(os, 'uname') else None,
                "Python version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "Experiment": f"Confusion matrix benchmark ({curr_backend})",
                "Number of results": len(df),
                "Slope size": slope_size,
                "Step": step,
                "Global mask": global_mask,
                "Thresholds": thresholds,
                "Runs per implementation": n_runs,
                "Detector": detector,
            }])
            info_df.to_csv(os.path.join(info_path, filename), index=False)

    return all_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='compare_confusion_matrix',
        description='Benchmark confusion matrix implementations over a whole dataset'
    )

    parser.add_argument('--dataset',     type=str,   default='tsb',  help='Dataset to benchmark on')
    parser.add_argument('--backend',     type=str,   choices=['numpy', 'torch', 'both'], default='both',
                        help='Which backend(s) to benchmark')
    parser.add_argument('--no_global_mask', action='store_true',     help='Disable global mask before the confusion matrix step')
    parser.add_argument('--slope_size',  type=int,   default=128,    help='Number of slopes used for computation')
    parser.add_argument('--step',        type=int,   default=1,      help='Step size between slopes')
    parser.add_argument('--thresholds',  type=str,   choices=['all_unique', 'subsampled', 'pr_relevant', 'pr_relevant_fuzzy'], default='pr_relevant_fuzzy',
                        help='Threshold selection strategy')
    parser.add_argument('--n_runs',      type=int,   default=10,     help='Timed runs per implementation per series')
    parser.add_argument('--test',        action='store_true',        help='Run in testing mode (limits the data for fast testing)')
    parser.add_argument('--experiment',  type=str,   default=None,   help='Directory to save experiment results and info')
    parser.add_argument('--detector',    type=str,   default=None,   help="TSB detector to use: a detector name or 'all'. Default: first detector")

    args = parser.parse_args()

    compare_over_dataset(
        dataset=args.dataset,
        backend=args.backend,
        global_mask=not args.no_global_mask,
        slope_size=args.slope_size,
        step=args.step,
        thresholds=args.thresholds,
        n_runs=args.n_runs,
        testing=args.test,
        experiment_dir=args.experiment,
        detector=args.detector,
    )
