"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""

import argparse
import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from legacy.sylli_metrics import sylli_get_metrics


# ---------------------------------------------------------------------------
# Local helpers (moved here from src/utils/ which is being deleted)
# ---------------------------------------------------------------------------

def _analyze_label(label):
    diff = np.diff(label)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if len(label) > 0 and label[-1]:
        ends = np.append(ends, len(label) - 1)
    if len(label) > 0 and label[0]:
        starts = np.append([0], starts)
    n = int(starts.shape[0])
    if n == 0:
        return len(label), 0, 0.0
    lengths = (ends + 1) - starts  # end is exclusive
    return len(label), n, float(np.mean(lengths))


def natural_keys(text):
    def atoi(t):
        return int(t) if t.isdigit() else t
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# torch.argsort/sort raise "dimension being sorted can not have more than
# INT_MAX elements" for tensors with more than 2**31 - 1 elements. The synthetic
# sweep goes up to 2**31 points, so cap the length just under the limit; dropping
# the tail 1-2 points is invisible on a log-scale scaling plot.
TORCH_SORT_MAX = 2**31 - 1


def load_synthetic(dataset, testing=False, iterator=False):
    dataset_path = os.path.join(TSADParser.benchmarks_root(), '_synthetic', dataset)
    files = sorted(os.listdir(dataset_path), key=natural_keys)

    for file in tqdm(files, desc="Loading synthetic"):
        data  = np.load(os.path.join(dataset_path, file))
        label = data['label']
        score = data['score'].round(2)
        if len(score) > TORCH_SORT_MAX:
            label = label[:TORCH_SORT_MAX]
            score = score[:TORCH_SORT_MAX]
        if iterator:
            yield (file, label, score)
        else:
            if testing:
                return [(file, label, score)]
    if not iterator:
        return [(f, np.load(os.path.join(dataset_path, f))['label'],
                 np.load(os.path.join(dataset_path, f))['score'].round(2))
                for f in files]


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metric(
        metric,
        data,
        global_mask=True,
        slope_size=None,
        step=None,
        slopes=None,
        existence=None,
        conf_matrix=None,
        thresholds=None,
        zita=(1 / math.sqrt(2)),
        save_path=None,
        n_thresholds=None,
):
    metric_name = metric.replace('_', '-').upper()
    results = []

    if metric == 'ff_vus':
        ff_vus = VUSNumpy(
            global_mask=global_mask,
            slope_size=slope_size,
            zita=zita,
            step=step,
            slopes=slopes,
            existence=existence,
            conf_matrix=conf_matrix,
            thresholds=thresholds,
            side='both',
            max_thresholds = n_thresholds if n_thresholds is not None else 2000,
        )
    elif metric == 'ff_vus_gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        existence = False if existence.lower() == "none" else bool(existence)
        slopes = 'function'
        ff_vus = VUSTorch(
            global_mask=global_mask,
            slope_size=slope_size,
            zita=zita,
            step=step,
            existence=existence,
            conf_matrix=conf_matrix,
            device=device,
            thresholds=thresholds,
            max_thresholds = n_thresholds if n_thresholds is not None else 2000,
        )

    for filename, label, score, detector in tqdm(data, desc=f'Computing {metric}'):
        length, n_anomalies, anomalies_avg_length = _analyze_label(label)
        curr_result = {
            "Time series":             filename,
            "Length":                  length,
            "Number of anomalies":     n_anomalies,
            "Anomalies average length": float(anomalies_avg_length),
        }
        if detector is not None:
            curr_result["Detector"] = detector

        if label.shape != score.shape:
            min_len = min(label.shape[0], score.shape[0])
            label = label[:min_len]
            score = score[:min_len]

        if metric in ('ff_vus', 'ff_vus_gpu'):
            curr_result.update({
                'Slope size':      slope_size,
                'Step':            step,
                'Global mask':     global_mask,
                'Slopes':          slopes,
                'Existence':       existence,
                'Confusion matrix': conf_matrix,
                'Thresholds':      thresholds,
            })

            with torch.no_grad():
                if metric == 'ff_vus_gpu':
                    label = torch.tensor(label, device=device)
                    score = torch.tensor(score, device=device)

                (metric_value, ff_vus_time_analysis), metric_time = ff_vus.compute(label, score)
            curr_result.update(ff_vus_time_analysis)
        else:
            if metric in ('range_auc', 'vus', 'softed'):
                curr_result.update({'Slope size': slope_size})
            if metric == 'vus':
                curr_result.update({'Existence': True if existence != 'None' else False})
            metric_value, metric_time = sylli_get_metrics(
                label, score, metric, slope_size,
                existence=True if existence != 'None' else False,
                **({"max_thresholds": n_thresholds} if n_thresholds is not None else {}),
            )
            if metric == 'softed':
                metric_value, metric_time = metric_value
                metric_value = metric_value['F1']

        curr_result.update({
            'Metric':       metric_name,
            'Metric value': float(metric_value),
            'Metric time':  metric_time,
        })

        if save_path is not None:
            df_row = pd.DataFrame([curr_result])
            write_header = not os.path.exists(save_path)
            df_row.to_csv(save_path, mode='a', header=write_header, index=False)

        results.append(curr_result)

    return pd.DataFrame(results)


def compute_metric_over_dataset(
        dataset,
        metric,
        global_mask=True,
        slope_size=100,
        step=1,
        slopes='function',
        existence='optimized',
        conf_matrix='dynamic',
        thresholds='pr_relevant_fuzzy',
        zita=(1 / math.sqrt(2)),
        testing=False,
        experiment_dir=None,
        n_thresholds=None,
        detector=None,
    ):

    # Load dataset
    if dataset == 'tsb':
        tsb = TSADParser()
        if testing:
            fnames, timeseries, labels = tsb.load_timeseries(dataset=['NAB'])
            fnames, timeseries, labels = fnames[:10], timeseries[:10], labels[:10]
        else:
            fnames, timeseries, labels = tsb.load_timeseries(load_all=True)

        detectors = tsb.get_detectors()
        # 'random' loads every detector then picks one per time series, so the
        # parser must load all of them (same cost as 'all').
        per_series_random = detector == 'random'
        if detector is None:
            selected = [detectors[0]]
        elif detector == 'all' or per_series_random:
            selected = detectors
        elif detector in detectors:
            selected = [detector]
        else:
            raise ValueError(f"Unknown detector '{detector}'. Available: {detectors}")
        print(f"Using detector(s): {'random per series, drawn from all' if per_series_random else selected}")

        scores_grouped, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=selected)
        # Each scores_grouped[i] has shape (len(selected), T).
        if per_series_random:
            # One random detector per time series.
            data = []
            for fname, label, score_stack in zip(fnames, labels, scores_grouped):
                i = random.randrange(len(selected))
                data.append((fname, label, score_stack[i], selected[i]))
        else:
            # Emit one data entry per (time series, detector) so every detector
            # gets its own row.
            data = [
                (fname, label, score, det)
                for fname, label, score_stack in zip(fnames, labels, scores_grouped)
                for det, score in zip(selected, score_stack)
            ]
    elif 'syn_' in dataset:
        data = [(f, label, score, None)
                for f, label, score in load_synthetic(dataset=dataset, testing=testing, iterator=True)]
    else:
        raise ValueError(f"Wrong argument for dataset: {dataset}")

    data = list(data)

    if metric == 'all':
        metrics = ['ff_vus_gpu', 'auc', 'ff_vus', 'affiliation', 'range_auc', 'vus', 'rf']
    else:
        metrics = [metric]

    if slope_size == -1:
        slope_sizes = [0, 16, 32, 64, 128, 256]
    else:
        slope_sizes = [slope_size]

    if experiment_dir is not None:
        save_path    = os.path.join('experiments', experiment_dir)
        results_path = os.path.join(save_path, 'results')
        info_path    = os.path.join(save_path, 'info')
        os.makedirs(save_path,    exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(info_path,    exist_ok=True)

    for metric in metrics:
        for slope_size in slope_sizes:
            if experiment_dir is not None:
                filename = f"{dataset}_{metric.replace('_', '-').upper()}"
                if metric in ('ff_vus', 'ff_vus_gpu', 'range_auc', 'vus'):
                    filename += f"_{slope_size}"
                if metric in ('ff_vus', 'ff_vus_gpu'):
                    filename += f"_{step}_{conf_matrix}_{'globalmask' if global_mask else 'noglobalmask'}_{slopes}_{existence}_{thresholds}"
                filename += ".csv"
                curr_save_path = os.path.join(results_path, filename)
            else:
                curr_save_path = None

            df = compute_metric(metric, data, global_mask, slope_size, step, slopes,
                                existence, conf_matrix, thresholds, zita, curr_save_path, n_thresholds)

            if not df.empty:
                print(df)
                print(f"Average computation time: {df['Metric time'].mean():.3f} seconds")
                # if metric in ('ff_vus', 'ff_vus_gpu'):
                #     time_cols = [c for c in df.columns if c.endswith(' time') and c != 'Metric time']
                #     if time_cols:
                #         avg_times = df[time_cols].mean()
                #         print("\nFF-VUS time analysis (averages):")
                #         for col, val in avg_times.items():
                #             print(f"  {col}: {val:.6f} s")

                #         labels = [c.replace(' time', '') for c in time_cols]
                #         raw    = [df[c].dropna().values for c in time_cols]
                #         order  = sorted(range(len(raw)), key=lambda i: -raw[i].mean())

                #         plot_data = []
                #         for i in order:
                #             vals = raw[i]
                #             q1, q3 = np.percentile(vals, [25, 75])
                #             iqr = q3 - q1
                #             mask = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
                #             for v in vals[mask]:
                #                 plot_data.append({"component": labels[i], "time": v})
                #         plot_df = pd.DataFrame(plot_data)
                #         ordered_labels = [labels[i] for i in order]

                #         fig, ax = plt.subplots(figsize=(10, 5))
                #         sns.boxplot(
                #             data=plot_df,
                #             x="component",
                #             y="time",
                #             order=ordered_labels,
                #             showfliers=True,
                #             ax=ax,
                #             showmeans=True,
                #         )
                #         ax.set_xlabel("")
                #         ax.set_ylabel("Time (s)")
                #         ax.set_title(f"FF-VUS runtime breakdown — {metric} / slope={slope_size}")
                #         ax.set_yscale('log')
                #         ax.grid(axis='y', linestyle='--', alpha=0.5)
                #         plt.xticks(rotation=30, ha='right')
                #         plt.tight_layout()
                #         plt.show()

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
                    "Experiment": f"Compute {metric}",
                    "Number of results": len(df),
                    "Slope size": slope_size,
                    "Step": step,
                    "Global mask": global_mask,
                    "Slopes": slopes,
                    "Existence": existence,
                    "Confusion matrix": conf_matrix,
                    "Thresholds": thresholds,
                    "Detector": detector,
                    "Time": df.iloc[:, -1].sum() if not df.empty else 0,
                }])
                info_df.to_csv(os.path.join(info_path, filename), index=False)

            if metric in ('auc', 'affiliation', 'rf'):
                break

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Compute a specific metric for a whole dataset'
    )

    parser.add_argument('--dataset',     type=str,   required=True, help='Path or name of the dataset')
    parser.add_argument('--metric',      type=str,   required=True, help='Metric to compute (e.g., VUS, AUC-PR, etc.)')
    parser.add_argument('--no_global_mask', action='store_true',    help='Disable global mask for metric computation')
    parser.add_argument('--slope_size',  type=int,   default=128,   help='Number of slopes used for computation')
    parser.add_argument('--step',        type=int,   default=1,     help='Step size between slopes')
    parser.add_argument('--slopes',      type=str,   choices=['precomputed', 'function'], default='function',
                        help='Slope generation method')
    parser.add_argument('--existence',   type=str,   choices=['None', 'trivial', 'optimized', 'matrix'], default='optimized',
                        help='Existence computation method')
    parser.add_argument('--conf_matrix', type=str,   choices=['trivial', 'dynamic', 'dynamic_plus'], default='dynamic_plus',
                        help='Type of confusion matrix computation')
    parser.add_argument('--thresholds',  type=str,   choices=['all_unique', 'subsampled', 'pr_relevant', 'pr_relevant_fuzzy'], default='pr_relevant_fuzzy',
                        help='Threshold selection strategy')
    parser.add_argument('--zita',        type=float, default=(1 / math.sqrt(2)),
                        help='FFVUS slope angle parameter, must be in [0, 1] (default: 1/sqrt(2))')
    parser.add_argument('--test',        action='store_true', help='Run in testing mode (limits the data for fast testing)')
    parser.add_argument('--experiment',  type=str,   default=None,  help='Directory to save experiment results and info')
    parser.add_argument('--n_thresholds', type=int,  default=None,  help='Override max thresholds for ff_vus, ff_vus_gpu, and vus (default: use each metric\'s built-in default)')
    parser.add_argument('--detector',    type=str,   default=None,  help="TSB detector to use: a detector name, 'random' (a random detector per time series), or 'all' (every detector). Default: first detector")

    args = parser.parse_args()

    if args.dataset == 'all_synthetic':
        synthetic_dir = os.path.join(TSADParser.benchmarks_root(), '_synthetic')
        datasets = os.listdir(synthetic_dir)
        datasets.sort()
    else:
        datasets = [args.dataset]
    datasets.sort(key=natural_keys)

    for dataset in datasets:
        compute_metric_over_dataset(
            dataset=dataset,
            metric=args.metric,
            global_mask=not args.no_global_mask,
            slope_size=args.slope_size,
            step=args.step,
            slopes=args.slopes,
            existence=args.existence,
            conf_matrix=args.conf_matrix,
            thresholds=args.thresholds,
            zita=args.zita,
            testing=args.test,
            experiment_dir=args.experiment,
            n_thresholds=args.n_thresholds,
            detector=args.detector,
        )
