"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""

import argparse
import math
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tsbadparser import TSBADParser
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


def load_synthetic(dataset, testing=False, iterator=False):
    dataset_path = os.path.join('data', 'synthetic', dataset)
    files = sorted(os.listdir(dataset_path), key=natural_keys)

    for file in tqdm(files, desc="Loading synthetic"):
        data  = np.load(os.path.join(dataset_path, file))
        label = data['label']
        score = data['score'].round(2)
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
            thresholds=thresholds if thresholds in ('all_unique', 'subsampled', 'pr_relevant_fuzzy') else 'pr_relevant_fuzzy',
        )

    for filename, label, score in tqdm(data, desc=f'Computing {metric}'):
        length, n_anomalies, anomalies_avg_length = _analyze_label(label)
        curr_result = {
            "Time series":             filename,
            "Length":                  length,
            "Number of anomalies":     n_anomalies,
            "Anomalies average length": float(anomalies_avg_length),
        }

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
                    label = torch.tensor(label, device=device, dtype=torch.uint8)
                    score = torch.tensor(score, device=device, dtype=torch.float16)

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
        scores_grouped, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=detectors[:1])
        # Each scores_grouped[i] is (1, T); extract single 1-D score per time series
        scores = [s[0] for s in scores_grouped]
        data = zip(fnames, labels, scores)
    elif 'syn_' in dataset:
        data = load_synthetic(dataset=dataset, testing=testing, iterator=True)
    else:
        raise ValueError(f"Wrong argument for dataset: {dataset}")

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
                                existence, conf_matrix, thresholds, zita, curr_save_path)

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
                #         data   = [df[c].dropna().values for c in time_cols]
                #         order  = sorted(range(len(data)), key=lambda i: -data[i].mean())
                #         fig, ax = plt.subplots(figsize=(10, 5))
                #         ax.boxplot(
                #             [data[i] for i in order],
                #             tick_labels=[labels[i] for i in order],
                #             vert=True,
                #             patch_artist=True,
                #         )
                #         ax.set_ylabel("Time (s)")
                #         ax.set_title(f"FF-VUS runtime breakdown — {metric} / slope={slope_size}")
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

    args = parser.parse_args()

    if args.dataset == 'all_synthetic':
        synthetic_dir = os.path.join('data', 'synthetic')
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
        )
