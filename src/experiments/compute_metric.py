"""Compute one evaluation measure over a dataset and save one row per (series, detector).

This is the data source of the accuracy and speed experiments (C1, C2):
run it once per measure on TSB-AD (all detectors) and on the synthetic sweeps.

    conda run -n autovus python src/experiments/compute_metric.py --dataset tsb --metric fastvus --detector all --experiment all_metrics_tsb
    conda run -n autovus python src/experiments/compute_metric.py --dataset syn_0 --metric fastvus_gpu --experiment scaling_syn

Measures: fastvus, fastvus_gpu, adjustedvus, adjustedvus_gpu, autovus,
autovus_gpu, autovus_per_side, autovus_gpu_per_side (ours), and the baselines auc, vus, range_auc,
rf, affiliation, pate (src/baselines). Upstream vus takes windowSize = 2L.
Datasets: 'tsb' (TSB-AD univariate, via tsad_parser), 'syn_<k>' or
'all_synthetic' (label + score .npz files under <benchmarks>/_synthetic/).
Never run the baselines on the synthetic length sweep (hours to months).
Results: experiments/<experiment>/results/<dataset>_<METRIC>_<L>_<step>.csv,
appended row by row, plus a machine/info file under experiments/<experiment>/info/.
"""
import argparse
import os
import re
import random
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tsad_parser import TSADParser

from autovus import FastVUS, FastVUSGPU, AdjustedVUS, AutoVUS
from baselines import compute as baseline_compute, METRICS as BASELINES

AUTO = ('autovus', 'autovus_gpu', 'autovus_per_side', 'autovus_gpu_per_side')   # sides: symmetric, per_side
ADJ = ('adjustedvus', 'adjustedvus_gpu')
OURS = ('fastvus', 'fastvus_gpu') + ADJ + AUTO
NEEDS_L = OURS + ('vus', 'range_auc', 'pate')
TORCH_SORT_MAX = 2 ** 31 - 1     # torch.sort limit; the longest synthetic series is cut to it


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def label_stats(label):
    """Length, number of anomalies, mean anomaly length."""
    starts = np.where(np.diff(np.r_[0, label, 0]) == 1)[0]
    ends = np.where(np.diff(np.r_[0, label, 0]) == -1)[0]
    n = len(starts)
    return len(label), n, float(np.mean(ends - starts)) if n else 0.0


def load_synthetic(dataset, testing=False):
    """Yield (name, label, score) from <benchmarks>/_synthetic/<dataset>/."""
    folder = os.path.join(TSADParser.benchmarks_root(), '_synthetic', dataset)
    files = sorted(os.listdir(folder), key=natural_keys)
    if testing:
        files = files[:1]
    for f in files:
        data = np.load(os.path.join(folder, f))
        label, score = data['label'], data['score'].round(2)
        if len(score) > TORCH_SORT_MAX:
            label, score = label[:TORCH_SORT_MAX], score[:TORCH_SORT_MAX]
        yield f, label, score, None


def load_tsb(detector, testing=False):
    """Yield (name, label, score, detector) from TSB-AD."""
    tsb = TSADParser()
    if testing:
        fnames, series, labels = tsb.load_timeseries(dataset=['NAB'])
        fnames, series, labels = fnames[:10], series[:10], labels[:10]
    else:
        fnames, series, labels = tsb.load_timeseries(load_all=True)
    detectors = tsb.get_detectors()
    if detector is None:
        selected = [detectors[0]]
    elif detector in ('all', 'random'):
        selected = detectors
    elif detector in detectors:
        selected = [detector]
    else:
        raise ValueError(f"Unknown detector {detector!r}. Available: {detectors}")
    print(f"Detectors: {'one random per series' if detector == 'random' else selected}")
    scores, _ = tsb.load_scores_and_delete(fnames, series, labels, detectors=selected)
    for fname, label, stack in zip(fnames, labels, scores):
        label = np.asarray(label).astype(np.int8)
        if detector == 'random':
            i = random.randrange(len(selected))
            yield fname, label, stack[i], selected[i]
        else:
            for det, score in zip(selected, stack):
                yield fname, label, score, det


# ---------------------------------------------------------------------------
# One measure over the data
# ---------------------------------------------------------------------------
def make_ours(metric, L, step, global_mask, n_thresholds):
    kwargs = dict(L=L, step=step, global_mask=global_mask)
    if n_thresholds is not None:
        kwargs['max_thresholds'] = n_thresholds
    if metric in ADJ + AUTO:
        kwargs['cumulative'] = True
    if 'gpu' in metric:
        engine = FastVUSGPU(**kwargs, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        engine = FastVUS(**kwargs)
    if metric in AUTO:
        return AutoVUS(engine, sides='per_side' if metric.endswith('per_side') else 'symmetric')
    if metric in ADJ:
        return AdjustedVUS(engine)
    return engine


def compute_metric(metric, data, L, step, global_mask, n_thresholds, save_path=None):
    ours = make_ours(metric, L, step, global_mask, n_thresholds) if metric in OURS else None
    rows = []
    for name, label, score, detector in tqdm(data, desc=f'{metric}'):
        score = np.asarray(score, dtype=np.float64)
        if len(label) != len(score):                      # a few TSB-AD scores are one point short
            n = min(len(label), len(score))
            label, score = label[:n], score[:n]
        length, n_anomalies, mean_len = label_stats(label)
        row = {'Time series': name, 'Detector': detector, 'Length': length,
               'Number of anomalies': n_anomalies, 'Anomalies average length': mean_len,
               'Metric': metric, 'L': L if metric in NEEDS_L else None, 'Step': step if metric in OURS else None,
               'Global mask': global_mask if metric in OURS else None}
        if ours is not None:
            t0 = time.perf_counter()
            value = ours.compute(label, score)
            if metric in ADJ + AUTO:
                value = value['adjusted']
            if 'gpu' in metric:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            seconds = time.perf_counter() - t0
            row['n_thresholds'] = None
            if metric not in AUTO:                        # AutoVUS calls the engine many times; only the total time is kept
                row.update({f'time {k}': v for k, v in getattr(ours, 'engine', ours).timings.items()})
        else:
            kwargs = {'thre': n_thresholds} if (n_thresholds is not None and metric in ('vus', 'range_auc')) else {}
            value, seconds = baseline_compute(metric, label, score, L=L, **kwargs)
        row.update({'Metric value': float(value), 'Metric time': seconds})
        if save_path is not None:
            pd.DataFrame([row]).to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        rows.append(row)
    return pd.DataFrame(rows)


def machine_info():
    return {
        'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'CPU': os.popen("lscpu | grep 'Model name' | awk -F ':' '{print $2}'").read().strip(),
        'Cores': os.cpu_count(),
        'RAM (GB)': round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 ** 3, 1),
        'Python': sys.version.split()[0], 'torch': torch.__version__,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, help="'tsb', 'syn_<k>' or 'all_synthetic'")
    p.add_argument('--metric', required=True, choices=OURS + BASELINES + ('all',), help="'all' = every measure except pate")
    p.add_argument('--L', type=int, default=128, help='buffer length per side (default 128)')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--no_global_mask', action='store_true')
    p.add_argument('--n_thresholds', type=int, default=None, help='threshold cap for fastvus/fastvus_gpu (default 2000) and vus/range_auc (default 250)')
    p.add_argument('--detector', default=None, help="TSB-AD detector name, 'all', or 'random' (default: the first detector)")
    p.add_argument('--test', action='store_true', help='10 NAB series or 1 synthetic file')
    p.add_argument('--experiment', default=None, help='save under experiments/<experiment>/')
    a = p.parse_args()

    metrics = ['fastvus', 'fastvus_gpu', 'auc', 'vus', 'range_auc', 'rf', 'affiliation'] if a.metric == 'all' else [a.metric]
    if a.dataset == 'all_synthetic':
        datasets = sorted(os.listdir(os.path.join(TSADParser.benchmarks_root(), '_synthetic')), key=natural_keys)
    else:
        datasets = [a.dataset]
    if any(m not in OURS for m in metrics) and any(d != 'tsb' for d in datasets):
        raise SystemExit('baselines on the synthetic sweep are not allowed (hours to months per file)')

    for dataset in datasets:
        for metric in metrics:
            data = list(load_tsb(a.detector, a.test) if dataset == 'tsb' else load_synthetic(dataset, a.test))
            save_path = info_path = None
            if a.experiment is not None:
                root = os.path.join('experiments', a.experiment)
                os.makedirs(os.path.join(root, 'results'), exist_ok=True)
                os.makedirs(os.path.join(root, 'info'), exist_ok=True)
                tag = f"{dataset}_{metric}" + (f"_L{a.L}" if metric in NEEDS_L else '') + (f"_step{a.step}" if metric in OURS else '') \
                    + ('_nomask' if (a.no_global_mask and metric in OURS) else '') + ('_test' if a.test else '')
                save_path = os.path.join(root, 'results', tag + '.csv')
                info_path = os.path.join(root, 'info', tag + '.csv')
                if os.path.exists(save_path):
                    os.remove(save_path)
            df = compute_metric(metric, data, a.L, a.step, not a.no_global_mask, a.n_thresholds, save_path)
            print(df[['Time series', 'Detector', 'Length', 'Number of anomalies', 'Metric value', 'Metric time']].to_string(max_rows=20))
            print(f"{metric} on {dataset}: {len(df)} rows, total {df['Metric time'].sum():.1f} s, mean {df['Metric time'].mean():.3f} s")
            if info_path is not None:
                pd.DataFrame([{**machine_info(), 'dataset': dataset, 'metric': metric, 'L': a.L, 'step': a.step,
                               'global_mask': not a.no_global_mask, 'n_thresholds': a.n_thresholds, 'detector': a.detector,
                               'rows': len(df), 'total_time_s': df['Metric time'].sum()}]).to_csv(info_path, index=False)


if __name__ == '__main__':
    main()
