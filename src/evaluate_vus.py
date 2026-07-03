"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""

import functools
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import argparse
from sklearn import metrics as sk_metrics
from tqdm import tqdm

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from legacy.sylli_metrics import sylli_get_metrics
from vus_debug import compare_vus_steps


# ---------------------------------------------------------------------------
# Local helpers (moved here from src/utils/ which is being deleted)
# ---------------------------------------------------------------------------

def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start_time
    return wrapper


def auc_pr_wrapper(label, score, interpolation='stepwise'):
    if interpolation == 'stepwise':
        return sk_metrics.average_precision_score(label, score)
    elif interpolation == 'linear':
        pr, rec, _ = sk_metrics.precision_recall_curve(label, score)
        return sk_metrics.auc(rec, pr)
    else:
        raise ValueError(f"Unknown argument for interpolation: {interpolation}")


def load_scores_aligned(tsb, fnames, timeseries, labels, detector):
    """Like TSBADParser.load_scores_and_delete but tolerant of score files whose
    length does not match their time series.

    load_scores_and_delete asserts that every score length equals its time
    series length and crashes otherwise (e.g. SAND emits a length-500 score for
    a length-1000 series). Here we instead drop those mismatched entries so the
    evaluation can continue. fnames/timeseries/labels are mutated in-place to
    stay aligned with the returned scores.
    """
    scores, problematic_files = tsb.load_scores(
        detectors=[detector], filenames=fnames, drop_missing=True,
    )
    if problematic_files:
        problematic_csv = [name.replace(".npy", ".csv") for name in problematic_files]
        indexes = [idx for idx, name in enumerate(fnames) if name in problematic_csv]
        tsb.delete_indexes_from_lists(fnames, timeseries, labels, indexes=indexes)

    # Drop any remaining entries whose score length disagrees with the series.
    bad = [
        i for i, (t, l, s) in enumerate(zip(timeseries, labels, scores))
        if not (t.shape[0] == l.shape[0] == s.shape[1])
    ]
    if bad:
        bad_names = [fnames[i] for i in bad]
        print(f"[{detector}] Dropping {len(bad)} score(s) with mismatched length: {bad_names}")
        tsb.delete_indexes_from_lists(fnames, timeseries, labels, scores, indexes=bad)

    return scores, problematic_files


def save_results(experiment_dir, results, problematic_timeseries, info_meta):
    """Write the collected results and run info to experiments/<experiment_dir>/.

    Called incrementally (after every detector) and from a finally block, so a
    crash, OOM-kill, or Ctrl-C still leaves the results gathered so far on disk.
    The filename encodes the running error stats, which change as more rows are
    added, so stale CSVs from earlier incremental writes are cleared first.
    """
    if not results:
        return

    saving_path  = os.path.join('experiments', experiment_dir)
    results_path = os.path.join(saving_path, 'results')
    info_path    = os.path.join(saving_path, 'info')
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(info_path,    exist_ok=True)

    df = pd.DataFrame(results)
    vus_mean_error     = df["VUS-PR - FF-VUS-PR"].mean()
    vus_max_error      = df["VUS-PR - FF-VUS-PR"].max()
    vus_gpu_mean_error = df["VUS-PR - FF-VUS-PR-GPU"].mean()
    vus_gpu_max_error  = df["VUS-PR - FF-VUS-PR-GPU"].max()
    exp_name = (f"evaluate_vusmean{vus_mean_error:.2e}_vusmax{vus_max_error:.2e}"
                f"_vusgpumean{vus_gpu_mean_error:.2e}_vusgpumax{vus_gpu_max_error:.2e}.csv")

    for directory in (results_path, info_path):
        for stale in glob.glob(os.path.join(directory, "*.csv")):
            os.remove(stale)

    df.to_csv(os.path.join(results_path, exp_name), index=False)

    info_df = pd.DataFrame([info_meta])
    info_df["Problematic time series"] = [problematic_timeseries]
    info_df.to_csv(os.path.join(info_path, exp_name), index=False)


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


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_vus(testing, experiment_dir=None, detector_name=None):
    tic = time.time()

    tsb = TSBADParser()
    all_detectors = tsb.get_detectors()

    # Validate detector name and build the list to iterate over
    if detector_name is not None and detector_name not in all_detectors:
        raise ValueError(
            f"Unknown detector '{detector_name}'. "
            f"Available: {all_detectors}"
        )
    detectors_to_run = [detector_name] if detector_name is not None else all_detectors

    # Load time series
    if testing:
        fnames, timeseries, labels = tsb.load_timeseries(dataset=['NAB'])
        fnames, timeseries, labels = fnames[:10], timeseries[:10], labels[:10]
    else:
        fnames, timeseries, labels = tsb.load_timeseries(load_all=True)
        sampled_idx = np.random.choice(len(fnames), size=min(100, len(fnames)), replace=False)
        fnames     = [fnames[i]     for i in sampled_idx]
        timeseries = [timeseries[i] for i in sampled_idx]
        labels     = [labels[i]     for i in sampled_idx]
        # load_scores aligns by filename (groupby alphabetical), so keep the same order
        sorted_order = sorted(range(len(fnames)), key=lambda i: fnames[i])
        fnames     = [fnames[i]     for i in sorted_order]
        timeseries = [timeseries[i] for i in sorted_order]
        labels     = [labels[i]     for i in sorted_order]

    slope_size = 128
    step = 1
    zita = (1 / math.sqrt(2))
    global_mask = True
    slopes = 'function'
    existence = 'optimized'
    conf_matrix = 'dynamic_plus'
    interpolation = 'stepwise'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    thresholds='subsampled'
    max_thresholds = 250

    ff_vus_numpy = VUSNumpy(
        slope_size=slope_size,
        step=step,
        zita=zita,
        global_mask=global_mask,
        slopes=slopes,
        existence=existence,
        conf_matrix=conf_matrix,
        interpolation=interpolation,
        thresholds=thresholds,
        max_thresholds=max_thresholds,
    )

    ff_auc_numpy = VUSNumpy(
        slope_size=0,
        step=1,
        zita=zita,
        global_mask=global_mask,
        slopes=slopes,
        existence='None',
        conf_matrix=conf_matrix,
        interpolation=interpolation,
        max_thresholds=1000,
    )

    ff_vus_torch = VUSTorch(
        slope_size=slope_size,
        step=1,
        zita=zita,
        global_mask=global_mask,
        conf_matrix=conf_matrix,
        device=device,
        thresholds=thresholds,
        max_thresholds=max_thresholds,
    )

    ff_auc_torch = VUSTorch(
        slope_size=0,
        step=1,
        zita=zita,
        global_mask=global_mask,
        existence=False,
        conf_matrix=conf_matrix,
        device=device,
        max_thresholds=1000,
    )
    print(f">> Settings: slope_size={slope_size}, step={step}, slopes={slopes}, "
          f"existence={existence}, conf_matrix={conf_matrix}, "
          f"interpolation={interpolation}, global_mask={global_mask}")
    print(f">> Detectors to run ({len(detectors_to_run)}): {detectors_to_run}")

    results = []
    problematic_timeseries = []

    # Pre-populate from a prior run so reruns expand rather than replace.
    if experiment_dir is not None:
        for f in glob.glob(os.path.join('experiments', experiment_dir, 'results', '*.csv')):
            results.extend(pd.read_csv(f).to_dict('records'))

    def _info_meta():
        return {
            "Experiment":       "Evaluate FF-VUS-PR",
            "Slope size":       slope_size,
            "Global mask":      global_mask,
            "Step":             step,
            "Slopes":           slopes,
            "Existence":        existence,
            "Confusion matrix": conf_matrix,
            "Time":             time.time() - tic,
            "Detector":         detector_name if detector_name is not None else 'all',
        }

    # Phase 1: load all scores up-front before running any comparisons.
    print(f"Loading scores for {len(detectors_to_run)} detector(s)...")
    all_detector_data = {}
    for detector in tqdm(detectors_to_run, desc='Loading scores', disable=len(detectors_to_run) == 1):
        fnames_det = list(fnames)
        ts_det     = list(timeseries)
        labels_det = list(labels)
        scores, _  = load_scores_aligned(tsb, fnames_det, ts_det, labels_det, detector)
        if not scores:
            print(f"[{detector}] No valid scores found, skipping.")
            continue
        all_detector_data[detector] = (fnames_det, ts_det, labels_det, scores)

    # Phase 2: run comparisons.
    try:
      for detector in tqdm(list(all_detector_data), desc='Detectors', disable=len(all_detector_data) == 1):
        fnames_det, ts_det, labels_det, scores = all_detector_data[detector]

        for i, (filename, label, curr_scores) in enumerate(zip(fnames_det, labels_det, scores)):
            score = curr_scores[0, :]  # (1, T) -> (T,)

            length, n_anomalies, anomalies_avg_length = _analyze_label(label)
            curr_result = {
                'Time series': filename,
                'Detector': detector,
                'Length': length,
                'Number of anomalies': n_anomalies,
                'Anomalies average length': float(anomalies_avg_length),
            }

            # AUC-PR
            auc_pr, auc_pr_time = time_it(auc_pr_wrapper)(label, score, interpolation)
            curr_result.update({'AUC-PR': auc_pr, 'AUC-PR time': auc_pr_time})

            # FF-AUC-PR
            (ff_auc_pr, ff_auc_time_analysis), ff_auc_time = ff_auc_numpy.compute(label, score)
            curr_result.update({'FF-AUC-PR': ff_auc_pr, 'FF-AUC-PR time': ff_auc_time})
            curr_result.update(ff_auc_time_analysis)

            # VUS-PR (reference)
            vus_pr, vus_time = sylli_get_metrics(label, score, 'vus', slope_size, max_thresholds=max_thresholds)
            curr_result.update({'VUS-PR': vus_pr, 'VUS-PR time': vus_time})

            # FF-VUS-PR
            (ff_vus_pr, ff_vus_time_analysis), ff_vus_time = ff_vus_numpy.compute(label, score)
            curr_result.update({'FF-VUS-PR': ff_vus_pr, 'FF-VUS-PR time': ff_vus_time})
            curr_result.update(ff_vus_time_analysis)

            # GPU variants — convert to tensors for torch compute
            label_t = torch.tensor(label, device=device)
            score_t = torch.tensor(score, device=device)

            (ff_vus_pr_gpu, ff_vus_gpu_time_analysis), ff_vus_gpu_time = ff_vus_torch.compute(label_t, score_t)
            curr_result.update({'FF-VUS-PR-GPU': ff_vus_pr_gpu.item(), 'FF-VUS-PR-GPU time': ff_vus_gpu_time})
            curr_result.update(ff_vus_gpu_time_analysis)

            (ff_auc_pr_gpu, ff_auc_gpu_time_analysis), ff_auc_gpu_time = ff_auc_torch.compute(label_t, score_t)
            curr_result.update({'FF-AUC-PR-GPU': ff_auc_pr_gpu.item(), 'FF-AUC-PR-GPU time': ff_auc_gpu_time})
            curr_result.update(ff_auc_gpu_time_analysis)

            tqdm.write(
                f"[{detector}/{i}] "
                f"ΔAUC: {abs(auc_pr - ff_auc_pr):.2e} | "
                f"ΔAUC-GPU: {abs(auc_pr - ff_auc_pr_gpu):.2e} | "
                f"ΔVUS: {abs(vus_pr - ff_vus_pr):.2e} | "
                f"ΔVUS-GPU: {abs(vus_pr - ff_vus_pr_gpu):.2e} | "
                f"VUS/{vus_time / ff_vus_time:.1f}x | "
                f"Len:{label.shape[0]} | "
                f"#Anom:{n_anomalies}"
            )

            curr_result.update({
                "AUC-PR - FF-AUC-PR":     abs(auc_pr - ff_auc_pr),
                "VUS-PR - FF-VUS-PR":     abs(vus_pr - ff_vus_pr),
                "AUC-PR Slow down":       ff_auc_time / auc_pr_time,
                "VUS-PR Speed up":        vus_time / ff_vus_time,
                "AUC-PR equal":           abs(auc_pr - ff_auc_pr) < 1e-10,
                "VUS-PR equal":           abs(vus_pr - ff_vus_pr) < 1e-10,
                "VUS-PR - FF-VUS-PR-GPU": abs(vus_pr - ff_vus_pr_gpu).item(),
                "AUC-PR - FF-AUC-PR-GPU": abs(auc_pr - ff_auc_pr_gpu).item(),
                "AUC-PR-GPU Slow down":   ff_auc_gpu_time / auc_pr_time,
                "VUS-PR-GPU Speed up":    vus_time / ff_vus_gpu_time,
                "AUC-PR-GPU equal":       abs(auc_pr - ff_auc_pr_gpu).item() < 1e-10,
                "VUS-PR-GPU equal":       abs(vus_pr - ff_vus_pr_gpu).item() < 1e-10,
            })
            if not curr_result["VUS-PR equal"]:
                problematic_timeseries.append((filename, detector))

            # Extra analysis when error is high (Abandoned idea)
            # cpu_err = curr_result["VUS-PR - FF-VUS-PR"]
            # gpu_err = curr_result["VUS-PR - FF-VUS-PR-GPU"]
            # if cpu_err > 1e-10 or gpu_err > 1e-4:
            #     debug_info = compare_vus_steps(label, score, ff_vus_numpy, ff_vus_torch)
            #     curr_result.update(debug_info)

            results.append(curr_result)

        # Persist progress after every detector so a crash, OOM-kill, or Ctrl-C
        # still leaves the results gathered so far on disk.
        if experiment_dir is not None:
            save_results(experiment_dir, results, problematic_timeseries, _info_meta())
    finally:
        if experiment_dir is not None:
            save_results(experiment_dir, results, problematic_timeseries, _info_meta())

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    print(df[['Time series', 'VUS-PR', 'FF-VUS-PR', 'FF-VUS-PR-GPU']])
    print(f"AUC-PR - FF-AUC-PR: average dif.: {df['AUC-PR - FF-AUC-PR'].mean()}, "
          f"max dif.: {df['AUC-PR - FF-AUC-PR'].max()}, "
          f"avg slow down: {df['AUC-PR Slow down'].mean()}")
    print(f"VUS-PR - FF-VUS-PR: average dif.: {df['VUS-PR - FF-VUS-PR'].mean()}, "
          f"max dif.: {df['VUS-PR - FF-VUS-PR'].max()}, "
          f"avg speed up: {df['VUS-PR Speed up'].mean()}")
    print(f"AUC-PR - FF-AUC-PR-GPU: average dif.: {df['AUC-PR - FF-AUC-PR-GPU'].mean()}, "
          f"max dif.: {df['AUC-PR - FF-AUC-PR-GPU'].max()}, "
          f"avg slow down: {df['AUC-PR-GPU Slow down'].mean()}")
    print(f"VUS-PR - FF-VUS-PR-GPU: average dif.: {df['VUS-PR - FF-VUS-PR-GPU'].mean()}, "
          f"max dif.: {df['VUS-PR - FF-VUS-PR-GPU'].max()}, "
          f"avg speed up: {df['VUS-PR-GPU Speed up'].mean()}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog='main',
        description='Run experiments on the current implementation of the VUS metric'
    )
    arg_parser.add_argument('--test',       action='store_true', help='Run in testing mode (limits the data for fast testing)')
    arg_parser.add_argument('--experiment', type=str,  default=None, help='Directory to save experiment results (optional)')
    arg_parser.add_argument('--detector',   type=str,  default=None, help='Detector name to evaluate (e.g. IForest). Omit to evaluate all detectors.')

    args = arg_parser.parse_args()

    evaluate_vus(
        testing=args.test,
        experiment_dir=args.experiment,
        detector_name=args.detector,
    )
