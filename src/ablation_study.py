"""
Ablation study for FF-VUS optimizations.

Incrementally enables each optimization and measures its effect on overall runtime.
Runs on the full TSB-UAD benchmark with a single detector.

Usage:
    python src/ablation_study.py                    # Full TSB-UAD
    python src/ablation_study.py --test              # Quick test (10 time series)
    python src/ablation_study.py --test --cpu_only   # CPU-only quick test
    python src/ablation_study.py --test --gpu_only   # GPU-only quick test
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from legacy.sylli_metrics import sylli_get_metrics


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
    lengths = (ends + 1) - starts
    return len(label), n, float(np.mean(lengths))


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# Each config is a dict of constructor kwargs.
# They are ordered so that each step adds exactly one optimization on top of
# the previous one.

CPU_CONFIGS = [
    {
        "name": "Baseline (no optimizations)",
        "kwargs": {
            "global_mask": False,
            "slopes": "function",
            "existence": "trivial",
            "conf_matrix": "trivial",
        },
    },
    {
        "name": "+ Precomputed slopes",
        "kwargs": {
            "global_mask": False,
            "slopes": "precomputed",
            "existence": "trivial",
            "conf_matrix": "trivial",
        },
    },
    {
        "name": "+ Optimized existence",
        "kwargs": {
            "global_mask": False,
            "slopes": "precomputed",
            "existence": "optimized",
            "conf_matrix": "trivial",
        },
    },
    {
        "name": "+ Dynamic confusion matrix",
        "kwargs": {
            "global_mask": False,
            "slopes": "precomputed",
            "existence": "optimized",
            "conf_matrix": "dynamic",
        },
    },
    {
        "name": "+ Dynamic+ confusion matrix",
        "kwargs": {
            "global_mask": False,
            "slopes": "precomputed",
            "existence": "optimized",
            "conf_matrix": "dynamic_plus",
        },
    },
    {
        "name": "+ Global mask (FF-VUS)",
        "kwargs": {
            "global_mask": True,
            "slopes": "precomputed",
            "existence": "optimized",
            "conf_matrix": "dynamic_plus",
        },
    },
]

GPU_CONFIGS = [
    {
        "name": "Baseline (no optimizations)",
        "kwargs": {
            "global_mask": False,
            "existence": False,
            "conf_matrix": "dynamic",
        },
    },
    {
        "name": "+ Existence",
        "kwargs": {
            "global_mask": False,
            "existence": True,
            "conf_matrix": "dynamic",
        },
    },
    {
        "name": "+ Dynamic+ confusion matrix",
        "kwargs": {
            "global_mask": False,
            "existence": True,
            "conf_matrix": "dynamic_plus",
        },
    },
    {
        "name": "+ Global mask (FF-VUS-GPU step=1)",
        "kwargs": {
            "global_mask": True,
            "existence": True,
            "conf_matrix": "dynamic_plus",
        },
    },
]


def build_gpu_step_configs(slope_size):
    """Generate additional GPU configs that vary the step parameter."""
    steps = []
    step = 2
    while step <= slope_size:
        if slope_size % step == 0:
            steps.append(step)
        step *= 2

    configs = []
    for s in steps:
        configs.append({
            "name": f"+ Fewer buffers (step={s})",
            "kwargs": {
                "global_mask": True,
                "existence": True,
                "conf_matrix": "dynamic_plus",
            },
            "step": s,
        })
    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_original_vus(filenames, labels, scores, slope_size, config_name="Original VUS"):
    """Run the original (unoptimized) VUS implementation over the full dataset."""
    rows = []

    for filename, label, score in tqdm(
        zip(filenames, labels, scores),
        total=len(filenames),
        desc=config_name,
        leave=False,
    ):
        length, n_anomalies, anomalies_avg_length = _analyze_label(label)

        if label.shape != score.shape:
            min_len = min(label.shape[0], score.shape[0])
            label = label[:min_len]
            score = score[:min_len]

        metric_value, metric_time = sylli_get_metrics(label, score, 'vus', slope_size, existence=True)

        row = {
            "Config": config_name,
            "Backend": "CPU",
            "Time series": filename,
            "Length": length,
            "N anomalies": n_anomalies,
            "Avg anomaly length": float(anomalies_avg_length),
            "Metric value": float(metric_value),
            "Total time (s)": metric_time,
        }
        rows.append(row)

    return rows


def run_single_config(vus_obj, filenames, labels, scores, backend, config_name):
    """Run a VUS configuration over the full dataset and return per-ts results."""
    is_gpu = backend == "gpu"
    device = vus_obj.device if is_gpu else "cpu"
    rows = []

    for filename, label, score in tqdm(
        zip(filenames, labels, scores),
        total=len(filenames),
        desc=config_name,
        leave=False,
    ):
        length, n_anomalies, anomalies_avg_length = _analyze_label(label)

        if label.shape != score.shape:
            min_len = min(label.shape[0], score.shape[0])
            label = label[:min_len]
            score = score[:min_len]

        with torch.no_grad():
            if is_gpu:
                label_t = torch.tensor(label, device=device, dtype=torch.uint8)
                score_t = torch.tensor(score, device=device, dtype=torch.float16)
                (metric_value, time_analysis), metric_time = vus_obj.compute(label_t, score_t)
            else:
                (metric_value, time_analysis), metric_time = vus_obj.compute(label, score)

        row = {
            "Config": config_name,
            "Backend": backend.upper(),
            "Time series": filename,
            "Length": length,
            "N anomalies": n_anomalies,
            "Avg anomaly length": float(anomalies_avg_length),
            "Metric value": float(metric_value),
            "Total time (s)": metric_time,
        }
        row.update({k: v for k, v in time_analysis.items()})
        rows.append(row)

    return rows


def run_ablation(filenames, labels, scores, slope_size, step_default, run_cpu, run_gpu, save_dir):
    """Run the full ablation study."""
    all_rows = []

    # ---- Original VUS (unoptimized reference) ----
    if run_cpu:
        print("\n" + "=" * 70)
        print("ORIGINAL VUS (unoptimized reference)")
        print("=" * 70)

        rows = run_original_vus(filenames, labels, scores, slope_size, "Original VUS")
        all_rows.extend(rows)
        total_time = sum(r["Total time (s)"] for r in rows)
        print(f"  {'Original VUS':45s}  total={total_time:10.2f}s  "
              f"avg={total_time / len(rows):8.4f}s")

    # ---- CPU ablation ----
    if run_cpu:
        print("\n" + "=" * 70)
        print("CPU ABLATION")
        print("=" * 70)

        for cfg in CPU_CONFIGS:
            kwargs = {**cfg["kwargs"], "slope_size": slope_size, "step": step_default}
            vus = VUSNumpy(**kwargs)
            rows = run_single_config(vus, filenames, labels, scores, "cpu", cfg["name"])
            all_rows.extend(rows)

            total_time = sum(r["Total time (s)"] for r in rows)
            print(f"  {cfg['name']:45s}  total={total_time:10.2f}s  "
                  f"avg={total_time / len(rows):8.4f}s")

    # ---- GPU ablation ----
    if run_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("\nWARNING: No CUDA device found - GPU ablation will run on CPU.")

        print("\n" + "=" * 70)
        print(f"GPU ABLATION (device={device})")
        print("=" * 70)

        gpu_all_configs = GPU_CONFIGS + build_gpu_step_configs(slope_size)

        for cfg in gpu_all_configs:
            step = cfg.get("step", step_default)
            kwargs = {**cfg["kwargs"], "slope_size": slope_size, "step": step, "device": device}
            vus = VUSTorch(**kwargs)
            rows = run_single_config(vus, filenames, labels, scores, "gpu", cfg["name"])
            all_rows.extend(rows)

            total_time = sum(r["Total time (s)"] for r in rows)
            print(f"  {cfg['name']:45s}  total={total_time:10.2f}s  "
                  f"avg={total_time / len(rows):8.4f}s")

    # ---- Save results ----
    df = pd.DataFrame(all_rows)
    os.makedirs(save_dir, exist_ok=True)

    results_path = os.path.join(save_dir, "results")
    info_path = os.path.join(save_dir, "info")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(info_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_path, f"ablation_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nPer-time-series results saved to {results_file}")

    # ---- Summary ----
    summary = (
        df.groupby(["Backend", "Config"])
        .agg(
            total_time=("Total time (s)", "sum"),
            mean_time=("Total time (s)", "mean"),
            mean_metric=("Metric value", "mean"),
            n_series=("Time series", "count"),
        )
        .reset_index()
    )

    # Compute speedup relative to Original VUS
    original_time = summary.loc[summary["Config"] == "Original VUS", "total_time"]
    original_time = original_time.values[0] if len(original_time) > 0 else None
    summary["speedup"] = summary.apply(
        lambda r: original_time / r["total_time"]
        if original_time is not None and r["total_time"] > 0 else float("nan"),
        axis=1,
    )
    summary = summary.sort_values(["Backend", "total_time"], ascending=[True, False])

    summary_file = os.path.join(results_path, f"ablation_summary_{timestamp}.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # ---- Print summary ----
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(f"{'Backend':<6} {'Config':<50} {'Total (s)':>10} {'Speedup':>8} {'Mean metric':>12}")
    print("-" * 100)
    for _, row in summary.iterrows():
        print(f"{row['Backend']:<6} {row['Config']:<50} {row['total_time']:>10.2f} "
              f"{row['speedup']:>7.1f}x {row['mean_metric']:>12.6f}")
    print("=" * 100)

    # ---- Save system info ----
    info = {
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "GPU Memory (GB)": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if torch.cuda.is_available() else None,
        "CPU": os.popen("lscpu | grep 'Model name' | awk -F ':' '{print $2}'").read().strip(),
        "Cores": os.cpu_count(),
        "RAM (GB)": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0 ** 3), 2)
            if hasattr(os, "sysconf") else None,
        "Slope size": slope_size,
        "Step (default)": step_default,
        "N time series": len(filenames),
        "Timestamp": timestamp,
    }
    info_df = pd.DataFrame([info])
    info_file = os.path.join(info_path, f"ablation_info_{timestamp}.csv")
    info_df.to_csv(info_file, index=False)
    print(f"System info saved to {info_file}")

    return df, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study for FF-VUS optimizations")
    parser.add_argument("--slope_size", type=int, default=128, help="Buffer size (default: 128)")
    parser.add_argument("--step", type=int, default=1, help="Default step size (default: 1)")
    parser.add_argument("--test", action="store_true", help="Quick test mode (10 time series from KDD21)")
    parser.add_argument("--cpu_only", action="store_true", help="Run only CPU ablation")
    parser.add_argument("--gpu_only", action="store_true", help="Run only GPU ablation")
    parser.add_argument("--experiment", type=str, default="ablation_study",
                        help="Experiment subdirectory under experiments/ (default: ablation_study)")
    args = parser.parse_args()

    # Ensure paths are relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    # Load data
    print("Loading TSB-AD dataset...")
    tsb = TSBADParser()
    if args.test:
        filenames, timeseries, labels = tsb.load_timeseries(dataset=['YAHOO'])
        filenames, timeseries, labels = filenames[:10], timeseries[:10], labels[:10]
    else:
        filenames, timeseries, labels = tsb.load_timeseries(load_all=True)
    detectors = tsb.get_detectors()
    scores_grouped, _ = tsb.load_scores_and_delete(filenames, timeseries, labels, detectors=detectors[:1])
    scores = [s[0] for s in scores_grouped]
    print(f"Loaded {len(filenames)} time series")

    save_dir = os.path.join("experiments", args.experiment)
    run_ablation(filenames, labels, scores, args.slope_size, args.step, run_cpu, run_gpu, save_dir)


if __name__ == "__main__":
    main()
