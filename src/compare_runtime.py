"""Compare detector inference runtimes vs VUS/FF-VUS/FF-VUS-GPU metric computation times."""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load detector inference times ---
det_path = os.path.join(BASE, "data", "results_from_MSAD", "detectors_inference_time.csv")
det_df = pd.read_csv(det_path)
det_df["ts_key"] = det_df["dataset"] + "/" + det_df["filename"]
detector_cols = [c for c in det_df.columns if c not in ("dataset", "filename", "ts_key")]
det_df["det_mean"] = det_df[detector_cols].mean(axis=1)
det_df["det_median"] = det_df[detector_cols].median(axis=1)
det_df["det_min"] = det_df[detector_cols].min(axis=1)
det_df["det_max"] = det_df[detector_cols].max(axis=1)

# --- Load metric computation times ---
results_dir = os.path.join(BASE, "experiments", "allmetrics_defparams_tsb", "results")

def load_metric(filename, metric_label):
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    return df[["Time series", "Metric time"]].rename(columns={
        "Time series": "ts_key",
        "Metric time": metric_label
    })

vus = load_metric("tsb_VUS_128.csv", "VUS_time")
ffvus = load_metric(
    "tsb_FF-VUS_128_1_dynamic_plus_globalmask_precomputed_optimized.csv", "FFVUS_time"
)
ffvus_gpu = load_metric(
    "tsb_FF-VUS-GPU_128_1_dynamic_plus_globalmask_precomputed_optimized.csv", "FFVUS_GPU_time"
)

# --- Merge on common time series ---
merged = det_df[["ts_key", "det_mean", "det_median", "det_min", "det_max"]].copy()
for mdf in [vus, ffvus, ffvus_gpu]:
    merged = merged.merge(mdf, on="ts_key", how="inner")

n = len(merged)
print(f"Matched time series: {n}")
print()

# --- Summary statistics ---
for metric_label, col in [("VUS", "VUS_time"), ("FF-VUS", "FFVUS_time"), ("FF-VUS-GPU", "FFVUS_GPU_time")]:
    times = merged[col]
    ratio_mean = merged["det_mean"] / times
    ratio_median = merged["det_median"] / times
    ratio_min = merged["det_min"] / times

    print(f"=== {metric_label} ===")
    print(f"  Metric time:  mean={times.mean():.4f}s, median={times.median():.4f}s, "
          f"min={times.min():.4f}s, max={times.max():.4f}s")
    print(f"  Ratio (mean det / metric):  mean={ratio_mean.mean():.1f}x, median={ratio_mean.median():.1f}x")
    print(f"  Ratio (median det / metric): mean={ratio_median.mean():.1f}x, median={ratio_median.median():.1f}x")
    print(f"  Ratio (fastest det / metric): mean={ratio_min.mean():.1f}x, median={ratio_min.median():.1f}x")

    # How often is the metric slower than the fastest detector?
    slower_than_fastest = (times > merged["det_min"]).sum()
    slower_than_median = (times > merged["det_median"]).sum()
    print(f"  Metric slower than fastest detector: {slower_than_fastest}/{n} ({100*slower_than_fastest/n:.1f}%)")
    print(f"  Metric slower than median detector:  {slower_than_median}/{n} ({100*slower_than_median/n:.1f}%)")
    print()

# --- Per-detector-family breakdown ---
print("=== Per-detector breakdown (mean inference time across series) ===")
print(f"{'Detector':<15} {'Mean time (s)':>13}")
for col in detector_cols:
    print(f"  {col:<13} {det_df[col].mean():>13.2f}")

print()
print(f"  {'VUS':<13} {merged['VUS_time'].mean():>13.4f}")
print(f"  {'FF-VUS':<13} {merged['FFVUS_time'].mean():>13.4f}")
print(f"  {'FF-VUS-GPU':<13} {merged['FFVUS_GPU_time'].mean():>13.4f}")

# --- Table for the rebuttal ---
print()
print("=== Rebuttal-ready summary table ===")
print("| Metric/Detector | Mean time (s) | Relative to VUS |")
print("|-----------------|--------------|-----------------|")
vus_mean = merged["VUS_time"].mean()
for label, col in [("VUS", "VUS_time"), ("FF-VUS", "FFVUS_time"), ("FF-VUS-GPU", "FFVUS_GPU_time")]:
    t = merged[col].mean()
    print(f"| {label:<15} | {t:>12.4f} | {t/vus_mean:>15.2f}x |")
print("|-----------------|--------------|-----------------|")
for col in detector_cols:
    t = merged.merge(det_df[["ts_key", col]], on="ts_key")[col].mean()
    print(f"| {col:<15} | {t:>12.2f} | {t/vus_mean:>15.1f}x |")
