"""
Ablation study for FF-VUS optimizations.

Incrementally enables each optimization and measures its effect on overall runtime.
Runs on the full TSB-AD benchmark. By default it uses the first detector; use
--detector to pick another, 'all' detectors, or a 'random' detector per series.

Usage:
    python src/ablation_study.py                     # Full TSB-AD, first detector
    python src/ablation_study.py --detector all      # Every detector
    python src/ablation_study.py --test              # Quick test (10 time series)
    python src/ablation_study.py --test --cpu_only   # CPU-only quick test
    python src/ablation_study.py --test --gpu_only   # GPU-only quick test
"""

import argparse
import os
import random
import time
from datetime import datetime

import matplotlib
import seaborn as sns
matplotlib.use("Agg")  # headless: never pop up a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from baselines import compute as baseline_compute


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


# Number of thresholds used by both the CPU (VUSNumpy) and GPU (VUSTorch)
# implementations. Overrides their default of 2000 to keep the ablation
# comparable and fast.
MAX_THRESHOLDS = 250

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

        metric_value, metric_time = baseline_compute('vus', label, score, L=slope_size)

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


# ---------------------------------------------------------------------------
# Summary / analysis
# ---------------------------------------------------------------------------

def summarize_results(df):
    """Aggregate per-time-series results into a per-config summary with speedups."""
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
    return summary


def print_summary(summary):
    """Print the ablation summary table plus the per-step incremental analysis."""
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY")
    print("=" * 100)
    print(f"{'Backend':<6} {'Config':<50} {'Total (s)':>10} {'Speedup':>8} {'Mean metric':>12}")
    print("-" * 100)
    for _, row in summary.iterrows():
        print(f"{row['Backend']:<6} {row['Config']:<50} {row['total_time']:>10.2f} "
              f"{row['speedup']:>7.1f}x {row['mean_metric']:>12.6f}")
    print("=" * 100)

    # ---- Incremental analysis: marginal contribution of each optimization ----
    # Within each backend, configs are added one optimization at a time. Order
    # them fastest-last (slowest-first) and report the step-over-step speedup so
    # the contribution of each optimization is visible.
    print("\nINCREMENTAL CONTRIBUTION (step-over-step, within each backend)")
    print("-" * 100)
    for backend in summary["Backend"].unique():
        sub = summary[summary["Backend"] == backend].sort_values(
            "total_time", ascending=False).reset_index(drop=True)
        print(f"\n[{backend}]")
        prev_time = None
        for _, row in sub.iterrows():
            if prev_time is not None and row["total_time"] > 0:
                step = prev_time / row["total_time"]
                marker = f"x{step:5.2f} vs prev"
            else:
                marker = "     (reference)"
            print(f"  {row['Config']:<50} {row['total_time']:>10.2f}s  {marker}")
            prev_time = row["total_time"]
    print("=" * 100)


# The ablation is a single incremental path. The CPU was the design playground
# where every step had a trivial baseline built for comparison; the GPU port was
# written once, on top of the fully-optimized CPU version, so it is not a
# per-step ablation but the final step of the same progression. We therefore
# render one continuous staircase: Original VUS -> ... -> AutoVUS (CPU) -> AutoVUS (GPU).
# Baseline and the plain "+ Dynamic confusion matrix" step are dropped from the
# figure: Baseline duplicates Original VUS, and only Dynamic+ is kept for the
# confusion matrix. All configs still run and are saved; this only trims the plot.
CPU_PLOT_ORDER = [
    "Original VUS",
    "+ Precomputed slopes",
    "+ Optimized existence",
    "+ Dynamic+ confusion matrix",
    "+ Global mask (FF-VUS)",
]
# The fully-optimized GPU config (all optimizations, step=1) is the terminal bar.
GPU_FINAL_CONFIG = "+ Global mask (FF-VUS-GPU step=1)"

# Short labels for the x-axis (the full config names are too long to read).
# Display names use "AutoVUS"; the CSV "Config" keys keep their original strings.
SHORT_LABELS = {
    "Original VUS": "Original\nVUS",
    "Baseline (no optimizations)": "Baseline",
    "+ Precomputed slopes": "+ Buffer\ninjection",
    "+ Optimized existence": "+ Optimized\nexistence",
    "+ Dynamic confusion matrix": "+ Dynamic\nconf.",
    "+ Dynamic+ confusion matrix": "+ Confusion\nmatrix",
    "+ Global mask (FF-VUS)": "+ Global mask\n(FastVUS)",
    GPU_FINAL_CONFIG: "+ GPU\n(FastVUS-GPU)",
}
# [Sylli] Paul comment: 
# - Remove baseline because it's confusing and write buffer injection instead of precomputed slopes, fastVUS instead of AutoVus, keep only one dynamic conf. (why two?) and fix its name. The title also has the wrong name, actually drop the title completely

def build_ablation_path(summary):
    """Return the ordered (config, backend, total_time, speedup) staircase.

    The CPU incremental configs come first, followed by the fully-optimized GPU
    config as the terminal step. Speedups are recomputed relative to Original VUS
    so the whole path shares one reference.
    """
    cpu = summary[summary["Backend"] == "CPU"].set_index("Config")
    gpu = summary[summary["Backend"] == "GPU"].set_index("Config")

    path = []
    for cfg in CPU_PLOT_ORDER:
        if cfg in cpu.index:
            path.append((cfg, "CPU", float(cpu.loc[cfg, "total_time"])))
    if GPU_FINAL_CONFIG in gpu.index:
        path.append((GPU_FINAL_CONFIG, "GPU", float(gpu.loc[GPU_FINAL_CONFIG, "total_time"])))

    ref = next((t for c, b, t in path if c == "Original VUS"), None)
    return [(c, b, t, (ref / t if ref and t > 0 else float("nan"))) for c, b, t in path]


def plot_ablation(summary, save_path):
    """Render the incremental ablation as a single log-scale staircase.

    Colors follow the paper palette: the Original VUS reference is red, the
    incremental FastVUS (CPU) steps are shaded yellow (light -> dark, darkest
    at the fully-optimized default), and the terminal GPU step is green.
    """
    sns.set_style("whitegrid")
    path = build_ablation_path(summary)
    if not path:
        return None

    configs = [c for c, _, _, _ in path]
    backends = [b for _, b, _, _ in path]
    times = [t for _, _, t, _ in path]
    speedups = [s for _, _, _, s in path]
    x = np.arange(len(configs))

    # Paper palette: VUS red, FastVUS yellow, FastVUS-GPU green.
    vus_red = "#b02a41"
    fastvus_gpu_green = "#40da70"
    fastvus_yellow = to_rgb("#c8b400")
    # Light -> base yellow ramp; the last (default) FastVUS step is the darkest.
    yellow_light = tuple(1.0 - 0.55 * (1.0 - c) for c in fastvus_yellow)
    yellow_cmap = LinearSegmentedColormap.from_list(
        "fastvus", [yellow_light, fastvus_yellow])

    n_yellow = sum(b == "CPU" and c != "Original VUS"
                   for c, b in zip(configs, backends))
    colors = []
    yi = 0
    for c, b in zip(configs, backends):
        if c == "Original VUS":
            colors.append(vus_red)
        elif b == "GPU":
            colors.append(fastvus_gpu_green)
        else:
            colors.append(yellow_cmap(yi / max(1, n_yellow - 1)))
            yi += 1

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x, times, width=0.72, color=colors, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(times) / 3, top=max(times) * 3)

    # Cumulative-speedup label above each bar.
    for xi, t, sp in zip(x, times, speedups):
        label = "1x" if abs(sp - 1.0) < 1e-9 else f"{sp:.0f}x"
        ax.annotate(label, (xi, t), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#1a1a1a")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS.get(c, c) for c in configs], fontsize=9,
                       rotation=15, ha="center")
    # Bold the FastVUS / FastVUS-GPU labels (the named contributions of this work).
    for tick, cfg in zip(ax.get_xticklabels(), configs):
        if "FastVUS" in SHORT_LABELS.get(cfg, cfg):
            tick.set_fontweight("bold")
    ax.set_ylabel("Total runtime (secs; log scale)", fontsize=9)
    # ax.grid(axis="y", which="both", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(visible=False, axis='x')
    # Keep the full box (top/right spines) that the style otherwise drops.
    for spine in ax.spines.values():
        spine.set_visible(True)

    n_series = int(summary["n_series"].max())
    # ax.set_title(
    #     f"AutoVUS ablation on TSB-AD ({n_series} time series x detector pairs)",
    #     fontsize=12, fontweight="bold",
    # )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0)

    # Also drop a canonical copy under experiments/figures/ for the paper.
    canonical_pdf = os.path.join("experiments", "figures", "ablation_study.pdf")
    os.makedirs(os.path.dirname(canonical_pdf), exist_ok=True)
    fig.savefig(canonical_pdf, bbox_inches="tight", pad_inches=0)

    plt.close(fig)
    print(f"Figure saved to {save_path} (and .pdf, and {canonical_pdf})")
    return save_path


def print_ablation_table(summary):
    """Print the ablation path (same order/steps as the figure) as a markdown
    table styled like the rebuttal: Backend | Configuration | Total Time (s) |
    Mean Time (s) | Mean Metric | #Series | Speed-up."""
    path = build_ablation_path(summary)
    if not path:
        return
    lut = summary.set_index(["Backend", "Config"])
    header = ["Backend", "Configuration", "Total Time (s)", "Mean Time (s)",
              "Mean Metric", "#Series", "Speed-up"]

    print("\n" + "=" * 100)
    print("ABLATION TABLE (figure path)")
    print("=" * 100)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for cfg, backend, total_time, speedup in path:
        row = lut.loc[(backend, cfg)]
        name = SHORT_LABELS.get(cfg, cfg).replace("\n", " ")
        print(f"| {backend} | {name} | {total_time:.4f} | {row['mean_time']:.4f} | "
              f"{row['mean_metric']:.4f} | {int(row['n_series'])} | {speedup:.2f}x |")
    print("=" * 100)


def analyze_results_file(path):
    """Load a saved per-time-series results CSV and print the summary + analysis."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    print(f"Analyzing {path}")
    print(f"Rows: {len(df)}  |  Configs: {df['Config'].nunique()}  |  "
          f"Time series: {df['Time series'].nunique()}")
    summary = summarize_results(df)
    print_summary(summary)
    print_ablation_table(summary)

    # Save the ablation figure next to the results file, under a figures/ dir.
    results_dir = os.path.dirname(os.path.abspath(path))
    fig_dir = os.path.join(os.path.dirname(results_dir), "figures")
    base = os.path.splitext(os.path.basename(path))[0]  # e.g. ablation_full_<ts>
    fig_path = os.path.join(fig_dir, f"{base}.png")
    plot_ablation(summary, fig_path)
    return summary


def run_ablation(filenames, labels, scores, slope_size, step_default, run_cpu, run_gpu, save_dir, mode="full"):
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
            kwargs = {**cfg["kwargs"], "slope_size": slope_size, "step": step_default,
                      "max_thresholds": MAX_THRESHOLDS}
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
            kwargs = {**cfg["kwargs"], "slope_size": slope_size, "step": step, "device": device,
                      "max_thresholds": MAX_THRESHOLDS}
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
    tag = f"{mode}_{timestamp}"
    results_file = os.path.join(results_path, f"ablation_{tag}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nPer-time-series results saved to {results_file}")

    # ---- Summary ----
    summary = summarize_results(df)

    summary_file = os.path.join(results_path, f"ablation_summary_{tag}.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    print_summary(summary)

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
        "Mode": mode,
        "Timestamp": timestamp,
    }
    info_df = pd.DataFrame([info])
    info_file = os.path.join(info_path, f"ablation_info_{tag}.csv")
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
    parser.add_argument("--detector", type=str, default=None,
                        help="TSB detector to use: a detector name, 'random' (a random detector "
                             "per time series), or 'all' (every detector). Default: first detector")
    parser.add_argument("--experiment", type=str, default="ablation_study",
                        help="Experiment subdirectory under experiments/ (default: ablation_study)")
    parser.add_argument("--analyze", type=str, default=None, metavar="RESULTS_CSV",
                        help="Analyze an existing per-time-series results CSV and print the "
                             "summary + analysis instead of running the ablation")
    args = parser.parse_args()

    # Ensure paths are relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Analysis-only mode: load an existing results file and print the summary.
    if args.analyze:
        analyze_results_file(args.analyze)
        return

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    # Load data
    print("Loading TSB-AD dataset...")
    tsb = TSADParser()
    if args.test:
        filenames, timeseries, labels = tsb.load_timeseries(dataset=['YAHOO'])
        filenames, timeseries, labels = filenames[:10], timeseries[:10], labels[:10]
    else:
        filenames, timeseries, labels = tsb.load_timeseries(load_all=True)
    detectors = tsb.get_detectors()
    # 'random' loads every detector then picks one per time series, so the
    # parser must load all of them (same cost as 'all').
    per_series_random = args.detector == 'random'
    if args.detector is None:
        selected = [detectors[0]]
    elif args.detector == 'all' or per_series_random:
        selected = detectors
    elif args.detector in detectors:
        selected = [args.detector]
    else:
        raise ValueError(f"Unknown detector '{args.detector}'. Available: {detectors}")
    print(f"Using detector(s): {'random per series, drawn from all' if per_series_random else selected}")

    scores_grouped, _ = tsb.load_scores_and_delete(filenames, timeseries, labels, detectors=selected)
    # Each scores_grouped[i] has shape (len(selected), T). Expand the parallel
    # (filename, label, score) lists to one entry per (time series, detector) so
    # the ablation runner sees every pair transparently.
    exp_filenames, exp_labels, scores = [], [], []
    for fname, label, score_stack in zip(filenames, labels, scores_grouped):
        if per_series_random:
            i = random.randrange(len(selected))
            exp_filenames.append(fname)
            exp_labels.append(label)
            scores.append(score_stack[i])
        else:
            for score in score_stack:
                exp_filenames.append(fname)
                exp_labels.append(label)
                scores.append(score)
    filenames, labels = exp_filenames, exp_labels
    print(f"Loaded {len(filenames)} (time series, detector) pairs "
          f"from {len(selected)} detector(s)")

    save_dir = os.path.join("experiments", args.experiment)
    mode = "test" if args.test else "full"
    run_ablation(filenames, labels, scores, args.slope_size, args.step, run_cpu, run_gpu, save_dir, mode=mode)


if __name__ == "__main__":
    main()
