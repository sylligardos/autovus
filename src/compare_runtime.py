"""Practicality closer: VUS metric time vs detector inference time on TSB-AD.

Refutes the reviewer's "VUS is probably not the bottleneck" claim by showing that
for a large class of detectors, computing the VUS score of a detector's output costs
more than running the detector itself. AutoVUS removes this cost entirely.

Detector inference times come from the TSB-AD benchmark (via tsad_parser.load_runtime).
Metric times come from experiments/all_metrics_tsb_def_params/results, per (series, detector).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsad_parser import TSADParser

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "experiments", "all_metrics_tsb_def_params", "results")
FIGDIR = os.path.join(BASE, "experiments", "figures")

# Paper naming + palette (consistent with the other figures).
VUS_NAME, CPU_NAME, GPU_NAME = "VUS", "AutoVUS", "AutoVUS-GPU"
DET_NAME = "Detector inference"
COLORS = {
    VUS_NAME: "#b02a41",   # main competitor (red)
    CPU_NAME: "#c8b400",   # AutoVUS (gold)
    GPU_NAME: "#40da70",   # AutoVUS-GPU (green)
    DET_NAME: "#4d4d4d",   # detector inference (dark grey)
}

METRIC_FILES = {
    VUS_NAME: "tsb_VUS_128.csv",
    CPU_NAME: "tsb_FF-VUS_128_1_dynamic_plus_globalmask_function_optimized_subsampled.csv",
    GPU_NAME: "tsb_FF-VUS-GPU_128_1_dynamic_plus_globalmask_function_optimized_subsampled.csv",
}


def load_data():
    """Merge metric times with detector inference times on (series, detector)."""
    merged = None
    for label, fname in METRIC_FILES.items():
        df = pd.read_csv(os.path.join(RESULTS, fname))
        df = df[["Time series", "Detector", "Metric time"]].rename(
            columns={"Metric time": label})
        merged = df if merged is None else merged.merge(df, on=["Time series", "Detector"])

    rt = TSADParser().load_runtime()[["File Name", "Model Name", "Time Cost"]].rename(
        columns={"File Name": "Time series", "Model Name": "Detector", "Time Cost": DET_NAME})

    df = merged.merge(rt, on=["Time series", "Detector"], how="inner")
    df = df[df[DET_NAME] > 0].reset_index(drop=True)  # drop unrecorded (0s) detector runs
    return df


def print_stats(df):
    n = len(df)
    print(f"Matched (series, detector) evaluations: {n}"
          f"  ({df['Time series'].nunique()} series x {df['Detector'].nunique()} detectors)\n")

    print(f"{'':<12}{'total (s)':>12}{'mean (s)':>10}{'median (s)':>12}")
    for c in [DET_NAME, VUS_NAME, CPU_NAME, GPU_NAME]:
        print(f"{c:<12}{df[c].sum():>12.1f}{df[c].mean():>10.3f}{df[c].median():>12.4f}")

    print("\n--- Rebuttal numbers ---")
    slower = (df[VUS_NAME] >= df[DET_NAME]).mean() * 100
    print(f"* VUS slower than the detector itself in {slower:.1f}% of the {n} evaluations.")

    # Per-detector medians: how many detectors are cheaper to run than to score with VUS?
    g = df.groupby("Detector").agg(**{
        DET_NAME: (DET_NAME, "median"), VUS_NAME: (VUS_NAME, "median"),
        CPU_NAME: (CPU_NAME, "median"), GPU_NAME: (GPU_NAME, "median")}).sort_values(DET_NAME)
    cheaper = g[g[DET_NAME] < g[VUS_NAME]]
    print(f"* {len(cheaper)}/{len(g)} detectors have a median run cheaper than a median VUS "
          f"evaluation: {', '.join(cheaper.index)}.")
    worst = (g[VUS_NAME] / g[DET_NAME]).max()
    worst_det = (g[VUS_NAME] / g[DET_NAME]).idxmax()
    print(f"* Worst case: VUS is {worst:.0f}x slower than {worst_det}.")
    print(f"* Median VUS {df[VUS_NAME].median():.3f}s vs median detector "
          f"{df[DET_NAME].median():.3f}s (same order of magnitude).")
    print(f"* AutoVUS cuts the median to {df[CPU_NAME].median():.4f}s "
          f"({df[VUS_NAME].median()/df[CPU_NAME].median():.0f}x smaller than VUS), "
          f"{100*df[CPU_NAME].median()/df[DET_NAME].median():.2f}% of the median detector run "
          f"and slower than the detector in only {100*(df[CPU_NAME] >= df[DET_NAME]).mean():.1f}% of cases.")
    print(f"* Aggregate caveat (be honest): summed over the whole benchmark, VUS is only "
          f"{100*df[VUS_NAME].sum()/df[DET_NAME].sum():.1f}% of total detection time, because a "
          f"few very slow deep detectors dominate that total.\n")
    return g


def make_figure(g):
    """Per-detector inference time vs the three metric engines (sorted by detector cost)."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    x = np.arange(len(g))
    # Shade the region where the detector is cheaper to run than VUS is to compute.
    n_cheaper = int((g[DET_NAME] < g[VUS_NAME]).sum())
    if n_cheaper:
        ax.axvspan(-0.5, n_cheaper - 0.5, color="#b02a41", alpha=0.06, zorder=0)
        ax.text((n_cheaper - 1) / 2, g[[VUS_NAME, DET_NAME]].max().max() * 1.3,
                "VUS $\\geq$ detector", ha="center", va="bottom",
                fontsize=8, color="#b02a41")

    ax.plot(x, g[DET_NAME], "-o", color=COLORS[DET_NAME], ms=3.5, lw=1.3,
            label="Detector inference", zorder=3)
    for name, style in [(VUS_NAME, "-"), (CPU_NAME, "--"), (GPU_NAME, "--")]:
        ax.plot(x, g[name], style, color=COLORS[name], lw=1.8, label=name, zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("Median runtime per series\n(secs; log scale)", fontsize=9)
    ax.set_xlabel("")
    ax.set_xlim(-0.6, len(g) - 0.4)
    ax.grid(visible=False, axis="x")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "runtime_vs_detector_tsb.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    df = load_data()
    g = print_stats(df)
    make_figure(g)
