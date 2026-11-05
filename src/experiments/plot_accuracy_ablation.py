"""Fig 7 + Fig 8 side by side (paper Sec IV-A): (a) error to VUS on TSB-AD, (b) per-step ablation.

Inputs: experiments/all_metrics_tsb_def_params/results (VUS, FastVUS, FastVUS-GPU at L=128, 24,704 pairs)
and ablation_study/results/ablation_summary_full_20260705_133315.csv.
Outputs: experiments/figures/error_ecdf_small.pdf and ablation_small.pdf, drawn at print size for the
0.42 / 0.56 \\linewidth subfigures (one IEEE column = 3.5 in), fonts 7 pt.
Usage: conda run -n autovus python src/experiments/plot_accuracy_ablation.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ARCH = os.path.join("experiments")  # all_metrics_tsb_def_params and ablation_study (moved here from the archive, 2026-11-05)
OUT = os.path.join("experiments", "figures")
os.makedirs(OUT, exist_ok=True)
COL = 3.5                      # IEEE column width in inches
H = 1.55
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "pdf.fonttype": 42})
RED, YEL, GRN = "#b02a41", "#c8b400", "#40da70"


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def panel_a():
    R = f"{ARCH}/all_metrics_tsb_def_params/results"
    k = ["Time series", "Detector"]
    rd = lambda f, n: pd.read_csv(f"{R}/{f}")[k + ["Metric value"]].rename(columns={"Metric value": n})
    d = (rd("tsb_VUS_128.csv", "vus")
         .merge(rd("tsb_FF-VUS_128_1_dynamic_plus_globalmask_function_optimized_subsampled.csv", "cpu"), on=k)
         .merge(rd("tsb_FF-VUS-GPU_128_1_dynamic_plus_globalmask_function_optimized_subsampled.csv", "gpu"), on=k))
    e_cpu = (d.cpu - d.vus).abs().values
    e_gpu = (d.gpu - d.vus).abs().values
    fig, ax = plt.subplots(figsize=(0.38 * COL, H), constrained_layout=True)
    for e, c, name in [(e_cpu, YEL, "FastVUS"), (e_gpu, GRN, "FastVUS-GPU")]:
        x = np.sort(np.clip(e, 1e-16, None)); y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, lw=1.4, label=name)
    ax.set_xscale("log"); ax.set_xlim(3e-17, 3e-2); ax.set_ylim(0, 1.02)
    ax.set_xticks([1e-15, 1e-9, 1e-3])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Absolute error"); ax.set_ylabel("Fraction of series")
    ax.grid(True, alpha=0.3, lw=0.4)
    for q in (0.5, 0.9, 0.99):
        ax.axhline(q, color="gray", lw=0.5, ls=(0, (4, 4)), alpha=0.6)
        va, dy = ("top", -0.01) if q == 0.9 else ("bottom", 0.01)
        ax.text(0.98, q + dy, f"p{int(q*100)}", transform=ax.transAxes, va=va, ha="right",
                fontsize=5.5, color="gray")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.3), ncol=1, frameon=False,
              handlelength=1.0, borderaxespad=0, labelspacing=0.2, fontsize=6)
    for s in ax.spines.values(): s.set_visible(True)
    fig.savefig(f"{OUT}/error_ecdf_small.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(f"{OUT}/error_ecdf_small.png", dpi=300, bbox_inches="tight", pad_inches=0)
    print("cpu max %.2e, gpu max %.2e, %d pairs" % (e_cpu.max(), e_gpu.max(), len(d)))


def panel_b():
    s = pd.read_csv(f"{ARCH}/ablation_study/results/ablation_summary_full_20260705_133315.csv")
    cpu = s[s.Backend == "CPU"].set_index("Config"); gpu = s[s.Backend == "GPU"].set_index("Config")
    order = [("Original VUS", "VUS", cpu), ("+ Precomputed slopes", "+ Buffer inj.", cpu),
             ("+ Optimized existence", "+ Existence", cpu),
             ("+ Dynamic+ confusion matrix", "+ Conf. matrix", cpu),
             ("+ Global mask (FF-VUS)", "+ Mask (FastVUS)", cpu),
             ("+ Global mask (FF-VUS-GPU step=1)", "FastVUS-GPU", gpu)]
    times = [float(t.loc[c, "total_time"]) for c, _, t in order]
    labels = [l for _, l, _ in order]
    ref = times[0]
    ylight = tuple(1.0 - 0.55 * (1.0 - c) for c in to_rgb(YEL))
    cmap = LinearSegmentedColormap.from_list("fastvus", [ylight, YEL])
    colors = [RED] + [cmap(i / 3) for i in range(4)] + [GRN]
    fig, ax = plt.subplots(figsize=(0.60 * COL, H), constrained_layout=True)
    y = np.arange(len(times))[::-1]
    ax.barh(y, times, color=colors, height=0.72, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.7, 1e5)
    ax.set_xticks([1, 100, 10000])
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for tick, l in zip(ax.get_yticklabels(), labels):
        if "FastVUS" in l: tick.set_fontweight("bold")
    for yi, t in zip(y, times):
        sp = ref / t
        ax.text(t * 1.4, yi, "1x" if abs(sp - 1) < 1e-9 else f"{sp:.0f}x", va="center", ha="left",
                fontsize=6.5, fontweight="bold", color="#1a1a1a")
    ax.set_xlabel("Total runtime (s)")
    ax.grid(True, axis="x", alpha=0.3, lw=0.4, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s_ in ax.spines.values(): s_.set_visible(True)
    fig.savefig(f"{OUT}/ablation_small.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(f"{OUT}/ablation_small.png", dpi=300, bbox_inches="tight", pad_inches=0)
    print("ablation:", [f"{ref/t:.0f}x" for t in times])


if __name__ == "__main__":
    panel_a(); panel_b()
