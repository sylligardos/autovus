"""Supplementary figure: VUS error versus the number of thresholds, for the two
threshold selections (evenly spread as in the original VUS, and the FastVUS selection).

Reads experiments/compare_threshold_cap/results/results_*.csv (run of 2026-06-22,
1,059 series-detector pairs, L = 128, error against the all-thresholds value).
Writes docs/supplementary/threshold_selection.{png,pdf} and prints the table.

    conda run -n autovus python src/experiments/plot_threshold_cap.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = "experiments/compare_threshold_cap/results"
OUT = "docs/supplementary/threshold_selection"
RED, YEL, INK, MUTED = "#b02a41", "#c8b400", "#1a1a1a", "#6b6b6b"
SERIES = [("subsampled", "Evenly spread (original)", RED, "o"),
          ("pr_relevant_fuzzy", "FastVUS selection", YEL, "s")]


def load():
    files = [f for f in glob.glob(f"{RES}/results_*.csv") if os.path.getsize(f) > 0]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
    d = d[(d.backend == "numpy") & (d.slope_size == 128)]
    key = ["filename", "detector"]
    ref = d[d.strategy == "all_unique"][key + ["vus_value"]].rename(columns={"vus_value": "ref"})
    m = d[d.strategy != "all_unique"].merge(ref, on=key)
    m = m[m.cap != float("inf")]
    m["err"] = (m.vus_value - m.ref).abs()
    return m.groupby(["strategy", "cap"]).err.agg(["mean", "max"]).reset_index(), len(ref)


def main():
    t, n_pairs = load()
    print(f"{n_pairs} pairs")
    print(t.pivot(index="cap", columns="strategy", values=["max", "mean"]).to_string(float_format="%.1e"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for ax, col, title in zip(axes, ["max", "mean"], ["Maximum absolute error", "Mean absolute error"]):
        for key, name, color, marker in SERIES:
            s = t[t.strategy == key].sort_values("cap")
            ax.plot(s.cap, s[col], color=color, marker=marker, markersize=4.5, linewidth=1.6,
                    markeredgecolor="white", markeredgewidth=0.6, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        caps = sorted(t.cap.unique())
        ax.set_xticks(caps)
        ax.set_xticklabels([f"{int(c)}" for c in caps])
        ax.minorticks_off()
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Number of thresholds", color=INK)
        ax.grid(True, which="major", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7.5)
    axes[0].set_ylabel("Error to all thresholds", color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=220)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    main()
