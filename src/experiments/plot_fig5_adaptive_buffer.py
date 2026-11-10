"""Fig 5: adaptive buffer selection (synthetic anomaly, current FastVUS/AutoVUS engine).
Cells 9 and 10 of src/notebooks/case_studies.ipynb, verbatim, except: smaller canvas (FIG_W x FIG_H)
so the same fonts print larger at the same width, and zero padding around the saved figure.
The canvas keeps the aspect of the paper figure (1892 x 675 pt), so its printed size does not change.

Output: experiments/figures/adaptive_buffer.pdf/.png (copy to ICDE/figures/adaptive_buffer.pdf)
Usage: conda run -n autovus python src/experiments/plot_fig5_adaptive_buffer.py [FIG_W FIG_H]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from autovus import FastVUS, AutoVUS

FIG_W, FIG_H = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (22.5, 6.92)

# -----------------------------
# Figure 5: data + selection
# -----------------------------
rng = np.random.default_rng(7)

T5 = 220
start5, end5 = 92, 122
L5 = 200 - end5  # 78
label5 = np.zeros(T5, dtype=int)
label5[start5:end5] = 1
x5 = np.arange(T5)

main_peak = 0.96 * np.exp(-0.5 * ((x5 - 134) / 11.0) ** 2)
shoulder = 0.24 * np.exp(-0.5 * ((x5 - 111) / 10.0) ** 2)
far_peak = 0.44 * np.exp(-0.5 * ((x5 - 184) / 7.5) ** 2)
background = 0.035 + 0.014 * np.sin(x5 / 9.0) + 0.010 * np.cos(x5 / 17.0)

raw_noise = rng.normal(0, 0.040, size=T5)
kernel = np.array([1, 2, 3, 2, 1], dtype=float)
kernel /= kernel.sum()
smooth_noise = np.convolve(raw_noise, kernel, mode="same")

score5 = main_peak + shoulder + far_peak + background + smooth_noise
score5 = np.clip(score5, 0, None)
score5 = score5 / score5.max()

vus5 = FastVUS(L=L5, cumulative=True)
auto5 = AutoVUS(vus5)
zita5 = float(vus5.zita)


def vus_style_soft_label(T, s, e, b, zita):
    y = np.zeros(T, dtype=float)

    if b == 0:
        y[s:e] = 1.0
        return y

    left_idx = np.arange(max(0, s - b), s)
    left_dist = s - left_idx
    y[left_idx] = 1.0 - (1.0 - zita) * left_dist / b

    right_idx = np.arange(e, min(T, e + b))
    right_dist = right_idx - e
    y[right_idx] = 1.0 - (1.0 - zita) * right_dist / b

    y[s:e] = 1.0
    return y


soft_max5 = vus_style_soft_label(T5, start5, end5, L5, zita5)

res5 = auto5.select(label5, score5)[0]["left"]

b5 = auto5.buffers

# Marginal per-buffer detector performance A_k(b;s) and random baseline g_k(b).
A5 = auto5.decumulate(res5["vus_curve"])
g5 = auto5.decumulate(res5["random_curve"])

# Selection rule r_k(b;s) and b*, straight from AutoVUS.
r5 = res5["ratio_curve"]
b_idx5 = res5["buffer_idx"]
b_star5 = res5["buffer_val"]

print("Figure 5 selected b* =", b_star5)

# -----------------------------
# Figure 5: final plot
# -----------------------------
plt.rcParams.update({
    "font.size": 36,
    "axes.titlesize": 40,
    "axes.labelsize": 40,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 38,
})

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.5, 1.13, 1.13],
    wspace=0.62,
)

blue = "#1f77b4"
orange = "#ff7f0e"
pink = "#f4a6c1"
label_color = "black"

# panel (i)
gs_left = gs[0, 0].subgridspec(
    2, 1,
    height_ratios=[1.0, 1.0],
    hspace=0.50,
)
ax1_top = fig.add_subplot(gs_left[0, 0])
ax1_bottom = fig.add_subplot(gs_left[1, 0], sharex=ax1_top)

ax1_top.plot(x5, soft_max5, color=orange, linewidth=4.8)
ax1_top.step(x5, label5, where="post", color=label_color, linewidth=5.4)
ax1_top.axvline(start5, linestyle="--", linewidth=2.5, color="0.35")
ax1_top.axvline(end5, linestyle="--", linewidth=2.5, color="0.35")
ax1_top.set_ylim(-0.03, 1.08)
ax1_top.set_yticks([0, zita5, 1.0])
ax1_top.set_yticklabels(["0", r"$\zeta$", "1"])
ax1_top.set_ylabel("label", labelpad=18)
ax1_top.tick_params(labelbottom=False)
ax1_top.text(
    0.12, 0.62, "(i)",
    transform=ax1_top.transAxes,
    fontsize=42,
    fontweight="bold",
    va="top",
    ha="left",
)

left0, left1 = max(0, start5 - b_star5), start5
mid0, mid1 = start5, end5
right0, right1 = end5, min(T5, end5 + b_star5)

for a, bspan in [(left0, left1), (mid0, mid1), (right0, right1)]:
    ax1_bottom.axvspan(
        a, bspan,
        color=pink,
        alpha=0.40,
        linewidth=0,
        zorder=1,
    )

ax1_bottom.plot(x5, score5, color=blue, linewidth=4.8, zorder=2)
ax1_bottom.axvline(start5, linestyle="--", linewidth=2.5, color="0.35")
ax1_bottom.axvline(end5, linestyle="--", linewidth=2.5, color="0.35")
ax1_bottom.set_ylim(-0.02, 1.05)
ax1_bottom.set_yticks([0, 0.5, 1.0])
ax1_bottom.set_ylabel("score", labelpad=18)
ax1_bottom.set_xlabel("time", labelpad=14)

ax1_bottom.legend(
    handles=[
        Patch(
            facecolor=pink,
            edgecolor="none",
            alpha=0.40,
            label="selected buffer",
        )
    ],
    frameon=False,
    loc="lower left",
    bbox_to_anchor=(0.00, 1.00),
    handlelength=1.0,
    borderaxespad=0.0,
    prop={"size": 34},
)

for ax in [ax1_top, ax1_bottom]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.13, linewidth=1.0)
    ax.tick_params(width=1.5, length=6)

# panel (ii)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(b5, A5, color=blue, linewidth=5.0, label=r"$A_k(b;s)$")
ax2.plot(b5, g5, color=orange, linewidth=5.0, label=r"$g_k(b)$")
ax2.axvline(b_star5, linestyle="--", linewidth=2.8, color="0.30")
ax2.set_xlim(0, L5)
ax2.set_xlabel("buffer size $b$", labelpad=14)
ax2.set_ylabel("score", labelpad=18)
ax2.text(
    0.97, 0.04, "(ii)",
    transform=ax2.transAxes,
    fontsize=42,
    fontweight="bold",
    va="bottom",
    ha="right",
)
# direct curve labels instead of a legend (larger fonts leave no room for one)
ax2.text(30, A5[30] - 0.07, r"$A_k(b;s)$", color=blue, fontsize=38, va="top", ha="left")
ax2.text(40, g5[40] - 0.06, r"$g_k(b)$", color=orange, fontsize=38, va="top", ha="left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(alpha=0.13, linewidth=1.0)
ax2.tick_params(width=1.5, length=6)

# panel (iii)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(b5, r5, color=blue, linewidth=5.0, label=r"$r_k(b;s)$")
ax3.axvline(b_star5, linestyle="--", linewidth=2.8, color="0.30")
ax3.scatter([b_star5], [r5[b_idx5]], s=220, color=blue, zorder=3)
ax3.set_xlim(0, L5)
ax3.set_xlabel("buffer size $b$", labelpad=14)
ax3.set_ylabel("normalized gain", labelpad=18)
ax3.text(
    0.97, 0.96, "(iii)",
    transform=ax3.transAxes,
    fontsize=42,
    fontweight="bold",
    va="top",
    ha="right",
)
ax3.text(28, 1.0, r"$r_k(b;s)$", color=blue, fontsize=38, va="bottom", ha="left")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(alpha=0.13, linewidth=1.0)
ax3.tick_params(width=1.5, length=6)

out = os.path.join(ROOT, "experiments", "figures", "adaptive_buffer")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0)
fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0, dpi=250)
print("saved:", out + ".pdf")

