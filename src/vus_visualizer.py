"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""

import pandas as pd
from tqdm import tqdm
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
import torch
import os

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch


def main(plot):
    if plot == 'existence_seq':
        visualize_existence_examples()
    elif plot == 'conf_matrix':
        confusion_matrix_visualization()
    elif plot == 'gpu_buffers':
        gpu_buffers_visualization()
    elif plot == 'gpu_existence':
        gpu_existence_visualization()
    elif plot == 'gpu_existence_segmax':
        gpu_existence_segmax_visualization()
    elif plot == 'steps':
        steps_visualization()
    elif plot == 'global_mask':
        global_mask_visualization()
    elif plot == 'progressive_vus':
        progressive_vus_visualization()
    else:
        raise ValueError('Unknown plot type')


def global_mask_visualization():
    sns.set_style("whitegrid")
    buffer_size = 20
    ffvus = VUSNumpy(buffer_size)
    global_mask_choices = [False, True]

    tsb = TSADParser()
    fn, _, y = tsb.load_timeseries(dataset=['MGAB'])
    ts_index = max(range(len(y)), key=lambda i: len(y[i]))
    label = y[ts_index]
    print(f"Time series '{fn[ts_index]}', Length: {len(label)}")
    # label = torch.tensor([0]*150 + [1]*20 + [0]*200 + [1]*50 + [0]*250 + [1]*20 + [0]*200, dtype=torch.float32)
    
    fig, axes = plt.subplots(2, 1, figsize=(2.5, 1.7), sharex=False)
    for i, global_mask in enumerate(global_mask_choices):
        ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), _ = ffvus.get_anomalies_coordinates(label)
        if global_mask:
            safe_mask, _ = ffvus.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=global_mask)
            label = label[safe_mask]
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), _ = ffvus.get_anomalies_coordinates(label)
        else:
            pass
        labels = ffvus.add_slopes_precomputed(label, start_no_edges, end_no_edges)
        
        sns.lineplot(labels.T, ax=axes[i], palette='flare_r', legend=False, linewidth=1.5)
        # axes[i].set_title(f'Global mask: {global_mask}, Length: {len(label)}')
        axes[i].set_ylabel(f'{global_mask}')
        axes[i].set_yticks([])
        if i == 0:
            axes[i].set_xticks([0, 50000, 100000], ['0', '50k', '100k'])
        print(f'Size reduction {(max_len / len(label)):.0f}x')
    fig.text(0.01, 0.55, 'Global mask:', ha='center', va='center', rotation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig("experiments/figures/global_mask.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def steps_visualization():
    sns.set_style("whitegrid")
    
    step_sizes = [1, 5]

    label = np.array([0]*50 + [1]*20 + [0]*30).astype(np.float64)    

    fig, axes = plt.subplots(len(step_sizes), 1, figsize=(2.5, 1.7), sharex=True)
    for i, step in enumerate(step_sizes):
        buffer_size = 20
        ffvus = VUSNumpy(
            slope_size=buffer_size,
            step=step
        )
        ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
        labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    
        sns.lineplot(labels.T, ax=axes[i], palette='flare_r', legend=False, linewidth=1.5)
        # axes[i].annotate(f'Buffer length {buffer_size}\nStep {step}', xy=(5, .2), xytext=(.5, .2))
        axes[i].annotate(f'Step {step}', xy=(5, .2), xytext=(.5, .2))
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.savefig("experiments/figures/buffer_steps.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def gpu_existence_visualization():
    sns.set_style("whitegrid")
    device = 'cpu'
    buffer_size = 10
    ffvus = VUSTorch(buffer_size)

    label = torch.tensor([0]*15 + [1]*20 + [0]*40 + [1]*10 + [0]*25 + [1]*1 + [0]*20, dtype=torch.float32)
    score = torch.tensor(np.zeros_like(label, dtype=float))
    score[10] = 0.5
    score[28] = 0.8
    score[55] = 0.6
    # score[63] = 0.6
    # score[71] = 0.9
    score[110] = 0.4
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)
    thresholds, _ = ffvus.get_unique_thresholds(score)
    score_mask, _ = ffvus.get_score_mask(score, thresholds)

    # Compute existence
    # We myltiply every label in labels with every score of all thresholds (both normalized; since we are only counting anomalies found we don't care about the value of the buffer)
    # This means that only the regions where label and score are both 1 will survive in the output
    # Since the label is encoded with the staircase, the output also contains the info of which anomaly was found (e.g. value is 4 for the 4th anonaly found)
    # The cumulative max carries this info accross all time steps after every anomaly found and not only at the point where it was found
    norm_labels = (labels > 0).int()
    diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
    diff = torch.clamp(diff, min=0, max=1)
    stairs = torch.cumsum(diff, dim=1)
    labels_stairs = norm_labels * stairs
    score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]  # Multiply every score mask with every staircased buffered label, shape (L, t, T)
    cm = torch.cummax(score_hat, dim=2).values                      # Now every anomaly that is found, encodes the number of the anomaly, shape (L, t, T)
    cm_diff = torch.diff(cm, dim=2)                                 
    cm_diff_norm = torch.clamp(cm_diff - 1, min=0)                  

    total_anomalies = stairs[:, -1][:, None]                                            # Tells the total number of anomalies
    final_anomalies_missed = total_anomalies - cm[:, :, -1]                             # CM dif can't show the last anomaly(-ies) that were not found, since it's a dif operation
    n_anomalies_not_found = torch.sum(cm_diff_norm, dim=2) + final_anomalies_missed     # The normalized cm dif shows the number of anomalies not found, if every anomaly is found then cm_diff is always 1, so cm_diff_norm is always 0
    n_anomalies_found = total_anomalies - n_anomalies_not_found                         
    
    # --- Visualization ---
    fig, axes = plt.subplots(3, 1, figsize=(6, 3.5), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # (a) Label, buffered label, staircase
    axes[0].plot(labels_stairs[-1], color="#C51B7D", lw=2, label="Staircase")           # magenta
    axes[0].plot(label, linestyle='dashed', color="#1F78B4", lw=2, label="Original")    # bright blue
    axes[0].plot(labels[-1], linestyle='dotted', color="#4D4D4D", lw=2, label="Buffered") # dark grey

    axes[0].set_title("(a) Original (dashed), buffered (dotted), staircase (solid)")
    axes[0].legend(loc="upper right").remove()
    axes[0].set_ylim(top=3.2)
    axes[0].set_yticks([0, 1, 2, 3])

    # (b) Scores, mask, score_hat
    axes[1].plot(score_hat[-1, -2, :], color="#FDB863", lw=2, label="Score hat")       # deep red
    axes[1].plot(score_mask[-2], linestyle='dotted', color="#B2182B", lw=2, label="Mask")  # dark grey
    # axes[1].plot(score, linestyle='dashed', color="#FBFF00", lw=2, label="Score")      # amber/orange
    axes[1].set_title("(b) Binary score mask (dotted), and $\\overline{Score}$ (solid)")
    axes[1].legend(loc="upper right").remove()
    axes[1].set_ylim(top=3.2)
    axes[1].set_yticks([0, 1, 2, 3])

    # (e) Cumulative max
    axes[2].plot(cm[-1, -2, :], color="dodgerblue", lw=2.5)
    axes[2].set_title("(c) Cumulative Max of $\\overline{Score}$")
    axes[2].hlines(y=[1, 3], xmin=0, xmax=label.shape[0], linestyle='--', color='k', lw=1.5)
    axes[2].annotate('Step difference 2 $\\Rightarrow$ 1 anomaly missed', xy=(40, 1), xytext=(20, 1.6))
    axes[2].annotate('', xy=(18, 1), xytext=(18, 3), arrowprops=dict(arrowstyle='<->', linestyle='--', color='k', lw=1.5))
    axes[2].set_xlabel("Time")
    axes[2].set_ylim(top=3.2)
    axes[2].set_yticks([0, 1, 2, 3])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("experiments/figures/gpu_existence.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def gpu_existence_segmax_visualization():
    """
    Step-by-step walkthrough of VUSTorch.compute_existence_segmax.

    Instead of the cummax scan over (s, t, T) score_hat (see
    gpu_existence_visualization), segmax reduces the score to ONE max per
    region (threshold-independent), then counts per threshold how many
    region-maxes clear it, via sort + searchsorted in (s, K) memory.
    """
    sns.set_style("whitegrid")
    device = 'cpu'
    ffvus = VUSTorch(slope_size=20, step=4, device=device)

    # Toy series exercising every mechanism of the method:
    #   A: strong peak inside            -> found at every threshold
    #   B: moderate peak inside          -> found at mid thresholds; rescues C after the merge
    #   C: no peak at all                -> missed alone, but MERGES with B once the slopes touch (b >= 8)
    #   D: peak 6 points AFTER the end   -> found only once the buffer extends the region over it
    T = 400
    rng = np.random.default_rng(7)
    regions = {'A': (60, 75), 'B': (150, 160), 'C': (176, 186), 'D': (300, 330)}
    label = torch.zeros(T, dtype=torch.float32)
    for start, end in regions.values():
        label[start:end] = 1
    score_np = rng.uniform(0.0, 0.15, T)
    for center, height in [(67, 0.95), (154, 0.55), (336, 0.75)]:
        bump = height * np.exp(-0.5 * ((np.arange(T) - center) / 6) ** 2)
        np.maximum(score_np, bump, out=score_np)
    score = torch.tensor(score_np, dtype=torch.float64)

    # Same preprocessing compute() runs before the existence call
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)                 # (s, T) soft labels
    buffers = ffvus.slope_values.numpy()
    thresholds = torch.linspace(1.0, 0.0, 41, dtype=torch.float64)
    highlight = [0.9, 0.6, 0.3]
    thresh_colors = {0.9: '#a50f15', 0.6: '#de2d26', 0.3: '#fb6a4a'}
    # Okabe-Ito subset (CVD-safe), fixed order, one per region id
    region_colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']

    # --- Replicate the internals, keeping every intermediate ---
    # Region grouping: identical stairs trick as compute_existence, so the
    # merge behaviour is preserved exactly.
    s = labels.shape[0]
    t = thresholds.shape[0]
    norm_labels = (labels > 0).int()
    diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((s, 1), device=device))
    diff = torch.clamp(diff, min=0, max=1)
    stairs = torch.cumsum(diff, dim=1)
    labels_stairs = norm_labels * stairs                     # (s, T), 0 = non-anomaly
    total_anomalies = stairs[:, -1][:, None]

    # Per-region max score: ONE scatter-amax, no threshold axis at all.
    # Empty bins (rows whose regions merged below K) stay -inf.
    K = int(labels_stairs.max().item())
    seg_max = torch.full((s, K + 1), float('-inf'), device=device, dtype=score.dtype)
    seg_max.scatter_reduce_(1, labels_stairs.long(), score[None, :].expand(s, T),
                            reduce='amax', include_self=True)
    seg_max = seg_max[:, 1:]                                 # (s, K)

    # Counting without the (s, K, t) comparison: sort ascending, then
    # searchsorted gives per threshold the number of maxes strictly below it;
    # found = K - below (-inf bins always count as below, so merged rows stay correct).
    sorted_seg, _ = torch.sort(seg_max, dim=1)
    below = torch.searchsorted(sorted_seg, thresholds[None, :].expand(s, t).contiguous(), right=False)
    existence = (K - below).to(torch.float32) / total_anomalies

    # Sanity: replica == real method == old cummax formulation
    segmax_result, _ = ffvus.compute_existence_segmax(labels, score, thresholds)
    score_mask, _ = ffvus.get_score_mask(score, thresholds)
    cummax_result, _ = ffvus.compute_existence(labels, score_mask)
    print(f"max |replica - compute_existence_segmax| = {(existence - segmax_result).abs().max():.2e}")
    print(f"max |segmax  - compute_existence(cummax)| = {(segmax_result - cummax_result).abs().max():.2e}")

    stairs_np = labels_stairs.numpy()
    seg_max_np = seg_max.numpy()
    focus_rows = [0, s // 2, s - 1]

    fig = plt.figure(figsize=(14, 17))
    gs = GridSpec(6, 2, figure=fig, height_ratios=[1.1, 0.9, 0.8, 0.8, 0.8, 1.2],
                  hspace=0.65, wspace=0.25)

    # (a) Inputs
    ax = fig.add_subplot(gs[0, :])
    ax.plot(score_np, color='#4d4d4d', lw=1.2)
    for name, (start, end) in regions.items():
        ax.axvspan(start, end, color='#bbbbbb', alpha=0.45)
        ax.text((start + end) / 2, 1.04, name, ha='center', fontsize=10, fontweight='bold')
    for th in highlight:
        ax.axhline(th, color=thresh_colors[th], ls='--', lw=1)
        ax.text(T + 2, th, f'θ={th}', color=thresh_colors[th], fontsize=8, va='center')
    ax.set_xlim(0, T + 25)
    ax.set_ylim(0, 1.12)
    ax.set_title('(a) Inputs: score (line), anomaly regions (grey), thresholds (dashed).\n'
                 'A: strong peak · B: moderate peak · C: no peak · D: peak 6 pts AFTER the region', fontsize=10)

    # (b) labels_stairs heatmap: region id per (slope, time) cell
    ax = fig.add_subplot(gs[1, :])
    cmap = mcolors.ListedColormap(['#ffffff'] + region_colors[:K])
    norm = mcolors.BoundaryNorm(np.arange(K + 2) - 0.5, cmap.N)
    ax.imshow(stairs_np, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm,
              extent=[0, T, s - 0.5, -0.5])
    ax.set_yticks(range(s))
    ax.set_yticklabels([f'b={b}' for b in buffers])
    for row in range(s):
        for rid in np.unique(stairs_np[row]):
            if rid > 0:
                span = np.where(stairs_np[row] == rid)[0]
                ax.text(span.mean(), row, str(int(rid)), ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
    ax.set_title('(b) labels_stairs = norm_labels * cumsum(rising edges): region id per (slope, time) cell.\n'
                 'Larger buffers widen regions; B and C touch at b≥8 and MERGE into one region (id 2)', fontsize=10)

    # (c-e) seg_max for three slope rows
    for i, row in enumerate(focus_rows):
        ax = fig.add_subplot(gs[2 + i, :])
        ax.plot(score_np, color='#4d4d4d', lw=0.8, alpha=0.5)
        found_at = {th: 0 for th in highlight}
        for rid in range(1, K + 1):
            span = np.where(stairs_np[row] == rid)[0]
            if span.size == 0:
                continue
            m = seg_max_np[row, rid - 1]
            color = region_colors[rid - 1]
            ax.axvspan(span[0], span[-1] + 1, color=color, alpha=0.18)
            ax.hlines(m, span[0], span[-1] + 1, color=color, lw=2.5)
            ax.text(span.mean(), m + 0.03, f'{m:.2f}', ha='center', fontsize=8, color=color)
            for th in highlight:
                found_at[th] += int(m >= th)
        for th in highlight:
            ax.axhline(th, color=thresh_colors[th], ls='--', lw=0.8, alpha=0.8)
        total = int(total_anomalies[row])
        counts = ',  '.join(f'θ={th}: {found_at[th]}/{total}' for th in highlight)
        ax.set_title(f'({chr(99 + i)}) seg_max at b={buffers[row]}  ({total} regions) — found: {counts}', fontsize=10)
        ax.set_xlim(0, T + 25)
        ax.set_ylim(0, 1.05)

    # (f) The searchsorted trick on the last slope row
    ax = fig.add_subplot(gs[5, 0])
    row = s - 1
    sorted_row = sorted_seg[row].numpy()
    xs = np.arange(1, K + 1)
    for x, v in zip(xs, sorted_row):
        if np.isinf(v):
            ax.bar(x, 0.04, bottom=-0.12, color='#999999', width=0.55)
            ax.text(x, -0.14, '-inf\n(empty bin)', ha='center', va='top', fontsize=8)
        else:
            ax.bar(x, v, color='#4d4d4d', width=0.55)
            ax.text(x, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for th in highlight:
        ax.axhline(th, color=thresh_colors[th], ls='--', lw=1)
        n_below = int(np.sum(sorted_row < th))
        ax.text(K + 0.45, th + 0.01, f'θ={th}: below={n_below} → found=K−below={K - n_below}',
                color=thresh_colors[th], fontsize=8)
    ax.set_xlim(0.4, K + 3.2)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xticks(xs)
    ax.set_xlabel('sorted position')
    ax.set_title(f'(f) searchsorted trick at b={buffers[row]}: sort seg_max ascending,\n'
                 'count entries < θ; -inf pads rows that merged below K', fontsize=10)

    # (g) Resulting existence surface
    ax = fig.add_subplot(gs[5, 1])
    im = ax.imshow(existence.numpy(), aspect='auto', interpolation='nearest',
                   cmap='Blues', vmin=0, vmax=1,
                   extent=[thresholds[0].item(), thresholds[-1].item(), s - 0.5, -0.5])
    ax.set_yticks(range(s))
    ax.set_yticklabels([f'b={b}' for b in buffers])
    ax.set_xlabel('threshold θ (descending)')
    ax.set_title('(g) Existence surface: n_found / total_anomalies per (slope, θ).\n'
                 'Steps up as θ drops below each seg_max; pattern shifts at b≥8 (merge + late D)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle('compute_existence_segmax, step by step', fontsize=13, y=0.995)
    plt.savefig("experiments/figures/gpu_existence_segmax.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def gpu_buffers_visualization():
    sns.set_style("whitegrid")
    buffer_size = 100
    ffvus = VUSTorch(buffer_size, step=10)

    label = torch.tensor([0]*150 + [1]*200 + [0]*250 + [1]*100 + [0]*100 + [1]*10 + [0]*200, dtype=torch.float32)
    T = label.shape[0]
    
    (_, (start_points, end_points)), _ = ffvus.get_anomalies_coordinates(label)
    pos, _ = ffvus.distance_from_anomaly(label, start_points, end_points)
    labels, _ = ffvus.add_slopes(label, pos)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5, 2))

    sns.lineplot(labels.T, ax=ax[1], palette='flare_r', legend=False, linewidth=1.5)
    ax[1].set_title(f'Label with {buffer_size} buffers')
    ax[0].plot(pos, color='k', linewidth=2.5)
    ax[0].set_title('Position sequence')
    ax[0].fill_between(range(T), 0, 200, where=pos >= buffer_size, color="pink", alpha=0.8, label="Not affected regions")
    ax[0].fill_between(range(T), 0, 200, where=label, color="pink", alpha=0.8, label="Original anomaly")
    ax[0].fill_between(range(T), 0, 200, where=np.logical_and(pos < buffer_size, np.logical_not(label)), color="lightgreen", alpha=0.8, label="Buffers")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig("experiments/figures/gpu_buffers.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

def confusion_matrix_visualization():
    sns.set_style("whitegrid")
    buffer_length = 50
    ffvus = VUSNumpy(buffer_length)

    label = np.array([0]*100 + [1]*200 + [0]*400).astype(np.float64)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    T = label.shape[0]
    
    score = np.zeros_like(label, dtype=float)
    score += np.random.uniform(0, .1, score.shape[0])
    score[200:400] += .7
    thresholds = np.linspace(1, 0, 11).round(2)
    sm, _ = ffvus.get_score_mask(score, thresholds)
    t = 0.6

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4.5, 2.2))

    ax[0].plot(labels[-1], color='k', linewidth=2.5)
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 0, color="lightskyblue", alpha=0.5, label="Non-anomalous")
    ax[0].fill_between(range(T), 0, 1, where=labels[-1] == 1, color="pink", alpha=0.5, label="Original anomaly")
    ax[0].fill_between(range(T), 0, 1, where=np.logical_and(labels[-1] < 1, labels[-1] > 0), color="purple", alpha=0.5, label="Buffer region")
    # ax[0].set_title(f'Label with buffer {buffer_length}')
    ax[0].set_ylabel('Label')
    ax[0].set_yticks([])
    ax[0].legend()

    ax[1].plot(score, color='k', linewidth=2.5)
    ax[1].hlines(y=t, xmin=0, xmax=len(label), lw=2.5, color='red', linestyle='--', label='threshold')
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), label == 1), color="lightgreen", alpha=0.5, label="$TP_{initial}$")
    ax[1].fill_between(range(T), 0, 1, where=np.logical_and(np.logical_and(score > t, labels[-1]), labels[-1] < 1), color="darkgreen", alpha=0.5, label="$TP_{buffer}$")
    # ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 1, labels[-1]  == 0), color="red", alpha=0.5, label="FP; O(t, T)")
    # ax[1].fill_between(range(T), 0, 1, where=np.logical_and(score == 0, label == 1), color="orange", alpha=0.5, label="FN; O(t, T)")
    ax[1].set_xlabel('Time')
    # ax[1].set_xticks([])
    # ax[1].set_title('Score of threshold t\'')
    ax[1].set_ylabel('Score')
    ax[1].set_yticks([])
    ax[1].legend()
    
    plt.tight_layout()
    # plt.savefig("experiments/figures/conf_matrix_mask.svg", bbox_inches="tight", pad_inches=0)
    plt.savefig("experiments/figures/conf_matrix_mask.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def visualize_existence_examples():
    thresholds = np.linspace(1, 0, 11).round(2) 
    ffvus = VUSNumpy(10)
    
    label = np.array([0]*20 + [1]*20 + [0]*20)
    ((start_points, end_points), _), _ = ffvus.get_anomalies_coordinates(label)
    labels = ffvus.add_slopes_precomputed(label, start_points, end_points)
    
    score = np.zeros_like(label, dtype=float)
    score[30] = 0.6
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_1", "Detected inside anomaly")

    score = np.zeros_like(label, dtype=float) # np.random.uniform(0, 0.05, label.shape[0])
    score[15] = 0.6
    score[30] = 0.4
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_2", "Left buffer + middle")

    score = np.zeros_like(label, dtype=float)
    score[13] = 0.6
    score[17] = 0.4
    score[30] = 0.2
    existence = ffvus.existence_optimized(labels, score, thresholds, start_points, end_points)
    plot_existence_toy(labels, score, existence, thresholds, "existence_seq_3", "Two in left buffer + inside anomaly")


def plot_existence_toy(labels, score, existence, thresholds, filename, title_suffix):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(6, 2.5))
    gs = GridSpec(2, 5, figure=fig, hspace=.6, wspace=.9)

    # Buffered label
    ax0 = fig.add_subplot(gs[0, :3])
    sns.lineplot(labels.T, ax=ax0, palette='flare_r', legend=False, linewidth=1.5)
    ax0.set_title(f"Label with buffers") #  ({title_suffix})

    # Score
    ax1 = fig.add_subplot(gs[1, :3], sharex=ax0)
    ax1.plot(score, markersize=3, color='black', linewidth=2.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.linspace(0, 1, 6))
    ax1.set_title("Score")
    ax1.set_xlabel("Time")

    # Existence heatmap
    ax2 = fig.add_subplot(gs[:, 3:])
    sns.heatmap(existence, cmap="flare_r", rasterized=True, ax=ax2)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Buffer")
    # 
    # ax2.set_xticklabels(thresholds)
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    ax2.set_xticks(np.arange(0, 11, 2))
    ax2.set_xticklabels(['1', '.8', '.6', '.4', '.2', '0'])
    ax2.invert_yaxis()

    # plt.tight_layout()global_mask
    plt.savefig(f"experiments/figures/{filename}.svg", bbox_inches='tight')
    plt.savefig(f"experiments/figures/{filename}.pdf", bbox_inches='tight')
    # plt.show()

    

# ---------------------------------------------------------------------------
# Progressive VUS
# ---------------------------------------------------------------------------

def _compute_progressive(label, score, slope_size, backend):
    if backend == 'torch':
        import torch as _torch
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        label_t = _torch.tensor(label, device=device, dtype=_torch.uint8)
        score_t = _torch.tensor(score, device=device, dtype=_torch.float32)
        vus = VUSTorch(slope_size=slope_size, step=1, cumulative=True, device=device)
        with _torch.no_grad():
            (cum_values, _), _ = vus.compute(label_t, score_t)
        return cum_values.cpu().numpy()
    else:
        vus = VUSNumpy(slope_size=slope_size, step=1, cumulative=True)
        (cum_values, _), _ = vus.compute(label, score)
        return cum_values


def _plot_single_progressive(fname, label, score, curve, L, backend, output, ts_values=None, detector=None):
    n_rows = 4 if ts_values is not None else 3
    height_ratios = [1.2, 0.7, 0.7, 2.5] if ts_values is not None else [1, 1, 3]
    fig = plt.figure(figsize=(14, 10 if ts_values is not None else 9))
    gs = GridSpec(n_rows, 1, figure=fig, height_ratios=height_ratios, hspace=0.45)

    row = 0
    if ts_values is not None:
        ax_ts = fig.add_subplot(gs[row]); row += 1
    ax_label = fig.add_subplot(gs[row], sharex=(ax_ts if ts_values is not None else None)); row += 1
    ax_score = fig.add_subplot(gs[row], sharex=ax_label); row += 1
    ax_vus = fig.add_subplot(gs[row])

    t = np.arange(len(label))

    if ts_values is not None:
        ax_ts.plot(t, ts_values, linewidth=0.7, color='dimgray', label='Time series')
        ax_ts.fill_between(t, ts_values.min(), ts_values.max(),
                           where=label.astype(bool), alpha=0.25, color='tomato', label='Anomaly region')
        ax_ts.set_ylabel('Value', fontsize=10)
        ax_ts.legend(loc='upper right', fontsize=8)
        ax_ts.set_title(
            f'{os.path.basename(fname)}  |  length={len(label)}  anomaly_ratio={label.mean():.3f}', fontsize=11
        )
        ax_ts.tick_params(labelbottom=False)

    ax_label.fill_between(t, label, alpha=0.5, color='tomato', label='Anomaly')
    ax_label.plot(t, label, linewidth=0.5, color='tomato', alpha=0.7)
    ax_label.set_ylabel('Label', fontsize=10)
    ax_label.set_ylim(-0.1, 1.3)
    if ts_values is None:
        ax_label.set_title(f'{os.path.basename(fname)}  |  length={len(label)}  anomaly_ratio={label.mean():.3f}', fontsize=11)
    ax_label.legend(loc='upper right', fontsize=8)
    ax_label.tick_params(labelbottom=False)
    ax_label.set_yticks([0, 1])

    score_label = f'Score  [{detector}]' if detector else 'Anomaly score'
    ax_score.plot(t, score, linewidth=0.7, color='steelblue', label=score_label)
    ax_score.set_ylabel('Score', fontsize=10)
    ax_score.legend(loc='upper right', fontsize=8)
    ax_score.set_xlabel('Time step', fontsize=10)

    x = np.arange(L + 1)
    ax_vus.plot(x, curve, linewidth=2, color='steelblue', zorder=3)
    ax_vus.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    auc_pr_val = curve[0]
    vus_max_val = curve[L]
    ax_vus.scatter([0], [auc_pr_val], color='tomato', zorder=5, s=70,
                   label=f'slope=0  (AUC-PR w/ existence) = {auc_pr_val:.4f}')
    ax_vus.scatter([L], [vus_max_val], color='seagreen', zorder=5, s=70,
                   label=f'slope={L}  (VUS-PR) = {vus_max_val:.4f}')

    ax_vus.set_xlabel('Slope size (buffer length)', fontsize=12)
    ax_vus.set_ylabel('VUS-PR', fontsize=12)
    ax_vus.set_title(f'VUS-PR vs Slope Size  |  Backend: {backend}', fontsize=13)
    ax_vus.set_xlim(0, L)
    ax_vus.legend(fontsize=10, loc='lower right')
    ax_vus.grid(True, alpha=0.3)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Figure saved to: {output}')
    else:
        plt.show()


def _plot_multiple_progressive(filenames, labels, curves, L, dataset, backend, output):
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(L + 1)
    cmap = plt.cm.tab10
    for i, (fname, label, curve) in enumerate(zip(filenames, labels, curves)):
        short_name = os.path.basename(fname)
        ratio = label.mean()
        ax.plot(x, curve, alpha=0.65, linewidth=1.3, color=cmap(i % 10),
                label=f'{short_name} (ratio={ratio:.3f})')

    mean_curve = np.mean(curves, axis=0)
    ax.plot(x, mean_curve, color='black', linewidth=2.5, label=f'Mean (n={len(curves)})', zorder=5)

    auc_pr_mean = mean_curve[0]
    vus_mean = mean_curve[L]
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.9, alpha=0.55)
    ax.scatter([0], [auc_pr_mean], color='tomato', zorder=6, s=60)
    ax.scatter([L], [vus_mean], color='seagreen', zorder=6, s=60)

    offset_y = (mean_curve.max() - mean_curve.min()) * 0.05
    ax.annotate(
        f'AUC-PR (slope=0)\nmean = {auc_pr_mean:.4f}',
        xy=(0, auc_pr_mean),
        xytext=(L * 0.05, auc_pr_mean + offset_y),
        fontsize=9, color='dimgray',
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1),
    )
    ax.annotate(
        f'VUS-PR (slope={L})\nmean = {vus_mean:.4f}',
        xy=(L, vus_mean),
        xytext=(L * 0.80, vus_mean + offset_y),
        fontsize=9, color='dimgray',
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1),
    )

    ax.set_xlabel('Slope size (buffer length)', fontsize=13)
    ax.set_ylabel('VUS-PR', fontsize=13)
    ax.set_title(f'VUS-PR vs Slope Size  |  Dataset: {dataset}  |  Backend: {backend}', fontsize=14)
    ax.set_xlim(0, L)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Figure saved to: {output}')
    else:
        plt.show()


def progressive_vus_visualization(
    dataset='YAHOO',
    n=5,
    slope_size=512,
    backend='numpy',
    ts_index=None,
    pick_random=False,
    seed=None,
    output=None,
):
    """Plot VUS-PR as a curve from slope=0 to slope=L on TSB-AD data."""
    print(f'Loading {n} time series from {dataset}...')
    tsb = TSADParser()
    filenames, timeseries_list, labels = tsb.load_timeseries(dataset=[dataset])
    filenames = filenames[:n]; timeseries_list = timeseries_list[:n]; labels = labels[:n]
    raw_scores, _ = tsb.load_scores_and_delete(filenames, timeseries_list, labels)
    scores = [s[0] for s in raw_scores]
    detector_name = tsb.get_detectors()[0]
    print(f'  Loaded {len(filenames)} time series.')

    single_idx = None
    if ts_index is not None:
        if ts_index >= len(filenames):
            raise ValueError(f'ts_index {ts_index} out of range (loaded {len(filenames)} series)')
        single_idx = ts_index
    elif pick_random:
        if seed is not None:
            random.seed(seed)
        single_idx = random.randrange(len(filenames))
        print(f'  Randomly selected index {single_idx}: {filenames[single_idx]}')

    if single_idx is not None:
        filenames = [filenames[single_idx]]
        timeseries_list = [timeseries_list[single_idx]]
        labels = [labels[single_idx]]
        scores = [scores[single_idx]]

    curves = []
    for i, (fname, label, score) in enumerate(zip(filenames, labels, scores)):
        print(f'  [{i+1}/{len(filenames)}] {fname}  '
              f'len={len(label)}  anomaly_ratio={label.mean():.3f}')
        curve = _compute_progressive(label, score, slope_size, backend)
        curves.append(curve)

    if len(curves) == 1:
        _plot_single_progressive(
            filenames[0], labels[0], scores[0], curves[0],
            slope_size, backend, output,
            ts_values=timeseries_list[0], detector=detector_name,
        )
    else:
        _plot_multiple_progressive(
            filenames, labels, curves,
            slope_size, dataset, backend, output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='main',
        description='Visualize different useful concepts of the ffvus metric'
    )

    parser.add_argument(
        '--plot',
        type=str,
        choices=['existence_seq', 'conf_matrix', 'gpu_buffers', 'gpu_existence', 'gpu_existence_segmax', 'steps', 'global_mask', 'progressive_vus'],
        default='existence_seq',
        help='Type of plot to generate'
    )
    args = parser.parse_args()

    main(plot=args.plot)