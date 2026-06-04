"""
Adaptive Buffer Selection — Real-World Examples (TSB-AD)
=========================================================

Loads real time series from the TSB-AD benchmark and visualises the adaptive
buffer selection for a single detector.

Series are binned by mean anomaly length into quartiles and N representatives
are sampled from each bin, giving a spread of structural cases.

Usage
-----
    conda run -n ffvus python src/test_adaptive_buffer_real.py
    conda run -n ffvus python src/test_adaptive_buffer_real.py --dataset NAB
    conda run -n ffvus python src/test_adaptive_buffer_real.py --detector MatrixProfile
    conda run -n ffvus python src/test_adaptive_buffer_real.py --n-per-bin 1

Layout (one figure per selected series)
----------------------------------------
    Row 0 : raw time series + label (red) + selected buffer dotted lines
    Row 1 : detector score + label (red)
    Row 2 : ratio curve (analytical denom, all 3 methods)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vus.vus_numpy import VUSNumpy, select_best_buffer
from tsbadparser import TSBADParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOPE_SIZE      = 1024
DEFAULT_DETECTOR = 'IForest'
METHODS          = ['argmax', 'elbow', 'first_peak']

METHOD_COLORS = {
    'argmax':     'royalblue',
    'elbow':      'darkorange',
    'first_peak': 'mediumpurple',
}

DENOM_STYLE = dict(color='crimson', linestyle='--')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_tag(fname):
    return fname.split('_')[1]


def _analyze_label(label):
    diff   = np.diff(label.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]
    if len(label) > 0 and label[-1]:
        ends = np.append(ends, len(label) - 1)
    if len(label) > 0 and label[0]:
        starts = np.insert(starts, 0, 0)
    n = int(starts.shape[0])
    if n == 0:
        return len(label), 0, 0.0
    lengths = (ends + 1) - starts
    return len(label), n, float(np.mean(lengths))


def _get_anomaly_bounds(label):
    diff   = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0] - 1
    return starts, ends


def _draw_label(ax, anom_starts, anom_ends, y_data=None):
    """Draw red spans for multi-point anomalies; red dots for point anomalies.

    y_data: if provided, point-anomaly dots are placed at the actual data value.
    """
    for s, e in zip(anom_starts, anom_ends):
        if s == e:
            y = float(y_data[s]) if y_data is not None else 0.0
            ax.scatter([s], [y], color='red', s=30, zorder=6)
        else:
            ax.axvspan(s, e, color='red', alpha=0.15, linewidth=0)


def _select_series(fnames, labels, dataset_filter, n_per_bin, rng):
    """
    Bin series by mean anomaly length (quartiles) and return n_per_bin
    representatives from each bin.
    """
    stats = []
    for i, (fn, lbl) in enumerate(zip(fnames, labels)):
        if dataset_filter and _dataset_tag(fn) != dataset_filter:
            continue
        T, n_anom, mean_len = _analyze_label(lbl)
        if n_anom == 0:
            continue
        stats.append((i, mean_len, n_anom, T, fn))

    if not stats:
        raise ValueError(f"No series found for dataset filter '{dataset_filter}'.")

    stats.sort(key=lambda x: x[1])  # sort by mean anomaly length
    n      = len(stats)
    q_size = max(1, n // 4)
    bins   = [stats[k:k + q_size] for k in range(0, n, q_size)]

    selected = []
    for b in bins:
        picks = rng.choice(len(b), size=min(n_per_bin, len(b)), replace=False)
        for p in picks:
            selected.append(b[p])

    return selected  # list of (original_idx, mean_len, n_anom, T, fname)


def _compute_method_bests(vus, labels_arr, score, vus_pr, rng):
    T   = len(score)
    res, rt = vus.find_adaptive_buffer(
        labels_arr, score, vus_pr, T,
        denom='analytical', method=METHODS[0], rng=rng,
    )
    ratio        = res['ratio']
    method_bests = {METHODS[0]: res['best']}
    for method in METHODS[1:]:
        best = select_best_buffer(ratio, vus.slope_values, method=method)
        best['vus_score'] = float(vus_pr[best['buffer_idx']])
        method_bests[method] = best
    return ratio, method_bests, rt


# ---------------------------------------------------------------------------
# Per-series figure
# ---------------------------------------------------------------------------

def plot_series(fname, timeseries, label, score, detector, vus, rng):
    T      = len(label)
    tag    = _dataset_tag(fname)
    _, n_anom, mean_len = _analyze_label(label)

    ((start_ne, end_ne), _), _ = vus.get_anomalies_coordinates(label)
    labels_arr = vus.add_slopes_precomputed(label, start_ne, end_ne)

    (vus_pr, _), _ = vus.compute(label, score, adaptive=False)
    ratio, method_bests, rt = _compute_method_bests(vus, labels_arr, score, vus_pr, rng)

    buffers      = vus.slope_values
    anom_starts, anom_ends = _get_anomaly_bounds(label)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[1].sharex(axes[0])
    vus_total = float(vus_pr[-1])
    fig.suptitle(
        f"[{tag}]  {fname}   T={T}  anomalies={n_anom}  mean_len={mean_len:.1f}"
        f"   |   detector: {detector}   VUS={vus_total:.4f}",
        fontsize=9, fontweight='bold',
    )

    # Row 0: raw time series + label (red) + buffer dotted lines
    ax = axes[0]
    _draw_label(ax, anom_starts, anom_ends, y_data=timeseries)
    ax.plot(timeseries, linewidth=0.7, color='dimgray')
    b   = method_bests['argmax']['buffer_val']
    col = METHOD_COLORS['argmax']
    for s, e in zip(anom_starts, anom_ends):
        ax.axvline(max(0, s - b),     color=col, linestyle=':', linewidth=1.2, alpha=0.9)
        ax.axvline(min(T - 1, e + b), color=col, linestyle=':', linewidth=1.2, alpha=0.9)
    ts_handles = [
        plt.matplotlib.lines.Line2D([], [], color='dimgray', linewidth=0.7, label='time series'),
        plt.matplotlib.patches.Patch(facecolor='red', alpha=0.3, label='label'),
        plt.matplotlib.lines.Line2D(
            [], [], color=METHOD_COLORS['argmax'], linestyle=':', linewidth=1.2,
            label=f"argmax  b*={method_bests['argmax']['buffer_val']}",
        ),
    ]
    ax.legend(handles=ts_handles, fontsize=7, loc='upper right')
    ax.set_ylabel("Time series", fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(labelsize=7, labelbottom=False)

    # Row 1: detector score
    ax = axes[1]
    ax.plot(score, linewidth=0.9, color='steelblue')
    ax.legend(
        handles=[plt.matplotlib.lines.Line2D([], [], color='steelblue', linewidth=0.9, label='score')],
        fontsize=7, loc='upper right',
    )
    ax.set_ylabel(detector, fontsize=8)
    ax.tick_params(labelsize=7)

    # Row 2: ratio curve
    ax = axes[2]
    ax.plot(buffers, ratio, linewidth=1.5, **DENOM_STYLE, label='analytical')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    for method in METHODS:
        best = method_bests[method]
        bval = best['buffer_val']
        rval = best['ratio']
        col  = METHOD_COLORS[method]
        ax.axvline(x=bval, color=col, linestyle=':', linewidth=1.2, alpha=0.8)
        ax.scatter([bval], [rval], color=col, zorder=5, s=45,
                   label=f"{method}  b*={bval}  vus={best['vus_score']:.3f}")
    ax.set_title(f"analytical   runtime: {rt * 1000:.1f} ms", fontsize=8)
    ax.set_ylabel("Ratio", fontsize=8)
    ax.set_xlabel("Buffer")
    ax.set_xlim(0, int(buffers[-1]))
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive buffer — real TSB-AD examples")
    p.add_argument('--dataset',    type=str,  default=None,
                   help="Filter to a single dataset, e.g. NAB, YAHOO, UCR")
    p.add_argument('--detector',   type=str,  default=DEFAULT_DETECTOR,
                   help=f"Detector to use (default: {DEFAULT_DETECTOR})")
    p.add_argument('--n-per-bin',  type=int,  default=2,
                   help="Number of series to pick from each structural bin (default 2)")
    p.add_argument('--seed',       type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    rng      = np.random.default_rng(args.seed)
    vus      = VUSNumpy(slope_size=SLOPE_SIZE, step=1, cumulative=True, global_mask=False, slopes='precomputed')
    detector = args.detector

    tsb = TSBADParser()

    valid_detectors = tsb.get_detectors()
    if detector not in valid_detectors:
        print(f"Error: detector '{detector}' not found.")
        print(f"Available detectors:\n  {', '.join(sorted(valid_detectors))}")
        sys.exit(1)

    print("Loading time series...")
    fnames, timeseries, labels = tsb.load_timeseries(load_all=True)

    if args.dataset is not None:
        valid_datasets = sorted(set(_dataset_tag(fn) for fn in fnames))
        if args.dataset not in valid_datasets:
            print(f"Error: dataset '{args.dataset}' not found.")
            print(f"Available datasets:\n  {', '.join(valid_datasets)}")
            sys.exit(1)

    print(f"Loading scores for [{detector}]...")
    scores, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=[detector])

    selected = _select_series(fnames, labels, args.dataset, args.n_per_bin, rng)
    print(f"Selected {len(selected)} series.")

    fname_to_idx = {fn: i for i, fn in enumerate(fnames)}

    for _, mean_len, n_anom, T, fn in selected:
        idx = fname_to_idx.get(fn)
        if idx is None:
            print(f"  Skipping {fn} (dropped during score loading)")
            continue
        print(f"  Plotting {fn}  (mean_len={mean_len:.1f}, n_anom={n_anom})")
        plot_series(fn, timeseries[idx], labels[idx], scores[idx][0], detector, vus, rng)
