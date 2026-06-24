"""
Rule ablation study for the adaptive buffer selection.

Activates one rule at a time against a no-rules baseline to find real examples
where each rule changes the selected buffer.

Usage:
    conda run -n ffvus python src/compare_rules.py [options]

Options:
    --dataset       Filter to one dataset, e.g. YAHOO, NAB, UCR
    --detector      Filter to one detector
    --n-interesting Number of interesting series to analyse (default 5)
    --min-anomalies Minimum anomalies per series (default 3)
    --slope-size    Max buffer to sweep in VUS (default 200)
    --output-dir    Where to save results (default experiments/rule_ablation)
    --show          Display plots interactively instead of saving
"""

import argparse
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vus.vus_numpy import VUSNumpy


# ---------------------------------------------------------------------------
# Rule configurations: each entry activates exactly one rule (ablation mode).
# 'baseline' has all rules off (pure penalized-obj argmax).
# 'all' is the production default.
# ---------------------------------------------------------------------------

RULE_NAMES = {
    'r1':  'R1 (no-skill guard)',
    'r2':  'R2 (first-plateau)',
    'r3':  'R3 (runaway guard)',
    'r4a': 'R4a (degenerate score)',
    'r4b': 'R4b (detection gate)',
}

RULE_CONFIGS = [
    ('baseline', {}),
    ('r1_only',  {'r1':  True}),
    ('r2_only',  {'r2':  True}),
    ('r3_only',  {'r3':  True}),
    ('r4a_only', {'r4a': True}),
    ('r4b_only', {'r4b': True}),
    ('all',      {'r1': True, 'r2': True, 'r3': True, 'r4a': True, 'r4b': True}),
]


# ---------------------------------------------------------------------------
# Helpers (shared with test_per_anomaly_buffer.py)
# ---------------------------------------------------------------------------

def _dataset_tag(fname):
    return fname.split('_')[1]


def _get_anomaly_lengths(label):
    diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends = np.where(diff < 0)[0] - 1
    if len(starts) == 0:
        return np.array([], dtype=int)
    return (ends - starts + 1).astype(int)


def _select_interesting(fnames, labels, min_anomalies=3, n_top=5, dataset_filter=None):
    candidates = []
    for i, (fn, lbl) in enumerate(zip(fnames, labels)):
        if dataset_filter and _dataset_tag(fn) != dataset_filter:
            continue
        lengths = _get_anomaly_lengths(lbl)
        if len(lengths) < min_anomalies:
            continue
        diversity = float(lengths.std() / (lengths.mean() + 1e-8))
        candidates.append((i, fn, lengths, diversity))
    candidates.sort(key=lambda x: -x[3])
    return candidates[:n_top]


# ---------------------------------------------------------------------------
# Per-anomaly run under a given rule config
# ---------------------------------------------------------------------------

def _run_config(vus_obj, label, score, config_dict, gate_margin):
    """Run per-anomaly adaptive slicing under a single rule config.

    Returns list of buffer_vals, one per anomaly.
    """
    per_anom = vus_obj.compute_per_anomaly_adaptive_slicing(
        label, score,
        plot=False,
        gate_margin=gate_margin,
        rules=config_dict if config_dict else {},
    )
    return [r['best']['buffer_val'] for r in per_anom]


# ---------------------------------------------------------------------------
# Plot for a single changed anomaly
# ---------------------------------------------------------------------------

def _plot_changed_anomaly(
    fname, label, score, anom_idx, anom_start, anom_end,
    buffers_by_config,   # dict config_name → buffer_val
    out_path, show=False,
):
    """Ratio-curve comparison for one anomaly across rule configs."""
    fig, (ax_ts, ax_sc, ax_buf) = plt.subplots(
        3, 1, figsize=(12, 7),
        gridspec_kw={'height_ratios': [1.5, 1, 1.5]},
        sharex=False,
    )

    T = len(label)
    x = np.arange(T)

    # --- label panel ---
    ax_ts.plot(x, label, color='lightcoral', linewidth=0.8, label='label')
    ax_ts.axvspan(anom_start, anom_end + 1, color='red', alpha=0.2, linewidth=0, label='anomaly')
    ax_ts.set_ylabel('label', fontsize=8)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.legend(fontsize=7, loc='upper right')
    ax_ts.set_title(
        f"{fname}  |  anomaly {anom_idx} [{anom_start}:{anom_end}]  len={anom_end - anom_start + 1}",
        fontsize=8, fontweight='bold',
    )
    ax_ts.tick_params(labelsize=7, labelbottom=False)

    # --- score panel ---
    ax_sc.plot(x, score, color='steelblue', linewidth=0.7)
    ax_sc.axvspan(anom_start, anom_end + 1, color='red', alpha=0.15, linewidth=0)
    ax_sc.set_ylabel('score', fontsize=8)
    ax_sc.set_xlabel('time', fontsize=8)
    ax_sc.set_xlim(0, T - 1)
    ax_sc.tick_params(labelsize=7)

    # --- buffer comparison panel ---
    config_names = list(buffers_by_config.keys())
    b_vals = [buffers_by_config[c] for c in config_names]
    colors = ['gray' if c == 'baseline' else ('crimson' if c == 'all' else 'steelblue')
              for c in config_names]
    bars = ax_buf.barh(config_names, b_vals, color=colors, alpha=0.75)
    ax_buf.set_xlabel('b* (selected buffer)', fontsize=8)
    ax_buf.tick_params(labelsize=7)
    for bar, val in zip(bars, b_vals):
        ax_buf.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va='center', fontsize=7)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(args):
    from tsbadparser import TSBADParser

    vus_obj = VUSNumpy(
        slope_size=args.slope_size, step=1,
        cumulative=True, global_mask=False,
    )

    tsb = TSBADParser()
    detectors = tsb.get_detectors()
    if args.detector:
        if args.detector not in detectors:
            print(f"Unknown detector '{args.detector}'. Available: {detectors}")
            return
        detectors = [args.detector]

    print("Loading time series...")
    load_kwargs = {'dataset': [args.dataset]} if args.dataset else {'load_all': True}
    fnames, timeseries, labels = tsb.load_timeseries(**load_kwargs)

    print(f"Loading scores for {len(detectors)} detectors...")
    scores, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=detectors)

    interesting = _select_interesting(
        fnames, labels,
        min_anomalies=args.min_anomalies,
        n_top=args.n_interesting,
        dataset_filter=args.dataset,
    )

    if not interesting:
        print("No interesting series found. Try lowering --min-anomalies.")
        return

    print(f"\nSelected {len(interesting)} series:")
    for _, fn, lengths, div in interesting:
        print(f"  {fn}  n_anom={len(lengths)}  lengths={sorted(lengths.tolist())}  diversity={div:.2f}")

    out_base = args.output_dir
    fname_to_idx = {fn: i for i, fn in enumerate(fnames)}

    # CSV summary
    csv_path = os.path.join(out_base, 'summary.csv')
    os.makedirs(out_base, exist_ok=True)
    csv_rows = []
    csv_header = ['series', 'detector', 'anomaly_idx', 'anomaly_len'] + [c for c, _ in RULE_CONFIGS]

    for _, fn, anom_lengths, _ in interesting:
        idx = fname_to_idx.get(fn)
        if idx is None:
            print(f"\n  Skipping {fn} (not in score table)")
            continue

        lbl = labels[idx]
        sc  = scores[idx]
        fname_safe = fn.replace('/', '_')

        # Anomaly coordinates (reuse existing helper)
        diff = np.diff(np.concatenate([[0], lbl.astype(int), [0]]))
        anom_starts = np.where(diff > 0)[0]
        anom_ends   = np.where(diff < 0)[0] - 1

        print(f"\n  {fn}  [{len(anom_lengths)} anomalies]")

        for det_idx, det in enumerate(detectors):
            score = sc[det_idx]

            # Run every config and collect buffer_vals per anomaly
            results_by_config = {}
            for cfg_name, cfg_dict in RULE_CONFIGS:
                try:
                    results_by_config[cfg_name] = _run_config(
                        vus_obj, lbl, score, cfg_dict, args.gate_margin,
                    )
                except Exception as ex:
                    print(f"    [{det}] [{cfg_name}] ERROR: {ex}")
                    results_by_config[cfg_name] = [None] * len(anom_starts)

            n_anom = len(anom_starts)
            changed_rules = set()

            for k in range(n_anom):
                row = [fn, det, k, int(anom_ends[k] - anom_starts[k] + 1)]
                row += [results_by_config[c][k] for c, _ in RULE_CONFIGS]
                csv_rows.append(row)

                baseline_b = results_by_config['baseline'][k]
                for cfg_name, _ in RULE_CONFIGS:
                    if cfg_name in ('baseline', 'all'):
                        continue
                    cfg_b = results_by_config[cfg_name][k]
                    if cfg_b != baseline_b:
                        changed_rules.add(cfg_name)
                        rule_key = cfg_name.replace('_only', '')
                        rule_label = RULE_NAMES.get(rule_key, cfg_name)
                        print(f"    [{det}] anom {k} [{anom_starts[k]}:{anom_ends[k]}] "
                              f"len={anom_ends[k]-anom_starts[k]+1}  "
                              f"baseline={baseline_b}  {cfg_name}={cfg_b}  "
                              f"→ {rule_label} fires here")

                        # Save a comparison plot for this anomaly × rule
                        buffers = {c: results_by_config[c][k] for c, _ in RULE_CONFIGS
                                   if results_by_config[c][k] is not None}
                        out_path = os.path.join(
                            out_base, 'plots', rule_key,
                            f"{fname_safe}_{det}_anom{k}.png",
                        )
                        _plot_changed_anomaly(
                            fn, lbl, score, k,
                            int(anom_starts[k]), int(anom_ends[k]),
                            buffers, out_path, show=args.show,
                        )

            if not changed_rules:
                all_b = [results_by_config['all'][k] for k in range(n_anom)]
                print(f"    [{det}] no rule changes any anomaly  b_all={all_b}")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"\nSummary saved to {os.path.abspath(csv_path)}")
    print(f"Plots saved under {os.path.abspath(os.path.join(out_base, 'plots'))}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Rule ablation study for adaptive buffer selection")
    p.add_argument('--dataset',        type=str,  default=None)
    p.add_argument('--detector',       type=str,  default=None)
    p.add_argument('--n-interesting',  type=int,  default=5)
    p.add_argument('--min-anomalies',  type=int,  default=3)
    p.add_argument('--slope-size',     type=int,  default=200)
    p.add_argument('--gate-margin',    type=int,   default=10)
    p.add_argument('--show',           action='store_true')
    p.add_argument('--output-dir',     type=str,
                   default=os.path.join(
                       os.path.dirname(__file__), '..', 'experiments', 'rule_ablation',
                   ))
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
