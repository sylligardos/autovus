"""
Adaptive buffer — three modes:

  explore  Loop one detector over a dataset; shows the 5-panel ratio plot per
           series (original behaviour — uses find_adaptive_buffer inside compute).

  compute  Compute analytical vs Monte-Carlo denominators for every series in a
           dataset (no detector needed).  Saves a pickle + prints a ranked table.

  review   Load a previously saved pickle and plot the worst --top_n cases.

Usage:
    conda run -n ffvus python src/find_adaptive_buffer.py --mode explore
    conda run -n ffvus python src/find_adaptive_buffer.py --mode explore --dataset YAHOO --slope_size 100
    conda run -n ffvus python src/find_adaptive_buffer.py --mode compute --dataset YAHOO --K 100
    conda run -n ffvus python src/find_adaptive_buffer.py --mode review  --input experiments/adaptive_buffer/YAHOO_K100_L100.pkl
"""

import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy


# ---------------------------------------------------------------------------
# Helpers shared by compute / review
# ---------------------------------------------------------------------------

def _analytical_denom(vus, label):
    """Closed-form E[cumVUS(b)] for an independent random score (label-only)."""
    ((start_ne, end_ne), _), _ = vus.get_anomalies_coordinates(label)
    labels_arr = vus.add_slopes_precomputed(label, start_ne, end_ne)
    g_b = labels_arr.mean(axis=1)
    indices = np.arange(1, vus.n_slopes + 1)
    return np.cumsum(g_b) / indices


def _mc_denom(vus, label, K, rng):
    """Monte Carlo E[cumVUS(b)] with K Bernoulli(p=label.mean()) random scores."""
    p = label.mean()
    T = len(label)
    total = np.zeros(vus.n_slopes)
    vus._skip_adaptive_plot = True
    try:
        for _ in range(K):
            rand_score = rng.binomial(1, p, size=T).astype(float)
            (cum_vus, _), _ = vus.compute(label, rand_score)
            total += cum_vus
    finally:
        vus._skip_adaptive_plot = False
    return total / K


def _default_output(dataset, K, slope_size):
    return os.path.join("experiments", "adaptive_buffer",
                        f"{dataset}_K{K}_L{slope_size}.pkl")


# ---------------------------------------------------------------------------
# Mode: explore  (original behaviour)
# ---------------------------------------------------------------------------

def run_explore(dataset, detector, slope_size):
    tsb = TSBADParser()
    fnames, _, labels = tsb.load_timeseries(dataset=[dataset])
    if not fnames:
        raise ValueError(f"No files found for dataset '{dataset}'")

    if detector is None:
        detector = tsb.get_detectors()[1]

    vus = VUSNumpy(slope_size=slope_size, step=1, cumulative=True)

    for i, (fname, label) in enumerate(zip(fnames, labels)):
        label = label.astype(float)

        scores, _ = tsb.load_scores(detectors=[detector], filenames=[fname])
        if not scores:
            print(f"[{i+1}/{len(fnames)}] Skipping {os.path.basename(fname)}: "
                  f"no scores for '{detector}'")
            continue
        score = scores[0][0].astype(float)

        n = min(len(label), len(score))
        label, score = label[:n], score[:n]

        print(f"\n[{i+1}/{len(fnames)}] {os.path.basename(fname)}")
        print(f"  Detector: {detector}  |  Length: {n:,}  |  Anomaly ratio: {label.mean():.4f}")

        vus._explore_detector_name = detector
        (cum_vus, _), _ = vus.compute(label, score)

        print(f"  VUS at b=0: {cum_vus[0]:.4f}  |  VUS at b={slope_size}: {cum_vus[-1]:.4f}")


# ---------------------------------------------------------------------------
# Mode: compute
# ---------------------------------------------------------------------------

def run_compute(datasets, K, top_n, slope_size, output):
    tsb = TSBADParser()
    vus = VUSNumpy(slope_size=slope_size, step=1, cumulative=True)
    rng = np.random.default_rng(42)

    records = []  # holds last dataset's records; used for single-dataset plot

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        fnames, _, labels = tsb.load_timeseries(dataset=[dataset])
        if not fnames:
            print(f"  No files found — skipping.")
            continue

        records = []

        for i, (fname, label) in enumerate(zip(fnames, labels)):
            label = label.astype(float)
            name  = os.path.basename(fname)
            p     = label.mean()
            T     = len(label)

            if p == 0.0:
                print(f"[{i+1}/{len(fnames)}] {name} — skipped (no anomalies)")
                continue

            n_anom = int(np.sum(np.diff(np.concatenate([[0], label.astype(int), [0]])) > 0))
            print(f"[{i+1}/{len(fnames)}] {name}  T={T:,}  p={p:.4f}  n_anom={n_anom}",
                  end="  ", flush=True)

            a = _analytical_denom(vus, label)
            m = _mc_denom(vus, label, K, rng)

            max_diff = float(np.max(np.abs(m - a)))
            rmse     = float(np.sqrt(np.mean((m - a) ** 2)))
            print(f"max|diff|={max_diff:.5f}  RMSE={rmse:.5f}")

            records.append({
                "fname":      name,
                "T":          T,
                "p":          p,
                "n_anom":     n_anom,
                "max_diff":   max_diff,
                "rmse":       rmse,
                "analytical": a,
                "mc":         m,
                "label":      label,
            })

        if not records:
            print("  No valid series — skipping.")
            continue

        records.sort(key=lambda r: r["max_diff"], reverse=True)

        print(f"\n  Ranked by max|diff| (worst first) — {dataset}")
        for rank, r in enumerate(records, 1):
            print(f"  {rank:3d}. {r['fname']:<45s}  max_diff={r['max_diff']:.5f}  "
                  f"RMSE={r['rmse']:.5f}  T={r['T']:,}  p={r['p']:.4f}  n_anom={r['n_anom']}")

        out = output if (output and len(datasets) == 1) else _default_output(dataset, K, slope_size)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({"dataset": dataset, "K": K, "slope_size": slope_size,
                         "records": records}, f)
        print(f"\n  Saved {out}")

    if len(datasets) == 1 and records:
        _plot_worst(records, top_n, datasets[0], K, slope_size, vus.slope_values)


# ---------------------------------------------------------------------------
# Label stats helper (used by review)
# ---------------------------------------------------------------------------

def _compute_label_stats(label):
    """Return anomaly-structure statistics from a binary label array."""
    diff   = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0]   # first index after each anomaly

    if len(starts) == 0:
        return {"mean_anom_len": 0.0, "min_anom_len": 0.0,
                "mean_gap": np.nan,  "min_gap": np.nan}

    lengths = ends - starts
    stats = {
        "mean_anom_len": float(lengths.mean()),
        "min_anom_len":  float(lengths.min()),
    }
    if len(starts) > 1:
        gaps = starts[1:] - ends[:-1]
        stats["mean_gap"] = float(gaps.mean())
        stats["min_gap"]  = float(gaps.min())
    else:
        stats["mean_gap"] = np.nan
        stats["min_gap"]  = np.nan

    return stats


# ---------------------------------------------------------------------------
# Mode: review
# ---------------------------------------------------------------------------

def run_review(input_paths, top_n):
    all_records = []
    K          = None
    slope_size = None

    for path in input_paths:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping.")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)

        if K is None:
            K, slope_size = data["K"], data["slope_size"]
        elif data["slope_size"] != slope_size:
            print(f"  Warning: {path} has slope_size={data['slope_size']} "
                  f"(expected {slope_size}) — skipping.")
            continue

        for r in data["records"]:
            r["dataset"] = data["dataset"]
        all_records.extend(data["records"])

    if not all_records:
        print("No records loaded.")
        return

    records    = sorted(all_records, key=lambda r: r["max_diff"], reverse=True)
    datasets   = sorted({r["dataset"] for r in records})
    buffers    = np.arange(len(records[0]["analytical"]))

    print(f"Loaded {len(records)} records  datasets={datasets}  K={K}  slope_size={slope_size}")
    print("\n=== Ranked by max|diff| (worst first) ===")
    for rank, r in enumerate(records, 1):
        print(f"  {rank:3d}. [{r['dataset']:<10s}] {r['fname']:<40s}  "
              f"max_diff={r['max_diff']:.5f}  RMSE={r['rmse']:.5f}  "
              f"T={r['T']:,}  p={r['p']:.4f}  n_anom={r['n_anom']}")

    # --- per-record label stats ---
    for r in records:
        s = _compute_label_stats(r["label"])
        r.update(s)
        r["buffer_ratio"] = (slope_size / s["mean_anom_len"]
                             if s["mean_anom_len"] > 0 else np.nan)

    # compare top-25 % (worst) vs bottom-25 % (best)
    n_split     = max(1, len(records) // 4)
    high_group  = records[:n_split]
    low_group   = records[-n_split:]

    stat_defs = [
        ("T",             "TS length",           "{:.0f}"),
        ("n_anom",        "N anomalies",          "{:.1f}"),
        ("p",             "Anomaly ratio",        "{:.4f}"),
        ("mean_anom_len", "Mean anom length",     "{:.1f}"),
        ("min_anom_len",  "Min anom length",      "{:.1f}"),
        ("mean_gap",      "Mean inter-anom gap",  "{:.1f}"),
        ("min_gap",       "Min inter-anom gap",   "{:.1f}"),
        ("buffer_ratio",  "Buffer / anom length", "{:.2f}"),
    ]

    def _gmean(group, key):
        vals = [r[key] for r in group
                if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else np.nan

    def _fmt(val, fmt_str):
        return "—" if (isinstance(val, float) and np.isnan(val)) else fmt_str.format(val)

    print(f"\n=== Stats: top {n_split} (worst divergence) vs bottom {n_split} (best) ===")
    print(f"  {'Stat':<25s}  {'High div (mean)':>15s}  {'Low div (mean)':>14s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*14}")
    for key, label, fmt_str in stat_defs:
        h = _gmean(high_group, key)
        l = _gmean(low_group,  key)
        print(f"  {label:<25s}  {_fmt(h, fmt_str):>15s}  {_fmt(l, fmt_str):>14s}")

    _plot_worst(records, top_n, ", ".join(datasets), K, slope_size, buffers)


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _plot_worst(records, top_n, dataset, K, slope_size, buffers):
    n_plot = min(top_n, len(records))
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3.2 * n_plot), squeeze=False)

    for row, r in enumerate(records[:n_plot]):
        ax_l, ax_d = axes[row]

        ax_l.plot(r["label"], linewidth=0.7, color="black")
        ds_tag = f"[{r['dataset']}] " if "dataset" in r else ""
        ax_l.set_title(
            f"#{row+1} {ds_tag}{r['fname']}\nT={r['T']:,}  p={r['p']:.4f}  n_anom={r['n_anom']}",
            fontsize=8,
        )
        ax_l.set_xlabel("Time")
        ax_l.set_ylabel("Label")

        ax_d.plot(buffers, r["mc"],         color="seagreen", linewidth=1.5,
                  label=f"MC (K={K}, p=label.mean())")
        ax_d.plot(buffers, r["analytical"], color="crimson",  linewidth=1.5,
                  linestyle="--", label="Analytical")
        ax_d.set_title(f"max|diff|={r['max_diff']:.5f}  RMSE={r['rmse']:.5f}", fontsize=8)
        ax_d.set_xlabel("Buffer")
        ax_d.set_ylabel("E[cumVUS]")
        ax_d.legend(fontsize=8)

    plt.suptitle(
        f"Worst {n_plot} cases — MC vs Analytical denominator\n"
        f"dataset={dataset}  K={K}  slope_size={slope_size}",
        y=1.01, fontsize=10,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive buffer — explore / compute / review")
    parser.add_argument("--mode",       type=str, default="explore",
                        choices=["explore", "compute", "review"])

    # shared
    parser.add_argument("--dataset",    type=str, nargs="+", default=["YAHOO"],
                        metavar="DS",
                        help="Dataset name(s), or 'all' to run every dataset")
    parser.add_argument("--slope_size", type=int, default=128)
    parser.add_argument("--top_n",      type=int, default=5,
                        help="Worst cases to plot (compute / review)")

    # explore only
    parser.add_argument("--detector",   type=str, default=None,
                        help="Detector name (explore, default: first available)")

    # compute only
    parser.add_argument("--K",          type=int, default=100,
                        help="Monte Carlo draws per series (compute, default: 100)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output pickle path (compute, default: experiments/adaptive_buffer/…)")

    # review only
    parser.add_argument("--input",      type=str, nargs="+", default=None,
                        metavar="PKL",
                        help="Input pickle path(s) (review). If omitted, resolved "
                             "from --dataset / --K / --slope_size.")

    args = parser.parse_args()

    # resolve dataset list
    datasets = args.dataset
    if datasets == ["all"]:
        from tsbadparser import TSBADParser as _TSB
        datasets = _TSB().get_datasets()

    if args.mode == "explore":
        run_explore(datasets[0], args.detector, args.slope_size)
    elif args.mode == "compute":
        run_compute(datasets, args.K, args.top_n, args.slope_size, args.output)
    elif args.mode == "review":
        if args.input is not None:
            input_paths = args.input
        else:
            input_paths = [_default_output(ds, args.K, args.slope_size)
                           for ds in datasets]
        run_review(input_paths, args.top_n)
