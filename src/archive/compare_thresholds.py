"""
Compare threshold selection strategies across the full TSB-AD benchmark.

For each (time series, detector) pair, every strategy is evaluated at two
operating points:

  slope_size=0, global_mask=False, existence=None
    → correctness: VUS must equal sklearn APS (gold standard)

  slope_size=SLOPE_SIZE, global_mask=True, existence=optimized
    → timing: speedup and accuracy in a real VUS run

One CSV row is written per (filename, detector, strategy, slope_size).
Results accumulate across runs; use --force to overwrite.

Output: experiments/compare_threshold_methods/results.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy

STRATEGIES     = ['all_unique', 'subsampled', 'pr_relevant', 'pr_relevant_fuzzy']
TEST_DATASET   = 'NAB'
TEST_DETECTOR  = 'AnomalyTransformer'
TEST_N_SERIES  = 20

# Cap used by the 'subsampled' strategy. None = no subsampling (identical to
# 'all_unique'). Picked so subsampled produces a meaningfully smaller set
# than all_unique on typical YAHOO/NAB series (which often have a few
# hundred unique scores).
MAX_THRESHOLDS_SUBSAMPLED = 100

OUTPUT_DIR = os.path.join('experiments', 'compare_threshold_methods')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'results.csv')

CSV_COLUMNS = [
    'filename', 'dataset', 'T', 'anomaly_rate', 'n_anomalies', 'n_unique_scores',
    'detector', 'slope_size', 'strategy',
    'n_thresholds', 'thresh_time', 'total_time', 'vus_value', 'sklearn_aps',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_anomalies(label):
    diff   = np.diff(label)
    starts = np.where(diff == 1)[0] + 1
    if len(label) > 0 and label[0]:
        starts = np.r_[0, starts]
    return int(len(starts))


def _dataset_from_fname(fname, dataset_names):
    fname_upper = fname.upper()
    for ds in dataset_names:
        if ds.upper() in fname_upper:
            return ds
    return 'unknown'


def _make_vus_instances(slope_size):
    """Build one VUSNumpy per (strategy, mode) — reused across all pairs."""
    instances = {}
    for strategy in STRATEGIES:
        max_thresh = MAX_THRESHOLDS_SUBSAMPLED if strategy == 'subsampled' else None
        instances[(strategy, 0)] = VUSNumpy(
            slope_size=0, step=1, global_mask=False,
            slopes='function', existence='None', conf_matrix='dynamic_plus',
            thresholds=strategy, max_thresholds=max_thresh,
        )
        instances[(strategy, slope_size)] = VUSNumpy(
            slope_size=slope_size, step=1, global_mask=True,
            slopes='function', existence='optimized', conf_matrix='dynamic_plus',
            thresholds=strategy, max_thresholds=max_thresh,
        )
    return instances


def _load_done(csv_path):
    """Return set of (filename, detector, strategy, slope_size) already in CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=['filename', 'detector', 'strategy', 'slope_size'])
        return set(zip(df['filename'], df['detector'], df['strategy'], df['slope_size']))
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

def _prep_correctness(df):
    """Attach err_sklearn and err_all_unique columns to a slope_size=0 subset."""
    ref = (
        df[df['strategy'] == 'all_unique'][['filename', 'detector', 'vus_value']]
        .rename(columns={'vus_value': 'ref_vus'})
    )
    df = df.merge(ref, on=['filename', 'detector'], how='left')
    df['err_sklearn']    = (df['vus_value'] - df['sklearn_aps']).abs()
    df['err_all_unique'] = (df['vus_value'] - df['ref_vus']).abs()
    return df


def _prep_timing(df):
    """Attach speedup, err_ref, and compression columns to a timing subset."""
    ref_t = (
        df[df['strategy'] == 'all_unique'][['filename', 'detector', 'total_time', 'n_thresholds']]
        .rename(columns={'total_time': 'ref_time', 'n_thresholds': 'ref_n_thresh'})
    )
    ref_v = (
        df[df['strategy'] == 'all_unique'][['filename', 'detector', 'vus_value']]
        .rename(columns={'vus_value': 'ref_vus'})
    )
    df = df.merge(ref_t, on=['filename', 'detector'], how='left')
    df = df.merge(ref_v, on=['filename', 'detector'], how='left')
    df['speedup']     = df['ref_time'] / df['total_time']
    df['err_ref']     = (df['vus_value'] - df['ref_vus']).abs()
    df['compression'] = df['n_thresholds'] / df['ref_n_thresh']
    return df


def _print_correctness_table(df, label, indent=0):
    pad = ' ' * indent
    n_series = df['filename'].nunique()
    n_det    = df['detector'].nunique()
    print(f"\n{pad}{label}  ({n_series} series, {n_det} detector{'s' if n_det != 1 else ''})")
    hdr = f"{pad}  {'Strategy':<22} {'pairs':>6} {'pass_sklearn%':>14} {'pass_ref%':>10} {'mean_err':>12} {'max_err':>12}"
    print(hdr)
    print(f"{pad}  " + "-" * 74)
    for strat in STRATEGIES:
        sub = df[df['strategy'] == strat]
        if sub.empty:
            continue
        n         = len(sub)
        p_sklearn = (sub['err_sklearn']    < 1e-5).mean() * 100
        p_ref     = (sub['err_all_unique'] < 1e-5).mean() * 100
        mean_err  = sub['err_sklearn'].mean()
        max_err   = sub['err_sklearn'].max()
        print(f"{pad}  {strat:<22} {n:>6} {p_sklearn:>13.1f}% {p_ref:>9.1f}% {mean_err:>12.2e} {max_err:>12.2e}")


def _print_timing_table(df, label, indent=0):
    pad = ' ' * indent
    n_series = df['filename'].nunique()
    n_det    = df['detector'].nunique()
    print(f"\n{pad}{label}  ({n_series} series, {n_det} detector{'s' if n_det != 1 else ''})")
    hdr = f"{pad}  {'Strategy':<22} {'compress':>9} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'mean':>8} {'pass%':>7}"
    print(hdr)
    print(f"{pad}  " + "-" * 81)
    for strat in STRATEGIES:
        sub = df[df['strategy'] == strat]
        if sub.empty:
            continue
        compression          = sub['compression'].mean()
        p25, p50, p75, p95   = np.percentile(sub['speedup'], [25, 50, 75, 95])
        mean_spd             = sub['speedup'].mean()
        p_err                = (sub['err_ref'] < 1e-5).mean() * 100
        print(f"{pad}  {strat:<22} {compression:>8.3f}x {p25:>7.2f}x {p50:>7.2f}x {p75:>7.2f}x {p95:>7.2f}x {mean_spd:>7.2f}x {p_err:>6.1f}%")


STRAT_COLORS = {
    'all_unique':        '#9E9E9E',
    'subsampled':        '#4CAF50',
    'pr_relevant':       '#FF9800',
    'pr_relevant_fuzzy': '#9C27B0',
}


def _bar_group(ax, datasets, values_by_strat, strats, ylabel, title, ylim=None, hline=None):
    """Grouped bar chart: one group per dataset, one bar per strategy."""
    x     = np.arange(len(datasets))
    width = min(0.8 / len(strats), 0.25)
    offset = (len(strats) - 1) / 2

    for i, strat in enumerate(strats):
        vals = [values_by_strat[strat].get(ds, np.nan) for ds in datasets]
        ax.bar(x + (i - offset) * width, vals, width,
               label=strat, color=STRAT_COLORS[strat], alpha=0.85)

    if hline is not None:
        ax.axhline(hline, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)


def _boxplot(ax, data_by_strat, strats, ylabel, title, log_scale=False, hline=None):
    """Box plot: one box per strategy, outliers hidden."""
    data   = [data_by_strat[s] for s in strats if s in data_by_strat]
    labels = [s for s in strats if s in data_by_strat]
    colors = [STRAT_COLORS[s] for s in labels]

    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5,
        medianprops=dict(color='black', linewidth=2),
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    if hline is not None:
        ax.axhline(hline, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=30)


def plot_review(csv_path=OUTPUT_CSV):
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    timing_slopes = sorted(s for s in df['slope_size'].unique() if s != 0)
    datasets      = sorted(df['dataset'].unique())

    # ---- Figure 1: Correctness (slope_size=0) ----
    df0 = df[df['slope_size'] == 0].copy()
    if not df0.empty:
        df0 = _prep_correctness(df0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Correctness  —  slope_size=0 | global_mask=False | existence=None', fontsize=11)

        # Panel 1: pass_sklearn% per strategy per dataset
        pass_by_strat = {}
        for strat in STRATEGIES:
            pass_by_strat[strat] = {}
            for ds in datasets:
                sub = df0[(df0['dataset'] == ds) & (df0['strategy'] == strat)]
                pass_by_strat[strat][ds] = (sub['err_sklearn'] < 1e-5).mean() * 100 if not sub.empty else np.nan
        _bar_group(axes[0], datasets, pass_by_strat, STRATEGIES,
                   ylabel='Pass rate (%)', title='Pass rate vs sklearn APS',
                   ylim=(0, 108), hline=100)

        # Panel 2: |err vs sklearn| distribution per strategy (log, zeros dropped)
        err_by_strat = {}
        for strat in STRATEGIES:
            vals = df0[df0['strategy'] == strat]['err_sklearn']
            vals = vals[vals > 0].dropna()
            if not vals.empty:
                err_by_strat[strat] = vals.values
        _boxplot(axes[1], err_by_strat, STRATEGIES,
                 ylabel='|error vs sklearn APS|  (log scale, zeros excluded)',
                 title='Error distribution', log_scale=True)

        plt.tight_layout()

    # ---- Figure 2+: Timing (one figure per non-zero slope_size) ----
    for slope_size in timing_slopes:
        dfL = _prep_timing(df[df['slope_size'] == slope_size].copy())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Timing  —  slope_size={slope_size} | global_mask=True | existence=optimized', fontsize=11)

        # Panel 1: speedup distribution per strategy (exclude all_unique — always 1x)
        non_ref = [s for s in STRATEGIES if s != 'all_unique']
        spd_by_strat = {}
        for strat in non_ref:
            vals = dfL[dfL['strategy'] == strat]['speedup'].dropna()
            if not vals.empty:
                spd_by_strat[strat] = vals.values
        _boxplot(axes[0], spd_by_strat, non_ref,
                 ylabel='Speedup vs all_unique',
                 title='Speedup distribution  (all_unique = 1×, dashed)',
                 hline=1.0)

        # Panel 2: match rate vs all_unique per strategy per dataset
        match_by_strat = {}
        for strat in non_ref:
            match_by_strat[strat] = {}
            for ds in datasets:
                sub = dfL[(dfL['dataset'] == ds) & (dfL['strategy'] == strat)]
                match_by_strat[strat][ds] = (sub['err_ref'] < 1e-5).mean() * 100 if not sub.empty else np.nan
        _bar_group(axes[1], datasets, match_by_strat, non_ref,
                   ylabel='Match rate vs all_unique (%)',
                   title='Correctness at real slope_size',
                   ylim=(0, 108), hline=100)

        plt.tight_layout()

    plt.show()


def review(csv_path=OUTPUT_CSV):
    if not os.path.exists(csv_path):
        print(f"No results file at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Results file is empty.")
        return

    timing_slopes = sorted(s for s in df['slope_size'].unique() if s != 0)
    datasets      = sorted(df['dataset'].unique())

    print("\n" + "=" * 92)
    print("  REVIEW")
    print("=" * 92)

    # ---- Correctness ----
    df0 = df[df['slope_size'] == 0].copy()
    if not df0.empty:
        df0 = _prep_correctness(df0)
        print("\nCORRECTNESS  slope_size=0 | global_mask=False | existence=None")
        _print_correctness_table(df0, "Overall")
        for ds in datasets:
            _print_correctness_table(df0[df0['dataset'] == ds], ds, indent=2)

    # ---- Timing ----
    for slope_size in timing_slopes:
        dfL = _prep_timing(df[df['slope_size'] == slope_size].copy())
        print(f"\nTIMING  slope_size={slope_size} | global_mask=True | existence=optimized")
        _print_timing_table(dfL, "Overall")
        for ds in datasets:
            _print_timing_table(dfL[dfL['dataset'] == ds], ds, indent=2)

    print()
    plot_review(csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slope_size', type=int,  default=128,  help='Slope size for the timing run (default: 128)')
    parser.add_argument('--dataset',    type=str,  default=None, help='Dataset name; None = all')
    parser.add_argument('--detector',   type=str,  default=None, help='Detector name; None = all')
    parser.add_argument('--test',       action='store_true',     help=f'Quick test: {TEST_DATASET} / {TEST_DETECTOR} / {TEST_N_SERIES} series')
    parser.add_argument('--force',      action='store_true',     help='Overwrite existing results file')
    parser.add_argument('--review',     action='store_true',     help='Print review of existing results and exit')
    args = parser.parse_args()

    if args.review:
        review()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.force and os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"Removed {OUTPUT_CSV}")

    done = _load_done(OUTPUT_CSV)
    if done:
        print(f"Resuming: {len(done)} rows already in CSV, skipping those.")

    # ---- Load data ----
    tsb = TSBADParser()
    all_dataset_names = tsb.get_datasets()

    dataset_filter = [TEST_DATASET] if args.test else ([args.dataset] if args.dataset else None)
    if dataset_filter:
        fnames, ts_list, labels = tsb.load_timeseries(dataset=dataset_filter)
    else:
        fnames, ts_list, labels = tsb.load_timeseries(load_all=True)

    all_detectors = tsb.get_detectors()
    if args.test:
        detector_filter = [TEST_DETECTOR]
    elif args.detector:
        detector_filter = [args.detector]
    else:
        detector_filter = all_detectors

    if args.test:
        fnames, ts_list, labels = fnames[:TEST_N_SERIES], ts_list[:TEST_N_SERIES], labels[:TEST_N_SERIES]

    # load_scores_and_delete mutates fnames / ts_list / labels in-place (removes bad entries)
    scores_grouped, _ = tsb.load_scores_and_delete(fnames, ts_list, labels, detectors=detector_filter)

    slope_sizes = [0, args.slope_size]
    vus_instances = _make_vus_instances(args.slope_size)

    total_work = len(fnames) * len(detector_filter) * len(STRATEGIES) * len(slope_sizes)
    print(f"Work: {len(fnames)} series × {len(detector_filter)} detectors × "
          f"{len(STRATEGIES)} strategies × {len(slope_sizes)} slope_sizes = {total_work} runs")

    with tqdm(total=total_work, desc='Overall') as pbar:
        for i, (fname, label) in enumerate(zip(fnames, labels)):
            scores_row   = scores_grouped[i]   # (n_det, T)
            dataset_name = _dataset_from_fname(fname, all_dataset_names)
            n_anom       = _n_anomalies(label)
            anomaly_rate = float(label.mean())

            for j, detector in enumerate(detector_filter):
                score   = scores_row[j]
                min_len = min(len(label), len(score))
                lbl     = label[:min_len]
                scr     = score[:min_len]

                if lbl.sum() == 0:
                    pbar.update(len(STRATEGIES) * len(slope_sizes))
                    continue

                n_unique    = int(np.unique(scr).shape[0])
                sklearn_aps = float(average_precision_score(lbl, scr))

                for slope_size in slope_sizes:
                    for strategy in STRATEGIES:
                        pbar.update(1)

                        key = (fname, detector, strategy, slope_size)
                        if key in done:
                            continue

                        try:
                            vus = vus_instances[(strategy, slope_size)]
                            (vus_val, time_analysis), total_time = vus.compute(lbl, scr)
                        except Exception as e:
                            tqdm.write(f"ERROR {fname} / {detector} / {strategy} / L={slope_size}: {e}")
                            continue

                        row = {
                            'filename':        fname,
                            'dataset':         dataset_name,
                            'T':               min_len,
                            'anomaly_rate':    anomaly_rate,
                            'n_anomalies':     n_anom,
                            'n_unique_scores': n_unique,
                            'detector':        detector,
                            'slope_size':      slope_size,
                            'strategy':        strategy,
                            'n_thresholds':    int(time_analysis['n_thresholds']),
                            'thresh_time':     float(time_analysis['Thresholds time']),
                            'total_time':      float(total_time),
                            'vus_value':       float(vus_val),
                            'sklearn_aps':     sklearn_aps if slope_size == 0 else float('nan'),
                        }

                        write_header = not os.path.exists(OUTPUT_CSV)
                        pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                        done.add(key)

    review()


if __name__ == '__main__':
    main()
