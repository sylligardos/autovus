"""
Effect of capping the pr_relevant_fuzzy threshold set vs uniform subsampling.

For each sampled (time series, detector) pair, two strategies are evaluated at
several cap sizes C, plus an uncapped reference:

  all_unique        (uncapped, reference)
  subsampled        at C ∈ CAP_VALUES          — uniform subsample of all unique scores
  pr_relevant_fuzzy at C ∈ CAP_VALUES + [None] — fuzzy-prune then TP-space cap to C
                                                  None = uncapped, must match all_unique

Two operating points per (strategy, cap):
  slope_size=0,   global_mask=False, existence=None   → must match sklearn APS
  slope_size=128, global_mask=True,  existence=optimized → accuracy vs all_unique + speedup

For torch: pr_relevant_fuzzy at caps {None, 1000}, slope_size=128 only (cross-validation).

Series selection: for each dataset, up to N_SERIES_PER_STRATUM series per T-stratum
(low / mid / high length), so we cover a wide range of series sizes cheaply.

Output: experiments/compare_threshold_cap/results.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
import torch

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

SLOPE_SIZE   = 128
N_SERIES_PER_STRATUM = 1   # series picked per (dataset, T-stratum)

CAP_VALUES = [50, 100, 250, 500, 1000, 2000, 5000]

# (strategy, cap) pairs to evaluate.  None cap = no limit.
# subsampled with None = identical to all_unique, so we omit it.
STRATEGY_CAP_PAIRS = (
    [('all_unique',          None)] +
    [('subsampled',          c) for c in CAP_VALUES] +
    [('pr_relevant_fuzzy',   c) for c in CAP_VALUES] +
    [('pr_relevant_fuzzy', None)]
)

# Torch: only these (strategy, cap) combos, slope_size=128 only
TORCH_PAIRS = [
    ('pr_relevant_fuzzy', None),
    ('pr_relevant_fuzzy', 1000),
]

OUTPUT_DIR  = os.path.join('experiments', 'compare_threshold_cap')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

CSV_COLUMNS = [
    'filename', 'dataset', 'T', 'anomaly_rate', 'n_anomalies', 'n_unique_scores_full',
    'detector', 'backend', 'slope_size', 'strategy', 'cap',
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


def _select_series(fnames, labels, n_per_stratum):
    """Pick up to n_per_stratum series from each T-stratum (low/mid/high)."""
    lengths = np.array([len(l) for l in labels])
    if len(lengths) == 0:
        return []
    p33, p67 = np.percentile(lengths, [33, 67])
    strata = np.digitize(lengths, [p33, p67])  # 0=low, 1=mid, 2=high
    selected = []
    for s in range(3):
        idxs = np.where(strata == s)[0]
        if len(idxs) == 0:
            continue
        # pick the median-length series within the stratum
        median_len = np.median(lengths[idxs])
        pick = idxs[np.argsort(np.abs(lengths[idxs] - median_len))[:n_per_stratum]]
        selected.extend(pick.tolist())
    return sorted(set(selected))


def _output_csv(dataset=None):
    if dataset is not None:
        return os.path.join(RESULTS_DIR, f'results_{dataset}.csv')
    return os.path.join(RESULTS_DIR, 'results.csv')


def _load_all_results():
    """Load all results_*.csv (per-dataset) plus results.csv (local run) if present."""
    import glob
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'results*.csv')))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype={'cap': str}))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_done(csv_path):
    """Return set of (filename, detector, backend, slope_size, strategy, cap_str) in CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={'cap': str}, keep_default_na=False,
                         usecols=['filename', 'detector', 'backend', 'slope_size', 'strategy', 'cap'])
        return set(zip(df['filename'], df['detector'], df['backend'],
                       df['slope_size'], df['strategy'], df['cap']))
    except Exception:
        return set()


def _cap_str(cap):
    return 'inf' if cap is None else str(cap)


# ---------------------------------------------------------------------------
# VUSNumpy / VUSTorch instance cache
# ---------------------------------------------------------------------------

def _make_numpy_instances():
    """One VUSNumpy per (strategy, cap, slope_size)."""
    instances = {}
    for strategy, cap in STRATEGY_CAP_PAIRS:
        for slope_size in [0, SLOPE_SIZE]:
            key = (strategy, cap, slope_size)
            instances[key] = VUSNumpy(
                slope_size    = slope_size,
                step          = 1,
                global_mask   = (slope_size > 0),
                slopes        = 'function',
                existence     = 'optimized' if slope_size > 0 else 'None',
                conf_matrix   = 'dynamic_plus',
                thresholds    = strategy,
                max_thresholds = cap,
            )
    return instances


def _make_torch_instances():
    """One VUSTorch per (strategy, cap) — slope_size=128 only."""
    instances = {}
    for strategy, cap in TORCH_PAIRS:
        key = (strategy, cap)
        instances[key] = VUSTorch(
            slope_size     = SLOPE_SIZE,
            step           = 1,
            global_mask    = True,
            thresholds     = strategy,
            max_thresholds = cap,
            device         = 'cpu',
        )
    return instances


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

def _prep_correctness(df):
    ref = (
        df[df['strategy'] == 'all_unique'][['filename', 'detector', 'vus_value']]
        .rename(columns={'vus_value': 'ref_vus'})
    )
    df = df.merge(ref, on=['filename', 'detector'], how='left')
    df['err_sklearn']    = (df['vus_value'] - df['sklearn_aps']).abs()
    df['err_all_unique'] = (df['vus_value'] - df['ref_vus']).abs()
    return df


def _prep_accuracy(df):
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


def _pair_label(strategy, cap):
    return f"{strategy}@{cap if cap is not None else 'inf'}"


def _print_correctness_table(df, label, indent=0):
    pad = ' ' * indent
    n_series = df['filename'].nunique()
    n_det    = df['detector'].nunique()
    print(f"\n{pad}{label}  ({n_series} series, {n_det} detector{'s' if n_det != 1 else ''})")
    hdr = f"{pad}  {'strategy@cap':<32} {'pairs':>6} {'pass_sklearn%':>14} {'pass_ref%':>10} {'mean_err':>12} {'max_err':>12}"
    print(hdr)
    print(f"{pad}  " + "-" * 86)
    for strategy, cap in STRATEGY_CAP_PAIRS:
        if strategy == 'subsampled' and slope_size_label(strategy, cap) is None:
            continue
        sub = df[(df['strategy'] == strategy) & (df['cap'] == _cap_str(cap))]
        if sub.empty:
            continue
        lbl_str  = _pair_label(strategy, cap)
        n        = len(sub)
        p_sk     = (sub['err_sklearn']    < 1e-5).mean() * 100
        p_ref    = (sub['err_all_unique'] < 1e-5).mean() * 100
        mean_err = sub['err_sklearn'].mean()
        max_err  = sub['err_sklearn'].max()
        print(f"{pad}  {lbl_str:<32} {n:>6} {p_sk:>13.1f}% {p_ref:>9.1f}% {mean_err:>12.2e} {max_err:>12.2e}")


def slope_size_label(strategy, cap):
    # subsampled with None cap is excluded
    if strategy == 'subsampled' and cap is None:
        return None
    return _pair_label(strategy, cap)


def _print_accuracy_table(df, label, indent=0):
    pad = ' ' * indent
    n_series = df['filename'].nunique()
    n_det    = df['detector'].nunique()
    print(f"\n{pad}{label}  ({n_series} series, {n_det} detector{'s' if n_det != 1 else ''})")
    hdr = f"{pad}  {'strategy@cap':<32} {'pairs':>6} {'compress':>9} {'speedup_med':>12} {'pass%':>7} {'mean_err':>12} {'max_err':>12}"
    print(hdr)
    print(f"{pad}  " + "-" * 98)
    for strategy, cap in STRATEGY_CAP_PAIRS:
        if strategy == 'all_unique':
            continue
        sub = df[(df['strategy'] == strategy) & (df['cap'] == _cap_str(cap))]
        if sub.empty:
            continue
        lbl_str    = _pair_label(strategy, cap)
        n          = len(sub)
        compress   = sub['compression'].mean()
        speedup    = np.median(sub['speedup'])
        p_err      = (sub['err_ref'] < 1e-5).mean() * 100
        mean_err   = sub['err_ref'].mean()
        max_err    = sub['err_ref'].max()
        print(f"{pad}  {lbl_str:<32} {n:>6} {compress:>8.3f}x {speedup:>11.2f}x {p_err:>6.1f}% {mean_err:>12.2e} {max_err:>12.2e}")


def _print_nthresh_table(df):
    """Median n_thresholds by n_unique_scores_full bin and (strategy, cap)."""
    bins   = [0, 100, 1000, 10000, np.inf]
    labels = ['<100', '100-1K', '1K-10K', '>10K']
    df = df.copy()
    df['score_bin'] = pd.cut(df['n_unique_scores_full'], bins=bins, labels=labels, right=False)

    print("\nMEDIAN n_thresholds  by score-count bin and (strategy, cap)  [slope_size=128, numpy]")
    pairs_to_show = [('all_unique', None)] + [('pr_relevant_fuzzy', c) for c in [None] + CAP_VALUES] + [('subsampled', c) for c in CAP_VALUES]
    header = f"  {'strategy@cap':<32}" + "".join(f"{b:>9}" for b in labels)
    print(header)
    print("  " + "-" * (32 + 9 * len(labels)))
    for strategy, cap in pairs_to_show:
        sub = df[(df['strategy'] == strategy) & (df['cap'] == _cap_str(cap))]
        if sub.empty:
            continue
        row = f"  {_pair_label(strategy, cap):<32}"
        for b in labels:
            vals = sub[sub['score_bin'] == b]['n_thresholds']
            row += f"{int(np.median(vals)):>9}" if not vals.empty else f"{'—':>9}"
        print(row)


def review():
    df = _load_all_results()
    if df.empty:
        print("Results file is empty.")
        return

    datasets = sorted(df['dataset'].unique())

    print("\n" + "=" * 106)
    print("  REVIEW — threshold cap experiment")
    print("=" * 106)

    # ---- Correctness (slope_size=0, numpy) ----
    df0 = df[(df['slope_size'] == 0) & (df['backend'] == 'numpy')].copy()
    if not df0.empty:
        df0 = _prep_correctness(df0)
        print("\nCORRECTNESS  slope_size=0 | global_mask=False | existence=None | backend=numpy")
        _print_correctness_table(df0, "Overall")
        for ds in datasets:
            _print_correctness_table(df0[df0['dataset'] == ds], ds, indent=2)

    # ---- Accuracy (slope_size=128, numpy) ----
    dfL = df[(df['slope_size'] == SLOPE_SIZE) & (df['backend'] == 'numpy')].copy()
    if not dfL.empty:
        dfL = _prep_accuracy(dfL)
        print(f"\nACCURACY  slope_size={SLOPE_SIZE} | global_mask=True | existence=optimized | backend=numpy")
        _print_accuracy_table(dfL, "Overall")
        for ds in datasets:
            _print_accuracy_table(dfL[dfL['dataset'] == ds], ds, indent=2)

        # ---- Threshold count by score bin ----
        _print_nthresh_table(dfL)

    # ---- Torch cross-check ----
    dfT = df[(df['slope_size'] == SLOPE_SIZE) & (df['backend'] == 'torch')].copy()
    if not dfT.empty:
        dfN = df[(df['slope_size'] == SLOPE_SIZE) & (df['backend'] == 'numpy')].copy()
        print(f"\nTORCH CROSS-CHECK  slope_size={SLOPE_SIZE}")
        hdr = f"  {'strategy@cap':<32} {'pairs':>6} {'mean|torch-numpy|':>18} {'max|torch-numpy|':>17}"
        print(hdr)
        print("  " + "-" * 75)
        for strategy, cap in TORCH_PAIRS:
            st = dfT[(dfT['strategy'] == strategy) & (dfT['cap'] == _cap_str(cap))]
            sn = dfN[(dfN['strategy'] == strategy) & (dfN['cap'] == _cap_str(cap))]
            merged = st[['filename', 'detector', 'vus_value']].merge(
                sn[['filename', 'detector', 'vus_value']], on=['filename', 'detector'],
                suffixes=('_torch', '_numpy')
            )
            if merged.empty:
                continue
            diff = (merged['vus_value_torch'] - merged['vus_value_numpy']).abs()
            lbl = _pair_label(strategy, cap)
            print(f"  {lbl:<32} {len(merged):>6} {diff.mean():>18.2e} {diff.max():>17.2e}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slope_size',        type=int, default=SLOPE_SIZE)
    parser.add_argument('--n_per_stratum',     type=int, default=N_SERIES_PER_STRATUM,
                        help='Series per T-stratum per dataset (default: 1)')
    parser.add_argument('--dataset',           type=str, default=None,
                        help='Restrict to one dataset name')
    parser.add_argument('--detector',          type=str, default=None,
                        help='Restrict to one detector name')
    parser.add_argument('--force',             action='store_true',
                        help='Delete existing results and recompute')
    parser.add_argument('--review',            action='store_true',
                        help='Print review of existing results and exit')
    parser.add_argument('--experiment',        type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.review:
        review()
        return

    output_csv = _output_csv(args.dataset)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.force and os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"Removed {output_csv}")

    done = _load_done(output_csv)
    if done:
        print(f"Resuming: {len(done)} rows already in CSV, skipping those.")

    # ---- Load and sample data ----
    tsb = TSADParser()
    all_dataset_names = tsb.get_datasets()
    all_detectors     = tsb.get_detectors()

    dataset_filter   = [args.dataset] if args.dataset else all_dataset_names
    detector_filter  = [args.detector] if args.detector else all_detectors

    # Build instance caches once
    numpy_instances = _make_numpy_instances()
    torch_instances = _make_torch_instances()
    slope_sizes     = [0, args.slope_size]
    runs_per_pair   = len(STRATEGY_CAP_PAIRS) * len(slope_sizes) + len(TORCH_PAIRS)

    print(f"Processing {len(dataset_filter)} dataset(s), {args.n_per_stratum} series/stratum, "
          f"{len(detector_filter)} detectors, {runs_per_pair} runs/pair ...")

    # Process one dataset at a time to avoid cross-dataset shape-mismatch in
    # load_scores_and_delete (which asserts homogeneous lengths across the batch).
    for ds in dataset_filter:
        ds_fnames, ds_ts, ds_labels = tsb.load_timeseries(dataset=[ds])
        if not ds_fnames:
            continue

        idxs = _select_series(ds_fnames, ds_labels, args.n_per_stratum)
        fn2 = [ds_fnames[i] for i in idxs]
        ts2 = [ds_ts[i]     for i in idxs]
        lb2 = [ds_labels[i] for i in idxs]

        if not fn2:
            continue

        try:
            scores_grouped, _ = tsb.load_scores_and_delete(fn2, ts2, lb2, detectors=detector_filter)
        except Exception as e:
            tqdm.write(f"SKIP dataset {ds}: load_scores_and_delete failed: {e}")
            continue

        total_work = len(fn2) * len(detector_filter) * runs_per_pair
        with tqdm(total=total_work, desc=ds, leave=False) as pbar:
            for i, (fname, label) in enumerate(zip(fn2, lb2)):
                scores_row   = scores_grouped[i]
                n_anom       = _n_anomalies(label)
                anomaly_rate = float(label.mean())

                for j, detector in enumerate(detector_filter):
                    score   = scores_row[j]
                    min_len = min(len(label), len(score))
                    lbl     = label[:min_len]
                    scr     = score[:min_len]

                    if lbl.sum() == 0:
                        pbar.update(runs_per_pair)
                        continue

                    n_unique_full = int(np.unique(scr).shape[0])
                    sklearn_aps   = float(average_precision_score(lbl, scr))

                    # ---- numpy: all (strategy, cap) × both slope sizes ----
                    for slope_size in slope_sizes:
                        for strategy, cap in STRATEGY_CAP_PAIRS:
                            pbar.update(1)
                            key_done = (fname, detector, 'numpy', slope_size, strategy, _cap_str(cap))
                            if key_done in done:
                                continue
                            try:
                                vus = numpy_instances[(strategy, cap, slope_size)]
                                (vus_val, time_analysis), total_time = vus.compute(lbl, scr)
                            except Exception as e:
                                tqdm.write(f"ERROR numpy {fname}/{detector}/{strategy}@{cap}/L={slope_size}: {e}")
                                continue

                            row = {
                                'filename':             fname,
                                'dataset':              ds,
                                'T':                    min_len,
                                'anomaly_rate':         anomaly_rate,
                                'n_anomalies':          n_anom,
                                'n_unique_scores_full': n_unique_full,
                                'detector':             detector,
                                'backend':              'numpy',
                                'slope_size':           slope_size,
                                'strategy':             strategy,
                                'cap':                  _cap_str(cap),
                                'n_thresholds':         int(time_analysis['n_thresholds']),
                                'thresh_time':          float(time_analysis['Thresholds time']),
                                'total_time':           float(total_time),
                                'vus_value':            float(vus_val),
                                'sklearn_aps':          sklearn_aps if slope_size == 0 else float('nan'),
                            }
                            write_header = not os.path.exists(output_csv)
                            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=write_header, index=False)
                            done.add(key_done)

                    # ---- torch: pr_relevant_fuzzy at {None, 1000}, slope_size=128 only ----
                    lbl_t = torch.tensor(lbl, dtype=torch.float32)
                    scr_t = torch.tensor(scr, dtype=torch.float32)
                    for strategy, cap in TORCH_PAIRS:
                        pbar.update(1)
                        key_done = (fname, detector, 'torch', args.slope_size, strategy, _cap_str(cap))
                        if key_done in done:
                            continue
                        try:
                            vus_t = torch_instances[(strategy, cap)]
                            (vus_val_t, time_analysis_t), total_time_t = vus_t.compute(lbl_t, scr_t)
                            vus_val_t = float(vus_val_t)
                        except Exception as e:
                            tqdm.write(f"ERROR torch {fname}/{detector}/{strategy}@{cap}: {e}")
                            continue

                        row = {
                            'filename':             fname,
                            'dataset':              ds,
                            'T':                    min_len,
                            'anomaly_rate':         anomaly_rate,
                            'n_anomalies':          n_anom,
                            'n_unique_scores_full': n_unique_full,
                            'detector':             detector,
                            'backend':              'torch',
                            'slope_size':           args.slope_size,
                            'strategy':             strategy,
                            'cap':                  _cap_str(cap),
                            'n_thresholds':         int(time_analysis_t.get('n_thresholds', -1)),
                            'thresh_time':          float(time_analysis_t.get('Thresholds time', 0.0)),
                            'total_time':           float(total_time_t),
                            'vus_value':            vus_val_t,
                            'sklearn_aps':          float('nan'),
                        }
                        write_header = not os.path.exists(output_csv)
                        pd.DataFrame([row]).to_csv(output_csv, mode='a', header=write_header, index=False)
                        done.add(key_done)

    review()


if __name__ == '__main__':
    main()
