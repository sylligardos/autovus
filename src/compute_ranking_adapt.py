"""
Stage 1 of the Adjusted-VUS ranking study: compute & save the raw scores.

For every (dataset, time series, detector) it runs the two-step adaptive pipeline
  1. selection  -> per-side per-anomaly b* (b*_left, b*_right) via
                   compute_per_anomaly_buffer(per_side=True), plus the
                   detection taxonomy (early/late/exact/missed) per anomaly
  2. scoring    -> score_adapt with the asymmetric per-side caps (raw VUS, ratio_*,
                   Adjusted VUS skill_*)
plus the community VUS baseline (legacy sylli implementation, same L), and writes
one long-format row per pair to a CSV. Floating-point score dust is floored once
(VUSNumpy.floor_dust) so selection and scoring use the identical cleaned score.
Runtimes (selection / scoring / baseline) are recorded so we can see whether the
method is prohibitively slow.

The CSV is the source of truth reused by analyze_ranking_adapt.py (Stage 2) and by
the paper; it is appended incrementally (crash-safe, progress visible).

Usage:
  conda run -n autovus python src/compute_ranking_adapt.py --dataset all
  conda run -n autovus python src/compute_ranking_adapt.py --dataset Stock
  conda run -n autovus python src/compute_ranking_adapt.py --dataset NAB --test
"""

import argparse
import os
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy
from legacy.sylli_metrics import sylli_get_metrics


def _n_anomalies(label):
    diff = np.diff(np.concatenate([[0], np.asarray(label).astype(int), [0]]))
    return int((diff > 0).sum())


# Fixed schema so every row (incl. no-anomaly rows) has identical columns — the
# incremental CSV append does not realign columns, so they must match exactly.
COLUMNS = [
    'dataset', 'filename', 'detector', 'n_anomalies', 'note',
    'b_star_mean', 'b_star_min', 'b_star_max',
    # per-side buffers (asymmetric selection); b_star_* above is the dominant side
    'b_left_mean', 'b_left_max', 'b_right_mean', 'b_right_max',
    # detection taxonomy counts (early/late/exact/missed) over this series' anomalies
    'n_early', 'n_late', 'n_exact', 'n_missed',
    'vus_adapt', 'mean_vus_adapt', 'vus_full', 'vus_community',
    'ratio_adapt_A', 'ratio_adapt_B', 'ratio_full_A', 'ratio_full_B',
    'skill_adapt_A', 'skill_adapt_B', 'skill_full_A', 'skill_full_B',
    'denom_adapt', 'denom_full',
    'selection_time', 'scoring_time', 'community_time', 'total_time',
]


def compute_ranking(dataset, slope_size, testing, experiment_dir, with_community, resume=False):
    tsb = TSADParser()
    if dataset == 'all':
        filenames, timeseries, labels = tsb.load_timeseries(load_all=True)
    else:
        filenames, timeseries, labels = tsb.load_timeseries(dataset=[dataset])

    if testing:
        filenames, timeseries, labels = filenames[:2], timeseries[:2], labels[:2]

    detectors = tsb.get_detectors()
    experiment_dir = os.path.join('experiments', experiment_dir or 'ranking_adapt')
    os.makedirs(experiment_dir, exist_ok=True)
    out_path = os.path.join(experiment_dir, f'scores_{dataset}.csv')

    # Resume: keep only fully-complete series already in the CSV (each complete
    # series has exactly len(detectors) rows; a series killed mid-write has fewer
    # and is dropped so it gets recomputed cleanly), then skip those series below.
    done = set()
    resume_existing = resume and os.path.exists(out_path)
    if resume_existing:
        prev = pd.read_csv(out_path)
        counts = prev.groupby('filename').size()
        done = set(counts[counts == len(detectors)].index)
        prev[prev['filename'].isin(done)].to_csv(out_path, index=False)  # drop partials
        print(f"Resume: {len(done)} complete series already done; recomputing the rest.")

    keep = [i for i, fn in enumerate(filenames) if fn not in done]
    filenames = [filenames[i] for i in keep]
    timeseries = [timeseries[i] for i in keep]
    labels = [labels[i] for i in keep]
    if not filenames:
        print("Nothing to do — all series already scored.")
        return

    # dataset name per file (filenames encode it)
    meta = tsb.parse_ts_filenames(filenames).reset_index().set_index('filename')
    ds_of = {fn: meta.loc[fn, 'dataset'] for fn in filenames}

    all_scores, _ = tsb.load_scores_and_delete(filenames, timeseries, labels, detectors=detectors)
    if len(all_scores) != len(labels):
        raise ValueError(f'Size mismatch: scores={len(all_scores)}, labels={len(labels)}')

    vus = VUSNumpy(slope_size=slope_size, step=1, cumulative=True, global_mask=False,
                   aggregation='mean', metric='vus_pr')

    n_pairs = len(filenames) * len(detectors)
    pbar = tqdm(total=n_pairs, desc=f'Scoring {dataset}')
    first = not resume_existing  # append to the resumed file (keep its header)

    for i, (fname, label, scores) in enumerate(zip(filenames, labels, all_scores)):
        label = np.asarray(label).ravel()
        n_anom = _n_anomalies(label)
        rows = []
        for det_idx, detector in enumerate(detectors):
            score = np.asarray(scores[det_idx]).ravel()
            row = {c: np.nan for c in COLUMNS}
            row.update({
                'dataset':     ds_of.get(fname, dataset),
                'filename':    fname,
                'detector':    detector,
                'n_anomalies': n_anom,
            })

            if n_anom == 0:
                # No anomaly: adaptive scoring is undefined; record NaNs but keep the row.
                pbar.update(1)
                row['note'] = 'no_anomaly'
                rows.append(row)
                continue

            # Clean floating-point dust ONCE so selection and scoring share the
            # identical score (compute_per_anomaly_buffer would otherwise
            # clean internally but score_adapt would not, desyncing the two steps).
            score = VUSNumpy.floor_dust(score)

            t0 = perf_counter()
            sel = vus.compute_per_anomaly_buffer(label, score, per_side=True, dust_floor=0)
            selection_time = perf_counter() - t0

            b_left  = np.array([r['best_left']['buffer_val']  for r in sel])
            b_right = np.array([r['best_right']['buffer_val'] for r in sel])
            b_star  = np.array([r['best']['buffer_val'] for r in sel])   # dominant side (back-compat)
            mean_vus_adapt = float(np.mean([r['best']['vus_score'] for r in sel]))
            types = [r['detection_type'] for r in sel]

            t1 = perf_counter()
            out = vus.score_adapt(label, score, b_star_left=b_left, b_star_right=b_right)
            scoring_time = perf_counter() - t1

            community_vus, community_time = (np.nan, np.nan)
            if with_community:
                t2 = perf_counter()
                community_vus = sylli_get_metrics(label, score, 'vus', slope_size, existence=True)
                if isinstance(community_vus, tuple):
                    community_vus = community_vus[0]
                community_time = perf_counter() - t2

            rows.append({
                **row,
                'b_star_mean': float(np.mean(b_star)),
                'b_star_min':  int(np.min(b_star)),
                'b_star_max':  int(np.max(b_star)),
                # per-side buffers
                'b_left_mean':  float(np.mean(b_left)),
                'b_left_max':   int(np.max(b_left)),
                'b_right_mean': float(np.mean(b_right)),
                'b_right_max':  int(np.max(b_right)),
                # detection taxonomy counts
                'n_early':  int(types.count('early')),
                'n_late':   int(types.count('late')),
                'n_exact':  int(types.count('exact')),
                'n_missed': int(types.count('missed')),
                # raw VUS
                'vus_adapt':      out['vus_adapt'],
                'mean_vus_adapt': mean_vus_adapt,
                'vus_full':       out['vus_full'],
                'vus_community': community_vus,
                # unbounded ratio diagnostic
                'ratio_adapt_A': out['ratio_adapt_A'],
                'ratio_adapt_B': out['ratio_adapt_B'],
                'ratio_full_A':  out['ratio_full_A'],
                'ratio_full_B':  out['ratio_full_B'],
                # Adjusted VUS (aVUS) — stored UNCLIPPED
                'skill_adapt_A': out['skill_adapt_A'],
                'skill_adapt_B': out['skill_adapt_B'],
                'skill_full_A':  out['skill_full_A'],
                'skill_full_B':  out['skill_full_B'],
                'denom_adapt':   out['denom_adapt'],
                'denom_full':    out['denom_full'],
                # runtimes (seconds)
                'selection_time': selection_time,
                'scoring_time':   scoring_time,
                'community_time': community_time,
                'total_time':     selection_time + scoring_time + (community_time if with_community else 0.0),
            })
            pbar.update(1)

        # incremental, crash-safe append (truncates any prior file on first write)
        df = pd.DataFrame(rows).reindex(columns=COLUMNS)
        df.to_csv(out_path, mode='w' if first else 'a', header=first, index=False)
        first = False

    pbar.close()
    print(f"\nSaved {n_pairs} rows to {os.path.abspath(out_path)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stage 1: compute Adjusted-VUS ranking scores')
    p.add_argument('--dataset', type=str, default='all', help="Dataset name or 'all'")
    p.add_argument('--slope-size', type=int, default=256, help='Buffer L (default 256)')
    p.add_argument('--test', action='store_true', help='Limit to 2 series for a quick check')
    p.add_argument('--experiment', type=str, default=None, help='Output subdir under experiments/')
    p.add_argument('--no-community', action='store_true', help='Skip the legacy VUS baseline')
    p.add_argument('--resume', action='store_true', help='Skip series already present in the output CSV')
    args = p.parse_args()

    compute_ranking(
        dataset=args.dataset,
        slope_size=args.slope_size,
        testing=args.test,
        experiment_dir=args.experiment,
        with_community=not args.no_community,
        resume=args.resume,
    )
