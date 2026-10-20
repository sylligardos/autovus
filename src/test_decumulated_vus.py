"""
Decumulated VUS — quick visual test.

Loads time series + detector scores from TSB-AD, runs cumulative VUS, and
plots, for each (series, detector):

  1. the time series (with anomalies shaded)
  2. the detector score
  3. the ORIGINAL cumulative VUS-per-buffer curve   C[b] = mean_{j<=b} A[j]
  4. the DECUMULATED per-buffer VUS                 A[b]      (inverts the mean)
  5. the per-step incremental gain                  A[b]-A[b-1]

Panels 4 and 5 are the two candidate "decumulated" views: A[b] removes the
running-mean smear; diff(A) localizes where each buffer first earns its reward.

Usage:
  conda run -n autovus python src/test_decumulated_vus.py
  conda run -n autovus python src/test_decumulated_vus.py --dataset NAB --detector IForest
  conda run -n autovus python src/test_decumulated_vus.py --n-series 3 --slope-size 200 --show
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vus.vus_numpy import VUSNumpy


def _anomaly_bounds(label):
    diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends = np.where(diff < 0)[0] - 1
    return starts, ends


def plot_decumulated(fname, detector, ts, score, label, slope_values,
                     vus_curve, marginal, increment, out_path, show):
    starts, ends = _anomaly_bounds(label)
    T = len(ts)

    fig, axes = plt.subplots(5, 1, figsize=(14, 11))
    ax_ts, ax_sc, ax_cum, ax_marg, ax_inc = axes
    fig.suptitle(
        f"{fname}  |  detector: {detector}  |  VUS(full)={vus_curve[-1]:.4f}",
        fontsize=10, fontweight='bold',
    )

    # 1. time series
    ax_ts.plot(ts, linewidth=0.6, color='dimgray', zorder=2)
    for s, e in zip(starts, ends):
        ax_ts.axvspan(s, e + 1, color='red', alpha=0.25, linewidth=0, zorder=3)
    ax_ts.set_ylabel('value', fontsize=8)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.tick_params(labelsize=7, labelbottom=False)

    # 2. score
    ax_sc.plot(score, linewidth=0.7, color='steelblue', zorder=2)
    for s, e in zip(starts, ends):
        ax_sc.axvspan(s, e + 1, color='red', alpha=0.15, linewidth=0, zorder=3)
    ax_sc.set_ylabel('score', fontsize=8)
    ax_sc.set_xlim(0, T - 1)
    ax_sc.tick_params(labelsize=7, labelbottom=False)
    ax_sc.set_xlabel('time', fontsize=8)

    # 3. original cumulative VUS per buffer
    ax_cum.plot(slope_values, vus_curve, color='crimson', linewidth=1.5)
    ax_cum.set_ylabel('cumulative\nVUS  C[b]', fontsize=8)
    ax_cum.set_xlim(0, slope_values[-1])
    ax_cum.tick_params(labelsize=7, labelbottom=False)

    # 4. decumulated (per-buffer) VUS
    ax_marg.plot(slope_values, marginal, color='seagreen', linewidth=1.5)
    ax_marg.set_ylabel('decumulated\nVUS  A[b]', fontsize=8)
    ax_marg.set_xlim(0, slope_values[-1])
    ax_marg.tick_params(labelsize=7, labelbottom=False)

    # 5. per-step incremental gain
    ax_inc.plot(slope_values, increment, color='darkorange', linewidth=1.2)
    ax_inc.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_inc.set_ylabel('per-step gain\nA[b]-A[b-1]', fontsize=8)
    ax_inc.set_xlabel('buffer', fontsize=8)
    ax_inc.set_xlim(0, slope_values[-1])
    ax_inc.tick_params(labelsize=7)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Decumulated VUS visual test")
    p.add_argument('--dataset', type=str, default=None,
                   help='Filter to one dataset, e.g. NAB, YAHOO, UCR (default: load all)')
    p.add_argument('--detector', type=str, default=None,
                   help='Filter to one detector (default: all detectors)')
    p.add_argument('--n-series', type=int, default=2,
                   help='Number of series to analyse (default 2)')
    p.add_argument('--min-anomalies', type=int, default=1,
                   help='Skip series with fewer anomalies than this (default 1)')
    p.add_argument('--slope-size', type=int, default=256,
                   help='Max buffer to sweep (default 256)')
    p.add_argument('--show', action='store_true',
                   help='Display interactively instead of saving to disk')
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..',
                                        'experiments', 'decumulated_vus'),
                   help='Output directory for plots')
    args = p.parse_args()

    from tsad_parser import TSADParser

    vus = VUSNumpy(slope_size=args.slope_size, step=1,
                   cumulative=True, global_mask=False, aggregation='mean')

    tsb = TSADParser()
    detectors = tsb.get_detectors()
    if args.detector:
        if args.detector not in detectors:
            print(f"Unknown detector '{args.detector}'. Available: {detectors}")
            return
        detectors = [args.detector]

    print("Loading time series...")
    load_kwargs = {'dataset': [args.dataset]} if args.dataset else {'load_all': True}
    fnames, timeseries, labels = tsb.load_timeseries(**load_kwargs)

    # pick the first n series that have enough anomalies
    chosen = []
    for i, lbl in enumerate(labels):
        starts, _ = _anomaly_bounds(np.asarray(lbl))
        if len(starts) >= args.min_anomalies:
            chosen.append(i)
        if len(chosen) >= args.n_series:
            break
    if not chosen:
        print("No series matched the filters.")
        return

    chosen_fnames = [fnames[i] for i in chosen]
    print(f"Loading scores for {len(detectors)} detector(s)...")
    scores, _ = tsb.load_scores_and_delete(
        chosen_fnames, [timeseries[i] for i in chosen],
        [labels[i] for i in chosen], detectors=detectors,
    )

    for j, i in enumerate(chosen):
        fname = fnames[i]
        ts = np.asarray(timeseries[i]).ravel()
        lbl = np.asarray(labels[i]).ravel()
        sc = scores[j]
        fname_safe = fname.replace('/', '_')
        print(f"\n{fname}  (T={len(ts)}, anomalies={len(_anomaly_bounds(lbl)[0])})")

        for det_idx, det in enumerate(detectors):
            score = np.asarray(sc[det_idx]).ravel()
            (vus_curve, _), _ = vus.compute(lbl, score)
            marginal = vus.decumulate(vus_curve)
            increment = np.diff(marginal, prepend=marginal[0])

            # sanity: re-cumulating the marginal must reproduce the curve
            idx = np.arange(1, len(marginal) + 1)
            assert np.allclose(np.cumsum(marginal) / idx, vus_curve), \
                "decumulate round-trip failed"

            print(f"  [{det}] VUS(full)={vus_curve[-1]:.4f}  "
                  f"A[0]={marginal[0]:.4f}  A[L]={marginal[-1]:.4f}")

            out_path = os.path.join(args.output_dir, fname_safe, f"{det}.png")
            plot_decumulated(fname, det, ts, score, lbl, vus.slope_values,
                             vus_curve, marginal, increment, out_path, args.show)

    if not args.show:
        print(f"\nAll plots saved to {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
