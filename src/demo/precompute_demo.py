#!/usr/bin/env python3
"""
Precompute all data needed for the FF-VUS adaptive buffer demo.

Steps
-----
  rankings  → experiments/precomputed/rankings_L{L}.npz
  existence → experiments/precomputed/existence_maps.npz

rankings output arrays
----------------------
  curves          float32  (n_files, n_det, L+1)  cumVUS per real (file, det, buffer)
  synth_curves    float32  (n_files, n_synth, L+1) cumVUS per synthetic detector
  denom           float32  (n_files, L+1)           analytical denominator per label
  b_star          int16    (n_files, n_det, 3)       best buffer: argmax / elbow / first_peak
  synth_b_star    int16    (n_files, n_synth, 3)     same for synthetic detectors
  mean_anom_len   float32  (n_files,)
  n_anomalies     int32    (n_files,)
  files           str      (n_files,)
  datasets        str      (n_files,)
  detectors       str      (n_det,)
  synth_names     str      (n_synth,)
  L               int      scalar

Usage
-----
    conda run -n autovus python src/demo/precompute_demo.py --L 1024
    conda run -n autovus python src/demo/precompute_demo.py --L 1024 --skip existence
    conda run -n autovus python src/demo/precompute_demo.py --L 1024 --skip rankings
    conda run -n autovus python src/demo/precompute_demo.py --L 1024 --force
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from tsad_parser import TSADParser
from vus.vus_numpy import VUSNumpy, select_best_buffer

try:
    from vus.vus_torch import VUSTorch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ['argmax', 'elbow', 'first_peak']

SYNTH_NAMES = [
    'perfect',        # score = label  (ideal detector)
    'const_0',        # always 0       (never alarms)
    'const_1',        # always 1       (always alarms)
    'rand_bernoulli', # Bernoulli(p)   (random, calibrated prevalence)
    'rand_uniform',   # Uniform(0,1)   (random, continuous)
    'C1',             # fires in left  buffer zone (early)
    'C2',             # fires in right buffer zone (late)
    'E1',             # isolated false positive far from anomaly
    'E3',             # first half of each anomaly only
    'E4',             # early over-prediction (extends before left boundary)
    'E5',             # late over-prediction  (extends after right boundary)
]
N_SYNTH = len(SYNTH_NAMES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(L):
    print('Loading time series and scores...')
    tsb = TSADParser()
    all_files, timeseries, all_labels = tsb.load_timeseries(load_all=True)
    detector_names = tsb.get_detectors()
    fname_to_dataset = dict(zip(tsb.meta['filename'], tsb.meta['dataset']))

    all_scores, _ = tsb.load_scores_and_delete(
        all_files, timeseries, all_labels, detectors=detector_names
    )
    all_datasets = [fname_to_dataset.get(f, 'unknown') for f in all_files]

    n_files = len(all_files)
    n_det   = len(detector_names)
    print(f'Loaded: {n_files} files | {n_det} detectors | L={L}')
    return all_files, all_labels, all_scores, detector_names, all_datasets


# ---------------------------------------------------------------------------
# Synthetic score helpers
# ---------------------------------------------------------------------------

def _get_anomaly_bounds(label):
    diff   = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0] - 1
    return starts, ends


def _make_synth_score(name, label, L, rng):
    T              = len(label)
    starts, ends   = _get_anomaly_bounds(label)
    has_anom       = len(starts) > 0
    avg_L          = int(round((ends - starts + 1).mean())) if has_anom else max(1, L // 4)

    if name == 'perfect':
        return label.astype(float)

    if name == 'const_0':
        return np.zeros(T)

    if name == 'const_1':
        return np.ones(T)

    if name == 'rand_bernoulli':
        p = label.mean() if label.mean() > 0 else 0.01
        return rng.binomial(1, p, size=T).astype(float)

    if name == 'rand_uniform':
        return rng.uniform(0.0, 1.0, size=T)

    if name == 'C1':
        score   = np.zeros(T)
        seg_len = max(1, avg_L // 2)
        for s in starts:
            seg_end   = s - 1
            seg_start = max(0, seg_end - seg_len + 1)
            if seg_end >= 0:
                score[seg_start:seg_end + 1] = 1.0
        return score

    if name == 'C2':
        score   = np.zeros(T)
        seg_len = max(1, avg_L // 2)
        for e in ends:
            seg_start = e + 1
            seg_end   = min(T - 1, seg_start + seg_len - 1)
            if seg_start < T:
                score[seg_start:seg_end + 1] = 1.0
        return score

    if name == 'E1':
        score = np.zeros(T)
        if not has_anom:
            score[T // 2] = 1.0
            return score
        gaps  = []
        if starts[0] > 0:
            gaps.append((0, starts[0] - 1))
        for k in range(len(ends) - 1):
            gaps.append((ends[k] + 1, starts[k + 1] - 1))
        if ends[-1] < T - 1:
            gaps.append((ends[-1] + 1, T - 1))
        valid      = [(gs, ge) for gs, ge in gaps if ge - gs >= 2 * L]
        gs, ge     = max(valid if valid else gaps, key=lambda g: g[1] - g[0])
        score[(gs + ge) // 2] = 1.0
        return score

    if name == 'E3':
        score = np.zeros(T)
        for s, e in zip(starts, ends):
            length = e - s + 1
            score[s:s + max(1, length // 2)] = 1.0
        return score

    if name == 'E4':
        score = np.zeros(T)
        over  = L + avg_L // 2
        for s, e in zip(starts, ends):
            score[max(0, s - over):e + 1] = 1.0
        return score

    if name == 'E5':
        score = np.zeros(T)
        over  = L + avg_L // 2
        for s, e in zip(starts, ends):
            score[s:min(T, e + over + 1)] = 1.0
        return score

    raise ValueError(f"Unknown synthetic score name: {name!r}")


# ---------------------------------------------------------------------------
# Label statistics + analytical denominator
# ---------------------------------------------------------------------------

def _label_stats(label):
    diff    = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts  = np.where(diff > 0)[0]
    ends    = np.where(diff < 0)[0]
    n_anom  = len(starts)
    if n_anom == 0:
        return 0, 0.0
    lengths = ends - starts
    return n_anom, float(lengths.mean())


def _analytical_denom(vus, label):
    """Closed-form E[cumVUS(b)] for a label-independent random score."""
    T = len(label)
    ((start_ne, end_ne), _), _ = vus.get_anomalies_coordinates(label)
    labels_arr = vus.add_slopes_precomputed(label, start_ne, end_ne)
    g_b        = labels_arr.mean(axis=1)          # == sum(axis=1) / T
    return np.cumsum(g_b) / np.arange(1, vus.n_slopes + 1)


# ---------------------------------------------------------------------------
# b* computation (post-processing, vectorised over methods)
# ---------------------------------------------------------------------------

def _compute_b_star(curve, denom, slope_values):
    """Return int16 array of shape (3,): b* for argmax / elbow / first_peak."""
    out   = np.zeros(3, dtype=np.int16)
    ratio = curve / np.clip(denom, 1e-10, None)
    for m, method in enumerate(METHODS):
        out[m] = select_best_buffer(ratio, slope_values, method=method)['buffer_val']
    return out


# ---------------------------------------------------------------------------
# Step: rankings (enhanced)
# ---------------------------------------------------------------------------

def step_rankings(all_files, all_labels, all_scores, detector_names, all_datasets,
                  L, backend, device, force):
    out_dir  = os.path.join(_ROOT_DIR, 'experiments', 'precomputed')
    out_path = os.path.join(out_dir, f'rankings_L{L}.npz')

    if not force and os.path.exists(out_path):
        print(f'\n[rankings] {out_path} already exists. Skipping. (--force to recompute)')
        return

    n_files = len(all_files)
    n_det   = len(detector_names)
    print(f'\n[rankings] {n_files} files × {n_det} real + {N_SYNTH} synthetic detectors | L={L}')

    # Allocate output arrays (NaN = missing / failed)
    curves       = np.full((n_files, n_det,   L + 1), np.nan, dtype=np.float32)
    synth_curves = np.full((n_files, N_SYNTH, L + 1), np.nan, dtype=np.float32)
    denom        = np.full((n_files,          L + 1), np.nan, dtype=np.float32)
    b_star       = np.full((n_files, n_det,   3),     -1,     dtype=np.int16)
    synth_b_star = np.full((n_files, N_SYNTH, 3),     -1,     dtype=np.int16)
    mean_anom_len = np.zeros(n_files, dtype=np.float32)
    n_anomalies   = np.zeros(n_files, dtype=np.int32)

    if backend == 'torch':
        vus = VUSTorch(slope_size=L, step=1, cumulative=True, device=device)
    else:
        vus = VUSNumpy(slope_size=L, step=1, cumulative=True, global_mask=False)

    vus._skip_adaptive_plot = True
    rng = np.random.default_rng(42)

    # ---- Main loop: one file at a time ----
    for i in tqdm(range(n_files), desc='files', unit='file'):
        label = all_labels[i]

        # Label statistics + analytical denominator
        n_anom, m_len           = _label_stats(label)
        n_anomalies[i]          = n_anom
        mean_anom_len[i]        = m_len

        if n_anom > 0:
            try:
                denom[i, :] = _analytical_denom(vus, label)
            except Exception:
                pass

        # Synthetic detectors
        for k, name in enumerate(SYNTH_NAMES):
            try:
                score = _make_synth_score(name, label, L, rng)
                (cum, _), _ = vus.compute(label, score)
                if backend == 'torch':
                    cum = cum.cpu().numpy()
                synth_curves[i, k, :] = cum
            except Exception:
                pass

        # Real detectors
        for j in range(n_det):
            score = all_scores[i][j, :]
            if np.isnan(score).all():
                continue
            try:
                (cum, _), _ = vus.compute(label, score)
                if backend == 'torch':
                    cum = cum.cpu().numpy()
                curves[i, j, :] = cum
            except Exception:
                pass

    # ---- b* post-processing ----
    print('Computing b* for all (file, detector) pairs...')
    slope_values = vus.slope_values if hasattr(vus, 'slope_values') else np.arange(L + 1)

    for i in tqdm(range(n_files), desc='b*', unit='file'):
        d = denom[i, :]
        if np.isnan(d[0]):
            continue
        for j in range(n_det):
            if not np.isnan(curves[i, j, 0]):
                b_star[i, j, :] = _compute_b_star(curves[i, j, :], d, slope_values)
        for k in range(N_SYNTH):
            if not np.isnan(synth_curves[i, k, 0]):
                synth_b_star[i, k, :] = _compute_b_star(synth_curves[i, k, :], d, slope_values)

    # ---- Save ----
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        curves        = curves,
        synth_curves  = synth_curves,
        denom         = denom,
        b_star        = b_star,
        synth_b_star  = synth_b_star,
        mean_anom_len = mean_anom_len,
        n_anomalies   = n_anomalies,
        files         = np.array(all_files),
        datasets      = np.array(all_datasets),
        detectors     = np.array(detector_names),
        synth_names   = np.array(SYNTH_NAMES),
        L             = np.array(L),
    )

    nan_real  = int(np.isnan(curves[:, :, 0]).sum())
    nan_synth = int(np.isnan(synth_curves[:, :, 0]).sum())
    print(f'\n[rankings] Saved → {out_path}')
    print(f'  Missing real detector pairs  : {nan_real} / {n_files * n_det}')
    print(f'  Missing synthetic pairs      : {nan_synth} / {n_files * N_SYNTH}')


# ---------------------------------------------------------------------------
# Step: existence  (unchanged)
# ---------------------------------------------------------------------------

def step_existence(all_files, all_labels, all_scores, detector_names, all_datasets, L, force):
    out_dir  = os.path.join(_ROOT_DIR, 'experiments', 'precomputed')
    out_path = os.path.join(out_dir, 'existence_maps.npz')

    if not force and os.path.exists(out_path):
        print(f'\n[existence] Output exists. Skipping. (--force to recompute)')
        return

    n_files = len(all_files)
    n_det   = len(detector_names)
    print(f'\n[existence] {n_files} files × {n_det} detectors | L={L}')

    vus = VUSNumpy(slope_size=L, step=1)

    anom_curves     = np.empty((n_files, n_det), dtype=object)
    mean_curves     = np.full((n_files, n_det, L + 1), np.nan, dtype=np.float32)
    anom_starts_all = np.empty(n_files, dtype=object)
    anom_ends_all   = np.empty(n_files, dtype=object)
    n_anomalies_all = np.zeros(n_files, dtype=np.int32)

    with tqdm(total=n_files * n_det, unit='pair') as pbar:
        for i in range(n_files):
            label = all_labels[i]
            try:
                ((_, _), (starts, ends)), _ = vus.get_anomalies_coordinates(label)
            except Exception:
                pbar.update(n_det)
                continue

            n_anom              = len(starts)
            anom_starts_all[i]  = starts.astype(np.int32)
            anom_ends_all[i]    = ends.astype(np.int32)
            n_anomalies_all[i]  = n_anom

            for j in range(n_det):
                score = all_scores[i][j, :]
                if n_anom == 0 or np.isnan(score).all():
                    anom_curves[i, j] = np.full((n_anom, L + 1), np.nan, dtype=np.float32)
                    pbar.update(1)
                    continue
                try:
                    _, _, c_curves = vus.compute_per_anomaly_existence(label, score)
                    anom_curves[i, j]    = c_curves
                    mean_curves[i, j, :] = c_curves.mean(axis=0)
                except Exception:
                    anom_curves[i, j] = np.full((n_anom, L + 1), np.nan, dtype=np.float32)
                pbar.update(1)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        files        = np.array(all_files),
        datasets     = np.array(all_datasets),
        detectors    = np.array(detector_names),
        L            = np.array(L),
        anom_curves  = anom_curves,
        mean_curves  = mean_curves,
        anom_starts  = anom_starts_all,
        anom_ends    = anom_ends_all,
        n_anomalies  = n_anomalies_all,
    )
    nan_pairs       = int(np.isnan(mean_curves[:, :, 0]).sum())
    total_anomalies = int(n_anomalies_all.sum())
    print(f'\n[existence] Saved → {out_path}')
    print(f'  Missing pairs   : {nan_pairs} / {n_files * n_det}')
    print(f'  Total anomalies : {total_anomalies}')



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Precompute all data for the FF-VUS adaptive buffer demo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--L',       type=int, default=1024,
                        help='Max buffer size (default: 1024)')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy',
                        help='Compute backend for rankings (default: numpy)')
    parser.add_argument('--skip',    nargs='+', default=[],
                        choices=['rankings', 'existence'],
                        help='Steps to skip')
    parser.add_argument('--force',   action='store_true',
                        help='Re-run even if output already exists')
    args = parser.parse_args()

    L       = args.L
    backend = args.backend
    skip    = set(args.skip)

    if backend == 'torch':
        if not _TORCH_AVAILABLE:
            print('Error: --backend torch requested but VUSTorch could not be imported.')
            sys.exit(1)
        import torch
        if not torch.cuda.is_available():
            print('Error: --backend torch requires a CUDA GPU. Use --backend numpy instead.')
            sys.exit(1)
        device   = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f'GPU: {gpu_name}')
    else:
        device = None

    all_files, all_labels, all_scores, detector_names, all_datasets = load_data(L)

    if 'rankings' not in skip:
        step_rankings(all_files, all_labels, all_scores, detector_names, all_datasets,
                      L, backend, device, args.force)
    else:
        print('\n[rankings] Skipped.')

    if 'existence' not in skip:
        step_existence(all_files, all_labels, all_scores, detector_names, all_datasets,
                       L, args.force)
    else:
        print('\n[existence] Skipped.')

    print('\nDone.')


if __name__ == '__main__':
    main()
