"""
Verify that the cumulative FFVUS computation is correct (numpy + torch).

Strategy:
- Compute standard FFVUS individually for slope_sizes 1, 2, 4, 8, ..., 512.
- Compute cumulative FFVUS once with slope_size=512.
- Compare: cumulative[k] should equal the individual run with slope_size=k.
- Run the same check for both VUSNumpy and VUSTorch on real TSB data.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch
from utils.utils import load_tsb


def verify_numpy(label, score, slope_sizes, max_slope):
    print(f"\n{'='*70}")
    print("VUSNumpy cumulative verification")
    print(f"{'='*70}")

    # Individual
    individual = {}
    for ss in slope_sizes:
        vus = VUSNumpy(slope_size=ss, step=1, cumulative=False)
        (value, _), _ = vus.compute(label, score)
        individual[ss] = value

    # Cumulative
    vus_cum = VUSNumpy(slope_size=max_slope, step=1, cumulative=True)
    (cum_values, _), _ = vus_cum.compute(label, score)

    return compare(individual, cum_values, slope_sizes, "VUSNumpy")


def verify_torch(label, score, slope_sizes, max_slope, device):
    print(f"\n{'='*70}")
    print(f"VUSTorch cumulative verification (device={device})")
    print(f"{'='*70}")

    label_t = torch.tensor(label, device=device, dtype=torch.uint8)
    score_t = torch.tensor(score, device=device, dtype=torch.float32)

    # Individual
    individual = {}
    for ss in slope_sizes:
        vus = VUSTorch(slope_size=ss, step=1, cumulative=False, device=device)
        with torch.no_grad():
            (value, _), _ = vus.compute(label_t, score_t)
        individual[ss] = float(value)

    # Cumulative
    vus_cum = VUSTorch(slope_size=max_slope, step=1, cumulative=True, device=device)
    with torch.no_grad():
        (cum_values, _), _ = vus_cum.compute(label_t, score_t)
    cum_values = cum_values.cpu().numpy()

    return compare(individual, cum_values, slope_sizes, f"VUSTorch ({device})")


def compare(individual, cum_values, slope_sizes, name):
    print(f"  Cumulative shape: {cum_values.shape}")
    print(f"  {'slope_size':>10} {'cumulative[k]':>18} {'individual':>18} {'diff':>12} {'status':>8}")
    print(f"  {'-'*66}")

    all_pass = True
    for ss in slope_sizes:
        cum_val = cum_values[ss]
        ind_val = individual[ss]
        diff = abs(cum_val - ind_val)
        match = np.isclose(cum_val, ind_val, atol=1e-6)
        all_pass = all_pass and match
        print(f"  {ss:>10} {cum_val:>18.12f} {ind_val:>18.12f} {diff:>12.2e} {'PASS' if match else 'FAIL':>8}")

    print(f"  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == '__main__':
    max_slope = 512
    slope_sizes = [2**i for i in range(10)]

    # Load real data
    print("Loading TSB data...")
    filenames, labels, scores, _ = load_tsb(testing=False, dataset='KDD21', n_timeseries=10)

    all_pass = True
    for i in range(len(filenames)):
        label = labels[i]
        score = scores[i]
        print(f"\n{'#'*70}")
        print(f"Time series: {filenames[i]}")
        print(f"Length: {len(label)}, anomaly_ratio: {label.mean():.4f}")
        print(f"{'#'*70}")

        all_pass &= verify_numpy(label, score, slope_sizes, max_slope)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        all_pass &= verify_torch(label, score, slope_sizes, max_slope, device)

    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")
