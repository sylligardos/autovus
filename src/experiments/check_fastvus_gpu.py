"""Checks for FastVUSGPU on toy arrays only.

Reference: FastVUS, the exact CPU engine. Runs on every available device
(cpu, cuda). The segment-max existence is the only approximation: it merges
two anomalies when their buffered regions touch (distance 2L + 1), where the
exact rule does not.

    conda run -n autovus python src/experiments/check_fastvus_gpu.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch

from autovus import FastVUS, FastVUSGPU
from check_fastvus import toy_cases, L, STEP

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def report(name, pairs, atol):
    """pairs: {case: (a, b)}; prints the max difference and whether it is within atol."""
    worst = max(np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in pairs.values())
    ok = worst <= atol
    print(f"  {'ok ' if ok else 'MISMATCH'}  {name}: max diff {worst:.1e} (tolerance {atol:.0e})")
    if not ok:
        for case, (a, b) in pairs.items():
            d = np.abs(np.asarray(a) - np.asarray(b)).max()
            if d > atol:
                print(f"      {case}: {d:.2e}")
    return ok


def no_overlap(label):
    """True when no two buffered regions touch at any buffer (segment-max existence is then exact)."""
    s, e = FastVUS(L=L).anomaly_coordinates(label)[1]
    return len(s) < 2 or np.all(s[1:] - e[:-1] > 2 * L + 1)


def check_device(device):
    print(f"device {device}:")
    r = []
    cpu = FastVUS(L=L, step=STEP, max_thresholds=None)
    gpu = FastVUSGPU(L=L, step=STEP, max_thresholds=None, device=device)
    cases = toy_cases()

    # Final value: exact where no buffers merge (float32 buffered labels), close elsewhere.
    plain_cases = {c: (cpu.compute(l, s), gpu.compute(l, s)) for c, (l, s) in cases.items() if no_overlap(l)}
    r.append(report('compute vs FastVUS, non-overlapping cases', plain_cases, 1e-6))
    merge_cases = {c: (cpu.compute(l, s), gpu.compute(l, s)) for c, (l, s) in cases.items() if not no_overlap(l)}
    r.append(report('compute vs FastVUS, overlapping cases (existence merge approximation)', merge_cases, 1e-2))

    # Configurations.
    for name, kw in {'cumulative': dict(cumulative=True), 'no global mask': dict(global_mask=False),
                     'step 4': dict(step=4), 'capped thresholds': dict(max_thresholds=20)}.items():
        a = FastVUS(L=L, step=STEP, max_thresholds=None); b = FastVUSGPU(L=L, step=STEP, max_thresholds=None, device=device)
        for k, v in kw.items():
            setattr(a, k, v); setattr(b, k, v)
        if 'step' in kw:
            a = FastVUS(L=L, step=4, max_thresholds=None); b = FastVUSGPU(L=L, step=4, max_thresholds=None, device=device)
        pairs = {c: (a.compute(l, s), b.compute(l, s)) for c, (l, s) in cases.items() if no_overlap(l)}
        r.append(report(f'compute ({name}) vs FastVUS, non-overlapping cases', pairs, 1e-6))

    # Per-anomaly caps.
    rng = np.random.default_rng(1)
    pairs = {}
    for c, (l, s) in cases.items():
        if not no_overlap(l):
            continue
        n = len(cpu.anomaly_coordinates(l)[1][0])
        cl, cr = rng.integers(0, L + 1, n), rng.integers(0, L + 1, n)
        pairs[c] = (cpu.compute(l, s, cl, cr), gpu.compute(l, s, cl, cr))
    r.append(report('compute with caps vs FastVUS, non-overlapping cases', pairs, 1e-6))
    for side in ('left', 'right'):
        a = FastVUS(L=L, step=STEP, max_thresholds=None, side=side)
        b = FastVUSGPU(L=L, step=STEP, max_thresholds=None, device=device, side=side)
        r.append(report(f'compute (side {side}) vs FastVUS, non-overlapping cases',
                        {c: (a.compute(l, s), b.compute(l, s)) for c, (l, s) in cases.items() if no_overlap(l)}, 1e-6))

    # Chunking: a tiny memory budget forces many chunks; the result must not move.
    whole = {c: gpu.compute(l, s) for c, (l, s) in cases.items()}
    gpu._update_memory_budget = lambda: setattr(gpu, 'memory_budget', 4000)
    chunked = {c: gpu.compute(l, s) for c, (l, s) in cases.items()}
    r.append(report('chunked (budget 4 kB) vs unchunked (float32 summation order)', {c: (whole[c], chunked[c]) for c in cases}, 1e-6))
    gpu._update_memory_budget = FastVUSGPU._update_memory_budget.__get__(gpu)

    return r


def main():
    results = []
    for device in DEVICES:
        results += check_device(device)
    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
