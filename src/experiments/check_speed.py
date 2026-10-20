"""Speed check: new engines against the old classes on one long random series.

In-memory data only (no benchmark files). The series has T points with
n_anomalies anomalies of ~20 points; the score is noise plus bumps.

    conda run -n autovus python src/experiments/check_speed.py [--T 2000000] [--n 200] [--L 128]
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch

from vus.fastvus import FastVUS
from vus.fastvus_gpu import FastVUSGPU
from vus.vus_numpy import VUSNumpy
from vus.vus_torch import VUSTorch


def make_series(T, n_anomalies, seed=0):
    rng = np.random.default_rng(seed)
    label = np.zeros(T, dtype=np.int8)
    starts = np.sort(rng.choice(T - 100, n_anomalies, replace=False))
    for s in starts:
        label[s:s + rng.integers(5, 40)] = 1
    score = rng.random(T) * 0.3
    for s in starts:
        score[s - rng.integers(0, 30):s + 30] += rng.random() * 0.7
    return label, np.clip(score, 0, 1)


def bench(name, fn, repeats):
    fn()                                   # warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); v = fn(); times.append(time.perf_counter() - t0)
    print(f"  {name:22s} {np.median(times) * 1000:9.1f} ms   value {float(v):.6f}")
    return np.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--T', type=int, default=2_000_000)
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--L', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    label, score = make_series(a.T, a.n)
    print(f"T={a.T:,}  anomalies={a.n}  L={a.L}")

    print('CPU:')
    old = VUSNumpy(slope_size=a.L)
    t_old = bench('old VUSNumpy', lambda: old.compute(label, score)[0][0], a.repeats)
    t_new = bench('new FastVUS', lambda: FastVUS(L=a.L).compute(label, score), a.repeats)
    print(f"  new / old = {t_new / t_old:.2f}")

    if torch.cuda.is_available():
        print('GPU:')
        oldg = VUSTorch(slope_size=a.L, device='cuda')
        newg = FastVUSGPU(L=a.L, device='cuda')
        lt = torch.tensor(label, device='cuda'); st = torch.tensor(score, device='cuda')

        def run_old():
            v = oldg.compute(lt, st)[0][0]; torch.cuda.synchronize(); return v

        def run_new():
            v = newg.compute(lt, st); torch.cuda.synchronize(); return v
        t_oldg = bench('old VUSTorch', run_old, a.repeats)
        t_newg = bench('new FastVUSGPU', run_new, a.repeats)
        print(f"  new / old = {t_newg / t_oldg:.2f}")


if __name__ == '__main__':
    main()
