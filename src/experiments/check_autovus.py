"""Checks for AutoVUS on toy arrays only.

Detections with a known timing must be classified as such, the FastVUS and
FastVUSGPU engines must select the same buffers, and the two scores must
agree. The selection rule itself (Mingyi's delta ratio) is not judged here.

    conda run -n autovus python src/experiments/check_autovus.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch

from autovus import FastVUS, FastVUSGPU, AutoVUS

L = 32


def detection_cases():
    """Series with anomalies whose detections have a known timing."""
    rng = np.random.default_rng(0)
    T = 1500
    cases = {}

    def series(spec):
        label = np.zeros(T, int)
        score = rng.random(T) * 0.05
        for start, length, lag, width in spec:
            label[start:start + length] = 1
            s = start + lag
            score[max(s, 0):max(s, 0) + width] += 0.8 + 0.2 * rng.random()
        return label, np.clip(score, 0, 1)

    cases['exact detections'] = series([(200, 20, 0, 20), (700, 30, 0, 30), (1200, 10, 0, 10)])
    cases['early detections'] = series([(200, 20, -12, 20), (700, 30, -20, 30), (1200, 10, -5, 10)])
    cases['late detections'] = series([(200, 20, 15, 20), (700, 30, 25, 30), (1200, 10, 8, 10)])
    cases['mixed, one missed'] = series([(200, 20, -10, 20), (700, 30, 0, 0), (1200, 10, 12, 10)])
    return cases


# None = not asserted: an anomaly with no bump sits in noise and the argmax
# rule still grows a buffer there. Known weakness, tracker D6.
EXPECTED = {'exact detections': ['exact'] * 3, 'early detections': ['early'] * 3,
            'late detections': ['late'] * 3, 'mixed, one missed': ['early', None, 'late']}


def ok(name, cond):
    print(f"  {'ok ' if cond else 'MISMATCH'}  {name}")
    return bool(cond)


def main():
    cpu = AutoVUS(FastVUS(L=L, cumulative=True))
    gpu = AutoVUS(FastVUSGPU(L=L, cumulative=True, device='cuda' if torch.cuda.is_available() else 'cpu'))
    results = []
    print('selection (b*_left / b*_right per anomaly):')
    agree = total = 0
    worst = 0.0
    for case, (label, score) in detection_cases().items():
        r_cpu, r_gpu = cpu.compute(label, score), gpu.compute(label, score)
        types = [a['detection'] for a in r_cpu['anomalies']]
        results.append(ok(f"{case}: detection types {types}", all(e is None or e == t for e, t in zip(EXPECTED[case], types))))
        for a, b in zip(r_cpu['anomalies'], r_gpu['anomalies']):
            ba = (a['left']['buffer_val'], a['right']['buffer_val'])
            bb = (b['left']['buffer_val'], b['right']['buffer_val'])
            agree += ba == bb; total += 1
            print(f"      anomaly {a['start']:5d}: cpu {ba}  gpu {bb}  {a['detection']}")
        worst = max(worst, abs(r_cpu['vus_adapt'] - r_gpu['vus_adapt']), abs(r_cpu['skill_adapt'] - r_gpu['skill_adapt']))
        print(f"      vus_full {r_cpu['vus_full']:.4f}  vus_adapt {r_cpu['vus_adapt']:.4f}  skill_adapt {r_cpu['skill_adapt']:.4f}")
    results.append(ok(f"b* agreement FastVUS vs FastVUSGPU engines: {agree}/{total}", agree == total))
    results.append(ok(f"scores agree between engines: max diff {worst:.1e}", worst < 1e-6))
    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
