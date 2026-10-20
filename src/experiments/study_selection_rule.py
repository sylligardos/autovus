"""G6: how the buffer selection rules behave on detections with a known lag (toy).

One anomaly (length 20) in a window of 400 points; the detector fires a bump
of the same length shifted by `lag` points (negative = early), plus noise.
Rules on the per-side cumulative curves (side = the lagged side):
  delta    (current, Mingyi): argmax (vus[b]-vus[0]+1) / (random[b]-random[0]+1)
  marginal (June design):     argmax A[b] / g[b] on the de-cumulated curves
Ideal b* = |lag| (the buffer that just reaches the bump).

    conda run -n autovus python src/experiments/study_selection_rule.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np

from autovus import FastVUS, AutoVUS

L = 64


def window(lag, noise, width=20, seed=0, bump=True):
    rng = np.random.default_rng(seed)
    label = np.zeros(400, int); label[190:210] = 1
    score = rng.random(400) * noise
    if bump:
        s = 190 + lag
        score[s:s + width] += 1.0
    return label, np.clip(score, 0, 1)


def rules(label, score, side):
    eng = FastVUS(L=L, cumulative=True, side=side)
    auto = AutoVUS(eng)
    vus, random = auto.curves(label, score)
    delta = auto.select_buffer(vus, random)['buffer_val']
    A, g = AutoVUS.decumulate(vus), np.clip(AutoVUS.decumulate(random), 1e-10, None)
    ratio = A / g
    marginal = int(eng.buffers[np.argmax(ratio)]) if np.any(ratio - 1 > 1e-4) else 0
    return delta, marginal


def main():
    print(f"L={L}. Cells: b* delta / b* marginal (ideal = |lag|)")
    print(f"{'noise':>6s} " + ' '.join(f"{'lag '+str(l):>9s}" for l in (-40, -20, -10, -5, 0, 5, 10, 20, 40)) + '   no bump (left / right)')
    for noise in (0.02, 0.1, 0.3, 0.6):
        cells = []
        for lag in (-40, -20, -10, -5, 0, 5, 10, 20, 40):
            label, score = window(lag, noise)
            side = 'left' if lag < 0 else 'right'
            d, m = rules(label, score, side)
            cells.append(f"{d:3d} /{m:3d}")
        label, score = window(0, noise, bump=False)
        dl, ml = rules(label, score, 'left'); dr, mr = rules(label, score, 'right')
        print(f"{noise:6.2f} " + ' '.join(f"{c:>9s}" for c in cells) + f"   {dl}/{ml} , {dr}/{mr}")


if __name__ == '__main__':
    main()
