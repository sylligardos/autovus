"""
Test script for the conf_matrix_new_idea confusion matrix.

Four scenarios with a single anomaly and a Gaussian score peak placed at
different positions relative to the anomaly and its buffer zone.
Run from the project root:
    conda run -n ffvus python src/test_new_idea.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from vus.vus_numpy import VUSNumpy

# ── Time series parameters ───────────────────────────────────────────────────
T             = 80
ANOMALY_START = 25
ANOMALY_END   = 34   # inclusive, anomaly length = 10
SLOPE_SIZE    = 10   # buffer extends 10 steps left and right of anomaly
# Buffer zone: left  [15, 24], right [35, 44]

# ── VUS instance ─────────────────────────────────────────────────────────────
vus = VUSNumpy(
    slope_size=SLOPE_SIZE,
    step=1,
    conf_matrix='new_idea',
    existence='None',     # skip existence so recall = tp / positives directly
    global_mask=False,    # keep all timesteps visible
)

# ── Helper: build a Gaussian score peak at a given position ─────────────────
def gaussian_score(peak: int, sigma: float = 2.0) -> np.ndarray:
    x = np.arange(T)
    return np.exp(-0.5 * ((x - peak) / sigma) ** 2)

# ── Ground-truth label ───────────────────────────────────────────────────────
label = np.zeros(T)
label[ANOMALY_START:ANOMALY_END + 1] = 1.0

# ── Scenarios ────────────────────────────────────────────────────────────────
scenarios = [
    (
        'Scenario 1 — Detection inside anomaly (position 29)',
        gaussian_score(29),
    ),
    (
        'Scenario 2 — Detection in right buffer, close (position 37)',
        gaussian_score(37),
    ),
    (
        'Scenario 3 — Detection in right buffer, far (position 43)',
        gaussian_score(43),
    ),
    (
        'Scenario 4 — Detection outside buffer, pure FP (position 58)',
        gaussian_score(58),
    ),
]

# ── Run ──────────────────────────────────────────────────────────────────────
for name, score in scenarios:
    sep = '=' * 64
    print(f'\n{sep}\n{name}\n{sep}')
    print(f'  Anomaly : [{ANOMALY_START}, {ANOMALY_END}]')
    print(f'  Buffer  : left [{ANOMALY_START - SLOPE_SIZE}, {ANOMALY_START - 1}]'
          f'  right [{ANOMALY_END + 1}, {ANOMALY_END + SLOPE_SIZE}]')
    print(f'  Score peak at position {int(score.argmax())}  (value {score.max():.3f})')

    (vus_pr, _), __ = vus.compute(label, score)
    print(f'  VUS-PR  : {vus_pr:.4f}')
