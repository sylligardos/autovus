"""Migration check: FastVUS (new) against VUSNumpy (old), step by step.

Runs on toy arrays only. Every ported step must give identical outputs.

    conda run -n autovus python src/experiments/check_fastvus.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np

from vus.fastvus import FastVUS
from vus.vus_numpy import VUSNumpy

L, STEP = 16, 1


def toy_cases():
    """Labels covering the edge cases, with a random score each."""
    rng = np.random.default_rng(0)
    cases = {}
    T = 400
    lab = np.zeros(T, int); lab[100:110] = 1; lab[250:252] = 1
    cases['two anomalies'] = lab
    lab = np.zeros(T, int); lab[:5] = 1
    cases['anomaly at start'] = lab
    lab = np.zeros(T, int); lab[-7:] = 1
    cases['anomaly at end'] = lab
    lab = np.zeros(T, int); lab[:3] = 1; lab[-3:] = 1; lab[200] = 1
    cases['start, end and single point'] = lab
    lab = np.zeros(T, int); lab[100:120] = 1; lab[121:130] = 1
    cases['adjacent anomalies (gap 1)'] = lab
    lab = np.zeros(T, int); lab[100:120] = 1; lab[120 + 2 * L:140 + 2 * L] = 1
    cases['anomalies 2L apart'] = lab
    lab = np.ones(T, int)
    cases['all anomalous'] = lab
    return {name: (lab, rng.random(len(lab))) for name, lab in cases.items()}


def same(a, b):
    """Exact equality for nested tuples/arrays."""
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return np.array_equal(np.asarray(a), np.asarray(b))


def check(step_name, fn_new, fn_old):
    """fn_old returns (result, seconds) because of the old @time_it decorator."""
    for name, (label, score) in toy_cases().items():
        out_new = fn_new(label, score)
        out_old, _ = fn_old(label, score)
        if not same(out_new, out_old):
            print(f"  MISMATCH [{step_name}] on '{name}'\n    new: {out_new}\n    old: {out_old}")
            return False
    print(f"  ok  {step_name}")
    return True


def main():
    new = FastVUS(L=L, step=STEP)
    old = VUSNumpy(slope_size=L, step=STEP)
    results = []

    results.append(check(
        'anomaly coordinates',
        lambda label, score: new.anomaly_coordinates(label),
        lambda label, score: old.get_anomalies_coordinates(label),
    ))

    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
