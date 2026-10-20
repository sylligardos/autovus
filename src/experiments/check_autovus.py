"""Migration check for AutoVUS on toy arrays only.

The selection rule, the helpers and the scoring are checked exactly against
the old VUSNumpy code on identical inputs. The end-to-end selection is then
compared old vs new (different engines: curved buffers, so only agreement is
reported) and FastVUS vs FastVUSGPU engines (should agree).

    conda run -n autovus python src/experiments/check_autovus.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import torch

from vus.fastvus import FastVUS
from vus.fastvus_gpu import FastVUSGPU
from vus.autovus import AutoVUS
from vus.vus_numpy import VUSNumpy

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


def ok(name, cond):
    print(f"  {'ok ' if cond else 'MISMATCH'}  {name}")
    return bool(cond)


def check_rule():
    """select_buffer == old _select_adaptive_buffer on identical curves."""
    new = AutoVUS(FastVUS(L=L, cumulative=True))
    old = VUSNumpy(slope_size=L, cumulative=True)
    rng = np.random.default_rng(3)
    r = []
    for trial in range(50):
        label = np.zeros(600, int); label[250:270] = 1
        buffered = new.engine.buffered_labels(label, *new.engine.distance_from_anomaly(label, *new._coordinates(label)))
        vus = np.sort(rng.random(L + 1)) * rng.random()            # some increasing curve
        if trial % 5 == 0:
            vus = np.full(L + 1, vus[0])                            # flat: nothing beats random
        T = 600
        random = np.cumsum(buffered.sum(axis=1) / T) / np.arange(1, L + 2)
        a = new.select_buffer(vus, random)
        (b, _) = old._select_adaptive_buffer(buffered, vus, T)
        same = (a['buffer_idx'] == b['best']['buffer_idx'] and a['buffer_val'] == b['best']['buffer_val']
                and a['ratio'] == b['best']['ratio'] and a['vus'] == b['best']['vus_score']
                and a['core_ratio'] == b['best']['core_ratio'] and np.array_equal(a['ratio_curve'], b['ratio'])
                and np.array_equal(random, b['denom']))
        r.append(same)
    return [ok('select_buffer == old _select_adaptive_buffer (50 random curves)', all(r))]


def check_helpers():
    old = VUSNumpy(slope_size=L, cumulative=True)
    rng = np.random.default_rng(4)
    r = []
    curves = [rng.random(L + 1) for _ in range(10)]
    r.append(ok('decumulate == old', all(np.array_equal(AutoVUS.decumulate(c), old.decumulate(c)) for c in curves)))
    scores = [rng.random(300) * np.where(rng.random(300) < 0.3, 1e-9, 1.0) for _ in range(10)]
    r.append(ok('floor_dust == old', all(np.array_equal(AutoVUS.floor_dust(s, 1e-6), old.floor_dust(s, 1e-6)) for s in scores)))
    same = True
    for bl in range(0, 4):
        for br in range(0, 4):
            for core in (0.5, 1.5):
                new_a = {'left': {'buffer_val': bl, 'core_ratio': core}, 'right': {'buffer_val': br}}
                old_a = {'best_left': {'buffer_val': bl}, 'best_right': {'buffer_val': br}, 'ratio_left': np.array([core])}
                same &= AutoVUS.classify_detection(new_a) == old.classify_detection(old_a)
    r.append(ok('classify_detection == old', same))
    return r


def check_selection():
    """End-to-end selection: expected timing, old vs new agreement, CPU vs GPU engines."""
    new = AutoVUS(FastVUS(L=L, cumulative=True))
    old = VUSNumpy(slope_size=L, cumulative=True)
    gpu = AutoVUS(FastVUSGPU(L=L, cumulative=True, device='cuda' if torch.cuda.is_available() else 'cpu'))
    r = []
    # None = not asserted: an anomaly with no bump sits in noise, and the argmax
    # rule still grows a buffer there (old and new alike). Known weakness, D6.
    expected = {'exact detections': ['exact'] * 3, 'early detections': ['early'] * 3,
                'late detections': ['late'] * 3, 'mixed, one missed': ['early', None, 'late']}
    print('  end-to-end selection (b*_left / b*_right per anomaly):')
    agree_old = agree_gpu = total = 0
    for case, (label, score) in detection_cases().items():
        a_new = new.select(label, score)
        a_old = old.compute_per_anomaly_buffer(label, score)
        a_gpu = gpu.select(label, score)
        types = [a['detection'] for a in a_new]
        r.append(ok(f"{case}: detection types {types}", all(e is None or e == t for e, t in zip(expected[case], types))))
        for n, o, g in zip(a_new, a_old, a_gpu):
            nb = (n['left']['buffer_val'], n['right']['buffer_val'])
            ob = (o['best_left']['buffer_val'], o['best_right']['buffer_val'])
            gb = (g['left']['buffer_val'], g['right']['buffer_val'])
            agree_old += nb == ob; agree_gpu += nb == gb; total += 1
            print(f"      anomaly {n['start']:5d}: new {nb}  old {ob}  gpu {gb}  {n['detection']}")
    print(f"  info  b* agreement new vs old: {agree_old}/{total} (different engines: curve, existence side handling)")
    r.append(ok(f"b* agreement FastVUS vs FastVUSGPU engines: {agree_gpu}/{total}", agree_gpu == total))
    return r


def check_scoring():
    """score() against old score_adapt on the same caps; old engine differs, so report."""
    new = AutoVUS(FastVUS(L=L, cumulative=True))
    old = VUSNumpy(slope_size=L, cumulative=True)
    print('  scoring vs old score_adapt (same caps, different engine):')
    worst = 0.0
    for case, (label, score) in detection_cases().items():
        anomalies = new.select(label, score)
        cl = np.array([a['left']['buffer_val'] for a in anomalies]); cr = np.array([a['right']['buffer_val'] for a in anomalies])
        s_new = new.score(label, new.floor_dust(score, 1e-6), cl, cr)
        (s_old, _) = old.score_adapt(label, old.floor_dust(score, 1e-6), b_star_left=cl, b_star_right=cr) if hasattr(old.score_adapt, '__wrapped__') else (old.score_adapt(label, old.floor_dust(score, 1e-6), b_star_left=cl, b_star_right=cr), 0)
        d = max(abs(s_new['vus_adapt'] - s_old['vus_adapt']), abs(s_new['vus_full'] - s_old['vus_full']),
                abs(s_new['skill_adapt'] - s_old['skill_adapt_A']), abs(s_new['skill_full'] - s_old['skill_full_A']),
                abs(s_new['skill_per_buffer_adapt'] - s_old['skill_adapt_B']))
        worst = max(worst, d)
        print(f"      {case:20s} vus_adapt {s_new['vus_adapt']:.4f} (old {s_old['vus_adapt']:.4f})  skill_adapt {s_new['skill_adapt']:.4f} (old {s_old['skill_adapt_A']:.4f})  max diff {d:.1e}")
    return [ok(f'scoring within 1e-2 of old (curve difference): max diff {worst:.1e}', worst < 1e-2)]


def main():
    results = []
    print('selection rule:'); results += check_rule()
    print('helpers:'); results += check_helpers()
    print('selection:'); results += check_selection()
    print('scoring:'); results += check_scoring()
    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
