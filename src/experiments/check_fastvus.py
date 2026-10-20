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


def plain(fn):
    """Call an old method and drop the timing that @time_it adds, when it does."""
    def call(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out[0] if hasattr(fn, '__wrapped__') else out
    return call


def check(step_name, fn_new, fn_old):
    for name, (label, score) in toy_cases().items():
        out_new = fn_new(label, score)
        out_old = fn_old(label, score)
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
        lambda label, score: plain(old.get_anomalies_coordinates)(label),
    ))

    def coords(label):
        return plain(old.get_anomalies_coordinates)(label)[1]   # (start, end) with edges

    for extra_safe in (False, True):
        results.append(check(
            f'safe mask (extra_safe={extra_safe})',
            lambda label, score: new.safe_mask(label, *coords(label), extra_safe=extra_safe),
            lambda label, score: plain(old.create_safe_mask)(label, *coords(label), extra_safe=extra_safe),
        ))
    results.append(check(
        'distance from anomaly',
        lambda label, score: new.distance_from_anomaly(label, *coords(label)),
        lambda label, score: plain(old.distance_from_anomaly)(label, *coords(label)),
    ))

    def dists(label):
        return plain(old.distance_from_anomaly)(label, *coords(label))

    for side in ('both', 'left', 'right'):
        new_s = FastVUS(L=L, step=STEP, side=side)
        old_s = VUSNumpy(slope_size=L, step=STEP, side=side)
        results.append(check(
            f'buffered labels (side={side})',
            lambda label, score: new_s.buffered_labels(label, *dists(label)),
            lambda label, score: plain(old_s.add_slopes_function)(label, *dists(label)),
        ))

    def caps(label):
        n = len(coords(label)[0])
        rng = np.random.default_rng(1)
        return rng.integers(0, L + 1, n), rng.integers(0, L + 1, n)

    results.append(check(
        'per-point caps',
        lambda label, score: new.per_point_caps(label, *coords(label), *caps(label)),
        lambda label, score: plain(old._per_point_cap)(label, *coords(label), *caps(label)),
    ))
    results.append(check(
        'buffered labels with caps',
        lambda label, score: new.buffered_labels(label, *dists(label), *new.per_point_caps(label, *coords(label), *caps(label))),
        lambda label, score: plain(old.add_slopes_function)(label, *dists(label), *old._per_point_cap(label, *coords(label), *caps(label))),
    ))

    def buffered(label):
        return new.buffered_labels(label, *dists(label))

    def thr(label, score):
        return new.thresholds(buffered(label), score)

    results.append(check(
        'thresholds',
        lambda label, score: thr(label, score),
        lambda label, score: plain(old.get_pr_relevant_thresholds_vus)(buffered(label), score, max_thresholds=old.max_thresholds),
    ))
    results.append(check(
        'score mask',
        lambda label, score: new.score_mask(score, thr(label, score)),
        lambda label, score: plain(old.get_score_mask)(score, thr(label, score)),
    ))
    results.append(check(
        'false positives outside the mask',
        lambda label, score: new.false_positives_outside(score[::3], thr(label, score)),
        lambda label, score: (score[::3] >= thr(label, score)[:, None]).sum(axis=1),
    ))
    results.append(check(
        'existence',
        lambda label, score: new.existence(buffered(label), score, thr(label, score), *coords(label)),
        lambda label, score: plain(old.existence_optimized)(buffered(label), score, thr(label, score), *coords(label)),
    ))
    for gm in (True, False):
        new_g = FastVUS(L=L, step=STEP, global_mask=gm)
        old_g = VUSNumpy(slope_size=L, step=STEP, global_mask=gm)
        results.append(check(
            f'confusion matrix (global_mask={gm})',
            lambda label, score: new_g.confusion_matrix(buffered(label), new.score_mask(score, thr(label, score))),
            lambda label, score: (lambda r: (r[2], r[0], r[3]))(plain(old_g.compute_confusion_matrix)(buffered(label), new.score_mask(score, thr(label, score)))),
        ))

    def pr_inputs(label, score):
        t = thr(label, score)
        sm = new.score_mask(score, t)
        tp, fp, pos = new.confusion_matrix(buffered(label), sm)
        ex = new.existence(buffered(label), score, t, *coords(label))
        return tp, fp, pos, ex

    results.append(check(
        'precision-recall',
        lambda label, score: new.precision_recall(*pr_inputs(label, score)),
        lambda label, score: plain(old.precision_recall_curve)(*pr_inputs(label, score)),
    ))
    for cum in (False, True):
        new_c = FastVUS(L=L, step=STEP, cumulative=cum)
        old_c = VUSNumpy(slope_size=L, step=STEP, cumulative=cum)
        results.append(check(
            f'auc (cumulative={cum})',
            lambda label, score: new_c.auc(*new.precision_recall(*pr_inputs(label, score))[::-1]),
            lambda label, score: plain(old_c.auc)(*new.precision_recall(*pr_inputs(label, score))[::-1]),
        ))

    # ---- the whole metric ----
    configs = {
        'default':            dict(L=L, step=STEP),
        'cumulative':         dict(L=L, step=STEP, cumulative=True),
        'no global mask':     dict(L=L, step=STEP, global_mask=False),
        'step 4':             dict(L=L, step=4),
        'side left':          dict(L=L, step=STEP, side='left'),
        'side right':         dict(L=L, step=STEP, side='right'),
        'L=0':                dict(L=0, step=0),
        'max_thresholds=20':  dict(L=L, step=STEP, max_thresholds=20),
    }
    for cfg_name, cfg in configs.items():
        old_cfg = dict(cfg); old_cfg['slope_size'] = old_cfg.pop('L')
        new_c, old_c = FastVUS(**cfg), VUSNumpy(**old_cfg)
        results.append(check(
            f'compute ({cfg_name})',
            lambda label, score: new_c.compute(label, score),
            lambda label, score: plain(old_c.compute)(label, score)[0],
        ))
    results.append(check(
        'compute with caps (cumulative)',
        lambda label, score: FastVUS(L=L, cumulative=True).compute(label, score, *caps(label)),
        lambda label, score: plain(VUSNumpy(slope_size=L, cumulative=True).compute)(
            label, score, max_buffer_per_anomaly_left=caps(label)[0], max_buffer_per_anomaly_right=caps(label)[1])[0],
    ))

    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
