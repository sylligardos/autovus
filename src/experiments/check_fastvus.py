"""Migration check for FastVUS (new) on toy arrays only.

Two references:
  * VUSNumpy (old): every step that did not change must give identical output.
  * upstream TSB-AD (src/baselines/tsb_ad): the buffer curve and the final
    VUS-PR. Upstream averages windows W = 0..2L and merges overlapping buffers
    by adding and clipping, so the final value is only close; per window, the
    even windows must match exactly on cases without overlapping buffers.

    conda run -n autovus python src/experiments/check_fastvus.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np

from vus.fastvus import FastVUS
from vus.vus_numpy import VUSNumpy
from baselines.tsb_ad.basic_metrics import basic_metricor

L, STEP = 16, 1
new = FastVUS(L=L, step=STEP)
old = VUSNumpy(slope_size=L, step=STEP)
upstream = basic_metricor()


# ---------------------------------------------------------------- helpers
def toy_cases():
    """Labels covering the edge cases, with a random score each."""
    rng = np.random.default_rng(0)
    T = 400
    cases = {}
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


def plain(fn):
    """Call an old method and drop the timing that its @time_it adds, when it does."""
    def call(*args, **kwargs):
        out = fn(*args, **kwargs)
        return out[0] if hasattr(fn, '__wrapped__') else out
    return call


def same(a, b, tol=0.0):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if tol:
        return np.allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=0)
    return np.array_equal(np.asarray(a), np.asarray(b))


def check(name, fn_a, fn_b, tol=0.0):
    """fn_a and fn_b take (label, score); their outputs must be identical on every
    case (or within relative tolerance tol, for results that differ only in
    floating-point summation order)."""
    for case, (label, score) in toy_cases().items():
        a, b = fn_a(label, score), fn_b(label, score)
        if not same(a, b, tol):
            print(f"  MISMATCH [{name}] on '{case}'\n    a: {a}\n    b: {b}")
            return False
    print(f"  ok  {name}")
    return True


# Inputs for the step checks, built with the already verified new methods.
def coords(label):
    return new.anomaly_coordinates(label)[1]            # (start, end)


def dists(label):
    return new.distance_from_anomaly(label, *coords(label))


def buffered(label):
    return new.buffered_labels(label, *dists(label))


def thr(label, score):
    return new.thresholds(buffered(label), score)


def caps(label):
    """Random per-anomaly caps, the same every call."""
    rng = np.random.default_rng(1)
    n = len(coords(label)[0])
    return rng.integers(0, L + 1, n), rng.integers(0, L + 1, n)


def no_overlap(label):
    s, e = coords(label)
    return len(s) < 2 or np.all(s[1:] - e[:-1] > 2 * L)


# ---------------------------------------------------------------- checks
def check_steps_vs_old():
    r = []
    r.append(check('anomaly coordinates',
                   lambda l, s: new.anomaly_coordinates(l),
                   lambda l, s: plain(old.get_anomalies_coordinates)(l)))
    for extra in (False, True):
        r.append(check(f'safe mask (extra_safe={extra})',
                       lambda l, s: new.safe_mask(l, *coords(l), extra_safe=extra),
                       lambda l, s: plain(old.create_safe_mask)(l, *coords(l), extra_safe=extra)))
    r.append(check('distance from anomaly',
                   lambda l, s: dists(l),
                   lambda l, s: plain(old.distance_from_anomaly)(l, *coords(l))))
    r.append(check('per-point caps',
                   lambda l, s: new.per_point_caps(l, *coords(l), *caps(l)),
                   lambda l, s: plain(old._per_point_cap)(l, *coords(l), *caps(l))))
    r.append(check('thresholds',
                   lambda l, s: thr(l, s),
                   lambda l, s: plain(old.get_pr_relevant_thresholds_vus)(buffered(l), s, max_thresholds=old.max_thresholds)))
    r.append(check('score mask',
                   lambda l, s: new.score_mask(s, thr(l, s)),
                   lambda l, s: plain(old.get_score_mask)(s, thr(l, s))))
    r.append(check('false positives outside the mask',
                   lambda l, s: new.false_positives_outside(s[::3], thr(l, s)),
                   lambda l, s: (s[::3] >= thr(l, s)[:, None]).sum(axis=1)))
    r.append(check('existence',
                   lambda l, s: new.existence(buffered(l), s, thr(l, s), *coords(l)),
                   lambda l, s: plain(old.existence_optimized)(buffered(l), s, thr(l, s), *coords(l))))
    for gm in (True, False):
        new_g, old_g = FastVUS(L=L, step=STEP, global_mask=gm), VUSNumpy(slope_size=L, step=STEP, global_mask=gm)
        r.append(check(f'confusion matrix (global_mask={gm})',
                       lambda l, s: new_g.confusion_matrix(buffered(l), new.score_mask(s, thr(l, s))),
                       lambda l, s: (lambda o: (o[2], o[0], o[3]))(
                           plain(old_g.compute_confusion_matrix)(buffered(l), new.score_mask(s, thr(l, s))))))

    def pr_inputs(l, s):
        t = thr(l, s)
        tp, fp, pos = new.confusion_matrix(buffered(l), new.score_mask(s, t))
        return tp, fp, pos, new.existence(buffered(l), s, t, *coords(l))

    r.append(check('precision-recall',
                   lambda l, s: new.precision_recall(*pr_inputs(l, s)),
                   lambda l, s: plain(old.precision_recall_curve)(*pr_inputs(l, s))))
    for cum in (False, True):
        new_c, old_c = FastVUS(L=L, step=STEP, cumulative=cum), VUSNumpy(slope_size=L, step=STEP, cumulative=cum)
        r.append(check(f'auc (cumulative={cum})',
                       lambda l, s: new_c.auc(*new.precision_recall(*pr_inputs(l, s))[::-1]),
                       lambda l, s: plain(old_c.auc)(*new.precision_recall(*pr_inputs(l, s))[::-1])))
    return r


def check_buffers_vs_upstream():
    """Buffer h of ours == upstream window 2h, where no two ramps meet."""
    def upstream_rows(label):
        seq = list(zip(*coords(label)))
        return np.array([upstream.sequencing(label, seq, 2 * h) for h in new.buffers])

    for case, (label, score) in toy_cases().items():
        if not no_overlap(label):
            continue
        diff = np.abs(buffered(label) - upstream_rows(label)).max()
        if diff > 1e-12:
            print(f"  MISMATCH [buffered labels vs upstream] on '{case}': max diff {diff}")
            return [False]
    print('  ok  buffered labels vs upstream curve (non-overlapping cases)')

    r = [True]
    left, right = FastVUS(L=L, step=STEP, side='left'), FastVUS(L=L, step=STEP, side='right')
    r.append(check('buffered labels: both == max(left, right)',
                   lambda l, s: buffered(l),
                   lambda l, s: np.maximum(left.buffered_labels(l, *dists(l)), right.buffered_labels(l, *dists(l)))))

    def capped(l, c_left, c_right):
        return new.buffered_labels(l, *dists(l), *new.per_point_caps(l, *coords(l), c_left, c_right))

    def n_anom(l):
        return len(coords(l)[0])

    def rows_below_caps(l):
        return min(caps(l)[0].min(), caps(l)[1].min()) // STEP + 1

    r.append(check('buffered labels: caps = L give the uncapped result',
                   lambda l, s: capped(l, np.full(n_anom(l), L), np.full(n_anom(l), L)),
                   lambda l, s: buffered(l)))
    r.append(check('buffered labels: rows below every cap are unchanged',
                   lambda l, s: capped(l, *caps(l))[:rows_below_caps(l)],
                   lambda l, s: buffered(l)[:rows_below_caps(l)]))
    return r


def check_compute_vs_upstream():
    """Final VUS-PR close to upstream; even windows exact where no ramps meet."""
    exact = FastVUS(L=L, step=STEP, max_thresholds=None)

    def our_areas(label, score):
        b = exact.buffered_labels(label, *dists(label))
        t = exact.thresholds(b, score)
        tp, fp, pos = exact.confusion_matrix(b, exact.score_mask(score, t))
        ex = exact.existence(b, score, t, *coords(label))
        precision, recall = exact.precision_recall(tp, fp, pos, ex)
        return np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)

    ok = True
    print('  compute vs upstream VUS-PR (all unique thresholds on both sides):')
    for case, (label, score) in toy_cases().items():
        if label.all():
            print(f"    {case:30s} skipped: upstream fails without an anomaly boundary")
            continue
        ours = exact.compute(label, score)
        tpr, _, prec, _, _, theirs = upstream.RangeAUC_volume_opt_mem(label, score, windowSize=2 * L, thre=len(score))
        their_areas = np.array([np.dot(t[1:-1] - t[:-2], p[1:]) for t, p in zip(tpr, prec)])
        line = f"    {case:30s} ours {ours:.6f}  upstream {theirs:.6f}  diff {ours - theirs:+.6f}"
        if no_overlap(label):
            even = np.abs(our_areas(label, score) - their_areas[::2]).max()
            line += f"  even-window max diff {even:.1e}"
            if even > 1e-12:
                ok = False
                line += '  <-- should be exact'
        print(line)
    return [ok]


def check_compute_properties():
    """Configurations that have no upstream counterpart: internal consistency."""
    r = []
    cum = FastVUS(L=L, step=STEP, cumulative=True)
    r.append(check('compute: cumulative[-1] == plain value (running vs plain mean)',
                   lambda l, s: cum.compute(l, s)[-1],
                   lambda l, s: new.compute(l, s), tol=1e-12))
    r.append(check('compute: cumulative[0] == L=0 value',
                   lambda l, s: cum.compute(l, s)[0],
                   lambda l, s: FastVUS(L=0, step=0).compute(l, s), tol=1e-12))
    r.append(check('compute: global mask changes nothing',
                   lambda l, s: new.compute(l, s),
                   lambda l, s: FastVUS(L=L, step=STEP, global_mask=False).compute(l, s)))
    r.append(check('compute: caps = L give the uncapped curve',
                   lambda l, s: cum.compute(l, s, np.full(len(coords(l)[0]), L), np.full(len(coords(l)[0]), L)),
                   lambda l, s: cum.compute(l, s)))
    return r


def main():
    results = []
    print('vs old VUSNumpy (unchanged steps):')
    results += check_steps_vs_old()
    print('vs upstream (buffer curve):')
    results += check_buffers_vs_upstream()
    print('vs upstream (final value):')
    results += check_compute_vs_upstream()
    print('internal properties:')
    results += check_compute_properties()
    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
