"""Checks for FastVUS on toy arrays only.

Reference: upstream TSB-AD (src/baselines/tsb_ad). Upstream averages windows
W = 0..2L and merges overlapping buffers by adding and clipping, so the final
value is only close; per window, the even windows must match exactly on
cases without overlapping buffers. Plus internal properties.

    conda run -n autovus python src/experiments/check_fastvus.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
from sklearn.metrics import roc_auc_score

from autovus import FastVUS
from baselines.tsb_ad.basic_metrics import basic_metricor

L, STEP = 16, 1
new = FastVUS(L=L, step=STEP)
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


def same(a, b, tol=0.0):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if tol:
        return np.allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=0)
    return np.array_equal(np.asarray(a), np.asarray(b))


def check(name, fn_a, fn_b, tol=0.0):
    """fn_a and fn_b take (label, score); outputs must be identical on every case
    (or within relative tolerance tol for results that differ only in summation order)."""
    for case, (label, score) in toy_cases().items():
        a, b = fn_a(label, score), fn_b(label, score)
        if not same(a, b, tol):
            print(f"  MISMATCH [{name}] on '{case}'\n    a: {a}\n    b: {b}")
            return False
    print(f"  ok  {name}")
    return True


def coords(label):
    return new.anomaly_coordinates(label)[1]


def dists(label):
    return new.distance_from_anomaly(label, *coords(label))


def buffered(label):
    return new.buffered_labels(label, *dists(label))


def caps(label):
    """Random per-anomaly caps, the same every call."""
    rng = np.random.default_rng(1)
    n = len(coords(label)[0])
    return rng.integers(0, L + 1, n), rng.integers(0, L + 1, n)


def no_overlap(label):
    s, e = coords(label)
    return len(s) < 2 or np.all(s[1:] - e[:-1] > 2 * L)


def decumulate(curve):
    return np.diff(np.asarray(curve) * np.arange(1, len(curve) + 1), prepend=0.0)


# ---------------------------------------------------------------- checks
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

    def caps_brute(l):
        """Left cap of the anomaly whose start is nearest ahead, right cap of the one whose end is nearest behind."""
        start, end = coords(l); cl, cr = caps(l)
        lft = np.array([cl[np.argmax(start >= t)] if (start >= t).any() else cl[-1] for t in range(len(l))], float)
        rgt = np.array([cr[len(end) - 1 - np.argmax((end <= t)[::-1])] if (end <= t).any() else cr[0] for t in range(len(l))], float)
        return lft, rgt

    r.append(check('per-point caps (anomaly ahead / behind rule, brute force)',
                   lambda l, s: new.per_point_caps(l, *coords(l), *caps(l)),
                   lambda l, s: caps_brute(l)))

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
    """Configurations without an upstream counterpart: internal consistency."""
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
    r.append(check('compute: step 4 equals every 4th buffer of step 1',
                   lambda l, s: FastVUS(L=L, step=4, cumulative=True).compute(l, s),
                   lambda l, s: np.cumsum(decumulate(cum.compute(l, s))[::4]) / np.arange(1, L // 4 + 2), tol=1e-12))
    return r


def check_roc():
    """VUS-ROC. Even windows exact vs upstream with all thresholds, for
    continuous and for tied (quantized) scores; L = 0 with one anomaly equals
    sklearn's AUC-ROC (existence is then 1 wherever TPR > 0); properties."""
    exact = FastVUS(L=L, step=STEP, max_thresholds=None, cumulative=True)
    ok = True
    print('  VUS-ROC vs upstream (all unique thresholds on both sides):')
    for quant in (None, 20):
        for case, (label, score) in toy_cases().items():
            if label.all():
                continue
            if quant:
                score = np.floor(score * quant) / quant
            cum = exact.compute(label, score, metric='roc')
            tpr, fpr, _, _, theirs, _ = upstream.RangeAUC_volume_opt_mem(label, score, windowSize=2 * L, thre=len(score))
            their_areas = np.array([np.dot(f[1:] - f[:-1], (t[1:] + t[:-1]) / 2) for t, f in zip(tpr, fpr)])
            tag = 'continuous' if quant is None else f'{quant} levels'
            line = f"    {case:30s} {tag:10s} ours {cum[-1]:.6f}  upstream {theirs:.6f}  diff {cum[-1] - theirs:+.6f}"
            if no_overlap(label):
                even = np.abs(decumulate(cum) - their_areas[::2]).max()
                line += f"  even-window max diff {even:.1e}"
                if even > 1e-12:
                    ok = False
                    line += '  <-- should be exact'
            print(line)

    print('  VUS-ROC at L = 0, one anomaly, vs sklearn AUC-ROC (continuous and tied scores):')
    rng = np.random.default_rng(1)
    worst = 0.0
    for quant in (None, 10, 3):
        for T, (a, b) in [(400, (100, 110)), (400, (0, 5)), (400, (393, 400)), (1000, (500, 501))]:
            label = np.zeros(T, int); label[a:b] = 1
            score = rng.random(T)
            if quant:
                score = np.floor(score * quant) / quant
            for gm in (True, False):
                ours = FastVUS(L=0, step=0, global_mask=gm, max_thresholds=None).compute(label, score, metric='roc')
                worst = max(worst, abs(ours - roc_auc_score(label, score)))
    print(f"    {'ok ' if worst <= 1e-12 else 'MISMATCH'}  max diff {worst:.1e} over 24 cases")
    ok = ok and worst <= 1e-12

    r = [ok]
    cum = FastVUS(L=L, step=STEP, cumulative=True)
    r.append(check('roc: cumulative[-1] == plain value',
                   lambda l, s: cum.compute(l, s, metric='roc')[-1] if not l.all() else 0,
                   lambda l, s: new.compute(l, s, metric='roc') if not l.all() else 0, tol=1e-12))
    r.append(check('roc: global mask changes nothing',
                   lambda l, s: new.compute(l, s, metric='roc') if not l.all() else 0,
                   lambda l, s: FastVUS(L=L, step=STEP, global_mask=False).compute(l, s, metric='roc') if not l.all() else 0, tol=1e-12))
    r.append(check("roc: metric='both' == ('pr', 'roc')",
                   lambda l, s: new.compute(l, s, metric='both') if not l.all() else 0,
                   lambda l, s: (new.compute(l, s), new.compute(l, s, metric='roc')) if not l.all() else 0))
    return r


def main():
    results = []
    print('vs upstream (buffer curve):')
    results += check_buffers_vs_upstream()
    print('vs upstream (final value):')
    results += check_compute_vs_upstream()
    print('internal properties:')
    results += check_compute_properties()
    print('VUS-ROC:')
    results += check_roc()
    print('ALL OK' if all(results) else 'FAILURES')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
