"""VUS-ROC of FastVUS against sklearn and upstream VUS on a few real pairs.

Three comparisons, all on src/baselines/tsb_ad (upstream TSB-AD HEAD):
  (1) L = 0 on a one-anomaly series equals sklearn's AUC-ROC exactly
      (existence is 1 wherever TPR > 0, so range TPR is the plain TPR);
  (2) upstream with ALL thresholds at a small L: our even windows must equal
      upstream's per-window area exactly, the final value differs only by
      upstream's odd windows and add-and-clip overlap, as for VUS-PR;
  (3) upstream with its default 250 index thresholds at L = 128: with its
      thresholds and labels plugged into our pipeline, the trapezoid over
      our (TPR, FPR) points must equal its per-window area exactly.
Upstream is slow (python loops over windows x thresholds x anomalies), so
(2) runs on the two short/sparse pairs only and (3) prints its time per pair.
Ask before enlarging.

    conda run -n autovus python src/experiments/check_roc_real.py
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
from sklearn.metrics import roc_auc_score
from tsad_parser import TSADParser

from baselines.tsb_ad.basic_metrics import basic_metricor
from autovus import FastVUS

PAIRS = [
    ('449_UCR_id_147_Sensor_tr_2847_1st_9330.csv', 'Sub_PCA'),        # 1 anomaly, T=11388
    ('011_NAB_id_11_Facility_tr_1007_1st_1526.csv', 'LOF'),           # typical, T=4031
    ('179_SMD_id_2_Facility_tr_5925_1st_17580.csv', 'IForest'),       # 1 anomaly, T=23703
    ('288_TODS_id_2_Synthetic_tr_500_1st_65.csv', 'IForest'),         # 112 anomalies, T=5000
    ('155_Stock_id_7_Finance_tr_500_1st_18.csv', 'LOF'),              # 924 anomalies, T=20000
]
ALL_THRESHOLDS = {'449_UCR_id_147_Sensor_tr_2847_1st_9330.csv', '011_NAB_id_11_Facility_tr_1007_1st_1526.csv'}
L_SMALL, L_FULL = 16, 128


def roc_points(eng, buffered, score, thresholds, start, end, T):
    """Our (TPR, FPR) per buffer at the given thresholds, closed by (0, 0) and (1, 1)."""
    sm = eng.score_mask(score, thresholds)
    tp, fp, pos = eng.confusion_matrix(buffered, sm)
    ex = eng.existence(buffered, score, thresholds, start, end)
    _, recall = eng.precision_recall(tp, fp, pos, ex)
    ones = np.ones((eng.n_buffers, 1))
    tpr = np.hstack((recall, ones))
    fpr = np.hstack((np.zeros((eng.n_buffers, 1)), fp / (T - pos), ones))
    return tpr, fpr


def trapezoid(tpr, fpr):
    return np.sum((fpr[:, 1:] - fpr[:, :-1]) * (tpr[:, 1:] + tpr[:, :-1]) / 2, axis=1)


def main():
    p = TSADParser()
    names = [n for n, _ in PAIRS]
    fnames, series, labels = p.load_timeseries(filename=sorted(set(names)))
    dets = sorted({d for _, d in PAIRS})
    scores, _ = p.load_scores_and_delete(fnames, series, labels, detectors=dets)
    data = {f: (np.asarray(l).astype(int), dict(zip(dets, s))) for f, l, s in zip(fnames, labels, scores)}
    up = basic_metricor()
    ok = True

    print('(1) L = 0, one anomaly, vs sklearn AUC-ROC:')
    for fname, det in PAIRS:
        label, score = data[fname][0], np.asarray(data[fname][1][det], dtype=np.float64)
        n_anom = len(FastVUS(L=0, step=0).anomaly_coordinates(label)[1][0])
        if n_anom != 1:
            continue
        ours = FastVUS(L=0, step=0).compute(label, score, metric='roc')
        ref = roc_auc_score(label, score)
        d = abs(ours - ref)
        ok = ok and d <= 1e-12
        print(f"    {fname[:40]:40s} {det:9s} ours {ours:.10f} sklearn {ref:.10f} diff {d:.1e}")

    print(f'(2) upstream with ALL thresholds, L = {L_SMALL} (windowSize {2 * L_SMALL}):')
    for fname, det in PAIRS:
        if fname not in ALL_THRESHOLDS:
            continue
        label, score = data[fname][0], np.asarray(data[fname][1][det], dtype=np.float64)
        T = len(label)
        eng = FastVUS(L=L_SMALL, max_thresholds=None, cumulative=True)
        t0 = time.perf_counter()
        cum = eng.compute(label, score, metric='roc')
        t_ours = time.perf_counter() - t0
        ours = np.diff(cum * np.arange(1, L_SMALL + 2), prepend=0.0)
        t0 = time.perf_counter()
        tpr, fpr, _, _, theirs, _ = up.RangeAUC_volume_opt(label, score, windowSize=2 * L_SMALL, thre=T)
        t_up = time.perf_counter() - t0
        their_areas = trapezoid(tpr, fpr)
        even = np.abs(ours - their_areas[::2]).max()
        ok = ok and even <= 1e-12
        print(f"    {fname[:40]:40s} {det:9s} T={T:6d}  ours {cum[-1]:.6f} ({t_ours:.2f}s)  upstream {theirs:.6f} ({t_up:.0f}s)  "
              f"diff {cum[-1] - theirs:+.6f}  even-window max diff {even:.1e}")

    print(f'(3) upstream default (250 index thresholds), L = {L_FULL}: their thresholds and labels in our pipeline:')
    eng = FastVUS(L=L_FULL, global_mask=False, max_thresholds=None)
    for fname, det in PAIRS:
        label, score = data[fname][0], np.asarray(data[fname][1][det], dtype=np.float64)
        T = len(label)
        _, (start, end) = eng.anomaly_coordinates(label)
        seq = list(zip(start, end))
        up_labels = np.array([up.sequencing(label, seq, 2 * h) for h in eng.buffers])
        up_thr = (-np.sort(-score))[np.linspace(0, T - 1, 250).astype(int)]
        ours_full = FastVUS(L=L_FULL).compute(label, score, metric='roc')
        v_ab = trapezoid(*roc_points(eng, up_labels, score, up_thr, start, end, T))
        t0 = time.perf_counter()
        tpr, fpr, _, _, theirs, _ = up.RangeAUC_volume_opt_mem(label, score, windowSize=2 * L_FULL, thre=250)
        t_up = time.perf_counter() - t0
        even = np.abs(v_ab - trapezoid(tpr, fpr)[::2]).max()
        ok = ok and even <= 1e-12
        print(f"    {fname[:40]:40s} {det:9s} T={T:6d} anomalies={len(start):4d}  ours (all thr) {ours_full:.5f}  "
              f"upstream {theirs:.5f} ({t_up:.0f}s)  even windows with their conventions: max diff {even:.1e}")
        sys.stdout.flush()

    print('ALL OK' if ok else 'FAILURES')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
