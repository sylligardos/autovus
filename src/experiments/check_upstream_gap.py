"""Decompose the gap between FastVUS and upstream VUS-PR on a few real pairs.

Three conventions differ. Each is switched on in turn, on our pipeline:
  (a) thresholds: upstream takes 250 by sorted-score index; we take every
      threshold where the surface can change;
  (b) overlap: upstream adds the ramps of neighbouring anomalies and clips at
      1; we take their max;
  (c) windows: upstream averages W = 0..2L (odd W have their own curve); we
      average the L+1 buffers, i.e. the even windows.
With upstream's thresholds and labels, our even-window areas must equal
upstream's per-window AP exactly; what is left is (c).
Small: the pairs listed below. Ask before enlarging.

    conda run -n autovus python src/experiments/check_upstream_gap.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
from tsad_parser import TSADParser

from baselines.tsb_ad.basic_metrics import basic_metricor
from vus.fastvus import FastVUS

L = 128
PAIRS = [
    ('449_UCR_id_147_Sensor_tr_2847_1st_9330.csv', 'Sub_PCA'),        # +0.24 vs upstream, 1 anomaly
    ('179_SMD_id_2_Facility_tr_5925_1st_17580.csv', 'IForest'),       # +0.034, 1 anomaly
    ('155_Stock_id_7_Finance_tr_500_1st_18.csv', 'LOF'),              # -0.034, 924 anomalies
    ('288_TODS_id_2_Synthetic_tr_500_1st_65.csv', 'IForest'),         # -0.023, 112 anomalies
    ('011_NAB_id_11_Facility_tr_1007_1st_1526.csv', 'LOF'),           # +0.003, typical
]
ALL_THRESHOLDS_ON = {'449_UCR_id_147_Sensor_tr_2847_1st_9330.csv', '179_SMD_id_2_Facility_tr_5925_1st_17580.csv'}


def areas_with(eng, buffered, score, thresholds, start, end):
    """Per-buffer AUC-PR of our pipeline on given buffered labels and thresholds."""
    sm = eng.score_mask(score, thresholds)
    tp, fp, pos = eng.confusion_matrix(buffered, sm)
    ex = eng.existence(buffered, score, thresholds, start, end)
    precision, recall = eng.precision_recall(tp, fp, pos, ex)
    return np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)


def main():
    p = TSADParser()
    names = [n for n, _ in PAIRS]
    fnames, series, labels = p.load_timeseries(filename=sorted(set(names)))
    dets = sorted({d for _, d in PAIRS})
    scores, _ = p.load_scores_and_delete(fnames, series, labels, detectors=dets)
    data = {f: (np.asarray(l).astype(int), dict(zip(dets, s))) for f, l, s in zip(fnames, labels, scores)}

    up = basic_metricor()
    eng = FastVUS(L=L, global_mask=False, max_thresholds=None)   # whole series, like upstream
    for fname, det in PAIRS:
        label, score = data[fname][0], np.asarray(data[fname][1][det], dtype=np.float64)
        T = len(label)
        _, (start, end) = eng.anomaly_coordinates(label)
        seq = list(zip(start, end))
        dl, dr = eng.distance_from_anomaly(label, start, end)
        ours_labels = eng.buffered_labels(label, dl, dr)
        up_labels = np.array([up.sequencing(label, seq, 2 * h) for h in eng.buffers])
        our_thr = eng.thresholds(ours_labels, score)
        up_thr = (-np.sort(-score))[np.linspace(0, T - 1, 250).astype(int)]

        v_ours = areas_with(eng, ours_labels, score, our_thr, start, end).mean()
        v_a = areas_with(eng, ours_labels, score, up_thr, start, end).mean()           # (a) their thresholds
        v_ab = areas_with(eng, up_labels, score, up_thr, start, end)                   # (a)+(b) their labels too
        tpr, _, prec, _, _, v_up = up.RangeAUC_volume_opt_mem(label, score, windowSize=2 * L, thre=250)
        up_ap = np.array([np.dot(t[1:-1] - t[:-2], pr[1:]) for t, pr in zip(tpr, prec)])
        even_gap = np.abs(v_ab - up_ap[::2]).max()
        print(f"{fname[:40]:40s} {det:9s} T={T:6d} anomalies={len(start):4d}")
        print(f"    ours (all thr, max overlap, even windows)      {v_ours:.5f}")
        print(f"    (a) + upstream thresholds (250 by index)       {v_a:.5f}   effect {v_a - v_ours:+.5f}")
        print(f"    (b) + upstream overlap (add and clip)          {v_ab.mean():.5f}   effect {v_ab.mean() - v_a:+.5f}")
        print(f"    (c) + upstream windows (odd ones included)     {v_up:.5f}   effect {v_up - v_ab.mean():+.5f}   = upstream")
        print(f"    even windows, ours with their conventions vs their own AP: max |diff| {even_gap:.1e}")
        if fname in ALL_THRESHOLDS_ON:
            *_, v_up_all = up.RangeAUC_volume_opt_mem(label, score, windowSize=2 * L, thre=T)
            print(f"    upstream with ALL thresholds (thre=T)          {v_up_all:.5f}   vs ours {v_ours:.5f}: diff {v_up_all - v_ours:+.5f}")


if __name__ == '__main__':
    main()
