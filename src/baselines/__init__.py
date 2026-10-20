"""Baseline metrics: the published implementations, called through one function.

    from baselines import compute
    value, seconds = compute('vus', label, score, L=128)

Metrics (names kept from the old legacy.sylli_metrics wrapper):
    'auc'         AUC-PR, sklearn average precision (upstream metric_PR).
    'vus'         VUS-PR, upstream TSB-AD RangeAUC_volume_opt_mem.
    'range_auc'   Range-AUC-PR at window 2L, taken from upstream's own VUS loop
                  (RangeAUC_volume_opt_mem, last window), see note below.
    'rf'          Range-based F1, upstream TSB-AD metric_RF1 (oracle threshold, 100 steps).
    'affiliation' Affiliation-F, upstream TSB-AD metric_Affiliation (oracle threshold, 100 steps).
    'pate'        PATE, the authors' package (e_buffer = d_buffer = L).

Conventions:
    L is OUR per-side buffer length. Upstream's windowSize is the total window, so
    'vus' and 'range_auc' are called with windowSize = 2 * L.
    'vus' and 'range_auc' use upstream's default 250 thresholds taken by sorted-score index.
    Upstream's standalone RangeAUC() is broken at HEAD for window > 0 (it runs
    range_convers_new on the extended float labels, finds no segment, divides by zero)
    and TSB-AD itself no longer calls it. Range-AUC here is therefore the single-window
    slice of the VUS computation, with upstream's own AP formula, so it is the same code.
    All code under baselines/tsb_ad is upstream, untouched (see its NOTE.md).
"""
import time

import numpy as np
from sklearn import metrics as _skmetrics

from .tsb_ad.basic_metrics import basic_metricor

METRICS = ('auc', 'vus', 'range_auc', 'rf', 'affiliation', 'pate')


def compute(metric, label, score, L=None, **kwargs):
    """Return (value, elapsed_seconds) for one baseline metric."""
    label = np.asarray(label).astype(int)
    score = np.asarray(score, dtype=np.float64)
    start = time.time()
    if metric == 'auc':
        value = _skmetrics.average_precision_score(label, score)
    elif metric == 'vus':
        _require_L(metric, L)
        *_, value = basic_metricor().RangeAUC_volume_opt_mem(
            labels_original=label, score=score, windowSize=2 * L, **kwargs)
    elif metric == 'range_auc':
        _require_L(metric, L)
        tpr_3d, _, prec_3d, *_ = basic_metricor().RangeAUC_volume_opt_mem(
            labels_original=label, score=score, windowSize=2 * L, **kwargs)
        tpr, prec = tpr_3d[-1], prec_3d[-1]
        value = np.dot(tpr[1:-1] - tpr[:-2], prec[1:])   # upstream's AP_range formula
    elif metric == 'rf':
        value = basic_metricor().metric_RF1(label, score, **kwargs)
    elif metric == 'affiliation':
        value = basic_metricor().metric_Affiliation(label, score, **kwargs)
    elif metric == 'pate':
        _require_L(metric, L)
        from pate.PATE_metric import PATE
        value = PATE(label, score, e_buffer=L, d_buffer=L, **kwargs)
    else:
        raise ValueError(f"Unknown baseline metric {metric!r}; choose from {METRICS}")
    return float(value), time.time() - start


def _require_L(metric, L):
    if L is None:
        raise ValueError(f"Metric {metric!r} needs the buffer length L")
