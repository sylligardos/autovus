"""
Step-by-step comparison of VUSNumpy vs VUSTorch intermediate outputs.

Called from evaluate_ffvus when a large numerical error is detected.
Both pipelines are replayed in their actual execution order so that the
comparison reflects real behaviour, not an idealized shared sequence.

Key known divergences (expected, not bugs):
  - Thresholds: numpy uses the MASKED score; torch uses the FULL score before masking.
  - Slopes: numpy uses directional (pos_left, pos_right) distances; torch uses a single pos.
  - Existence: numpy runs existence_optimized; torch runs existence_matrix (different algo, ~0.005 avg diff).
  - dtype: numpy is float64; torch is float32 (~1e-7 rounding everywhere).
"""

import numpy as np
import torch


def compare_vus_steps(label, score, np_instance, torch_instance):
    """
    Walk VUSNumpy and VUSTorch through their pipelines step by step and
    return max absolute differences for 5 key intermediate results.

    Torch chunking is bypassed (max_memory_tokens set to 1e30 temporarily)
    so the full series is processed in one pass, matching the numpy path.

    Args:
        label          (np.ndarray): binary ground truth, shape (T,)
        score          (np.ndarray): detector scores, shape (T,), float64
        np_instance    (VUSNumpy)  : configured instance (not modified)
        torch_instance (VUSTorch)  : configured instance (max_memory_tokens restored after call)

    Returns:
        dict with keys:
            debug_thresholds_diff      — max |Δ| between threshold arrays
            debug_slopes_diff          — max |Δ| between fuzzy label matrices (n_slopes × T)
            debug_existence_diff       — max |Δ| between existence matrices (n_slopes × n_thresholds)
            debug_tp_diff              — max |Δ| between TP confusion-matrix entries
            debug_auc_per_slope_diff   — max |Δ| between per-buffer AUC before averaging
    """
    device = torch_instance.device
    label_t = torch_instance._to_tensor(label, dtype=torch.int8)
    score_t = torch_instance._to_tensor(score.astype(np.float32), dtype=torch.float32)

    orig_mem = torch_instance.max_memory_tokens
    torch_instance.max_memory_tokens = 1e30   # forces n_splits=1 in all chunk loops
    try:
        return _compare_impl(label, score, label_t, score_t, np_instance, torch_instance)
    finally:
        torch_instance.max_memory_tokens = orig_mem


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _compare_impl(label, score, label_t, score_t, nv, tv):
    device = tv.device
    out = {}

    def _to_f64(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def _max_diff(a, b):
        a, b = _to_f64(a), _to_f64(b)
        if a.shape != b.shape:
            if a.ndim == 1 and b.ndim == 1:
                n = min(len(a), len(b))
                return float(np.max(np.abs(a[:n] - b[:n])))
            r = min(a.shape[0], b.shape[0])
            c = min(a.shape[1], b.shape[1])
            return float(np.max(np.abs(a[:r, :c] - b[:r, :c])))
        return float(np.max(np.abs(a - b)))

    # =========================================================================
    # NUMPY PIPELINE  (follows VUSNumpy.compute order exactly)
    # =========================================================================

    # 1. Anomaly coordinates
    ((sne_np, ene_np), (swe_np, ewe_np)), _ = nv.get_anomalies_coordinates(label)

    # 2. Initial safe mask (extra_safe=global_mask)
    isafe_np, _ = nv.create_safe_mask(label, swe_np, ewe_np, extra_safe=nv.global_mask)

    # 3. Global mask: slice label/score, recompute anomaly coords; keep isafe_np as safe_mask
    if nv.global_mask:
        score_nonmask_np = score[~isafe_np]
        label_np  = label[isafe_np]
        score_np  = score[isafe_np]
        ((sne_np, ene_np), (swe_np, ewe_np)), _ = nv.get_anomalies_coordinates(label_np)
        safe_np = isafe_np   # numpy keeps the pre-mask safe_mask
    else:
        label_np, score_np = label, score
        score_nonmask_np = np.array([])
        safe_np = isafe_np

    # 4. Directional distances
    (pos_left_np, pos_right_np), _ = nv.distance_from_anomaly(label_np, swe_np, ewe_np)

    # 5. Slopes  (numpy uses directional pos_left / pos_right)
    slopes_np, _ = nv.add_slopes(label_np, sne_np, ene_np, pos_left_np, pos_right_np)

    # 6. Thresholds  (numpy dispatches AFTER slopes, from the MASKED score)
    thresh_np, _ = nv._dispatch_thresholds(label_np, score_np, slopes_np)

    # 7. Score mask
    sm_np, _ = nv.get_score_mask(score_np, thresh_np)

    # 8. Extra FP (searchsorted approach used in current numpy compute())
    if nv.global_mask:
        snms = np.sort(score_nonmask_np)
        extra_fp_np = len(score_nonmask_np) - np.searchsorted(snms, thresh_np, side='left')
    else:
        extra_fp_np = np.zeros(len(thresh_np))

    # 9. Existence
    existence_np, _ = nv.compute_existence(
        slopes_np, sm_np, score_np, thresh_np, swe_np, ewe_np, safe_np
    )

    # 10. Confusion matrix
    (fp_np, fn_np, tp_np, pos_np_v, neg_np, _), _ = nv.compute_confusion_matrix(slopes_np, sm_np)
    fp_np = fp_np + extra_fp_np  # (n_slopes, n_thresholds) + (n_thresholds,) → broadcasts

    # 11. Precision-recall
    (prec_np, rec_np), _ = nv.precision_recall_curve(tp_np, fp_np, pos_np_v, existence_np)

    # 12. Per-buffer AUC (stepwise, before averaging)
    auc_slope_np = np.sum((rec_np[:, 1:] - rec_np[:, :-1]) * prec_np[:, 1:], axis=1)

    # =========================================================================
    # TORCH PIPELINE  (follows VUSTorch.compute order exactly, no chunking)
    # =========================================================================

    # 1. Anomaly coordinates
    ((sne_t, ene_t), (swe_t, ewe_t)), _ = tv.get_anomalies_coordinates(label_t)

    # 2. Initial safe mask
    isafe_t, _ = tv.create_safe_mask(label_t, swe_t, ewe_t, extra_safe=tv.global_mask)

    # 3. Thresholds  (torch computes BEFORE masking, from the FULL score + initial safe_mask)
    #    Mirror the exact call in VUSTorch.compute() line: self._dispatch_thresholds(label, score, safe_mask)
    thresh_t, _ = tv._dispatch_thresholds(label_t, score_t, isafe_t)

    # 4. Global mask: slice label/score, recompute anomaly coords AND safe_mask
    if tv.global_mask:
        score_nonmask_t = score_t[~isafe_t]
        label_t_m  = label_t[isafe_t]
        score_t_m  = score_t[isafe_t]
        ((sne_t, ene_t), (swe_t, ewe_t)), _ = tv.get_anomalies_coordinates(label_t_m)
        safe_t, _ = tv.create_safe_mask(label_t_m, swe_t, ewe_t)  # torch recomputes safe_mask
    else:
        label_t_m, score_t_m = label_t, score_t
        score_nonmask_t = torch.tensor([], dtype=torch.float32, device=device)
        safe_t = isafe_t

    # 5. Single combined distance (torch does not split by direction)
    pos_t, _ = tv.distance_from_anomaly(label_t_m, swe_t, ewe_t)

    # 6. Slopes (torch uses single pos)
    slopes_t, _ = tv.add_slopes(label_t_m, pos_t)

    # 7. Score mask
    sm_t, _ = tv.get_score_mask(score_t_m, thresh_t)

    # 8. Extra FP (torch materialises the chunk score mask)
    extra_fp_t = torch.zeros(thresh_t.shape[0], device=device)
    if tv.global_mask:
        for chunk in tv.process_in_chunks(score_nonmask_t):
            sm_chunk, _ = tv.get_score_mask(chunk, thresh_t)
            extra_fp_t += sm_chunk.sum(dim=1)

    # 9. Existence
    existence_t, _ = tv.compute_existence(slopes_t, sm_t, safe_t, normalize=True)

    # 10. Confusion matrix
    (fp_t, fn_t, tp_t, pos_t_v, neg_t, _), _ = tv.compute_confusion_matrix(slopes_t, sm_t)
    fp_t = fp_t + extra_fp_t

    # 11. Precision-recall
    (prec_t, rec_t), _ = tv.precision_recall_curve(tp_t, fp_t, pos_t_v, existence_t)

    # 12. Per-buffer AUC (stepwise, before averaging)
    auc_slope_t = ((rec_t[:, 1:] - rec_t[:, :-1]) * prec_t[:, 1:]).sum(dim=1)

    # =========================================================================
    # COMPARE
    # =========================================================================

    out['debug_thresholds_diff']    = _max_diff(thresh_np, thresh_t)
    out['debug_slopes_diff']        = _max_diff(slopes_np, slopes_t)
    out['debug_existence_diff']     = _max_diff(existence_np, existence_t) if existence_np is not None else float('nan')
    out['debug_tp_diff']            = _max_diff(tp_np, tp_t)
    out['debug_auc_per_slope_diff'] = _max_diff(auc_slope_np, auc_slope_t)

    return out
