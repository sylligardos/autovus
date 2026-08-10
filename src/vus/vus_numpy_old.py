"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""

import numpy as np
import math
from skimage.util.shape import view_as_windows as viewW
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.signal import find_peaks

import time
import functools
def time_it(func):
    """Wrapper to measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper


def _plot_select_best_buffer_penalized(
    ratio, slope_values, obj, argmax_ratio_idx,
    obj_argmax_idx, plateau_idx, result_idx,
    rule1, rule3, thresh, scale, plateau_eps, show=True,
):
    """Debug plot for select_best_buffer with method='penalized'."""
    sv = slope_values
    n  = len(sv)

    fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    rules_str = ('Rule 1' if rule1 else '') + (' Rule 3' if rule3 else '') + ('' if (rule1 or rule3) else 'Rule 2')
    fig.suptitle(
        f"select_best_buffer  |  method=penalized  τ={scale:.0f}  "
        f"b*={sv[result_idx]}  [{rules_str}]",
        fontsize=9, fontweight='bold',
    )

    # ---- Panel 1: raw ratio ----
    ax_r.plot(sv, ratio, color='steelblue', linewidth=1.5, label='ratio')
    ax_r.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, label='ratio=1 (random)')
    ax_r.axvline(sv[argmax_ratio_idx], color='gray', linestyle='--', linewidth=1.2,
                 alpha=0.8, label=f'argmax(ratio) = {sv[argmax_ratio_idx]}  (no rules)')
    if rule3:
        ax_r.axvline(sv[argmax_ratio_idx], color='purple', linestyle='-.', linewidth=1.8,
                     alpha=0.85, label=f'Rule 3: runaway at edge → b*=0')
    color_star = 'crimson' if (rule1 or rule3 or result_idx != argmax_ratio_idx) else 'steelblue'
    ax_r.axvline(sv[result_idx], color='crimson', linewidth=2.0, label=f'b* = {sv[result_idx]}')
    ax_r.scatter([sv[result_idx]], [ratio[result_idx]], color='crimson', s=60, zorder=5)
    ax_r.set_ylabel('ratio', fontsize=8)
    ax_r.tick_params(labelsize=7, labelbottom=False)
    ax_r.legend(fontsize=7, loc='upper right')

    # ---- Panel 2: penalized objective ----
    ax_o.plot(sv, obj, color='darkorange', linewidth=1.5, label='obj = (ratio−1)·e^{−b/τ}')
    ax_o.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    if rule1:
        # Shade the whole area below zero to show nothing is positive.
        ax_o.fill_between(sv, obj, 0, where=(obj <= 0), color='red', alpha=0.18,
                          label='Rule 1: all obj ≤ 0 → b*=0')
        mid_b = sv[n // 2]
        obj_min = float(obj.min())
        ax_o.text(mid_b, obj_min * 0.75, 'Rule 1 fired: no positive skill → b*=0',
                  color='red', fontsize=7, ha='center', va='top',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.7))

    if rule3:
        ax_o.axvline(sv[argmax_ratio_idx], color='purple', linestyle='-.', linewidth=1.8,
                     alpha=0.8, label=f'Rule 3: runaway guard')

    if not rule1 and not rule3:
        # Rule 2: show plateau region and threshold.
        ax_o.axhline(thresh, color='goldenrod', linestyle='--', linewidth=1.0,
                     alpha=0.85, label=f'plateau threshold (±{plateau_eps*100:.1f}% of max)')
        in_plateau = obj >= thresh
        ax_o.fill_between(sv, obj, thresh, where=in_plateau, color='darkorange',
                          alpha=0.22, label='Rule 2: plateau region')
        ax_o.axvline(sv[obj_argmax_idx], color='gray', linestyle='--', linewidth=1.2,
                     alpha=0.8, label=f'argmax(obj) = {sv[obj_argmax_idx]}  (before plateau)')

    ax_o.axvline(sv[result_idx], color='crimson', linewidth=2.0, label=f'b* = {sv[result_idx]}')
    ax_o.scatter([sv[result_idx]], [obj[result_idx]], color='crimson', s=60, zorder=5)

    ax_o.set_ylabel('penalized obj', fontsize=8)
    ax_o.set_xlabel(f'buffer', fontsize=8)
    ax_o.tick_params(labelsize=7)
    ax_o.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_select_best_buffer_simple(ratio, slope_values, argmax_idx, result_idx, rules_fired, method, show=True):
    """Debug plot for select_best_buffer for argmax / elbow / first_peak."""
    sv = slope_values

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    rules_str = ', '.join(name for name, *_ in rules_fired)
    fig.suptitle(
        f"select_best_buffer  |  method={method}  b*={sv[result_idx]}  [{rules_str}]",
        fontsize=9, fontweight='bold',
    )

    ax.plot(sv, ratio, color='steelblue', linewidth=1.5, label='ratio')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, label='ratio=1 (random)')

    if argmax_idx != result_idx:
        ax.axvline(sv[argmax_idx], color='gray', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'argmax(ratio) = {sv[argmax_idx]}  (before rules)')

    for i, (name, desc, _) in enumerate(rules_fired):
        ax.annotate(
            f'{name}: {desc}',
            xy=(0.02, 0.95 - i * 0.09),
            xycoords='axes fraction', fontsize=7, color='dimgray', va='top',
        )

    ax.axvline(sv[result_idx], color='crimson', linewidth=2.0, label=f'b* = {sv[result_idx]}')
    ax.scatter([sv[result_idx]], [ratio[result_idx]], color='crimson', s=60, zorder=5)

    ax.set_ylabel('ratio', fontsize=8)
    ax.set_xlabel('buffer', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def _apply_penalized_rules(
    obj, ratio, slope_values,
    argmax_ratio_idx, obj_argmax_idx, plateau_idx,
    score=None, core_mask=None, gate_thr=None,
    rules=None, flat_eps=1e-4,
):
    """Apply all buffer-selection rules and return (result_idx, fired_dict).

    Rules (evaluated in priority order):
        R4a — degenerate score (near-constant, no signal)          → b*=0
        R4b — detection gate (core max score ≤ gate percentile)    → b*=0
        R1  — no buffer offers meaningful skill                     → b*=0
              fires when the skill ratio is flat (np.ptp(ratio) <= flat_eps)
              or no buffer beats random by more than flat_eps
              (obj = ratio-1 never exceeds flat_eps). The flat_eps tolerance
              prevents a genuinely no-skill side (ratio ≡ 1 up to ~1e-14
              floating-point jitter) from defeating the guard and letting
              argmax wander to a large buffer.
        R3  — runaway guard (raw argmax at rightmost edge)         → b*=0
        R2  — first-plateau preference (parsimony tie-break)
              if disabled, fall back to pure obj argmax

    Args:
        obj               : penalized objective array, shape (n,)
        ratio             : raw ratio array, shape (n,)
        slope_values      : buffer values, shape (n,)
        argmax_ratio_idx  : index of argmax(ratio)
        obj_argmax_idx    : index of argmax(obj)
        plateau_idx       : index of first entry in obj plateau (R2 candidate)
        score             : detector score array (needed for R4a/R4b)
        core_mask         : integer index array of anomaly core positions (R4b)
        gate_thr          : score threshold for the detection gate (R4b)
        rules             : dict of rule flags; None means all active (production
                            default). Any dict key not present defaults to False,
                            so passing e.g. {'r2': True} activates only R2.

    Returns:
        result_idx (int), fired (dict[str, bool])
    """
    if rules is None:
        r = {'r1': True, 'r2': True, 'r3': True, 'r4a': True, 'r4b': True}
    else:
        r = {'r1': False, 'r2': False, 'r3': False, 'r4a': False, 'r4b': False}
        r.update(rules)

    fired = {k: False for k in ('r1', 'r2', 'r3', 'r4a', 'r4b')}
    n = len(ratio)

    # R4a — degenerate (near-constant) score
    if r['r4a'] and score is not None:
        fired['r4a'] = bool(np.ptp(score) <= 1e-12)

    # R4b — detection gate: core max doesn't beat the background
    if r['r4b'] and not fired['r4a'] and core_mask is not None and gate_thr is not None:
        if core_mask.size > 0:
            fired['r4b'] = bool(float(np.max(score[core_mask])) <= gate_thr)

    if fired['r4a'] or fired['r4b']:
        return 0, fired

    # R1 — no buffer offers meaningful skill (flat ratio, or skill never
    # exceeds flat_eps). The flat-ratio clause (np.ptp) catches a side with no
    # reachable detection: its ratio is constant (≈1, or any constant level)
    # up to ~1e-14 jitter, which would otherwise slip past `obj > flat_eps` and
    # send argmax to a large buffer on pure rounding noise.
    if r['r1']:
        fired['r1'] = bool(np.ptp(ratio) <= flat_eps or not np.any(obj > flat_eps))

    # R3 — runaway guard
    if r['r3'] and not fired['r1']:
        fired['r3'] = bool(n > 1 and argmax_ratio_idx == n - 1)

    if fired['r1'] or fired['r3']:
        return 0, fired

    # R2 — first-plateau preference (parsimony tie-break)
    fired['r2'] = bool(r['r2'])
    return (plateau_idx if r['r2'] else obj_argmax_idx), fired


def select_best_buffer(ratio, slope_values, method='penalized', tau=None, plateau_eps=1e-3, plot=False,
                       rules=None, score=None, core_mask=None, gate_thr=None, flat_eps=1e-4):
    """
    Given a ratio curve ratio[b] = VUS_detector(b) / E[VUS_random(b)], select the best
    buffer index using the specified method.

    Args:
        ratio        (np.ndarray): shape (n_slopes,)
        slope_values (np.ndarray): shape (n_slopes,); actual buffer lengths (e.g. [0,1,...,L])
        method       (str): one of 'argmax', 'elbow', 'first_peak', 'penalized'
        tau          (float): tolerance scale for the 'penalized' method; the
            skill-above-random is discounted by exp(-b / tau). Larger tau tolerates
            larger buffers. Defaults to the full buffer range when None.
        plateau_eps  (float): for 'penalized', buffers whose penalized objective is
            within this relative epsilon of the best are treated as tied, and the
            smallest such buffer is chosen (rule 2, first-plateau preference).
        plot         (bool): if True, show a debug figure with the ratio curve,
            intermediate objectives, and an annotation for every rule that fired.

    Returns:
        dict with keys:
            'buffer_idx' (int):   index into slope_values
            'buffer_val' (int):   actual buffer length
            'ratio'      (float): ratio value at that buffer
    """
    n = len(ratio)

    def _pack(idx):
        return {
            'buffer_idx': int(idx),
            'buffer_val': int(slope_values[idx]),
            'ratio':      float(ratio[idx]),
        }

    argmax_idx = int(np.argmax(ratio))

    if method == 'penalized':
        # b* = argmax of skill-above-random (ratio - 1). No distance penalty: the
        # skill ratio already self-limits — extending the buffer past the detection
        # adds false-positive area and lowers the ratio, so over-prediction peaks at
        # b=0 on its own. Runaways from *missed* detections are handled by the gate
        # (R4) and the runaway guard (R3), not a penalty. (tau is retained only for
        # the debug plot's annotation and as a no-op for API compatibility.)
        gain  = ratio - 1.0
        scale = tau if (tau is not None and tau > 0) else float(slope_values[-1])
        obj   = gain

        # Always compute plateau candidates so they're available for the plot.
        best_obj_val   = float(obj.max())
        thresh         = best_obj_val - plateau_eps * abs(best_obj_val)
        obj_argmax_idx = int(np.argmax(obj))
        plateau_idx    = int(np.argmax(obj >= thresh))   # R2 candidate

        result_idx, fired = _apply_penalized_rules(
            obj, ratio, slope_values,
            argmax_idx, obj_argmax_idx, plateau_idx,
            score=score, core_mask=core_mask, gate_thr=gate_thr,
            rules=rules, flat_eps=flat_eps,
        )

        if plot:
            rule1 = fired['r1']
            rule3 = fired['r3']
            _plot_select_best_buffer_penalized(
                ratio, slope_values, obj, argmax_idx,
                obj_argmax_idx, plateau_idx, result_idx,
                rule1, rule3, thresh, scale, plateau_eps,
            )

        return _pack(result_idx)

    if method == 'argmax':
        rules_fired = []
        result_idx  = argmax_idx

        # Rule 1 — detector never beats random.
        if ratio[argmax_idx] < 1.0:
            rules_fired.append(('Rule 1', 'ratio < 1 everywhere → b*=0', 0))
            result_idx = 0
        else:
            gain_scalar = (ratio[argmax_idx] - ratio[0]) / (abs(ratio[0]) + 1e-10)
            # Rule 2 — flat curve, negligible gain.
            if gain_scalar < 1e-3:
                rules_fired.append(('Rule 2', f'flat curve (gain={gain_scalar:.2e}) → b*=0', 0))
                result_idx = 0
            else:
                rules_fired.append(('argmax', f'b={slope_values[argmax_idx]}', argmax_idx))

        if plot:
            _plot_select_best_buffer_simple(ratio, slope_values, argmax_idx, result_idx, rules_fired, 'argmax')

        return _pack(result_idx)

    if method == 'elbow':
        x      = np.linspace(0.0, 1.0, n)
        y      = ratio.astype(float)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
        dy     = float(y_norm[-1] - y_norm[0])
        dist   = (x * dy - (y_norm - y_norm[0])) / (np.sqrt(1.0 + dy ** 2) + 1e-10)
        result_idx = int(np.argmax(np.abs(dist)))
        rules_fired = [('elbow', f'max dist from chord → b={slope_values[result_idx]}', result_idx)]

        if plot:
            _plot_select_best_buffer_simple(ratio, slope_values, result_idx, result_idx, rules_fired, 'elbow')

        return _pack(result_idx)

    if method == 'first_peak':
        peaks, _ = find_peaks(ratio)
        result_idx = int(peaks[0]) if len(peaks) > 0 else argmax_idx
        rules_fired = [('first_peak', f'b={slope_values[result_idx]}', result_idx)]

        if plot:
            _plot_select_best_buffer_simple(ratio, slope_values, result_idx, result_idx, rules_fired, 'first_peak')

        return _pack(result_idx)

    raise ValueError(f"Unknown method: {method!r}. Choose 'auto', 'argmax', 'elbow', or 'first_peak'.")


class VUSNumpy():
    def __init__(
            self,
            slope_size=100,
            step=1,
            zita=(1/math.sqrt(2)),
            global_mask=True,
            slopes='function',
            existence='optimized',
            conf_matrix='dynamic_plus',
            interpolation='stepwise',
            metric='vus_pr',
            side='both',
            cumulative=False,
            aggregation='mean',
            fn_alpha=0.0,
            thresholds='pr_relevant_fuzzy',
            max_thresholds=2000,
        ):
        """
        Initialize the VUSNumpy metric class.

        Args:
            slope_size (int): Number of slope values for anomaly encoding.
            step (int): Increment between slope values; must evenly divide slope_size.
            zita (float): Slope parameter, must be in [0, 1].
            global_mask (bool): Whether to apply a global mask to the data.
            slopes (str): Slope computation mode, either 'precomputed' or 'function'.
            existence (str): Existence calculation mode: 'None', 'trivial', 'optimized', or 'matrix'.
            conf_matrix (str): Confusion matrix calculation mode: 'trivial', 'dynamic', or 'dynamic_plus'.
            interpolation (str): Interpolation method for PR curve: 'linear' or 'stepwise'.
            metric (str): Metric to compute: 'auc_pr', 'auc_roc', 'vus_pr', 'vus_roc', or 'all'.
            aggregation (str): How to aggregate AUC values across buffer sizes: 'mean', 'geometric', or 'harmonic'.
            fn_alpha (float): FN strictness in [0, 1]. 0 = standard VUS (hard FN only); 1 = fuzzy FN (buffer misses also penalized). Only applies to conf_matrix='trivial'.
            thresholds (str): Threshold selection strategy: 'all_unique', 'subsampled', 'pr_relevant', or 'pr_relevant_fuzzy'.
            max_thresholds (int or None): Max thresholds kept for 'subsampled' strategy; ignored otherwise.
        """
        if slope_size < 0 or step < 0:
            raise ValueError(f"Error with the slope_size {slope_size} or step {step}. They should be positive values.")
        if step == 0 and slope_size > 0:
            raise ValueError(f"Error with step {step} and slope_size {slope_size}. Step can't be 0 with non-zero slope_size.")
        if (step > 0 and slope_size > 0) and (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 or zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")
        
        self.slope_size, self.step = slope_size, step
        self.zita, self.global_mask = zita, global_mask
        self.cumulative = cumulative
        
        self.slope_values = np.arange(self.slope_size + 1, step=self.step) if step > 0 else np.array([0])
        self.n_slopes = self.slope_values.shape[0]

        # Define valid argument sets
        slopes_args = ['precomputed', 'function']
        existence_args = ['None', 'trivial', 'optimized', 'matrix']
        conf_matrix_args = ['trivial', 'dynamic', 'dynamic_plus']
        interpolation_args = ['linear', 'stepwise']
        metric_args = ['auc_pr', 'auc_roc', 'vus_pr', 'vus_roc', 'all']
        side_args = ['both', 'left', 'right']
        aggregation_args = ['mean', 'geometric', 'harmonic']
        thresholds_args = ['all_unique', 'subsampled', 'pr_relevant', 'pr_relevant_fuzzy']

        # Validate and set modes
        self.add_slopes_mode = self.validate_args(slopes, slopes_args)
        self.existence_mode = self.validate_args(existence, existence_args)
        self.conf_matrix_mode = self.validate_args(conf_matrix, conf_matrix_args)
        self.interpolation_mode = self.validate_args(interpolation, interpolation_args)
        self.metric_mode = self.validate_args(metric, metric_args)
        self.reward_side = self.validate_args(side, side_args)
        self.aggregation_mode = self.validate_args(aggregation, aggregation_args)
        self.fn_alpha = float(fn_alpha)
        self.thresholds_mode = self.validate_args(thresholds, thresholds_args)
        self.max_thresholds = max_thresholds

        # Precompute slopes
        self.prepare_precomputed_slopes()


    def validate_args(self, value, valid_options):
        if value not in valid_options:
            raise ValueError(f"Unknown argument {value}. Must be one of {valid_options}.")
        return value
    
    def prepare_precomputed_slopes(self):
        if self.add_slopes_mode == 'precomputed':
            if self.slope_size > 0:
                self.pos_slopes = self.precompute_slopes()
                self.neg_slopes = self.pos_slopes[:, ::-1]
            else:
                self.pos_slopes, self.neg_slopes = np.array([]), np.array([])

    def _dispatch_thresholds(self, label, score, label_slopes, score_full=None):
        # Thresholds must range over the FULL (unmasked) score so that global_mask
        # stays a pure optimization and does not change the threshold set. The
        # legacy VUS and VUSTorch both select thresholds from the full score.
        # score_full carries it for the score-only modes; the pr_relevant modes
        # stay coupled to the masked (label-aligned) score.
        score_unmasked = score if score_full is None else score_full
        if self.thresholds_mode == 'all_unique':
            return self.get_unique_thresholds(score_unmasked, max_thresholds=None)
        elif self.thresholds_mode == 'subsampled':
            return self.get_unique_thresholds(score_unmasked, max_thresholds=self.max_thresholds)
        elif self.thresholds_mode == 'pr_relevant':
            return self.get_pr_relevant_thresholds(label, score)
        elif self.thresholds_mode == 'pr_relevant_fuzzy':
            return self.get_pr_relevant_thresholds_vus(label_slopes, score, max_thresholds=self.max_thresholds)

    @time_it
    def compute(self, label, score, thresholds=None, adaptive=False, max_buffer_per_anomaly=None,
                max_buffer_per_anomaly_left=None, max_buffer_per_anomaly_right=None):
        """
        Main computation function for the VUS metric.

        Args:
            label (np.ndarray): Binary ground truth vector.
            score (np.ndarray): Prediction scores.
            max_buffer_per_anomaly (np.ndarray, optional): Per-anomaly maximum
                buffer length, shape (n_anomalies,), ordered like the anomalies
                returned by get_anomalies_coordinates. When given, each anomaly's
                soft buffer is capped at its own value instead of the uniform
                slope_size (see add_slopes_function). When None (the default,
                original pipeline), buffers grow uniformly to slope_size.
            adaptive (bool): when True, return the per-buffer b* selection from
                _select_adaptive_buffer (argmax of the marginal skill ratio)
                instead of the scalar VUS. Requires cumulative=True.

        Returns:
            vus_pr (float): Computed VUS-PR metric.
            time_analysis (dict): Timing breakdown for each computation step.
        """
        T = len(label)

        score_full = score  # full, unmasked score for threshold selection

        ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), time_anom_coord = self.get_anomalies_coordinates(label)
        safe_mask, time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=self.global_mask)

        if self.global_mask:
            score_nonmask = score[~safe_mask]
            label, score = label[safe_mask], score[safe_mask]
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), extra_time_anom_coord = self.get_anomalies_coordinates(label)
            time_anom_coord += extra_time_anom_coord

        (pos_left, pos_right), time_pos = self.distance_from_anomaly(label, start_with_edges, end_with_edges)

        cap_left = cap_right = None
        b_left, b_right = self._resolve_caps(
            max_buffer_per_anomaly, max_buffer_per_anomaly_left, max_buffer_per_anomaly_right)
        if b_left is not None:
            cap_left, cap_right = self._per_point_cap(
                label, start_with_edges, end_with_edges, b_left, b_right)

        labels, time_slopes = self.add_slopes(
            label, start_no_edges, end_no_edges, pos_left, pos_right,
            cap_left=cap_left, cap_right=cap_right)

        if thresholds is None:
            thresholds, time_thresholds = self._dispatch_thresholds(label, score, labels, score_full=score_full)
        else:
            thresholds = np.array(thresholds) if isinstance(thresholds, list) else thresholds
            if not isinstance(thresholds, np.ndarray):
                raise ValueError(f"thresholds must be a list or numpy array, got {type(thresholds)}")
            time_thresholds = 0

        sm, time_sm = self.get_score_mask(score, thresholds)
        if self.global_mask:
            # print(score_nonmask.shape, thresholds.shape)
            # extra_sm, extra_time_sm = self.get_score_mask(score_nonmask, thresholds)
            # extra_fp = extra_sm.sum(axis=1)
            # time_sm += extra_time_sm

            # Alternative: avoids materializing the (n_thresholds, T_nonmask) matrix
            t0 = time.time()
            score_nonmask_sorted = np.sort(score_nonmask)
            extra_fp = len(score_nonmask) - np.searchsorted(score_nonmask_sorted, thresholds, side='left')
            extra_time_sm = time.time() - t0
            time_sm += extra_time_sm
            # print(f"score_nonmask time: {extra_time_sm}s")
            # print(f"match: {np.array_equal(extra_fp, extra_fp_v2)}  old: {extra_time_sm}s  new: {t_v2}s")
            # if not np.array_equal(extra_fp, extra_fp_v2):
            #     raise ValueError("CHEEEECK")

        existence, time_existence = self.compute_existence(labels, sm, score, thresholds, start_with_edges, end_with_edges, safe_mask)
        (fp, fn, tp, positives, negatives, fpr), time_confusion = self.compute_confusion_matrix(labels, sm)

        if self.global_mask:
            fp += extra_fp

        (precision, recall), time_pr_rec = self.precision_recall_curve(tp, fp, positives, existence)
        vus_pr, time_integral = self.auc(recall, precision)

        time_analysis = {
            "Anomaly coordinates time": time_anom_coord,
            "Safe mask time": time_safe_mask,
            "Thresholds time": time_thresholds,
            "n_thresholds": len(thresholds),
            "Score mask time": time_sm,
            "Position time": time_pos,
            "Slopes time": time_slopes,
            "Existence time": time_existence,
            "Confusion matrix time": time_confusion,
            "Precision recall curve time": time_pr_rec,
            "Integral time": time_integral,
        }

        if adaptive:
            if not self.cumulative:
                raise ValueError("adaptive=True requires the VUSNumpy instance to be initialized with cumulative=True")
            result, adaptive_time = self._select_adaptive_buffer(labels, vus_pr, T)
            time_analysis['Adaptive buffer time'] = adaptive_time
            return result, time_analysis

        return vus_pr, time_analysis

    def _compute_denom_curve(self, labels, T, denom, K, rng):
        """Compute the expected random-VUS denominator curve (shape: n_slopes).

        Args:
            labels (np.ndarray): Slope array (n_slopes, T_masked); row 0 is the binary label.
            T      (int): Original series length (before global masking).
            denom  (str): 'analytical', 'mc_bernoulli', or 'mc_uniform'.
            K      (int): Number of MC draws (ignored for 'analytical').
            rng    : numpy random Generator; created fresh when None (MC only).

        Returns:
            np.ndarray of shape (n_slopes,).
        """
        if denom == 'analytical':
            # E[AUC-PR(b)] = g_b = sum of buffered label / T (original length).
            # Dividing by T (not T_masked) is correct: the denominator must reflect
            # the full series so the ratio is consistent with how VUS is normalised.
            g_b = labels.sum(axis=1) / T
            return np.cumsum(g_b) / np.arange(1, self.n_slopes + 1)

        if rng is None:
            rng = np.random.default_rng()
        masked_label = labels[0]
        masked_T     = labels.shape[1]
        total        = np.zeros(self.n_slopes)

        if denom == 'mc_bernoulli':
            p = masked_label.mean()
            for _ in range(K):
                rand_score = rng.binomial(1, p, size=masked_T).astype(float)
                (rand_vus, _), _ = self.compute(masked_label, rand_score)
                total += rand_vus

        elif denom == 'mc_uniform':
            for _ in range(K):
                rand_score = rng.uniform(0, 1, size=masked_T)
                (rand_vus, _), _ = self.compute(masked_label, rand_score)
                total += rand_vus

        else:
            raise ValueError(f"Unknown denom: {denom!r}. Choose 'analytical', 'mc_bernoulli', or 'mc_uniform'.")

        return total / K

    @time_it
    def _select_adaptive_buffer(self, labels, vus_cumulative, T):
        """Select b* by argmax of the per-buffer (marginal) skill ratio.

        The cumulative VUS curve is the running mean of AUC over buffers [0, b].
        De-cumulating it ONCE recovers the per-buffer AUC(b); dividing by the
        analytical expected-random AUC gives the marginal skill ratio

            ratio[b] = AUC_detector(b) / E[AUC_random(b)].

        b* is its argmax: the buffer width that best captures the detector's
        skill over random, consistent with VUS ("count any detection within b").
        The only guard kept here returns b*=0 when no buffer beats random
        (ratio < 1 everywhere), i.e. there is nothing worth including.

        Returns a dict with the selection ('best') and the curves used, so
        callers (per-anomaly windows, debug plots) can reuse them.
        """
        denom = self._compute_denom_curve(labels, T, 'analytical', 0, None)

        n_idx    = np.arange(1, self.n_slopes + 1)
        marg_vus = np.diff(vus_cumulative * n_idx, prepend=0.0)   # per-buffer AUC(b)
        marg_den = np.diff(denom          * n_idx, prepend=0.0)
        ratio    = marg_vus / np.clip(marg_den, 1e-10, None)

        b_idx = int(np.argmax(ratio))
        # Guard: no buffer beats random -> nothing worth including -> b*=0.
        if not np.any((ratio - 1.0) > 1e-4):
            b_idx = 0

        best = {
            'buffer_idx': b_idx,
            'buffer_val': int(self.slope_values[b_idx]),
            'ratio':      float(ratio[b_idx]),
            'vus_score':  float(vus_cumulative[b_idx]),
        }
        return {'vus': vus_cumulative, 'ratio': ratio, 'denom': denom, 'best': best}

    def _select_window(self, win_label, win_score, score_scale, min_signal_frac):
        """Pick b* for one anomaly window using the current self.reward_side.

        Runs cumulative VUS on the window, takes the marginal-ratio argmax
        (see _select_adaptive_buffer), then clamps b* to the buffer room
        available on the rewarded side(s) and zeroes it when the window holds no
        real signal. Callers flip self.reward_side around this call to select a
        buffer for one side at a time.
        """
        (res, _), _ = self.compute(win_label, win_score, adaptive=True)
        best  = res['best']
        ratio = res['ratio']

        # Reachable-buffer cap. Past the window edge there is no series, so a
        # buffer cannot be placed there. The VUS soft label still grows beyond the
        # edge (a larger slope_size only makes the ramp shallower, inflating the
        # soft value at already-included points), which can make the marginal
        # ratio climb to slope_size when a detection sits flush against the edge.
        # Restrict b* to the room actually available: left/right room for one-sided
        # selection, their min for the shared symmetric ('both') buffer.
        ones = np.flatnonzero(win_label > 0)
        if ones.size:
            Lw         = len(win_label)
            left_room  = int(ones[0])
            right_room = int(Lw - 1 - ones[-1])
            if self.reward_side == 'left':
                room = left_room
            elif self.reward_side == 'right':
                room = right_room
            else:
                room = min(left_room, right_room)

            if best['buffer_val'] > room:
                allowed = self.slope_values <= room
                new_idx = int(np.argmax(np.where(allowed, ratio, -np.inf)))
                # No reachable buffer beats random -> b*=0 (mirrors the guard above).
                if not np.any((ratio[allowed] - 1.0) > 1e-4):
                    new_idx = 0
                best['buffer_idx'] = new_idx
                best['buffer_val'] = int(self.slope_values[new_idx])
                best['ratio']      = float(ratio[new_idx])
                best['vus_score']  = float(res['vus'][new_idx])

            # Hide the unreachable tail from the debug plot (pure ramp inflation).
            res['ratio'] = np.where(self.slope_values <= room, ratio, np.nan)
            res['reachable_buffer'] = int(room)

        # No-signal guard: a window whose score never rises to a meaningful
        # fraction of the detector's global scale holds no detection -> b*=0.
        slice_max = float(np.max(win_score)) if win_score.size else 0.0
        if best['buffer_val'] != 0 and (score_scale <= 0 or slice_max < min_signal_frac * score_scale):
            best['buffer_idx'] = 0
            best['buffer_val'] = 0
            best['ratio']      = float(ratio[0])
            best['vus_score']  = float(res['vus'][0])

        # Optional guard (off by default): near-constant score carries no signal.
        # if np.ptp(win_score) <= 1e-12: best['buffer_idx'] = best['buffer_val'] = 0

        return res

    @staticmethod
    def classify_detection(r, tol=0):
        """Label a per-anomaly result by detection timing (no extra computation).

        Reads only the buffer widths already chosen by
        ``compute_per_anomaly_buffer``. A grown pre-anomaly (left)
        buffer means the detector fired before the anomaly; a grown post-anomaly
        (right) buffer means it fired after. ``b*=0`` is overloaded (exact vs
        missed), so it is split by the inside skill, i.e. whether the marginal
        ratio at buffer 0 beats random.

        per_side results (carry 'best_left'/'best_right'/'ratio_left'):
            'missed' : no skill inside and neither buffer grew
            'exact'  : detected inside, neither buffer needed (b*_left,b*_right<=tol)
            'early'  : left buffer grew  (detection precedes the anomaly)
            'late'   : right buffer grew (detection lags the anomaly)
                       when both grow, the larger side wins (ties -> 'late').

        symmetric results (single 'best', direction unknown):
            'missed' / 'exact' / 'lagged' (b* > tol)

        Args:
            r   (dict): one entry from compute_per_anomaly_buffer.
            tol (int) : buffer widths <= tol count as "not grown" (default 0).
        """
        if 'best_left' in r:
            b_left  = r['best_left']['buffer_val']
            b_right = r['best_right']['buffer_val']
            if b_left <= tol and b_right <= tol:
                return 'exact' if float(r['ratio_left'][0]) > 1.0 else 'missed'
            return 'early' if b_left - b_right > tol else 'late'

        b = r['best']['buffer_val']
        if b <= tol:
            return 'exact' if float(r['ratio'][0]) > 1.0 else 'missed'
        return 'lagged'

    @staticmethod
    def floor_dust(score, dust_floor=1e-6):
        """Snap floating-point dust to zero; returns a cleaned COPY (never mutates).

        Some detectors (e.g. AnomalyTransformer, via exp()/softmax underflow) emit
        scores like 5e-33 or 1e-24 that are numerically nonzero but carry no signal.
        Each distinct dust value becomes its own threshold at the very bottom of the
        PR curve, and when an adaptive buffer grows just far enough to reach a dust
        point sitting outside the core it flips that point from a false-positive into
        a fractional true-positive. That nudges precision in the bottom band,
        producing a spurious micro-bump in the per-buffer skill ratio exactly at the
        buffer width that reaches the dust -- which argmax then latches onto (e.g.
        b*_right=4 for a detection that actually sits squarely inside the anomaly).

        The floor is RELATIVE to the detector's own scale (|score| < dust_floor *
        max|score|), so it is invariant to un-normalised score ranges; with the usual
        [0, 1] min-max normalisation it is just an absolute 1e-6 cutoff. dust_floor
        of 0/None disables it (returns score unchanged, as an array).
        """
        score = np.asarray(score, dtype=float)
        if not dust_floor or score.size == 0:
            return score
        smax = float(np.max(np.abs(score)))
        if smax <= 0:
            return score
        return np.where(np.abs(score) < dust_floor * smax, 0.0, score)

    def compute_per_anomaly_buffer(self, label, score, per_side=True, plot=False,
                                   dust_floor=1e-6, min_signal_frac=1e-3):
        """Per-anomaly adaptive buffer via marginal-ratio argmax.

        For each anomaly k, slice a window spanning half the gap to each
        neighbour, run cumulative VUS on that window, and pick
        b* = argmax of the per-buffer skill ratio (see _select_adaptive_buffer),
        clamped to the buffer room actually available inside the window. Short
        and long anomalies thus pick different buffers.

        per_side=True (default): pick a separate b* for each side of every
        anomaly -- the window is scored once with reward_side='left' (b*_left,
        the pre-anomaly buffer) and once with 'right' (b*_right, the post-anomaly
        buffer) -- so the two sides can grow to different widths. per_side=False
        uses a single symmetric b* shared by both sides.

        Guards (only ever shrink a runaway to 0; they never drop a real
        detection, they fire only when there is no signal to include):
            - no buffer beats random              -> b*=0 (in _select_adaptive_buffer)
            - window max score < min_signal_frac * global max -> b*=0 (noise)
            - dust_floor: |score| < dust_floor * max|score| snapped to 0 first

        After selecting b* for every anomaly, the full series is run once through
        cumulative VUS with these per-anomaly caps (see compute's
        max_buffer_per_anomaly* arguments); the result is stored on every dict as
        'final_vus' (and 'baseline_vus' is the no-caps VUS). Each result is
        stamped with classify_detection's 'detection_type'.

        Args:
            label (np.ndarray): binary ground truth, shape (T,)
            score (np.ndarray): detector scores, shape (T,)
            per_side (bool)   : two-sided selection (default) vs one symmetric b*.
            plot  (bool)      : show a per-anomaly debug figure for each anomaly.
            dust_floor (float): relative cutoff below which scores are snapped to
                zero before selection (floating-point underflow, not signal). Set
                to 0/None to disable. See floor_dust for the rationale.
            min_signal_frac (float): a window whose max score is below this
                fraction of the detector's global max is treated as signal-free.

        Returns:
            list of per-anomaly dicts with keys: 'start', 'end', 'window',
            'best' (buffer_idx/buffer_val/ratio/vus_score), 'vus', 'ratio',
            'final_vus', 'baseline_vus', 'detection_type'. In per_side mode each
            dict also carries 'best_left'/'best_right', 'vus_left'/'vus_right',
            'ratio_left'/'ratio_right'; 'best' is the larger-b* side (kept for
            backward compatibility with symmetric consumers).
        """
        if not self.cumulative:
            raise ValueError("compute_per_anomaly_buffer requires cumulative=True")

        T = len(label)
        ((_, _), (starts, ends)), _ = self.get_anomalies_coordinates(label)
        n_anomalies = len(starts)
        if n_anomalies == 0:
            return []

        # Floor floating-point dust to zero before selection (see floor_dust).
        # Selection and any downstream score_adapt MUST run on the SAME cleaned
        # score, so callers driving both should clean once and pass dust_floor=0.
        score = self.floor_dust(score, dust_floor)
        score_scale = float(np.max(score)) if score.size else 0.0

        orig_side = self.reward_side
        results = []
        try:
            for k in range(n_anomalies):
                l_k = int((ends[k - 1] + starts[k]) // 2) if k > 0 else 0
                r_k = int((ends[k] + starts[k + 1]) // 2) if k < n_anomalies - 1 else T
                win_label, win_score = label[l_k:r_k], score[l_k:r_k]

                r = {'start': int(starts[k]), 'end': int(ends[k]), 'window': (l_k, r_k)}

                if not per_side:
                    res = self._select_window(win_label, win_score, score_scale, min_signal_frac)
                    r['best'], r['vus'], r['ratio'] = res['best'], res['vus'], res['ratio']
                else:
                    self.reward_side = 'left'
                    res_l = self._select_window(win_label, win_score, score_scale, min_signal_frac)
                    self.reward_side = 'right'
                    res_r = self._select_window(win_label, win_score, score_scale, min_signal_frac)
                    self.reward_side = orig_side

                    r['best_left'],  r['best_right']  = res_l['best'],  res_r['best']
                    r['vus_left'],   r['vus_right']   = res_l['vus'],   res_r['vus']
                    r['ratio_left'], r['ratio_right'] = res_l['ratio'], res_r['ratio']
                    # Backward-compat single 'best': the side with the larger b*.
                    dom = res_l if res_l['best']['buffer_val'] >= res_r['best']['buffer_val'] else res_r
                    r['best'], r['vus'], r['ratio'] = dom['best'], dom['vus'], dom['ratio']

                results.append(r)
        finally:
            self.reward_side = orig_side

        # Recompute VUS on the full series, capping each anomaly at its b*.
        if per_side:
            b_left  = np.array([r['best_left']['buffer_val']  for r in results])
            b_right = np.array([r['best_right']['buffer_val'] for r in results])
            (vus_curve, _), _ = self.compute(
                label, score,
                max_buffer_per_anomaly_left=b_left,
                max_buffer_per_anomaly_right=b_right)
        else:
            b_star = np.array([r['best']['buffer_val'] for r in results])
            (vus_curve, _), _ = self.compute(label, score, max_buffer_per_anomaly=b_star)
        final_vus = float(vus_curve[-1])

        # Baseline: full-buffer VUS with no adaptive caps.
        (baseline_curve, _), _ = self.compute(label, score)
        baseline_vus = float(baseline_curve[-1])

        for r in results:
            r['final_vus']      = final_vus
            r['baseline_vus']   = baseline_vus
            r['detection_type'] = self.classify_detection(r)

        if plot:
            for k, r in enumerate(results):
                if per_side:
                    self._plot_anomaly_debug_per_side(label, score, r, k, n_anomalies, T)
                else:
                    self._plot_anomaly_debug(label, score, r, k, n_anomalies, T)

        return results

    def compute_per_anomaly_adaptive_masking(self, label, score, plot=False):
        """Per-anomaly adaptive buffer selection via masking.

        For each anomaly k, keeps the whole series (so all normal regions stay,
        preserving the global score scale and false-positive context) but removes
        the core regions of every OTHER anomaly from both label and score. Then
        runs the normal cumulative VUS on these full-length-minus-other-anomalies
        arrays and selects b* independently per anomaly.

        Unlike the slicing strategy, no external thresholds are injected:
        compute() derives thresholds from the masked score it receives, which
        guarantees TP+FP >= 1 for every threshold (no 0/0 NaN).

        Args:
            label (np.ndarray): binary ground truth, shape (T,)
            score (np.ndarray): detector scores, shape (T,)
            plot  (bool)      : reserved; debug plotting is not yet supported here

        Returns:
            list of dicts, one per anomaly:
                'start'  (int)        : anomaly start index in original series
                'end'    (int)        : anomaly end index in original series
                'window' (None)       : no slice is used in the masking strategy
                'best'   (dict)       : select_best_buffer result (buffer_idx, buffer_val, ratio)
                'vus'    (np.ndarray) : per-buffer VUS curve for this anomaly
        """
        if not self.cumulative:
            raise ValueError("compute_per_anomaly_adaptive_masking requires cumulative=True")

        T = len(label)
        ((_, _), (starts, ends)), _ = self.get_anomalies_coordinates(label)
        n_anomalies = len(starts)

        if n_anomalies == 0:
            return []

        results = []
        for k in range(n_anomalies):
            # Keep the whole series, then drop the core region of every other anomaly
            # from both arrays so they count as neither TP nor FP.
            keep = np.ones(T, dtype=bool)
            for j in range(n_anomalies):
                if j == k:
                    continue
                keep[int(starts[j]):int(ends[j]) + 1] = False

            label_k = label[keep]
            score_k = score[keep]

            # No thresholds= argument: let compute() derive in-range thresholds.
            (result_k, _), _ = self.compute(label_k, score_k, adaptive=True)

            r = {
                'start':  int(starts[k]),   # original-series index
                'end':    int(ends[k]),     # original-series index
                'window': None,             # not a slice in the masking strategy
                'best':   result_k['best'],
                'vus':    result_k['vus'],
            }
            results.append(r)

            if plot:
                # TODO: _plot_anomaly_debug was written for the slicing strategy
                # (full-length arrays + a window). Masking has no window, so skip
                # the per-anomaly debug plot for now rather than crash.
                pass

        return results

    def _plot_anomaly_debug(self, label, score, r, k, n_anomalies, T, timeseries=None):
        s, e     = r['start'], r['end']
        l_k, r_k = r['window']
        b        = r['best']['buffer_val']
        b_idx    = r['best']['buffer_idx']
        vus_score = float(r['vus'][b_idx])
        ratio     = r.get('ratio')

        fig, (ax_ts, ax_sc, ax_ratio) = plt.subplots(
            3, 1, figsize=(14, 7),
            gridspec_kw={'height_ratios': [2, 1, 1.5]},
            sharex=False,
        )
        ax_sc.sharex(ax_ts)
        final_vus    = r.get('final_vus')
        baseline_vus = r.get('baseline_vus')
        final_vus_str    = f"  |  VUS(caps)={final_vus:.4f}"    if final_vus    is not None else ""
        baseline_vus_str = f"  |  VUS(no-adapt)={baseline_vus:.4f}" if baseline_vus is not None else ""
        dtype_str        = f"  |  {r['detection_type'].upper()}" if r.get('detection_type') else ""
        fig.suptitle(
            f"Anomaly {k + 1}/{n_anomalies}  |  "
            f"range [{s}, {e}]  len={e - s + 1}  |  "
            f"window [{l_k}, {r_k}]  |  "
            f"b*={b}  VUS(local)={vus_score:.4f}{final_vus_str}{baseline_vus_str}{dtype_str}",
            fontsize=9, fontweight='bold',
        )

        # parse all anomaly spans for context shading
        _diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
        all_starts = np.where(_diff > 0)[0]
        all_ends   = np.where(_diff < 0)[0] - 1

        # --- time series ---
        if timeseries is not None:
            ax_ts.plot(np.arange(T), timeseries, color='dimgray', linewidth=0.6, zorder=2)
        else:
            ax_ts.plot(np.arange(T), label.astype(float), color='lightcoral',
                       linewidth=0.8, alpha=0.6, label='label')
        ax_ts.axvspan(l_k, r_k, color='steelblue', alpha=0.06, linewidth=0, zorder=1)
        ax_ts.axvline(l_k, color='steelblue', linewidth=1, linestyle='--', alpha=0.5)
        ax_ts.axvline(r_k, color='steelblue', linewidth=1, linestyle='--', alpha=0.5)
        for as_, ae in zip(all_starts, all_ends):
            if as_ == s and ae == e:
                ax_ts.axvspan(s, e + 1, color='red', alpha=0.45, linewidth=0, zorder=3)
            else:
                ax_ts.axvspan(as_, ae + 1, color='red', alpha=0.2, linewidth=0, zorder=2)
        if b > 0:
            ax_ts.axvspan(max(0, s - b), s,         color='orange', alpha=0.25, linewidth=0, zorder=2)
            ax_ts.axvspan(e + 1, min(T, e + 1 + b), color='orange', alpha=0.25, linewidth=0, zorder=2)
        ax_ts.legend(
            handles=[
                plt.matplotlib.patches.Patch(facecolor='steelblue', alpha=0.2, label='window'),
                plt.matplotlib.patches.Patch(facecolor='red',        alpha=0.45, label='current anomaly'),
                plt.matplotlib.patches.Patch(facecolor='red',        alpha=0.2, label='other anomalies'),
                plt.matplotlib.patches.Patch(facecolor='orange',     alpha=0.3, label=f'buffer b*={b}'),
            ],
            fontsize=7, loc='upper right',
        )
        ax_ts.set_ylabel('value' if timeseries is not None else 'label', fontsize=8)
        ax_ts.set_xlim(0, T - 1)
        ax_ts.tick_params(labelsize=7, labelbottom=False)

        # --- score ---
        ax_sc.plot(np.arange(T), score, color='dimgray', linewidth=0.7, zorder=2)
        ax_sc.axvspan(l_k, r_k, color='steelblue', alpha=0.06, linewidth=0, zorder=1)
        for as_, ae in zip(all_starts, all_ends):
            if as_ == s and ae == e:
                ax_sc.axvspan(s, e + 1, color='red', alpha=0.3, linewidth=0, zorder=3)
            else:
                ax_sc.axvspan(as_, ae + 1, color='red', alpha=0.12, linewidth=0, zorder=2)
        if b > 0:
            ax_sc.axvspan(max(0, s - b), s,         color='orange', alpha=0.25, linewidth=0, zorder=2)
            ax_sc.axvspan(e + 1, min(T, e + 1 + b), color='orange', alpha=0.25, linewidth=0, zorder=2)
        ax_sc.set_ylabel('score', fontsize=8)
        ax_sc.tick_params(labelsize=7, labelbottom=False)

        # --- ratio curve ---
        if ratio is not None:
            sv = self.slope_values
            ax_ratio.plot(sv, ratio, color='steelblue', linewidth=1.5, label='ratio')
            ax_ratio.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_ratio.axvline(sv[b_idx], color='crimson', linewidth=2.0, label=f'b*={b}')
            ax_ratio.scatter([sv[b_idx]], [ratio[b_idx]], color='crimson', s=60, zorder=5)
            ax_ratio.legend(fontsize=8, loc='upper right')
        ax_ratio.set_ylabel('ratio', fontsize=8)
        ax_ratio.set_xlabel('buffer', fontsize=8)
        ax_ratio.set_xlim(0, int(self.slope_values[-1]))
        ax_ratio.tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()

    def _plot_anomaly_debug_per_side(self, label, score, r, k, n_anomalies, T, timeseries=None):
        """Per-anomaly debug plot for per_side selection.

        Time series + score on top, then the LEFT and RIGHT marginal-ratio curves
        side by side, each with its own b* marked. Mirrors _plot_anomaly_debug but
        renders the two one-sided selections of a per_side result dict (keys
        best_left/best_right, ratio_left/ratio_right, vus_left/vus_right).
        """
        s, e     = r['start'], r['end']
        l_k, r_k = r['window']
        b_left   = r['best_left']['buffer_val']
        b_right  = r['best_right']['buffer_val']
        bi_left  = r['best_left']['buffer_idx']
        bi_right = r['best_right']['buffer_idx']
        ratio_l  = r.get('ratio_left')
        ratio_r  = r.get('ratio_right')

        fig, axd = plt.subplot_mosaic(
            [['ts', 'ts'], ['sc', 'sc'], ['rl', 'rr']],
            figsize=(14, 8),
            gridspec_kw={'height_ratios': [2, 1, 1.5]},
        )
        ax_ts, ax_sc, ax_rl, ax_rr = axd['ts'], axd['sc'], axd['rl'], axd['rr']
        ax_sc.sharex(ax_ts)
        ax_rr.sharey(ax_rl)   # common ratio scale so the two sides are comparable

        final_vus    = r.get('final_vus')
        baseline_vus = r.get('baseline_vus')
        final_vus_str    = f"  |  VUS(caps)={final_vus:.4f}"        if final_vus    is not None else ""
        baseline_vus_str = f"  |  VUS(no-adapt)={baseline_vus:.4f}" if baseline_vus is not None else ""
        dtype_str        = f"  |  {r['detection_type'].upper()}" if r.get('detection_type') else ""
        fig.suptitle(
            f"Anomaly {k + 1}/{n_anomalies}  |  range [{s}, {e}]  len={e - s + 1}  |  "
            f"window [{l_k}, {r_k}]  |  b*_left={b_left}  b*_right={b_right}"
            f"{final_vus_str}{baseline_vus_str}{dtype_str}",
            fontsize=9, fontweight='bold',
        )

        # parse all anomaly spans for context shading
        _diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
        all_starts = np.where(_diff > 0)[0]
        all_ends   = np.where(_diff < 0)[0] - 1

        # --- time series (left buffer orange, right buffer green) ---
        if timeseries is not None:
            ax_ts.plot(np.arange(T), timeseries, color='dimgray', linewidth=0.6, zorder=2)
        else:
            ax_ts.plot(np.arange(T), label.astype(float), color='lightcoral',
                       linewidth=0.8, alpha=0.6, label='label')
        ax_ts.axvspan(l_k, r_k, color='steelblue', alpha=0.06, linewidth=0, zorder=1)
        ax_ts.axvline(l_k, color='steelblue', linewidth=1, linestyle='--', alpha=0.5)
        ax_ts.axvline(r_k, color='steelblue', linewidth=1, linestyle='--', alpha=0.5)
        for as_, ae in zip(all_starts, all_ends):
            if as_ == s and ae == e:
                ax_ts.axvspan(s, e + 1, color='red', alpha=0.45, linewidth=0, zorder=3)
            else:
                ax_ts.axvspan(as_, ae + 1, color='red', alpha=0.2, linewidth=0, zorder=2)
        if b_left > 0:
            ax_ts.axvspan(max(0, s - b_left), s, color='orange', alpha=0.25, linewidth=0, zorder=2)
        if b_right > 0:
            ax_ts.axvspan(e + 1, min(T, e + 1 + b_right), color='seagreen', alpha=0.25, linewidth=0, zorder=2)
        ax_ts.legend(
            handles=[
                plt.matplotlib.patches.Patch(facecolor='steelblue', alpha=0.2, label='window'),
                plt.matplotlib.patches.Patch(facecolor='red',        alpha=0.45, label='current anomaly'),
                plt.matplotlib.patches.Patch(facecolor='red',        alpha=0.2, label='other anomalies'),
                plt.matplotlib.patches.Patch(facecolor='orange',     alpha=0.3, label=f'left b*={b_left}'),
                plt.matplotlib.patches.Patch(facecolor='seagreen',   alpha=0.3, label=f'right b*={b_right}'),
            ],
            fontsize=7, loc='upper right',
        )
        ax_ts.set_ylabel('value' if timeseries is not None else 'label', fontsize=8)
        ax_ts.set_xlim(0, T - 1)
        ax_ts.tick_params(labelsize=7, labelbottom=False)

        # --- score ---
        ax_sc.plot(np.arange(T), score, color='dimgray', linewidth=0.7, zorder=2)
        ax_sc.axvspan(l_k, r_k, color='steelblue', alpha=0.06, linewidth=0, zorder=1)
        for as_, ae in zip(all_starts, all_ends):
            if as_ == s and ae == e:
                ax_sc.axvspan(s, e + 1, color='red', alpha=0.3, linewidth=0, zorder=3)
            else:
                ax_sc.axvspan(as_, ae + 1, color='red', alpha=0.12, linewidth=0, zorder=2)
        if b_left > 0:
            ax_sc.axvspan(max(0, s - b_left), s, color='orange', alpha=0.25, linewidth=0, zorder=2)
        if b_right > 0:
            ax_sc.axvspan(e + 1, min(T, e + 1 + b_right), color='seagreen', alpha=0.25, linewidth=0, zorder=2)
        ax_sc.set_ylabel('score', fontsize=8)
        ax_sc.tick_params(labelsize=7)

        # --- ratio curves side by side ---
        sv = self.slope_values
        for ax, ratio, b_idx, b_val, title, color in [
            (ax_rl, ratio_l, bi_left,  b_left,  'left (pre-anomaly buffer)',   'orange'),
            (ax_rr, ratio_r, bi_right, b_right, 'right (post-anomaly buffer)', 'seagreen'),
        ]:
            if ratio is not None:
                ax.plot(sv, ratio, color='steelblue', linewidth=1.5, label='ratio')
                ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
                ax.axvline(sv[b_idx], color=color, linewidth=2.0, label=f'b*={b_val}')
                ax.scatter([sv[b_idx]], [ratio[b_idx]], color=color, s=60, zorder=5)
                ax.legend(fontsize=8, loc='upper right')
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('buffer', fontsize=8)
            ax.set_xlim(0, int(sv[-1]))
            ax.tick_params(labelsize=7)
        ax_rl.set_ylabel('ratio', fontsize=8)

        plt.tight_layout()
        plt.show()

    @time_it
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    @time_it
    def get_unique_thresholds(self, score, max_thresholds=None):
        """
        Returns the sorted unique thresholds in descending order.

        Args:
            max_thresholds: if None (default), all unique values are returned.
                            If an int, the array is uniformly subsampled to at
                            most that many thresholds.
        """
        thresholds = np.sort(np.unique(score))[::-1]

        if max_thresholds is not None and len(thresholds) > max_thresholds:
            k = len(thresholds) // max_thresholds
            thresholds = thresholds[::k]
        return thresholds

    @time_it
    def get_pr_relevant_thresholds(
        self,
        label,
        score,
        min_area=1e-8,          # None = no area pruning
    ):
        """
        Returns PR-curve-relevant thresholds (descending), pruned by collinearity
        and optionally by local area contribution.

        WARNING — wrong for VUS, correct only for standard AUC-PR (slope_size=0,
        existence='None').  Two reasons it fails for VUS:

        1. Slope buffers: points inside the buffer zone around an anomaly have a
           fractional slope value > 0, so they contribute partial TP to VUS-PR.
           A threshold step where the newly-predicted points are all inside a
           buffer appears collinear in binary PR space but bends the VUS curve.

        2. Global mask: with global_mask=True, create_safe_mask keeps only the
           anomaly core plus slope_size+2 points on each side.  Virtually all
           safe-region points are inside a slope buffer, so almost every threshold
           step touches buffer-zone points — the collinearity pruning keeps nearly
           every threshold, giving no benefit over get_unique_thresholds(max_thresholds=None).

        Use get_unique_thresholds(max_thresholds=None) (the default) for correct VUS.
        """

        # ---- Sort by score descending ----
        order = np.argsort(score)[::-1]
        score_sorted = score[order]
        label_sorted = label[order]

        # ---- Keep only distinct score values ----
        # LAST occurrence so the cumulative TP/FP at each unique score
        # includes every tied point. See get_pr_relevant_thresholds_vus
        # for the matching note.
        distinct = np.r_[score_sorted[:-1] != score_sorted[1:], True]
        score_unique = score_sorted[distinct]

        # ---- Cumulative TP ----
        tp = np.cumsum(label_sorted)[distinct]

        # ---- Keep only thresholds where cumulative TP changes ----
        # Exact for stepwise (right-Riemann) AUC-PR: thresholds that add only
        # FPs leave recall unchanged and contribute zero area, so they can
        # be dropped without affecting the integral. This replaces the older
        # collinearity + area pruning, which was only exact for trapezoidal
        # interpolation and produced ~1e-3 deviations on rare PR curves
        # under the default stepwise interpolation.
        keep = np.diff(tp, prepend=0) > 0
        return score_unique[keep]

    @time_it
    def get_pr_relevant_thresholds_vus(
        self,
        label_slopes,
        score,
        min_area=1e-8,          # kept for backward-compatible signature; unused
        max_thresholds=1000,    # cap: TP-space uniform sampling when exceeded
    ):
        """
        VUS-aware threshold pruning: correct for VUS-PR under stepwise
        interpolation.

        Uses label_slopes.max(axis=0) — the fuzzy label at the outermost slope
        level — to decide which thresholds matter:

          - Normal points (max_fuzzy == 0): adding them across a threshold step
            only changes FP, never TP at any slope, so recall is unchanged for
            every slope and the stepwise area contribution is zero. Dropped.
          - Buffer/anomaly points (max_fuzzy > 0): contribute fractional TP for
            at least one slope, so recall changes for that slope. Kept.
        The existence reward is automatically handled: an anomaly is "found"
        when the threshold crosses the max score inside its buffer window, and
        that max point has max_fuzzy > 0 by construction, so the corresponding
        threshold is kept.

        NOTE: must be called after add_slopes — pass the already-computed
        label_slopes to avoid recomputing it.

        WARNING: speedup degrades with anomaly rate.  With global_mask=True and
        slope_size=L, the buffer zones cover roughly anomaly_rate × 2L / T of
        the series.  For high anomaly rates (>10 %) most threshold steps touch
        buffer points and the pruning gives little benefit over
        get_unique_thresholds(max_thresholds=None).

        NOTE on tied scores: the `distinct` mask uses the LAST occurrence of
        each unique score so the cumulative TP/FP at that threshold accounts
        for every tied point.  Using FIRST-occurrence (the older form) drops
        any tied TPs that follow the first instance of a repeated score and
        produces wildly wrong (near-zero) VUS values whenever the detector
        emits flat score plateaus over anomaly regions.

        NOTE on interpolation: the kept-set is the minimal exact set for
        stepwise (right-Riemann) integration. For trapezoidal/linear AUC the
        previous collinearity+area pruning would have been slightly tighter,
        but stepwise is the default in `auc()` and is what `sklearn.metrics.
        average_precision_score` matches.

        NOTE on capping: when the exact set exceeds max_thresholds, TP-space
        uniform sampling is applied.  N targets are placed at equal intervals
        in [0, total_TP] and mapped to the nearest threshold via searchsorted.
        The worst-case stepwise AUC error is bounded by 1/max_thresholds
        (one recall step ≤ total_TP / (N * total_TP) = 1/N).  At the default
        of 10 000 this is ≤ 1e-4, below any meaningful numerical precision.
        Pass max_thresholds=None to disable the cap and recover exact behaviour.
        """
        fuzzy_label = label_slopes.max(axis=0)

        # ---- Sort by score descending ----
        order = np.argsort(score)[::-1]
        score_sorted  = score[order]
        fuzzy_sorted  = fuzzy_label[order]

        # ---- Keep only distinct score values ----
        # Use LAST occurrence of each tied score so the cumulative TP/FP
        # at that threshold includes every point sharing the same score.
        # FIRST-occurrence (the previous behaviour) silently drops tied TPs
        # and produces a wildly wrong PR curve when scores have plateaus.
        distinct     = np.r_[score_sorted[:-1] != score_sorted[1:], True]
        score_unique = score_sorted[distinct]

        # ---- Cumulative fuzzy-TP ----
        tp = np.cumsum(fuzzy_sorted)[distinct]

        # ---- Keep only thresholds where cumulative fuzzy-TP changes ----
        # See get_pr_relevant_thresholds for the rationale.
        keep = np.diff(tp, prepend=0.0) > 0
        thresholds = score_unique[keep]

        # ---- TP-space uniform sampling cap ----
        if max_thresholds is not None and len(thresholds) > max_thresholds:
            tp_kept = tp[keep]
            total_tp = tp_kept[-1]
            targets = np.linspace(0.0, total_tp, max_thresholds + 1)[1:]
            idx = np.searchsorted(tp_kept, targets, side='left')
            idx = np.clip(idx, 0, len(tp_kept) - 1)
            idx = np.unique(idx)
            thresholds = thresholds[idx]

        return thresholds

    @time_it
    def create_safe_mask(self, label, start_points, end_points, extra_safe=False):
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """
        length = label.shape[0]
        mask = np.zeros(length, dtype=np.int8)

        safe_extension = self.slope_size + 1 + int(extra_safe)
        
        start_safe_points = np.maximum(start_points - safe_extension, 0)
        end_safe_points = np.minimum(end_points + safe_extension + 1, length - 1)

        np.add.at(mask, start_safe_points, 1)
        np.add.at(mask, end_safe_points, -1)

        mask = np.cumsum(mask)
        mask = mask > 0
        # TODO: verify correctness of the no-anomalies case end-to-end (empty label, label with all-normal, etc.)
        mask[-1] = (len(end_safe_points) > 0 and end_safe_points[-1] == (length - 1)) if extra_safe else label[-1]
        
        return mask
    
    def _strided_indexing_roll(self, a, r):
        # Concatenate with sliced to cover all rolls
        a_ext = np.concatenate((a, a[:, : - 1]), axis=1)

        # Get sliding windows; use advanced-indexing to select appropriate ones
        n = a.shape[1]
        return viewW(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]
    
    def _slopes_mask(self, n, m, as_type='int'):
        k = m // n
        row_indices = np.arange(1, n + 1)[:, None]
        col_indices = np.arange(m)

        mask = col_indices < (row_indices * k)
        return ~mask
    
    def precompute_slopes(self):
        '''
        Create all the required slopes at once.
        step should be an exact divider of l.
        The very first slopes, which is the slope of length 0, is not provided
        Every row is a slope, row 0 is the smallest and the last row the biggest
        These are the positive slopes
        '''
        steps = ((1 - self.zita) / self.slope_values[1:])
        stops = 1 + steps[::-1] * self.slope_values[:-1]
        
        slopes = np.linspace(self.zita, stops, self.slope_size + 1).T[::-1]
        slopes[self._slopes_mask(*slopes.shape)] = 0
        
        rolls = self.slope_size - self.slope_values[1:]
        shifted_slopes = self._strided_indexing_roll(slopes, rolls)

        return shifted_slopes
    

    @time_it
    def distance_from_anomaly(self, label, start_points, end_points, clip=False):
        '''
        For every point, return TWO direction-aware distance arrays:

          pos_left[t]  = distance to the nearest anomaly START at or ahead of t
                         (start - t for start >= t), inf if none. Drives the LEFT
                         buffer, the pre-anomaly ramp on [start - slope_size, start].
          pos_right[t] = distance to the nearest anomaly END at or behind t
                         (t - end for end <= t), inf if none. Drives the RIGHT
                         buffer, the post-anomaly ramp on [end, end + slope_size].

        For side='both', add_slopes_function takes the per-point minimum of the
        two, which reproduces the old min|t - boundary| behaviour exactly for
        points outside anomalies (inside points are forced to 1 regardless). The
        old code conflated the two sides via abs(); splitting them lets the
        function path honour reward_side like the precomputed path does.
        '''
        length = len(label)

        pos_left = np.full(length, np.inf)
        pos_right = np.full(length, np.inf)
        if start_points.shape[0] == 0 and end_points.shape[0] == 0:
            return pos_left, pos_right

        indices = np.arange(length)[:, None]

        if start_points.shape[0]:
            sd = start_points[None, :] - indices       # start - t ; >=0 when t at/left of start
            sd = np.where(sd >= 0, sd, np.inf)
            pos_left = sd.min(axis=1)
        if end_points.shape[0]:
            ed = indices - end_points[None, :]          # t - end ; >=0 when t at/right of end
            ed = np.where(ed >= 0, ed, np.inf)
            pos_right = ed.min(axis=1)

        if clip:
            # Inside-anomaly points contribute 0 distance on both sides. Clip the
            # ramp reach to slope_size, but preserve inf (no boundary on that side)
            # so a one-sided point gets no spurious ramp.
            inside = label.astype(bool)
            pos_left[inside] = 0
            pos_right[inside] = 0
            fl = np.isfinite(pos_left)
            fr = np.isfinite(pos_right)
            pos_left[fl] = np.clip(pos_left[fl], 0, self.slope_size)
            pos_right[fr] = np.clip(pos_right[fr], 0, self.slope_size)

        return pos_left, pos_right

    def _per_point_cap(self, label, start_points, end_points, b_left, b_right):
        """Per-point left/right buffer caps from per-anomaly caps.

        Each point inherits the cap of its nearest anomaly (the one whose
        [start, end] interval is closest, 0 distance if inside). The left and
        right caps are looked up separately so each side of an anomaly can carry
        a different b*. Points inside an anomaly are forced to 1 in
        add_slopes_function regardless, so their cap is moot.

        Args:
            label (np.ndarray): binary label, shape (T,)
            start_points, end_points (np.ndarray): anomaly boundaries with edges,
                shape (n_anomalies,)
            b_left, b_right (np.ndarray): per-anomaly caps, shape (n_anomalies,)

        Returns:
            (cap_left, cap_right): two float arrays of shape (T,), the left and
            right cap of each point's nearest anomaly.
        """
        b_left = np.asarray(b_left, dtype=float)
        b_right = np.asarray(b_right, dtype=float)
        n = start_points.shape[0]
        if b_left.shape[0] != n or b_right.shape[0] != n:
            raise ValueError(
                f"per-anomaly caps have {b_left.shape[0]}/{b_right.shape[0]} entries "
                f"but there are {n} anomalies"
            )

        T = len(label)
        idx = np.arange(T)[:, None]                       # (T, 1)
        # Distance from each point to each anomaly interval (0 if inside).
        dist = np.maximum(start_points[None, :] - idx,    # left of the interval
                          idx - end_points[None, :])      # right of the interval
        np.clip(dist, a_min=0, a_max=None, out=dist)      # inside -> 0
        nearest = np.argmin(dist, axis=1)
        return b_left[nearest], b_right[nearest]

    def _resolve_caps(self, sym, left, right):
        """Normalise the three cap kwargs into per-anomaly (b_left, b_right).

        Accepts either the symmetric ``max_buffer_per_anomaly`` (applied to both
        sides) or the per-side pair, never both. Returns (None, None) when no cap
        is requested.
        """
        if left is not None or right is not None:
            if sym is not None:
                raise ValueError(
                    "pass either max_buffer_per_anomaly or the per-side "
                    "max_buffer_per_anomaly_left/right, not both"
                )
            if left is None or right is None:
                raise ValueError(
                    "per-side caps require BOTH max_buffer_per_anomaly_left and "
                    "max_buffer_per_anomaly_right"
                )
            return np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if sym is not None:
            sym = np.asarray(sym, dtype=float)
            return sym, sym
        return None, None

    @time_it
    def get_anomalies_coordinates(self, label: np.array):
        """
        Return the starting and ending points of all anomalies in label,
        both with and without edge inclusion.

        Args:
            label (np.array): vector of 1s and 0s

        Returns:
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges))
        """
        diff = np.diff(label)
        start_no_edges = np.where(diff == 1)[0] + 1
        end_no_edges = np.where(diff == -1)[0]
        
        start_with_edges = np.append([0], start_no_edges) if label[0] else start_no_edges
        end_with_edges = np.append(end_no_edges, len(label) - 1) if label[-1] else end_no_edges

        if start_with_edges.shape != end_with_edges.shape:
            raise ValueError(f'The number of start and end points of anomalies does not match, {start_with_edges} != {end_with_edges}')
            
        return (start_no_edges, end_no_edges), (start_with_edges, end_with_edges)

    def _side_ramp(self, pos, cap):
        """Buffer ramp contribution for ONE side, shape (n_slopes-1, T).

        ``pos`` is the direction-aware distance for this side (pos_left or
        pos_right); points with no boundary on this side carry inf and are masked
        out (stay 0). ``cap`` (shape (T,) or None) caps each point's effective
        slope size at min(slope_values[i], cap[t]); None gives the uniform buffer.
        """
        T = pos.shape[0]
        out = np.zeros((self.n_slopes - 1, T))
        pos_b = pos[None, :]               # (1, T)

        if cap is None:
            if not hasattr(self, '_inv_slope_factor'):
                self._inv_slope_factor = (1.0 - self.zita) / self.slope_values[1:, None]

            sv = self.slope_values[1:, None]   # (n_slopes-1, 1)
            mask = sv >= pos_b                 # one mask, reused

            np.subtract(1.0, self._inv_slope_factor * pos_b,
                        out=out, where=mask)
        else:
            # Per-point effective slope size: min(buffer index, nearest anomaly's cap).
            sv_eff = np.minimum(self.slope_values[1:, None], cap[None, :])  # (n_slopes-1, T)
            mask = (sv_eff >= pos_b) & (sv_eff > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ramp = 1.0 - (1.0 - self.zita) * pos_b / sv_eff
            np.copyto(out, ramp, where=mask)

        return out

    def add_slopes_function(self, label, pos_left, pos_right, cap_left=None, cap_right=None):
        """
        Compute the slopes of a label using a predefined function.

        This function is may be a bit slower than the precomputed version for small buffer lengths or small time series.
        It is significantly faster for big buffer lengths and long time series though.

        The pos part requires about 10% of the total execution time.

        Honours ``reward_side``: the left buffer (driven by ``pos_left``) and the
        right buffer (``pos_right``) are computed independently and combined per
        the configured side ('both' -> elementwise max, matching the precomputed
        path; 'left'/'right' -> that side only). ``cap_left``/``cap_right`` (shape
        (T,) or None) cap each side's per-point effective slope size; None keeps
        the uniform-buffer behaviour for that side.
        """
        T = len(label)
        f_pos = np.zeros((self.n_slopes, T))
        f_pos[0] = label

        if self.n_slopes > 1:
            if self.reward_side == 'left':
                f_pos[1:] = self._side_ramp(pos_left, cap_left)
            elif self.reward_side == 'right':
                f_pos[1:] = self._side_ramp(pos_right, cap_right)
            else:  # 'both': max of the two one-sided ramps
                f_pos[1:] = self._side_ramp(pos_left, cap_left)
                np.maximum(f_pos[1:], self._side_ramp(pos_right, cap_right), out=f_pos[1:])

        f_pos[1:, label.astype(bool)] = 1

        return f_pos

    def add_slopes_precomputed(self, label, start_points, end_points):
        """
        Start points and end points of anomalies are without edges
        """
        result = np.repeat([label.astype(np.float64)], self.n_slopes, axis=0)

        if self.n_slopes == 1:
            return result

        if self.reward_side in ['both', 'left']:
            for curr_point in start_points:
                slope_start = max(curr_point - self.slope_size, 0)
                slope_end = curr_point + 1  # to include it in the transformation, read desc
                adjusted_l = slope_end - slope_start
                result[1:, slope_start: slope_end] = np.maximum(result[1:, slope_start: slope_end], self.pos_slopes[:, -adjusted_l:])

        if self.reward_side in ['both', 'right']:
            for curr_point in end_points:
                slope_start = curr_point
                slope_end = min(curr_point + self.slope_size, len(label) - 1) + 1       # to include it in the transformation, read desc
                adjusted_l = slope_end - slope_start
                result[1:, slope_start: slope_end] = np.maximum(result[1:, slope_start: slope_end], self.neg_slopes[:, :adjusted_l])

        return result
    
    @time_it
    def add_slopes(self, label, start_points, end_points, pos_left, pos_right,
                   cap_left=None, cap_right=None, plot=False):
        if cap_left is not None or cap_right is not None:
            # The precomputed path applies a uniform slope_size to every anomaly
            # and cannot cap per anomaly, so route through the function path.
            slopes = self.add_slopes_function(label, pos_left, pos_right,
                                              cap_left=cap_left, cap_right=cap_right)
        elif self.add_slopes_mode == 'precomputed':
            slopes = self.add_slopes_precomputed(label, start_points, end_points)
        else:
            slopes = self.add_slopes_function(label, pos_left, pos_right)

        # t0 = time.time(); slopes     = self.add_slopes_precomputed(label, start_points, end_points);      t_orig = time.time() - t0
        # t0 = time.time(); slopes_new = self.add_slopes_function(label, pos); t_fast = time.time() - t0
        # match = np.allclose(slopes, slopes_new)
        # print(f"[slopes bench] orig={t_orig*1e3:.2f}ms  fast={t_fast*1e3:.2f}ms  speedup={t_orig/t_fast:.1f}x  match={match}")
        # if not match:
        #     raise ValueError("STOOOOOOP")
        # sns.lineplot(slopes.T, legend=None)
        # plt.show()

        if plot:
            # Example used in paper is the first time series of Daphnet
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(2, 1, figsize=(5, 4), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})

            plot_start = 13150 + 25
            plot_end = 13490 - 200

            sns.lineplot(label[plot_start:plot_end], ax=ax[0], linewidth=2.5, color='#000000')
            ax[0].set_title('Label')
            ax[0].grid(alpha=0.2)
            ax[0].annotate(f'(a)', xy=(7, 0.7), color='k', fontsize=15, weight='bold', alpha=.8)

            ax[1].annotate(f'Z', xy=(111, self.zita),
                        xytext=(115, self.zita + 0.15),
                        arrowprops=dict(arrowstyle="->", color='purple', lw=1.5, alpha=.5),
                        color='purple', fontsize=12, weight='bold', alpha=.5)
            ax[1].axhline(y=self.zita, xmin=0, xmax=1000, color='purple', linestyle='-.', zorder=0, alpha=.5)
            sns.lineplot(ax=ax[1], data=slopes.T[plot_start:plot_end], linewidth=1.5, palette='flare_r', legend=False)
            ax[1].set_title('Label with buffers')
            ax[1].set_xlabel('Time')
            ax[1].grid(alpha=0.2)

            plt.tight_layout()
            plt.savefig("experiments/figures/ts_label_buffer_example.svg", bbox_inches='tight')
            plt.savefig("experiments/figures/ts_label_buffer_example.pdf", bbox_inches='tight')
            plt.show()
            exit()

        return slopes
    
    @time_it
    def compute_existence(self, labels, sm, score, thresholds, start_points, end_points, safe_mask, plot=False):
        if self.existence_mode == 'optimized':
            existence = self.existence_optimized(labels, score, thresholds, start_points, end_points)
        elif self.existence_mode == 'matrix':
            existence = self.existence_matrix(labels, sm, safe_mask)
        elif self.existence_mode == 'trivial':
            existence = self.existence_trivial(labels, sm, thresholds, start_points, end_points)
        elif self.existence_mode == 'None':
            existence = None
        
        if plot:
            # Example used in paper is the first time series of YAHOO
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))

            sns.heatmap(
                existence,
                cmap="flare_r",
                ax=ax,
                annot=False,
                linewidths=0,
                linecolor=None,
                rasterized=True
            )
            
            ax.annotate(f'(b)', xy=(68, 18), color='k', fontsize=15, weight='bold', alpha=.8)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Buffer")
            ax.invert_yaxis()
            ax.set_facecolor('white')
            ax.set_xticks(np.arange(len(thresholds)), thresholds)

            plt.locator_params(axis='y', nbins=11)
            plt.locator_params(axis='x', nbins=9)
            plt.xticks(rotation=0)

            plt.tight_layout()
            plt.savefig("experiments/figures/existence_heatmap.svg", bbox_inches='tight')
            plt.savefig("experiments/figures/existence_heatmap.pdf", bbox_inches='tight')
            plt.show()
            exit()

        return existence
    

    def existence_trivial(self, labels, score_mask, thresholds, start_points, end_points):
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        n_anomalies = np.full(n_slopes, start_points.shape[0], dtype=int)

        all_start_points = np.maximum((np.tile(start_points, (n_slopes, 1)) - self.slope_values[:, None]), 0)
        all_end_points = np.minimum((np.tile(end_points, (n_slopes, 1)) + self.slope_values[:, None]), T - 1)

        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = (2 * self.slope_size) - distance
            if overlap >= 0:  # >= 0 (not > 0) so the exact distance == 2*slope_size touch case merges, matching existence_optimized and the legacy VUS
                index = math.ceil((self.slope_size - (overlap // 2)) / self.step)
                n_anomalies[index:] -= 1
                all_start_points[index:, (k + 1)] = -1
                all_end_points[index:, k] = -1

        for i in range(n_slopes):
            for j in range(n_thresholds):
                for k in range(n_anomalies[i]):
                    start = all_start_points[i, all_start_points[i] >= 0][k]
                    end = all_end_points[i, all_end_points[i] >= 0][k]  
                    if np.any(score_mask[j, start: end + 1]):
                        existence[i, j] += 1

        return existence / n_anomalies[:, None]

    def existence_optimized(self, labels, score, thresholds, start_points, end_points):
        """
        Optimized existence computation (~400x faster than trivial implementation).
        Suitable for CPU applications.

        Step-by-step explanation of the algorithm:
        1) Find the starting and ending points of every anomaly, for every buffer. 2 arrays of shape (L x n)
        2) Find all anomalies were their buffers overlap, and at which point (or in other words, which buffers overlap). This is important for 2 reasons. First, the number of anomalies in a time series changes after buffers collide. Second, if two anomalies merge and the first one is found, every anomaly merged to this one is also conidered found.
        3) Iterate over the anomalies and fill in the (L x t) existence matrix, where L is the number of buffers and t is the number of thresholds. Every cell of this matrix represents the numbers of anomalies found for the according buffer x threshold pair. Iterating over the anomalies we will slowly add the found anomalies in this buffer and finally divide by the number of anomalies to get the fraction.
        3.1) If the anomaly is found (the maximum score value) in the middle of the anomaly, and not on the buffers, then the existence value of the main label is the same as the one of the buffers and it only depends on the threshold where the maximum value of the score lies. All values above the threshold are 0 and all values equal and below the threshold are 1.
        3.2) If the anomaly is found in the buffer region, then identify the exact buffer where the maximum score value lies. All the buffers smaller than this buffer have also found the anomaly so we can complete them. We can repeat the same process for the region after the buffer.
        4) Divide every cell of the existence matrix with the total number of anomalies of the according buffer.
        """
        n_slopes, T = labels.shape
        n_thresholds = thresholds.shape[0]
        existence = np.zeros((n_slopes, n_thresholds))
        # thresholds_dict = {value: index for index, value in enumerate(thresholds)}
        n_anomalies = np.full(n_slopes, start_points.shape[0], dtype=int)
        thresholds_inv = thresholds[::-1]
        
        all_start_points = np.maximum((np.tile(start_points, (n_slopes, 1)) - self.slope_values[:, None]), 0)
        all_end_points = np.minimum((np.tile(end_points, (n_slopes, 1)) + self.slope_values[:, None]), T - 1)

        anomalies_overlaps = np.zeros(start_points.shape, dtype=int)
        for k, (start, end) in enumerate(zip(start_points[1:], end_points[:-1])):
            distance = start - end
            overlap = (2 * self.slope_size) - distance
            if overlap >= 0:
                index = math.ceil((self.slope_size - (overlap // 2)) / self.step)
                n_anomalies[index:] -= 1
                anomalies_overlaps[k] = index
        
        prev_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
        for k in range(n_anomalies[0]):
            overlap_index = anomalies_overlaps[k - 1]       # This works because the last one never has an overlap ;)
            
            og_start = start_points[k]
            og_end = end_points[k]
            
            curr_existence = np.zeros((n_slopes, n_thresholds), dtype=int)
            tmp_existence = np.zeros((n_slopes, n_thresholds), dtype=int)

            i = n_slopes - 1
            while i >= 0:
                start = all_start_points[i, k]
                end = all_end_points[i, k]
                
                argmax_t = [score[start:og_start + 1][::-1].argmax(), score[og_start:og_end + 1].argmax(), score[og_end:end + 1].argmax()]
                max_t = [score[og_start - argmax_t[0]], score[og_start + argmax_t[1]], score[og_end + argmax_t[2]]]
            
                if max_t[1] >= max_t[0] and max_t[1] >= max_t[2]:   # Case middle
                    # threshold_index = thresholds_dict[max_t[1]]
                    threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[1], side="right")
                    curr_existence[:i+1, threshold_index:] += 1
                    break
                elif max_t[0] > max_t[2]:    # Case left
                    # threshold_index = thresholds_dict[max_t[0]]
                    threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[0], side="right")
                    curr_existence[math.ceil(argmax_t[0] / self.step):i+1, threshold_index:] += 1
                    i = math.ceil(argmax_t[0] / self.step) - 1
                elif max_t[2] >= max_t[0]:   # Case right
                    # threshold_index = thresholds_dict[max_t[2]]
                    threshold_index = n_thresholds - np.searchsorted(thresholds_inv, max_t[2], side="right")
                    curr_existence[math.ceil(argmax_t[2] / self.step):i+1, threshold_index:] += 1
                    i = math.ceil(argmax_t[2] / self.step) - 1

            if overlap_index > 0:
                tmp_existence[overlap_index:] = np.logical_or(curr_existence[overlap_index:], prev_existence[overlap_index:])
                curr_existence[overlap_index:] = np.logical_and(curr_existence[overlap_index:], np.logical_not(prev_existence[overlap_index:]))
            existence += curr_existence
            prev_existence = np.logical_or(tmp_existence, curr_existence)

        return existence / n_anomalies[:, None]

    def existence_matrix(self, labels, score_mask, safe_mask):
        """
        This implementation enables parallel computation for the existence.
        However, it is only ~5 times faster than the trivial implementation
        and ~80 times slower than the optimized one on CPU.

        Grouping correction (see the `onset` block below): the raw contiguity-based
        stairs merges two anomalies as soon as their buffers become *adjacent*, whereas
        the reference (existence_optimized / the legacy VUS) merges them only when the
        buffers *overlap*. Those two conditions differ by exactly one slope row, and only
        when the inter-anomaly distance D is odd (this was the historical ~0.005 average
        divergence). Re-inserting the missing group boundary makes this match
        existence_optimized and the legacy VUS exactly when step divides (D-1)/2 (always
        true for step=1). VUSTorch::compute_existence uses the same algorithm and needs
        the same correction to gain parity; see vus_torch.py::compute_existence.

        TODO: this is the old (s, t, T) score_hat + cummax formulation and is the
        slowest path. It should be reworked into the segment-max formulation used by
        VUSTorch::compute_existence_segmax: a region is found at a threshold iff its
        max in-region score clears it, so reduce the score to a per-region max once
        (threshold-independent) and compare against the thresholds, dropping the 3-D
        tensor and the cummax scan entirely. Proven exactly equivalent and ~74x
        faster on GPU; port here when there's time (numpy default is
        existence_optimized, so this is low priority).
        """
        # Compute mask for relevant points
        if not self.global_mask:
            labels = labels[:, safe_mask]
            score_mask = score_mask[:, safe_mask]

        # Normalize labels to 0 or 1
        norm_labels = (labels > 0).astype(np.int8)

        # Compute step function (stairs): mark the start of each contiguous slope run
        onset = np.diff(norm_labels, prepend=np.zeros((norm_labels.shape[0], 1)), axis=1)
        onset = np.clip(onset, 0, 1)

        # Grouping correction: split anomaly pairs whose buffers are merely adjacent (not
        # overlapping) so the grouping matches existence_optimized / the legacy VUS. The
        # split lands on the single slope row where adjacency first occurs (buffer w =
        # (D-1)/2), at the boundary cell the reference uses (ends[k] + w + 1 == starts[k+1] - w).
        base = norm_labels[0]  # row 0 = unbuffered label = anomaly cores
        edges = np.diff(np.concatenate([[0], base, [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        for k in range(len(starts) - 1):
            distance = starts[k + 1] - ends[k]
            half = (distance - 1) // 2
            if distance % 2 == 1 and 0 < half <= self.slope_size and half % self.step == 0:
                onset[half // self.step, ends[k] + half + 1] = 1

        stairs = np.cumsum(onset, axis=1)
        labels_stairs = norm_labels * stairs

        # Multiply every score with every slope
        score_hat = labels_stairs[:, None, :] * score_mask[None, :, :]

        # Compute cumulative max along the time dimension
        cm = np.maximum.accumulate(score_hat, axis=2)

        # Compute differences and normalize
        cm_diff = np.diff(cm, axis=2)
        cm_diff_norm = np.maximum(cm_diff - 1, 0)

        # Compute total anomalies and missed anomalies
        total_anomalies = stairs[:, -1][:, None]
        final_anomalies_missed = total_anomalies - cm[:, :, -1]
        n_anomalies_not_found = np.sum(cm_diff_norm, axis=2) + final_anomalies_missed
        n_anomalies_found = total_anomalies - n_anomalies_not_found

        return (n_anomalies_found / total_anomalies)
    
    @time_it
    def compute_confusion_matrix(self, labels, sm):
        """
        Scikit-learn order: tn, fp, fn, tp
        """

        if self.conf_matrix_mode == "trivial":
            conf_matrix = self.conf_matrix_trivial(labels, sm)
        elif self.conf_matrix_mode == "dynamic":
            conf_matrix = self.conf_matrix_dyn(labels, sm)
        elif self.conf_matrix_mode == "dynamic_plus":
            conf_matrix = self.conf_matrix_dyn_plus(labels, sm)
        else:
            raise ValueError('Unknown confusion matrix computation mode')
        fn, tp, positives = conf_matrix

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives

        return fp, fn, tp, positives, negatives, fpr
    
    def conf_matrix_trivial(self, labels, sm):
        label = labels[0]
        sm_inv = ~sm

        true_positives = np.matmul(labels, sm.T)
        fn_hard = np.matmul(label, sm_inv.T)[:, np.newaxis].T  # (1, n_thresholds)

        if self.fn_alpha > 0.0:
            fn_fuzzy = np.matmul(labels, sm_inv.T)  # (n_slopes, n_thresholds)
            false_negatives = (1.0 - self.fn_alpha) * fn_hard + self.fn_alpha * fn_fuzzy
        else:
            false_negatives = fn_hard

        positives = ((true_positives + false_negatives) + label.sum()) / 2

        return false_negatives, true_positives, positives
    
    def sparse_conf_matrix(self, labels, sm):
        """
        Compute confusion matrix components using sparse matrix operations.
        This is efficient when labels and/or sm contain many zeros.
        
        Uses scipy.sparse for efficient matrix multiplications on CPU.
        Produces the same results as conf_matrix_trivial.
        
        :param labels: 2D numpy array of shape (n_slopes, T) with float values [0, 1]
        :param sm: 2D boolean numpy array of shape (n_thresholds, T) - score mask
        :returns: (false_negatives, true_positives, positives)
        """
        label = labels[0]
        
        # Convert to sparse CSR format (efficient for row slicing and matrix multiplication)
        labels_sparse = sparse.csr_matrix(labels)
        sm_sparse = sparse.csr_matrix(sm.astype(np.float64))
        
        # true_positives = labels @ sm.T -> shape (n_slopes, n_thresholds)
        true_positives = labels_sparse.dot(sm_sparse.T).toarray()
        
        # For false_negatives, we need label @ (~sm).T -> shape (1, n_thresholds)
        # label is 1D (T,), when converted to sparse becomes (1, T)
        sm_inv_sparse = sparse.csr_matrix((~sm).astype(np.float64))
        label_sparse = sparse.csr_matrix(label.astype(np.float64))
        # Result: (1, T) @ (T, n_thresholds) = (1, n_thresholds)
        false_negatives = label_sparse.dot(sm_inv_sparse.T).toarray()
        
        positives = ((true_positives + false_negatives) + label.sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn(self, labels, sm):
        if not self.global_mask:
            mask = np.where(labels[-1] > 0)[0]
            labels = labels[:, mask]
            sm = sm[:, mask]

        true_positives = np.matmul(labels, sm.T)
        false_negatives = np.matmul(labels[0], ~sm.T)[None, :]
        positives = ((true_positives + false_negatives) + labels[0].sum()) / 2

        return false_negatives, true_positives, positives
    
    def conf_matrix_dyn_plus(self, labels, sm):
        if not self.global_mask:
            slope_mask = np.where((labels[0] == 0) & (labels[-1] > 0))[0]
        else:
            slope_mask = np.where(labels[0] == 0)[0]
        label_as_mask = np.where(labels[0])[0]
        
        initial_tps = sm[:, label_as_mask].sum(axis=1)  # same as np.dot(sm, labels[0])
        slope_tps = np.matmul(labels[:, slope_mask], sm[:, slope_mask].T) # same as np.matmul(sm[:, slope_mask], labels[:, slope_mask].T).T
        true_positives = initial_tps + slope_tps
        fn_hard = (~sm)[:, label_as_mask].sum(axis=1)  # (n_thresholds,)
        if self.fn_alpha > 0.0:
            fn_slope = np.matmul(labels[:, slope_mask], (~sm)[:, slope_mask].T)  # (n_slopes, n_thresholds)
            false_negatives = (1.0 - self.fn_alpha) * fn_hard + self.fn_alpha * (fn_hard[np.newaxis, :] + fn_slope)
        else:
            false_negatives = fn_hard
        positives = ((true_positives + false_negatives) + label_as_mask.shape[0]) / 2

        return false_negatives, true_positives, positives

    
    @time_it
    def precision_recall_curve(self, tp, fp, positives, existence, plot=False):
        ones, zeros = np.ones(self.n_slopes)[:, np.newaxis], np.zeros(self.n_slopes)[:, np.newaxis]
        precision = np.hstack((ones, (tp / (tp + fp))))
        recall = np.hstack((zeros, (tp / positives)))
        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = np.multiply(recall[:, 1:], existence)

        if plot:
            # Example used in paper is the first time series of YAHOO
            P = precision[:, 1:] # Remove the very first values which are always 1 or 0
            R = recall[:, 1:]
            n_slopes, n_thresholds = existence.shape
            S = np.repeat(np.arange(n_slopes)[:, None], n_thresholds, axis=1)
            plot_methods = ['surface', 'curves']
            
            for method in plot_methods:
                fig = plt.figure(figsize=(5, 4))
                ax = fig.add_subplot(111, projection='3d')

                if method == 'surface':
                    ax.plot_surface(R, S, P, cmap='flare_r', alpha=1)
                else:
                    for i in range(n_slopes):
                        ax.plot(
                            R[i],
                            np.full_like(R[i], i),
                            P[i],
                            color=sns.color_palette("flare_r", n_slopes)[i],
                            linewidth=1.5,
                        )

                ax.set_xlabel('Recall')
                ax.set_ylabel('Buffer')
                ax.set_zlabel('Precision')            
                ax.text(
                    x=5, y=5, z=7,  # slightly inside top-left, on top of surface
                    s='(c)',
                    transform=ax.transAxes,      # use axes fraction coordinates
                    fontsize=14,
                    fontweight='bold',
                    color='k',
                    alpha=0.8
                )

                plt.tight_layout()
                plt.savefig(f"experiments/figures/precision_recall_{method}.svg", bbox_inches='tight', bbox_extra_artists=(ax.xaxis.label, ax.yaxis.label, ax.zaxis.label))
                plt.savefig(f"experiments/figures/precision_recall_{method}.pdf", bbox_inches='tight', bbox_extra_artists=(ax.xaxis.label, ax.yaxis.label, ax.zaxis.label))
                plt.show()
            exit()
        return precision, recall
    
    @time_it
    def auc(self, x, y):
        if self.interpolation_mode == 'linear':
            auc_per_slope = self.linear_interpolation(x, y)
        else:
            auc_per_slope = self.stepwise_interpolation(x, y)

        n = len(auc_per_slope)
        indices = np.arange(1, n + 1)

        if self.aggregation_mode == 'geometric':
            log_auc = np.log(np.clip(auc_per_slope, 1e-10, None))
            if self.cumulative:
                return np.exp(np.cumsum(log_auc) / indices)
            else:
                return np.exp(log_auc.mean())
        elif self.aggregation_mode == 'harmonic':
            inv_auc = 1.0 / np.clip(auc_per_slope, 1e-10, None)
            if self.cumulative:
                return indices / np.cumsum(inv_auc)
            else:
                return n / inv_auc.sum()
        else:
            if self.cumulative:
                return np.cumsum(auc_per_slope) / indices
            else:
                return auc_per_slope.mean()

    def linear_interpolation(self, x, y):
        return np.trapezoid(y, x, axis=1)

    def stepwise_interpolation(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        return np.sum(np.multiply(width_pr, height_pr), axis=1)

    def decumulate(self, vus_curve):
        """Invert the running-mean aggregation of a cumulative VUS curve.

        The cumulative VUS curve produced by auc() with cumulative=True is a
        running mean over buffers: C[b] = cumsum(A)[b] / (b+1), where A[b] is
        the AUC-PR at buffer level b on its own. This recovers A[b] exactly:

            A = diff(C * indices, prepend=0)

        so each entry is the per-buffer reward, NOT averaged with all smaller
        buffers. A late detection caught at buffer b no longer smears into the
        running mean of every larger buffer (though A[b] itself can still be
        monotone, since the slope labels are cumulative in b; the first
        difference np.diff(A) gives the incremental per-step gain).

        Only valid for aggregation='mean': geometric/harmonic aggregate via
        cumsum of log / inverse, which diff does not invert.

        Args:
            vus_curve (np.ndarray): cumulative VUS curve, shape (n_slopes,).

        Returns:
            np.ndarray of shape (n_slopes,): per-buffer (marginal) VUS.
        """
        if self.aggregation_mode != 'mean':
            raise ValueError(
                f"decumulate is only defined for aggregation='mean', "
                f"got {self.aggregation_mode!r}."
            )
        vus_curve = np.asarray(vus_curve, dtype=float)
        indices = np.arange(1, len(vus_curve) + 1)
        return np.diff(vus_curve * indices, prepend=0.0)


    def _build_capped_labels(self, label, max_buffer_per_anomaly,
                             max_buffer_per_anomaly_left=None,
                             max_buffer_per_anomaly_right=None):
        """Rebuild the (masked) capped slope array + original T.

        Mirrors compute()'s preamble (get_anomalies_coordinates -> create_safe_mask
        -> global mask -> distance_from_anomaly -> _per_point_cap -> add_slopes) so
        the slope array matches exactly the one compute() builds internally for the
        same per-anomaly caps. compute() builds this array but does not return it;
        score_adapt needs it to compute the random-VUS denominator on the identical
        buffer configuration.

        Caps follow the same convention as compute(): pass either the symmetric
        ``max_buffer_per_anomaly`` (both sides) or the per-side pair
        ``max_buffer_per_anomaly_left`` / ``max_buffer_per_anomaly_right``. All
        None reproduces the uniform full-buffer (non-adaptive) configuration.

        Returns:
            (labels, T): the slope array (n_slopes, T_masked) and the ORIGINAL series
            length T (denominator analytics divide by full T, not T_masked).
        """
        T = len(label)
        ((s_no, e_no), (s_ed, e_ed)), _ = self.get_anomalies_coordinates(label)
        safe_mask, _ = self.create_safe_mask(label, s_ed, e_ed, extra_safe=self.global_mask)
        if self.global_mask:
            label = label[safe_mask]
            ((s_no, e_no), (s_ed, e_ed)), _ = self.get_anomalies_coordinates(label)
        (pos_left, pos_right), _ = self.distance_from_anomaly(label, s_ed, e_ed)
        cap_left = cap_right = None
        b_left, b_right = self._resolve_caps(
            max_buffer_per_anomaly, max_buffer_per_anomaly_left, max_buffer_per_anomaly_right)
        if b_left is not None:
            cap_left, cap_right = self._per_point_cap(label, s_ed, e_ed, b_left, b_right)
        labels, _ = self.add_slopes(label, s_no, e_no, pos_left, pos_right,
                                    cap_left=cap_left, cap_right=cap_right)
        return labels, T

    def _normalized_vus(self, label, score, max_buffer_per_anomaly,
                        max_buffer_per_anomaly_left=None, max_buffer_per_anomaly_right=None,
                        denom='analytical', K=100, rng=None):
        """Per-buffer VUS over its random baseline, under one cap configuration.

        Numerator: cumulative VUS curve under the given per-anomaly caps (the
        detector). Denominator: expected random VUS on the SAME caps (analytical
        base rate). Both are cumulative running-means, so decumulate() recovers the
        per-buffer values A[b] (detector) and g[b] (random base rate).

        Returns two normalized variants, both calibrated to E[.|random]=1:
            ratio_A = VUS / E[VUS_random]          (ratio of means; integral/integral)
            ratio_B = mean_b( A[b] / g[b] )         (mean of ratios; per-buffer divide)

        Caps follow compute()'s convention: symmetric ``max_buffer_per_anomaly`` OR
        the per-side pair (left/right), never both; all None gives the uniform
        full-buffer (non-adaptive) case. PR-only: the b-dependent random baseline is
        only meaningful for VUS-PR.
        """
        (vus_curve, _), _ = self.compute(
            label, score,
            max_buffer_per_anomaly=max_buffer_per_anomaly,
            max_buffer_per_anomaly_left=max_buffer_per_anomaly_left,
            max_buffer_per_anomaly_right=max_buffer_per_anomaly_right)
        labels, T   = self._build_capped_labels(
            label, max_buffer_per_anomaly,
            max_buffer_per_anomaly_left, max_buffer_per_anomaly_right)
        denom_curve = self._compute_denom_curve(labels, T, denom, K, rng)

        A = self.decumulate(vus_curve)
        g = np.clip(self.decumulate(denom_curve), 1e-10, None)

        # Adjusted VUS (aVUS): chance-adjusted skill score, identical in form to the
        # Adjusted Rand Index / Adjusted Mutual Information:
        #     (index - E[random]) / (max - E[random]),  index=VUS, max=1, E[random]=g.
        # random -> 0, perfect -> 1, comparable across b* (both anchors removed at
        # each buffer). Stored UNCLIPPED (<= 1, may be slightly negative when
        # worse-than-chance, exactly like ARI/AMI); clip at 0 only for display.
        # ratio_* (VUS / g) is the unbounded diagnostic.
        vus_last, denom_last = float(vus_curve[-1]), float(denom_curve[-1])
        skill_A = (vus_last - denom_last) / max(1.0 - denom_last, 1e-10)
        skill_B = float(np.mean((A - g) / np.clip(1.0 - g, 1e-10, None)))
        return {
            'vus':         vus_last,
            'denom':       denom_last,
            'ratio_A':     float(vus_last / max(denom_last, 1e-10)),
            'ratio_B':     float(np.mean(A / g)),
            'skill_A':     float(skill_A),
            'skill_B':     skill_B,
            'vus_curve':   vus_curve,
            'denom_curve': denom_curve,
        }

    def score_adapt(self, label, score, b_star=None, b_star_left=None, b_star_right=None,
                    denom='analytical', K=100, rng=None):
        """Normalized VUS_adapt — the scoring step, fully separate from selection.

        Takes the per-anomaly b* chosen by a selection step (e.g.
        compute_per_anomaly_buffer) and returns the normalized adaptive
        score plus the non-adaptive (full-buffer) counterparts, so experiments can
        compare ratio_A vs ratio_B vs ratio_full vs raw VUS on the same call.

        Caps: pass either the symmetric ``b_star`` (one buffer per anomaly, applied
        to both sides) or the per-side pair ``b_star_left`` / ``b_star_right`` (the
        asymmetric widths from compute_per_anomaly_buffer(per_side=True)),
        never both. The 'full' (non-adaptive) counterparts are unaffected by the
        choice.

        Does NOT select b*. PR-only (metric='vus_pr'); analytical denom only.

        Returns dict:
            vus_adapt      raw adaptive VUS (today's number)
            ratio_adapt_A  VUS_adapt / E[VUS_random_adapt]      (ratio of means, unbounded)
            ratio_adapt_B  mean_b( A_adapt[b] / g_adapt[b] )    (mean of ratios, unbounded)
            skill_adapt_A  (VUS_adapt - g)/(1 - g)              Adjusted VUS (aVUS), ARI/AMI-style
            skill_adapt_B  mean_b (A-g)/(1-g)                   per-buffer adjusted, then averaged
            vus_full, ratio_full_A/B, skill_full_A/B            non-adaptive (uniform L) counterparts
            denom_adapt, denom_full
            _adapt, _full  full per-buffer curves (for plotting / debugging)

        ratio_* is the unbounded skill-over-random diagnostic; skill_* is the
        chance-adjusted Adjusted VUS (random -> 0, perfect -> 1), stored UNCLIPPED
        (<= 1, may be slightly negative); clip at 0 for display.
        """
        if not self.cumulative:
            raise ValueError("score_adapt requires cumulative=True")
        adapt = self._normalized_vus(
            label, score, b_star,
            max_buffer_per_anomaly_left=b_star_left,
            max_buffer_per_anomaly_right=b_star_right,
            denom=denom, K=K, rng=rng)
        full  = self._normalized_vus(label, score, None, denom=denom, K=K, rng=rng)
        return {
            'vus_adapt':     adapt['vus'],
            'ratio_adapt_A': adapt['ratio_A'],
            'ratio_adapt_B': adapt['ratio_B'],
            'skill_adapt_A': adapt['skill_A'],
            'skill_adapt_B': adapt['skill_B'],
            'denom_adapt':   adapt['denom'],
            'vus_full':      full['vus'],
            'ratio_full_A':  full['ratio_A'],
            'ratio_full_B':  full['ratio_B'],
            'skill_full_A':  full['skill_A'],
            'skill_full_B':  full['skill_B'],
            'denom_full':    full['denom'],
            '_adapt':        adapt,
            '_full':         full,
        }


    def compute_per_anomaly_existence(self, label, score):
        """
        Returns per-anomaly existence curves independent for each anomaly.

        For anomaly k at buffer b: checks if any score in the expanded window
        [start_k - b, end_k + b] exceeds a given threshold. The per-anomaly
        curve value at b is the fraction of thresholds where this holds.

        At runtime, averaging curves[k][:b+1] gives the existence score for
        anomaly k when the buffer slider is set to b.

        Args:
            label (np.ndarray): binary ground truth, shape (T,)
            score (np.ndarray): detector scores, shape (T,)

        Returns:
            anom_starts (np.ndarray): (n_anomalies,) start positions
            anom_ends   (np.ndarray): (n_anomalies,) end positions
            curves      (np.ndarray): (n_anomalies, n_slopes) float32
        """
        ((_, _), (start_with_edges, end_with_edges)), _ = \
            self.get_anomalies_coordinates(label)

        n_anomalies = len(start_with_edges)
        T = len(score)
        empty = np.zeros((0, self.n_slopes), dtype=np.float32)

        if n_anomalies == 0:
            return start_with_edges, end_with_edges, empty

        thresholds, _ = self.get_unique_thresholds(score)
        # ascending order for np.searchsorted
        thresholds_asc = thresholds[::-1]
        n_thresholds = len(thresholds)

        b_vals = self.slope_values   # shape (n_slopes,) = [0, 1, ..., L]
        curves = np.zeros((n_anomalies, self.n_slopes), dtype=np.float32)

        for k in range(n_anomalies):
            s0 = int(start_with_edges[k])
            e0 = int(end_with_edges[k])
            core_max = float(score[s0:e0 + 1].max())

            # Running max extending left from s0 (index i → distance i+1)
            left_ext = score[max(0, s0 - self.slope_size):s0][::-1]
            right_ext = score[e0 + 1:min(T, e0 + self.slope_size + 1)]

            # Index 0 = buffer 0 (core only); index b = buffer b
            left_max = np.maximum.accumulate(
                np.concatenate([[core_max], left_ext])
            )
            right_max = np.maximum.accumulate(
                np.concatenate([[core_max], right_ext])
            )

            l_idx = np.minimum(b_vals, len(left_ext))
            r_idx = np.minimum(b_vals, len(right_ext))
            window_max = np.maximum(left_max[l_idx], right_max[r_idx])

            # Fraction of thresholds <= window_max → anomaly "found" rate
            curves[k] = (
                np.searchsorted(thresholds_asc, window_max, side='right')
                / n_thresholds
            )

        return start_with_edges, end_with_edges, curves

    def compute_localization_map(self, label, score):
        """
        Returns per-timestamp arrays for building a TP/FN/FP/TN performance map.

        avg_label[t]: mean fuzzy label across all slope sizes — 1 inside the
        core anomaly, decreasing through the buffer zone, 0 in normal regions.
        score_norm[t]: detector score min-max normalized to [0, 1].

        Interpretation:
          avg_label > 0, score_norm high  -> TP (detected anomaly)
          avg_label > 0, score_norm low   -> FN (missed anomaly)
          avg_label = 0, score_norm high  -> FP (false alarm)
          avg_label = 0, score_norm low   -> TN (correct silence)

        Args:
            label (np.ndarray): binary ground truth, shape (T,)
            score (np.ndarray): detector scores, shape (T,)

        Returns:
            avg_label  (np.ndarray): shape (T,), values in [0, 1]
            score_norm (np.ndarray): shape (T,), values in [0, 1]
        """
        ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges)), _ = \
            self.get_anomalies_coordinates(label)
        (pos_left, pos_right), _ = self.distance_from_anomaly(label, start_with_edges, end_with_edges, clip=True)
        labels, _ = self.add_slopes(label, start_no_edges, end_no_edges, pos_left, pos_right)

        avg_label = labels.mean(axis=0)

        lo, hi = score.min(), score.max()
        score_norm = (score - lo) / (hi - lo + 1e-8)

        return avg_label, score_norm

