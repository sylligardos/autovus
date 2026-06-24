"""
REFERENCE ONLY -- not imported, not executed.

This file preserves the original, more complicated per-anomaly adaptive buffer
machinery that was replaced by the lean argmax pipeline in vus_numpy.py
(VUSNumpy._select_adaptive_buffer / ._select_window / .compute_per_anomaly_buffer).

What it contained that the new pipeline drops:
  - select_best_buffer methods (penalized/elbow/first_peak) and the R1-R4 rule
    system (_apply_penalized_rules) -- the new pipeline is pure argmax of the
    marginal skill ratio, with only the "no buffer beats random -> 0" guard and
    the no-signal guards (dust_floor / min_signal_frac).
  - tau / tau_floor distance penalty, detection_gate / gate_margin /
    gate_strong_frac -- all no-ops in the production path.
  - the DOUBLE de-cumulation (np.diff applied twice) of the skill ratio; the new
    pipeline de-cumulates ONCE (ratio[b] = AUC_detector(b) / E[AUC_random(b)]).

Kept here verbatim as a record of what was tried. The methods below were
VUSNumpy methods (they reference self.* and module-level select_best_buffer,
which still lives in vus_numpy.py).
"""


class _LegacyVUSNumpyPerAnomaly:  # reference-only stub so this file stays valid Python
    @time_it
    def find_adaptive_buffer(
        self, 
        labels, 
        score, 
        vus_cumulative, 
        T, 
        denom='analytical', 
        method='penalized',
        K=100, 
        rng=None, 
        plot=False,
        tau=None,
        detection_gate=False,
        gate_margin=10,
        gate_strong_frac=0.8,
        marginal=True,
        rules=None
    ):
        """Select the best buffer b* from a precomputed cumulative VUS curve.

        Args:
            labels         (np.ndarray): Slope array (n_slopes, T_masked); row 0 is the binary label.
            score          (np.ndarray): Detector scores, used for MC denom paths.
            vus_cumulative (np.ndarray): VUS per buffer, shape (n_slopes,).
            T              (int): Original series length before global masking.
            denom          (str): 'analytical', 'mc_bernoulli', or 'mc_uniform'.
            method         (str): Buffer selection method — 'argmax', 'elbow', or 'first_peak'.
            K              (int): Number of MC draws (only used for mc_* denoms).
            rng            : numpy random Generator; created fresh if None.
            plot           (bool): if True, show label, score, and ratio curve with b* marked.
            rules          (dict or None): rule toggle flags forwarded to _apply_penalized_rules.
                None (default) activates all rules (production mode). Pass e.g.
                {'r2': True} to run only R2 (ablation mode). detection_gate=True
                is equivalent to rules={'r4a': True, 'r4b': True} and is kept for
                backward compatibility; when rules is not None it takes precedence.
            marginal       (bool): if True (and aggregation is 'mean'), select b*
                on the de-cumulated per-buffer skill ratio instead of the cumulative
                ratio, so b* lands at the detection's end rather than ~2x past it.

        Returns:
            dict with keys 'vus', 'ratio' (the ratio used for selection: marginal
            when enabled, else cumulative), 'ratio_cumulative', 'denom', 'best'.
        """
        denom_curve = self._compute_denom_curve(labels, T, denom, K, rng)

        ratio = vus_cumulative / np.clip(denom_curve, 1e-10, None)

        # Buffer selection runs on the PER-STEP INCREMENTAL GAIN ratio (View 2 from
        # IDEAS §12). View 1 (inverted running mean) still grows monotonically because
        # slope labels are cumulative in b — a detection at b stays a TP for all b'>b.
        # View 2 takes the first difference of View 1, isolating the reward *first*
        # earned at each buffer step. This produces a spike at the detection's true
        # extent and is flat elsewhere.
        #
        # Old method (View 1 — inverted running mean, still monotone):
        # if marginal and self.aggregation_mode == 'mean':
        #     n_idx      = np.arange(1, len(vus_cumulative) + 1)
        #     marg_vus   = np.diff(vus_cumulative * n_idx, prepend=0.0)
        #     marg_denom = np.diff(denom_curve   * n_idx, prepend=0.0)
        #     sel_ratio  = marg_vus / np.clip(marg_denom, 1e-10, None)
        # else:
        #     sel_ratio  = ratio
        if marginal and self.aggregation_mode == 'mean':
            n_idx      = np.arange(1, len(vus_cumulative) + 1)
            marg_vus   = np.diff(vus_cumulative * n_idx, prepend=0.0)
            marg_denom = np.diff(denom_curve   * n_idx, prepend=0.0)
            incr_vus   = np.diff(marg_vus,   prepend=0.0)
            incr_denom = np.diff(marg_denom, prepend=0.0)
            sel_ratio  = incr_vus / np.clip(incr_denom, 1e-10, None)
        else:
            sel_ratio  = ratio

        # Compute R4 gate inputs so select_best_buffer can pass them to
        # _apply_penalized_rules. The gate asks whether the detector's *strongest*
        # response (>= gate_strong_frac of its own max) lands within +/- gate_margin
        # of the core. A contiguous late/early detection touches the core edge so
        # it passes even though its block extends further; an isolated peak on
        # neighbouring (e.g. periodic) structure sits beyond the margin and is
        # rejected. gate_margin is a small, anomaly-length-independent tolerance.
        core_mask = None
        gate_thr  = None
        if detection_gate or (rules is not None and (rules.get('r4a') or rules.get('r4b'))):
            core_idx = np.where(labels[0] > 0)[0]
            if core_idx.size:
                margin = int(gate_margin)
                lo = max(0, core_idx[0] - margin)
                hi = min(len(score), core_idx[-1] + margin + 1)
                core_mask = np.arange(lo, hi)
                gate_thr  = gate_strong_frac * float(np.max(score))

        # Default selection is PURE ARGMAX on the (marginal) skill ratio: no rule
        # heuristically drops a detection. This keeps us consistent with VUS, where
        # a buffer L means "count any detection within L of the anomaly" — argmax
        # picks the width that best captures whatever detection exists in [0, L].
        # The rule set is applied only when explicitly requested (ablation studies)
        # or via the legacy detection_gate flag.
        if rules is not None:
            effective_rules = rules
        elif detection_gate:
            effective_rules = {'r1': True, 'r2': not marginal, 'r3': True, 'r4a': True, 'r4b': True}
        else:
            # Default: argmax on the marginal ratio with ONLY the no-signal guards —
            # R1 (no buffer beats random -> 0) and R4a (constant score -> 0). These
            # never drop a real detection; they return 0 only when there is nothing
            # to include. The detection-dropping rules (R2/R3/R4b) stay off.
            effective_rules = {'r1': True, 'r4a': True}

        best = select_best_buffer(
            sel_ratio, self.slope_values, method=method, tau=tau,
            rules=effective_rules, score=score, core_mask=core_mask, gate_thr=gate_thr,
        )
        best['vus_score'] = float(vus_cumulative[best['buffer_idx']])

        if plot:
            label_1d = labels[0]
            b_val    = best['buffer_val']
            b_idx    = best['buffer_idx']

            fig, (ax_lbl, ax_sc, ax_ratio) = plt.subplots(
                3, 1, figsize=(14, 7),
                gridspec_kw={'height_ratios': [1, 1, 2]},
                sharex=False,
            )
            ax_sc.sharex(ax_lbl)
            fig.suptitle(
                f"find_adaptive_buffer  |  method={method}  b*={b_val}  VUS={best['vus_score']:.4f}",
                fontsize=9, fontweight='bold',
            )

            ax_lbl.fill_between(range(len(labels[b_idx])), labels[b_idx], color='steelblue', alpha=0.3, label=f'b*={b_val}')
            ax_lbl.plot(label_1d, color='crimson', linewidth=0.8, label='original')
            ax_lbl.legend(fontsize=7, loc='upper right')
            ax_lbl.set_ylabel('label', fontsize=8)
            ax_lbl.tick_params(labelsize=7, labelbottom=False)

            ax_sc.plot(score, color='dimgray', linewidth=0.7)
            ax_sc.set_ylabel('score', fontsize=8)
            ax_sc.set_xlabel('time', fontsize=8)
            ax_sc.tick_params(labelsize=7)

            ax_ratio.plot(self.slope_values, sel_ratio, color='steelblue', linewidth=1.5, label='ratio (marginal)' if (marginal and self.aggregation_mode == 'mean') else 'ratio')
            ax_ratio.plot(self.slope_values, ratio, color='lightsteelblue', linewidth=1.0, linestyle='--', alpha=0.7, label='ratio (cumulative)')
            ax_ratio.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax_ratio.axvline(b_val, color='crimson', linewidth=2.0, label=f'b*={b_val} ({method})')
            ax_ratio.scatter([b_val], [sel_ratio[b_idx]], color='crimson', s=60, zorder=5)
            ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_xlabel('buffer', fontsize=8)
            ax_ratio.set_xlim(0, int(self.slope_values[-1]))
            ax_ratio.legend(fontsize=8)
            ax_ratio.tick_params(labelsize=7)

            plt.tight_layout()
            plt.show()

        return {
            'vus':            vus_cumulative,
            'ratio':          sel_ratio,      # marginal (used for selection)
            'ratio_cumulative': ratio,
            'denom':          denom_curve,
            'best':           best,
        }



    def _select_window_buffer(self, win_label, win_score, tau_k, score_scale,
                              gate_margin, gate_strong_frac, min_signal_frac, rules):
        """Run adaptive VUS on one anomaly window and return its result dict.

        Uses the current self.reward_side, so callers can flip the side around
        this call to select a buffer for one side only. Applies the no-signal
        guard (generalised R4a) and the reachable-buffer cap in place on
        result['best'].
        """
        (result_k, _), _ = self.compute(
            win_label, win_score, adaptive=True,
            adaptive_opts={
                'method': 'penalized',
                'marginal': True,
                'tau': tau_k,
                'detection_gate': False,
                'gate_margin': gate_margin,
                'gate_strong_frac': gate_strong_frac,
                'plot': False,
                'rules': rules,   # None -> pure argmax; dict -> ablation rules
            },
        )

        best = result_k['best']

        # Reachable-buffer cap. The window is a hard slice: past its edges there
        # is no time series, so a buffer cannot be placed there. The VUS soft
        # label still grows beyond the edge (a larger slope_size makes the ramp
        # shallower, so the soft value at a fixed position keeps rising toward 1)
        # -- that is pure ramp inflation over already-included points, not a wider
        # detection. When a detection sits flush against a window edge this makes
        # the marginal skill ratio climb monotonically and argmax runs to
        # slope_size. Restrict b* to the room actually available on the rewarded
        # side(s): left/right room for one-sided selection, their min for the
        # symmetric ('both') buffer (it is shared, so it must fit on BOTH sides).
        ones = np.flatnonzero(win_label > 0)
        if ones.size:
            Lw         = len(win_label)
            left_room  = int(ones[0])               # samples left of the core
            right_room = int(Lw - 1 - ones[-1])     # samples right of the core
            if self.reward_side == 'left':
                room = left_room
            elif self.reward_side == 'right':
                room = right_room
            else:                                   # 'both' -> shared buffer
                room = min(left_room, right_room)

            if best['buffer_val'] > room:
                sel_ratio = result_k['ratio']
                allowed   = self.slope_values <= room
                new_idx   = int(np.argmax(np.where(allowed, sel_ratio, -np.inf)))
                # No-skill guard within the reachable range (mirrors R1): if no
                # reachable buffer beats random, none is worth placing -> b*=0.
                if not np.any((sel_ratio[allowed] - 1.0) > 1e-4):
                    new_idx = 0
                best['buffer_idx'] = new_idx
                best['buffer_val'] = int(self.slope_values[new_idx])
                best['ratio']      = float(sel_ratio[new_idx])
                if 'vus_score' in best:
                    best['vus_score'] = float(result_k['vus'][new_idx])

            # Truncate the selection ratio beyond the reachable buffer. Past the
            # window edge the curve is pure ramp inflation over already-included
            # points (see above), so it is not a meaningful skill signal. NaN-ing
            # it stops the debug plot at the window edge instead of drawing the
            # misleading climbing tail. reachable_buffer is stored for the plot.
            result_k['ratio'] = np.where(self.slope_values <= room, result_k['ratio'], np.nan)
            result_k['reachable_buffer'] = int(room)

        # No-signal guard (generalised R4a): if this window holds no score above
        # min_signal_frac of the detector's global scale, there is no detection to
        # include -> b*=0, regardless of where argmax fell.
        slice_max = float(np.max(win_score)) if win_score.size else 0.0
        if best['buffer_val'] != 0 and (score_scale <= 0 or slice_max < min_signal_frac * score_scale):
            best['buffer_idx'] = 0
            best['buffer_val'] = 0
            best['ratio']      = float(result_k['ratio'][0])
            if 'vus_score' in best:
                best['vus_score'] = float(result_k['vus'][0])
        return result_k



    def compute_per_anomaly_adaptive_slicing(self, label, score, plot=False, tau_floor=10, gate_margin=10, gate_strong_frac=0.8, min_signal_frac=1e-3, rules=None, per_side=True, dust_floor=1e-6):
        """Per-anomaly adaptive buffer selection.

        For each anomaly k, slices a window spanning half the distance to its
        neighbors on each side, runs cumulative VUS on that window, and selects
        b* independently. This lets short and long anomalies pick different buffers.

        When ``per_side=True`` (the default), each anomaly selects TWO buffers, one
        per side: the window is scored once with reward_side='left' (b*_left, the
        pre-anomaly buffer) and once with reward_side='right' (b*_right, the
        post-anomaly buffer). This roughly doubles the per-window cost but lets the
        left and right of an anomaly grow to different widths. The final full-series
        VUS is then computed with these asymmetric per-side caps. reward_side is
        restored afterwards. Pass ``per_side=False`` to fall back to the original
        symmetric selection (a single b* per anomaly, shared by both sides).

        Buffer selection (default) is PURE ARGMAX on the marginal (per-buffer)
        skill ratio: b* is the buffer width that best captures the detector's skill
        over random within [0, L]. This is consistent with VUS — a buffer L means
        "count any detection within L of the anomaly" — and applies no heuristic
        rule that would drop a detection. Pass rules=<dict> to instead run the
        rule-based ablation (R1/R2/R3/R4 detection gate, etc.).

        Args:
            label (np.ndarray): binary ground truth, shape (T,)
            score (np.ndarray): detector scores, shape (T,)
            plot  (bool)      : if True, show a debug figure after each anomaly
            tau_floor (float) : retained for API compatibility (no longer affects
                the default pure-argmax selection).
            gate_margin (int): forwarded to the detection gate only when an
                ablation `rules` dict enables R4b; ignored by the default path.
            gate_strong_frac (float): score fraction defining a "strong" response
                for the gate; only used when `rules` enables R4b.
            rules (dict or None): None (default) selects pure argmax on the marginal
                ratio. A dict (e.g. {'r4b': True}) runs the corresponding rule
                ablation through _apply_penalized_rules.
            dust_floor (float): relative cutoff below which scores are snapped to
                zero before any buffer selection. Scores with magnitude <
                dust_floor * max(|score|) are treated as numerical underflow, not
                signal (see body for why this removes spurious buffer growth). Set
                to 0/None to disable.

        After selecting b* for every anomaly, the full series is run through the
        normal cumulative VUS once with these per-anomaly caps (see
        compute's max_buffer_per_anomaly argument), and the resulting final VUS
        value is stored on each returned dict under 'final_vus' and annotated on
        the per-anomaly debug plots.

        Returns:
            list of dicts, one per anomaly:
                'start'     (int)        : anomaly start index in original series
                'end'       (int)        : anomaly end index in original series
                'window'    (tuple)      : (l_k, r_k) slice indices used
                'best'      (dict)       : select_best_buffer result (buffer_idx, buffer_val, ratio).
                    In per_side mode this is the side with the larger b* (for
                    backward compatibility with symmetric consumers).
                'vus'       (np.ndarray) : per-buffer VUS curve for this anomaly's window
                'final_vus' (float)      : full-series VUS with per-anomaly caps applied
                'detection_type' (str)   : timing label from classify_detection
                    (per_side: early/late/exact/missed; symmetric:
                    exact/missed/lagged).
            In per_side mode each dict additionally carries 'best_left',
            'best_right', 'vus_left', 'vus_right', 'ratio_left', 'ratio_right'.
        """
        if not self.cumulative:
            raise ValueError("compute_per_anomaly_adaptive_slicing requires cumulative=True")

        T = len(label)
        ((_, _), (starts, ends)), _ = self.get_anomalies_coordinates(label)
        n_anomalies = len(starts)

        if n_anomalies == 0:
            return []

        # Floor floating-point dust to zero before any buffer selection (see
        # floor_dust for the full rationale). Selection and the downstream
        # score_adapt MUST run on the SAME cleaned score, so callers driving both
        # steps should clean once with floor_dust and pass dust_floor=0 here.
        score = self.floor_dust(score, dust_floor)

        # Detector scale over the whole series: a slice whose scores never rise to a
        # meaningful fraction of this has no detection in it (just numerical noise),
        # so b*=0. This generalises R4a (exactly-constant -> no signal) to the common
        # "numerically flat" case, e.g. a slice maxing at 1e-9 against a global max of
        # 1, where argmax would otherwise latch onto a floating-point blip.
        score_scale = float(np.max(score)) if score.size else 0.0

        orig_side = self.reward_side
        results = []
        try:
            for k in range(n_anomalies):
                l_k = int((ends[k - 1] + starts[k]) // 2) if k > 0 else 0
                r_k = int((ends[k] + starts[k + 1]) // 2) if k < n_anomalies - 1 else T

                tau_k = max(int(ends[k] - starts[k] + 1), tau_floor)
                win_label, win_score = label[l_k:r_k], score[l_k:r_k]

                r = {'start': int(starts[k]), 'end': int(ends[k]), 'window': (l_k, r_k)}

                if not per_side:
                    # Default: pure argmax on the marginal skill ratio (rules=None).
                    # Pass rules=<dict> to run the rule-based ablation instead.
                    result_k = self._select_window_buffer(
                        win_label, win_score, tau_k, score_scale,
                        gate_margin, gate_strong_frac, min_signal_frac, rules)
                    r['best']  = result_k['best']
                    r['vus']   = result_k['vus']
                    r['ratio'] = result_k['ratio']
                else:
                    # Select a buffer for each side independently by scoring the
                    # same window once per reward_side.
                    self.reward_side = 'left'
                    res_l = self._select_window_buffer(
                        win_label, win_score, tau_k, score_scale,
                        gate_margin, gate_strong_frac, min_signal_frac, rules)
                    self.reward_side = 'right'
                    res_r = self._select_window_buffer(
                        win_label, win_score, tau_k, score_scale,
                        gate_margin, gate_strong_frac, min_signal_frac, rules)
                    self.reward_side = orig_side

                    r['best_left']   = res_l['best']
                    r['best_right']  = res_r['best']
                    r['vus_left']    = res_l['vus']
                    r['vus_right']   = res_r['vus']
                    r['ratio_left']  = res_l['ratio']
                    r['ratio_right'] = res_r['ratio']
                    # Backward-compat single 'best': the side with the larger b*.
                    dom = res_l if res_l['best']['buffer_val'] >= res_r['best']['buffer_val'] else res_r
                    r['best']  = dom['best']
                    r['vus']   = dom['vus']
                    r['ratio'] = dom['ratio']

                results.append(r)
        finally:
            self.reward_side = orig_side

        # Recompute VUS on the full series, capping each anomaly's buffer at its b*.
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

        # Baseline: full-buffer VUS with no adaptive selection (largest buffer).
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

