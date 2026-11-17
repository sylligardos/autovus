"""AutoVUS: VUS-PR with a buffer chosen per (anomaly, detection) pair.

    from autovus import FastVUS, AutoVUS     # or FastVUSGPU
    auto = AutoVUS(FastVUS(L=128, cumulative=True))
    result = auto.compute(label, score)      # dict, see compute(); the AutoVUS value is result['adjusted']
    auto.anomalies                           # per-anomaly selection of the last compute()
    auto.compute_all_L(label, score)         # the b* and the AutoVUS value for every L' up to L

Selection (select): every anomaly gets a window reaching halfway to its
neighbours. The engine scores the window with a flat buffer, once rewarding
only the buffer before the anomaly (side='left') and once only the buffer
after it (side='right'), or both at once (sides='symmetric', the paper's
AutoVUS); b* is the first argmax of the rule over the buffers (select_buffer).

Scoring (compute): the whole series is scored once by AdjustedVUS with
every anomaly's buffers capped at its b*; the AutoVUS value is 'adjusted'.

The engine only has to provide compute() with cumulative=True, per-anomaly
caps, a settable side and zita, and label_mass; FastVUS and FastVUSGPU both do.
"""
import numpy as np

from .adjustedvus import AdjustedVUS


class AutoVUS:
    RULES = ('paper', 'ratio')
    SIDES = ('symmetric', 'per_side')

    def __init__(self, engine, rule='paper', sides='symmetric', dust_floor=1e-6, min_signal_frac=1e-3):
        """
        Args:
            engine: FastVUS or FastVUSGPU built with cumulative=True.
            rule (str): the selection rule over b, on the marginal curves:
                'paper'  (A(b) - A(0) + 1) / (g(b) - g(0) + 1)
                'ratio'  A(b) / g(b)
            sides (str): 'symmetric' selects one buffer used on both sides of
                every anomaly; 'per_side' selects the left and the right
                buffer separately.
            dust_floor (float): scores below dust_floor * max|score| are set
                to 0 before selection (relative). Numerically-nonzero dust
                (e.g. softmax underflow) would otherwise add bottom thresholds
                that a growing buffer reaches one by one, bumping the marginal
                ratio and the argmax with it. 0 or None disables it.
            min_signal_frac (float): an anomaly whose window holds no score
                above min_signal_frac * max|score| gets b* = 0 (no signal to
                fit a buffer to). 0 or None disables it.
        AutoVUS is defined over VUS-PR only.
        """
        if not engine.cumulative:
            raise ValueError("AutoVUS needs an engine built with cumulative=True")
        if rule not in self.RULES:
            raise ValueError(f"rule={rule!r} must be one of {self.RULES}")
        if sides not in self.SIDES:
            raise ValueError(f"sides={sides!r} must be one of {self.SIDES}")
        self.engine = engine
        self.adjusted = AdjustedVUS(engine)
        self.rule = rule
        self.sides = sides
        self.dust_floor = dust_floor or 0.0
        self.min_signal_frac = min_signal_frac or 0.0
        self.anomalies = []

    @property
    def buffers(self):
        return np.asarray(self.engine.buffers.cpu() if hasattr(self.engine.buffers, 'cpu') else self.engine.buffers)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @staticmethod
    def decumulate(curve):
        """Per-buffer values from a cumulative (running mean) curve."""
        curve = np.asarray(curve, dtype=float)
        return np.diff(curve * np.arange(1, len(curve) + 1), prepend=0.0)

    def rule_curve(self, vus, random):
        """The selection rule over b on the marginal curves (see __init__)."""
        A, g = self.decumulate(vus), self.decumulate(random)
        if self.rule == 'paper':
            return (A - A[0] + 1.0) / np.clip(g - g[0] + 1.0, 1e-10, None)
        return A / np.clip(g, 1e-10, None)

    def select_buffer(self, vus, random, no_signal=False):
        """b* from the cumulative curves of the detector and of a random detector.

        b* is the first argmax of the rule curve (0 when no_signal, see min_signal_frac). core_ratio = vus[0] / random[0]
        says whether the anomaly is detected inside its own interval at all.
        """
        ratio = self.rule_curve(vus, random)
        idx = 0 if no_signal else int(np.argmax(ratio))
        return {
            'buffer_idx': idx,
            'buffer_val': int(self.buffers[idx]),
            'ratio': float(ratio[idx]),
            'vus': float(vus[idx]),
            'core_ratio': float(vus[0] / max(float(random[0]), 1e-10)),
            'ratio_curve': ratio,
            'vus_curve': vus,
            'random_curve': random,
        }

    @staticmethod
    def classify_detection(anomaly):
        """A label for an anomaly, from its selected buffers only.

        No buffer: 'exact' if the detector beats random inside the anomaly
        (core_ratio > 1), else 'missed'. Only the left buffer: 'early'; only
        the right: 'late'; both: 'wide'. With sides='symmetric' both sides
        share one buffer, so the label is 'exact', 'missed' or 'wide'.
        """
        b_left, b_right = anomaly['left']['buffer_val'], anomaly['right']['buffer_val']
        if b_left == 0 and b_right == 0:
            return 'exact' if anomaly['left']['core_ratio'] > 1.0 else 'missed'
        if b_left > 0 and b_right > 0:
            return 'wide'
        return 'early' if b_left > 0 else 'late'

    def select(self, label, score):
        """Per-anomaly buffers: list of dicts with start, end, window, left, right, detection.

        'left' and 'right' are select_buffer dicts. With sides='symmetric' they
        are the same dict (one selection with side='both'). 'detection' is
        classify_detection's label. Selection uses a flat buffer (zita = 1).
        Also stored in self.anomalies.
        """
        label = np.asarray(label)
        score = np.asarray(score, dtype=float)
        T = len(label)
        start, end = self._coordinates(label)
        smax = float(np.max(np.abs(score))) if T else 0.0
        if self.dust_floor and smax > 0:
            score = np.where(np.abs(score) < self.dust_floor * smax, 0.0, score)

        side, zita = self.engine.side, self.engine.zita
        per_anomaly = getattr(self.engine, 'per_anomaly', False)
        self.engine.zita = 1.0
        if per_anomaly:
            self.engine.per_anomaly = False      # local scores only for the final scoring pass
        anomalies = []
        try:
            for k in range(len(start)):
                lo = int((end[k - 1] + start[k]) // 2) if k > 0 else 0
                hi = int((end[k] + start[k + 1]) // 2) if k < len(start) - 1 else T
                anomaly = {'start': int(start[k]), 'end': int(end[k]), 'window': (lo, hi)}
                for name in (('left', 'right') if self.sides == 'per_side' else ('both',)):
                    self.engine.side = name
                    no_signal = bool(self.min_signal_frac) and float(np.max(np.abs(score[lo:hi]))) < self.min_signal_frac * smax
                    anomaly[name] = self.select_buffer(*self.adjusted.curves(label[lo:hi], score[lo:hi]), no_signal=no_signal)
                if self.sides == 'symmetric':
                    anomaly['left'] = anomaly['right'] = anomaly.pop('both')
                anomaly['detection'] = self.classify_detection(anomaly)
                anomalies.append(anomaly)
        finally:
            self.engine.side, self.engine.zita = side, zita
            if per_anomaly:
                self.engine.per_anomaly = True
        self.anomalies = anomalies
        return anomalies

    @staticmethod
    def _coordinates(label):
        """Start and end index of every anomaly (same rule as the engines)."""
        diff = np.diff(label)
        start, end = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0]
        start = np.append([0], start) if label[0] else start
        end = np.append(end, len(label) - 1) if label[-1] else end
        return start, end

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def compute(self, label, score):
        """select(), then AdjustedVUS with every anomaly's buffers capped at its b*.

        Returns AdjustedVUS.compute()'s dict (vus, random, adjusted = the
        AutoVUS value, vus_curve, random_curve) plus 'anomalies'.
        """
        anomalies = self.select(label, score)
        if not anomalies:
            raise ValueError("label has no anomaly: VUS-PR is undefined")
        cap_left = np.array([a['left']['buffer_val'] for a in anomalies])
        cap_right = np.array([a['right']['buffer_val'] for a in anomalies])
        result = self.adjusted.compute(label, np.asarray(score, dtype=float), cap_left, cap_right)
        result['anomalies'] = anomalies
        return result

    # ------------------------------------------------------------------
    # Every L' up to L at once
    # ------------------------------------------------------------------
    def compute_all_L(self, label, score):
        """AutoVUS for every L' in the engine's buffers, from one selection at L.

        The rule curve at buffer b does not depend on L, so the b* of an
        anomaly at L' is the first argmax of its rule curve up to L'. The
        final value needs every anomaly capped at its b*, so the series is
        scored once per distinct set of b* (the set only changes at the L'
        where some b* jumps), and every L' reads its value from the pass of
        its own set.

        Returns a dict; column / element i is L' = self.buffers[i]:
            bstar_left, bstar_right: (n_anomalies, n_buffers) selected buffers
                (the same matrix with sides='symmetric'); the last column is
                what compute() selects.
            set_start: indices i where the set of b* changes (starts with 0);
                one scoring pass each.
            adjusted: the AutoVUS value per L'; vus, random: its two parts.
            anomalies: select()'s list at L.
        Local (per_anomaly) scores are not computed here.
        """
        anomalies = self.select(label, score)
        if not anomalies:
            raise ValueError("label has no anomaly: VUS-PR is undefined")
        buffers = self.buffers
        n = len(buffers)

        def running(selection):
            if selection['buffer_idx'] == 0:              # 0 at L (or no signal) is 0 at every L'
                return np.zeros(n, dtype=int)
            ratio = selection['ratio_curve']
            return buffers[[int(np.argmax(ratio[:i + 1])) for i in range(n)]]

        bstar_left = np.array([running(a['left']) for a in anomalies])
        bstar_right = bstar_left if self.sides == 'symmetric' else np.array([running(a['right']) for a in anomalies])

        changed = np.any(bstar_left[:, 1:] != bstar_left[:, :-1], axis=0) | np.any(bstar_right[:, 1:] != bstar_right[:, :-1], axis=0)
        set_start = np.where(np.r_[True, changed])[0]

        score = np.asarray(score, dtype=float)
        vus, random = np.zeros(n), np.zeros(n)
        per_anomaly = getattr(self.engine, 'per_anomaly', False)
        if per_anomaly:
            self.engine.per_anomaly = False
        try:
            for s, e in zip(set_start, np.r_[set_start[1:], n]):
                v, r = self.adjusted.curves(label, score, bstar_left[:, s], bstar_right[:, s])
                vus[s:e], random[s:e] = v[s:e], r[s:e]
        finally:
            if per_anomaly:
                self.engine.per_anomaly = True
        return {
            'bstar_left': bstar_left,
            'bstar_right': bstar_right,
            'set_start': set_start,
            'adjusted': self.adjusted.adjust(vus, random),
            'vus': vus,
            'random': random,
            'anomalies': anomalies,
        }
