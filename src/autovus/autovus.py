"""AutoVUS: VUS-PR with a buffer chosen per (anomaly, detection) pair.

    from autovus import FastVUS, AutoVUS     # or FastVUSGPU
    auto = AutoVUS(FastVUS(L=128, cumulative=True))
    result = auto.compute(label, score)      # dict, see score()
    auto.anomalies                           # per-anomaly selection of the last compute()

Selection (select): every anomaly gets a window reaching halfway to its
neighbours. The engine scores the window twice, once rewarding only the
buffer before the anomaly (side='left') and once only the buffer after it
(side='right'); each side's b* is the argmax of the skill ratio over the
random baseline (select_buffer), limited to the room inside the window and
set to 0 when the window holds no signal.

Scoring (score): the whole series is scored once with every anomaly's
buffers capped at its b*, and once with the full buffer L; both are reported
raw and adjusted for the random baseline.

The engine only has to provide compute() with cumulative=True, per-anomaly
caps, a settable side, and label_mass; FastVUS and FastVUSGPU both do.
"""
import numpy as np


class AutoVUS:
    RULES = ('cumulative', 'paper', 'ratio')
    SIDES = ('per_side', 'symmetric')

    def __init__(self, engine, dust_floor=1e-6, min_signal_frac=1e-3,
                 rule='cumulative', sides='per_side', select_zita=None):
        """
        Args:
            engine: FastVUS or FastVUSGPU built with cumulative=True.
            dust_floor (float): scores below dust_floor * max|score| are set
                to 0 before selection (floating-point dust, not signal).
                0 or None disables it.
            min_signal_frac (float): a window whose max score is below this
                fraction of the series' max score is treated as signal-free.
            rule (str): the selection rule, see select_buffer.
                'cumulative'  (C(b) - C(0) + 1) / (G(b) - G(0) + 1)  on the cumulative curves
                'paper'       (A(b) - A(0) + 1) / (g(b) - g(0) + 1)  on the marginal curves
                'ratio'       A(b) / g(b)                            on the marginal curves
            sides (str): 'per_side' selects a buffer for the left and the
                right side of every anomaly separately; 'symmetric' selects
                one buffer used on both sides.
            select_zita (float or None): zita used by the engine during
                selection only (1.0 = flat buffer); None keeps the engine's
                zita. Scoring always uses the engine's zita.
        AutoVUS is defined over VUS-PR only; the engines' VUS-ROC has no
        random baseline here (see TRACKER.md).
        """
        if not engine.cumulative:
            raise ValueError("AutoVUS needs an engine built with cumulative=True")
        if rule not in self.RULES:
            raise ValueError(f"rule={rule!r} must be one of {self.RULES}")
        if sides not in self.SIDES:
            raise ValueError(f"sides={sides!r} must be one of {self.SIDES}")
        self.engine = engine
        self.dust_floor = dust_floor
        self.min_signal_frac = min_signal_frac
        self.rule = rule
        self.sides = sides
        self.select_zita = select_zita
        self.anomalies = []

    @property
    def buffers(self):
        return np.asarray(self.engine.buffers.cpu() if hasattr(self.engine.buffers, 'cpu') else self.engine.buffers)

    # ------------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------------
    def curves(self, label, score, cap_left=None, cap_right=None):
        """Cumulative VUS-PR of the detector and of a random detector, per buffer.

        The random curve is analytical: a random detector's expected AUC-PR
        at buffer b is the mass of the buffered label over T, and the
        cumulative curve is its running mean over buffers.
        """
        vus = np.asarray(self.engine.compute(label, score, cap_left, cap_right, metric='pr'))
        mass = np.asarray(self.engine.label_mass) / len(label)
        random = np.cumsum(mass) / np.arange(1, len(mass) + 1)
        return vus, random

    @staticmethod
    def decumulate(curve):
        """Per-buffer values from a cumulative (running mean) curve."""
        curve = np.asarray(curve, dtype=float)
        return np.diff(curve * np.arange(1, len(curve) + 1), prepend=0.0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def rule_curve(self, vus, random):
        """The selection rule's curve over b, from the cumulative curves."""
        if self.rule == 'cumulative':
            return (vus - vus[0] + 1.0) / np.clip(random - random[0] + 1.0, 1e-10, None)
        A, g = self.decumulate(vus), self.decumulate(random)
        if self.rule == 'paper':
            return (A - A[0] + 1.0) / np.clip(g - g[0] + 1.0, 1e-10, None)
        return A / np.clip(g, 1e-10, None)

    def select_buffer(self, vus, random):
        """b* from the cumulative curves of the detector and of a random detector.

        The rule (see __init__) gives a curve over b whose value at b is the
        gain of the detector relative to the gain of a random detector. b* is
        the first argmax of the curve, or 0 when no buffer beats b = 0
        (curve[b] <= curve[0] everywhere).
        core_ratio = vus[0] / random[0] says whether the anomaly is detected
        inside its own interval at all.
        """
        ratio = self.rule_curve(vus, random)
        idx = int(np.argmax(ratio))
        if not np.any(ratio[1:] - ratio[0] > 1e-4):
            idx = 0
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

    def _select_side(self, label, score, side, score_scale):
        """b* of one side of one anomaly window."""
        self.engine.side = side
        vus, random = self.curves(label, score)
        best = self.select_buffer(vus, random)
        ratio = best['ratio_curve']

        # A buffer cannot extend past the window: limit b* to the room there.
        inside = np.flatnonzero(label > 0)
        room_left, room_right = int(inside[0]), int(len(label) - 1 - inside[-1])
        room = {'left': room_left, 'right': room_right, 'both': min(room_left, room_right)}[side]
        if best['buffer_val'] > room:
            allowed = self.buffers <= room
            idx = int(np.argmax(np.where(allowed, ratio, -np.inf)))
            if not np.any(ratio[allowed] - ratio[0] > 1e-4):
                idx = 0
            best.update(buffer_idx=idx, buffer_val=int(self.buffers[idx]), ratio=float(ratio[idx]), vus=float(vus[idx]))
        best['room'] = room

        # A window whose score never rises above noise holds no detection.
        if best['buffer_val'] != 0 and (score_scale <= 0 or float(np.max(score)) < self.min_signal_frac * score_scale):
            best.update(buffer_idx=0, buffer_val=0, ratio=float(ratio[0]), vus=float(vus[0]))
        return best

    @staticmethod
    def floor_dust(score, dust_floor):
        """Copy of score with |values| below dust_floor * max|score| set to 0."""
        score = np.asarray(score, dtype=float)
        if not dust_floor or score.size == 0 or np.max(np.abs(score)) <= 0:
            return score
        return np.where(np.abs(score) < dust_floor * np.max(np.abs(score)), 0.0, score)

    @staticmethod
    def classify_detection(anomaly, tol=0):
        """'exact', 'early', 'late' or 'missed' from an anomaly's selected buffers.

        Neither buffer grown: 'exact' if the detector beats random inside the
        anomaly (core_ratio > 1), else 'missed'. Otherwise the larger side
        tells the timing: left = 'early', right = 'late' (ties go to 'late').
        """
        b_left, b_right = anomaly['left']['buffer_val'], anomaly['right']['buffer_val']
        if b_left <= tol and b_right <= tol:
            return 'exact' if anomaly['left']['core_ratio'] > 1.0 else 'missed'
        return 'early' if b_left - b_right > tol else 'late'

    def select(self, label, score):
        """Per-anomaly buffers: list of dicts with start, end, window, left, right, detection.

        'left' and 'right' are select_buffer dicts (plus 'room'). With
        sides='symmetric' they are the same dict (one selection with
        side='both') and 'detection' is 'exact', 'missed' or 'both'. Also
        stored in self.anomalies.
        """
        label = np.asarray(label)
        score = self.floor_dust(score, self.dust_floor)
        T = len(label)
        start, end = self._coordinates(label)
        score_scale = float(np.max(score)) if score.size else 0.0

        side, zita = self.engine.side, self.engine.zita
        if self.select_zita is not None:
            self.engine.zita = self.select_zita
        anomalies = []
        try:
            for k in range(len(start)):
                lo = int((end[k - 1] + start[k]) // 2) if k > 0 else 0
                hi = int((end[k] + start[k + 1]) // 2) if k < len(start) - 1 else T
                anomaly = {'start': int(start[k]), 'end': int(end[k]), 'window': (lo, hi)}
                if self.sides == 'per_side':
                    anomaly['left'] = self._select_side(label[lo:hi], score[lo:hi], 'left', score_scale)
                    anomaly['right'] = self._select_side(label[lo:hi], score[lo:hi], 'right', score_scale)
                    anomaly['detection'] = self.classify_detection(anomaly)
                else:
                    both = self._select_side(label[lo:hi], score[lo:hi], 'both', score_scale)
                    anomaly['left'] = anomaly['right'] = both
                    if both['buffer_val'] == 0:
                        anomaly['detection'] = 'exact' if both['core_ratio'] > 1.0 else 'missed'
                    else:
                        anomaly['detection'] = 'both'
                anomalies.append(anomaly)
        finally:
            self.engine.side, self.engine.zita = side, zita
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
    def score(self, label, score, cap_left, cap_right):
        """VUS-PR with the given per-anomaly caps and with the full buffer, raw and adjusted.

        Returns a dict:
            vus_adapt, vus_full        the cumulative VUS-PR at L (raw)
            random_adapt, random_full  the random baseline at L
            skill_adapt, skill_full    (vus - random) / (1 - random): 0 = random, 1 = perfect
            skill_per_buffer_adapt/_full  the same per buffer, averaged
            curves_adapt, curves_full  (vus curve, random curve)
        """
        out = {}
        for name, caps in (('adapt', (cap_left, cap_right)), ('full', (None, None))):
            vus, random = self.curves(label, score, *caps)
            per_buffer = self.decumulate(vus)
            per_random = np.clip(self.decumulate(random), 1e-10, None)
            out[f'vus_{name}'] = float(vus[-1])
            out[f'random_{name}'] = float(random[-1])
            out[f'skill_{name}'] = float((vus[-1] - random[-1]) / max(1.0 - random[-1], 1e-10))
            out[f'skill_per_buffer_{name}'] = float(np.mean((per_buffer - per_random) / np.clip(1.0 - per_random, 1e-10, None)))
            out[f'curves_{name}'] = (vus, random)
        return out

    def compute(self, label, score):
        """select() then score() with the selected buffers; the score dict plus 'anomalies'."""
        anomalies = self.select(label, score)
        if not anomalies:
            raise ValueError("label has no anomaly: VUS-PR is undefined")
        score = self.floor_dust(score, self.dust_floor)
        cap_left = np.array([a['left']['buffer_val'] for a in anomalies])
        cap_right = np.array([a['right']['buffer_val'] for a in anomalies])
        result = self.score(label, score, cap_left, cap_right)
        result['anomalies'] = anomalies
        return result
