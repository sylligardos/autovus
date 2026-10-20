"""FastVUS: exact CPU implementation of VUS-PR.

    from autovus import FastVUS
    vus = FastVUS(L=128)
    value = vus.compute(label, score)
    vus.timings   # seconds per step of the last compute()

L is the buffer length added on each side of every anomaly. The buffers
0, step, 2*step, ..., L are all evaluated; VUS-PR is the mean AUC-PR over them.
"""
import functools
import math
import time

import numpy as np


def timed(method):
    """Add the method's wall time to self.timings[<method name>]."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        self.timings[method.__name__] = self.timings.get(method.__name__, 0.0) + time.perf_counter() - start
        return result
    return wrapper


class FastVUS:
    def __init__(self, L=100, step=1, zita=1 / math.sqrt(2), global_mask=True,
                 cumulative=False, side='both', max_thresholds=2000):
        """
        Args:
            L (int): buffer length added on each side of every anomaly.
            step (int): spacing of the evaluated buffer lengths; must divide L.
            zita (float): ramp value at the outer end of a buffer, in [0, 1].
            global_mask (bool): drop the parts of the series that cannot be
                affected by any buffer before the expensive steps.
            cumulative (bool): compute() returns one VUS per buffer length
                (VUS at L = 0, step, ..., L) instead of a single value.
            side (str): 'both', 'left' or 'right': where buffers are added.
            max_thresholds (int or None): cap on the number of thresholds.
        """
        if L < 0 or step < 0:
            raise ValueError(f"L={L} and step={step} must be non-negative")
        if L > 0 and (step == 0 or L % step != 0):
            raise ValueError(f"step={step} must be a positive divisor of L={L}")
        if not 0 <= zita <= 1:
            raise ValueError(f"zita={zita} must be in [0, 1]")
        if side not in ('both', 'left', 'right'):
            raise ValueError(f"side={side!r} must be 'both', 'left' or 'right'")

        self.L = L
        self.step = step
        self.zita = zita
        self.global_mask = global_mask
        self.cumulative = cumulative
        self.side = side
        self.max_thresholds = max_thresholds

        self.buffers = np.arange(0, L + 1, step) if L > 0 else np.array([0])
        self.n_buffers = len(self.buffers)
        self.timings = {}      # seconds per step, filled by compute()
        self.label_mass = None  # sum of the buffered label per buffer, from the last compute()

    # ------------------------------------------------------------------
    # Step 1: anomaly coordinates
    # ------------------------------------------------------------------
    @timed
    def anomaly_coordinates(self, label):
        """Start and end index of every anomaly, as two pairs of arrays.

        Returns ((start_inner, end_inner), (start, end)). The second pair is
        the full list. The first pair leaves out an anomaly touching the first
        or last point of the series, where no buffer can be added on that side.
        """
        diff = np.diff(label)
        start_inner = np.where(diff == 1)[0] + 1
        end_inner = np.where(diff == -1)[0]
        start = np.append([0], start_inner) if label[0] else start_inner
        end = np.append(end_inner, len(label) - 1) if label[-1] else end_inner
        if start.shape != end.shape:
            raise ValueError(f"Unbalanced anomaly starts and ends: {start} vs {end}")
        return (start_inner, end_inner), (start, end)

    # ------------------------------------------------------------------
    # Step 2: global mask
    # ------------------------------------------------------------------
    @timed
    def safe_mask(self, label, start, end, extra_safe=False):
        """Boolean mask of the points that any buffer up to L can reach.

        Every anomaly is extended by L + 1 points on each side (one more with
        extra_safe), so dropping the points outside the mask changes nothing
        in the buffered labels. The score outside the mask only contributes
        false positives, counted separately in compute().
        """
        T = len(label)
        reach = self.L + 1 + int(extra_safe)
        start_safe = np.maximum(start - reach, 0)
        end_safe = np.minimum(end + reach + 1, T - 1)

        delta = np.zeros(T, dtype=np.int8)
        np.add.at(delta, start_safe, 1)
        np.add.at(delta, end_safe, -1)
        mask = np.cumsum(delta) > 0
        # TODO (D2): the last point is set by hand; verify the no-anomaly case end to end.
        mask[-1] = (len(end_safe) > 0 and end_safe[-1] == T - 1) if extra_safe else label[-1]
        return mask

    # ------------------------------------------------------------------
    # Step 3: distance of every point from the anomalies
    # ------------------------------------------------------------------
    @timed
    def distance_from_anomaly(self, label, start, end):
        """Per-point distance to the nearest anomaly boundary, one array per side.

        dist_left[t]  = nearest anomaly start at or after t, minus t (drives the
                        buffer before an anomaly); inf if there is none.
        dist_right[t] = t minus the nearest anomaly end at or before t (drives
                        the buffer after an anomaly); inf if there is none.
        Points inside an anomaly get distance 0 on the relevant side.
        """
        # TODO: builds a (T, n_anomalies) matrix; revisit if memory bites.
        T = len(label)
        dist_left = np.full(T, np.inf)
        dist_right = np.full(T, np.inf)
        if len(start) == 0:
            return dist_left, dist_right
        t = np.arange(T)[:, None]
        to_start = start[None, :] - t
        dist_left = np.where(to_start >= 0, to_start, np.inf).min(axis=1)
        from_end = t - end[None, :]
        dist_right = np.where(from_end >= 0, from_end, np.inf).min(axis=1)
        return dist_left, dist_right

    # ------------------------------------------------------------------
    # Step 4: buffered labels, one row per buffer length
    # ------------------------------------------------------------------
    def _ramp(self, dist, cap):
        """Ramp of one side for buffers step..L, shape (n_buffers - 1, T).

        A point at distance d from the boundary gets
            sqrt(1 - (1 - zita^2) * d / b)
        for every buffer b >= d, else 0: the curve of the original VUS, which
        equals zita at the outer end of the buffer. dist = inf (no boundary on
        this side) stays 0. cap (shape (T,) or None) limits each point's
        buffer to min(b, cap[t]); None means the uniform buffer.
        """
        T = len(dist)
        out = np.zeros((self.n_buffers - 1, T))
        b = self.buffers[1:, None]                                   # (n_buffers - 1, 1)
        d = dist[None, :]                                            # (1, T)
        b_eff = b if cap is None else np.minimum(b, cap[None, :])
        reached = (b_eff >= d) & (b_eff > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            coef = (1.0 - self.zita ** 2) / b_eff
            np.sqrt(1.0 - coef * d, out=out, where=reached)
        return out

    @timed
    def buffered_labels(self, label, dist_left, dist_right, cap_left=None, cap_right=None):
        """Label with buffers added, shape (n_buffers, T): row i is buffer self.buffers[i].

        Row 0 is the label itself. The left and right ramps are combined with
        the per-point maximum (side='both'), or one of them is used alone.
        Points inside an anomaly are 1 in every row.
        """
        T = len(label)
        out = np.zeros((self.n_buffers, T))
        out[0] = label
        if self.n_buffers > 1:
            if self.side == 'left':
                out[1:] = self._ramp(dist_left, cap_left)
            elif self.side == 'right':
                out[1:] = self._ramp(dist_right, cap_right)
            else:
                out[1:] = self._ramp(dist_left, cap_left)
                np.maximum(out[1:], self._ramp(dist_right, cap_right), out=out[1:])
        out[1:, label.astype(bool)] = 1
        return out

    def per_point_caps(self, label, start, end, b_left, b_right):
        """Spread per-anomaly caps to per-point caps, one array per side.

        A point's left ramp comes from the anomaly whose start is nearest
        ahead of it, so it takes that anomaly's left cap; its right ramp comes
        from the anomaly whose end is nearest behind it, so it takes that
        anomaly's right cap. This is the same pairing distance_from_anomaly
        makes, so the capped label is the max over anomalies of their own
        capped ramps. Used by AutoVUS to cap each anomaly at its own b*.
        """
        b_left = np.asarray(b_left, dtype=float)
        b_right = np.asarray(b_right, dtype=float)
        n = len(start)
        if len(b_left) != n or len(b_right) != n:
            raise ValueError(f"{len(b_left)}/{len(b_right)} caps for {n} anomalies")
        t = np.arange(len(label))
        ahead = np.minimum(np.searchsorted(start, t, side='left'), n - 1)      # first start >= t
        behind = np.maximum(np.searchsorted(end, t, side='right') - 1, 0)     # last end <= t
        return b_left[ahead], b_right[behind]

    # ------------------------------------------------------------------
    # Step 5: thresholds and score mask
    # ------------------------------------------------------------------
    @timed
    def thresholds(self, buffered, score):
        """The score values at which the VUS-PR surface can change, descending.

        Under stepwise integration a threshold only matters if crossing it
        adds true positives for some buffer, i.e. if the point it admits has
        a non-zero value in the widest buffered label. Thresholds that admit
        only plain normal points are dropped. Tied scores are handled by
        keeping the last occurrence, so all tied points count at once.
        If more than max_thresholds survive, they are subsampled uniformly
        in cumulative-TP space (worst-case AUC error 1 / max_thresholds).
        """
        fuzzy = buffered.max(axis=0)
        order = np.argsort(score)[::-1]
        score_sorted = score[order]
        distinct = np.r_[score_sorted[:-1] != score_sorted[1:], True]
        tp = np.cumsum(fuzzy[order])[distinct]
        keep = np.diff(tp, prepend=0.0) > 0
        thresholds = score_sorted[distinct][keep]

        if self.max_thresholds is not None and len(thresholds) > self.max_thresholds:
            tp_kept = tp[keep]
            targets = np.linspace(0.0, tp_kept[-1], self.max_thresholds + 1)[1:]
            idx = np.searchsorted(tp_kept, targets, side='left')
            idx = np.unique(np.clip(idx, 0, len(tp_kept) - 1))
            thresholds = thresholds[idx]
        return thresholds

    @timed
    def score_mask(self, score, thresholds):
        """Boolean (n_thresholds, T): score[t] >= threshold."""
        return score >= thresholds[:, None]

    @timed
    def false_positives_outside(self, score_outside, thresholds):
        """Per-threshold count of points outside the global mask at or above it.

        Those points are all normal, so they are false positives for every
        buffer. Counted with a sort and a search instead of a full mask.
        """
        score_sorted = np.sort(score_outside)
        return len(score_outside) - np.searchsorted(score_sorted, thresholds, side='left')

    # ------------------------------------------------------------------
    # Step 6: existence reward
    # ------------------------------------------------------------------
    @timed
    def existence(self, buffered, score, thresholds, start, end, cap_left=None, cap_right=None):
        """Fraction of anomalies detected, shape (n_buffers, n_thresholds).

        An anomaly is detected at a threshold if the max score inside its
        buffered region reaches it. The region of anomaly k at buffer b is
        [start - reach_left, end + reach_right], the same one the buffered
        label covers: reach = min(b, cap) on each rewarded side, 0 on a side
        that self.side excludes. For each anomaly the max is located by part
        (inside, left buffer, right buffer); once a buffer detects it, every
        wider buffer does too, so only the narrower buffers are revisited.
        Anomalies whose regions meet merge into one: the count per buffer
        drops and a merged anomaly is detected if any of its parts is.

        cap_left, cap_right: per-anomaly caps (one value per anomaly) or None.
        """
        n_buffers, T = buffered.shape
        n_anomalies_total = len(start)
        n_thresholds = len(thresholds)
        thresholds_asc = thresholds[::-1]
        existence = np.zeros((n_buffers, n_thresholds))
        b = self.buffers[:, None]
        reach_left = (b if cap_left is None else np.minimum(b, np.asarray(cap_left, dtype=int)[None, :])) \
            if self.side in ('both', 'left') else np.zeros((n_buffers, n_anomalies_total), dtype=int)
        reach_right = (b if cap_right is None else np.minimum(b, np.asarray(cap_right, dtype=int)[None, :])) \
            if self.side in ('both', 'right') else np.zeros((n_buffers, n_anomalies_total), dtype=int)
        starts = np.maximum(start[None, :] - reach_left, 0)      # (n_buffers, n_anomalies)
        ends = np.minimum(end[None, :] + reach_right, T - 1)

        # Buffer index from which anomaly k and k+1 share one region (n_buffers = never).
        n_anomalies = np.full(n_buffers, n_anomalies_total, dtype=int)
        merge_from = np.full(n_anomalies_total, n_buffers, dtype=int)
        for k in range(n_anomalies_total - 1):
            meets = ends[:, k] >= starts[:, k + 1]
            if meets.any():
                index = int(np.argmax(meets))
                n_anomalies[index:] -= 1
                merge_from[k] = index

        def threshold_index(value):
            return n_thresholds - np.searchsorted(thresholds_asc, value, side='right')

        prev = np.zeros((n_buffers, n_thresholds), dtype=int)
        for k in range(n_anomalies_total):
            s, e = start[k], end[k]
            curr = np.zeros((n_buffers, n_thresholds), dtype=int)
            i = n_buffers - 1
            while i >= 0:
                left = score[starts[i, k]:s + 1][::-1]
                inside = score[s:e + 1]
                right = score[e:ends[i, k] + 1]
                arg_l, arg_in, arg_r = left.argmax(), inside.argmax(), right.argmax()
                max_l, max_in, max_r = left[arg_l], inside[arg_in], right[arg_r]
                if max_in >= max_l and max_in >= max_r:            # found inside the anomaly
                    curr[:i + 1, threshold_index(max_in):] += 1
                    break
                elif max_l > max_r:                                # found in the left buffer
                    first = math.ceil(arg_l / self.step)
                    curr[first:i + 1, threshold_index(max_l):] += 1
                    i = first - 1
                else:                                              # found in the right buffer
                    first = math.ceil(arg_r / self.step)
                    curr[first:i + 1, threshold_index(max_r):] += 1
                    i = first - 1

            merged = np.zeros_like(curr)
            m = merge_from[k - 1]           # the last anomaly never merges forward
            if m < n_buffers:
                merged[m:] = np.logical_or(curr[m:], prev[m:])
                curr[m:] = np.logical_and(curr[m:], np.logical_not(prev[m:]))
            existence += curr
            prev = np.logical_or(merged, curr)

        return existence / n_anomalies[:, None]

    # ------------------------------------------------------------------
    # Step 7: confusion matrix
    # ------------------------------------------------------------------
    @timed
    def confusion_matrix(self, buffered, score_mask):
        """True positives, false positives and positives per (buffer, threshold).

        TP is the dot product of the buffered label and the thresholded score.
        It splits into the anomaly points, which are 1 in every buffer (one
        sum shared by all buffers), and the buffer points (one matmul over
        the buffer columns only). Positives are the mean of TP + FN and the
        number of anomaly points, so recall stays at most 1.
        """
        label = buffered[0]
        inside = np.where(label)[0]
        if self.global_mask:
            in_buffer = np.where(label == 0)[0]
        else:
            in_buffer = np.where((label == 0) & (buffered[-1] > 0))[0]

        tp_inside = score_mask[:, inside].sum(axis=1)                          # (n_thresholds,)
        tp = tp_inside + buffered[:, in_buffer] @ score_mask[:, in_buffer].T    # (n_buffers, n_thresholds)
        fn = (~score_mask)[:, inside].sum(axis=1)
        positives = (tp + fn + len(inside)) / 2
        fp = score_mask.sum(axis=1) - tp
        return tp, fp, positives

    # ------------------------------------------------------------------
    # Step 8: precision-recall curves and their area
    # ------------------------------------------------------------------
    @timed
    def precision_recall(self, tp, fp, positives, existence):
        """Precision and recall per buffer, each (n_buffers, n_thresholds + 1).

        The first column is the empty prediction (precision 1, recall 0).
        Recall is multiplied by the existence reward.
        """
        ones = np.ones((self.n_buffers, 1))
        zeros = np.zeros((self.n_buffers, 1))
        precision = np.hstack((ones, tp / (tp + fp)))
        recall = np.hstack((zeros, tp / positives))
        recall[recall > 1] = 1
        recall[:, 1:] *= existence
        return precision, recall

    @timed
    def auc(self, recall, precision):
        """Stepwise area under each PR curve, averaged over the buffers.

        Returns the mean over all buffers, or with cumulative=True the running
        mean: element i is VUS-PR with L = self.buffers[i].
        """
        area = np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)
        if self.cumulative:
            return np.cumsum(area) / np.arange(1, len(area) + 1)
        return area.mean()

    # ------------------------------------------------------------------
    # The metric
    # ------------------------------------------------------------------
    def compute(self, label, score, cap_left=None, cap_right=None):
        """VUS-PR of a score against a binary label.

        Args:
            label: 0/1 array of length T.
            score: anomaly score of length T, higher = more anomalous.
            cap_left, cap_right: optional per-anomaly buffer caps (one value
                per anomaly, in order), used by AutoVUS. Pass both or neither.

        Returns:
            float, or with cumulative=True an array with one VUS-PR per
            buffer length in self.buffers. Step times are in self.timings.
            self.label_mass holds the sum of the buffered label per buffer:
            label_mass[i] / T is the expected AUC-PR of a random detector at
            buffer self.buffers[i], which AutoVUS uses as its baseline.
        """
        if (cap_left is None) != (cap_right is None):
            raise ValueError("pass both cap_left and cap_right, or neither")
        self.timings = {}
        label = np.asarray(label)
        score = np.asarray(score, dtype=np.float64)

        _, (start, end) = self.anomaly_coordinates(label)
        mask = self.safe_mask(label, start, end, extra_safe=self.global_mask)
        if self.global_mask:
            score_outside = score[~mask]
            label, score = label[mask], score[mask]
            _, (start, end) = self.anomaly_coordinates(label)

        dist_left, dist_right = self.distance_from_anomaly(label, start, end)
        point_caps = (None, None)
        if cap_left is not None:
            point_caps = self.per_point_caps(label, start, end, cap_left, cap_right)
        buffered = self.buffered_labels(label, dist_left, dist_right, *point_caps)
        self.label_mass = buffered.sum(axis=1)

        thresholds = self.thresholds(buffered, score)
        score_mask = self.score_mask(score, thresholds)
        existence = self.existence(buffered, score, thresholds, start, end, cap_left, cap_right)
        tp, fp, positives = self.confusion_matrix(buffered, score_mask)
        if self.global_mask:
            fp = fp + self.false_positives_outside(score_outside, thresholds)

        precision, recall = self.precision_recall(tp, fp, positives, existence)
        return self.auc(recall, precision)
