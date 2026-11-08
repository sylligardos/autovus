"""FastVUS: exact CPU implementation of VUS-PR and VUS-ROC.

    from autovus import FastVUS
    vus = FastVUS(L=128)
    value = vus.compute(label, score)                 # VUS-PR
    value = vus.compute(label, score, metric='roc')   # VUS-ROC
    vus.timings   # seconds per step of the last compute()

L is the buffer length added on each side of every anomaly. The buffers
0, step, 2*step, ..., L are all evaluated; VUS-PR (VUS-ROC) is the mean
AUC-PR (AUC-ROC) over them.
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
                 cumulative=False, side='both', max_thresholds=2000, per_anomaly=False):
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
            per_anomaly (bool): also compute one local VUS-PR per anomaly
                (see local_scores) into self.local. The returned value is
                unchanged.
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
        self.per_anomaly = per_anomaly
        self.local = None       # per-anomaly results of the last compute(), see local_scores

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

    def _mean(self, area):
        """Mean over all buffers, or with cumulative=True the running mean:
        element i is the VUS with L = self.buffers[i]."""
        if self.cumulative:
            return np.cumsum(area) / np.arange(1, len(area) + 1)
        return area.mean()

    @timed
    def auc(self, recall, precision):
        """Stepwise area under each PR curve, averaged over the buffers (VUS-PR)."""
        area = np.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], axis=1)
        return self._mean(area)

    # ------------------------------------------------------------------
    # Step 9 (metric='roc' only): ROC curves and their area
    # ------------------------------------------------------------------
    @timed
    def tied_false_positives(self, buffered, score, thresholds, score_outside=None):
        """False positives admitted exactly at each threshold, (n_buffers, n_thresholds).

        A point whose score equals a threshold enters together with the
        points that threshold admits: the ROC curve moves diagonally there,
        not flat and then up. Counted as the points tied with the threshold
        minus their buffered label mass. Points outside the global mask are
        plain normals, so their tie count is added as is.
        """
        n = len(thresholds)
        pos = np.searchsorted(-thresholds, -score)                 # thresholds are descending
        hit = pos < n
        hit[hit] = thresholds[pos[hit]] == score[hit]
        n_tied = np.bincount(pos[hit], minlength=n).astype(np.float64)
        if score_outside is not None:
            s = np.sort(score_outside)
            n_tied += np.searchsorted(s, thresholds, side='right') - np.searchsorted(s, thresholds, side='left')
        tp_tied = np.zeros((n, self.n_buffers))
        np.add.at(tp_tied, pos[hit], buffered[:, hit].T)
        return n_tied[None, :] - tp_tied.T

    @timed
    def roc(self, recall, fp, positives, fp_tied, T, P):
        """Area under each ROC curve, averaged over the buffers (VUS-ROC).

        TPR is the existence-weighted recall, FPR = FP / (T - positives), with
        T the full series length and P the anomaly points (the positives
        before any threshold). The thresholds are the ones where the surface
        can change, so between two of them TPR is flat and the area is a
        rectangle at the previous TPR. At a threshold the curve moves
        diagonally, from the point just before it (the false positives
        admitted earlier, over the previous negatives count) to the point at
        it; the trapezoid rule credits that width at half the TPR jump. The
        curve closes flat at TPR 1 to (1, 1). This equals the trapezoid rule
        over every distinct score value, the VUS definition.
        """
        negatives = np.hstack((np.full((self.n_buffers, 1), float(T - P)), T - positives))
        fpr = np.hstack((np.zeros((self.n_buffers, 1)), fp / negatives[:, 1:]))
        before = (fp - fp_tied) / negatives[:, :-1]
        tpr = recall
        flat = np.sum((fpr[:, 1:] - fpr[:, :-1]) * tpr[:, :-1], axis=1) + (1 - fpr[:, -1]) * tpr[:, -1]
        diagonal = np.sum((fpr[:, 1:] - before) * (tpr[:, 1:] - tpr[:, :-1]) / 2, axis=1)
        return self._mean(flat + diagonal)

    # ------------------------------------------------------------------
    # Step 10 (per_anomaly=True only): one local VUS-PR per anomaly
    # ------------------------------------------------------------------
    @staticmethod
    def regions(start, end, T):
        """Region of every anomaly: halfway to its neighbours, [lo, hi).

        Same windows as AutoVUS.select: lo = (end[k-1] + start[k]) // 2,
        hi = (end[k] + start[k+1]) // 2; the regions tile the series.
        """
        mid = (end[:-1] + start[1:]) // 2
        lo = np.r_[0, mid].astype(int)
        hi = np.r_[mid, T].astype(int)
        return lo, hi

    @timed
    def own_labels(self, label, start, end, lo, hi, dist_left, dist_right, point_caps):
        """Buffered label of every point credited to the anomaly of its region only.

        Shape (n_buffers, T). A point before its region's anomaly takes that
        anomaly's left ramp, a point after it its right ramp, a point inside
        is 1. Unlike buffered_labels (max of both ramps), a neighbour's ramp
        reaching past the midpoint is not credited to anyone here.
        """
        T = len(label)
        out = np.zeros((self.n_buffers, T))
        out[0] = label
        if self.n_buffers > 1:
            rid = np.repeat(np.arange(len(start)), hi - lo)
            before = np.arange(T) < start[rid]
            after = np.arange(T) > end[rid]
            if self.side in ('both', 'left'):
                out[1:, before] = self._ramp(dist_left, point_caps[0])[:, before]
            if self.side in ('both', 'right'):
                out[1:, after] = self._ramp(dist_right, point_caps[1])[:, after]
            out[1:, label.astype(bool)] = 1
        return out

    @timed
    def local_existence(self, score, thresholds, start, end, lo, hi, cap_left=None, cap_right=None):
        """Per anomaly and buffer, the index of the first threshold at which
        the anomaly is detected, shape (n_anomalies, n_buffers).

        The window of anomaly k at buffer b is [start - reach_left, end +
        reach_right] clipped to its region [lo, hi): a neighbour's detection
        never counts. reach = min(b, cap) on a rewarded side, 0 otherwise.
        The anomaly is detected at threshold i (descending) when the max
        score in its window reaches thresholds[i]; n_thresholds = never.
        """
        K = len(start)
        n_thr = len(thresholds)
        thresholds_asc = thresholds[::-1]
        b = self.buffers
        caps_l = np.full(K, b[-1]) if cap_left is None else np.asarray(cap_left, dtype=int)
        caps_r = np.full(K, b[-1]) if cap_right is None else np.asarray(cap_right, dtype=int)
        out = np.empty((K, self.n_buffers), dtype=np.int64)
        for k in range(K):
            s, e = start[k], end[k]
            best = np.full(self.n_buffers, score[s:e + 1].max())
            if self.side in ('both', 'left') and s > lo[k]:
                left = np.maximum.accumulate(score[lo[k]:s][::-1])          # max at distance 1..
                reach = np.minimum(np.minimum(b, caps_l[k]), len(left))
                has = reach > 0
                best[has] = np.maximum(best[has], left[reach[has] - 1])
            if self.side in ('both', 'right') and e + 1 < hi[k]:
                right = np.maximum.accumulate(score[e + 1:hi[k]])
                reach = np.minimum(np.minimum(b, caps_r[k]), len(right))
                has = reach > 0
                best[has] = np.maximum(best[has], right[reach[has] - 1])
            out[k] = n_thr - np.searchsorted(thresholds_asc, best, side='right')
        return out

    def local_scores(self, label, score, thresholds, start, end, dist_left, dist_right,
                     point_caps, cap_left, cap_right, buffered, fp, T_full, mask, score_outside,
                     budget=4_000_000):
        """Local VUS-PR of every anomaly k (variant C), one value per buffer.

        For anomaly k at buffer b and threshold i:
          TP_k     own buffered label (own_labels) of the points of its region
                   at or above the threshold;
          recall   min(TP_k / pos_k, 1) * E_k, with pos_k = (TP_k + FN_k + a_k) / 2
                   (the positives of compute(), restricted to k) and E_k the
                   existence of k in its clipped window (local_existence);
          precision TP_k / (TP_k + FP / K), FP the false positives of the whole
                   series (from compute()), K the number of anomalies.
        area_k(b) is the stepwise area, like auc(). Random baseline per buffer:
        mass_k / (mass_k + (T - label_mass) / K). False-alarm cost: per point,
        the part of mean_k (1 - area_k) it causes, averaged over the buffers;
        sum(cost) = mean over k and b of (1 - area_k(b)).

        TP_k is a histogram of the points' own label at the first threshold
        that admits them, accumulated over the thresholds; anomalies are done
        in batches of at most `budget` (anomaly, buffer, threshold) cells.

        Returns a dict: start, end (original coordinates, inclusive), area and
        random (n_anomalies, n_buffers), vus and adjusted (running means over
        the buffers if cumulative, else means), cost (length T_full).
        """
        T = len(label)
        K = len(start)
        n_b, n_thr = self.n_buffers, len(thresholds)
        lo, hi = self.regions(start, end, T)
        own = self.own_labels(label, start, end, lo, hi, dist_left, dist_right, point_caps)
        j = self.local_existence(score, thresholds, start, end, lo, hi, cap_left, cap_right)
        idx = np.searchsorted(-thresholds, -score, side='left')    # first threshold admitting each point
        rid = np.repeat(np.arange(K), hi - lo)
        inside = label.astype(bool)
        in_buf = np.where((own[-1] > 0) & ~inside)[0]              # points with an own buffer value
        in_pts = np.where(inside)[0]
        a = (end - start + 1).astype(float)
        mass = np.add.reduceat(own, lo, axis=1).T                   # (K, n_b)
        label_mass = buffered.sum(axis=1)
        random = mass / (mass + (T_full - label_mass)[None, :] / K)
        fp_k = fp / K                                                # (n_b, n_thr)

        area = np.zeros((K, n_b))
        G = np.zeros((n_b, n_thr))          # sum over k of dR_k / (TP_k + FP / K), for the cost
        step = max(1, budget // (n_b * (n_thr + 1)))
        thr = np.arange(n_thr)
        for k0 in range(0, K, step):
            k1 = min(k0 + step, K)
            B = k1 - k0
            p = in_pts[(in_pts >= lo[k0]) & (in_pts < hi[k1 - 1])]
            hit = np.bincount((rid[p] - k0) * (n_thr + 1) + idx[p], minlength=B * (n_thr + 1))
            hit = np.cumsum(hit.reshape(B, n_thr + 1)[:, :n_thr], axis=1)        # anomaly points admitted
            q = in_buf[(in_buf >= lo[k0]) & (in_buf < hi[k1 - 1])]
            flat = (((rid[q] - k0)[None, :] * n_b + np.arange(n_b)[:, None]) * (n_thr + 1) + idx[q][None, :]).ravel()
            tp = np.bincount(flat, weights=own[:, q].ravel(), minlength=B * n_b * (n_thr + 1)).astype(float)   # empty input gives int
            tp = np.cumsum(tp.reshape(B, n_b, n_thr + 1)[:, :, :n_thr], axis=2) + hit[:, None, :]
            fn = a[k0:k1, None] - hit
            pos = (tp + fn[:, None, :] + a[k0:k1, None, None]) / 2
            exists = thr[None, None, :] >= j[k0:k1, :, None]
            recall = np.minimum(tp / pos, 1.0) * exists
            denom = tp + fp_k[None]
            precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
            d_recall = np.diff(recall, axis=2, prepend=0.0)
            area[k0:k1] = np.sum(d_recall * precision, axis=2)
            G += np.sum(np.divide(d_recall, denom, out=np.zeros_like(tp), where=denom > 0), axis=0)

        # false-alarm cost per point: sum over b and admitting thresholds of G * fp weight / (K * n_b)
        Gc = np.cumsum(G[:, ::-1], axis=1)[:, ::-1] / (K * K * n_b)   # thresholds i and lower
        Gc = np.hstack((Gc, np.zeros((n_b, 1))))                      # never admitted
        g1 = Gc.sum(axis=0)
        cost = g1[idx]
        support = np.where(buffered[-1] > 0)[0]                      # widest buffer covers every narrower one
        for c0 in range(0, len(support), 100_000):
            c = support[c0:c0 + 100_000]
            cost[c] -= np.einsum('bt,bt->t', Gc[:, idx[c]], buffered[:, c])
        cost_full = np.zeros(T_full)
        if mask is not None:
            cost_full[mask] = cost
            idx_out = np.searchsorted(-thresholds, -score_outside, side='left')
            cost_full[~mask] = g1[idx_out]
            where = np.where(mask)[0]
            start_o, end_o = where[start], where[end]
        else:
            cost_full[:] = cost
            start_o, end_o = start, end

        if self.cumulative:
            n = np.arange(1, n_b + 1)
            vus, rnd = np.cumsum(area, axis=1) / n, np.cumsum(random, axis=1) / n
        else:
            vus, rnd = area.mean(axis=1), random.mean(axis=1)
        return {
            'start': start_o, 'end': end_o,
            'area': area, 'random': random,
            'vus': vus, 'adjusted': (vus - rnd) / np.maximum(1.0 - rnd, 1e-10),
            'cost': cost_full,
        }

    # ------------------------------------------------------------------
    # The metric
    # ------------------------------------------------------------------
    def compute(self, label, score, cap_left=None, cap_right=None, metric='pr'):
        """VUS-PR or VUS-ROC of a score against a binary label.

        Args:
            label: 0/1 array of length T.
            score: anomaly score of length T, higher = more anomalous.
            cap_left, cap_right: optional per-anomaly buffer caps (one value
                per anomaly, in order), used by AutoVUS. Pass both or neither.
            metric: 'pr' (VUS-PR), 'roc' (VUS-ROC) or 'both' (a tuple of the two).

        Returns:
            float, or with cumulative=True an array with one value per
            buffer length in self.buffers. Step times are in self.timings.
            self.label_mass holds the sum of the buffered label per buffer:
            label_mass[i] / T is the expected AUC-PR of a random detector at
            buffer self.buffers[i], which AutoVUS uses as its baseline.
        """
        if (cap_left is None) != (cap_right is None):
            raise ValueError("pass both cap_left and cap_right, or neither")
        if metric not in ('pr', 'roc', 'both'):
            raise ValueError(f"metric={metric!r} must be 'pr', 'roc' or 'both'")
        self.timings = {}
        label = np.asarray(label)
        score = np.asarray(score, dtype=np.float64)
        if len(label) != len(score):
            raise ValueError(f"label has {len(label)} points, score {len(score)}")
        if not label.any():
            raise ValueError("label has no anomaly: VUS is undefined")
        if metric != 'pr' and label.all():
            raise ValueError("label has no normal point: VUS-ROC is undefined")
        T = len(label)

        _, (start, end) = self.anomaly_coordinates(label)
        mask = self.safe_mask(label, start, end, extra_safe=self.global_mask)
        score_outside = None
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

        self.local = None
        if self.per_anomaly and metric != 'roc':
            self.local = self.local_scores(label, score, thresholds, start, end, dist_left, dist_right,
                                           point_caps, cap_left, cap_right, buffered, fp, T,
                                           mask if self.global_mask else None, score_outside)

        precision, recall = self.precision_recall(tp, fp, positives, existence)
        if metric == 'pr':
            return self.auc(recall, precision)
        fp_tied = self.tied_false_positives(buffered, score, thresholds, score_outside)
        roc = self.roc(recall, fp, positives, fp_tied, T, int(label.sum()))
        if metric == 'roc':
            return roc
        return self.auc(recall, precision), roc
