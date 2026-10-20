"""FastVUS: exact CPU implementation of VUS-PR.

    from vus.fastvus import FastVUS
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
        self.timings = {}   # seconds per step, filled by compute()

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

        A point at distance d from the boundary gets 1 - (1 - zita) * d / b
        for every buffer b >= d, else 0. dist = inf (no boundary on this side)
        stays 0. cap (shape (T,) or None) limits each point's buffer to
        min(b, cap[t]); None means the uniform buffer.
        """
        # TODO (B2): linear ramp, to become the curved one once the engine matches the old code.
        T = len(dist)
        out = np.zeros((self.n_buffers - 1, T))
        b = self.buffers[1:, None]           # (n_buffers - 1, 1)
        d = dist[None, :]                    # (1, T)
        if cap is None:
            reached = b >= d
            np.subtract(1.0, (1.0 - self.zita) / b * d, out=out, where=reached)
        else:
            b_eff = np.minimum(b, cap[None, :])
            reached = (b_eff >= d) & (b_eff > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ramp = 1.0 - (1.0 - self.zita) * d / b_eff
            np.copyto(out, ramp, where=reached)
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

        Every point inherits the caps of its nearest anomaly (distance 0 when
        inside). Used by AutoVUS to cap each anomaly's buffer at its own b*.
        """
        b_left = np.asarray(b_left, dtype=float)
        b_right = np.asarray(b_right, dtype=float)
        n = len(start)
        if len(b_left) != n or len(b_right) != n:
            raise ValueError(f"{len(b_left)}/{len(b_right)} caps for {n} anomalies")
        t = np.arange(len(label))[:, None]
        dist = np.maximum(start[None, :] - t, t - end[None, :])
        np.clip(dist, 0, None, out=dist)
        nearest = np.argmin(dist, axis=1)
        return b_left[nearest], b_right[nearest]
