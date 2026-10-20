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
