"""FastVUSGPU: vectorized torch implementation of VUS-PR.

    from vus.fastvus_gpu import FastVUSGPU
    vus = FastVUSGPU(L=128)            # device: cuda if available, else cpu
    value = vus.compute(label, score)
    vus.timings                         # seconds per step of the last compute()

Same steps and conventions as FastVUS (see fastvus.py), with two differences:
  * the series is processed in chunks that fit the device memory, split at
    points no buffer can reach, and the counts are summed over chunks;
  * the existence reward is computed from the per-region maximum score
    (compute once, compare with every threshold), which merges two anomalies
    as soon as their buffered regions touch. FastVUS merges them when the
    buffers overlap by the exact VUS rule, so the two can differ slightly
    when anomalies lie an odd distance apart. This is the only approximation.

Set FASTVUS_PROFILE=1 to synchronize the device around every step, so that
self.timings is exact (slower; off by default).
"""
import functools
import math
import os
import time

import numpy as np
import psutil
import torch

PROFILE = os.environ.get('FASTVUS_PROFILE', '0') == '1'


def timed(method):
    """Add the method's wall time to self.timings[<method name>].

    CUDA runs asynchronously, so without FASTVUS_PROFILE=1 a step's time is
    only its launch cost and the real work is charged to the next step that
    waits for the device.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        sync = PROFILE and self.device.type == 'cuda'
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = method(self, *args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        self.timings[method.__name__] = self.timings.get(method.__name__, 0.0) + time.perf_counter() - start
        return result
    return wrapper


class FastVUSGPU:
    def __init__(self, L=100, step=1, zita=1 / math.sqrt(2), global_mask=True,
                 cumulative=False, side='both', max_thresholds=2000, device=None):
        """Same arguments as FastVUS, plus device ('cuda', 'cpu' or None = auto)."""
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
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        buffers = np.arange(0, L + 1, step) if L > 0 else np.array([0])
        self.buffers = torch.tensor(buffers, device=self.device)
        self.n_buffers = len(buffers)
        self.timings = {}
        self.label_mass = None       # sum of the buffered label per buffer, from the last compute() (numpy)
        self.memory_budget = None    # bytes one chunk may use, set per compute()
        self._warm_up()

    def _warm_up(self):
        """First torch calls pay a one-off setup cost; keep it out of timings."""
        label = torch.zeros(64, dtype=torch.int8, device=self.device)
        label[20:25] = 1
        self.compute(label, torch.rand(64, device=self.device))
        self.timings = {}

    def _tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return torch.tensor(np.asarray(x), dtype=dtype, device=self.device)

    def _update_memory_budget(self, fraction=1 / 50):
        """A fraction of the memory currently free on the device."""
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
        else:
            free = psutil.virtual_memory().available
        self.memory_budget = free * fraction

    # ------------------------------------------------------------------
    # Step 1: anomaly coordinates
    # ------------------------------------------------------------------
    @timed
    def anomaly_coordinates(self, label):
        """Start and end index of every anomaly, as two pairs of tensors.

        Returns ((start_inner, end_inner), (start, end)); see FastVUS.
        """
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        if label.numel() == 0:
            return (empty, empty), (empty, empty)
        diff = torch.diff(label)
        start_inner = torch.where(diff == 1)[0] + 1
        end_inner = torch.where(diff == -1)[0]
        first = torch.tensor([0], device=self.device)
        last = torch.tensor([len(label) - 1], device=self.device)
        start = torch.cat((first, start_inner)) if label[0] else start_inner
        end = torch.cat((end_inner, last)) if label[-1] else end_inner
        if start.shape != end.shape:
            raise ValueError(f"Unbalanced anomaly starts and ends: {start} vs {end}")
        return (start_inner, end_inner), (start, end)

    # ------------------------------------------------------------------
    # Step 2: global mask
    # ------------------------------------------------------------------
    @timed
    def safe_mask(self, label, start, end, extra_safe=False):
        """Boolean mask of the points that any buffer up to L can reach (see FastVUS)."""
        T = len(label)
        reach = self.L + 1 + int(extra_safe)
        start_safe = torch.clamp(start - reach, min=0)
        end_safe = torch.clamp(end + reach + 1, max=T - 1)

        delta = torch.zeros(T, dtype=torch.int8, device=self.device)
        delta.index_add_(0, start_safe, torch.ones_like(start_safe, dtype=torch.int8))
        delta.index_add_(0, end_safe, -torch.ones_like(end_safe, dtype=torch.int8))
        mask = delta.cumsum(0) > 0
        # TODO (D2): the last point is set by hand; verify the no-anomaly case end to end.
        mask[-1] = (end_safe.numel() > 0 and bool(end_safe[-1] == T - 1)) if extra_safe else bool(label[-1])
        return mask

    # ------------------------------------------------------------------
    # Step 3: distance of every point from the anomalies
    # ------------------------------------------------------------------
    @timed
    def distance_from_anomaly(self, label, start, end):
        """Per-point distance to the nearest anomaly boundary, one tensor per side (see FastVUS).

        The (T, n_anomalies) distance matrix is built in slices of anomalies
        when it would exceed the memory budget.
        """
        T = len(label)
        inf = float('inf')
        dist_left = torch.full((T,), inf, device=self.device)
        dist_right = torch.full((T,), inf, device=self.device)
        if start.numel() == 0:
            return dist_left, dist_right

        t = torch.arange(T, device=self.device)[:, None]
        per_anomaly = T * 4 * 2
        slice_size = max(1, int(self.memory_budget // per_anomaly))
        for s, e in zip(start.split(slice_size), end.split(slice_size)):
            to_start = (s[None, :] - t).float()
            dist_left = torch.minimum(dist_left, torch.where(to_start >= 0, to_start, inf).min(dim=1).values)
            from_end = (t - e[None, :]).float()
            dist_right = torch.minimum(dist_right, torch.where(from_end >= 0, from_end, inf).min(dim=1).values)
        return dist_left, dist_right

    # ------------------------------------------------------------------
    # Step 4: buffered labels
    # ------------------------------------------------------------------
    def _ramp(self, dist, cap, buffers=None):
        """Ramp of one side for the given buffers (default step..L), shape (len(buffers), T); see FastVUS._ramp."""
        b = (self.buffers[1:] if buffers is None else buffers)[:, None].float()
        d = dist[None, :]
        b_eff = b if cap is None else torch.minimum(b, cap[None, :])
        reached = (b_eff >= d) & (b_eff > 0)
        coef = (1.0 - self.zita ** 2) / b_eff
        return torch.where(reached, torch.sqrt(1.0 - coef * d), 0.0)

    @timed
    def buffered_labels(self, label, dist_left, dist_right, cap_left=None, cap_right=None):
        """Label with buffers added, shape (n_buffers, T), float32; see FastVUS."""
        out = torch.zeros((self.n_buffers, len(label)), device=self.device)
        out[0] = label
        if self.n_buffers > 1:
            if self.side == 'left':
                out[1:] = self._ramp(dist_left, cap_left)
            elif self.side == 'right':
                out[1:] = self._ramp(dist_right, cap_right)
            else:
                out[1:] = torch.maximum(self._ramp(dist_left, cap_left), self._ramp(dist_right, cap_right))
        out[1:, label.bool()] = 1
        return out

    @timed
    def widest_label(self, label, dist_left, dist_right, cap_left=None, cap_right=None):
        """The buffered label of the widest buffer alone, shape (T,): row -1 of buffered_labels."""
        widest = self.buffers[-1:]
        if self.side == 'left':
            out = self._ramp(dist_left, cap_left, widest)[0]
        elif self.side == 'right':
            out = self._ramp(dist_right, cap_right, widest)[0]
        else:
            out = torch.maximum(self._ramp(dist_left, cap_left, widest)[0], self._ramp(dist_right, cap_right, widest)[0])
        out[label.bool()] = 1
        return out

    def per_point_caps(self, label, start, end, b_left, b_right):
        """Spread per-anomaly caps to per-point caps, one tensor per side; see FastVUS."""
        b_left = self._tensor(b_left, torch.float32)
        b_right = self._tensor(b_right, torch.float32)
        n = start.numel()
        if b_left.numel() != n or b_right.numel() != n:
            raise ValueError(f"{b_left.numel()}/{b_right.numel()} caps for {n} anomalies")
        t = torch.arange(len(label), device=self.device)
        ahead = torch.clamp(torch.searchsorted(start, t, side='left'), max=n - 1)
        behind = torch.clamp(torch.searchsorted(end, t, side='right') - 1, min=0)
        return b_left[ahead], b_right[behind]

    # ------------------------------------------------------------------
    # Step 5: thresholds and score mask
    # ------------------------------------------------------------------
    @timed
    def thresholds(self, widest, score):
        """Score values at which the surface can change, descending; see FastVUS.thresholds.

        widest is the buffered label of the widest buffer (widest_label), the
        same quantity FastVUS reads off its buffered labels, so both engines
        pick the same thresholds.
        """
        fuzzy = widest.to(score.dtype)
        order = torch.argsort(score, descending=True)
        score_sorted = score[order]
        last = torch.ones(1, dtype=torch.bool, device=self.device)
        distinct = torch.cat((score_sorted[:-1] != score_sorted[1:], last))
        tp = torch.cumsum(fuzzy[order], dim=0)[distinct]
        keep = torch.diff(tp, prepend=torch.zeros(1, dtype=tp.dtype, device=self.device)) > 0
        thresholds = score_sorted[distinct][keep]

        if self.max_thresholds is not None and thresholds.numel() > self.max_thresholds:
            tp_kept = tp[keep]
            targets = torch.linspace(0.0, float(tp_kept[-1]), self.max_thresholds + 1,
                                     device=self.device, dtype=tp_kept.dtype)[1:]
            idx = torch.searchsorted(tp_kept, targets, side='left')
            idx = torch.unique(torch.clamp(idx, 0, tp_kept.numel() - 1))
            thresholds = thresholds[idx]
        return thresholds

    @timed
    def score_mask(self, score, thresholds):
        """Boolean (n_thresholds, T): score[t] >= threshold."""
        return score >= thresholds[:, None]

    @timed
    def false_positives_outside(self, score_outside, thresholds):
        """Per-threshold count of points outside the global mask at or above it."""
        score_sorted, _ = torch.sort(score_outside)
        return score_outside.numel() - torch.searchsorted(score_sorted, thresholds, side='left')

    # ------------------------------------------------------------------
    # Step 6: existence reward
    # ------------------------------------------------------------------
    @timed
    def existence(self, buffered, score, thresholds):
        """Anomalies found per (buffer, threshold) and anomalies per buffer.

        Returns (found, total): found has shape (n_buffers, n_thresholds),
        total (n_buffers, 1). Regions are the runs of non-zero buffered label
        in each row; a region is found at a threshold if its maximum score
        reaches it. The threshold comparison is a searchsorted over the
        sorted region maxima, so memory stays (n_buffers, n_regions).
        Division is left to the caller so that chunks can be summed first.
        """
        n_buffers, T = buffered.shape
        on = (buffered > 0).int()
        starts = torch.clamp(torch.diff(on, dim=1, prepend=torch.zeros((n_buffers, 1), dtype=on.dtype, device=self.device)), 0, 1)
        region = on * torch.cumsum(starts, dim=1)         # (n_buffers, T), 0 outside regions
        total = torch.cumsum(starts, dim=1)[:, -1:]       # regions per buffer, (n_buffers, 1)

        n_regions = int(region.max())
        if n_regions == 0:
            return torch.zeros((n_buffers, len(thresholds)), device=self.device), total

        region_max = torch.full((n_buffers, n_regions + 1), float('-inf'), device=self.device, dtype=score.dtype)
        region_max.scatter_reduce_(1, region.long(), score[None, :].expand(n_buffers, T), reduce='amax', include_self=True)
        region_max, _ = torch.sort(region_max[:, 1:], dim=1)                 # drop bin 0, ascending
        below = torch.searchsorted(region_max, thresholds[None, :].expand(n_buffers, -1).contiguous(), right=False)
        found = (n_regions - below).float()
        return found, total

    # ------------------------------------------------------------------
    # Step 7: confusion matrix
    # ------------------------------------------------------------------
    @timed
    def confusion_matrix(self, buffered, score_mask):
        """True positives, false positives and positives per (buffer, threshold); see FastVUS."""
        label = buffered[0]
        inside = torch.where(label)[0]
        if self.global_mask:
            in_buffer = torch.where(label == 0)[0]
        else:
            in_buffer = torch.where((label == 0) & (buffered[-1] > 0))[0]

        tp_inside = score_mask[:, inside].sum(dim=1)
        tp = tp_inside + buffered[:, in_buffer] @ score_mask[:, in_buffer].float().T
        fn = (~score_mask)[:, inside].sum(dim=1)
        positives = (tp + fn + inside.numel()) / 2
        fp = score_mask.sum(dim=1) - tp
        return tp, fp, positives

    # ------------------------------------------------------------------
    # Step 8: precision-recall curves and their area
    # ------------------------------------------------------------------
    @timed
    def precision_recall(self, tp, fp, positives, existence):
        """Precision and recall per buffer, each (n_buffers, n_thresholds + 1); see FastVUS."""
        ones = torch.ones((self.n_buffers, 1), device=self.device)
        zeros = torch.zeros((self.n_buffers, 1), device=self.device)
        precision = torch.cat((ones, tp / (tp + fp)), dim=1)
        recall = torch.cat((zeros, tp / positives), dim=1)
        recall[recall > 1] = 1
        recall[:, 1:] *= existence
        return precision, recall

    @timed
    def auc(self, recall, precision):
        """Stepwise area under each PR curve, mean or running mean over buffers; see FastVUS."""
        area = torch.sum((recall[:, 1:] - recall[:, :-1]) * precision[:, 1:], dim=1)
        if self.cumulative:
            return torch.cumsum(area, 0) / torch.arange(1, len(area) + 1, device=self.device)
        return area.mean()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def chunk_bounds(self, label, mask, n_thresholds, n_anomalies):
        """End index of each chunk, chosen outside the mask so no buffer is cut.

        The number of chunks comes from the memory the per-chunk tensors need
        (buffered labels, score mask, region maxima) against the budget.
        """
        T = len(label)
        needed = self.n_buffers * T * 4 + n_thresholds * T + self.n_buffers * (n_anomalies + n_thresholds) * 8
        n_chunks = int(min(max(math.ceil(needed / self.memory_budget), 1), T))
        if n_chunks == 1:
            return [T]

        outside = torch.nonzero(~mask).squeeze(1)
        if outside.numel() == 0:
            return [T]
        ideal = torch.linspace(0, T - 1, steps=n_chunks + 1, device=self.device)[1:-1]
        nearest = torch.abs(outside[:, None] - ideal[None, :]).argmin(dim=0)
        bounds = [b for b in torch.unique(outside[nearest]).tolist() if 0 < b < T]
        return bounds + [T]

    # ------------------------------------------------------------------
    # The metric
    # ------------------------------------------------------------------
    def compute(self, label, score, cap_left=None, cap_right=None):
        """VUS-PR of a score against a binary label; same interface as FastVUS.compute.

        Returns a float, or with cumulative=True a numpy array with one VUS-PR
        per buffer length in self.buffers. See FastVUS.compute for timings
        and label_mass.
        """
        if (cap_left is None) != (cap_right is None):
            raise ValueError("pass both cap_left and cap_right, or neither")
        self.timings = {}
        self._update_memory_budget()
        label = self._tensor(label, torch.int8)
        score = self._tensor(score, torch.float64)
        if label.numel() == 0:
            raise ValueError("empty label")

        _, (start, end) = self.anomaly_coordinates(label)
        mask = self.safe_mask(label, start, end, extra_safe=self.global_mask)
        if self.global_mask:
            score_outside = score[~mask]
            label, score = label[mask], score[mask]
            _, (start, end) = self.anomaly_coordinates(label)
            mask = self.safe_mask(label, start, end)

        dist_left, dist_right = self.distance_from_anomaly(label, start, end)
        if cap_left is not None:
            cap_left, cap_right = self.per_point_caps(label, start, end, cap_left, cap_right)
        thresholds = self.thresholds(self.widest_label(label, dist_left, dist_right, cap_left, cap_right), score)
        if self.global_mask:
            fp_outside = self.false_positives_outside(score_outside, thresholds)

        tp = fp = positives = found = total = 0
        label_mass = 0
        prev = 0
        for bound in self.chunk_bounds(label, mask, len(thresholds), start.numel()):
            label_c, score_c = label[prev:bound], score[prev:bound]
            caps_c = (None, None) if cap_left is None else (cap_left[prev:bound], cap_right[prev:bound])
            buffered = self.buffered_labels(label_c, dist_left[prev:bound], dist_right[prev:bound], *caps_c)
            label_mass = label_mass + buffered.sum(dim=1, dtype=torch.float64)
            score_mask = self.score_mask(score_c, thresholds)
            tp_c, fp_c, positives_c = self.confusion_matrix(buffered, score_mask)
            found_c, total_c = self.existence(buffered, score_c, thresholds)
            tp, fp, positives = tp + tp_c, fp + fp_c, positives + positives_c
            found, total = found + found_c, total + total_c
            prev = bound

        self.label_mass = label_mass.cpu().numpy()
        if self.global_mask:
            fp = fp + fp_outside
        precision, recall = self.precision_recall(tp, fp, positives, found / total)
        result = self.auc(recall, precision)
        return result.cpu().numpy() if self.cumulative else float(result)
