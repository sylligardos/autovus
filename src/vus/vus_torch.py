"""
@who: Anonymous
@where: Anonymous
@when: Anonymous
@what: FF-VUS
"""


import torch
import numpy as np
from tqdm import tqdm
import psutil


    
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

class VUSTorch():
    def __init__(
            self, 
            slope_size=100, 
            step=1, 
            zita=(1/torch.sqrt(torch.tensor(2))), 
            global_mask=True,
            existence=True,
            conf_matrix='dynamic_plus',
            device=None,
            cumulative=False,
            aggregation='mean',
            thresholds='pr_relevant_fuzzy',
            max_thresholds=None,
        ):
        """
        Initialize the torch version of the VUS metric.

        Args:
            TODO: Write the arguments description when done
        """
        if slope_size < 0 or step < 0:
            raise ValueError(f"Error with the slope_size {slope_size} or step {step}. They should be positive values.")
        if step == 0 and slope_size > 0:
            raise ValueError(f"Error with step {step} and slope_size {slope_size}. Step can't be 0 with non-zero slope_size.")
        if (step > 0 and slope_size > 0) and (slope_size % step) != 0:
            raise ValueError(f"Error with the step: {step}. It should be a positive value and a perfect divider of the slope_size.")
        if zita < 0 or zita > 1:
            raise ValueError(f"Error with the zita: {zita}. It should be a value between 0 and 1.")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        if self.device == 'cpu':
            print('You are using the GPU version of VUS on a CPU. Performance will be limited!')

        self.slope_size, self.step, self.zita, self.global_mask = slope_size, step, zita, global_mask    
        self.existence = False if existence == 'None'else existence
        self.cumulative = cumulative
        
        self.slope_values = torch.arange(
            start=0, 
            end=self.slope_size + 1, 
            step=self.step, 
            device=self.device, 
            dtype=torch.int16 if self.slope_size < 32000 else torch.int32 
        ) if self.slope_size > 0 else torch.tensor([0], device=self.device)
        self.n_slopes = self.slope_values.shape[0]

        conf_matrix_args = ['dynamic', 'dynamic_plus']
        if conf_matrix in conf_matrix_args:
            self.conf_matrix_mode = conf_matrix
        else:
            raise ValueError(f"Unknown argument for conf_matrix: {conf_matrix}. 'conf_matrix' should be one of {conf_matrix_args}")

        aggregation_args = ['mean', 'geometric', 'harmonic']
        if aggregation in aggregation_args:
            self.aggregation_mode = aggregation
        else:
            raise ValueError(f"Unknown argument for aggregation: {aggregation}. 'aggregation' should be one of {aggregation_args}")

        thresholds_args = ['all_unique', 'subsampled', 'pr_relevant_fuzzy']
        if thresholds in thresholds_args:
            self.thresholds_mode = thresholds
        else:
            raise ValueError(f"Unknown argument for thresholds: {thresholds}. 'thresholds' should be one of {thresholds_args}")
        self.max_thresholds = max_thresholds

        self.warmup()

    def warmup(self):
        '''
        Torch first computation is slower than the rest because of JIT, etc. This function does a dummy computation
        to remove this overhead from the first actual computation.
        '''
        label = torch.randint(0, 2, (20,), dtype=torch.int8, device=self.device)
        score = torch.rand(20, device=self.device)
        self.compute(label, score)


    def update_max_memory_tokens(self, divider=50):
        """
        Update self.max_memory_tokens according to the currently available memory.

        Changing the divider makes a little difference
       """
        if self.device == 'cuda':
            available_memory, _ = torch.cuda.mem_get_info()
            self.max_memory_tokens = available_memory / divider
        else:
            available_memory = psutil.virtual_memory().available
            self.max_memory_tokens = available_memory / divider

    def _to_tensor(self, arr, dtype):
        """Convert numpy array or plain tensor to a tensor on the correct device and dtype."""
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=dtype)
        return torch.tensor(arr, dtype=dtype, device=self.device)

    @time_it
    def compute(self, label, score):
        """
        The main computing function of the metric
        
        TODO: There is still some error on MITDB two first time series, CPU version too
        """
        label = self._to_tensor(label, dtype=torch.int8)
        score = self._to_tensor(score, dtype=torch.float32)
        if label.numel() == 0:
            raise ValueError("label is empty — cannot compute VUS on an empty sequence")
        self.update_max_memory_tokens()
        fp = tp = positives = anomalies_found = total_anomalies = time_sm = time_slopes = time_existence = time_confusion = time_pos = 0

        ((_), (start_with_edges, end_with_edges)), time_anomalies_coord = self.get_anomalies_coordinates(label)
        safe_mask, time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges, extra_safe=self.global_mask)
        thresholds, time_thresholds = self._dispatch_thresholds(label, score, safe_mask)

        # Apply global mask
        if self.global_mask:
            nonmask_score = score[~safe_mask]
            extra_fp = torch.zeros((thresholds.shape[0]), device=self.device)

            # The score mask materialised here is (n_thresholds, chunk) so the
            # safe chunk length must scale INVERSELY with the number of
            # thresholds. process_in_chunks assumes ~100 thresholds, which
            # silently overflows VRAM for the AUC variant (uncapped thresholds,
            # tens of thousands on long series). Size the chunk from the actual
            # threshold count instead.
            n_thresholds = thresholds.shape[0]
            chunk_len = max(1, int(self.max_memory_tokens // max(1, n_thresholds)))
            for start in tqdm(range(0, nonmask_score.shape[0], chunk_len),
                              desc='Computing extra FPs', leave=False):
                chunk = nonmask_score[start:start + chunk_len]
                sm, extra_time_sm = self.get_score_mask(chunk, thresholds)
                extra_fp += sm.sum(axis=1)
                time_sm += extra_time_sm

            label, score = label[safe_mask], score[safe_mask]
            ((_), (start_with_edges, end_with_edges)), extra_time_anomalies_coord = self.get_anomalies_coordinates(label)
            safe_mask, extra_time_safe_mask = self.create_safe_mask(label, start_with_edges, end_with_edges)
            time_anomalies_coord += extra_time_anomalies_coord
            time_safe_mask += extra_time_safe_mask

        # Number of chunks required to fit into memory. Peak per-chunk memory is
        # dominated by the (n_slopes x chunk) slope labels AND the
        # (n_thresholds x chunk) score mask; both dimensions must drive the split
        # count, otherwise a large threshold set overflows VRAM even when
        # n_slopes is tiny (the AUC variant has n_slopes == 1).
        labels_mem_size = self.n_slopes * len(label) * 4
        score_mask_mem_size = thresholds.shape[0] * len(label)
        mem_size = labels_mem_size + score_mask_mem_size
        n_splits = np.ceil(mem_size / self.max_memory_tokens).astype(int)
        n_splits = int(min(max(n_splits, 1), len(label)))
        split_points = self.find_safe_splits(label, safe_mask, n_splits)

        # Computation in chunks
        prev_split = 0
        for curr_split in tqdm(split_points, desc='In distance from anomaly', disable=(n_splits <= 10), leave=False):
            label_c = label[prev_split:curr_split]
            score_c = score[prev_split:curr_split]
            safe_mask_c = safe_mask[prev_split:curr_split]
            
            # Preprocessing
            ((_), (start_with_edges_c, end_with_edges_c)), chunk_time_anomalies_coord = self.get_anomalies_coordinates(label_c)
            pos, chunk_time_pos = self.distance_from_anomaly(label_c, start_with_edges_c, end_with_edges_c)
            sm, chunk_time_sm = self.get_score_mask(score_c, thresholds)

            # Main computation
            labels_c, chunk_time_slope = self.add_slopes(label_c, pos)
            (anomalies_found_c, total_anomalies_c), chunk_time_existence = self.compute_existence(labels_c, sm, safe_mask_c, normalize=False)
            (fp_c, fn_c, tp_c, positives_c, neg_c, fpr_c), chunk_time_conf = self.compute_confusion_matrix(labels_c, sm)

            # Accumulate results and timing from each chunk
            tp += tp_c
            fp += fp_c
            positives += positives_c
            anomalies_found += anomalies_found_c
            total_anomalies += total_anomalies_c
            time_anomalies_coord += chunk_time_anomalies_coord
            time_pos += chunk_time_pos
            time_sm += chunk_time_sm
            time_slopes += chunk_time_slope
            time_existence += chunk_time_existence
            time_confusion += chunk_time_conf

            prev_split = curr_split

        # Combine existence of all chunks
        existence = anomalies_found / total_anomalies if self.existence else torch.ones((self.n_slopes, thresholds.shape[0]), device=self.device)

        if self.global_mask:
            fp += extra_fp

        # After chunking
        (precision, recall), time_pr_rec = self.precision_recall_curve(tp, fp, positives, existence)
        vus_pr, time_integral = self.auc(recall, precision)

        time_analysis = {
            "Anomaly coordinates time": time_anomalies_coord,
            "Safe mask time": time_safe_mask,
            "Thresholds time": time_thresholds,
            "Score mask time": time_sm,
            "Position time": time_pos,
            "Slopes time": time_slopes,
            "Existence time": time_existence,
            "Confusion matrix time": time_confusion,
            "Precision recall curve time": time_pr_rec,
            "Integral time": time_integral,
        }

        return vus_pr, time_analysis

    def process_in_chunks(self, data, divider=100):
        '''
        Divider is used to simulate the number of thresholds, however some times
        we process single dimension arrays, as in unique_thresholds
        '''
        if len(data) == 0:
            return
        data_mem_size = data.nbytes * divider  # Around 100 thresholds 
        n_splits = np.ceil(data_mem_size / self.max_memory_tokens).astype(int)
        chunk_size = len(data) // n_splits

        for i in range(n_splits):
            start = i * chunk_size
            end = len(data) if i == n_splits - 1 else (i + 1) * chunk_size
            chunk = data[start:end]
            yield chunk

    def find_safe_splits(self, label, safe_mask, n_splits):
        """
        Finds approximately evenly spaced safe split points where pos is a local maximum
        and far enough from anomalies.
        """
        length = torch.tensor(label.shape[0], device=self.device)
        valid_splits_mask = ~safe_mask

        step = max(1, valid_splits_mask.sum().item() // (n_splits * 10))
        valid_splits = torch.nonzero(valid_splits_mask)[::step].squeeze(1)

        if valid_splits.numel() == 0 or n_splits < 1:
            return torch.tensor([length], device=self.device)

        ideal_splits = torch.linspace(0, length - 1, steps=n_splits + 1, device=self.device)[1:-1]        

        dists = torch.abs(valid_splits[:, None] - ideal_splits[None, :])
        min_indices = dists.argmin(dim=0)
        selected_splits = valid_splits[min_indices]
        selected_splits = torch.unique(selected_splits.sort().values, sorted=True)

        # Final safety check
        assert torch.all(label[selected_splits] != 1), "Some selected splits fall inside anomalies!"
        assert torch.all(safe_mask[selected_splits] == 0), "Some splits are too close to anomalies!"

        total_splits = torch.cat((selected_splits, length[None]), dim=0)
        return total_splits
    
    @time_it
    def get_score_mask(self, score, thresholds):
        return score >= thresholds[:, None]
    
    @time_it
    def get_unique_thresholds(self, score, compute_in_chunks=True, max_thresholds=200, debug=False):
        if not compute_in_chunks:
            sorted_thresholds, _ = torch.sort(torch.unique(score), descending=True)
        else:
            chunks = self.process_in_chunks(score, divider=5)
            all_unique = torch.empty(0, device=score.device, dtype=score.dtype)
            for chunk in chunks:
                chunk_unique = torch.unique(chunk)
                all_unique = torch.cat((all_unique, chunk_unique))
            thresholds = torch.unique(all_unique)
            sorted_thresholds, _ = torch.sort(thresholds, descending=True)

            if debug:
                sorted_thresholds_whole, _ = torch.sort(torch.unique(score), descending=True)
                if not torch.equal(sorted_thresholds, sorted_thresholds_whole):
                    raise ValueError("Chunked thresholds differ from full-sort thresholds.")

        if max_thresholds is not None and len(sorted_thresholds) > max_thresholds:
            k = len(sorted_thresholds) // max_thresholds
            sorted_thresholds = sorted_thresholds[::k]
        return sorted_thresholds

    @time_it
    def get_pr_relevant_thresholds_vus(self, label_slopes, score, min_area=1e-8,
                                       max_thresholds=1000):
        """
        VUS-aware threshold pruning (torch entry point, runs in numpy internally).

        Keeps only the thresholds at which the cumulative fuzzy-TP changes.
        Score levels that add only false positives leave recall unchanged and
        contribute zero stepwise area, so they can be dropped without
        affecting the resulting VUS value.

        Tied scores are handled by indexing the cumulative TP at the LAST
        occurrence of each unique score, so every tied true positive is
        counted (FIRST-occurrence silently drops tied TPs on flat score
        plateaus and yields near-zero VUS).

        When the exact set exceeds max_thresholds, TP-space uniform sampling
        is applied: N targets are placed evenly in [0, total_TP] and snapped
        to the nearest real threshold via searchsorted. Worst-case AUC error
        is bounded by 1/max_thresholds. Pass max_thresholds=None to disable.

        Matches the rule in vus_numpy.get_pr_relevant_thresholds_vus.

        label_slopes: float tensor of shape (n_slopes, T) — pass safe_mask.float().unsqueeze(0)
                      as the proxy when called from _dispatch_thresholds.
        score:        float tensor of shape (T,).
        min_area:     kept for backward-compatible signature; unused.
        Returns a 1-D descending torch tensor of thresholds.
        """
        ls_np = label_slopes.cpu().numpy()
        sc_np = score.cpu().numpy()

        fuzzy_label = ls_np.max(axis=0)

        order = np.argsort(sc_np)[::-1]
        score_sorted = sc_np[order]
        fuzzy_sorted = fuzzy_label[order]

        # LAST occurrence of each tied score.
        distinct     = np.r_[score_sorted[:-1] != score_sorted[1:], True]
        score_unique = score_sorted[distinct]

        # Cumulative fuzzy-TP; keep thresholds where it changes.
        tp = np.cumsum(fuzzy_sorted)[distinct]
        keep = np.diff(tp, prepend=0.0) > 0
        thresholds = score_unique[keep]

        # TP-space uniform sampling cap.
        if max_thresholds is not None and len(thresholds) > max_thresholds:
            tp_kept = tp[keep]
            total_tp = tp_kept[-1]
            targets = np.linspace(0.0, total_tp, max_thresholds + 1)[1:]
            idx = np.searchsorted(tp_kept, targets, side='left')
            idx = np.clip(idx, 0, len(tp_kept) - 1)
            idx = np.unique(idx)
            thresholds = thresholds[idx]

        return torch.tensor(thresholds, dtype=score.dtype, device=self.device)

    def _dispatch_thresholds(self, _label, score, safe_mask):
        if self.thresholds_mode == 'all_unique':
            return self.get_unique_thresholds(score, max_thresholds=None)
        elif self.thresholds_mode == 'subsampled':
            cap = self.max_thresholds if self.max_thresholds is not None else 200
            return self.get_unique_thresholds(score, max_thresholds=cap)
        elif self.thresholds_mode == 'pr_relevant_fuzzy':
            proxy = safe_mask.float().unsqueeze(0)  # (1, T)
            return self.get_pr_relevant_thresholds_vus(proxy, score, max_thresholds=self.max_thresholds)

    @time_it
    def get_anomalies_coordinates(self, label: torch.Tensor):
        """
        Return the starting and ending points of all anomalies in label,
        both with and without edge inclusion.

        Args:
            label (torch.tensor): vector of 1s and 0s
        Returns:
            ((start_no_edges, end_no_edges), (start_with_edges, end_with_edges))
        """
        device = label.device
        empty = torch.tensor([], dtype=torch.long, device=device)

        if label.numel() == 0:
            return (empty, empty), (empty, empty)

        diff = torch.diff(label)
        start_no_edges = torch.where(diff == 1)[0] + 1
        end_no_edges = torch.where(diff == -1)[0]

        start_with_edges = torch.cat((torch.tensor([0], device=device), start_no_edges)) if label[0] else start_no_edges
        end_with_edges = torch.cat((end_no_edges, torch.tensor([len(label) - 1], device=device))) if label[-1] else end_no_edges

        if start_with_edges.shape != end_with_edges.shape:
            raise ValueError(f'The number of start and end points of anomalies does not match, {start_with_edges} != {end_with_edges}')

        return (start_no_edges, end_no_edges), (start_with_edges, end_with_edges)
    
    @time_it
    def create_safe_mask(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor, extra_safe: bool = False) -> torch.Tensor:
        """
        A safe mask is a mask of the label that every anomaly is one point bigger (left and right)
        than the bigger slope. This allows us to mask the label safely, without changing anything in the implementation.
        """
        length = torch.tensor(label.shape[0], device=self.device)
        mask = torch.zeros(length, dtype=torch.int8, device=self.device)

        safe_extension = self.slope_size + 1 + int(extra_safe)
        
        start_safe_points = torch.clamp(start_points - safe_extension, min=0)
        end_safe_points = torch.clamp(end_points + safe_extension + 1, max=length - 1)
        
        add_start = torch.ones_like(start_safe_points, dtype=torch.int8, device=self.device)
        add_end = torch.ones_like(end_safe_points, dtype=torch.int8, device=self.device).mul(-1)

        mask.index_add_(0, start_safe_points, add_start)
        mask.index_add_(0, end_safe_points, add_end)

        mask = mask.cumsum(dim=0)
        mask = mask > 0
        if extra_safe:
            mask[-1] = end_safe_points.numel() > 0 and end_safe_points[-1] == (length - 1)
        else:
            mask[-1] = label[-1]

        return mask
    
    @time_it
    def distance_from_anomaly(self, label: torch.Tensor, start_points: torch.Tensor, end_points: torch.Tensor, clip: bool = False) -> torch.Tensor:
        """
        Computes distance to closest anomaly boundary for each point. Uses PyTorch for GPU compatibility.
        """
        device = label.device
        length = label.size(0)

        pos = torch.full((length,), float('inf'), device=device)
        if start_points.numel() == 0 and end_points.numel() == 0:
            return pos

        anomaly_boundaries = torch.cat([start_points, end_points])
        indices = torch.arange(length, device=device)[:, None]

        pos_mem_size = len(anomaly_boundaries) * length * pos.element_size()
        if pos_mem_size > self.max_memory_tokens:
            n_chunks = max(1, pos_mem_size // self.max_memory_tokens)
            chunk_size = int(max(1, len(anomaly_boundaries) // n_chunks))

            for chunk in tqdm(anomaly_boundaries.split(chunk_size), desc='In distance from anomaly', disable=False if n_chunks > 10000 else True):
                curr_distances = torch.abs(indices - chunk[None, :])
                curr_min_distances = curr_distances.min(dim=1).values
                pos = torch.minimum(pos, curr_min_distances)
        else:
            distances = torch.abs(indices - anomaly_boundaries)
            pos = torch.min(distances, dim=1).values
        
        if clip:
            pos = pos.clone()
            pos[label.bool()] = 0
            pos = torch.clamp(pos, max=self.slope_size)

        return pos.to(torch.float32)
    
    @time_it
    def add_slopes(self, label: torch.Tensor, pos: torch.Tensor, plot: bool = False) -> torch.Tensor:
        """
        Computes slope transformations for multiple slope values.

        Args:
            label: Binary tensor indicating anomaly regions (shape: [T], dtype: float).
            pos  : Distance to the nearest anomaly boundary (shape: [T]).
            plot : If True, render a visual check of label, pos, and f_pos.

        Returns:
            Tensor of shape (n_slopes, T) with transformed slope values.

        Optimizations vs. original:
          - Drops always-True `valid_mask`. The per-row mask `sv >= pos` handles
            out-of-buffer cells directly.
          - Uses torch.where in a single fused write (no fancy-indexed writes,
            no second masking pass, and inf-safe via per-cell selection).
          - Precomputes (1 - zita) / slope_values once (cached on self).
        """
        device = label.device
        T = label.size(0)

        if not hasattr(self, '_inv_slope_factor'):
            self._inv_slope_factor = (1.0 - self.zita) / self.slope_values[1:, None].to(torch.float32)

        sv = self.slope_values[1:, None]   # (n_slopes-1, 1)
        pos_b = pos[None, :]               # (1, T)
        mask = sv >= pos_b                 # one mask, reused

        f_pos = torch.zeros((self.n_slopes, T), device=device)
        f_pos[0] = label
        f_pos[1:] = torch.where(mask, 1.0 - self._inv_slope_factor * pos_b, 0.0)
        f_pos[1:, label.bool()] = 1

        if plot:
            import matplotlib.pyplot as plt

            label_np = label.detach().cpu().numpy()
            pos_np   = pos.detach().cpu().float().numpy()
            f_pos_np = f_pos.detach().cpu().numpy()
            sv_np    = self.slope_values.detach().cpu().numpy()

            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

            axes[0].plot(label_np, color='black', lw=1.5)
            axes[0].set_title('Label')
            axes[0].set_ylim(-0.1, 1.1)
            axes[0].grid(alpha=0.2)

            finite_mask = np.isfinite(pos_np)
            axes[1].plot(np.where(finite_mask, pos_np, np.nan), color='steelblue', lw=1)
            axes[1].axhline(self.slope_size, color='red', linestyle='--', alpha=0.5,
                            label=f'slope_size={self.slope_size}')
            axes[1].set_title('pos (distance to nearest anomaly boundary)')
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(alpha=0.2)

            cmap = plt.get_cmap('viridis', f_pos_np.shape[0])
            for i in range(f_pos_np.shape[0]):
                axes[2].plot(f_pos_np[i], color=cmap(i), lw=1, alpha=0.85,
                             label=f'b={int(sv_np[i])}')
            axes[2].set_title('f_pos (per-buffer slopes)')
            axes[2].set_xlabel('time')
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].grid(alpha=0.2)
            if f_pos_np.shape[0] <= 12:
                axes[2].legend(loc='upper right', fontsize=7, ncol=2)

            plt.tight_layout()
            plt.show()

        return f_pos

    @time_it
    def compute_existence(self, labels: torch.Tensor, score_mask: torch.Tensor, safe_mask: torch.Tensor = None, normalize: bool = True, allow_recursion: bool = True) -> torch.Tensor:
        """
        PyTorch version of the existence matrix computation. This function uses an approximation fallback mechanism
        in case the memory requirements exceed the limit set by 'max_memory_tokens'.

        This implements the same algorithm as VUSNumpy::existence_matrix (parallel/matrix approach).
        The numpy default is existence_optimized, which is a different algorithm and diverges from
        this one by ~0.005 on average in edge cases where the slopes of two anomalies meet on an
        even step. The exact conditions and worst-case error are not yet characterized.
        TODO: identify the edge cases, quantify the worst-case error, and decide if alignment is needed.
        See vus_numpy.py::existence_matrix.

        Args:
            labels: Tensor of shape [s, T], binary anomaly label one for each slope.
            score_mask: Tensor of shape [t, T], score mask over every threshold.

        Returns:
            Tensor of shape [s, t] representing existence score (fraction of anomalies found).
        """
        if not self.existence:
            return 1 if normalize else (1, 1)
        device = labels.device

        # Compute mask for relevant points
        if safe_mask is not None and not self.global_mask:
            labels = labels[:, safe_mask]
            score_mask = score_mask[:, safe_mask]
        
        if labels.shape[1] < 1:
            return 0 if normalize else (0, 0)

        s, T = labels.shape
        t = score_mask.shape[0]

        # Stairs / labels_stairs / total_anomalies depend only on the labels (not
        # on the thresholds), so compute them once.
        norm_labels = (labels > 0).int()
        diff = torch.diff(norm_labels, dim=1, prepend=torch.zeros((norm_labels.size(0), 1), device=device))
        diff = torch.clamp(diff, min=0, max=1)
        stairs = torch.cumsum(diff, dim=1)
        labels_stairs = norm_labels * stairs
        total_anomalies = stairs[:, -1][:, None]  # (s, 1), threshold-independent

        # The (s, t, T) score_hat / cummax tensors dominate memory. Chunk along the
        # threshold axis so the peak stays within max_memory_tokens, accumulating
        # exact per-threshold results. This replaces the previous linear
        # interpolation across buffer rows, which was lossy: it caused the
        # ~0.12-0.14 VUS errors on long multi-anomaly series (the existence-matrix
        # merge approximation itself is ~1e-4). Each chunk is sized so that
        # s * t_chunk * T <= max_memory_tokens, the same bound the old code used to
        # decide a one-shot computation was safe; small inputs still run in a
        # single pass.
        per_threshold = max(1, s * T)
        t_chunk = max(1, int(self.max_memory_tokens // per_threshold))
        found_parts = []
        for start in range(0, t, t_chunk):
            sm_c = score_mask[start:start + t_chunk]                  # (t_chunk, T)
            score_hat = labels_stairs[:, None, :] * sm_c[None, :, :]  # (s, t_chunk, T)
            cm = torch.cummax(score_hat, dim=2).values
            cm_diff_norm = torch.clamp(torch.diff(cm, dim=2) - 1, min=0)
            final_missed = total_anomalies - cm[:, :, -1]             # (s, t_chunk)
            not_found = torch.sum(cm_diff_norm, dim=2) + final_missed
            found_parts.append(total_anomalies - not_found)
        n_anomalies_found = torch.cat(found_parts, dim=1)             # (s, t)

        if normalize:
            return n_anomalies_found / total_anomalies
        else:
            return n_anomalies_found, total_anomalies

    @time_it
    def compute_confusion_matrix(self, labels: torch.Tensor, sm: torch.Tensor):
        if self.conf_matrix_mode == "dynamic":
            conf_matrix = self.conf_matrix_dyn(labels, sm)
        elif self.conf_matrix_mode == "dynamic_plus":
            conf_matrix = self.conf_matrix_dyn_plus(labels, sm)
        else:
            raise ValueError(f"Unknown conf_matrix_mode: {self.conf_matrix_mode}. Must be 'dynamic' or 'dynamic_plus'.")
        
        fn, tp, positives = conf_matrix

        fp = sm.sum(axis=1) - tp
        negatives = labels[0].shape[0] - positives
        fpr = fp / negatives

        return fp, fn, tp, positives, negatives, fpr
    

    def conf_matrix_trivial(self, labels: torch.Tensor, sm: torch.Tensor):
        label = labels[0]
        sm_inv = ~sm

        true_positives = torch.matmul(labels, sm.float().T)
        false_negatives = torch.matmul(label, sm_inv.float().T).unsqueeze(0)

        positives = ((true_positives + false_negatives) + label.sum()) / 2

        return false_negatives, true_positives, positives
    

    def conf_matrix_dyn(self, labels: torch.Tensor, sm: torch.Tensor):
        if not self.global_mask:
            mask = torch.where(labels[-1] > 0)[0]
            labels = labels[:, mask]
            sm = sm[:, mask]
        
        true_positives = torch.matmul(labels, sm.float().T)
        false_negatives = torch.matmul(labels[0], (~sm).float().T)[None, :]

        positives = ((true_positives + false_negatives) + labels[0].sum()).div(2.0)

        return false_negatives, true_positives, positives

    def conf_matrix_dyn_plus(self, labels: torch.Tensor, sm: torch.Tensor):
        if not self.global_mask:
            slope_mask = torch.where((labels[0] == 0) & (labels[-1] > 0))[0]
        else:
            slope_mask = torch.where(labels[0] == 0)[0]
        label_as_mask = torch.where(labels[0])[0]

        initial_tps = sm[:, label_as_mask].sum(dim=1)
        slope_tps = torch.matmul(labels[:, slope_mask], sm[:, slope_mask].float().T)
        true_positives = initial_tps + slope_tps
        false_negatives = (~sm)[:, label_as_mask].sum(dim=1)
        positives = ((true_positives + false_negatives) + label_as_mask.size(0)).div(2)

        return false_negatives, true_positives, positives
    
    def conf_matrix_cusparse(self, labels: torch.Tensor, sm: torch.Tensor):
        """
        Confusion matrix computation using cuSparse via PyTorch sparse tensors.
        
        Uses COO (coordinate) format for flexible sparse operations that work on both CPU and CUDA.
        
        Args:
            labels: Tensor of shape [n_slopes, T]
            sm: Tensor of shape [S, T]
        
        Returns:
            false_negatives: Tensor [S]
            true_positives: Tensor [n_slopes, S]
            positives: Tensor [n_slopes, S]
        """
        # Convert to sparse COO format for flexible operations
        sm_sparse = sm.to_sparse_coo().float()
        
        # Transpose for @ operation: [S, T] -> [T, S]
        sm_t_indices = torch.stack([sm_sparse.indices()[1], sm_sparse.indices()[0]])
        sm_t_sparse = torch.sparse_coo_tensor(sm_t_indices, sm_sparse.values(), (sm_sparse.shape[1], sm_sparse.shape[0]), device=sm_sparse.device)
        
        # True positives: labels @ sm.T = [n_slopes, T] @ [T, S] = [n_slopes, S]
        true_positives = torch.sparse.mm(labels.float(), sm_t_sparse)
        
        # False negatives: labels[0] @ (~sm).T = [1, T] @ [T, S] = [1, S]
        sm_inv_sparse = (~sm).to_sparse_coo().float()
        sm_inv_t_indices = torch.stack([sm_inv_sparse.indices()[1], sm_inv_sparse.indices()[0]])
        sm_inv_t_sparse = torch.sparse_coo_tensor(sm_inv_t_indices, sm_inv_sparse.values(), (sm_inv_sparse.shape[1], sm_inv_sparse.shape[0]), device=sm_inv_sparse.device)
        
        fn_result = torch.sparse.mm(labels[0:1].float(), sm_inv_t_sparse)
        false_negatives = fn_result.squeeze(0)  # [1, S] -> [S]
        
        # Positives calculation
        positives = ((true_positives + false_negatives) + labels[0].sum()).div(2)
        
        return false_negatives, true_positives, positives
    
    @time_it
    def precision_recall_curve(self, tp: torch.Tensor, fp: torch.Tensor, positives: torch.Tensor, existence: torch.Tensor = None):
        """
        Computes precision-recall curve points given TP, FP, positives, and optional existence weighting.
        
        Args:
            tp: Tensor of shape [n_slopes]
            fp: Tensor of shape [n_slopes]
            positives: Tensor of shape [n_slopes]
            existence: Optional tensor of shape [n_slopes, N] or None

        Returns:
            precision: Tensor of shape [n_slopes, N+1]
            recall: Tensor of shape [n_slopes, N+1]
        """
        device = tp.device

        ones = torch.ones(self.n_slopes, 1, device=device)
        zeros = torch.zeros(self.n_slopes, 1, device=device)
        precision = torch.cat([ones, tp / (tp + fp)], dim=1)
        recall = torch.cat([zeros, tp / positives], dim=1)

        recall[recall > 1] = 1
        if existence is not None:
            recall[:, 1:] = recall[:, 1:] * existence
            
        return precision, recall

    @time_it
    def auc(self, x, y):
        width_pr = x[:, 1:] - x[:, :-1]
        height_pr = y[:, 1:]
        auc_per_slope = torch.sum(torch.multiply(width_pr, height_pr), axis=1)

        n = len(auc_per_slope)
        indices = torch.arange(1, n + 1, device=auc_per_slope.device, dtype=auc_per_slope.dtype)

        if self.aggregation_mode == 'geometric':
            log_auc = torch.log(torch.clamp(auc_per_slope, min=1e-10))
            if self.cumulative:
                return torch.exp(torch.cumsum(log_auc, dim=0) / indices)
            else:
                return torch.exp(log_auc.mean())
        elif self.aggregation_mode == 'harmonic':
            inv_auc = 1.0 / torch.clamp(auc_per_slope, min=1e-10)
            if self.cumulative:
                return indices / torch.cumsum(inv_auc, dim=0)
            else:
                return n / inv_auc.sum()
        else:
            if self.cumulative:
                return torch.cumsum(auc_per_slope, dim=0) / indices
            else:
                return auc_per_slope.mean()
    