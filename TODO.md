# VUS v2 — TODO

_Deadline: 2026-06-11_

---

## Phase 0 — Quick win: All-threshold evaluation (§9)

- [x] Fix `get_unique_thresholds` in `vus_numpy.py`: change default `max_thresholds=200` to `max_thresholds=None` so all unique score values are used by default; subsampling still available by passing an int
- [x] Add warning comment to `get_pr_relevant_thresholds` explaining why it is wrong for VUS: slope buffers give fractional TP credit to buffer-zone points (invisible to the binary collinearity check), and with `global_mask=True` almost all safe-region points are in buffer zones so the pruning gives no benefit
- [x] Write `get_pr_relevant_thresholds_vus`: VUS-aware pruning using `label_slopes.max(axis=0)` (fuzzy label) instead of the binary label — purely normal points (fuzzy=0) are collinear exactly (both cross-product sides = 0), buffer/anomaly points are always kept; verified 0 error vs `all_unique` across YAHOO, SMD, LTDB
- [x] Write `compare_thresholds.py`: benchmarks `all_unique` / `subsampled` / `pr_relevant` / `pr_relevant_fuzzy` on correctness (VUS must match sklearn APS at slope_size=0) and timing (real VUS settings); supports `--dataset`, `--slope_size`, `--n_runs`, `--n` flags
- [x] Hook `get_pr_relevant_thresholds_vus` into `compute()` as an optional threshold strategy — added `thresholds` param to `VUSNumpy.__init__` (default `'pr_relevant_fuzzy'`), `_dispatch_thresholds` helper, and `--thresholds` CLI arg to `compute_metric.py`
- [x] Apply the same threshold strategy hook to `VUSTorch`: add `thresholds` param to `__init__`, add `_dispatch_thresholds` (torch-native: `safe_mask.float().unsqueeze(0)` as proxy), wire into `compute()` and `compute_metric.py`
- [x] Rewrite `compare_thresholds.py`: full benchmark over all TSB-AD (series × detectors), CSV output to `experiments/compare_threshold_methods/results.csv`, resume/force, `--test/--dataset/--detector` flags; `n_thresholds` added to `time_analysis` in `vus_numpy.py`
- [ ] Run full benchmark: `conda run -n ffvus python src/compare_thresholds.py` (all datasets, all detectors)
- [x] Added TP-space uniform sampling cap to `get_pr_relevant_thresholds_vus`: when the exact minimal set exceeds `max_thresholds=10_000`, N targets are placed evenly in `[0, total_TP]` and mapped to the nearest real threshold via `searchsorted`. Deterministic; worst-case AUC error bound is `1/N` = 1e-4 at the default cap. See `notes/thresholds_fix.md`.
- [ ] Validate the cap: run `compare_thresholds.py` on a high-anomaly-rate dataset (one that previously produced 90k+ thresholds) and measure actual max error vs `all_unique`. Decide whether `max_thresholds=10_000` should be higher or lower based on the observed error and runtime.

**Fallback guarantee:** if nothing else ships before the deadline, §9 alone is a valid paper contribution.

- [x] Updated `add_slopes_function` in `vus_numpy.py`; wrote `compare_add_slopes.py` to verify correctness and benchmark speedup vs `add_slopes_precomputed`

---

## Phase 1 — Code & directory audit

### VUS core code
- [x] Read through `vus_numpy.py` carefully — document what is actually implemented vs. dead/experimental
- [x] Remove dead code, commented-out experiments, and unused parameters (`vus_numpy.py` done, `vus_torch.py` done)
- [x] Read through `vus_torch.py` carefully — dead experiments removed, compare_conf_matrix moved to `compare_confusion_matrix.py` (--backend torch added), zita bug fixed
- [x] Cross-compare `VUSNumpy` and `VUSTorch` implementations — intentional gaps documented (torch is stripped-down GPU version by design); two bugs fixed in numpy (zita validation, create_safe_mask empty crash); existence algorithm divergence (~0.005) written down in both files

### Directory & experiment inventory
- [x] Map out the current directory structure (`src/`, `scripts/`, `data/`, `experiments/`)
- [x] Identify what experiment outputs exist and what they correspond to
- [x] List all experiments that have been run so far and on which dataset
- [x] Flag anything that will need to be re-run on TSB-AD

### Rename to VUS v2
- [ ] Settle on final name (VUS v2 vs VUS 2.0)
- [ ] Update all references: repo name, paper title, code imports, README, citation keys

---

## Phase 2 — Migrate from TSB-UAD to TSB-AD

**Completed (2026-05-21, two sessions):**
- [x] `evaluate_ffvus.py` — replaced Dataloader/Scoreloader/analyze_label with TSBADParser; helpers (time_it, auc_pr_wrapper, _analyze_label) are now local
- [x] `compute_metric.py` — replaced load_tsb/analyze_label with TSBADParser; natural_keys and load_synthetic are now local; fixed pre-existing `to_no_global_maskcsv` typo → `to_csv`
- [x] `generate_synthetic.py` — copied _analyze_label locally, removed utils.utils import
- [x] `verify_cumulative.py` — replaced load_tsb with TSBADParser (dataset YAHOO, 10 series)
- [x] `existence_skip_analysis.py` — replaced load_tsb with TSBADParser
- [x] `ablation_study.py` — replaced load_tsb + analyze_label with TSBADParser + local _analyze_label; updated dataset from ECG → YAHOO
- [x] `precompute_rankings.py` — replaced Dataloader/Scoreloader with TSBADParser; output path → experiments/precomputed/
- [x] `precompute_existence.py` — same pattern as rankings; output path → experiments/precomputed/
- [x] `compare_rank_vus.py` — replaced Dataloader/Scoreloader; fixed score shape (n_det, T) axis
- [x] `demo_vus.py` — replaced all 6 helper functions + 2 precomputed paths; removed utils.dataloader/scoreloader imports

### Key API facts

**All files migrated. Remaining tasks below.**
- `from tsbadparser import TSBADParser` — parser is installed as editable in ffvus conda env
- `TSBADParser()` with no args uses `/home/sylli/Documents/tsb-ad-parser/data/` automatically
- `tsb.load_timeseries(load_all=True)` → `(fnames, timeseries, labels)` — lists of numpy arrays
- `tsb.load_timeseries(dataset=['YAHOO'])` — filter by dataset name
- `tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=...)` → `(scores, bad_fnames)` — mutates all 3 lists in-place, removes problematic entries; `scores[i]` is shape `(n_det, T)` **not `(T, n_det)`**
- `tsb.get_detectors()` → list of detector names
- `tsb.get_datasets()` → list of dataset names
- Precomputed output dir: `experiments/precomputed/` (changed from `data/precomputed/`)
- `analyze_label` has no replacement in tsbadparser — copy `_analyze_label` from `evaluate_ffvus.py`

### Remaining migrations (not in original plan)

- [x] `src/compare_confusion_matrix.py` — 2026-05-21
- [x] `src/compare_existence_reward.py` — 2026-05-21
- [x] `src/vus_visualizer.py` — 2026-05-21

### Cleanup
- [x] Delete `src/utils/` entirely — 2026-05-21
- [ ] Re-run all baseline experiments on TSB-AD and record numbers in `experiments/`

---

## Phase 3 — Brainstorming session

For each idea below: write a proposed solution, estimate implementation effort, assess paper impact. The outcome of this session decides what goes into Phase 4.

### §1 — Per-anomaly decomposition
_Claim: VUS = (1/K) Σ VUS_k. New `compute_per_anomaly()` method. The inner loop in `existence_optimized` already iterates per anomaly — scaffolding mostly exists. Enables §2 and informs §8._

- [ ] Write down proposed solution and open questions

### §8 — Detection taxonomy
_Classify each detection event as C-1 (Early), C-2 (Late), C-3 (Exact), E-1 (Isolated FP), E-2 (Missed), E-3 (Partial), E-4 (Early-over-prediction), E-5 (Late-over-prediction). Buffer overlap geometry is already computed per anomaly in `existence_optimized` — may need no second pass._

- [ ] Write down proposed solution and open questions

### §10 — Adaptive buffer expansion
_Per-anomaly buffer modulated by a label shape statistic (e.g. variance) to avoid penalizing detectors when annotators already labeled wide regions. Open questions: which statistic, smooth vs. hard suppression, interaction with §1._

**Current approach (2026-05-22):** use a random baseline to find the buffer where the detector's advantage peaks.

#### Core idea
For each (label, score) pair, compute VUS for all buffers (using `cumulative=True`). Do the same for a random baseline score. The ratio `VUS_detector(b) / VUS_random(b)` captures how much better the detector is than chance at each buffer — the optimal buffer is where this ratio peaks or elbows.

#### Story
FF-VUS computes VUS for all buffers in a single pass. This unlocks automatic buffer selection — something the original VUS formulation couldn't afford in practice. We select b* as the argmax of `ratio(b) = cumVUS(b) / E[cumVUS_random(b)]`. The denominator encodes how much a random detector would benefit from the buffer zone alone, given this specific anomaly geometry. The ratio isolates the detector's true timing advantage over that baseline. The peak is the buffer at which the detector's advantage is strongest relative to chance — theoretically grounded, and free once FF-VUS has already run.

#### What we derived
- `E[AUC-PR(b)]` for a Bernoulli(p) baseline = `g_b = labels_arr.mean(axis=1)[b]`, the mean fuzzy label at buffer b. Reason: precision of an independent random score is always `g_b` in expectation, and the stepwise AUC reduces to `g_b × Σ(Δrecall) = g_b`.
- The analytical denominator is therefore `cumsum(g_b) / indices` — computable from the label alone, no random draws needed. It is more stable than MC alternatives (no variance, no random seed dependence) and fast enough that MC is not needed in practice.
- Selection method not yet decided: three candidates are implemented (`argmax`, `elbow`, `first_peak`); no winner chosen.

#### What is implemented
- `vus_numpy.py`: `find_adaptive_buffer()` computes the ratio curve using the analytical denominator and calls `select_best_buffer()` to pick b*; `compute(adaptive=True)` is the end-to-end entry point.
- `src/test_adaptive_buffer.py`: synthetic suite — 7 label configs × 8 detection score types, visualises ratio curves and b* selection for all three methods and all three denominators.
- `src/test_adaptive_buffer_real.py`: real-data counterpart — samples TSB-AD series stratified by mean anomaly length, plots time series + detector score + analytical ratio curve.

#### Next steps
- [x] Decide selection method — **argmax on the MARGINAL (de-cumulated) skill ratio** (2026-06-02, supervisor-approved). The marginal ratio peaks at the detection's true extent; the cumulative argmax overshot by ~2x. Kept only the no-signal guards (R1, constant-score R4a, `min_signal_frac=1e-3` for numerically-flat slices); dropped the detection gate, runaway guard, plateau, and hard cap (they departed from VUS by dropping/limiting real detections). Rules still toggleable via `rules=` for ablation. See memory `project_per_anomaly_buffer`.
- [x] Evaluate the proposed approach — implemented in `compute_per_anomaly_adaptive_slicing`; 13/13 synthetic unit tests in `test_per_anomaly_buffer.py` (early/late/exact/partial/missed/FP/over-prediction); cross-dataset sweep over 23 TSB-AD datasets shows non-trivial, sensible b* distributions (~56% zero, rest varied; large buffers only on clustered/oscillatory regions = honest VUS inclusions). Showcase + failure case plots saved under `experiments/`.

**After brainstorming:** tick the ideas going into this submission and update Phase 4 accordingly.

---

## Phase 4 — Implement chosen ideas

_To be filled in after the brainstorming session._

- [ ] Implement chosen idea(s)
- [ ] Run experiments on TSB-AD with new implementations
- [ ] §2 Localization map — add if §1 is chosen (trivial once §1 exists: one method + one plotting function)

---

## Phase 5 — Paper rewrite

_Start once Phase 2 results are in hand and at least one of §1 / §8 / §10 is done._
_Target start: 2026-05-27. Hard deadline: 2026-06-11._

- [ ] Rewrite from scratch incorporating VLDB and KDD reviewer feedback
- [ ] Address Sørbø & Ruocco corner cases in related work / discussion section
- [ ] Include §9 results (exact curve, correctness argument for point anomalies)
- [ ] Include results from Phase 4 chosen ideas
- [ ] Update all figures and tables with TSB-AD numbers

---

## Unified precompute script (`src/precompute_demo.py`)

_Done 2026-05-22._

- [x] Created `src/precompute_demo.py` with `load_data`, `step_rankings`, `step_existence`, `step_adaptive_buffer`
- [x] All CLI flags implemented: `--L`, `--K`, `--aggregation`, `--backend`, `--datasets`, `--skip`, `--force`
- [x] Skip/force logic: per-step, mode-aware for rankings (only missing aggregation modes recomputed)
- [x] `_skip_adaptive_plot = True` set on VUS instances — no plots spawned in batch mode
- [x] Deleted `precompute_rankings.py` and `precompute_existence.py` (redundant)
- [x] Fixed slider crash in demo (`min_value == max_value` when only 1 adaptive buffer record)
- [ ] **Pending: run full precompute at home** — `conda run -n ffvus python src/precompute_demo.py --L 1024 --K 100 --aggregation all --force`
  - Note: stale `rankings.npz` at L=100 exists in `experiments/precomputed/` — `--force` required
  - `existence_maps.npz` missing entirely
  - Adaptive buffer PKLs (YAHOO, NAB, UCR, CATSv2, Daphnet) already exist at L=1024 K=100

---

## Deferred — out of scope for this submission

- §10 Adaptive buffer (if not chosen in Phase 3)
- §3 Multi-detector comparison library
- §6 Unified metric suite
- §11 Dense anomaly / buffer saturation (revisit after §10)
- NN fine-tuning with fast VUS
- Clean public repo for VUS v2 (do after submission)
