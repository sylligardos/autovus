# FF-VUS — Research Ideas & V2 Directions

_Last updated: 2026-05-23_

---

## At a Glance

### Ideas by Section

| § | Title | Section | Status |
|---|-------|---------|--------|
| 9 | All-threshold evaluation | Main Focus | Active |
| 1 | Per-anomaly decomposition | Main Focus | Active |
| 8 | Detection taxonomy | Main Focus | Active |
| 10 | Adaptive buffer expansion | Main Focus | Active |
| 12 | Decumulated ("denoised") VUS | Future Steps | Parked (prototype done) |
| — | Corner cases (Sørbø & Ruocco) | Ideas to Decide On | Under review |
| 2 | Performance localization map | Future Steps | Pending §1 |
| 4 | Penalizing buffer / strict VUS | Done | Dropped |
| 5 | Geometric mean over slope sizes | Done | Dropped |
| 7 | Ranking instability visualizations | Done | Done |
| 3 | Multi-detector comparison library | Extras | Deferred |
| 6 | Unified metric suite | Extras | Deferred |
| 11 | Dense anomaly / buffer saturation | Extras | Deferred (revisit after §10) |
| — | NN fine-tuning with fast VUS | Extras | Low priority |


---

## Main Focus

### 9. All-threshold evaluation (eliminate the 300-threshold sampling artifact)

**Problem:** VUS currently samples 300 thresholds to approximate the R-AUC ROC/PR curves. This sampling makes the curve uneven and can severely underestimate accuracy. The failure mode is acute for point anomalies: if the true anomaly is a single timestep, the highest anomaly score is slightly off, and the series is long, the sampled thresholds may all land on the wrong side of the relevant score — collapsing accuracy from 1.0 to ~0.01. This problem is not unique to VUS; it affects all range-based measures that use threshold sampling (e.g., Nesime's NeurIPS paper).

**Proposed fix:** With the 100x speedup, use every unique score value as a threshold. This is now computationally feasible and produces an exact curve rather than an approximation. The PR/ROC areas become exact integrals over the true staircase, not a sampled approximation.

**Implementation:** Replace the linspace threshold grid in the evaluation pipeline with `np.unique(score)` (plus boundary values 0 and 1). The rest of the pipeline is unchanged.

**Paper angle:** Turns a known approximation error into an exact computation — framed as a quality improvement that the efficiency work unlocks for free.

---

### 1. Per-anomaly decomposition (the theoretical core)

**Claim:** VUS is decomposable — VUS = (1/K) Σ VUS_k, where VUS_k is the VUS score computed with only anomaly k's label active. This is a structural property no existing evaluation metric has.

**Implementation:** New method `compute_per_anomaly(label, score)` that iterates over anomalies, zeroes out all others, calls the existing pipeline, and collects individual scores. The key is that `existence_optimized` already iterates per anomaly (see `for k in range(n_anomalies[0])` in `vus_numpy.py`), so the structural work is mostly already there.

**Paper angle:** Frame this as a theorem: VUS decomposes by anomaly. No existing metric (AUC-PR, AUC-ROC, PA%K) has this property. This is the contribution reviewers respond to — it's a structural result, not just a feature.

**Per-anomaly averaging as an alternative aggregation (2026-06-02):** Now that we have a VUS per anomaly, we can *average the per-anomaly scores* instead of recomputing one global VUS. The two give very different numbers, and that difference is the point. Observed example: Chronos scores per anomaly `[0.5, 0.116, 1, 1]` (mean 0.654) but the global capped VUS-PR is 0.062 — below the per-anomaly *minimum*. Same asymmetry seen across detectors: a detector that finds the 3 point anomalies but misses the wide one scores low globally (e.g. CNN), while one that finds the wide anomaly and misses the points scores high (e.g. Sub_OCSVM).

Why the gap (and why 0.062 is NOT a bug):
1. **PR-area is non-additive.** Precision is a global ratio over a shared threshold axis, so VUS-PR ≠ mean of per-anomaly VUS. A point that ranks #1 inside its own local window can rank below series-wide FPs globally. → The literal `VUS = (1/K) Σ VUS_k` claim above holds (approximately) for **VUS-ROC** but **not VUS-PR**; fix this framing before calling it a theorem.
2. **Recall is point-counted, so wide anomalies dominate the recall axis.** Well-detected narrow anomalies contribute negligible recall mass; a poorly-detected wide anomaly owns most of it and lives at low thresholds where precision is already dead.

Equal-weight per-anomaly averaging removes the recall-mass weighting and gives every event the same vote — arguably fairer, and a natural companion to the localization map (§2). Decision: too much material for the current paper; park here for V2.

---

### 8. Detection taxonomy — qualitative error analysis (Paul, 2026-05-20)

**Motivation:** Speed alone has not convinced reviewers (VLDB, KDD rejections). A second novelty axis is interpretability: VUS can explain *how* a detector is right or wrong, not just how much. Reference: Felix's NeurIPS 2025 paper ([arxiv 2510.23261](https://arxiv.org/abs/2510.23261)) as related prior art to position against.

**Correct detection types:**

| Type | Name | Definition |
|------|------|------------|
| C-1 | Early | Predicted segment overlaps the left buffer |
| C-2 | Late | Predicted segment overlaps the right buffer only |
| C-3 | Exact | No overlap with either buffer — lands squarely inside the anomaly |

**Error types:**

| Type | Name | Definition |
|------|------|------------|
| E-1 | Isolated | Pure false positive — no anomaly nearby |
| E-2 | Missed | No prediction at all (including in both buffers) |
| E-3 | Partial | Some points inside the anomalous segment are missed |
| E-4 | Early-Over-Prediction | Contiguous segment fires too early — starts before the left buffer |
| E-5 | Late-Over-Prediction | Contiguous segment fires too late — ends after the right buffer |

**Implementation angle:** The buffer overlap geometry is already computed per anomaly in the `existence_optimized` loop (`vus_numpy.py`). The detection type should be readable with minimal overhead — no second pass needed. Open question from Paul: confirm exactly what intermediate values are already available and whether the full taxonomy can be derived from them.

**Paper angle:** Combined with the per-anomaly decomposition (§1), this turns VUS into a full diagnostic framework: a per-anomaly *score* (§1) plus a per-anomaly *label* (§8). The contribution is then: faster + more interpretable — two independent novelty axes in one paper.

---

### 10. Adaptive buffer expansion per anomaly

**Problem (reported by Mingyi):** The current VUS applies the same buffer length uniformly to every anomaly. This causes two distinct failure modes:

1. **Over-extended labels:** The true anomaly is a single spike, but the annotator labeled a large surrounding region. VUS then adds a buffer on top of that already-inflated label. A detector that precisely identifies the actual spike ends up penalized because it doesn't cover the redundant buffer extension.
2. **Undersized buffer for large anomalies:** The buffer is capped at a fixed length. For a large contiguous anomaly region, this may be too small. For a spike, the same fixed length is disproportionately large relative to the anomaly.

**Proposed fix:** Compute a per-anomaly statistic — e.g., internal variance of the ground-truth label segment — to determine whether the label is already over-extended. High variance (a spike surrounded by a wide labeled region) is a signal that the annotator's label already incorporates tolerance, so further buffer expansion should be suppressed or reduced. This makes buffer expansion *adaptive*: proportional to the anomaly's shape, not a global constant. The same metric could serve for identifying the right buffer length for a given anomaly.

**Current implementation (2026-05-23):** Rather than modulating buffer by label shape, the implemented approach selects the best buffer from a ratio curve: `ratio(b) = cumVUS_detector(b) / E[cumVUS_random(b)]`. Three denominator variants are supported (`analytical`, `mc_bernoulli`, `mc_uniform`) and three selection methods (`argmax`, `elbow`, `first_peak`). A synthetic test suite (`test_adaptive_buffer.py`) covers 7 label structures × 8 detection types (C-1 through E-5) to validate selection behavior across the full taxonomy.

**Open questions:**
- What statistic best captures "already-extended" labels? Variance, IQR, or a spike detector on the label boundary?
- Should the buffer be suppressed entirely, or smoothly reduced (e.g., buffer = max(0, L - existing_extension))?
- Does this interact with the per-anomaly decomposition (§1)? Each anomaly's buffer could be decided independently.
- Which denominator (analytical vs MC) and which selection method (argmax / elbow / first\_peak) should go in the paper? Pending validation on real data.

**Paper angle:** Moves VUS from a fixed-geometry measure to one that respects the annotator's intent. Directly addresses a known limitation of the original design.

---

## Ideas to Decide On

_From "Navigating the Metric Maze" — Sørbø & Ruocco, arXiv:2303.01272 (20 metrics, 14 case studies)_

**Corner cases the paper identifies:**

- **Partial detection vs. covering** (Figs. 12 & 23): predicting two anomalies partially vs. covering one anomaly fully. VUS-PR scores 0.86 (full cover) vs. 0.83 (detect both) — it does not reward finding new anomaly events. Structural: VUS is point-based. Per-anomaly decomposition (§1) makes missed events visible but does not change the preference.
- **Short-anomaly cluster bias** (Fig. 21): VUS-PR prefers 2 short anomalies (score 0.74) over 1 long anomaly (0.69) because buffer zones inflate short events proportionally more. The per-anomaly VUS_k scores would expose this discrepancy.
- **Proximity reliability** (Figs. 24 & 25): VUS is the *only* non-binary metric that rewards predictions close to (but not overlapping) labeled anomalies. However, this relies on the anomaly score being smooth — not guaranteed to be fair for sharp/peaked detectors.
- **Class imbalance in VUS-ROC** (Figs. 26 & 27): adding normal points changes which detector VUS-ROC prefers. VUS-PR does not have this problem.
- **Ambiguous labeling** (Figs. 16 & 19): AUC-PR collapses when a detector fires slightly early or late; VUS-PR is tolerant.

**Criticisms of VUS from this paper:**

- Still depends on max *l* — not truly parameter-free
- Point-based only: no event-detection preference (partial-detection corner case above)
- Short-anomaly bias from disproportionate buffer widening
- Proximity property vanishes at l=0 (reduces to AUC)
- Proximity relies on score smoothness — not always fair
- VUS-ROC is sensitive to true-negative count (prefer VUS-PR)

**What our multi-buffer speedup can address:**

- *l-dependence*: show score-vs-l curve across all l simultaneously — makes the choice visible and principled
- *Short-anomaly bias*: per-anomaly VUS_k (§1) surfaces per-event discrepancies instead of hiding them in the aggregate
- *Proximity reliability*: show the proximity property holds across the full l range, not just one chosen value

**Open question:** Does PATE (2024) address the partial-detection or short-anomaly bias corner cases? This paper predates PATE.

---

## Future Steps

### 2. Performance localization map (the visual payoff)

**Claim:** Given a label and a detector's score, produce a 1D array (same length as the time series) where each anomalous region is colored by its per-anomaly VUS_k. Non-anomalous regions are neutral (NaN or 0).

**Why it's more than a visualization:** It's the geometric proof-of-concept for (§1). An engineer can see at a glance that detector A misses anomaly #3 while detector B nails it.

**Implementation:** One new method `localization_map(label, score)` that calls `compute_per_anomaly` and fills a time-aligned array. A separate plotting function renders it as a color overlay on the raw time series. Trivial once (§1) exists.

**Paper angle:** Turns evaluation from a scalar into a diagnostic. The heatmap concretizes the decomposition — reviewers can see it, not just read about it.

---

### 12. Decumulated ("denoised") VUS over the buffer axis (2026-06-04)

**Observation:** As the buffer `b` grows, VUS grows. The reported VUS curve is a running *mean* over buffers, `C[b] = cumsum(A)[b] / (b+1)`, where `A[b]` is the AUC-PR at buffer level `b` on its own. Because it's a running mean, a reward first earned at buffer 20 (e.g. a late detection caught there) is folded into `C[b]` for *every* `b ≥ 20`. The cumulative curve therefore hides *where* along the buffer axis a detector actually earns its score.

**Idea:** Decumulate the curve so the per-buffer reward is visible, and ask whether a "denoised" VUS — one that doesn't keep climbing just because the buffer widened — is a better/cleaner metric. Long-term goal (in theory; too early to know): a better VUS. Collapsing the decumulated curve back into a single headline number is deferred — considered easy/secondary.

**Two candidate decumulated views (prototype computes both):**
1. `A[b] = diff(C * indices, prepend=0)` — inverts the running mean, recovering per-buffer AUC. This is the same formula already used for `marginal` buffer selection in `find_adaptive_buffer`. Removes the running-*mean* smear.
2. `A[b] − A[b−1]` — per-step incremental gain. Localizes where each buffer *first* earns reward (spike at 20, flat after).

**Key finding from the prototype (2026-06-04):** Decumulating the mean (view 1) does **not** stop the growth — on real NAB data `A[0] < A[L]` for every detector. Reason: the slope labels are themselves cumulative in `b` (a point caught at buffer 20 stays a TP for all larger buffers), so per-buffer AUC `A[b]` is still ~monotone. The view that actually localizes per-buffer reward — and matches the "gain at 20 shouldn't reappear at 30, 40, …" intuition — is the **first difference (view 2)**. So if this becomes "the decumulated VUS," it's the per-step gain, not the inverted mean.

**Status / what exists:**
- `VUSNumpy.decumulate(vus_curve)` — inverts the running mean (mean aggregation only; guarded). `vus_numpy.py`, end of the aggregation section near `auc()`.
- `src/test_decumulated_vus.py` — loads any TSB-AD series/detectors and plots: time series, score, cumulative `C[b]`, decumulated `A[b]`, and per-step gain `A[b]−A[b−1]`. Includes a round-trip assert that re-cumulating `A` reproduces `C`.

**Open questions:**
- Which curve becomes "the decumulated VUS" of record — `A[b]` or `A[b]−A[b−1]`? (Prototype suggests the latter.)
- How to collapse it into a single number, and does that number rank detectors more sensibly than plain VUS?
- Relation to the adaptive-buffer ratio (§10): both decumulate, but this is a *global metric* concern, explicitly **not** tied to the per-anomaly work.
- Only defined/cared-for under `aggregation='mean'`; geometric/harmonic don't invert via `diff`.

**Scope note:** Parked deliberately — prototype done, not pursuing further right now.

---

## Done

### 4. Penalizing buffer variant — strict VUS

Tried. Does not meaningfully change rankings. A parameter α ∈ [0, 1] interpolates between hard FN (current) and fuzzy FN. The asymmetry it was meant to fix does not produce a visible difference in practice on TSB-AD.

---

### 5. Geometric mean over slope sizes

Implemented as `aggregation='geometric'` in `VUSNumpy`. Does not change outcomes relative to arithmetic mean in practice.

---

### 7. Ranking instability — visualizations and theoretical framing

Done. Score-vs-L curves, crossing points, and stochastic dominance framing provide good insights. Good material for the demo. Does not add enough novelty on its own to change a reviewer's opinion.

---

## Extras

### 3. Multi-detector, multi-series comparison library

Build a library-level API that takes N time series and M detectors and returns: global ranking, per-anomaly ranking matrix (detectors × anomalies), and localization maps per detector. No reviewer would change their opinion based on this alone. Defer until the core contributions are solid.

---

### 6. Metric suite with a unified score

Compute AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, PA, PA%K together and combine via geometric mean or rank aggregation. Defer — same reasoning as §3.

---

### 11. Dense anomaly / buffer saturation problem

When anomalies are extremely dense (e.g., STOCK series in TSB-AD), buffer zones merge and can cover nearly the entire series, causing VUS to overestimate detector quality. Likely resolved naturally by adaptive buffer expansion (§10). Revisit after §10 is implemented.

---

### NN fine-tuning with fast VUS

Find a neural-network-based TSAD trained with AUC-PR, fine-tune it with fast VUS, and show performance improves. Interesting but not a priority right now.
