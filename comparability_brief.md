# Adjusted VUS — comparability brief (for the math review)

**Goal of this note.** We have a new anomaly-detection metric that selects a
*different buffer size for every anomaly–detector pair* and then collapses
everything into a single score. We need a proof that these single scores are
**comparable across detectors and across time series**, even though each was
computed with different per-anomaly buffers. This note gives only the two pieces
you need — (1) per-anomaly buffer selection, (2) the final single-score
computation — with pointers to the code. Everything else in the repo is
irrelevant to the proof.

---

## 1. Background: VUS in one paragraph

For a time series of length `T` we have a binary ground-truth label
`y ∈ {0,1}^T` (anomalies are contiguous segments) and a detector score
`s ∈ ℝ^T`. VUS introduces a **buffer** `b`: the label is "softened" by extending
a tolerance region of width `b` on each side of every anomaly, so a detection up
to `b` steps away still earns (partial) credit. For each buffer size
`b ∈ {0,1,…,L}` one computes a precision–recall area

```
A(b) = AUC-PR( detector s  vs  the b-buffered label )  ∈ [0,1].
```

Standard VUS (the accepted metric, with `L` fixed by the user, here `L=256`) is
the average over all buffers:

```
VUS = (1/(L+1)) · Σ_{b=0}^{L} A(b).
```

Our pipeline keeps `L=256` fixed for everyone. Two derived quantities matter:

- **Cumulative curve** `C(b) = (1/(b+1)) Σ_{j=0}^{b} A(j)` (so `VUS = C(L)`).
- **Marginal (de-cumulated) value** `A(b)`, recovered exactly from `C` by
  `A(b) = (b+1)·C(b) − b·C(b−1)`. *(code: `decumulate`)*

**Random baseline.** Under a random detector the expected AUC-PR at buffer `b`
equals the base rate of the `b`-buffered label:

```
g(b) = E[ A(b) | random detector ] = (positive area of the b-buffered label) / T.
```

This is exact for AUC-PR and is computed analytically (no Monte Carlo).
*(code: `_compute_denom_curve`, `denom='analytical'`)*. Note `g(b)` increases with
`b` (a wider buffer means more positives, so a higher chance baseline).

---

## 2. Step 1 — per-anomaly buffer selection

Let the label have anomalies `k = 1,…,K`. For each anomaly `k`:

1. Restrict to a window `W_k` around it (split the gap to each neighbour in half).
2. On that window compute the **marginal skill ratio**

   ```
   r_k(b) = A_k(b) / g_k(b),      b = 0,…,L
   ```

   where `A_k, g_k` are the marginal detector and baseline values of Step 1.
3. Select

   ```
   b*_k = argmax_{b}  r_k(b)
   ```

   (plus two “no-signal’’ guards: if the detector never beats chance, or the
   window contains no real signal, set `b*_k = 0`. These never drop a genuine
   detection.)

Intuition: `r_k(b)` is how many times better than chance the detector is when we
allow tolerance `b`; its argmax is the buffer that best captures this anomaly’s
detection. *(code: `compute_per_anomaly_adaptive_slicing` → `find_adaptive_buffer`
→ `select_best_buffer`; the marginal ratio is built at the line
`sel_ratio = marg_vus / marg_denom`.)*

**Selection and scoring are deliberately separate steps.** Step 1 only outputs the
vector `b* = (b*_1,…,b*_K)`. It does not produce the final score.

---

## 3. Step 2 — the final single score

Given `b*`, rebuild the label so that anomaly `k`’s buffer is **capped at `b*_k`**
(each anomaly may have a different cap), then run the usual VUS sweep `b=0,…,L`
once on this **adapted** label. Write `A*(b)` and `g*(b)` for the adapted
detector and baseline marginals, and

```
VUS_adapt = (1/(L+1)) Σ_b A*(b),     denom = (1/(L+1)) Σ_b g*(b).
```

From these we report (all in code: `score_adapt` / `_normalized_vus`):

| name        | formula                                            | range            |
|-------------|----------------------------------------------------|------------------|
| `ratio_A`   | `VUS_adapt / denom`                                 | `[0, ∞)`         |
| `ratio_B`   | `mean_b [ A*(b) / g*(b) ]`                          | `[0, ∞)`         |
| **`skill_A`** (headline, "Adjusted VUS") | `(VUS_adapt − denom) / (1 − denom)` | `(−·, 1]`, 0=chance |
| `skill_B`   | `mean_b [ (A*(b) − g*(b)) / (1 − g*(b)) ]`          | `(−·, 1]`, 0=chance |

The headline metric is **`skill_A`**, the *chance-adjusted* score. It is exactly
the “adjustment for chance’’ used by the **Adjusted Rand Index** and **Adjusted
Mutual Information**:

```
adjusted = (index − E[index | random]) / (max index − E[index | random]),
```

with `index = VUS_adapt`, `E[index|random] = denom`, `max = 1`. So `skill_A = 0`
for a random detector and `skill_A = 1` for a perfect one.

---

## 4. What we would like you to prove

The supervisor’s objection: *“each anomaly–detector pair picks a different `b*`,
so the single scores are computed under different conditions — why are they
comparable?”* Concretely, we’d like (any subset of) the following made rigorous.

**Proposition A (calibration, fixed buffers).** For any label `y` and any fixed
per-anomaly cap vector `b*` (not chosen from the data), if the detector is random
then `E[skill_A] = 0`; and `skill_A ≤ 1` always, with `= 1` iff the detector is
perfect. Same for `skill_B`. *(Note `denom` is deterministic given the
configuration, which should make this clean.)*

**Proposition B (cross-pair comparability — the main one).** For two detectors on
the same label, each with its own selected `b*`, the resulting `skill_A` values
lie on a common scale (`0` = chance, `1` = perfect) **independent of which `b*`
each chose**. Make precise the sense in which “0 and 1 mean the same thing across
different buffer configurations,” and show ranking by `skill_A` is therefore
well-defined across detectors — and across different time series.

**Proposition C (selection bias / winner’s curse).** Propositions A–B assume the
buffers are fixed. But `b*` is chosen by `argmax` over a noisy curve (Step 1), so
even a random detector, *after selection*, scores slightly above chance. We
observe empirically `E[skill_A | random] ≈ 0` for fixed buffers but `> 0` after
selection. We’d like: a bound/characterization of this optimistic bias as a
function of `L` and the number of anomalies `K`, and an argument that because the
bias is **shared** (every detector gets the same argmax freedom over the same
buffer grid) the **ranking** between detectors is still preserved.

---

## 5. Minimal code map

All in `src/vus/vus_numpy.py` (class `VUSNumpy`):

| concept                                   | location                                  |
|-------------------------------------------|-------------------------------------------|
| per-buffer VUS sweep `C(b)` (`cumulative=True`) | `compute(...)`, aggregation in `auc` (line ~1786) |
| de-cumulate `C(b) → A(b)`                 | `decumulate` (line ~1821)                 |
| random baseline `g(b)` (analytical)       | `_compute_denom_curve` (line ~538)        |
| **Step 1**: per-anomaly `b*` selection    | `compute_per_anomaly_adaptive_slicing` (line ~735) → `find_adaptive_buffer` (line ~583) → `select_best_buffer` (line ~215) |
| marginal skill ratio `r_k(b)`             | `find_adaptive_buffer`, `sel_ratio = marg_vus / marg_denom` (line ~641) |
| build adapted (capped) label              | `_build_capped_labels` (line ~1855)       |
| **Step 2**: final single score            | `score_adapt` (line ~1928) → `_normalized_vus` (line ~1885) |

Scope reminder: only Sections 2 and 3 (Steps 1 and 2) are in scope. The rest of
the file is the underlying VUS machinery, which you can treat as a black box that
returns `A(b)` and `g(b)`.
