"""
Adaptive Buffer Selection — Test Suite
=======================================

PURPOSE
-------
VUS sweeps over buffer sizes b in [0, L] and computes AUC-PR(b) for each.
The adaptive buffer problem: automatically select the best buffer b* without
prior knowledge of the anomaly structure or the detector.


THE HARD BOUNDARY PROBLEM
--------------------------
Any fixed buffer b creates a cliff: a detection at distance b is rewarded,
a detection at b+1 is not. This is arbitrary and unfair to detectors that
fire slightly outside the window. The adaptive approach sidesteps this by
asking a relative question instead.


THE RATIO APPROACH (what VUS currently does)
---------------------------------------------
Instead of picking a fixed b, compute:

    ratio(b) = cumVUS_detector(b) / E[cumVUS_random(b)]

This answers: "how much better than a random score is this detector, on
average up to buffer b?" A detection is rewarded as long as the detector
beats random — the exact distance no longer matters, only relative quality.

The adaptive buffer b* is then selected from the ratio curve, e.g.:
    - argmax  : buffer where the relative advantage is largest
    - elbow   : point of maximum curvature on the ratio curve
    - first_peak : first local maximum (early saturation point)


THE ANALYTICAL DENOMINATOR
---------------------------
Key claim: for a random i.i.d. Bernoulli(p) score independent of the label,

    E[AUC-PR(b)]  =  g_b

where g_b = mean of the buffered label at buffer size b:

    g_b = (1/T) * sum_i  buffered_label_i(b)

The buffered label at buffer b is 1 inside anomalies, linearly decreasing
from 1 to zita over the buffer zone, and 0 elsewhere.

Derivation sketch (stepwise AUC):
  1. A Bernoulli(p) score has exactly 2 distinct values: {0, 1}.
     The PR curve has only 2 real threshold steps.
  2. At threshold=0 (flag everything):
       precision = sum(buffered_label) / T = g_b   (exact, not random)
       recall    = 1
  3. At threshold=1 (flag only 1s):
       Given k flagged points, each position is equally likely (independence).
       E[TP_b | k] = k * g_b  →  E[precision | k] = g_b  for any k > 0.
       When k=0: precision=1 by convention but recall=0, contribution=0.
  4. AUC (stepwise) = precision_t1 * Δrecall_t1 + precision_t0 * Δrecall_t0
                    = g_b * Δrecall_t1 + g_b * Δrecall_t0
                    = g_b * (Δrecall_t1 + Δrecall_t0)
                    = g_b * 1  =  g_b

CRITICAL: This derivation only holds for STEPWISE interpolation.
With linear interpolation: E[AUC] = g_b + p*(1 - g_b)/2, which depends on p.
sklearn's average_precision_score uses stepwise; precision_recall_curve+auc
uses linear. Verified empirically in src/test_analytical_denom.py.

Therefore:

    E[cumVUS_random(b)]  =  cumsum(g_b)[b] / (b+1)

which is exactly what the code computes:
    analytical_denom = np.cumsum(g_b) / np.arange(1, n_slopes + 1)


KNOWN LIMITATION: SHORT ANOMALIES
-----------------------------------
The formula E[AUC-PR(b)] = g_b is approximate. It assumes:

    E[precision * Δrecall]  ≈  g_b * E[Δrecall]

i.e., precision and recall change are approximately independent. This holds
when the expected number of flagged points inside the anomaly window is >> 1:

    condition:  p * T * g_b  >>  1

For short anomalies (anomaly_length << T), p ≈ anomaly_length/T is small,
T * g_b ≈ anomaly_length + buffer_zone, so:

    p * T * g_b  ≈  anomaly_length * g_b  →  small when anomaly_length → 1

Failure mode: when a random 1 hits a 1-point anomaly, precision ≈ 1/(1+FP)
and recall jumps from 0 to 1 simultaneously. These two quantities are strongly
correlated (not independent). This makes E[AUC] > g_b, meaning the analytical
denominator UNDERESTIMATES the true E[VUS_random], inflating the ratio for
detectors on short anomalies.

Direction of bias:  analytical < MC(Bernoulli) for short anomalies.
Worst when:         slope_size / anomaly_length >> 1  (buffer_ratio >> 1)

Potential fix: use a continuous uniform random score as the MC baseline.
For a continuous score (random permutation of uniform [0,1] values), at every
rank position r, E[precision(r)] = g_b EXACTLY by symmetry (no Jensen gap).
The analytical g_b formula is exact for this score distribution.


WHAT TO TEST
-------------
Separate two concerns:

1. DENOMINATOR CALIBRATION
   - Input: random score, various label structures
   - Expected: ratio(b) ≈ 1.0 for all b
   - Test: analytical vs MC Bernoulli vs MC continuous
   - Failure signal: ratio systematically > 1 or < 1

2. BUFFER SELECTION QUALITY
   - Input: a "good" detector score + known label structure
   - Expected: selected b* makes structural sense (e.g., near anomaly length)
   - Test: argmax vs elbow vs first_peak on ratio curve
   - Failure signal: selected buffer is at wrong end of the range


SYNTHETIC PARAMETERS TO SWEEP
-------------------------------
    T              : total series length           (e.g., 100 to 10000)
    L              : anomaly length                (e.g., 1 to 200)
    N              : number of anomalies           (e.g., 1 to 20)
    G              : inter-anomaly gap             (e.g., L to 10*L)
    slope_size     : max buffer to sweep           (e.g., 50 to 200)

Key derived ratios (expect these to drive failures):
    L / T          : anomaly density               (small → hard case)
    slope_size / L : buffer-to-anomaly ratio       (> 1 → hard case)
    G / slope_size : gap-to-buffer ratio           (< 2 → overlapping buffers)


SYNTHETIC SCORE TYPES
----------------------
    random_bernoulli  : Bernoulli(p=label.mean()) — calibration baseline
    random_uniform    : Uniform(0, 1) — continuous calibration baseline
    perfect           : 1 inside anomaly, 0 outside — upper bound
    offset            : perfect but shifted by d steps — tests detection lag
    noisy             : perfect + Gaussian noise — realistic
    flat              : constant score — worst case


APPROACHES TO COMPARE
----------------------
    A. Analytical g_b (current)
    B. MC Bernoulli(p)
    C. MC continuous uniform
    D. Label-only heuristic: buffer = f(mean_anomaly_length), no score needed

Questions:
    - Does A match C? (validates the analytical formula for continuous scores)
    - Does A match B? When does it diverge? (the short anomaly issue)
    - Does D give a sensible buffer without needing any score?
    - For a good detector, does the selected b* make structural sense?


FOR THE PAPER
--------------
Claim to prove formally:
    "For a random i.i.d. score S_i ~ Bernoulli(p), independent of the label,
     the expected AUC-PR under stepwise interpolation equals g_b, the mean
     of the buffered label. Therefore E[cumVUS(b)] = cumsum(g_b)/(b+1)."

Proof relies on:
    - Stepwise AUC formula: sum_i precision_i * (recall_{i+1} - recall_i)
    - Independence of S and label: E[TP_b | k flagged] = k * g_b
    - Binary score → only 2 real threshold steps
    - Precision = g_b at both steps (shown above)

Limitation to state:
    The approximation holds when p * T * g_b >> 1 (enough expected positives
    inside the anomaly window). For very short anomalies (L << T, large
    slope_size/L), the formula underestimates the denominator.

Empirical validation needed:
    - Show ratio ≈ 1 for random scores on typical benchmark series
    - Show ratio > 1 for a good detector, with meaningful b* selection
    - Show where the formula breaks (short anomaly regime)
    - Compare to continuous-score MC as the "true" denominator
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vus.vus_numpy import VUSNumpy, select_best_buffer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOPE_SIZE = 1024

DENOMS  = ['analytical', 'mc_bernoulli', 'mc_uniform']
METHODS = ['argmax', 'elbow', 'first_peak']

DENOM_STYLES = {
    'analytical':   dict(color='crimson',   linestyle='--'),
    'mc_bernoulli': dict(color='seagreen',  linestyle='-'),
    'mc_uniform':   dict(color='steelblue', linestyle='-.'),
}
METHOD_COLORS = {
    'argmax':     'royalblue',
    'elbow':      'darkorange',
    'first_peak': 'mediumpurple',
}


# ---------------------------------------------------------------------------
# Label configs — each stresses one structural property
# ---------------------------------------------------------------------------

LABEL_CONFIGS = [
    dict(T=600, L=20, N=3, G=100, tag="baseline"),
    dict(T=600, L=5,  N=3, G=100, tag="short_anom"),      # slope_size / L = 8  → hard case
    dict(T=600, L=60, N=3, G=120, tag="long_anom"),        # slope_size / L = 0.67 → buffer < anomaly
    dict(T=600, L=20, N=5, G=25,  tag="dense_overlap"),    # G=25 < 2*slope_size=80 → buffer overlap
    dict(T=500, L=20, N=1, G=0,   tag="single"),           # single anomaly
    dict(T=800, L=3,  N=5, G=120, tag="tiny_sparse"),      # extreme: slope_size / L = 13
    dict(T=800, L=80, N=2, G=350, tag="wide_sparse"),      # buffer << anomaly, very sparse
]


# ---------------------------------------------------------------------------
# Synthetic label generator
# ---------------------------------------------------------------------------

def make_label(T, L, N, G):
    """
    Create a binary label with N anomalies of length L separated by gap G.

    Returns:
        label (np.ndarray): binary array of length T
        meta  (dict): actual parameters used (may be clipped if T is exceeded)
    """
    label = np.zeros(T, dtype=float)
    starts = []
    for k in range(N):
        s = k * (L + G)
        e = s + L
        if e > T:
            break
        label[s:e] = 1.0
        starts.append(s)
    return label, {"T": T, "L": L, "N": len(starts), "G": G, "p": label.mean()}


# ---------------------------------------------------------------------------
# Detection score helpers
# ---------------------------------------------------------------------------

def _get_anomaly_bounds(label):
    """Returns (starts, ends) with ends inclusive."""
    diff   = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0] - 1
    return starts, ends


def score_C1_early(label, slope_size):
    """C-1: fires in the left buffer zone only (before the anomaly starts)."""
    score  = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L  = int(round((ends - starts + 1).mean())) if len(starts) else 10
    seg_len = max(1, avg_L // 2)
    for s in starts:
        seg_end   = s - 1
        seg_start = max(0, seg_end - seg_len + 1)
        if seg_end >= 0:
            score[seg_start:seg_end + 1] = 1.0
    return score


def score_C2_late(label, slope_size):
    """C-2: fires in the right buffer zone only (after the anomaly ends)."""
    T      = len(label)
    score  = np.zeros(T)
    starts, ends = _get_anomaly_bounds(label)
    avg_L  = int(round((ends - starts + 1).mean())) if len(starts) else 10
    seg_len = max(1, avg_L // 2)
    for e in ends:
        seg_start = e + 1
        seg_end   = min(T - 1, seg_start + seg_len - 1)
        if seg_start < T:
            score[seg_start:seg_end + 1] = 1.0
    return score


def score_C3_exact(label, slope_size):
    """C-3: fires squarely inside the anomaly, no buffer overlap."""
    score  = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    for s, e in zip(starts, ends):
        L = e - s + 1
        q = max(1, L // 4)
        inner_s = s + q
        inner_e = e - q + 1
        if inner_s < inner_e:
            score[inner_s:inner_e] = 1.0
        else:
            score[s:e + 1] = 1.0  # anomaly too short to trim, flag the whole thing
    return score


def score_E1_isolated(label, slope_size):
    """E-1: isolated false positive well away from any anomaly."""
    T      = len(label)
    score  = np.zeros(T)
    starts, ends = _get_anomaly_bounds(label)
    if len(starts) == 0:
        score[T // 2] = 1.0
        return score
    gaps = []
    if starts[0] > 0:
        gaps.append((0, starts[0] - 1))
    for i in range(len(ends) - 1):
        gaps.append((ends[i] + 1, starts[i + 1] - 1))
    if ends[-1] < T - 1:
        gaps.append((ends[-1] + 1, T - 1))
    # prefer a gap that has a buffer-free zone; fall back to largest gap
    valid = [(gs, ge) for gs, ge in gaps if ge - gs >= 2 * slope_size]
    gs, ge = max(valid if valid else gaps, key=lambda g: g[1] - g[0])
    score[(gs + ge) // 2] = 1.0
    return score


def score_E2_missed(label, slope_size):
    """E-2: no prediction at all."""
    return np.zeros(len(label))


def score_E3_partial(label, slope_size):
    """E-3: only first half of each anomaly is flagged."""
    score  = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    for s, e in zip(starts, ends):
        L = e - s + 1
        score[s:s + max(1, L // 2)] = 1.0
    return score


def score_E4_early_over(label, slope_size):
    """E-4: contiguous segment starts before the left buffer boundary."""
    score  = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L  = int(round((ends - starts + 1).mean())) if len(starts) else 10
    over   = slope_size + avg_L // 2
    for s, e in zip(starts, ends):
        score[max(0, s - over):e + 1] = 1.0
    return score


def score_E5_late_over(label, slope_size):
    """E-5: contiguous segment ends after the right buffer boundary."""
    T      = len(label)
    score  = np.zeros(T)
    starts, ends = _get_anomaly_bounds(label)
    avg_L  = int(round((ends - starts + 1).mean())) if len(starts) else 10
    over   = slope_size + avg_L // 2
    for s, e in zip(starts, ends):
        score[s:min(T, e + over + 1)] = 1.0
    return score


DETECTION_SCORES = [
    ("C1", "C-1: Early (left buffer)",     score_C1_early),
    ("C2", "C-2: Late (right buffer)",     score_C2_late),
    ("C3", "C-3: Exact (inside anomaly)",  score_C3_exact),
    ("E1", "E-1: Isolated FP",             score_E1_isolated),
    ("E2", "E-2: Missed",                  score_E2_missed),
    ("E3", "E-3: Partial",                 score_E3_partial),
    ("E4", "E-4: Early Over-Prediction",   score_E4_early_over),
    ("E5", "E-5: Late Over-Prediction",    score_E5_late_over),
]


# ---------------------------------------------------------------------------
# Per-(label, score) figure
# ---------------------------------------------------------------------------

def plot_pair(label, meta, score, score_key, score_display, vus, rng):
    """
    For a single (label, score) pair:
      - Compute cumulative VUS once.
      - Call find_adaptive_buffer once per denom (3 calls total).
        Runtime comes from @time_it on that call.
      - Run select_best_buffer for the remaining 2 methods per denom
        (reuses the already-computed ratio curve — no redundant denom work).
      - Plot a 5-row figure:
          Row 0: label
          Row 1: score (with anomaly regions shaded)
          Row 2: ratio curve for analytical denom
          Row 3: ratio curve for mc_bernoulli denom
          Row 4: ratio curve for mc_uniform denom
        Each ratio row marks all 3 method selections and annotates runtime.
    """
    T       = meta['T']
    buffers = vus.slope_values

    # Slopes array — computed outside compute() so find_adaptive_buffer can
    # be called independently. Consistent with global_mask=False (no masking).
    ((start_ne, end_ne), _), _ = vus.get_anomalies_coordinates(label)
    labels_arr = vus.add_slopes_precomputed(label, start_ne, end_ne)

    (vus_pr, _), _ = vus.compute(label, score, adaptive=False)

    # One find_adaptive_buffer call per denom (runtime measured here).
    # Then reuse the returned ratio curve for the other two methods.
    denom_data   = {}  # denom → (ratio, denom_curve, runtime_s)
    method_bests = {}  # (denom, method) → best dict

    for denom in DENOMS:
        res, rt = vus.find_adaptive_buffer(
            labels_arr, score, vus_pr, T,
            denom=denom, method=METHODS[0], rng=rng,
        )
        ratio       = res['ratio']
        denom_curve = res['denom']
        denom_data[denom] = (ratio, denom_curve, rt)

        method_bests[(denom, METHODS[0])] = res['best']
        for method in METHODS[1:]:
            best = select_best_buffer(ratio, vus.slope_values, method=method)
            best['vus_score'] = float(vus_pr[best['buffer_idx']])
            method_bests[(denom, method)] = best

    # ---- figure ----
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(
        f"[{meta['tag']}]  T={T}  L={meta['L']}  N={meta['N']}  "
        f"G={meta['G']}  p={meta['p']:.4f}   |   Score: {score_display}",
        fontsize=10, fontweight='bold',
    )

    # Row 0: score + label shading + selected buffer zones per method
    ax = axes[0]
    anom_starts, anom_ends = _get_anomaly_bounds(label)
    for s, e in zip(anom_starts, anom_ends):
        ax.axvspan(s, e, color='red', alpha=0.15, linewidth=0)
    ax.plot(score, linewidth=0.9, color='steelblue')
    for method in METHODS:
        best = method_bests[('analytical', method)]
        b    = best['buffer_val']
        col  = METHOD_COLORS[method]
        for s, e in zip(anom_starts, anom_ends):
            ax.axvline(max(0, s - b),        color=col, linestyle=':', linewidth=1.2, alpha=0.9)
            ax.axvline(min(T - 1, e + b), color=col, linestyle=':', linewidth=1.2, alpha=0.9)

    handles = [
        plt.matplotlib.lines.Line2D(
            [], [], color='steelblue', linestyle='-', linewidth=0.9,
            label='score',
        ),
        plt.matplotlib.patches.Patch(
            facecolor='red', alpha=0.3, label='label',
        ),
        *[
            plt.matplotlib.lines.Line2D(
                [], [], color=METHOD_COLORS[m], linestyle=':', linewidth=1.2,
                label=f"{m}  b*={method_bests[('analytical', m)]['buffer_val']}",
            )
            for m in METHODS
        ],
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right')
    ax.set_ylabel(score_key, fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.tick_params(labelsize=7)

    # Rows 1–3: one ratio curve per denom
    for row_i, denom in enumerate(DENOMS, start=1):
        ratio, _, rt = denom_data[denom]
        ax = axes[row_i]

        ax.plot(buffers, ratio, linewidth=1.5, **DENOM_STYLES[denom], label=denom)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

        for method in METHODS:
            best = method_bests[(denom, method)]
            bval = best['buffer_val']
            rval = best['ratio']
            col  = METHOD_COLORS[method]
            ax.axvline(x=bval, color=col, linestyle=':', linewidth=1.2, alpha=0.8)
            ax.scatter([bval], [rval], color=col, zorder=5, s=45,
                       label=f"{method}  b*={bval}  vus={best['vus_score']:.3f}")

        ax.set_title(f"{denom}   runtime: {rt * 1000:.1f} ms", fontsize=8)
        ax.set_ylabel("Ratio", fontsize=8)
        ax.set_xlabel("Buffer" if row_i == 3 else "")
        ax.set_xlim(0, int(buffers[-1]))
        ax.legend(fontsize=7, ncol=4, loc='upper right')
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    vus = VUSNumpy(slope_size=SLOPE_SIZE, step=1, cumulative=True, global_mask=False, slopes='precomputed')

    for cfg in LABEL_CONFIGS:
        label, meta = make_label(cfg['T'], cfg['L'], cfg['N'], cfg['G'])
        meta['tag'] = cfg['tag']

        for score_key, score_display, score_fn in DETECTION_SCORES:
            score = score_fn(label, SLOPE_SIZE)
            print(f"[{meta['tag']}] {score_key}  score_sum={score.sum():.0f}")
            plot_pair(label, meta, score, score_key, score_display, vus, rng)
