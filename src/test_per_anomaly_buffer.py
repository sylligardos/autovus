"""
Per-Anomaly Adaptive Buffer — Test Script

Two modes:
  python src/test_per_anomaly_buffer.py           # unit tests (synthetic)
  python src/test_per_anomaly_buffer.py --real    # real TSB-AD analysis

Real-data mode loads the most "interesting" series (those with the most
varied anomaly lengths), runs compute_per_anomaly_adaptive_slicing for every detector,
and saves one PNG per (series, detector) to experiments/per_anomaly_buffer/.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vus.vus_numpy import VUSNumpy


# ---------------------------------------------------------------------------
# SYNTHETIC UNIT TESTS
# ---------------------------------------------------------------------------

SLOPE_SIZE = 256
T = 1000
vus = VUSNumpy(slope_size=SLOPE_SIZE, cumulative=True)
_PLOT = False       # set to True via --show at runtime
_PER_SIDE = False   # default per_side for score tests; set via --per-side at runtime


def make_label(anomalies, length=T):
    label = np.zeros(length)
    for s, e in anomalies:
        label[s:e + 1] = 1
    return label


def offset_score(label, offset, noise=0.05, seed=0):
    """Detector that fires `offset` steps after each anomaly segment."""
    rng = np.random.default_rng(seed)
    score = np.zeros_like(label, dtype=float)
    if offset > 0:
        score[offset:] = label[:-offset]
    elif offset < 0:
        score[:offset] = label[-offset:]
    else:
        score[:] = label
    score += rng.normal(0, noise, size=len(label))
    return np.clip(score, 0, None)


# ---------------------------------------------------------------------------
# Label configs — each stresses one structural property (N=1 except
# dense_overlap which requires N=2 by definition)
# ---------------------------------------------------------------------------

LABEL_CONFIGS = [
    dict(anomalies=[(400, 419)],              tag="baseline",     L=20),
    dict(anomalies=[(400, 404)],              tag="short_anom",   L=5),
    dict(anomalies=[(300, 359)],              tag="long_anom",    L=60),
    dict(anomalies=[(400, 402)],              tag="tiny_sparse",  L=3),
    dict(anomalies=[(200, 279)],              tag="wide_sparse",  L=80),
    dict(anomalies=[(300, 319), (345, 364)],  tag="dense_overlap", L=20),
]


# ---------------------------------------------------------------------------
# Score helpers — one anomaly at a time
# ---------------------------------------------------------------------------

def _get_anomaly_bounds(label):
    diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends   = np.where(diff < 0)[0] - 1
    return starts, ends


def score_C1_early(label):
    """C-1: fires in the left buffer zone only (before the anomaly starts)."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L   = int(round((ends - starts + 1).mean())) if len(starts) else 10
    seg_len = max(1, avg_L // 2)
    for s in starts:
        seg_end   = s - 1
        seg_start = max(0, seg_end - seg_len + 1)
        if seg_end >= 0:
            score[seg_start:seg_end + 1] = 1.0
    return score


def score_C2_late(label):
    """C-2: fires in the right buffer zone only (after the anomaly ends)."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L   = int(round((ends - starts + 1).mean())) if len(starts) else 10
    seg_len = max(1, avg_L // 2)
    for e in ends:
        seg_start = e + 1
        seg_end   = min(len(label) - 1, seg_start + seg_len - 1)
        if seg_start < len(label):
            score[seg_start:seg_end + 1] = 1.0
    return score


def score_C3_exact(label):
    """C-3: fires squarely inside the anomaly, no buffer overlap."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    for s, e in zip(starts, ends):
        L = e - s + 1
        q = max(1, L // 4)
        inner_s = s + q
        inner_e = e - q + 1
        if inner_s < inner_e:
            score[inner_s:inner_e] = 1.0
        else:
            score[s:e + 1] = 1.0
    return score


def score_E1_isolated(label):
    """E-1: isolated false positive well away from any anomaly."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    if len(starts) == 0:
        score[len(label) // 2] = 1.0
        return score
    gaps = []
    if starts[0] > 0:
        gaps.append((0, starts[0] - 1))
    for i in range(len(ends) - 1):
        gaps.append((ends[i] + 1, starts[i + 1] - 1))
    if ends[-1] < len(label) - 1:
        gaps.append((ends[-1] + 1, len(label) - 1))
    valid = [(gs, ge) for gs, ge in gaps if ge - gs >= 2 * SLOPE_SIZE]
    gs, ge = max(valid if valid else gaps, key=lambda g: g[1] - g[0])
    score[(gs + ge) // 2] = 1.0
    return score


def score_E2_missed(label):
    """E-2: no prediction at all."""
    return np.zeros(len(label))


def score_E3_partial(label):
    """E-3: only first half of each anomaly is flagged."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    for s, e in zip(starts, ends):
        L = e - s + 1
        score[s:s + max(1, L // 2)] = 1.0
    return score


def score_E4_early_over(label):
    """E-4: contiguous segment starts before the left buffer boundary."""
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L = int(round((ends - starts + 1).mean())) if len(starts) else 10
    over  = SLOPE_SIZE + avg_L // 2
    for s, e in zip(starts, ends):
        score[max(0, s - over):e + 1] = 1.0
    return score


def score_E5_late_over(label):
    """E-5: contiguous segment ends after the right buffer boundary.

    Score is 1.0 inside the anomaly, then linearly decays to 0 over the
    over-region (avoids a flat constant score which degenerates the PR curve).
    """
    score = np.zeros(len(label))
    starts, ends = _get_anomaly_bounds(label)
    avg_L = int(round((ends - starts + 1).mean())) if len(starts) else 10
    over  = SLOPE_SIZE + avg_L // 2
    for s, e in zip(starts, ends):
        score[s:e + 1] = 1.0
        decay_start = e + 1
        decay_end   = min(len(label), e + 1 + over)
        n = decay_end - decay_start
        if n > 0:
            score[decay_start:decay_end] = np.linspace(1.0, 0.0, n)
    return score


# ---------------------------------------------------------------------------
# Original structural tests (1–5)
# ---------------------------------------------------------------------------

def test_single_anomaly_matches_global():
    print("\n=== Test 1: single anomaly matches global adaptive ===")
    label = make_label([(300, 350)])
    score = offset_score(label, offset=5)

    (global_result, _), _ = vus.compute(label, score, adaptive=True)
    per_anom = vus.compute_per_anomaly_adaptive_slicing(label, score, plot=_PLOT, per_side=_PER_SIDE)

    lag = 5
    assert len(per_anom) == 1, f"Expected 1 result, got {len(per_anom)}"
    assert per_anom[0]['window'] == (0, T), f"Single anomaly window should span full series (0, {T}), got {per_anom[0]['window']}"
    if not _PER_SIDE:
        # 'vus' is the both-sided curve only in symmetric mode; per-side stores
        # one-sided curves under vus_left/vus_right instead.
        assert np.allclose(per_anom[0]['vus'], global_result['vus']), "VUS curves must match for single anomaly (same window = full series)"
    b = per_anom[0]['best']['buffer_val']
    tol = lag // 2
    assert abs(b - lag) <= tol, f"lag={lag}  b*={b}  expected b* in [{lag - tol}, {lag + tol}]"

    print(f"  b* = {b}  (global b*={global_result['best']['buffer_val']})")
    print("  PASS")


def test_window_bounds_slicing():
    print("\n=== Test 2: window bounds are midpoints to neighbors (slicing) ===")
    anomalies = [(100, 110), (300, 320), (600, 650)]
    label = make_label(anomalies)
    score = offset_score(label, offset=0)

    per_anom = vus.compute_per_anomaly_adaptive_slicing(label, score, plot=_PLOT, per_side=_PER_SIDE)

    assert per_anom[0]['window'] == (0, 205),   f"Got {per_anom[0]['window']}"
    assert per_anom[1]['window'] == (205, 460), f"Got {per_anom[1]['window']}"
    assert per_anom[2]['window'] == (460, T),   f"Got {per_anom[2]['window']}"

    for k, r in enumerate(per_anom):
        b = r['best']['buffer_val']
        print(f"  anomaly {k}: [{r['start']}, {r['end']}]  window={r['window']}  b*={b}")
        assert b == 0, f"anomaly {k}: b*={b}, expected 0 (exact detection, no buffer needed)"

    print("  PASS")


def test_no_anomalies():
    print("\n=== Test 3: no anomalies returns [] ===")
    label = np.zeros(T)
    score = np.random.default_rng(0).uniform(0, 1, T)
    result = vus.compute_per_anomaly_adaptive_slicing(label, score, per_side=_PER_SIDE)
    assert result == [], f"Expected [], got {result}"
    print("  PASS")


def test_requires_cumulative():
    print("\n=== Test 4: non-cumulative instance raises ValueError ===")
    vus_nc = VUSNumpy(slope_size=SLOPE_SIZE, cumulative=False)
    label = make_label([(200, 210)])
    score = offset_score(label, offset=0)
    try:
        vus_nc.compute_per_anomaly_adaptive_slicing(label, score)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"  Caught: {e}")
    print("  PASS")


def test_length_vs_buffer():
    print("\n=== Test 5: offset detector — b* vs anomaly length ===")
    lag = 15
    tol = lag // 2
    anomalies = [(150, 150), (400, 409), (700, 749)]
    label = make_label(anomalies)
    score = offset_score(label, offset=lag)

    per_anom = vus.compute_per_anomaly_adaptive_slicing(label, score, plot=_PLOT, per_side=_PER_SIDE)

    expectations = [
        (lag, f"b*≈{lag} (point anomaly, late detection within L is included — lag={lag})"),
        (lag, f"b*≈{lag} (lag={lag})"),
        (lag, f"b*≈{lag} (lag={lag})"),
    ]

    print(f"  {'anomaly':>10}  {'length':>8}  {'b*':>6}  result")
    for k, (r, (expected, desc)) in enumerate(zip(per_anom, expectations)):
        b      = r['best']['buffer_val']
        length = r['end'] - r['start'] + 1
        if expected == 0:
            ok  = (b == 0)
            why = f"expected 0, got b*={b}" if not ok else "PASS"
        else:
            ok  = abs(b - expected) <= tol
            why = f"lag={lag}  b*={b}  expected b* in [{expected - tol}, {expected + tol}]" if not ok else "PASS"
        print(f"  {k:>10}  {length:>8}  {b:>6}  {why}")
        assert ok, why

    print("  PASS")


# ---------------------------------------------------------------------------
# Score-type tests (6–13) — one per detection pattern, across all label configs
# ---------------------------------------------------------------------------

def _run_score_test(score_fn, label_configs, check_fn, test_name, per_side=None):
    # per_side=None -> follow the runtime default (_PER_SIDE, set by --per-side).
    # Tests that guard a symmetric-only property pass per_side=False explicitly.
    if per_side is None:
        per_side = _PER_SIDE
    print(f"\n=== {test_name} ===")
    failures = []
    for cfg in label_configs:
        label    = make_label(cfg['anomalies'])
        score    = score_fn(label)
        per_anom = vus.compute_per_anomaly_adaptive_slicing(label, score, plot=_PLOT, per_side=per_side)
        if len(per_anom) != len(cfg['anomalies']):
            msg = f"[{cfg['tag']}] Expected {len(cfg['anomalies'])} result(s), got {len(per_anom)}"
            print(f"  FAIL  {msg}")
            failures.append(msg)
            continue
        for k, r in enumerate(per_anom):
            b = r['best']['buffer_val']
            try:
                check_fn(b, cfg)
                print(f"  [{cfg['tag']}] anomaly {k}  b*={b}")
            except AssertionError as e:
                print(f"  [{cfg['tag']}] anomaly {k}  b*={b}  FAIL: {e}")
                failures.append(str(e))
    if failures:
        print(f"  FAIL ({len(failures)} failure(s))")
    else:
        print("  PASS")


def test_score_C1_early():
    """C-1: score fires just before the anomaly — b* must be >= L/2 to capture it."""
    def check(b, cfg):
        min_b = max(1, cfg['L'] // 2)
        assert b >= min_b, f"[{cfg['tag']}] b*={b} < {min_b}, buffer too small to reach early detection"
    _run_score_test(score_C1_early, LABEL_CONFIGS, check, "Test 6: C-1 Early detection")


def test_score_C2_late():
    """C-2: score fires just after the anomaly — b* must be >= L/2 to capture it."""
    def check(b, cfg):
        min_b = max(1, cfg['L'] // 2)
        assert b >= min_b, f"[{cfg['tag']}] b*={b} < {min_b}, buffer too small to reach late detection"
    _run_score_test(score_C2_late, LABEL_CONFIGS, check, "Test 7: C-2 Late detection")


def test_score_C3_exact():
    """C-3: score is strictly inside the anomaly — b*=0 is sufficient."""
    def check(b, cfg):
        assert b == 0, f"[{cfg['tag']}] b*={b}, expected 0 (exact detection needs no buffer)"
    _run_score_test(score_C3_exact, LABEL_CONFIGS, check, "Test 8: C-3 Exact detection")


def test_score_E1_isolated_fp():
    """E-1: isolated FP placed beyond any reachable buffer — b* should be 0."""
    def check(b, cfg):
        assert b == 0, f"[{cfg['tag']}] b*={b}, expected 0 (FP is unreachable, buffer should not chase it)"
    _run_score_test(score_E1_isolated, LABEL_CONFIGS, check, "Test 9: E-1 Isolated FP")


def test_score_E2_missed():
    """E-2: all-zero score — no crash and b*=0 (flat score, nothing detected)."""
    def check(b, cfg):
        assert b == 0, f"[{cfg['tag']}] b*={b}, expected 0 (missed detection, flat score)"
    _run_score_test(score_E2_missed, LABEL_CONFIGS, check, "Test 10: E-2 Missed detection")


def test_score_E3_partial():
    """E-3: first half of anomaly flagged — detection is inside, b*=0 sufficient."""
    def check(b, cfg):
        assert b == 0, f"[{cfg['tag']}] b*={b}, expected 0 (partial detection is inside anomaly)"
    _run_score_test(score_E3_partial, LABEL_CONFIGS, check, "Test 11: E-3 Partial detection")


def test_score_E4_early_over():
    """E-4: score extends beyond the left buffer boundary — b* should be small.

    The over-prediction bound is a symmetric-mode property; under --per-side the
    flooded side may legitimately absorb the over-prediction into its buffer, so
    we only report the b* values there (no verdict) and assert the bound when
    symmetric.
    """
    def check(b, cfg):
        if not _PER_SIDE:
            assert b <= SLOPE_SIZE // 2, \
                f"[{cfg['tag']}] b*={b}, over-prediction should not inflate b* beyond SLOPE_SIZE/2={SLOPE_SIZE // 2}"
    _run_score_test(score_E4_early_over, LABEL_CONFIGS, check, "Test 12: E-4 Early over-prediction")


def test_score_E5_late_over():
    """E-5: score decays past the right buffer boundary — b* should be small.

    See test_score_E4_early_over: bound asserted in symmetric mode only.
    """
    def check(b, cfg):
        if not _PER_SIDE:
            assert b <= SLOPE_SIZE // 2, \
                f"[{cfg['tag']}] b*={b}, over-prediction should not inflate b* beyond SLOPE_SIZE/2={SLOPE_SIZE // 2}"
    _run_score_test(score_E5_late_over, LABEL_CONFIGS, check, "Test 13: E-5 Late over-prediction")


# ---------------------------------------------------------------------------
# Detection taxonomy (early/late/exact/missed) — validated against the
# synthetic scores whose timing is known by construction.
# ---------------------------------------------------------------------------

def test_detection_taxonomy():
    """classify_detection labels each synthetic score by its true timing.

    Runs in per_side mode (direction needs the two-sided b*). Each score fn
    has a known ground-truth label; assert the stamped detection_type matches.
    """
    print("\n=== Test 14: detection taxonomy (per_side) ===")
    cases = [
        (score_C1_early,    'early'),
        (score_C2_late,     'late'),
        (score_C3_exact,    'exact'),
        (score_E1_isolated, 'missed'),
        (score_E2_missed,   'missed'),
        (score_E3_partial,  'exact'),
    ]
    failures = []
    for score_fn, expected in cases:
        for cfg in LABEL_CONFIGS:
            label    = make_label(cfg['anomalies'])
            score    = score_fn(label)
            per_anom = vus.compute_per_anomaly_adaptive_slicing(label, score, per_side=True)
            for k, r in enumerate(per_anom):
                got = r['detection_type']
                ok  = (got == expected)
                tag = "PASS" if ok else f"FAIL expected {expected}"
                print(f"  {score_fn.__name__:<18} [{cfg['tag']:<13}] anomaly {k}  -> {got:<7} {tag}")
                if not ok:
                    failures.append(f"{score_fn.__name__} [{cfg['tag']}] anomaly {k}: got {got}, expected {expected}")
    # Symmetric mode collapses early/late into 'lagged'.
    label = make_label([(400, 419)])
    r_sym = vus.compute_per_anomaly_adaptive_slicing(label, score_C2_late(label), per_side=False)[0]
    if r_sym['detection_type'] != 'lagged':
        failures.append(f"symmetric late: got {r_sym['detection_type']}, expected lagged")
    print(f"  symmetric late -> {r_sym['detection_type']} "
          f"({'PASS' if r_sym['detection_type'] == 'lagged' else 'FAIL expected lagged'})")
    print("  PASS" if not failures else f"  FAIL ({len(failures)} failure(s))")
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_structural_test(fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")


def run_unit_tests(plot=False, slope_size=None, per_side=False):
    global _PLOT, _PER_SIDE, SLOPE_SIZE, vus
    _PLOT = plot
    _PER_SIDE = per_side
    if slope_size is not None and slope_size != SLOPE_SIZE:
        SLOPE_SIZE = slope_size
        vus = VUSNumpy(slope_size=SLOPE_SIZE, cumulative=True)
    _run_structural_test(test_single_anomaly_matches_global)
    _run_structural_test(test_window_bounds_slicing)
    _run_structural_test(test_no_anomalies)
    _run_structural_test(test_requires_cumulative)
    _run_structural_test(test_length_vs_buffer)
    test_score_C1_early()
    test_score_C2_late()
    test_score_C3_exact()
    test_score_E1_isolated_fp()
    test_score_E2_missed()
    test_score_E3_partial()
    test_score_E4_early_over()
    test_score_E5_late_over()
    _run_structural_test(test_detection_taxonomy)
    print("\nAll unit tests done.")


# ---------------------------------------------------------------------------
# REAL DATA
# ---------------------------------------------------------------------------

def _dataset_tag(fname):
    return fname.split('_')[1]


def _get_anomaly_lengths(label):
    diff = np.diff(np.concatenate([[0], label.astype(int), [0]]))
    starts = np.where(diff > 0)[0]
    ends = np.where(diff < 0)[0] - 1
    if len(starts) == 0:
        return np.array([], dtype=int)
    return (ends - starts + 1).astype(int)


def _select_interesting(fnames, labels, min_anomalies=3, n_top=5, dataset_filter=None):
    """Pick series with >= min_anomalies and the most varied anomaly lengths."""
    candidates = []
    for i, (fn, lbl) in enumerate(zip(fnames, labels)):
        if dataset_filter and _dataset_tag(fn) != dataset_filter:
            continue
        lengths = _get_anomaly_lengths(lbl)
        if len(lengths) < min_anomalies:
            continue
        diversity = float(lengths.std() / (lengths.mean() + 1e-8))
        candidates.append((i, fn, lengths, diversity))

    candidates.sort(key=lambda x: -x[3])
    return candidates[:n_top]


def plot_per_anomaly_buffers(fname, timeseries, score, detector, per_anom, out_path, show=False, global_vus=None, mean_vus_adapt=None, per_side=False):
    T = len(timeseries)
    dataset = _dataset_tag(fname)

    fig, (ax_ts, ax_sc) = plt.subplots(
        2, 1, figsize=(16, 6), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )
    final_vus = per_anom[0].get('final_vus') if per_anom else None
    final_vus_str = f"  |  VUS (per-anomaly caps)={final_vus:.4f}" if final_vus is not None else ""
    global_vus_str = f"  |  VUS (no caps)={global_vus:.4f}" if global_vus is not None else ""
    mean_vus_str = f"  |  mean VUS (per-anomaly)={mean_vus_adapt:.4f}" if mean_vus_adapt is not None else ""
    fig.suptitle(
        f"[{dataset}]  {fname}  |  detector: {detector}{global_vus_str}{mean_vus_str}{final_vus_str}",
        fontsize=9, fontweight='bold',
    )

    ax_ts.plot(timeseries, linewidth=0.6, color='dimgray', zorder=2)

    ts_max = float(timeseries.max())
    ts_min = float(timeseries.min())
    label_y = ts_max + 0.02 * (ts_max - ts_min or 1.0)

    for r in per_anom:
        s, e = r['start'], r['end']
        length = e - s + 1
        ax_ts.axvspan(s, e + 1, color='red', alpha=0.25, linewidth=0, zorder=3)
        mid = (s + e) / 2.0
        dtype = r.get('detection_type', '')
        if per_side:
            # Asymmetric buffers: left (orange) sized by b*_left, right (green) by b*_right.
            b_left   = r['best_left']['buffer_val']
            b_right  = r['best_right']['buffer_val']
            vus_l    = float(r['best_left']['vus_score'])
            vus_r    = float(r['best_right']['vus_score'])
            if b_left > 0:
                ax_ts.axvspan(max(0, s - b_left), s, color='orange', alpha=0.25, linewidth=0, zorder=2)
            if b_right > 0:
                ax_ts.axvspan(e + 1, min(T, e + 1 + b_right), color='seagreen', alpha=0.25, linewidth=0, zorder=2)
            ax_ts.annotate(
                f'{dtype.upper()}\nL={b_left}  R={b_right}\nlen={length}\nvL={vus_l:.3f}  vR={vus_r:.3f}',
                xy=(mid, label_y), xycoords='data',
                fontsize=6, ha='center', va='bottom',
                color='darkorange', fontweight='bold',
            )
        else:
            b = r['best']['buffer_val']
            if b > 0:
                if s - b >= 0:
                    ax_ts.axvspan(max(0, s - b), s, color='orange', alpha=0.25, linewidth=0, zorder=2)
                if e + 1 + b <= T:
                    ax_ts.axvspan(e + 1, min(T, e + 1 + b), color='orange', alpha=0.25, linewidth=0, zorder=2)
            vus_score = float(r['vus'][r['best']['buffer_idx']])
            ax_ts.annotate(
                f'{dtype.upper()}\nb*={b}\nlen={length}\nvus={vus_score:.3f}',
                xy=(mid, label_y), xycoords='data',
                fontsize=6, ha='center', va='bottom',
                color='darkorange', fontweight='bold',
            )

    if per_side:
        legend_handles = [
            plt.matplotlib.patches.Patch(facecolor='red',      alpha=0.3, label='anomaly'),
            plt.matplotlib.patches.Patch(facecolor='orange',   alpha=0.3, label='left buffer (b*_left)'),
            plt.matplotlib.patches.Patch(facecolor='seagreen', alpha=0.3, label='right buffer (b*_right)'),
        ]
    else:
        legend_handles = [
            plt.matplotlib.patches.Patch(facecolor='red',    alpha=0.3, label='anomaly'),
            plt.matplotlib.patches.Patch(facecolor='orange', alpha=0.3, label='buffer (b*)'),
        ]
    ax_ts.legend(handles=legend_handles, fontsize=7, loc='upper right')
    ax_ts.set_ylabel('value', fontsize=8)
    ax_ts.tick_params(labelsize=7, labelbottom=False)
    ax_ts.set_xlim(0, T - 1)

    ax_sc.plot(score, linewidth=0.7, color='steelblue', zorder=2)
    for r in per_anom:
        ax_sc.axvspan(r['start'], r['end'] + 1, color='red', alpha=0.15, linewidth=0, zorder=3)
    ax_sc.set_ylabel('score', fontsize=8)
    ax_sc.set_xlabel('time', fontsize=8)
    ax_sc.tick_params(labelsize=7)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)


def run_real_data(args):
    from tsbadparser import TSBADParser

    slope_size = args.slope_size
    vus_real = VUSNumpy(slope_size=slope_size, step=1, cumulative=True, global_mask=False)

    tsb = TSBADParser()
    detectors = tsb.get_detectors()

    if args.detector:
        if args.detector not in detectors:
            print(f"Unknown detector '{args.detector}'. Available: {detectors}")
            return
        detectors = [args.detector]

    print("Loading time series...")
    load_kwargs = {'dataset': [args.dataset]} if args.dataset else {'load_all': True}
    fnames, timeseries, labels = tsb.load_timeseries(**load_kwargs)

    if args.list_series:
        print(f"\n{len(fnames)} series available:")
        for i, fn in enumerate(fnames):
            lengths = _get_anomaly_lengths(labels[i])
            print(f"  {fn}  n_anom={len(lengths)}  lengths={sorted(lengths.tolist())}")
        return

    if args.series:
        matched = [(i, fn) for i, fn in enumerate(fnames) if args.series in fn]
        if not matched:
            print(f"No series matching '{args.series}'. Use --list-series to see available names.")
            return
        selected = [(i, fn, _get_anomaly_lengths(labels[i]), 0.0) for i, fn in matched]
        print(f"\nSelected {len(selected)} series matching '{args.series}':")
        for i, fn, lengths, _ in selected:
            print(f"  {fn}  n_anom={len(lengths)}  lengths={sorted(lengths.tolist())}")
    else:
        selected_raw = _select_interesting(
            fnames, labels,
            min_anomalies=args.min_anomalies,
            n_top=args.n_interesting,
        )
        if not selected_raw:
            print("No interesting series found. Try lowering --min-anomalies.")
            return
        fname_to_idx = {fn: i for i, fn in enumerate(fnames)}
        selected = [(fname_to_idx[fn], fn, lengths, diversity) for _, fn, lengths, diversity in selected_raw]
        print(f"\nSelected {len(selected)} interesting series:")
        for _, fn, lengths, diversity in selected:
            print(f"  {fn}  n_anom={len(lengths)}  lengths={sorted(lengths.tolist())}  diversity={diversity:.2f}")

    print(f"Loading scores for {len(detectors)} detectors...")
    scores, _ = tsb.load_scores_and_delete(fnames, timeseries, labels, detectors=detectors)

    out_base = args.output_dir
    fname_to_idx = {fn: i for i, fn in enumerate(fnames)}

    for _, fn, lengths, _ in selected:
        idx = fname_to_idx.get(fn)
        if idx is None:
            print(f"\n  Skipping {fn} (dropped during score loading)")
            continue

        ts  = timeseries[idx]
        lbl = labels[idx]
        sc  = scores[idx]
        fname_safe = fn.replace('/', '_')

        print(f"\n  {fn}  [{len(lengths)} anomalies, lengths={sorted(lengths.tolist())}]")

        for det_idx, det in enumerate(detectors):
            score = sc[det_idx]
            try:
                per_anom = vus_real.compute_per_anomaly_adaptive_slicing(lbl, score, per_side=args.per_side)
            except Exception as ex:
                print(f"    [{det}] ERROR: {ex}")
                continue

            final_vus = per_anom[0].get('final_vus') if per_anom else None
            mean_vus_adapt = float(np.mean([r['best']['vus_score'] for r in per_anom])) if per_anom else None

            (global_vus_curve, _), _ = vus_real.compute(lbl, score)
            global_vus = float(global_vus_curve[-1])

            if args.per_side:
                b_str = "b*(L,R) = " + str([(r['best_left']['buffer_val'], r['best_right']['buffer_val']) for r in per_anom])
            else:
                b_str = "b* = " + str([r['best']['buffer_val'] for r in per_anom])
            final_vus_str = f"  VUS(caps)={final_vus:.4f}" if final_vus is not None else ""
            mean_vus_str = f"  mean_vus_adapt={mean_vus_adapt:.4f}" if mean_vus_adapt is not None else ""
            types_str = "  types=" + str([r.get('detection_type') for r in per_anom]) if per_anom else ""
            print(f"    [{det}] {b_str}  VUS(no caps)={global_vus:.4f}{mean_vus_str}{final_vus_str}{types_str}")

            if args.show_detail:
                for k, r in enumerate(per_anom):
                    if args.per_side:
                        vus_real._plot_anomaly_debug_per_side(lbl, score, r, k, len(per_anom), len(lbl), timeseries=ts)
                    else:
                        vus_real._plot_anomaly_debug(lbl, score, r, k, len(per_anom), len(lbl), timeseries=ts)

            out_path = os.path.join(out_base, fname_safe, f"{det}.png")
            plot_per_anomaly_buffers(fn, ts, score, det, per_anom, out_path, show=args.show, global_vus=global_vus, mean_vus_adapt=mean_vus_adapt, per_side=args.per_side)

    print(f"\nAll plots saved to {os.path.abspath(out_base)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Per-anomaly adaptive buffer tests")
    p.add_argument('--real',           action='store_true',
                   help='Run real TSB-AD analysis instead of unit tests')
    p.add_argument('--show',           action='store_true',
                   help='Display overview plots interactively instead of saving them to disk')
    p.add_argument('--show-detail',    action='store_true',
                   help='Display per-anomaly debug plots (ratio curve, b* marked) one by one. '
                        'With --per-side, shows the left and right ratio curves side by side.')
    p.add_argument('--per-side',       action='store_true',
                   help='Activate per-side adaptive buffer selection (a separate b* for each '
                        'side of every anomaly). Applies to --real, --show, --show-detail and '
                        'the unit tests; omit it to use the symmetric single-b* selection.')
    p.add_argument('--dataset',        type=str, default=None,
                   help='Filter to one dataset, e.g. YAHOO, NAB, UCR')
    p.add_argument('--series',         type=str, default=None,
                   help='Select a specific time series by substring match against its filename '
                        '(e.g. "260" or "id_3"). Bypasses --n-interesting selection. '
                        'Multiple matches are all processed; use --list-series to see available names.')
    p.add_argument('--list-series',    action='store_true',
                   help='List all available series filenames (after applying --dataset filter) and exit.')
    p.add_argument('--detector',       type=str, default=None,
                   help='Filter to one detector. If omitted, all detectors are loaded.')
    p.add_argument('--n-interesting',  type=int, default=5,
                   help='Number of interesting series to analyse (default 5)')
    p.add_argument('--min-anomalies',  type=int, default=3,
                   help='Minimum number of anomalies for a series to be considered (default 3)')
    p.add_argument('--slope-size',     type=int, default=256,
                   help='Max buffer to sweep in VUS (default 256)')
    p.add_argument('--output-dir',     type=str,
                   default=os.path.join(
                       os.path.dirname(__file__), '..', 'experiments', 'per_anomaly_buffer'
                   ),
                   help='Output directory for plots')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.real:
        run_real_data(args)
    else:
        run_unit_tests(plot=args.show, slope_size=args.slope_size, per_side=args.per_side)
