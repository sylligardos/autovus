"""
Interactive VUS explorer (Streamlit).

Run with:
    conda run -n ffvus streamlit run src/demo_vus.py
"""

import base64
import io
import json
import os
import sys
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from PIL import Image
from plotly.subplots import make_subplots as _plotly_subplots
from scipy.stats import kendalltau

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_SRC_DIR, '..')
sys.path.insert(0, _SRC_DIR)

from tsbadparser import TSBADParser
from vus.vus_numpy import VUSNumpy

try:
    from vus.vus_torch import VUSTorch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_dataset_names():
    return sorted(TSBADParser().get_datasets())


@st.cache_data
def get_file_list(dataset: str) -> list[str]:
    tsb = TSBADParser()
    return sorted(tsb.meta[tsb.meta['dataset'] == dataset]['filename'].tolist())


@st.cache_data
def get_detector_names() -> list[str]:
    return TSBADParser().get_detectors()


@st.cache_data
def load_timeseries(fname: str):
    tsb = TSBADParser()
    _, ts, labels = tsb.load_timeseries(filename=[fname])
    return ts[0], labels[0].astype(int)


@st.cache_data
def load_score(fname: str, detector: str):
    scores, _ = TSBADParser().load_scores(detectors=[detector], filenames=[fname])
    if not scores:
        return None
    return scores[0][0]  # scores[0] is (1, T); first detector row → (T,)


_DISPLAY_MAX_POINTS = 6000   # max points kept when plotting a long line trace
_HEATMAP_MAX_COLS   = 4000   # localization-map heatmap raster width cap


def _minmax_downsample(ts: np.ndarray, max_points: int = _DISPLAY_MAX_POINTS):
    """
    Min/max-bucket downsample a 1-D signal to roughly `max_points` samples.
    Two samples per bucket (min + max, in original time order) so the line
    plot keeps visual envelope. Returns (x_indices, y_values).
    """
    n = len(ts)
    if n <= max_points:
        return np.arange(n, dtype=np.int64), ts
    n_buckets = max(1, max_points // 2)
    bucket_size = n // n_buckets
    if bucket_size < 2:
        return np.arange(n, dtype=np.int64), ts
    trim = n_buckets * bucket_size
    view = ts[:trim].reshape(n_buckets, bucket_size)
    base = np.arange(n_buckets) * bucket_size
    mins_idx = view.argmin(axis=1) + base
    maxs_idx = view.argmax(axis=1) + base
    pairs = np.stack([mins_idx, maxs_idx], axis=1)
    pairs.sort(axis=1)                         # keep original time order in pair
    x_out = pairs.ravel()
    return x_out.astype(np.int64), ts[x_out]


@st.cache_data
def load_timeseries_display(fname: str, max_points: int = _DISPLAY_MAX_POINTS):
    """Cached down-sampled (x, y, label) used by the preview / map line traces."""
    ts_values, label = load_timeseries(fname)
    x, y = _minmax_downsample(ts_values, max_points)
    return x, y, label


@st.cache_data
def load_score_display(fname: str, detector: str, max_points: int = _DISPLAY_MAX_POINTS):
    """Down-sampled detector score for plotting only."""
    score = load_score(fname, detector)
    if score is None:
        return None
    x, y = _minmax_downsample(score, max_points)
    return x, y


@st.cache_data
def compute_avg_label_cached(fname: str, slope_size: int):
    """Buffered label, independent of detector — one call per (fname, slope_size)."""
    _, label = load_timeseries(fname)
    # slopes='precomputed' is required: this function uses add_slopes_precomputed,
    # which reads self.pos_slopes (only built in precomputed mode). The class default
    # changed to 'function', so it must be requested explicitly here.
    vus = VUSNumpy(slope_size=slope_size, step=1, slopes='precomputed')
    # get_anomalies_coordinates is wrapped with @time_it → (result, elapsed)
    ((start_no_edges, end_no_edges), _), _ = vus.get_anomalies_coordinates(label)
    # `add_slopes_precomputed` (default mode) ignores `pos`, so we skip the
    # O(T × n_anomalies) distance scan that used to dominate this call on
    # long series like MITDB.
    labels = vus.add_slopes_precomputed(label, start_no_edges, end_no_edges)
    return labels.mean(axis=0)


@st.cache_data
def compute_score_norm_cached(fname: str, detector: str):
    """Min-max normalized score, independent of slope_size — survives slider moves."""
    score = load_score(fname, detector)
    if score is None:
        return None
    lo, hi = score.min(), score.max()
    return (score - lo) / (hi - lo + 1e-8)


@st.cache_data
def compute_vus_curve(fname: str, detector: str, slope_size: int, backend: str = 'numpy') -> np.ndarray | None:
    _, label = load_timeseries(fname)
    score = load_score(fname, detector)
    if score is None:
        return None
    try:
        if backend == 'torch':
            vus = VUSTorch(slope_size=slope_size, step=1, cumulative=True)
            (vus_pr, _), _ = vus.compute(label, score)
            return vus_pr.cpu().numpy()
        vus = VUSNumpy(slope_size=slope_size, step=1, cumulative=True)
        (cum_values, _), _ = vus.compute(label, score)
        return cum_values
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def delta_metric(curve) -> float:
    return float(curve[-1] - curve[0])


def auc_above_baseline(curve) -> float:
    """Normalized area between the VUS curve and the horizontal AUC-PR baseline."""
    return float(np.trapezoid(curve - curve[0]) / (len(curve) - 1))


@st.cache_data
def load_precomputed_existence():
    path = os.path.join(_ROOT_DIR, 'experiments', 'precomputed', 'existence_maps.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'files':       d['files'].tolist(),
        'datasets':    d['datasets'].tolist(),
        'detectors':   d['detectors'].tolist(),
        'L':           int(d['L']),
        'anom_curves': d['anom_curves'],     # object (n_files, n_det)
        'mean_curves': d['mean_curves'],     # float32 (n_files, n_det, L+1)
        'anom_starts': d['anom_starts'],     # object (n_files,)
        'anom_ends':   d['anom_ends'],       # object (n_files,)
        'n_anomalies': d['n_anomalies'],     # int32 (n_files,)
    }


@st.cache_data
def load_precomputed_adaptive():
    path = os.path.join(_ROOT_DIR, 'experiments', 'precomputed', 'rankings_L1024.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'curves':        d['curves'],
        'synth_curves':  d['synth_curves'],
        'denom':         d['denom'],
        'b_star':        d['b_star'],
        'synth_b_star':  d['synth_b_star'],
        'mean_anom_len': d['mean_anom_len'],
        'n_anomalies':   d['n_anomalies'],
        'files':         d['files'].tolist(),
        'datasets':      d['datasets'].tolist(),
        'detectors':     d['detectors'].tolist(),
        'synth_names':   d['synth_names'].tolist(),
        'L':             int(d['L']),
    }


@st.cache_data
def load_ranking_adapt():
    """Adjusted-VUS ranking results (Stage-1 CSV from compute_ranking_adapt.py).

    Long-format: one row per (series, detector) at L=256 over the 772 complete-case
    series. Adds a display-clipped Adjusted VUS column `avus` (= skill_adapt_A, the
    chance-adjusted headline metric, clipped to [0, 1]).
    """
    path = os.path.join(_ROOT_DIR, 'experiments', 'ranking_adapt', 'scores_all.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'note' in df.columns:
        df = df[df['note'].isna()]
    df = df.dropna(subset=['skill_adapt_A', 'vus_full'])
    df['avus'] = df['skill_adapt_A'].clip(lower=0.0)
    return df


@st.cache_data
def load_precomputed():
    path = os.path.join(_ROOT_DIR, 'experiments', 'precomputed', 'rankings.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'curves':    d['curves'],               # float32 (n_files, n_det, L+1)
        'files':     d['files'].tolist(),
        'datasets':  d['datasets'].tolist(),
        'detectors': d['detectors'].tolist(),
        'L':         int(d['L']),
    }


# ---------------------------------------------------------------------------
# Notes (persisted to notes/notes.json)
# ---------------------------------------------------------------------------

def _notes_path() -> str:
    return os.path.join(_ROOT_DIR, 'notes', 'notes.json')


def load_notes() -> list[dict]:
    path = _notes_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_notes(notes: list[dict]) -> None:
    path = _notes_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(notes, f, indent=2)


def add_note(dataset: str, fname: str, detector: str, buffer_size: int,
             tags: list[str], text: str) -> dict:
    notes = load_notes()
    note = {
        'id': uuid.uuid4().hex[:12],
        'created': datetime.now().isoformat(timespec='seconds'),
        'dataset': dataset,
        'fname': fname,
        'detector': detector,
        'buffer': int(buffer_size),
        'tags': tags,
        'text': text,
    }
    notes.append(note)
    save_notes(notes)
    return note


def update_note(note_id: str, **fields) -> None:
    notes = load_notes()
    for n in notes:
        if n.get('id') == note_id:
            n.update(fields)
            n['updated'] = datetime.now().isoformat(timespec='seconds')
            break
    save_notes(notes)


def delete_note(note_id: str) -> None:
    notes = [n for n in load_notes() if n.get('id') != note_id]
    save_notes(notes)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _contiguous_regions(mask: np.ndarray):
    """Return (start, end) index pairs for contiguous True runs in mask."""
    arr = mask.astype(int)
    changes = np.diff(arr, prepend=0, append=0)
    return list(zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))


def _rdylgn(n: int) -> list[str]:
    """n colors from RdYlGn, index 0 = best (green), index n-1 = worst (red)."""
    return pc.sample_colorscale('RdYlGn', [1.0 - i / max(n - 1, 1) for i in range(n)])


def make_preview_figure(fname, ts_x, ts_y, label, score_x, score_y, detector):
    """Caller is expected to pass already-downsampled (x, y) for ts and score
    (use load_timeseries_display / load_score_display)."""
    T = len(label)
    fig = _plotly_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[2, 0.6, 0.9], vertical_spacing=0.04,
    )

    fig.add_trace(go.Scattergl(x=ts_x, y=ts_y, mode='lines',
                               line=dict(color='dimgray', width=0.8),
                               name='Time series'), row=1, col=1)
    for s, e in _contiguous_regions(label.astype(bool)):
        fig.add_vrect(x0=s, x1=e, fillcolor='tomato', opacity=0.22,
                      line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='tomato', size=10, symbol='square', opacity=0.4),
                             name='Anomaly region'), row=1, col=1)

    # Label band: render only the contiguous anomaly regions as filled rects
    # (was a T-long Scatter with fill='tozeroy' — wasteful on long series).
    for s, e in _contiguous_regions(label.astype(bool)):
        fig.add_shape(type='rect', x0=s, x1=e, y0=0.0, y1=1.0,
                      fillcolor='rgba(255,99,71,0.5)', line=dict(width=0),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, T - 1], y=[0, 0], mode='lines',
                             line=dict(width=0), showlegend=False,
                             hoverinfo='skip'), row=2, col=1)

    fig.add_trace(go.Scattergl(x=score_x, y=score_y, mode='lines',
                               line=dict(color='steelblue', width=0.8),
                               name=f'Score  [{detector}]'), row=3, col=1)

    fig.update_layout(
        title=f'{os.path.basename(fname)}  ·  length={T:,}  ·  anomaly ratio={label.mean():.3f}',
        height=550,
        hovermode='x unified',
        legend=dict(x=1.0, y=1.0, xanchor='left'),
    )
    fig.update_yaxes(title_text='Value', row=1, col=1)
    fig.update_yaxes(title_text='Label', row=2, col=1, range=[-0.1, 1.3],
                     tickvals=[0, 1])
    fig.update_yaxes(title_text='Score', row=3, col=1)
    fig.update_xaxes(title_text='Time step', row=3, col=1)
    return fig


def make_vus_figure(curve, slope_size, fname, detector, fixed_ylim=True):
    x = np.arange(slope_size + 1)
    grad = np.gradient(curve)
    auc_pr_val = curve[0]
    vus_max_val = curve[slope_size]
    auc_above = float(np.trapezoid(curve - auc_pr_val) / (len(curve) - 1))
    peak_slope = int(np.argmax(np.abs(grad)))

    fig = _plotly_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[2, 1], vertical_spacing=0.06,
    )

    # --- VUS curve (row 1) ---
    fig.add_trace(go.Scatter(x=x, y=curve, mode='lines',
                             line=dict(color='steelblue', width=2),
                             name='VUS-PR'), row=1, col=1)
    # Baseline
    fig.add_hline(y=auc_pr_val, line=dict(color='gray', dash='dash', width=1),
                  opacity=0.6, row=1, col=1)
    # Fill above baseline (steelblue)
    c_above = np.where(curve >= auc_pr_val, curve, auc_pr_val)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([c_above, np.full(len(x), auc_pr_val)[::-1]]),
        fill='toself', fillcolor='rgba(70,130,180,0.18)', line_width=0,
        name=f'AUC above baseline = {auc_above:.5f}',
    ), row=1, col=1)
    # Fill below baseline (tomato)
    c_below = np.where(curve < auc_pr_val, curve, auc_pr_val)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([c_below, np.full(len(x), auc_pr_val)[::-1]]),
        fill='toself', fillcolor='rgba(255,99,71,0.18)', line_width=0,
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0], y=[auc_pr_val], mode='markers',
        marker=dict(color='tomato', size=9),
        name=f'slope=0 → AUC-PR = {auc_pr_val:.4f}',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[slope_size], y=[vus_max_val], mode='markers',
        marker=dict(color='seagreen', size=9),
        name=f'slope={slope_size} → VUS-PR = {vus_max_val:.4f}',
    ), row=1, col=1)
    fig.add_vline(x=peak_slope, line=dict(color='darkorange', dash='dot', width=1.2),
                  opacity=0.7, row=1, col=1)
    # Legend proxy for the vline
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='darkorange', dash='dot', width=1.5),
                             name=f'peak Δ at slope={peak_slope}'), row=1, col=1)

    # --- Gradient (row 2) ---
    fig.add_trace(go.Scatter(x=x, y=grad, mode='lines',
                             line=dict(color='darkorange', width=1.5),
                             name='dVUS/dslope'), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dash', width=0.8),
                  opacity=0.6, row=2, col=1)
    g_pos = np.where(grad >= 0, grad, 0.0)
    g_neg = np.where(grad < 0,  grad, 0.0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([g_pos, np.zeros(len(x))[::-1]]),
        fill='toself', fillcolor='rgba(46,139,87,0.25)', line_width=0, showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([g_neg, np.zeros(len(x))[::-1]]),
        fill='toself', fillcolor='rgba(255,99,71,0.25)', line_width=0, showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[peak_slope], y=[grad[peak_slope]], mode='markers',
        marker=dict(color='darkorange', size=8),
        name=f'peak = {grad[peak_slope]:.5f} at slope={peak_slope}',
    ), row=2, col=1)

    fig.update_layout(
        title=f'VUS-PR progression  ·  {os.path.basename(fname)}  ·  [{detector}]',
        height=700,
        hovermode='x unified',
        xaxis_range=[0, slope_size],
        legend=dict(x=1.0, y=1.0, xanchor='left', font=dict(size=10)),
    )
    fig.update_yaxes(title_text='VUS-PR', row=1, col=1)
    fig.update_yaxes(title_text='dVUS / dslope', row=2, col=1)
    fig.update_xaxes(title_text='Slope size (buffer length)', row=2, col=1)
    if fixed_ylim:
        fig.update_yaxes(range=[0, 1], row=1, col=1)

    tick_step = max(10, round(slope_size / 12 / 10) * 10)
    fig.update_xaxes(tickvals=list(range(0, slope_size + 1, tick_step)))

    return fig, grad


def make_impact_figure(df: pd.DataFrame):
    """2×2 figure: both metrics (Δ and AUC-above) × (per detector and per TS/dataset)."""
    multi_ds = df['dataset'].nunique() > 1
    right_label = 'per Dataset' if multi_ds else 'top 20 Time Series'
    metrics = [
        ('delta',     'Δ  (VUS − AUC-PR)'),
        ('auc_above', 'AUC above baseline'),
    ]
    titles = []
    for _, metric_name in metrics:
        titles.append(f'{metric_name} — per Detector')
        titles.append(f'{metric_name} — {right_label}')
    fig = _plotly_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    for r, (metric_col, _) in enumerate(metrics, start=1):
        # Left: per detector
        by_det = df.groupby('detector')[metric_col].mean().sort_values(ascending=False)
        fig.add_trace(go.Bar(
            x=by_det.index.tolist(), y=by_det.values.tolist(),
            marker_color='steelblue', showlegend=False,
        ), row=r, col=1)
        fig.update_xaxes(tickangle=45, row=r, col=1)
        fig.update_yaxes(title_text='Mean', row=r, col=1)

        # Right: per dataset (multiple) or top-20 time series (single)
        if multi_ds:
            by_right = df.groupby('dataset')[metric_col].mean().sort_values(ascending=False)
            fig.add_trace(go.Bar(
                x=by_right.index.tolist(), y=by_right.values.tolist(),
                marker_color='seagreen', showlegend=False,
            ), row=r, col=2)
            fig.update_xaxes(tickangle=45, row=r, col=2)
        else:
            by_right = (df.groupby('file')[metric_col].mean()
                          .nlargest(20).sort_values(ascending=True))
            fig.add_trace(go.Bar(
                y=by_right.index.tolist(), x=by_right.values.tolist(),
                orientation='h', marker_color='seagreen', showlegend=False,
            ), row=r, col=2)

    fig.update_layout(title='Impact of Slope Size L on VUS-PR', height=750)
    return fig


def make_ranking_bar(mean_vus_b: np.ndarray, detectors: list[str], b: int, agg_label: str = 'mean'):
    """Horizontal bar chart of detector ranking at a single buffer value b."""
    n = len(detectors)
    order      = np.argsort(-mean_vus_b)[::-1]   # worst → best (plotly: first y at bottom, so best ends up on top)
    plot_names = [detectors[i] for i in order]
    plot_vals  = mean_vus_b[order]

    colors = _rdylgn(n)[::-1]
    x_max  = max(float(np.nanmax(plot_vals)) * 1.15, 0.1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_names, x=plot_vals,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.4f}' if not np.isnan(v) else '' for v in plot_vals],
        textposition='outside',
        showlegend=False,
    ))
    fig.update_layout(
        title=f'Detector Ranking  ·  buffer = {b}  ·  aggregation = {agg_label}',
        xaxis_title=f'VUS-PR [{agg_label}]  (buffer = {b})',
        height=max(400, n * 45),
        xaxis_range=[0, x_max],
    )
    return fig


def instability_stats(ranks_2d: np.ndarray) -> dict:
    """
    ranks_2d: (n_det, L+1) integer ranks (1 = best).
    Returns scalar instability statistics for a single scope.
    """
    n_det, n_b = ranks_2d.shape
    L = n_b - 1

    rank_range = ranks_2d.max(axis=1) - ranks_2d.min(axis=1)   # (n_det,)
    mean_disp = float(rank_range.mean())
    max_disp = int(rank_range.max())
    worst_det_idx = int(rank_range.argmax())

    top1_at_0 = int(np.argmin(ranks_2d[:, 0]))
    top1_frac = float(np.sum(np.argmin(ranks_2d, axis=0) != top1_at_0)) / n_b

    # Kendall-τ between ranking at b=0 and b=L — vectorized (no scipy dep)
    r0 = ranks_2d[:, 0].astype(float)
    rL = ranks_2d[:, L].astype(float)
    sign_prod = np.sign((r0[:, None] - r0[None, :]) *
                        (rL[:, None] - rL[None, :]))
    iu = np.triu_indices(n_det, k=1)
    upper = sign_prod[iu]
    concordant = int((upper > 0).sum())
    discordant = int((upper < 0).sum())
    n_pairs = n_det * (n_det - 1) // 2
    kendall_tau = (concordant - discordant) / n_pairs if n_pairs > 0 else 1.0

    # Total rank-swap events: pairs (i,j) that swap order in any adjacent b-step
    diff = ranks_2d[:, np.newaxis, :] - ranks_2d[np.newaxis, :, :]   # (n_det, n_det, L+1)
    crossings = (np.sign(diff[:, :, :-1]) * np.sign(diff[:, :, 1:])) < 0
    mask_upper = np.triu(np.ones((n_det, n_det), dtype=bool), k=1)
    n_crossings = int(crossings[mask_upper].sum())

    return {
        'mean_disp':     mean_disp,
        'max_disp':      max_disp,
        'worst_det_idx': worst_det_idx,
        'top1_frac':     top1_frac,
        'kendall_tau':   kendall_tau,
        'n_crossings':   n_crossings,
    }


def make_bump_chart(ranks: np.ndarray, detectors: list[str], b: int, L: int, agg_label: str = 'mean'):
    """Rank-vs-buffer line chart for all detectors with a vertical marker at b."""
    n_det  = len(detectors)
    x      = np.arange(L + 1)
    palette = (pc.qualitative.D3 + pc.qualitative.Plotly + pc.qualitative.Safe
               + pc.qualitative.Vivid) * 3

    fig = go.Figure()
    for j, det in enumerate(detectors):
        fig.add_trace(go.Scatter(
            x=x, y=ranks[j],
            mode='lines',
            line=dict(color=palette[j % len(palette)], width=1.5),
            name=det,
            opacity=0.85,
        ))
    fig.add_vline(x=b, line=dict(color='black', dash='dash', width=1.5),
                  opacity=0.75,
                  annotation=dict(text=f'b = {b}', yanchor='top'))

    # Map each rank position at b=0 to the detector that holds it
    ranks_at_zero = ranks[:, 0]
    ticktext_left = [''] * n_det
    for j, det in enumerate(detectors):
        r = int(ranks_at_zero[j])
        if 1 <= r <= n_det:
            ticktext_left[r - 1] = det

    fig.update_layout(
        title=f'Detector Ranking vs Buffer Size  ·  aggregation = {agg_label}',
        xaxis_title='Buffer size  (b)',
        yaxis_title=None,
        yaxis=dict(range=[n_det + 0.5, 0.5],
                   tickmode='array',
                   tickvals=list(range(1, n_det + 1)),
                   ticktext=ticktext_left),
        xaxis_range=[0, L],
        height=max(500, n_det * 45),
        legend=dict(x=1.01, y=0.5, xanchor='left', font=dict(size=10)),
        hovermode='x unified',
    )
    return fig


# ---------------------------------------------------------------------------
# Localization map helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple:
    h = h.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def _sort_detectors_by_vus(selected_dets: list[str], fname: str, slope_size: int, data) -> tuple[list[str], dict]:
    """
    Sort detectors best→worst by precomputed VUS-PR at the given slope_size.
    Returns (sorted_list, {det: vus_score}) — falls back to original order if data unavailable.
    """
    if data is None:
        return selected_dets, {}

    files     = data['files']
    detectors = data['detectors']
    curves    = data['curves']   # (n_files, n_det, L+1)
    L         = data['L']

    try:
        file_idx = files.index(fname)
    except ValueError:
        return selected_dets, {}

    b = min(slope_size, L)
    vus_scores = {}
    for det in selected_dets:
        try:
            det_idx = detectors.index(det)
            vus_scores[det] = float(curves[file_idx, det_idx, b])
        except (ValueError, IndexError):
            vus_scores[det] = float('nan')

    sorted_dets = sorted(selected_dets, key=lambda d: -(vus_scores.get(d) or 0))
    return sorted_dets, vus_scores


def _rasterize_localization_row(
    orig_anom_1d, buffer_z_1d, score_norm, avg_label,
    tp_color, fn_color, fp_color, tn_color,
    show_tp, show_fn, show_fp, show_tn,
    width: int,
):
    """
    Returns a (1, width, 4) uint8 RGBA image summarizing one detector's row.
    Buckets are picked by max-significance so single-pixel anomalies survive
    aggressive horizontal downsampling.
    """
    T = len(score_norm)
    s_f = score_norm.astype(np.float32, copy=False)
    combined = (avg_label * s_f).astype(np.float32, copy=False)

    # Per-source-pixel RGBA, then bucket.
    if show_tp:
        tp = np.array(tp_color, dtype=np.float32)
    else:
        tp = np.array([1, 1, 1, 1], dtype=np.float32)
    if show_fn:
        fn = np.array(fn_color, dtype=np.float32)
    else:
        fn = np.array([1, 1, 1, 1], dtype=np.float32)
    if show_fp:
        fp = np.array(fp_color, dtype=np.float32)
    else:
        fp = np.array([1, 1, 1, 1], dtype=np.float32)
    if show_tn:
        tn = np.array(tn_color, dtype=np.float32)
    else:
        tn = np.array([1, 1, 1, 1], dtype=np.float32)

    # Bucket reduction: choose bucket size so width <= cap and >= 1
    if T <= width:
        n_buckets = T
        bucket_size = 1
        chosen_idx = np.arange(T, dtype=np.int64)
    else:
        n_buckets = width
        bucket_size = T // n_buckets
        trim = n_buckets * bucket_size
        # interestingness: prioritize anomaly > buffer > score magnitude
        interest = (orig_anom_1d.astype(np.float32) * 2.0
                    + buffer_z_1d.astype(np.float32) * 1.0
                    + s_f * 0.001)
        view = interest[:trim].reshape(n_buckets, bucket_size)
        local_arg = view.argmax(axis=1)
        chosen_idx = local_arg + np.arange(n_buckets) * bucket_size

    # Vectorized colour pick at the chosen pixels
    sel_anom = orig_anom_1d[chosen_idx]
    sel_buf  = buffer_z_1d[chosen_idx]
    sel_norm = ~(sel_anom | sel_buf)
    sel_score = s_f[chosen_idx][:, None]            # (W, 1)
    sel_combo = combined[chosen_idx][:, None]       # (W, 1)

    out = np.empty((n_buckets, 4), dtype=np.float32)
    # normal: TN → FP by score
    np.copyto(out, tn + (fp - tn) * sel_score)
    # buffer: TN → TP by combined
    mask = sel_buf
    out[mask] = (tn + (tp - tn) * sel_combo)[mask]
    # anomaly: FN → TP by score
    mask = sel_anom
    out[mask] = (fn + (tp - fn) * sel_score)[mask]

    img = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img[None, :, :]   # (1, W, 4)


def _png_data_uri(img_uint8: np.ndarray) -> str:
    """Encode a (H, W, 4) uint8 RGBA array as a base64 PNG data URI."""
    pil = Image.fromarray(img_uint8, mode='RGBA')
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=False)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def make_localization_figure(fname, ts_x, ts_y, label, detector_maps,
                              tp_color, fn_color, fp_color, tn_color,
                              show_tp=True, show_fn=True, show_fp=True, show_tn=True,
                              vus_scores=None):
    """
    detector_maps: dict {detector_name: (avg_label, score_norm)}, already sorted best→worst.
    ts_x, ts_y: down-sampled time-series line for row 1 (use load_timeseries_display).
    *_color: RGBA tuples with floats in [0, 1].
    """
    n_dets = len(detector_maps)
    T = len(label)

    def _hex(c):
        return '#{:02x}{:02x}{:02x}'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255))

    row_heights = [2.5] + [0.7] * n_dets
    s_total = sum(row_heights)
    row_heights = [h / s_total for h in row_heights]

    fig = _plotly_subplots(
        rows=1 + n_dets, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
    )

    # --- Time series panel (downsampled, WebGL) ---
    y_min = float(np.min(ts_y))
    y_max = float(np.max(ts_y))
    fig.add_trace(
        go.Scattergl(
            x=ts_x, y=ts_y, mode='lines',
            line=dict(width=0.8, color='dimgray'),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1, col=1,
    )
    orig_anom_1d = (label > 0)
    diff = np.diff(np.concatenate([[0], orig_anom_1d.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        fig.add_shape(
            type='rect', x0=int(s), x1=int(e), y0=y_min, y1=y_max,
            fillcolor='tomato', opacity=0.18, line=dict(width=0),
            layer='below', row=1, col=1,
        )

    # --- Detector heatmap rows: each rendered as one rasterized PNG image. ---
    raster_W = min(T, _HEATMAP_MAX_COLS)
    for i, (det, (avg_label, score_norm)) in enumerate(detector_maps.items()):
        row = i + 2
        buffer_z_1d = (avg_label > 0) & (~orig_anom_1d)
        normal_1d   = ~(avg_label > 0)
        combined    = (avg_label * score_norm).astype(np.float32)
        s_f         = score_norm.astype(np.float32)

        img = _rasterize_localization_row(
            orig_anom_1d, buffer_z_1d, score_norm, avg_label,
            tp_color, fn_color, fp_color, tn_color,
            show_tp, show_fn, show_fp, show_tn,
            raster_W,
        )
        uri = _png_data_uri(img)

        fig.add_layout_image(dict(
            source=uri,
            xref=f'x{row}', yref=f'y{row}',
            x=0, y=1,                        # top-left in data coords
            sizex=T - 1, sizey=1,
            sizing='stretch', layer='below',
            opacity=1.0,
        ))
        # Invisible placeholder trace so the subplot has its own axes.
        fig.add_trace(go.Scatter(
            x=[0, T - 1], y=[0, 1], mode='lines',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ), row=row, col=1)

        # --- Per-detector confusion-matrix counts (threshold 0.5) ---
        det_pos = score_norm > 0.5
        cnt_tp = int((orig_anom_1d & det_pos).sum() + (buffer_z_1d & (combined > 0.5)).sum())
        cnt_fn = int((orig_anom_1d & ~det_pos).sum())
        cnt_fp = int((normal_1d & det_pos).sum())
        cnt_tn = int((buffer_z_1d & (combined <= 0.5)).sum() + (normal_1d & ~det_pos).sum())

        vus_val = (vus_scores or {}).get(det)
        head = f'<b>{det}</b>' + (f'  (VUS={vus_val:.4f})'
                                   if vus_val is not None and not np.isnan(vus_val) else '')

        def _chip(hex_color, name, n):
            return (f'<span style="color:{hex_color}">■</span> {name}:{n:,}')

        parts = []
        if show_tp: parts.append(_chip(_hex(tp_color), 'TP', cnt_tp))
        if show_fn: parts.append(_chip(_hex(fn_color), 'FN', cnt_fn))
        if show_fp: parts.append(_chip(_hex(fp_color), 'FP', cnt_fp))
        if show_tn: parts.append(_chip(_hex(tn_color), 'TN', cnt_tn))
        counts_line = '  '.join(parts)

        label_text = head + (f'<br>{counts_line}' if counts_line else '')
        fig.add_annotation(
            x=0.0, y=0.5,
            xref='paper', yref=f'y{row} domain',
            text=label_text, showarrow=False,
            xanchor='right', yanchor='middle', xshift=-10,
            font=dict(size=10, color='black'),
            align='right',
        )

        fig.update_yaxes(
            showticklabels=False, showgrid=False,
            zeroline=False, fixedrange=True,
            range=[0, 1], row=row, col=1,
        )

    for r in range(1, 1 + n_dets + 1):
        fig.update_xaxes(range=[0, T - 1], row=r, col=1)
    fig.update_xaxes(title_text='Time step', row=1 + n_dets, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)

    fig.update_layout(
        title=dict(
            text=(f'{os.path.basename(fname)}  ·  length={T:,}  ·  '
                  f'anomaly ratio={label.mean():.3f}'),
            font=dict(size=12),
            x=0.0, xanchor='left',
        ),
        height=int(260 + n_dets * 60),
        margin=dict(l=320, r=20, t=50, b=40),
        plot_bgcolor='white',
        showlegend=False,
        hovermode='closest',
        dragmode='zoom',
    )

    return fig


# ---------------------------------------------------------------------------
# Adaptive buffer helpers
# ---------------------------------------------------------------------------

def _expand_label(label: np.ndarray, b: int) -> np.ndarray:
    """Hard binary expansion of label by b steps in each direction."""
    if b <= 0:
        return label.astype(bool)
    return np.convolve(label.astype(float), np.ones(2 * b + 1), mode='same') > 0


def make_adaptive_buffer_figure(fname, ts_x, ts_y, label, det_names, expanded_labels, b_stars):
    """
    Shared-axis subplot figure: panel 0 = raw TS + GT label, panels 1..n = expanded label per detector.
    Red = original anomaly, blue = buffer expansion zone.
    """
    n_det = len(det_names)
    T = len(label)

    row_heights = [3.0] + [1.0] * n_det
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = _plotly_subplots(
        rows=1 + n_det, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.008,
        row_heights=row_heights,
    )

    # Panel 0: raw time series + GT label as red spans
    ts_y_arr = np.asarray(ts_y)
    y_min, y_max = float(ts_y_arr.min()), float(ts_y_arr.max())
    y_pad = (y_max - y_min) * 0.05
    fig.add_trace(
        go.Scattergl(x=ts_x, y=ts_y, mode='lines',
                     line=dict(color='dimgray', width=0.8),
                     showlegend=False, hoverinfo='skip'),
        row=1, col=1,
    )
    for s, e in _contiguous_regions(label.astype(bool)):
        fig.add_shape(type='rect', x0=s, x1=e,
                      y0=y_min - y_pad, y1=y_max + y_pad,
                      fillcolor='rgba(255,99,71,0.2)', line=dict(width=0),
                      layer='below', row=1, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)

    orig_bool = label.astype(bool)

    # Panels 1..n_det
    for k, (name, exp, b_val) in enumerate(zip(det_names, expanded_labels, b_stars)):
        row = k + 2
        buf_only = exp & ~orig_bool   # buffer zone only (not anomaly itself)

        # Blue: full expanded region (anomaly + buffer)
        for s, e in _contiguous_regions(exp):
            fig.add_shape(type='rect', x0=s, x1=e, y0=0, y1=1,
                          fillcolor='rgba(70,130,180,0.35)', line=dict(width=0),
                          layer='below', row=row, col=1)
        # Red on top: original anomaly
        for s, e in _contiguous_regions(orig_bool):
            fig.add_shape(type='rect', x0=s, x1=e, y0=0, y1=1,
                          fillcolor='rgba(255,99,71,0.6)', line=dict(width=0),
                          layer='below', row=row, col=1)

        # Invisible trace so the subplot has its own axes
        fig.add_trace(
            go.Scatter(x=[0, T - 1], y=[0, 0], mode='lines',
                       line=dict(width=0), showlegend=False, hoverinfo='skip'),
            row=row, col=1,
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                         fixedrange=True, range=[0, 1], row=row, col=1)
        fig.add_annotation(
            x=0.0, y=0.5,
            xref='paper', yref=f'y{row} domain',
            text=f'<b>{name}</b>  ·  b* = {b_val}',
            showarrow=False, xanchor='right', yanchor='middle', xshift=-8,
            font=dict(size=10),
        )

    for r in range(1, 2 + n_det):
        fig.update_xaxes(range=[0, T - 1], row=r, col=1)
    fig.update_xaxes(title_text='Time step', row=1 + n_det, col=1)

    fig.update_layout(
        title=dict(
            text=f'{os.path.basename(fname)}  ·  length={T:,}  ·  anomaly ratio={label.mean():.3f}',
            font=dict(size=12), x=0.0, xanchor='left',
        ),
        height=max(500, 280 + n_det * 55),
        margin=dict(l=220, r=20, t=50, b=40),
        plot_bgcolor='white',
        showlegend=False,
        hovermode='x unified',
    )
    return fig


def make_convergence_figure(det_names, b_stars, mean_anom_len):
    """
    Strip plot: x = detector name, y = mean_anom_len + b*.
    Dashed line at mean_anom_len (baseline for a perfect detector).
    """
    ys = [mean_anom_len + b for b in b_stars]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=det_names, y=ys,
        mode='markers',
        marker=dict(size=12, color='steelblue', symbol='circle'),
        name='mean_anom_len + b*',
        hovertemplate='%{x}<br>mean_anom_len + b* = %{y:.1f}<extra></extra>',
    ))
    fig.add_hline(
        y=mean_anom_len,
        line=dict(color='tomato', dash='dash', width=2),
        annotation=dict(text=f'mean anom len = {mean_anom_len:.1f}',
                         font=dict(color='tomato'), yanchor='bottom'),
    )
    fig.update_layout(
        title='Convergence — mean_anom_len + b*  per detector',
        xaxis_title='Detector',
        yaxis_title='mean_anom_len + b*',
        height=420,
        hovermode='x unified',
        xaxis_tickangle=45,
    )
    return fig


def make_adapt_ranking_figure(sub):
    """Grouped horizontal bars: VUS vs Adjusted VUS per detector, sorted by aVUS.

    The 'VUS squashes detectors near 1.0 while aVUS separates them' view, per series.
    """
    sub = sub.sort_values('avus', ascending=True)  # ascending -> best on top in barh
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sub['detector'], x=sub['vus_full'], orientation='h',
        name='VUS', marker_color='steelblue',
        hovertemplate='%{y}<br>VUS = %{x:.4f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=sub['detector'], x=sub['avus'], orientation='h',
        name='Adjusted VUS', marker_color='crimson',
        hovertemplate='%{y}<br>aVUS = %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title='VUS vs Adjusted VUS per detector  (this series)',
        barmode='group', xaxis_title='score', xaxis_range=[0, 1],
        height=max(420, 22 * len(sub)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Existence map helpers
# ---------------------------------------------------------------------------

def _existence_scores_at_b(curves_1d_per_anom, b):
    """
    curves_1d_per_anom: (n_anomalies, L+1) float32 — precomputed per-anomaly curves.
    Returns (n_anomalies,) array: mean existence score at each anomaly for buffer b.
    Each curves[k, b'] = fraction of thresholds that detected anomaly k at buffer b'.
    Averaging over b' = 0..b gives the score for the chosen buffer size.
    """
    if curves_1d_per_anom is None or len(curves_1d_per_anom) == 0:
        return np.array([])
    return curves_1d_per_anom[:, :b + 1].mean(axis=1)


def make_existence_map_figure(fname, ts_x, ts_y, T, label, anom_starts, anom_ends,
                               detector_existence, sorted_dets, vus_scores=None):
    """
    detector_existence: dict {det_name: ndarray(n_anomalies)} — existence scores in [0,1].
    anom_starts / anom_ends: integer arrays of anomaly boundary positions.
    ts_x, ts_y: down-sampled (x, y) for the row-1 line trace; T is the full length.
    Time-aligned: anomaly blocks are placed at their true time positions.
    Returns a Plotly figure with a shared x-axis so zoom/pan applies to all rows.
    """
    # Colormap: existence=0 blends into the gray background, existence=1 is
    # black. Avoids the green/red collision on very thin anomalies.
    BG_GRAY = 224
    def _rgb(score):
        s = max(0.0, min(1.0, float(score)))
        v = int(round(BG_GRAY * (1.0 - s)))
        return f'rgb({v},{v},{v})'

    n_dets = len(sorted_dets)
    n_anom = len(anom_starts)

    row_heights = [2.5] + [0.7] * n_dets
    s_total = sum(row_heights)
    row_heights = [h / s_total for h in row_heights]

    fig = _plotly_subplots(
        rows=1 + n_dets, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        row_heights=row_heights,
    )

    # --- Time series panel (down-sampled, WebGL, no hover) ---
    y_min = float(np.min(ts_y))
    y_max = float(np.max(ts_y))
    fig.add_trace(
        go.Scattergl(
            x=ts_x, y=ts_y,
            mode='lines',
            line=dict(width=0.8, color='dimgray'),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1, col=1,
    )
    for k, (s, e) in enumerate(zip(anom_starts, anom_ends)):
        fig.add_shape(
            type='rect', x0=int(s), x1=int(e), y0=y_min, y1=y_max,
            fillcolor='tomato', opacity=0.18, line=dict(width=0),
            layer='below', row=1, col=1,
        )
        mid = (int(s) + int(e)) / 2
        fig.add_annotation(
            x=mid, y=y_max, text=str(k + 1),
            showarrow=False, yanchor='bottom',
            font=dict(size=9, color='tomato'),
            row=1, col=1,
        )

    # --- Detector heatmap rows ---
    for i, det in enumerate(sorted_dets):
        row = i + 2
        scores = detector_existence.get(det)

        # gray background spanning the full series
        fig.add_shape(
            type='rect', x0=0, x1=T - 1, y0=0, y1=1,
            fillcolor='rgb(224,224,224)', line=dict(width=0),
            layer='below', row=row, col=1,
        )

        if scores is not None and len(scores) > 0:
            for k, (s, e) in enumerate(zip(anom_starts, anom_ends)):
                fig.add_shape(
                    type='rect',
                    x0=int(s), x1=int(e) + 1, y0=0, y1=1,
                    fillcolor=_rgb(scores[k]), line=dict(width=0),
                    layer='below', row=row, col=1,
                )
            # invisible markers for hover on each anomaly
            mids = (np.asarray(anom_starts) + np.asarray(anom_ends)) / 2.0
            customdata = np.stack(
                [np.arange(1, len(scores) + 1), np.asarray(scores, dtype=float)],
                axis=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=mids, y=[0.5] * len(mids),
                    mode='markers',
                    marker=dict(size=14, opacity=0,
                                color=np.asarray(scores, dtype=float),
                                coloraxis='coloraxis'),
                    customdata=customdata,
                    hovertemplate=(
                        f'<b>{det}</b><br>'
                        'anom #%{customdata[0]:d}<br>'
                        'score=%{customdata[1]:.3f}<extra></extra>'
                    ),
                    showlegend=False,
                ),
                row=row, col=1,
            )

        mean_score = (float(np.nanmean(scores))
                      if scores is not None and len(scores) > 0
                      else float('nan'))
        vus_val = (vus_scores or {}).get(det)
        label_text = det
        if vus_val is not None and not np.isnan(vus_val):
            label_text += f'  (VUS={vus_val:.3f}'
        if not np.isnan(mean_score):
            sep = ', ' if vus_val is not None and not np.isnan(vus_val) else '  ('
            label_text += f'{sep}exist={mean_score:.3f})'
        elif label_text != det:
            label_text += ')'

        # Row label in the left margin (horizontal y-axis title), so it
        # never covers data when the user zooms in.
        fig.add_annotation(
            x=0.0, y=0.5,
            xref='paper',
            yref=f'y{row} domain',
            text=label_text,
            showarrow=False,
            xanchor='right', yanchor='middle',
            xshift=-6,
            font=dict(size=10, color='black'),
        )

        fig.update_yaxes(
            range=[0, 1], showticklabels=False, showgrid=False,
            zeroline=False, fixedrange=True,
            row=row, col=1,
        )

    # Lock x range and label only on the last row
    for r in range(1, 1 + n_dets + 1):
        fig.update_xaxes(range=[0, T - 1], row=r, col=1)
    fig.update_xaxes(title_text='Time step', row=1 + n_dets, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)

    fig.update_layout(
        title=dict(
            text=(f'{os.path.basename(fname)}  ·  length={T:,}  ·  '
                  f'{n_anom} anomaly region{"s" if n_anom != 1 else ""}'),
            font=dict(size=12),
            x=0.0, xanchor='left',
        ),
        height=int(220 + n_dets * 55),
        margin=dict(l=240, r=20, t=50, b=40),
        plot_bgcolor='white',
        showlegend=False,
        hovermode='closest',
        dragmode='zoom',
        coloraxis=dict(
            colorscale=[[0.0, f'rgb({BG_GRAY},{BG_GRAY},{BG_GRAY})'],
                        [1.0, 'rgb(0,0,0)']],
            cmin=0, cmax=1,
            colorbar=dict(
                title='Existence',
                thickness=10, len=0.6, x=1.01, xanchor='left',
            ),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# App pages
# ---------------------------------------------------------------------------

@st.fragment
def explorer_page(fname: str, slope_size: int):
    data = load_precomputed()
    L_max = data['L'] if data is not None else 1024
    slope_size = min(max(slope_size, 1), L_max)

    detectors = get_detector_names()
    detector = st.selectbox('Detector', detectors, key='explorer_detector')

    ts_x, ts_y, label = load_timeseries_display(fname)
    score_disp = load_score_display(fname, detector)

    if score_disp is None:
        st.error(f'No scores found for **{os.path.basename(fname)}** with detector **{detector}**.')
        return
    score_x, score_y = score_disp

    preview_fig = make_preview_figure(fname, ts_x, ts_y, label, score_x, score_y, detector)
    st.plotly_chart(preview_fig, width='stretch')

    st.divider()

    if data is None:
        st.error(
            'Precomputed data not found at `experiments/precomputed/rankings.npz`.\n\n'
            'Generate it with:\n```\npython scripts/precompute_rankings.py --L 1024\n```'
        )
        return

    files_pc     = data['files']
    detectors_pc = data['detectors']
    curves_pc    = data['curves']   # (n_files, n_det, L+1)

    if fname not in files_pc:
        st.warning(f'**{os.path.basename(fname)}** not found in precomputed data. Re-run the precomputation script.')
        return
    if detector not in detectors_pc:
        st.warning(f'Detector **{detector}** not found in precomputed data. Re-run the precomputation script.')
        return

    file_idx = files_pc.index(fname)
    det_idx  = detectors_pc.index(detector)
    full_curve = curves_pc[file_idx, det_idx, :]   # (L+1,)

    if np.all(np.isnan(full_curve)):
        st.error('No VUS data available for this (time series, detector) pair.')
        return

    curve = full_curve[:slope_size + 1]

    grad = np.gradient(curve)
    peak_slope = int(np.argmax(np.abs(grad)))
    peak_grad = grad[peak_slope]
    auc_above = auc_above_baseline(curve)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('AUC-PR  (slope = 0)', f'{curve[0]:.4f}')
    c2.metric(f'VUS-PR  (slope = {slope_size})', f'{curve[slope_size]:.4f}')
    c3.metric('Δ  (VUS − AUC-PR)', f'{curve[slope_size] - curve[0]:+.4f}')
    c4.metric('AUC above baseline', f'{auc_above:.5f}')
    c5.metric('Peak gradient', f'{peak_grad:.5f}', delta=f'at slope = {peak_slope}',
              delta_color='off')

    fixed_ylim = st.checkbox('Fix y-axis [0, 1]', value=True)
    vus_fig, _ = make_vus_figure(curve, slope_size, fname, detector,
                                 fixed_ylim=fixed_ylim)
    st.plotly_chart(vus_fig, width='stretch')


@st.fragment
def impact_page(dataset: str, fname: str, slope_size: int):
    st.header('Impact Analysis')
    st.caption(
        'Impact of slope size on VUS-PR across (time series × detector) pairs. '
        'Dataset and time series are selected from the left panel.'
    )

    data = load_precomputed()
    if data is None:
        st.error(
            'Precomputed data not found at `experiments/precomputed/rankings.npz`.\n\n'
            'Generate it with:\n```\npython scripts/precompute_rankings.py --L 1024\n```'
        )
        return

    files     = data['files']
    all_ds    = data['datasets']
    detectors = data['detectors']
    curves    = data['curves']   # (n_files, n_det, L+1)
    L         = data['L']

    scope = st.radio('Scope', ['Global', 'Dataset', 'Time series'],
                     horizontal=True, index=2, key='impact_scope')

    b = min(slope_size, L)

    if scope == 'Global':
        file_indices = list(range(len(files)))
    elif scope == 'Dataset':
        file_indices = [i for i, d in enumerate(all_ds) if d == dataset]
    else:
        file_indices = [i for i, f in enumerate(files) if f == fname]

    if not file_indices:
        st.warning(
            f'No matching files found for the selected scope. '
            f'Re-run: `python scripts/precompute_rankings.py --L {L}`'
        )
        return

    rows = []
    for file_idx in file_indices:
        fname_full = files[file_idx]
        ds_label   = all_ds[file_idx]
        for det_idx, detector in enumerate(detectors):
            full_curve = curves[file_idx, det_idx, :]   # (L+1,)
            if np.all(np.isnan(full_curve)):
                continue
            auc_pr_val = float(full_curve[0])
            vus_pr_val = float(full_curve[b])
            trunc      = full_curve[:b + 1]
            auc_ab     = float(np.trapezoid(trunc - auc_pr_val) / b) if b > 0 else 0.0
            rows.append({
                'file':      os.path.basename(fname_full),
                'dataset':   ds_label,
                'detector':  detector,
                'auc_pr':    auc_pr_val,
                'vus_pr':    vus_pr_val,
                'delta':     vus_pr_val - auc_pr_val,
                'auc_above': auc_ab,
            })

    if not rows:
        st.warning('No data found for the selected scope in the precomputed rankings.')
        return

    scope_label = scope
    if scope == 'Dataset':
        scope_label += f'  /  {dataset}'
    elif scope == 'Time series':
        scope_label += f'  /  {os.path.basename(fname)}'
    st.caption(
        f'Scope: **{scope_label}**  ·  buffer = {b}  ·  '
        f'{len(file_indices)} time series  ·  {len(detectors)} detectors'
    )

    df = pd.DataFrame(rows)
    impact_fig = make_impact_figure(df)
    st.plotly_chart(impact_fig, width='stretch')

    st.subheader('Summary by Detector')
    det_summary = (
        df.groupby('detector')
          .agg(mean_delta=('delta', 'mean'),
               mean_auc_above=('auc_above', 'mean'),
               n_pairs=('file', 'count'))
          .sort_values('mean_delta', ascending=False)
          .reset_index()
    )
    st.dataframe(
        det_summary.style.format({'mean_delta': '{:.4f}', 'mean_auc_above': '{:.5f}'}),
        width='stretch',
    )


def find_unstable_series(
    curves: np.ndarray,
    mask: np.ndarray,
    L_low: int,
    L_high: int,
    k: int,
) -> dict:
    """
    Find per-time-series instability in the buffer window [L_low, L_high].

    "Pool" = detectors whose best (min) global rank within the window is <= k.
    A crossing among (i, j) is counted only if BOTH i and j are in the pool, so
    e.g. a detector moving from rank 3 to rank 2 counts as a swap with whoever
    was rank 2 (a pool member).

    Returns dict with arrays aligned to np.where(mask)[0]:
        idx           : file indices into original curves array
        n_crossings   : crossings among pool pairs in [L_low, L_high]
        n_distinct_top1: distinct detectors holding rank #1 across the window
        pool_dets     : list of pool detector indices per file (sorted by min rank)
        ranks_window  : (N, n_det, B) global ranks restricted to the window
    """
    file_idx = np.where(mask)[0]
    if len(file_idx) == 0:
        return {
            'idx': file_idx, 'n_crossings': np.zeros(0, dtype=int),
            'n_distinct_top1': np.zeros(0, dtype=int),
            'pool_dets': [], 'ranks_window': np.zeros((0, 0, 0), dtype=int),
        }

    sub = curves[file_idx][:, :, L_low:L_high + 1]              # (N, n_det, B)
    sub_safe = np.where(np.isnan(sub), -np.inf, sub)
    # Global rank per (file, buffer): 1 = best. argsort twice = ranks.
    ranks = (-sub_safe).argsort(axis=1).argsort(axis=1) + 1     # (N, n_det, B)

    N, n_det, B = ranks.shape
    in_pool = ranks.min(axis=2) <= k                            # (N, n_det)

    if B < 2:
        n_cross = np.zeros(N, dtype=int)
    else:
        diff_sign = np.sign(ranks[:, :, None, :] - ranks[:, None, :, :])  # (N, nd, nd, B)
        cross = (diff_sign[:, :, :, :-1] * diff_sign[:, :, :, 1:]) < 0     # (N, nd, nd, B-1)
        pool_pair = in_pool[:, :, None] & in_pool[:, None, :]              # (N, nd, nd)
        upper = np.triu(np.ones((n_det, n_det), dtype=bool), k=1)
        n_cross = (cross & pool_pair[:, :, :, None] & upper[None, :, :, None]).sum(axis=(1, 2, 3))

    top1 = ranks.argmin(axis=1)                                  # (N, B)
    n_distinct_top1 = np.array([len(np.unique(top1[i])) for i in range(N)], dtype=int)

    pool_lists = []
    for i in range(N):
        dets = np.where(in_pool[i])[0]
        dets = sorted(dets, key=lambda j: int(ranks[i, j].min()))
        pool_lists.append(dets)

    return {
        'idx':             file_idx,
        'n_crossings':     n_cross.astype(int),
        'n_distinct_top1': n_distinct_top1,
        'pool_dets':       pool_lists,
        'ranks_window':    ranks,
    }


def make_unstable_bump_chart(
    ranks_window: np.ndarray,
    detectors: list[str],
    pool_idx: list[int],
    L_low: int,
    L_high: int,
    title: str,
) -> go.Figure:
    """Bump chart of pool detectors' global ranks across the [L_low, L_high] window."""
    palette = (pc.qualitative.D3 + pc.qualitative.Plotly + pc.qualitative.Safe
               + pc.qualitative.Vivid) * 3
    x = np.arange(L_low, L_high + 1)

    fig = go.Figure()
    max_rank = 1
    for c, j in enumerate(pool_idx):
        y = ranks_window[j]
        max_rank = max(max_rank, int(y.max()))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers',
            line=dict(color=palette[c % len(palette)], width=2.5),
            marker=dict(size=7),
            name=detectors[j],
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Buffer size  (b)',
        yaxis_title='Global rank  (1 = best)',
        yaxis=dict(range=[max_rank + 0.5, 0.5],
                   tickvals=list(range(1, max_rank + 1))),
        xaxis=dict(tickmode='linear', dtick=max(1, (L_high - L_low) // 15)),
        height=420,
        hovermode='x unified',
        legend=dict(x=1.01, y=0.5, xanchor='left'),
    )
    return fig


@st.cache_data
def _ranking_mean_curves_and_ranks(scope: str, dataset: str | None, fname: str | None):
    """
    Cached: mean-VUS-PR curves and ranks for the chosen scope.
    Returns (mean_curves, ranks) — both shape (n_det, L+1) — so that moving
    the buffer slider doesn't redo the np.nanmean / argsort double pass.
    """
    data = load_precomputed()
    curves = data['curves']
    files  = data['files']
    all_ds = data['datasets']

    if scope == 'Global':
        mask = np.ones(len(files), dtype=bool)
    elif scope == 'Dataset':
        mask = np.array(all_ds) == dataset
    else:
        mask = np.array(files) == fname

    mean_curves = np.nanmean(curves[mask], axis=0)
    ranks       = np.argsort(np.argsort(-mean_curves, axis=0), axis=0) + 1
    return mean_curves, ranks, int(mask.sum())


@st.cache_data
def _instability_summary_by_dataset() -> list[dict]:
    """Per-dataset instability summary — independent of sidebar selectors."""
    data = load_precomputed()
    if data is None:
        return []
    curves    = data['curves']
    detectors = data['detectors']
    all_ds    = data['datasets']
    ds_arr    = np.array(all_ds)
    unique_datasets = sorted(set(all_ds))

    summary_rows = []
    for ds in unique_datasets:
        ds_mask = ds_arr == ds
        n_files_ds = int(ds_mask.sum())
        if n_files_ds == 0:
            continue
        mc_ds = np.nanmean(curves[ds_mask], axis=0)
        if np.all(np.isnan(mc_ds)):
            continue
        r_ds = np.argsort(np.argsort(-mc_ds, axis=0), axis=0) + 1
        s_ds = instability_stats(r_ds)
        summary_rows.append({
            'Dataset':            ds,
            'Files':              n_files_ds,
            'Mean rank displ.':   round(s_ds['mean_disp'], 2),
            'Max rank displ.':    s_ds['max_disp'],
            'Most unstable det.': detectors[s_ds['worst_det_idx']],
            'Top-1 unstable %':   round(s_ds['top1_frac'] * 100, 1),
            'Kendall-τ (0→L)':    round(s_ds['kendall_tau'], 3),
            'Rank crossings':     s_ds['n_crossings'],
        })
    return summary_rows


@st.fragment
def ranking_instability_page(dataset: str, fname: str, slope_size: int):
    st.header('Ranking Instability')
    st.caption(
        'How do detector rankings (by VUS-PR) shift as the buffer size b increases from 0 to L? '
        'Run the precomputation script first: '
        '`python scripts/precompute_rankings.py --L 1024`'
    )

    data = load_precomputed()
    if data is None:
        st.error(
            'Precomputed data not found at `experiments/precomputed/rankings.npz`.\n\n'
            'Generate it with:\n```\npython scripts/precompute_rankings.py --L 1024\n```'
        )
        return

    files     = data['files']
    all_ds    = data['datasets']
    detectors = data['detectors']
    L         = data['L']

    unique_datasets = sorted(set(all_ds))
    ds_arr   = np.array(all_ds)
    file_arr = np.array(files)

    curves = data['curves']   # (n_files, n_det, L+1)

    scope = st.radio('Scope', ['Global', 'Dataset', 'Time series'],
                     index=2, key='ri_scope')

    # Dataset and time series are driven by the left panel
    selected_dataset = dataset if scope in ('Dataset', 'Time series') else None
    selected_file    = fname   if scope == 'Time series' else None

    b = min(slope_size, L)

    mean_curves, ranks, n_masked = _ranking_mean_curves_and_ranks(
        scope, selected_dataset, selected_file
    )

    scope_label = scope
    if selected_dataset:
        scope_label += f'  /  {selected_dataset}'
    if selected_file:
        scope_label += f'  /  {os.path.basename(selected_file)}'
    st.caption(f'Aggregating over **{n_masked}** time series  ·  {scope_label}')

    # --- Bar chart (interactive via slider) ---
    bar_fig = make_ranking_bar(mean_curves[:, b], detectors, b)
    st.plotly_chart(bar_fig, width='stretch')

    st.divider()

    # --- Bump chart (full trajectory, vline tracks slider) ---
    st.subheader('Rank Trajectory Across All Buffer Sizes')
    bump_fig = make_bump_chart(ranks, detectors, b, L)
    st.plotly_chart(bump_fig, width='stretch')

    st.divider()

    # --- Current-scope instability stats ---
    st.subheader('Instability Statistics  ·  Current Scope')
    s = instability_stats(ranks)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Mean rank displacement', f'{s["mean_disp"]:.2f}',
              help='Average range (max−min rank) per detector across all b values. Higher = more unstable.')
    m2.metric('Max rank displacement', f'{s["max_disp"]}',
              help=f'Worst offender: **{detectors[s["worst_det_idx"]]}**. Max rank range across all b.')
    m3.metric('Top-1 instability', f'{s["top1_frac"]*100:.1f}%',
              help='Fraction of b values where the #1 detector differs from the one at b=0.')
    m4.metric('Kendall-τ  (b=0 → b=L)', f'{s["kendall_tau"]:.3f}',
              help='Rank correlation between the ranking at b=0 and b=L. 1=identical, −1=fully reversed.')
    m5.metric('Total rank crossings', f'{s["n_crossings"]:,}',
              help='Number of (detector pair, b-step) events where two detectors swap relative order.')

    st.divider()

    # --- Per-dataset instability summary (always shown) ---
    st.subheader('Instability Summary by Dataset')
    st.caption(
        'Each row averages VUS-PR curves across all time series in the dataset, then derives '
        'detector ranks at every b. Sorted by mean rank displacement (most unstable first).'
    )

    summary_rows = _instability_summary_by_dataset()

    summary_df = (
        pd.DataFrame(summary_rows)
          .sort_values('Mean rank displ.', ascending=False)
          .reset_index(drop=True)
    )

    st.dataframe(
        summary_df.style
            .format({
                'Mean rank displ.': '{:.2f}',
                'Top-1 unstable %': '{:.1f}',
                'Kendall-τ (0→L)':  '{:.3f}',
            })
            .background_gradient(
                subset=['Mean rank displ.', 'Top-1 unstable %'],
                cmap='RdYlGn_r',
            )
            .background_gradient(
                subset=['Kendall-τ (0→L)'],
                cmap='RdYlGn',   # green = 1 (stable), red = low (unstable)
            ),
        width='stretch',
    )

    worst_row = summary_df.iloc[0]
    st.warning(
        f'**Most unstable dataset: {worst_row["Dataset"]}**  '
        f'(mean rank displ. = {worst_row["Mean rank displ."]:.2f}, '
        f'top-1 changes at {worst_row["Top-1 unstable %"]:.1f}% of b values, '
        f'Kendall-τ = {worst_row["Kendall-τ (0→L)"]:.3f})  '
        f'— select it in the **Dataset** scope above to inspect its bump chart.'
    )

    st.divider()

    # --- Find Unstable Time Series ----------------------------------------
    st.subheader('Find Unstable Time Series')
    st.caption(
        'Locate individual time series where the **top-k detectors** swap relative order '
        'in an early-buffer window [L_low, L_high]. A swap among (i, j) is counted only '
        'if both detectors enter the top-k pool somewhere in the window — so a detector '
        'climbing from rank 3 → rank 2 counts (it swaps with whoever was at rank 2).'
    )

    unstable_scope = st.radio(
        'Search scope',
        ['Global', 'Dataset'],
        horizontal=True,
        key='ri_unstable_scope',
        help='Per-time-series instability is not meaningful at the Time-series scope.',
    )
    if unstable_scope == 'Dataset':
        st.caption(f'Searching within dataset **{dataset}** (change in left panel).')
        search_mask = ds_arr == dataset
    else:
        search_mask = np.ones(len(files), dtype=bool)

    n_in_scope = int(search_mask.sum())
    if n_in_scope == 0:
        st.info('No time series in the selected scope.')
        return

    c_low, c_high, c_k, c_n = st.columns(4)
    with c_low:
        L_low = st.number_input('L_low', min_value=0, max_value=L, value=0,
                                step=1, key='ri_unstable_Llow')
    with c_high:
        default_high = min(L, 15)
        L_high = st.number_input('L_high', min_value=int(L_low) + 1, max_value=L,
                                 value=max(int(L_low) + 1, default_high),
                                 step=1, key='ri_unstable_Lhigh')
    with c_k:
        n_det_total = len(detectors)
        k_top = st.number_input('Top-k', min_value=1, max_value=n_det_total, value=min(2, n_det_total),
                                step=1, key='ri_unstable_k')
    with c_n:
        max_rows = st.number_input('Max results', min_value=5, max_value=200,
                                   value=20, step=5, key='ri_unstable_nrows')

    result = find_unstable_series(curves, search_mask,
                                  int(L_low), int(L_high), int(k_top))

    if len(result['idx']) == 0:
        st.info('No time series match the scope.')
        return

    rows = []
    for local_i, file_i in enumerate(result['idx']):
        n_cross = int(result['n_crossings'][local_i])
        if n_cross == 0:
            continue
        pool = result['pool_dets'][local_i]
        rows.append({
            '_local_i':       local_i,
            '_file_i':        int(file_i),
            'Dataset':        all_ds[file_i],
            'Time series':    os.path.basename(files[file_i]),
            'Crossings':      n_cross,
            'Distinct top-1': int(result['n_distinct_top1'][local_i]),
            'Pool size':      len(pool),
            'Top-k pool':     ', '.join(detectors[j] for j in pool),
        })

    if not rows:
        st.success(
            f'No unstable cases found in **{n_in_scope}** time series '
            f'within buffers [{int(L_low)}, {int(L_high)}] using top-{int(k_top)}. '
            'Try widening the window or increasing k.'
        )
        return

    unstable_df = (
        pd.DataFrame(rows)
          .sort_values(['Crossings', 'Distinct top-1'], ascending=False)
          .reset_index(drop=True)
          .head(int(max_rows))
    )

    st.caption(
        f'Found **{len(rows)}** unstable time series (showing top {len(unstable_df)}) '
        f'out of {n_in_scope} in scope  ·  window = [{int(L_low)}, {int(L_high)}]  '
        f'·  top-k = {int(k_top)}.'
    )

    st.dataframe(
        unstable_df.drop(columns=['_local_i', '_file_i']).style
            .background_gradient(subset=['Crossings', 'Distinct top-1'], cmap='RdYlGn_r'),
        width='stretch',
        hide_index=True,
    )

    # --- Inspection: pick a row and view its focused bump chart -----------
    st.markdown('**Inspect a case**')
    label_for_pick = [
        f'{r["Dataset"]} / {r["Time series"]}  ·  {r["Crossings"]} crossings, '
        f'{r["Distinct top-1"]} top-1'
        for _, r in unstable_df.iterrows()
    ]
    pick = st.selectbox(
        'Pick a time series to visualize',
        options=list(range(len(unstable_df))),
        format_func=lambda i: label_for_pick[i],
        key='ri_unstable_pick',
    )

    chosen = unstable_df.iloc[int(pick)]
    local_i = int(chosen['_local_i'])
    pool_idx = result['pool_dets'][local_i]
    ranks_window = result['ranks_window'][local_i]                # (n_det, B)

    title = (
        f'{chosen["Dataset"]} / {chosen["Time series"]}  ·  '
        f'buffers [{int(L_low)}, {int(L_high)}]  ·  '
        f'{int(chosen["Crossings"])} crossings among top-{int(k_top)} pool '
        f'({chosen["Pool size"]} detectors)'
    )
    bump_focus = make_unstable_bump_chart(
        ranks_window, detectors, pool_idx,
        int(L_low), int(L_high), title,
    )
    st.plotly_chart(bump_focus, width='stretch')
    st.caption(
        'Tip: copy the time-series name into the left panel and switch to the '
        '**Time series** scope to view its full-range bump chart and dive deeper '
        'in other tabs.'
    )


@st.fragment
def localization_map_page(fname: str, slope_size: int):
    st.header('Localization Map')
    st.caption(
        'Color each timestep by detection quality: **green** = TP, **orange** = FN, '
        '**red** = FP, **light** = TN. Stack detectors vertically to compare coverage. '
        'Time series is selected from the left panel.'
    )

    detectors = get_detector_names()
    selected_dets = st.multiselect('Detectors', detectors, default=detectors[:3],
                                   key='lm_dets')

    with st.expander('Color scheme'):
        cc1, cc2, cc3, cc4 = st.columns(4)
        tp_hex = cc1.color_picker('TP — detected anomaly', '#27ae60', key='lm_tp')
        tp_on  = cc1.checkbox('Show TP', value=True, key='lm_tp_on')
        fn_hex = cc2.color_picker('FN — missed anomaly',   '#f39c12', key='lm_fn')
        fn_on  = cc2.checkbox('Show FN', value=True, key='lm_fn_on')
        fp_hex = cc3.color_picker('FP — false alarm',      '#e74c3c', key='lm_fp')
        fp_on  = cc3.checkbox('Show FP', value=True, key='lm_fp_on')
        tn_hex = cc4.color_picker('TN — correct silence',  '#ecf0f1', key='lm_tn')
        tn_on  = cc4.checkbox('Show TN', value=True, key='lm_tn_on')

    if not selected_dets:
        st.info('Select at least one detector to display.')
        return

    ts_x, ts_y, label = load_timeseries_display(fname)

    # Sort detectors best→worst using precomputed VUS-PR rankings
    precomputed = load_precomputed()
    sorted_dets, vus_scores = _sort_detectors_by_vus(selected_dets, fname, slope_size, precomputed)

    avg_label = compute_avg_label_cached(fname, slope_size)
    detector_maps = {}
    missing = []
    for det in sorted_dets:
        score_norm = compute_score_norm_cached(fname, det)
        if score_norm is not None:
            detector_maps[det] = (avg_label, score_norm)
        else:
            missing.append(det)

    if missing:
        st.warning(f'No scores found for: **{", ".join(missing)}** — skipped.')

    if not detector_maps:
        st.error('No scores found for any selected detector on this time series.')
        return

    neutral = (1.0, 1.0, 1.0, 1.0)  # disabled categories fade to white
    tp_color = _hex_to_rgba(tp_hex) if tp_on else neutral
    fn_color = _hex_to_rgba(fn_hex) if fn_on else neutral
    fp_color = _hex_to_rgba(fp_hex) if fp_on else neutral
    tn_color = _hex_to_rgba(tn_hex) if tn_on else neutral

    fig = make_localization_figure(
        fname, ts_x, ts_y, label, detector_maps,
        tp_color, fn_color, fp_color, tn_color,
        show_tp=tp_on, show_fn=fn_on, show_fp=fp_on, show_tn=tn_on,
        vus_scores=vus_scores,
    )
    st.plotly_chart(fig, width='stretch')


@st.fragment
def existence_map_page(dataset: str, fname: str, slope_size: int):
    st.header('Existence Map')
    st.caption(
        'Per-anomaly existence score: for each detector and each labeled anomaly, '
        'how often is the anomaly detected across buffer sizes 0 → b and all score thresholds? '
        '**Green = always found · Red = never found · Gray = normal region.** '
        'Run the precomputation script first: '
        '`conda run -n ffvus python scripts/precompute_existence.py --L 1024`'
    )

    data = load_precomputed_existence()
    if data is None:
        st.error(
            'Precomputed existence data not found at `experiments/precomputed/existence_maps.npz`.\n\n'
            'Generate it with:\n```\n'
            'conda run -n ffvus python scripts/precompute_existence.py --L 1024\n```'
        )
        return

    files      = data['files']
    all_ds     = data['datasets']
    detectors  = data['detectors']
    L          = data['L']
    anom_curves = data['anom_curves']   # object (n_files, n_det)
    mean_curves = data['mean_curves']   # float32 (n_files, n_det, L+1)
    anom_starts_all = data['anom_starts']
    anom_ends_all   = data['anom_ends']

    b = min(slope_size, L)

    # -----------------------------------------------------------------------
    # Per-time-series view
    # -----------------------------------------------------------------------
    st.subheader('Per-anomaly view')

    if fname not in files:
        st.warning(f'**{os.path.basename(fname)}** not found in precomputed existence data. '
                   'Re-run the precomputation script.')
    else:
        file_idx = files.index(fname)
        anom_starts = anom_starts_all[file_idx]
        anom_ends   = anom_ends_all[file_idx]
        n_anom      = len(anom_starts)

        if n_anom == 0:
            st.info('This time series has no labeled anomalies.')
        else:
            all_det_names = get_detector_names()
            selected_dets = st.multiselect('Detectors', all_det_names,
                                           default=all_det_names[:3], key='em_dets')

            if selected_dets:
                ts_x, ts_y, label = load_timeseries_display(fname)
                T = len(label)

                # Sort by mean existence score (best → worst) at chosen b
                def _mean_exist(det):
                    if det not in detectors:
                        return float('nan')
                    di = detectors.index(det)
                    c  = anom_curves[file_idx, di]
                    if c is None or len(c) == 0:
                        return float('nan')
                    return float(np.nanmean(_existence_scores_at_b(c, b)))

                sorted_dets = sorted(selected_dets, key=lambda d: -(_mean_exist(d) or 0))

                # Get VUS scores for labels
                precomputed_vus = load_precomputed()
                _, vus_scores = _sort_detectors_by_vus(sorted_dets, fname, slope_size, precomputed_vus)

                # Build per-detector existence scores at chosen b
                detector_existence = {}
                missing = []
                for det in sorted_dets:
                    if det not in detectors:
                        missing.append(det)
                        continue
                    di = detectors.index(det)
                    c  = anom_curves[file_idx, di]
                    if c is None or len(c) == 0 or np.all(np.isnan(c)):
                        missing.append(det)
                        continue
                    detector_existence[det] = _existence_scores_at_b(c, b)

                if missing:
                    st.warning(f'No existence data for: **{", ".join(missing)}** — skipped.')

                if detector_existence:
                    em_fig = make_existence_map_figure(
                        fname, ts_x, ts_y, T, label,
                        anom_starts, anom_ends,
                        detector_existence, list(detector_existence.keys()),
                        vus_scores=vus_scores,
                    )
                    st.plotly_chart(em_fig, width='stretch')

                    # Anomaly-level table
                    st.caption(f'Existence scores at buffer b = {b}  '
                               f'(mean over thresholds and buffers 0 → {b})')
                    rows = {'Anomaly': [f'#{k+1}  [{anom_starts[k]}–{anom_ends[k]}]'
                                        for k in range(n_anom)]}
                    for det in detector_existence:
                        rows[det] = [
                            f'{v:.3f}' for v in detector_existence[det]
                        ]
                    st.dataframe(pd.DataFrame(rows).set_index('Anomaly'), width='stretch')

    st.divider()

    # -----------------------------------------------------------------------
    # Global statistics
    # -----------------------------------------------------------------------
    st.subheader('Existence Statistics')
    st.caption(
        'Mean existence score per detector / time series / dataset at the selected buffer. '
        'Scores are averaged over anomalies, buffers 0 → b, and thresholds.'
    )

    scope = st.radio('Scope', ['Global', 'Dataset', 'Time series'],
                     horizontal=True, key='em_scope')

    ds_arr   = np.array(all_ds)
    file_arr = np.array(files)

    if scope == 'Global':
        mask = np.ones(len(files), dtype=bool)
    elif scope == 'Dataset':
        mask = ds_arr == dataset
    else:
        mask = file_arr == fname

    n_masked = int(mask.sum())
    unique_datasets = sorted(set(all_ds))

    if n_masked == 0:
        st.warning('No files match the selected scope.')
    else:
        # scores_2d[i, j] = mean existence score for file i, detector j at buffer b
        # = mean over anomalies of curves[k, :b+1].mean()
        # = mean_curves[i, j, :b+1].mean(axis=-1)
        masked_mean_curves = mean_curves[mask]               # (n_masked, n_det, L+1)
        scores_2d = np.nanmean(masked_mean_curves[:, :, :b + 1], axis=-1)  # (n_masked, n_det)

        scope_label = scope
        if scope == 'Dataset':
            scope_label += f'  /  {dataset}'
        elif scope == 'Time series':
            scope_label += f'  /  {os.path.basename(fname)}'
        st.caption(f'Scope: **{scope_label}**  ·  {n_masked} time series  ·  '
                   f'{len(detectors)} detectors  ·  buffer b = {b}')

        # --- Per-detector bar chart ---
        per_det = np.nanmean(scores_2d, axis=0)   # (n_det,)
        order   = np.argsort(-per_det)             # best → worst
        n_d     = len(detectors)
        sorted_det_names = [detectors[i] for i in order]   # best→worst (plotly: first at top)
        sorted_det_vals  = per_det[order]
        colors_det = pc.sample_colorscale('RdYlGn', np.clip(sorted_det_vals, 0, 1).tolist())

        fig_det = go.Figure()
        fig_det.add_trace(go.Bar(
            y=sorted_det_names, x=sorted_det_vals,
            orientation='h',
            marker_color=colors_det,
            text=[f'{v:.3f}' if not np.isnan(v) else '' for v in sorted_det_vals],
            textposition='outside',
            showlegend=False,
        ))
        fig_det.update_layout(
            title=f'Detector existence  ·  scope = {scope_label}  ·  b = {b}',
            xaxis_title='Mean existence score',
            height=max(400, n_d * 45),
            xaxis_range=[0, max(float(np.nanmax(sorted_det_vals)) * 1.15, 0.1)],
        )
        st.plotly_chart(fig_det, width='stretch')

        # --- Per-file (or per-dataset) chart ---
        per_file_score = np.nanmean(scores_2d, axis=1)   # (n_masked,)
        masked_files   = file_arr[mask]
        masked_ds      = ds_arr[mask]

        if scope in ('Global', 'Dataset'):
            # Show worst 20 time series (worst = lowest existence score)
            order_f = np.argsort(per_file_score)   # worst first
            n_show  = min(20, len(order_f))
            # Include dataset name so each entry is uniquely identifiable
            worst_names = [f'{masked_ds[i]} / {os.path.basename(masked_files[i])}'
                           for i in order_f[:n_show]]
            worst_vals  = per_file_score[order_f[:n_show]]
            colors_f = pc.sample_colorscale('RdYlGn', np.clip(worst_vals, 0, 1).tolist())

            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(
                y=worst_names, x=worst_vals,
                orientation='h',
                marker_color=colors_f,
                text=[f'{v:.3f}' if not np.isnan(v) else '' for v in worst_vals],
                textposition='outside',
                showlegend=False,
            ))
            fig_f.update_layout(
                title=f'Hardest time series (lowest existence)  ·  b = {b}',
                xaxis_title='Mean existence score (avg over detectors)',
                height=max(400, n_show * 45),
                xaxis_range=[0, 1],
            )
            st.plotly_chart(fig_f, width='stretch')

        # --- Hardest individual anomalies ---
        anom_labels   = []
        anom_scores   = []
        anom_lengths  = []
        for fi in np.where(mask)[0]:
            starts = anom_starts_all[fi]
            ends   = anom_ends_all[fi]
            n_a    = len(starts)
            if n_a == 0:
                continue
            per_det_anom = np.full((n_a, len(detectors)), np.nan, dtype=np.float32)
            for di in range(len(detectors)):
                c = anom_curves[fi, di]
                if c is None or len(c) == 0:
                    continue
                per_det_anom[:, di] = c[:, :b + 1].mean(axis=1)
            mean_per_anom = np.nanmean(per_det_anom, axis=1)   # (n_a,)
            for k in range(n_a):
                s = int(starts[k]); e = int(ends[k])
                anom_labels.append(
                    f'{all_ds[fi]} / {os.path.basename(files[fi])}  ·  '
                    f'#{k+1} [{s}–{e}]'
                )
                anom_scores.append(float(mean_per_anom[k]))
                anom_lengths.append(e - s + 1)

        if anom_scores:
            anom_scores  = np.asarray(anom_scores)
            anom_lengths = np.asarray(anom_lengths)
            order_a = np.argsort(anom_scores)               # worst (lowest) first
            n_show_a = min(20, len(order_a))
            top_idx  = order_a[:n_show_a]
            top_names   = [anom_labels[i] for i in top_idx]
            top_vals    = anom_scores[top_idx]
            top_lengths = anom_lengths[top_idx]
            colors_a = pc.sample_colorscale('RdYlGn', np.clip(top_vals, 0, 1).tolist())

            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                y=top_names, x=top_vals,
                orientation='h',
                marker_color=colors_a,
                customdata=top_lengths.reshape(-1, 1),
                text=[f'{v:.3f}  (len={int(L_a)})'
                      for v, L_a in zip(top_vals, top_lengths)],
                textposition='outside',
                hovertemplate=(
                    '%{y}<br>mean existence=%{x:.3f}'
                    '<br>length=%{customdata[0]:d}<extra></extra>'
                ),
                showlegend=False,
            ))
            fig_a.update_layout(
                title=f'Hardest anomalies (lowest existence)  ·  scope = {scope_label}  ·  b = {b}',
                xaxis_title='Mean existence score (avg over detectors)',
                height=max(400, n_show_a * 45),
                xaxis_range=[0, 1],
                yaxis=dict(autorange='reversed'),
            )
            st.plotly_chart(fig_a, width='stretch')

        st.divider()

        # --- Per-dataset summary table ---
        st.subheader('Existence Summary by Dataset')
        st.caption(
            'Each row aggregates all time series in the dataset. '
            'Sorted by mean existence (hardest first).'
        )
        summary_rows = []
        for ds in unique_datasets:
            ds_mask = ds_arr == ds
            n_files_ds = int(ds_mask.sum())
            if n_files_ds == 0:
                continue
            mc_ds     = mean_curves[ds_mask]   # (n_ds_files, n_det, L+1)
            sc_ds     = np.nanmean(mc_ds[:, :, :b + 1], axis=-1)  # (n_ds_files, n_det)
            det_means = np.nanmean(sc_ds, axis=0)                  # (n_det,)
            file_means= np.nanmean(sc_ds, axis=1)                  # (n_ds_files,)
            if np.all(np.isnan(det_means)) or np.all(np.isnan(file_means)):
                continue
            best_det_idx  = int(np.nanargmax(det_means))
            worst_det_idx = int(np.nanargmin(det_means))
            summary_rows.append({
                'Dataset':           ds,
                'Files':             n_files_ds,
                'Mean existence':    round(float(np.nanmean(sc_ds)), 3),
                'Worst detector':    detectors[worst_det_idx],
                'Worst det. score':  round(float(det_means[worst_det_idx]), 3),
                'Best detector':     detectors[best_det_idx],
                'Best det. score':   round(float(det_means[best_det_idx]), 3),
                'Hardest TS':        os.path.basename(files[
                    np.where(ds_arr == ds)[0][int(np.nanargmin(file_means))]
                ]),
            })

        summary_df = (
            pd.DataFrame(summary_rows)
              .sort_values('Mean existence')
              .reset_index(drop=True)
        )
        st.dataframe(
            summary_df.style
                .format({'Mean existence': '{:.3f}',
                         'Worst det. score': '{:.3f}',
                         'Best det. score': '{:.3f}'})
                .background_gradient(subset=['Mean existence'], cmap='RdYlGn'),
            width='stretch',
        )

        worst_row = summary_df.iloc[0]
        st.warning(
            f'**Hardest dataset: {worst_row["Dataset"]}**  '
            f'(mean existence = {worst_row["Mean existence"]:.3f}, '
            f'worst detector = {worst_row["Worst detector"]} at {worst_row["Worst det. score"]:.3f})  '
            f'— select it in the **Dataset** scope above to drill in.'
        )


# ---------------------------------------------------------------------------
# Notes review page
# ---------------------------------------------------------------------------

@st.fragment
def notes_page() -> None:
    st.header('Notes')
    st.caption(
        'Observations you flagged via the sidebar "Add note" form. '
        'Use Open to jump the main selectors to the note\'s time series.'
    )

    notes = load_notes()
    if not notes:
        st.info('No notes yet. Use the **Add note** expander in the sidebar to record an observation.')
        return

    all_tags = sorted({t for n in notes for t in n.get('tags', [])})
    all_datasets = sorted({n['dataset'] for n in notes})

    c1, c2, c3 = st.columns(3)
    with c1:
        ds_filter = st.multiselect('Dataset', all_datasets, key='notes_filter_dataset')
    with c2:
        tag_filter = st.multiselect('Tags', all_tags, key='notes_filter_tags')
    with c3:
        text_filter = st.text_input('Search text', key='notes_filter_text')

    filtered = notes
    if ds_filter:
        filtered = [n for n in filtered if n['dataset'] in ds_filter]
    if tag_filter:
        filtered = [n for n in filtered if any(t in tag_filter for t in n.get('tags', []))]
    if text_filter:
        q = text_filter.lower()
        filtered = [n for n in filtered if q in n.get('text', '').lower()]

    filtered = sorted(filtered, key=lambda n: n.get('created', ''), reverse=True)
    st.caption(f'Showing {len(filtered)} of {len(notes)} notes')

    for n in filtered:
        with st.container(border=True):
            st.markdown(
                f"**{os.path.basename(n['fname'])}**  ·  "
                f"detector: `{n['detector']}`  ·  b = {n['buffer']}"
            )
            meta = f"{n['dataset']}  ·  created {n['created']}"
            if n.get('updated'):
                meta += f"  ·  edited {n['updated']}"
            st.caption(meta)
            if n.get('tags'):
                st.markdown('  '.join(f'`{t}`' for t in n['tags']))
            st.markdown(n.get('text', ''))

            c_open, c_del, _ = st.columns([1, 1, 6])
            with c_open:
                if st.button('Open', key=f'notes_open_{n["id"]}',
                             help='Jump the sidebar selectors to this time series'):
                    st.session_state['_jump_dataset'] = n['dataset']
                    st.session_state['_jump_fname'] = n['fname']
                    st.session_state['_jump_buffer'] = n['buffer']
                    st.session_state['_jump_detector'] = n['detector']
                    st.rerun()
            with c_del:
                if st.button('Delete', key=f'notes_del_{n["id"]}'):
                    delete_note(n['id'])
                    st.rerun()

            with st.expander('Edit'):
                new_text = st.text_area('Text', value=n.get('text', ''),
                                        key=f'notes_edit_text_{n["id"]}')
                new_tags = st.text_input('Tags (comma-separated)',
                                         value=', '.join(n.get('tags', [])),
                                         key=f'notes_edit_tags_{n["id"]}')
                if st.button('Save changes', key=f'notes_save_{n["id"]}'):
                    update_note(
                        n['id'],
                        text=new_text.strip(),
                        tags=[t.strip() for t in new_tags.split(',') if t.strip()],
                    )
                    st.rerun()


# ---------------------------------------------------------------------------
# Adaptive buffer page  (Adjusted VUS ranking)
# ---------------------------------------------------------------------------

# Final measures from score_adapt (CSV column -> display name). Skill (Adjusted
# VUS) columns are clipped at 0 for display; ratio columns are unbounded.
_ADAPT_MEASURES = {
    'vus_full':      'VUS',
    'vus_adapt':     'VUS_adapt (raw)',
    'skill_adapt_A': 'aVUS-A (adaptive)',
    'skill_adapt_B': 'aVUS-B (adaptive)',
    'skill_full_A':  'aVUS-A (full)',
    'skill_full_B':  'aVUS-B (full)',
    'ratio_adapt_A': 'ratio-A (adaptive)',
    'ratio_adapt_B': 'ratio-B (adaptive)',
    'ratio_full_A':  'ratio-A (full)',
    'ratio_full_B':  'ratio-B (full)',
}
_ADAPT_SKILL_COLS = {'skill_adapt_A', 'skill_adapt_B', 'skill_full_A', 'skill_full_B'}
_ADAPT_DEFAULT_MEASURES = ['vus_full', 'skill_adapt_A']


def _clip_measure(df, col):
    v = df[col].astype(float)
    return v.clip(lower=0.0) if col in _ADAPT_SKILL_COLS else v


def make_adapt_barplot(df, cols):
    """Per-detector bar chart: bar = mean of a measure across the series in scope,
    so detector RANKING is visible. Grouped by measure when several are selected,
    to compare how the ranking changes from one metric to another.

    Detectors are ordered by the mean of the first selected measure (best on top).
    """
    palette = pc.qualitative.Plotly
    agg = {col: pd.DataFrame({'detector': df['detector'], 'v': _clip_measure(df, col)})
                  .groupby('detector')['v'].mean()
           for col in cols}
    # ascending so the highest-scoring detector sits at the top of the horizontal axis
    order = agg[cols[0]].sort_values().index.tolist()

    fig = go.Figure()
    for i, col in enumerate(cols):
        a = agg[col].reindex(order)
        fig.add_trace(go.Bar(
            x=a.values, y=a.index, orientation='h',
            name=_ADAPT_MEASURES.get(col, col),
            marker_color=palette[i % len(palette)],
            hovertemplate='%{y}<br>' + _ADAPT_MEASURES.get(col, col) + ' = %{x:.4f}<extra></extra>',
        ))
    fig.update_layout(
        title='Detector ranking by mean score  (across scope)',
        xaxis_title='mean score', barmode='group',
        yaxis=dict(categoryorder='array', categoryarray=order, title='detector'),
        height=max(500, 22 * len(order)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


@st.cache_data
def compute_ranking_change_table():
    """Per-series ranking change between VUS and Adjusted VUS (aVUS-A).

    For each series: Kendall-tau between the per-detector VUS and aVUS values
    (low tau = ranking changes a lot), whether the best detector flips, and the
    spread gain (aVUS spread - VUS spread). Cached; used to surface interesting cases.
    """
    rk = load_ranking_adapt()
    if rk is None:
        return None
    rows = []
    for (ds, fn), g in rk.groupby(['dataset', 'filename']):
        v = g['vus_full'].to_numpy(dtype=float)
        a = g['avus'].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(a)
        if m.sum() < 2:
            continue
        tau = kendalltau(v[m], a[m]).correlation
        det = g['detector'].to_numpy()
        rows.append({
            'dataset': ds, 'filename': fn,
            'kendall_tau': float(tau) if tau is not None and np.isfinite(tau) else np.nan,
            'top1_flip': bool(det[np.nanargmax(np.where(m, v, -np.inf))]
                              != det[np.nanargmax(np.where(m, a, -np.inf))]),
            'best_vus':  det[np.nanargmax(np.where(m, v, -np.inf))],
            'best_avus': det[np.nanargmax(np.where(m, a, -np.inf))],
            'vus_spread':  float(np.nanmax(v[m]) - np.nanmin(v[m])),
            'avus_spread': float(np.nanmax(a[m]) - np.nanmin(a[m])),
        })
    d = pd.DataFrame(rows)
    if len(d):
        d['sep_gain'] = d['avus_spread'] - d['vus_spread']
    return d


@st.fragment
def adaptive_buffer_page(dataset: str, fname: str):
    st.header('Adaptive Buffer')
    st.caption(
        'Adjusted VUS (aVUS = chance-adjusted skill score) vs VUS, from '
        '`experiments/ranking_adapt/scores_all.csv` (L=256, 772 complete-case series). '
        'Where VUS squashes detectors near 1.0, aVUS separates them and the ranking changes.'
    )

    rk = load_ranking_adapt()
    if rk is None:
        st.info(
            'Ranking results not found. Generate with:\n```\n'
            'conda run -n ffvus python src/compute_ranking_adapt.py --dataset all --no-community\n```'
        )
        return

    # --- Scope (mirrors Ranking Instability) ---
    scope = st.radio('Scope', ['Global', 'Dataset', 'Time series'],
                     index=2, horizontal=True, key='ab_scope')
    if scope == 'Global':
        scoped = rk
    elif scope == 'Dataset':
        scoped = rk[rk['dataset'] == dataset]
    else:
        scoped = rk[rk['filename'] == fname]

    scope_label = scope
    if scope in ('Dataset', 'Time series'):
        scope_label += f'  /  {dataset}'
    if scope == 'Time series':
        scope_label += f'  /  {os.path.basename(fname)}'

    if scoped.empty:
        st.warning(
            f'No ranking data in scope ({scope_label}). The selected series may have '
            'been dropped (a detector lacked scores) or the run is partial.'
        )
        return

    st.caption(
        f'Aggregating **{scoped["filename"].nunique()}** series · '
        f'**{len(scoped)}** (series, detector) rows · {scope_label}'
    )

    # --- Measure selection + boxplot ---
    avail = [c for c in _ADAPT_MEASURES if c in scoped.columns and scoped[c].notna().any()]
    chosen = st.multiselect(
        'Measures', avail,
        default=[c for c in _ADAPT_DEFAULT_MEASURES if c in avail],
        format_func=lambda c: _ADAPT_MEASURES.get(c, c),
        key='ab_measures',
    )
    if chosen:
        st.plotly_chart(make_adapt_barplot(scoped, chosen), width='stretch')
    else:
        st.info('Select at least one measure to plot.')

    # Single series in scope: also show its per-detector ranking.
    if scope == 'Time series':
        sub = scoped.copy()
        sub['rank_vus'] = sub['vus_full'].rank(ascending=False, method='min').astype(int)
        sub['rank_avus'] = sub['avus'].rank(ascending=False, method='min').astype(int)
        best_v = sub.loc[sub['vus_full'].idxmax(), 'detector']
        best_a = sub.loc[sub['avus'].idxmax(), 'detector']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('VUS spread', f"{sub['vus_full'].max() - sub['vus_full'].min():.3f}")
        m2.metric('aVUS spread', f"{sub['avus'].max() - sub['avus'].min():.3f}")
        m3.metric('Best · VUS', best_v)
        m4.metric('Best · aVUS', best_a)
        if best_v != best_a:
            st.warning(f'Top-1 reranked — VUS picks **{best_v}**, aVUS picks **{best_a}**.')
        show = sub[['detector', 'vus_full', 'avus', 'skill_adapt_B', 'ratio_adapt_A',
                    'b_star_mean', 'rank_vus', 'rank_avus']].rename(columns={
            'vus_full': 'VUS', 'avus': 'aVUS (A)', 'skill_adapt_B': 'aVUS (B)',
            'ratio_adapt_A': 'ratio (A)', 'b_star_mean': 'mean b*',
            'rank_vus': 'rank VUS', 'rank_avus': 'rank aVUS',
        })
        st.dataframe(show.sort_values('rank aVUS'), width='stretch', hide_index=True)

    st.divider()

    # --- Find series where the ranking changes most ---
    st.subheader('Find series with large ranking change')
    st.caption(
        'Per series: Kendall-τ between the per-detector VUS and Adjusted VUS values '
        '(low τ = the ranking changes a lot), whether the best detector flips, and the '
        'spread gain (aVUS spread − VUS spread). Surfaces the interesting cases to check.'
    )

    change = compute_ranking_change_table()
    if change is None or change.empty:
        st.info('No ranking-change data available.')
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        search_scope = st.radio('Search scope', ['Global', 'Dataset'],
                                horizontal=True, key='ab_change_scope')
    with f2:
        sort_by = st.selectbox(
            'Rank cases by',
            ['Most reranked (low τ)', 'Largest spread gain', 'Top-1 flips first'],
            key='ab_change_sort',
        )
    with f3:
        max_rows = st.number_input('Max results', min_value=5, max_value=200,
                                   value=25, step=5, key='ab_change_nrows')

    csub = change if search_scope == 'Global' else change[change['dataset'] == dataset]
    if csub.empty:
        st.info('No series in the selected scope.')
        return

    if sort_by == 'Most reranked (low τ)':
        csub = csub.sort_values('kendall_tau', ascending=True)
    elif sort_by == 'Largest spread gain':
        csub = csub.sort_values('sep_gain', ascending=False)
    else:
        csub = csub.sort_values(['top1_flip', 'kendall_tau'], ascending=[False, True])

    disp = csub.head(int(max_rows)).copy()
    disp['Time series'] = disp['filename'].map(os.path.basename)
    table = disp[['dataset', 'Time series', 'kendall_tau', 'top1_flip',
                  'best_vus', 'best_avus', 'vus_spread', 'avus_spread', 'sep_gain']].rename(columns={
        'dataset': 'Dataset', 'kendall_tau': 'Kendall-τ', 'top1_flip': 'Top-1 flip',
        'best_vus': 'Best VUS', 'best_avus': 'Best aVUS',
        'vus_spread': 'VUS spread', 'avus_spread': 'aVUS spread', 'sep_gain': 'Spread gain',
    })
    st.caption(
        f'{len(csub)} series in scope · {int(disp["top1_flip"].sum())} of the shown '
        f'{len(disp)} flip their best detector.'
    )
    st.dataframe(
        table.style
            .format({'Kendall-τ': '{:.3f}', 'VUS spread': '{:.3f}',
                     'aVUS spread': '{:.3f}', 'Spread gain': '{:.3f}'})
            .background_gradient(subset=['Kendall-τ'], cmap='RdYlGn')
            .background_gradient(subset=['Spread gain'], cmap='RdYlGn_r'),
        width='stretch', hide_index=True,
    )
    st.caption(
        'Tip: set the **Dataset / Time series** selectors in the left panel to a row '
        'above, then switch Scope to **Time series** to inspect its per-detector ranking.'
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title='Progressive VUS Explorer', layout='wide')
    st.title('Progressive VUS Explorer')

    datasets = get_dataset_names()

    # If a Notes "Open" was clicked, pre-seed the sidebar selectors before they render.
    if '_jump_dataset' in st.session_state:
        jd = st.session_state.pop('_jump_dataset', None)
        jf = st.session_state.pop('_jump_fname', None)
        jb = st.session_state.pop('_jump_buffer', None)
        jdet = st.session_state.pop('_jump_detector', None)
        if jd in datasets:
            st.session_state['sel_dataset'] = jd
            if jf is not None:
                files_j = get_file_list(jd)
                if jf in files_j:
                    st.session_state['sel_ts_idx'] = files_j.index(jf)
        if jb is not None:
            st.session_state['sel_buffer'] = int(jb)
        if jdet is not None:
            st.session_state['explorer_detector'] = jdet
            st.session_state['note_form_detector'] = jdet

    # Shared dataset + time series selection — drives Explorer and Localization Map tabs.
    with st.sidebar:
        st.header('Selection')
        dataset = st.selectbox('Dataset', datasets, key='sel_dataset')
        files = get_file_list(dataset)
        basenames = [os.path.basename(f) for f in files]
        # Clamp stale ts_idx after dataset switch. Coerce defensively in case an
        # older session stored a filename string under this key.
        _cur_idx = st.session_state.get('sel_ts_idx', 0)
        if isinstance(_cur_idx, str):
            _cur_idx = basenames.index(_cur_idx) if _cur_idx in basenames else 0
        elif not isinstance(_cur_idx, int):
            try:
                _cur_idx = int(_cur_idx)
            except (TypeError, ValueError):
                _cur_idx = 0
        if _cur_idx < 0 or _cur_idx >= len(files):
            _cur_idx = 0
        st.session_state['sel_ts_idx'] = _cur_idx
        ts_idx = st.selectbox('Time series', range(len(basenames)),
                              format_func=lambda i: basenames[i],
                              key='sel_ts_idx')
        fname = files[ts_idx]
        st.divider()
        slope_size = st.slider('Buffer size  (b)', min_value=0, max_value=1024,
                               value=512, step=1, key='sel_buffer')

        st.divider()
        with st.expander('Add note', expanded=False):
            detectors_all = get_detector_names()
            if 'note_form_detector' not in st.session_state:
                cur_det = st.session_state.get('explorer_detector')
                if cur_det in detectors_all:
                    st.session_state['note_form_detector'] = cur_det
            note_det = st.selectbox('Detector', detectors_all, key='note_form_detector')
            note_tags_raw = st.text_input('Tags (comma-separated)',
                                          key='note_form_tags',
                                          placeholder='e.g. wrong-best, label-issue')
            note_text = st.text_area('Note', key='note_form_text',
                                     placeholder='What did you notice?')
            st.caption(
                f'Will be saved against **{os.path.basename(fname)}**  ·  '
                f'detector `{note_det}`  ·  b = {slope_size}'
            )
            if st.button('Save note', type='primary', use_container_width=True,
                         key='note_form_save'):
                if not note_text.strip():
                    st.warning('Note text is empty.')
                else:
                    tags = [t.strip() for t in note_tags_raw.split(',') if t.strip()]
                    add_note(dataset, fname, note_det, slope_size, tags, note_text.strip())
                    st.session_state.pop('note_form_tags', None)
                    st.session_state.pop('note_form_text', None)
                    st.toast('Note saved.')
                    st.rerun()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        'Explorer', 'Impact Analysis', 'Ranking Instability',
        'Localization Map', 'Existence Map', 'Adaptive Buffer', 'Notes',
    ])

    with tab1:
        explorer_page(fname, slope_size)

    with tab2:
        impact_page(dataset, fname, slope_size)

    with tab3:
        ranking_instability_page(dataset, fname, slope_size)

    with tab4:
        localization_map_page(fname, slope_size)

    with tab5:
        existence_map_page(dataset, fname, slope_size)

    with tab6:
        adaptive_buffer_page(dataset, fname)

    with tab7:
        notes_page()


if __name__ == '__main__':
    main()
