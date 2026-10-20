"""
Adaptive Buffer explorer (Streamlit) — standalone, single page.

Detector ranking under a chosen metric, scoped to the whole benchmark, one
dataset, or one time series. Reads the ranking-experiment CSV produced by
compute_ranking_adapt.py.

Run with:
    conda run -n autovus streamlit run src/demo/demo_adaptive.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots as _plotly_subplots

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_DEMO_DIR)
sys.path.insert(0, _SRC_DIR)
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_SCORES_CSV = os.path.join(_ROOT_DIR, 'experiments', 'ranking_adapt', 'scores_all.csv')

# Final measures (CSV column -> display name). Skill (Adjusted VUS) columns are
# clipped at 0 for display; ratio columns are unbounded.
MEASURES = {
    'vus_full':       'VUS',
    'mean_vus_adapt': 'Mean per-anomaly VUS',
    'skill_adapt_A':  'Adjusted VUS  (A, adaptive)',
    'skill_adapt_B':  'Adjusted VUS  (B, adaptive)',
    'skill_full_A':   'Adjusted VUS  (A, full)',
    'skill_full_B':   'Adjusted VUS  (B, full)',
    'vus_adapt':      'VUS_adapt (raw)',
    'ratio_adapt_A':  'ratio (A, adaptive)',
    'ratio_adapt_B':  'ratio (B, adaptive)',
    'ratio_full_A':   'ratio (A, full)',
    'ratio_full_B':   'ratio (B, full)',
}
SKILL_COLS = {'skill_adapt_A', 'skill_adapt_B', 'skill_full_A', 'skill_full_B'}


@st.cache_data
def load_scores():
    if not os.path.exists(_SCORES_CSV):
        return None
    df = pd.read_csv(_SCORES_CSV)
    if 'note' in df.columns:
        df = df[df['note'].isna()]
    return df.dropna(subset=['vus_full'])


def detector_ranking(df, col):
    """Mean of `col` per detector (skill columns clipped at 0), sorted ascending
    (so the best detector ends up at the top of a horizontal bar chart)."""
    v = df[col].astype(float)
    if col in SKILL_COLS:
        v = v.clip(lower=0.0)
    s = pd.DataFrame({'detector': df['detector'], 'v': v})
    return s.groupby('detector')['v'].mean().sort_values()


def make_diff_ranking_bar(s: pd.Series, label_a: str, label_b: str, title: str):
    """Horizontal bar of entries sorted by signed diff (ascending → most positive on top).
    steelblue = A > B, crimson = A < B.  Zero line drawn for reference."""
    s = s.sort_values()
    display = ['/'.join(i.replace('\\', '/').split('/')[-2:]) if '/' in i else i
               for i in s.index]
    colors = ['steelblue' if v >= 0 else 'crimson' for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=display, orientation='h',
        marker_color=colors, customdata=list(s.index),
        hovertemplate='%{customdata}<br>' + label_a + ' − ' + label_b
                      + ' = %{x:.4f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_dash='dot', line_color='black', line_width=1)
    fig.update_layout(
        title=title, xaxis_title=f'{label_a} − {label_b}',
        height=max(400, 22 * max(len(s), 1)),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def make_ranking_bar(ranking, metric_label):
    fig = go.Figure(go.Bar(
        x=ranking.values, y=ranking.index, orientation='h',
        marker_color='crimson',
        hovertemplate='%{y}<br>' + metric_label + ' = %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=metric_label,
        xaxis_title=f'mean {metric_label}', yaxis_title='detector',
        height=max(500, 22 * len(ranking)),
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


TAXONOMY = [('n_exact', 'exact', 'seagreen'), ('n_early', 'early', 'orange'),
            ('n_late', 'late', 'goldenrod'), ('n_missed', 'missed', 'crimson')]


def has_taxonomy(df):
    return all(c in df.columns for c in (c for c, _, _ in TAXONOMY))


def make_taxonomy_figure(scoped):
    """Stacked horizontal bar: per-detector detection-type mix (fraction of the
    detector's anomalies that are exact / early / late / missed) over the scope.
    Detectors sorted by their 'exact' fraction (cleanest detector on top)."""
    cols = [c for c, _, _ in TAXONOMY]
    g = scoped.groupby('detector')[cols].sum()
    totals = g.sum(axis=1).replace(0, np.nan)
    frac = g.div(totals, axis=0).fillna(0.0)
    frac = frac.loc[frac['n_exact'].sort_values().index]   # ascending -> best last (top)
    fig = go.Figure()
    for col, name, color in TAXONOMY:
        fig.add_trace(go.Bar(
            y=frac.index, x=frac[col], name=name, orientation='h', marker_color=color,
            hovertemplate='%{y}<br>' + name + ' = %{x:.1%}<extra></extra>',
        ))
    fig.update_layout(
        barmode='stack', title='Detection taxonomy (per-detector anomaly mix)',
        xaxis_title='fraction of anomalies', yaxis_title='detector',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        height=max(500, 22 * len(frac)), legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=10, r=10, t=70, b=40),
    )
    return fig


@st.cache_data
def load_timeseries(fname):
    """Raw time series + binary label for one series (via TSB-AD)."""
    from tsad_parser import TSADParser
    _, ts, labels = TSADParser().load_timeseries(filename=[fname])
    return np.asarray(ts[0]), np.asarray(labels[0]).astype(int).ravel()


def _anomaly_regions(label):
    d = np.diff(np.concatenate([[0], label, [0]]))
    return list(zip(np.where(d > 0)[0], np.where(d < 0)[0]))  # [start, end)


@st.cache_data
def load_detector_score(fname, detector):
    from tsad_parser import TSADParser
    scores, _ = TSADParser().load_scores(detectors=[detector], filenames=[fname])
    return np.asarray(scores[0][0]).ravel() if scores else None


@st.cache_data
def compute_per_anomaly_bstar(fname, detector):
    """Live per-anomaly b* for one (series, detector), at L=256 (matching the CSV).

    Per-side: each anomaly carries an independent left/right buffer and a detection
    type (early/late/exact/missed) from the taxonomy."""
    from vus.vus_numpy import VUSNumpy
    _, label = load_timeseries(fname)
    score = load_detector_score(fname, detector)
    if score is None:
        return None
    vus = VUSNumpy(slope_size=256, step=1, cumulative=True, global_mask=False)
    res = vus.compute_per_anomaly_buffer(np.asarray(label), score, per_side=True)
    return [{'start': int(r['start']), 'end': int(r['end']),
             'b_left':  int(r['best_left']['buffer_val']),
             'b_right': int(r['best_right']['buffer_val']),
             'type':    r['detection_type'],
             'vus': float(r['best']['vus_score'])} for r in res]


def make_buffer_score_figure(ts, label, score, bstars, fname, detector, max_points=6000):
    """Two stacked panels: the series (with per-anomaly buffer zones) and the
    detector's score. Each anomaly k uses asymmetric buffers: left zone
    [start-b*_left, start] (orange), right zone [end, end+b*_right] (green)."""
    T = len(label)
    ts1 = ts if ts.ndim == 1 else ts[:, 0]
    step = max(1, T // max_points)
    x = np.arange(0, T, step)
    fig = _plotly_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           subplot_titles=('value + selected buffers '
                                           '(red=anomaly, orange=left b*, green=right b*)',
                                           f'{detector} — score'))
    fig.add_trace(go.Scatter(x=x, y=ts1[::step], mode='lines',
                             line=dict(width=1, color='dimgray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=score[::step], mode='lines',
                             line=dict(width=1, color='steelblue')), row=2, col=1)
    many = len(bstars) > 150
    if not many:
        for r in bstars:
            s, e = r['start'], r['end']
            bl, br = r['b_left'], r['b_right']
            for row in (1, 2):
                if bl > 0:
                    fig.add_vrect(x0=max(0, s - bl), x1=s,
                                  fillcolor='orange', opacity=0.18, line_width=0, row=row, col=1)
                if br > 0:
                    fig.add_vrect(x0=e + 1, x1=min(T, e + 1 + br),
                                  fillcolor='seagreen', opacity=0.18, line_width=0, row=row, col=1)
                fig.add_vrect(x0=s, x1=e + 1, fillcolor='red', opacity=0.3, line_width=0, row=row, col=1)
    else:
        idx = np.where(label > 0)[0]
        if len(idx) > 4000:
            idx = idx[::len(idx) // 4000]
        fig.add_trace(go.Scatter(x=idx, y=ts1[idx], mode='markers',
                                 marker=dict(color='red', size=3)), row=1, col=1)
    fig.update_layout(height=540, showlegend=False, margin=dict(l=10, r=10, t=50, b=30))
    return fig


def make_timeseries_figure(ts, label, fname, max_points=6000):
    """Line plot of the series with anomaly regions shaded. Long series are
    strided for the line; shading uses the full-resolution anomaly indices."""
    T = label.shape[0]
    ts = ts if ts.ndim > 1 else ts.reshape(-1, 1)
    step = max(1, T // max_points)
    x = np.arange(0, T, step)
    fig = go.Figure()
    multi = ts.shape[1] > 1
    for c in range(ts.shape[1]):
        fig.add_trace(go.Scatter(
            x=x, y=ts[::step, c], mode='lines', line=dict(width=1),
            name=f'ch{c}' if multi else 'value', showlegend=multi,
        ))
    # Shade anomaly ranges when there are few; for point-anomaly-heavy series
    # (hundreds of regions) shapes lag the browser, so overlay red markers instead.
    regions = _anomaly_regions(label)
    if len(regions) <= 300:
        for s, e in regions:
            fig.add_vrect(x0=s, x1=max(e, s + 1), fillcolor='red', opacity=0.2, line_width=0)
    else:
        idx = np.where(label > 0)[0]
        if len(idx) > 4000:
            idx = idx[::len(idx) // 4000]
        fig.add_trace(go.Scatter(
            x=idx, y=ts[idx, 0], mode='markers',
            marker=dict(color='red', size=3), name='anomaly',
        ))
    fig.update_layout(
        title=os.path.basename(fname), xaxis_title='time', height=260,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig


def main():
    st.set_page_config(page_title='Adaptive Buffer Explorer', layout='wide')
    st.title('Adaptive Buffer Explorer')

    df = load_scores()
    if df is None:
        st.error(
            'Ranking results not found at '
            '`experiments/ranking_adapt/scores_all.csv`.\n\n'
            'Generate with:\n```\n'
            'conda run -n autovus python src/compute_ranking_adapt.py --dataset all --no-community\n```'
        )
        return

    with st.sidebar:
        st.header('Scope')
        scope = st.radio('Aggregate over', ['Whole benchmark', 'Dataset', 'Time series'])

        dataset = ts_file = None
        if scope in ('Dataset', 'Time series'):
            dataset = st.selectbox('Dataset', sorted(df['dataset'].unique()))
        if scope == 'Time series':
            files = sorted(df[df['dataset'] == dataset]['filename'].unique())
            ts_file = st.selectbox('Time series', files,
                                   format_func=os.path.basename)

        st.header('Metrics')
        avail = [c for c in MEASURES if c in df.columns and df[c].notna().any()]
        metrics = st.multiselect(
            'Metrics  (compared side by side)', avail,
            default=[c for c in ('vus_full', 'skill_adapt_A') if c in avail],
            format_func=lambda c: MEASURES[c],
        )

        st.header('Metric comparison')
        _cmp_defaults = [c for c in ('vus_full', 'skill_adapt_A') if c in avail]
        comp_a = st.selectbox(
            'Metric A', avail,
            index=avail.index(_cmp_defaults[0]) if _cmp_defaults else 0,
            format_func=lambda c: MEASURES[c], key='comp_a',
        )
        comp_b = st.selectbox(
            'Metric B', avail,
            index=avail.index(_cmp_defaults[1]) if len(_cmp_defaults) > 1
                  else min(1, len(avail) - 1),
            format_func=lambda c: MEASURES[c], key='comp_b',
        )
        top_n = st.slider('Top N', 5, 50, 20)

    # Apply scope
    if scope == 'Whole benchmark':
        scoped = df
        scope_label = 'whole benchmark'
    elif scope == 'Dataset':
        scoped = df[df['dataset'] == dataset]
        scope_label = f'dataset {dataset}'
    else:
        scoped = df[df['filename'] == ts_file]
        scope_label = f'{dataset} / {os.path.basename(ts_file)}'

    if scoped.empty:
        st.warning(f'No data in scope ({scope_label}).')
        return

    # Show the selected time series itself.
    if scope == 'Time series':
        ts, label = load_timeseries(ts_file)
        st.plotly_chart(make_timeseries_figure(ts, label, ts_file), width='stretch')

    st.caption(
        f'{scope_label}  ·  {scoped["filename"].nunique()} series, '
        f'{scoped["detector"].nunique()} detectors'
    )

    if not metrics:
        st.info('Select at least one metric.')
        return

    # One ranking chart per metric, side by side. In Time-series scope the bars
    # are clickable to pick a detector for the buffer/score view below.
    clickable = scope == 'Time series'
    if clickable:
        st.caption('Click a bar to inspect that detector\'s buffers and score below.')
    cols = st.columns(len(metrics))
    rankings = {}
    clicked = None
    for c, metric in zip(cols, metrics):
        ranking = detector_ranking(scoped, metric)
        rankings[metric] = ranking
        if clickable:
            ev = c.plotly_chart(make_ranking_bar(ranking, MEASURES[metric]),
                                width='stretch', on_select='rerun', key=f'rank_{metric}')
            for p in (ev.get('selection', {}) or {}).get('points', []) if ev else []:
                if p.get('y'):
                    clicked = p['y']
        else:
            c.plotly_chart(make_ranking_bar(ranking, MEASURES[metric]), width='stretch')

    # Detection taxonomy mix across the current scope (new per-side feature).
    if has_taxonomy(scoped) and scoped[[c for c, _, _ in TAXONOMY]].to_numpy().sum() > 0:
        st.divider()
        st.subheader('Detection taxonomy')
        st.caption('Each anomaly is classified by where the detector fired relative '
                   'to it: **exact** (inside), **early** (left buffer grew), **late** '
                   '(right buffer grew), **missed** (no skill).')
        st.plotly_chart(make_taxonomy_figure(scoped), width='stretch')

    # Metric difference ranking
    st.divider()
    st.subheader('Metric difference ranking')
    label_a, label_b = MEASURES[comp_a], MEASURES[comp_b]
    if comp_a == comp_b:
        st.info('Select two different metrics in the sidebar (Metric A ≠ Metric B).')
    else:
        tmp = scoped[['dataset', 'filename', 'detector']].copy()
        tmp['diff'] = scoped[comp_a].astype(float).values - scoped[comp_b].astype(float).values
        tmp = tmp.dropna(subset=['diff'])
        if tmp.empty:
            st.warning('No overlapping data for the selected metrics.')
        elif scope == 'Time series':
            by_det = tmp.groupby('detector')['diff'].mean()
            by_det_top = by_det.reindex(by_det.abs().sort_values(ascending=False).index).head(top_n)
            st.caption(f'Mean **{label_a}** − **{label_b}** per detector on this series '
                       f'(steelblue = A > B, crimson = A < B).')
            st.plotly_chart(
                make_diff_ranking_bar(by_det_top, label_a, label_b,
                                      f'Detector diff: {label_a} − {label_b}'),
                width='stretch',
            )
        else:
            by_ts = tmp.groupby('filename')['diff'].mean()
            by_ts_top = by_ts.reindex(by_ts.abs().sort_values(ascending=False).index).head(top_n)
            st.caption(f'Top {top_n} entries by |mean **{label_a}** − **{label_b}**|  '
                       f'(steelblue = A > B, crimson = A < B).')
            if scope == 'Whole benchmark':
                by_dataset = tmp.groupby('dataset')['diff'].mean()
                by_dataset_top = by_dataset.reindex(
                    by_dataset.abs().sort_values(ascending=False).index).head(top_n)
                c1, c2 = st.columns(2)
                c1.markdown('**Datasets**')
                c1.plotly_chart(
                    make_diff_ranking_bar(by_dataset_top, label_a, label_b,
                                         f'Dataset diff: {label_a} − {label_b}'),
                    width='stretch',
                )
                c2.markdown('**Time series**')
                c2.plotly_chart(
                    make_diff_ranking_bar(by_ts_top, label_a, label_b,
                                         f'Time series diff: {label_a} − {label_b}'),
                    width='stretch',
                )
            else:
                st.plotly_chart(
                    make_diff_ranking_bar(by_ts_top, label_a, label_b,
                                         f'Time series diff: {label_a} − {label_b}'),
                    width='stretch',
                )

    if clickable:
        _detector_detail(ts_file, rankings[metrics[0]], clicked)


def _detector_detail(ts_file, primary_ranking, clicked):
    """Time-series scope: per-anomaly buffers (b*) + score for one detector.
    Defaults to the top (best) detector of the first metric; click/selectbox to change."""
    st.divider()
    st.subheader('Detector buffers & score')

    order = list(primary_ranking.index)          # ascending -> best is last
    top_det = order[-1]
    if clicked in order:
        st.session_state['ad_detector'] = clicked
    if st.session_state.get('ad_detector') not in order:
        st.session_state['ad_detector'] = top_det

    detector = st.selectbox('Detector  (or click a bar above)', sorted(order),
                            key='ad_detector')

    ts, label = load_timeseries(ts_file)
    score = load_detector_score(ts_file, detector)
    if score is None:
        st.warning(f'No score available for **{detector}** on this series.')
        return
    bstars = compute_per_anomaly_bstar(ts_file, detector) or []

    if bstars:
        bl  = np.array([r['b_left']  for r in bstars])
        br  = np.array([r['b_right'] for r in bstars])
        vus = np.array([r['vus'] for r in bstars])
        types = [r['type'] for r in bstars]
        mix = ' · '.join(f'{t}: {types.count(t)}' for t in ('early', 'late', 'exact', 'missed')
                         if types.count(t))
        st.caption(
            f'**{detector}** · {len(bstars)} anomalies · '
            f'mean per-anomaly VUS = **{vus.mean():.4f}** · '
            f'b*_left mean {bl.mean():.1f} (max {bl.max()}) · '
            f'b*_right mean {br.mean():.1f} (max {br.max()}) · '
            f'detection mix — {mix}'
        )
    st.plotly_chart(
        make_buffer_score_figure(ts, label, score, bstars, ts_file, detector),
        width='stretch',
    )


if __name__ == '__main__':
    main()
