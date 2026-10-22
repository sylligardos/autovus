"""
AutoVUS demo: the buffer selected for every anomaly and the curves behind it.

Everything is computed when a case is picked, nothing offline. The main view
follows the paper's "Adaptive buffer selection" figure: (i) label, score and
the selected buffer, (ii) the detector curve and the random baseline over b,
(iii) the rule curve whose first argmax is b*.

Run with:
    conda run -n autovus streamlit run src/demo/app.py
"""
import os
import sys
import time

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from autovus import FastVUS, AutoVUS
try:
    from autovus import FastVUSGPU
    ENGINES = ('FastVUS', 'FastVUSGPU')
except ImportError:
    ENGINES = ('FastVUS',)

RULE_LABELS = {
    'cumulative': 'cumulative   (C(b) - C(0) + 1) / (G(b) - G(0) + 1)',
    'paper':      'paper   (A(b) - A(0) + 1) / (g(b) - g(0) + 1)',
    'ratio':      'ratio   A(b) / g(b)',
}
SIDES_LABELS = {'per_side': 'per side (left, right)', 'symmetric': 'symmetric (one buffer)'}
SELECT_BUFFER = {'ramp': 'ramp (zita = 0.707, as scored)', 'flat': 'flat (zita = 1, selection only)'}
MAX_WINDOW_POINTS = 20_000


# ---------------------------------------------------------------- synthetic cases (paper Fig. "Controlled case study")
T13 = 160
GT_START13, GT_END13 = 70, 90
GT_CENTER13 = (GT_START13 + GT_END13) / 2


def block13(start, length):
    y = np.zeros(T13)
    a = max(0, int(round(start)))
    b = min(T13, int(round(start + length)))
    if b > a:
        y[a:b] = 1.0
    return y


def centered_block13(center, length):
    return block13(center - length / 2, length)


def union13(*arrays):
    out = np.zeros(T13)
    for a in arrays:
        out = np.maximum(out, a)
    return out


def synthetic_cases():
    """name -> score, exactly as the paper's case-study notebook builds them."""
    cases = {}
    for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
        cases[f'A{i}'] = centered_block13(GT_CENTER13 + shift, 20)
    for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
        cases[f'B{i}'] = centered_block13(GT_CENTER13 + shift, 10)
    for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
        cases[f'C{i}'] = centered_block13(GT_CENTER13 + shift, 30)
    inside13 = block13(75, 10)
    for i, extra_start in enumerate([86, 90, 96, 102], start=1):
        cases[f'D1-{i}'] = union13(inside13, block13(extra_start, 8))
    for i, (extra_start, extra_len) in enumerate([(96, 6), (96, 10), (96, 14), (96, 18)], start=1):
        cases[f'D2-{i}'] = union13(inside13, block13(extra_start, extra_len))
    cases['E1'] = union13(block13(50, 20), block13(90, 20))
    return cases


SYNTHETIC = synthetic_cases()
GT13 = np.zeros(T13, int)
GT13[GT_START13:GT_END13] = 1


# ---------------------------------------------------------------- TSB-AD
@st.cache_data
def tsb_datasets():
    from tsad_parser import TSADParser
    return TSADParser().get_datasets()


@st.cache_data
def tsb_files(dataset):
    from tsad_parser import TSADParser
    meta = TSADParser().meta
    return sorted(meta[meta['dataset'] == dataset]['filename'].tolist())


@st.cache_data
def tsb_detectors():
    from tsad_parser import TSADParser
    return TSADParser().get_detectors()


@st.cache_data
def tsb_load(fname, detector):
    """series, label, score for one (series, detector); None when no score."""
    from tsad_parser import TSADParser
    p = TSADParser()
    _, ts, labels = p.load_timeseries(filename=[fname])
    scores, _ = p.load_scores(detectors=[detector], filenames=[fname])
    if not scores or not len(scores[0]):
        return None
    series = np.asarray(ts[0], dtype=float).ravel()
    label = np.asarray(labels[0]).astype(int).ravel()
    score = np.asarray(scores[0][0], dtype=float).ravel()
    n = min(len(label), len(score), len(series))              # a few scores are one point short
    return series[:n], label[:n], score[:n]


# ---------------------------------------------------------------- selection
@st.cache_data
def run_selection(key, label, score, L, rule, sides, select_buffer, engine_name):
    """AutoVUS.select for one case; returns (anomalies, buffers, zita, seconds)."""
    t0 = time.perf_counter()
    cls = FastVUS if engine_name == 'FastVUS' else FastVUSGPU
    engine = cls(L=L, cumulative=True)
    auto = AutoVUS(engine, rule=rule, sides=sides, select_zita=1.0 if select_buffer == 'flat' else None)
    anomalies = auto.select(label, score)
    seconds = time.perf_counter() - t0
    return anomalies, auto.buffers, engine.zita, seconds


# ---------------------------------------------------------------- figure
def soft_label(n, start, end, b_left, b_right, zita):
    """Label with the selected buffers, in window coordinates (end inclusive)."""
    y = np.zeros(n)
    y[start:end + 1] = 1.0
    for d in range(1, b_left + 1):
        if start - d >= 0:
            y[start - d] = 1.0 - (1.0 - zita) * d / b_left
    for d in range(1, b_right + 1):
        if end + d < n:
            y[end + d] = 1.0 - (1.0 - zita) * d / b_right
    return y


def scaled(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def make_figure(anomaly, series, label, score, buffers, zita, rule, sides, L):
    s, e = anomaly['start'], anomaly['end']
    lo, hi = anomaly['window']
    if hi - lo > MAX_WINDOW_POINTS:                            # display only
        lo, hi = max(lo, s - MAX_WINDOW_POINTS // 2), min(hi, e + MAX_WINDOW_POINTS // 2)
    x = np.arange(lo, hi)
    zoom = max(4 * L, 2 * (e - s + 1))                         # initial x range, the window stays reachable by zooming out
    bl, br = anomaly['left']['buffer_val'], anomaly['right']['buffer_val']
    # Column 1: series (row 1) and score (row 2), x axes linked. Columns 2 and 3:
    # one row per side (left, right), or one block spanning both rows (symmetric).
    if sides == 'per_side':
        blocks = [(1, 'left', 'left side'), (2, 'right', 'right side')]
        specs = [[{}, {}, {}], [{}, {}, {}]]
        titles = ['(i) series', '(ii) curves, left side', '(iii) rule curve, left side',
                  '(i) score and label', '(ii) curves, right side', '(iii) rule curve, right side']
    else:
        blocks = [(1, 'left', 'both sides')]
        specs = [[{}, {'rowspan': 2}, {'rowspan': 2}], [{}, None, None]]
        titles = ['(i) series', '(ii) curves', '(iii) rule curve', '(i) score and label', '', '']
    fig = make_subplots(rows=2, cols=3, specs=specs, column_widths=[0.5, 0.25, 0.25],
                        horizontal_spacing=0.06, vertical_spacing=0.12, subplot_titles=titles)
    # (i) series, anomalous points in red; score, scaled, with the label and its selected buffers
    for r in (1, 2):
        fig.add_vrect(x0=s - bl - 0.5, x1=e + br + 0.5, fillcolor='pink', opacity=0.4, line_width=0, row=r, col=1)
    if series is not None:
        win = series[lo:hi]
        fig.add_trace(go.Scatter(x=x, y=win, name='series', line=dict(color='#444', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=np.where(label[lo:hi] > 0, win, np.nan), name='anomaly',
                                 line=dict(color='red', width=2)), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=label[lo:hi], name='label', line=dict(color='red', width=2)), row=1, col=1)
    y = soft_label(hi - lo, s - lo, e - lo, bl, br, zita)
    fig.add_trace(go.Scatter(x=x, y=y, name=f'label + selected buffer ({bl}, {br})', line=dict(color='orange', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=scaled(score[lo:hi]), name='score (scaled)', line=dict(color='#1f77b4', width=1.5)), row=2, col=1)
    fig.update_xaxes(range=[max(lo, s - zoom), min(hi, e + zoom)], row=1, col=1)
    fig.update_xaxes(matches='x', row=2, col=1)                # zoom together
    fig.update_xaxes(title_text='time', row=2, col=1)
    fig.update_yaxes(title_text='series', row=1, col=1)
    fig.update_yaxes(title_text='score / label', row=2, col=1)
    # (ii) and (iii) per side
    for r, side, name in blocks:
        sel = anomaly[side]
        b = sel['buffer_val']
        first = r == 1
        C, G = sel['vus_curve'], sel['random_curve']
        A, g = AutoVUS.decumulate(C), AutoVUS.decumulate(G)
        if rule == 'cumulative':                                # the curves exactly as the rule divides them
            det, rnd, n1, n2 = C - C[0] + 1, G - G[0] + 1, 'C(b) - C(0) + 1 detector', 'G(b) - G(0) + 1 random'
        elif rule == 'paper':
            det, rnd, n1, n2 = A - A[0] + 1, g - g[0] + 1, 'A(b) - A(0) + 1 detector', 'g(b) - g(0) + 1 random'
        else:
            det, rnd, n1, n2 = A, g, 'A(b) detector', 'g(b) random'
        fig.add_trace(go.Scatter(x=buffers, y=det, name=n1, line=dict(color='#1f77b4'), showlegend=first), row=r, col=2)
        fig.add_trace(go.Scatter(x=buffers, y=rnd, name=n2, line=dict(color='orange'), showlegend=first), row=r, col=2)
        fig.add_vline(x=b, line=dict(color='black', dash='dash', width=1), row=r, col=2)
        rc = sel['ratio_curve']
        fig.add_trace(go.Scatter(x=buffers, y=rc, name='rule curve', line=dict(color='#1f77b4'), showlegend=first), row=r, col=3)
        fig.add_trace(go.Scatter(x=[b], y=[rc[sel['buffer_idx']]], mode='markers', marker=dict(color='#1f77b4', size=10),
                                 name='b*', showlegend=False), row=r, col=3)
        fig.add_vline(x=b, line=dict(color='black', dash='dash', width=1), row=r, col=3)
        fig.update_xaxes(title_text='buffer b', row=r, col=2)
        fig.update_xaxes(title_text='buffer b', row=r, col=3)
        fig.update_yaxes(title_text='score', row=r, col=2)
        fig.update_yaxes(title_text=f'gain, b* = {b}', row=r, col=3)
    fig.update_layout(height=620, margin=dict(l=40, r=20, t=60, b=40),
                      legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
    return fig


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title='AutoVUS demo', layout='wide')
    st.title('AutoVUS: buffer selection')

    with st.sidebar:
        source = st.radio('Source', ['Synthetic', 'TSB-AD'])
        if source == 'Synthetic':
            case = st.radio('Case', list(SYNTHETIC), horizontal=True)
            key = f'syn/{case}'
            series, label, score = None, GT13, SYNTHETIC[case]
            L_default, L_max = 20, 80
        else:
            dataset = st.selectbox('Dataset', tsb_datasets())
            fname = st.selectbox('Series', tsb_files(dataset))
            detector = st.selectbox('Detector', tsb_detectors())
            key = f'tsb/{fname}/{detector}'
            loaded = tsb_load(fname, detector)
            if loaded is None:
                st.error('No score for this series and detector.')
                return
            series, label, score = loaded
            L_default, L_max = 128, 512
        st.divider()
        rule = st.radio('Rule', list(RULE_LABELS), format_func=RULE_LABELS.get)
        sides = st.radio('Sides', list(SIDES_LABELS), format_func=SIDES_LABELS.get)
        select_buffer = st.radio('Selection buffer', list(SELECT_BUFFER), format_func=SELECT_BUFFER.get)
        L = st.slider('L', 1, L_max, L_default)
        engine_name = st.radio('Engine', ENGINES, horizontal=True)
        time_box = st.empty()

    if label.sum() == 0:
        st.error('The label has no anomaly.')
        return
    anomalies, buffers, zita, seconds = run_selection(key, label, score, L, rule, sides, select_buffer, engine_name)
    time_box.metric('Selection time', f'{seconds * 1000:.1f} ms')

    names = [f"{k + 1}: [{a['start']}, {a['end']}]  b* = ({a['left']['buffer_val']}, {a['right']['buffer_val']})  {a['detection']}"
             for k, a in enumerate(anomalies)]
    k = 0 if len(anomalies) == 1 else st.selectbox(f'Anomaly ({len(anomalies)})', range(len(anomalies)), format_func=lambda i: names[i])
    st.caption(f'{key}   ·   {names[k]}   ·   window {anomalies[k]["window"]}')
    st.plotly_chart(make_figure(anomalies[k], series, label, score, buffers, zita, rule, sides, L), width='stretch')


if __name__ == '__main__':
    main()
