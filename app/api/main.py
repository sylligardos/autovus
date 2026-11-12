"""
The Python side of the AutoVUS demo: FastAPI on localhost.

The web app (app/web) reads the benchmark through it and computes live:
buffer selection on the Series page, the toy cases on the Intro, timings
and uploads on the Speed page. The tables the Rankings, Cases and Speed
pages read come from app/export.py and live under app/data/<benchmark>/.

Run from the repo root:
    conda run -n autovus uvicorn main:app --app-dir app/api --port 8765 --reload
"""
import io
import json
import os
import sys
import time
import uuid
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
DATA = os.path.join(_ROOT, 'app', 'data')

from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
from autovus import FastVUS, AdjustedVUS, AutoVUS  # noqa: E402
try:
    from autovus import FastVUSGPU  # noqa: E402
    import torch  # noqa: E402
    HAS_GPU = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAS_GPU = False
from baselines import compute as baseline  # noqa: E402

app = FastAPI(title='AutoVUS demo API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

BENCHMARKS = {'tsb-ad': {'label': 'TSB-AD'}}
_parsers = {}
UPLOADS = {}                     # id -> {'label', 'score', 'name', 'n'}
ORACLE_DELAY = 64                # the delayed oracle reference: label shifted later by this many points
REFERENCES = ('Delayed oracle', 'Random')
MAX_LIVE_T = 200_000             # upstream VUS and PATE run live only below this length


def parser(benchmark: str) -> TSADParser:
    if benchmark not in BENCHMARKS:
        raise HTTPException(404, f'unknown benchmark {benchmark}')
    if benchmark not in _parsers:
        _parsers[benchmark] = TSADParser()
    return _parsers[benchmark]


def read_table(benchmark, name):
    path = os.path.join(DATA, benchmark, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- data
def reference_score(name, label, seed=0):
    """The two reference detectors: the label shifted later, and a uniform random score."""
    label = np.asarray(label).astype(float)
    if name == 'Delayed oracle':
        out = np.zeros_like(label)
        out[ORACLE_DELAY:] = label[:len(label) - ORACLE_DELAY]
        return out
    if name == 'Random':
        return np.random.default_rng(seed).random(len(label))
    raise HTTPException(404, f'unknown reference {name}')


@lru_cache(maxsize=64)
def load_pair(benchmark, file, detector):
    """label, score (float64, same length) and the series values."""
    if file.startswith('upload:'):
        u = UPLOADS.get(file[7:])
        if u is None:
            raise HTTPException(404, 'upload expired')
        return u['label'], u['score'], u.get('values')
    p = parser(benchmark)
    try:
        values, label = load_series(p, file)
    except FileNotFoundError:
        raise HTTPException(404, f'unknown series {file}')
    if detector in REFERENCES:
        score = reference_score(detector, label)
    else:
        score = load_score(p, file, detector)
        if score is None:
            raise HTTPException(404, f'no score for {detector} on {file}')
    n = min(len(label), len(score), len(values))
    return label[:n], score[:n], values[:n]


def runs(label):
    d = np.diff(np.concatenate([[0], label, [0]]))
    return [[int(s), int(e) - 1] for s, e in zip(np.where(d > 0)[0], np.where(d < 0)[0])]


def binned(x, bins):
    """Min and max per bin, so a long series keeps its spikes when drawn."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= bins:
        return x.tolist(), x.tolist(), n
    edges = np.linspace(0, n, bins + 1).astype(int)
    lo = [float(np.min(x[a:b])) for a, b in zip(edges[:-1], edges[1:])]
    hi = [float(np.max(x[a:b])) for a, b in zip(edges[:-1], edges[1:])]
    return lo, hi, bins


def scaled(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# ---------------------------------------------------------------- endpoints: the benchmark
@app.get('/health')
def health():
    return {'ok': True, 'gpu': HAS_GPU}


@app.get('/summary')
def summary(b: str = Query('tsb-ad')):
    p = parser(b)
    meta = p.meta
    exported = read_table(b, 'summary.json') or {}
    return {
        'benchmark': b,
        'label': BENCHMARKS[b]['label'],
        'n_series': int(len(meta)),
        'n_datasets': int(meta['dataset'].nunique()),
        'n_detectors': int(len(p.get_detectors())),
        'n_anomalies': exported.get('n_anomalies'),
        'n_points': exported.get('n_points'),
        'datasets': sorted(meta['dataset'].unique().tolist()),
        'detectors': list(p.get_detectors()),
        'references': list(REFERENCES),
        'exported': bool(exported),
        'gpu': HAS_GPU,
    }


@app.get('/series')
def series_list(b: str = Query('tsb-ad'), dataset: str = Query(...)):
    meta = parser(b).meta
    rows = meta[meta['dataset'] == dataset]
    exported = read_table(b, 'series.json') or {}
    out = []
    for f in sorted(rows['filename']):
        e = exported.get(f, {})
        out.append({'file': f, 'dataset': dataset, 'n': e.get('n'), 'anomalies': e.get('anomalies')})
    return out


def window(n, start, end):
    """The point range [start, end) clipped to the series, the whole series by default."""
    start = max(0, min(int(start), n - 1))
    end = n if end is None else max(start + 1, min(int(end), n))
    return start, end


@app.get('/series/{file}')
def series_doc(file: str, b: str = Query('tsb-ad'), bins: int = Query(2000), start: int = Query(0), end: int | None = Query(None)):
    label, _, values = load_pair(b, file, 'Random')
    s, e = window(len(label), start, end)
    lo, hi, nb = binned(values[s:e], bins)
    return {'file': file, 'n': int(len(label)), 'start': s, 'end': e, 'bins': nb, 'lo': lo, 'hi': hi, 'labels': runs(label)}


@app.get('/score')
def score_doc(file: str, detector: str, b: str = Query('tsb-ad'), bins: int = Query(2000), start: int = Query(0), end: int | None = Query(None)):
    label, score, _ = load_pair(b, file, detector)
    s, e = window(len(label), start, end)
    lo, hi, nb = binned(scaled(score)[s:e], bins)
    return {'file': file, 'detector': detector, 'n': int(len(label)), 'start': s, 'end': e, 'bins': nb, 'lo': lo, 'hi': hi}


# ---------------------------------------------------------------- endpoints: selection
@lru_cache(maxsize=256)
def selection(benchmark, file, detector, L):
    """AutoVUS (paper rule, symmetric) on the CPU, plus plain VUS-PR and Adjusted VUS under 'full'."""
    label, score, _ = load_pair(benchmark, file, detector)
    if label.sum() == 0:
        raise HTTPException(400, 'the label has no anomaly')
    eng = FastVUS(L=L, cumulative=True)
    auto = AutoVUS(eng)
    t0 = time.perf_counter()
    result = auto.compute(label, score)
    seconds = time.perf_counter() - t0
    result['full'] = AdjustedVUS(eng).compute(label, score)          # plain VUS-PR and Adjusted VUS, no selection
    return result, auto.buffers.tolist(), seconds


def anomaly_summary(k, a):
    return {
        'k': k, 'start': a['start'], 'end': a['end'], 'window': list(a['window']),
        'b_left': int(a['left']['buffer_val']), 'b_right': int(a['right']['buffer_val']),
        'type': a['detection'],
        'core_ratio': float(a['left']['core_ratio']),
    }


@app.get('/select')
def select(file: str, detector: str, b: str = Query('tsb-ad'), L: int = Query(128)):
    result, buffers, seconds = selection(b, file, detector, L)
    return {
        'file': file, 'detector': detector, 'L': L,
        'seconds': seconds,
        'autovus': result['adjusted'], 'vus_adapt': result['vus'],
        'vus_full': result['full']['vus'], 'skill_full': result['full']['adjusted'],
        'random_full': result['full']['random'],
        'anomalies': [anomaly_summary(k, a) for k, a in enumerate(result['anomalies'])],
    }


@app.get('/curves')
def curves(file: str, detector: str, k: int, b: str = Query('tsb-ad'), L: int = Query(128)):
    """The three curves of one anomaly: the detector curve, the random curve, the rule curve."""
    result, buffers, _ = selection(b, file, detector, L)
    if k < 0 or k >= len(result['anomalies']):
        raise HTTPException(404, 'no such anomaly')
    a = result['anomalies'][k]
    out = {'k': k, 'buffers': buffers, 'sides': {}}
    for side in ('both',):
        s = a['left']
        out['sides'][side] = {
            'b': int(s['buffer_val']), 'idx': int(s['buffer_idx']), 'ratio': float(s['ratio']),
            'vus_curve': np.asarray(s['vus_curve'], dtype=float).tolist(),
            'random_curve': np.asarray(s['random_curve'], dtype=float).tolist(),
            'rule_curve': np.asarray(s['ratio_curve'], dtype=float).tolist(),
        }
    label, score, _ = load_pair(b, file, detector)
    lo, hi = a['window']
    span = max(4 * L, 2 * (a['end'] - a['start'] + 1))
    zl, zh = max(lo, a['start'] - span), min(hi, a['end'] + 1 + span)
    zlo, zhi, nb = binned(scaled(score)[zl:zh], 600)
    out['zoom'] = {'from': int(zl), 'to': int(zh), 'bins': nb, 'lo': zlo, 'hi': zhi}
    return out


@app.get('/rank')
def rank(file: str, b: str = Query('tsb-ad'), L: int = Query(128), metric: str = Query('autovus')):
    """Every detector on this series, best first under the metric (from the export when present, else computed live)."""
    table = read_table(b, 'scores.json')
    rows = []
    if table and file in table.get('files', {}) and L == table.get('L'):
        for det, vals in table['files'][file].items():
            rows.append({'detector': det, 'autovus': vals.get('autovus'), 'adj_vus': vals.get('skill_full'), 'vus_pr': vals.get('vus_pr'),
                         'vus_roc': vals.get('vus_roc'), 'auc_pr': vals.get('auc_pr'), 'auc_roc': vals.get('auc_roc')})
    else:
        dets = list(parser(b).get_detectors()) + list(REFERENCES) if not file.startswith('upload:') else []
        for det in dets:
            try:
                result, _, _ = selection(b, file, det, L)
                label, score, _ = load_pair(b, file, det)
                rows.append({'detector': det, 'autovus': result['adjusted'], 'adj_vus': result['full']['adjusted'], 'vus_pr': result['full']['vus'],
                             'vus_roc': float(FastVUS(L=L).compute(label, score, metric='roc')) if 0 < label.sum() < len(label) else None,
                             'auc_pr': float(average_precision_score(label, score)),
                             'auc_roc': float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else None})
            except HTTPException:
                continue
    key = metric if metric in ('autovus', 'adj_vus', 'vus_pr', 'vus_roc', 'auc_pr', 'auc_roc') else 'autovus'
    rows.sort(key=lambda r: -(r[key] if r.get(key) is not None else -1e9))
    return rows


# ---------------------------------------------------------------- endpoints: the toy cases
T13, GT_START, GT_END = 160, 70, 90


def _block(start, length):
    y = np.zeros(T13)
    a, b_ = max(0, int(round(start))), min(T13, int(round(start + length)))
    if b_ > a:
        y[a:b_] = 1.0
    return y


def _centered(center, length):
    return _block(center - length / 2, length)


def toy_cases():
    c = (GT_START + GT_END) / 2
    cases = {}
    for fam, length in (('A', 20), ('B', 10), ('C', 30)):
        for i, shift in enumerate([-20, -10, 0, 10, 20], start=1):
            cases[f'{fam}{i}'] = {'score': _centered(c + shift, length), 'shift': shift, 'length': length}
    inside = _block(75, 10)
    for i, extra in enumerate([86, 90, 96, 102], start=1):
        cases[f'D1-{i}'] = {'score': np.maximum(inside, _block(extra, 8)), 'shift': extra, 'length': 8}
    for i, (extra, ln) in enumerate([(96, 6), (96, 10), (96, 14), (96, 18)], start=1):
        cases[f'D2-{i}'] = {'score': np.maximum(inside, _block(extra, ln)), 'shift': extra, 'length': ln}
    cases['E1'] = {'score': np.maximum(_block(50, 20), _block(90, 20)), 'shift': 0, 'length': 20}
    return cases


TOY = toy_cases()
TOY_LABEL = np.zeros(T13, int)
TOY_LABEL[GT_START:GT_END] = 1


@app.get('/toy')
def toy(case: str = Query('A4'), L: int = Query(20)):
    if case not in TOY:
        raise HTTPException(404, 'unknown case')
    score = TOY[case]['score']
    eng = FastVUS(L=L, cumulative=True)
    auto = AutoVUS(eng)
    t0 = time.perf_counter()
    result = auto.compute(TOY_LABEL, score)
    seconds = time.perf_counter() - t0
    full = AdjustedVUS(eng).compute(TOY_LABEL, score)
    a = result['anomalies'][0]
    out = {
        'case': case, 'cases': list(TOY), 'label': TOY_LABEL.tolist(), 'score': score.tolist(),
        'seconds': seconds, 'autovus': result['adjusted'], 'vus_full': full['vus'],
        'anomaly': anomaly_summary(0, a), 'buffers': auto.buffers.tolist(), 'sides': {},
    }
    for side in ('both',):
        s = a['left']
        out['sides'][side] = {'b': int(s['buffer_val']), 'idx': int(s['buffer_idx']),
                              'vus_curve': np.asarray(s['vus_curve']).tolist(),
                              'random_curve': np.asarray(s['random_curve']).tolist(),
                              'rule_curve': np.asarray(s['ratio_curve']).tolist()}
    return out


# ---------------------------------------------------------------- endpoints: timing and upload
def _median_time(fn, repeats):
    times, value = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), value


@app.get('/time')
def time_metrics(file: str, detector: str, b: str = Query('tsb-ad'), L: int = Query(128), repeats: int = Query(3),
                 metrics: str = Query('autovus,vus_cpu,vus_gpu,vus_upstream,auc_pr,auc_roc,rf,affiliation,pate')):
    label, score, _ = load_pair(b, file, detector)
    T = int(len(label))
    want = [m for m in metrics.split(',') if m]
    out = {'file': file, 'detector': detector, 'n': T, 'L': L, 'repeats': repeats, 'results': []}
    for m in want:
        row = {'metric': m, 'seconds': None, 'value': None, 'note': None}
        try:
            if m == 'autovus':
                eng = FastVUS(L=L, cumulative=True)
                row['seconds'], r = _median_time(lambda: AutoVUS(eng).compute(label, score), repeats)
                row['value'] = r['adjusted']
            elif m == 'vus_cpu':
                eng = FastVUS(L=L)
                row['seconds'], v = _median_time(lambda: eng.compute(label, score), repeats)
                row['value'] = float(v)
            elif m == 'vus_roc':
                eng = FastVUS(L=L)
                row['seconds'], v = _median_time(lambda: eng.compute(label, score, metric='roc'), repeats)
                row['value'] = float(v)
            elif m == 'vus_gpu':
                if not HAS_GPU:
                    row['note'] = 'no gpu'
                else:
                    eng = FastVUSGPU(L=L)
                    eng.compute(label, score)                      # warm up once
                    row['seconds'], v = _median_time(lambda: eng.compute(label, score), repeats)
                    row['value'] = float(v.cpu() if hasattr(v, 'cpu') else v)
            elif m == 'vus_upstream':
                if T > MAX_LIVE_T:
                    row['note'] = f'skipped above {MAX_LIVE_T} points'
                else:
                    row['seconds'], (v, _) = _median_time(lambda: baseline('vus', label, score, L=L), 1)
                    row['value'] = float(v)
            elif m == 'auc_pr':
                row['seconds'], v = _median_time(lambda: average_precision_score(label, score), repeats)
                row['value'] = float(v)
            elif m == 'auc_roc':
                row['seconds'], v = _median_time(lambda: roc_auc_score(label, score), repeats)
                row['value'] = float(v)
            elif m in ('rf', 'affiliation'):
                row['seconds'], (v, _) = _median_time(lambda: baseline(m, label, score, L=L), 1)
                row['value'] = float(v)
            elif m == 'pate':
                if T > 20_000:
                    row['note'] = 'skipped above 20000 points'
                else:
                    row['seconds'], (v, _) = _median_time(lambda: baseline('pate', label, score, L=L), 1)
                    row['value'] = float(v)
            else:
                row['note'] = 'unknown metric'
        except Exception as e:  # noqa: BLE001
            row['note'] = str(e)[:200]
        out['results'].append(row)
    return out


def _read_column(data: bytes):
    text = data.decode('utf-8', errors='ignore')
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return np.asarray(arr, dtype=float).ravel()
    except Exception:  # noqa: BLE001
        pass
    rows = [r.strip() for r in text.replace(';', ',').splitlines() if r.strip()]
    vals = []
    for r in rows:
        cell = r.split(',')[-1]
        try:
            vals.append(float(cell))
        except ValueError:
            continue                                            # a header row
    return np.asarray(vals, dtype=float)


@app.post('/upload')
async def upload(label: UploadFile = File(...), score: UploadFile = File(...)):
    lab = _read_column(await label.read())
    sco = _read_column(await score.read())
    n = min(len(lab), len(sco))
    if n < 10:
        raise HTTPException(400, 'could not read two columns of numbers')
    lab = (lab[:n] > 0.5).astype(int)
    if lab.sum() == 0:
        raise HTTPException(400, 'the label has no anomaly')
    uid = uuid.uuid4().hex[:12]
    UPLOADS[uid] = {'label': lab, 'score': sco[:n], 'values': sco[:n], 'name': score.filename, 'n': n}
    return {'id': uid, 'file': f'upload:{uid}', 'n': n, 'anomalies': len(runs(lab)), 'name': score.filename}


# ---------------------------------------------------------------- endpoints: notes
class Notes(BaseModel):
    notes: dict


@app.get('/notes')
def notes_get():
    path = os.path.join(DATA, 'notes.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@app.put('/notes')
def notes_put(body: Notes):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, 'notes.json'), 'w') as f:
        json.dump(body.notes, f, indent=1)
    return {'ok': True}


# ---------------------------------------------------------------- endpoints: local evaluation
# The Local, Finder, Coverage, Buffers and Suspicious pages read the tables
# app/export_local.py writes to app/data/<benchmark>/local/; the game scores
# its rounds live.
LOCAL_TABLES = ('pairs', 'cover', 'buffers_20', 'buffers_128', 'odd', 'game')


@lru_cache(maxsize=32)
def _local_bytes(path, mtime):
    with open(path, 'rb') as f:
        return f.read()


def local_file(benchmark, *parts):
    parser(benchmark)                                  # 404 on an unknown benchmark
    path = os.path.join(DATA, benchmark, 'local', *parts)
    if not os.path.exists(path):
        raise HTTPException(404, 'not exported; run app/export_local.py')
    return Response(content=_local_bytes(path, os.path.getmtime(path)), media_type='application/json')


@app.get('/local/table/{name}')
def local_table(name: str, b: str = Query('tsb-ad')):
    if name not in LOCAL_TABLES:
        raise HTTPException(404, f'unknown table {name}')
    return local_file(b, name + '.json')


@app.get('/local/series/{file}')
def local_series(file: str, b: str = Query('tsb-ad')):
    if '/' in file or file.startswith('.'):
        raise HTTPException(400, 'bad file name')
    return local_file(b, 'series', file + '.json')


@lru_cache(maxsize=64)
def game_round(benchmark, file, detector, k, L):
    """One round: the window around anomaly k, and the per-side buffers AutoVUS picks there."""
    label, score, values = load_pair(benchmark, file, detector)
    auto = AutoVUS(FastVUS(L=L, cumulative=True), sides='per_side')
    t0 = time.perf_counter()
    auto.compute(label, score)
    seconds = time.perf_counter() - t0
    if not 0 <= k < len(auto.anomalies):
        raise HTTPException(404, 'no such anomaly')
    a = auto.anomalies[k]
    st, en = int(a['start']), int(a['end'])
    lo, hi = max(0, st - 2 * L), min(len(label), en + 2 * L + 1)
    v = scaled(values[lo:hi]) if values is not None else np.zeros(hi - lo)
    s = scaled(score)[lo:hi]
    return {'file': file, 'detector': detector, 'k': k, 'L': L, 'lo': lo, 'hi': hi, 'start': st, 'end': en,
            'b_left': int(a['left']['buffer_val']), 'b_right': int(a['right']['buffer_val']), 'type': a['detection'],
            'seconds': seconds, 'values': [round(float(x), 4) for x in v], 'score': [round(float(x), 4) for x in s]}


@app.get('/game/rounds')
def game_rounds(b: str = Query('tsb-ad')):
    path = os.path.join(DATA, b, 'local', 'game.json')
    if not os.path.exists(path):
        raise HTTPException(404, 'no game.json; run app/export_local.py')
    with open(path) as f:
        g = json.load(f)
    out = []
    for r in g['rounds']:
        try:
            out.append(game_round(b, r['file'], r['detector'], int(r['k']), int(g.get('L', 20))))
        except HTTPException:
            continue
    return {'L': g.get('L', 20), 'rounds': out}
