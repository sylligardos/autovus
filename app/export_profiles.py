"""
The tables of the Profiles page: the anomalies of one dataset grouped by their
per-detector local AutoVUS@20, one file per dataset in
app/data/<benchmark>/local/profiles/ plus index.json. The grouping itself is
src/experiments/local_eval/check_profiles.py; no metric is recomputed here.

    nice -n 19 conda run -n autovus python app/export_profiles.py                  # every dataset in range
    nice -n 19 conda run -n autovus python app/export_profiles.py --dataset YAHOO  # one, the index is updated

Every dataset is listed in index.json. A dataset with more than MAX_ANOMALIES
anomalies (Stock) is listed with 'too_many' and gets no file: too many points
for this drawing.

Inputs:  app/data/<benchmark>/local/cover.json and series/<series>.json (app/export_local.py)
Output:  profiles/<dataset>.json
    series_graph           the series as dots: a series is the share of its anomalies in each profile;
                           each series linked to its 10 nearest by Euclidean distance on these shares
                           (fewer when the dataset holds fewer series), positions in [0, 1], edges
    cover                  the 3 detectors that cover the dataset. The score of a set = the mean over the
                           anomalies of the best local AutoVUS any detector of the set reaches; the set is
                           greedy (the best detector alone, then twice the one that raises the score most).
                           Per detector: 'best' = the score of the set up to it, its own mean, its mean
                           seconds per series (the benchmark's runtime/<detector>.csv), and 'similar' =
                           among the detectors that take less time, the one whose scores over the anomalies
                           correlate most with its own (Pearson; null when none is faster).
                           'all' = the score of every detector together.
    left_out, dropped      detectors left out (check_profiles.EXCLUDE) and anomalies dropped for a missing run
    detectors, series      names
    s, a, start, end       per anomaly: its series, its index in the series, its points
    X                      per anomaly: local AutoVUS x100, one value per detector
    graph                  each anomaly linked to its 10 nearest by Euclidean distance on the raw
                           scores: positions in [0, 1], edges, Louvain groups (largest first,
                           -1 = fewer than 3 anomalies), stability over 10 seeds
"""
import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                        # threaded linear algebra segfaults in this env

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'experiments' / 'local_eval'))
from tsad_parser import TSADParser  # noqa: E402
from check_profiles import FOUND, K_LINKS, SEEDS, edges_dist, load, louvain  # noqa: E402

MAX_ANOMALIES = 3000


def unit(P):
    """Positions scaled to [0, 1]; an axis without spread sits in the middle."""
    span = P.max(0) - P.min(0)
    return np.where(span > 1e-9, (P - P.min(0)) / np.maximum(span, 1e-9), 0.5)


def graph(label, n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G, weight='weight', seed=0)
    P = np.array([pos[i] for i in range(n)])
    P = unit(P)
    labs = [louvain(n, edges, s) for s in SEEDS]
    return {'label': label, 'pos': [[round(float(x), 4), round(float(y), 4)] for x, y in P],
            'edges': [[i, j, round(w, 5)] for i, j, w in G.edges(data='weight')],
            'group': labs[0].tolist(), 'counts': [len(set(l[l >= 0])) for l in labs],
            'stability': round(float(np.mean([ari(a, b) for a, b in combinations(labs, 2)])), 2)}


def series_graph(group, s, S):
    """The series by what they contain: the share of the anomalies of a series in each profile."""
    g = np.asarray(group) + 1                             # 0 = the anomalies outside every profile
    H = np.zeros((S, g.max() + 1))
    np.add.at(H, (np.asarray(s), g), 1)
    H /= H.sum(1, keepdims=True)
    k = min(K_LINKS, S - 1)
    G = nx.Graph()
    G.add_nodes_from(range(S))
    if k > 0:
        dist, nb = NearestNeighbors(n_neighbors=k + 1).fit(H).kneighbors(H)
        G.add_weighted_edges_from((i, int(j), 1 / (1 + float(d))) for i in range(S) for d, j in zip(dist[i], nb[i]) if j != i)
    pos = nx.spring_layout(G, weight='weight', seed=0)
    P = np.array([pos[i] for i in range(S)])
    P = unit(P)
    return {'pos': [[round(float(x), 4), round(float(y), 4)] for x, y in P],
            'edges': [[i, j, round(w, 5)] for i, j, w in G.edges(data='weight')]}


def cover(X, detectors, files, runtime):
    """The greedy set of 3 detectors, each with its time and the most similar detector that is faster."""
    best = lambda S: float(X[:, S].max(1).mean())
    mean = X.mean(0)
    sub = runtime[runtime['File Name'].isin(files)].groupby('Model Name')['Time Cost'].mean()
    sec = np.array([float(sub.get(d, np.nan)) for d in detectors])
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.nan_to_num(np.corrcoef(X.T))               # a constant detector correlates with none
    info = lambda j: {'detector': detectors[j], 'score': round(float(mean[j]), 3), 'seconds': None if np.isnan(sec[j]) else round(float(sec[j]), 3)}
    S, rows = [], []
    for _ in range(3):
        j = max((d for d in range(len(detectors)) if d not in S), key=lambda d: best(S + [d]))
        S.append(j)
        rows.append({**info(j), 'best': round(best(S), 3)})
    for j, row in zip(S, rows):                           # the similar detector is never one of the three
        quicker = [i for i in range(len(detectors)) if sec[i] < sec[j] and i not in S]
        like = max(quicker, key=lambda i: C[j, i]) if quicker else None
        row['similar'] = None if like is None else {**info(like), 'corr': round(float(C[j, like]), 2)}
    return {'all': round(best(list(range(len(detectors)))), 3), 'rows': rows}


def write(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def export(benchmark, dataset, local, runtime):
    X, series, k, names, detectors, dropped, left_out = load(benchmark, dataset)
    n = len(X)
    used = sorted(set(series.tolist()))
    at = {s: i for i, s in enumerate(used)}
    spans = {s: json.load(open(local / 'series' / (names[s] + '.json')))['anomalies'] for s in used}
    out = {
        'dataset': dataset, 'k': K_LINKS, 'found': FOUND, 'dropped': dropped, 'left_out': left_out,
        'detectors': detectors, 'series': [names[s] for s in used],
        's': [at[s] for s in series.tolist()], 'a': k.tolist(),
        'start': [spans[s][j][0] for s, j in zip(series.tolist(), k.tolist())],
        'end': [spans[s][j][1] for s, j in zip(series.tolist(), k.tolist())],
        'X': np.round(X * 100).astype(int).tolist(),
        'graph': graph('Raw distance', n, edges_dist(X)),
    }
    g = out['graph']
    out['series_graph'] = series_graph(g['group'], out['s'], len(used))
    out['cover'] = cover(X, detectors, out['series'], runtime)
    write(out, local / 'profiles' / (dataset + '.json'))
    row = {'dataset': dataset, 'anomalies': n, 'series': len(used), 'detectors': len(detectors),
           'profiles': max(g['group']) + 1, 'stability': g['stability']}
    print(f"{dataset:<10} {n:>5} anomalies {len(used):>4} series {len(detectors)} detectors (left out: {', '.join(left_out) or 'none'}; "
          f"{dropped} anomalies dropped)  {row['profiles']:>2} profiles, stability {g['stability']:.2f}   " + ', '.join(f"{x['detector']} {x['best']:.2f} ({x['seconds']}s; like {x['similar']['detector']} r={x['similar']['corr']} {x['similar']['seconds']}s)" if x['similar'] else f"{x['detector']} {x['best']:.2f} ({x['seconds']}s; none faster)" for x in out['cover']['rows']), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='tsb-ad')
    ap.add_argument('--dataset', default=None)
    a = ap.parse_args()
    local = ROOT / 'app' / 'data' / a.benchmark / 'local'
    index_path = local / 'profiles' / 'index.json'
    runtime = pd.concat([pd.read_csv(f) for f in sorted(TSADParser().runtime_path.glob('*.csv'))], ignore_index=True)
    if a.dataset:
        rows = {r['dataset']: r for r in (json.load(open(index_path)) if index_path.exists() else [])}
        rows[a.dataset] = export(a.benchmark, a.dataset, local, runtime)
    else:
        cover = json.load(open(local / 'cover.json'))
        count = {}
        for name, K in zip(cover['series'], cover['K']):
            count[name.split('_')[1]] = count.get(name.split('_')[1], 0) + K
        series = {}
        for name in cover['series']:
            series[name.split('_')[1]] = series.get(name.split('_')[1], 0) + 1
        rows = {d: export(a.benchmark, d, local, runtime) if c <= MAX_ANOMALIES
                else {'dataset': d, 'anomalies': c, 'series': series[d], 'too_many': True} for d, c in count.items()}
    write(sorted(rows.values(), key=lambda r: -r['anomalies']), index_path)
    print(f'wrote {index_path}')


if __name__ == '__main__':
    main()
