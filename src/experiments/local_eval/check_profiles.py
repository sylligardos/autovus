"""
Check: do the anomalies of one dataset group into detection profiles?

An anomaly is its vector of local AutoVUS@20, one value per detector
(app/data/<benchmark>/local/cover.json). Three ways to group them:

    pyxida   Pyxida's graph as it is: profiles mean-centred per anomaly, edge =
             covariance, 10 strongest positive links per node, Louvain.
    dist     the same graph pipeline on the raw scores: 10 nearest anomalies by
             Euclidean distance, edge = 1 / (1 + distance), Louvain.
    kmeans   k-means on the raw scores, k = 2..15 chosen by silhouette.

Per method: the number of profiles and their sizes, the stability over seeds
(mean pairwise ARI), how far the profiles follow the series (purity against a
shuffled baseline), and where the anomalies found by no detector land.

    conda run -n autovus python src/experiments/local_eval/check_profiles.py --dataset YAHOO
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
K_LINKS = 10      # edges kept per node (Pyxida)
MIN_SIZE = 3      # smaller communities count as "other" (Pyxida)
EXCLUDE = ('Lag_Llama',)   # no run on many anomalies of several datasets (user, 2026-11-23)
FOUND = 0.20      # found: local AutoVUS at least this (the Coverage default)
SEEDS = range(10)


def load(benchmark, dataset):
    """Scores (anomalies x detectors), and per anomaly its series (index into names) and its index in the series.
    The detectors in EXCLUDE are left out; then the anomalies that still miss a run are left out."""
    d = json.load(open(ROOT / 'app' / 'data' / benchmark / 'local' / 'cover.json'))
    D = len(d['detectors'])
    X, series, k = [], [], []
    for i, name in enumerate(d['series']):
        if name.split('_')[1] != dataset:
            continue
        M = np.array(d['M'][i]).reshape(D, -1).T          # (K, D)
        X.append(M); series += [i] * len(M); k += range(len(M))
    X, series, k = np.concatenate(X), np.array(series), np.array(k)
    keep = np.array([n not in EXCLUDE for n in d['detectors']])
    X = X[:, keep]
    ok = (X != -999).all(1)
    return X[ok] / 100, series[ok], k[ok], d['series'], [n for n in d['detectors'] if n not in EXCLUDE], int((~ok).sum()), list(EXCLUDE)


def louvain(n, edges, seed):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(edges)
    comms = sorted(nx.community.louvain_communities(G, weight='weight', seed=seed), key=len, reverse=True)
    lab = np.full(n, -1)
    for c, members in enumerate(c for c in comms if len(c) >= MIN_SIZE):
        lab[list(members)] = c
    return lab                                            # -1 = "other"


def edges_pyxida(X):
    S = X - X.mean(1, keepdims=True)
    W = S @ S.T / S.shape[1]
    np.fill_diagonal(W, -np.inf)
    return [(i, int(j), float(W[i, j])) for i in range(len(X)) for j in np.argsort(W[i])[::-1][:K_LINKS] if W[i, j] > 0]


def edges_dist(X):
    dist, nb = NearestNeighbors(n_neighbors=min(K_LINKS, len(X) - 1) + 1).fit(X).kneighbors(X)   # fewer links in a tiny dataset
    return [(i, int(j), 1 / (1 + float(dd))) for i in range(len(X)) for dd, j in zip(dist[i], nb[i]) if j != i]


def purity(lab, series):
    """Share of anomalies in the main profile of their series, over the series with 2+ anomalies."""
    hit = tot = split = multi = 0
    for s in np.unique(series):
        l = lab[series == s]
        if len(l) < 2:
            continue
        counts = np.unique(l, return_counts=True)[1]
        hit += counts.max(); tot += len(l); multi += 1; split += len(counts) > 1
    return hit / tot, split, multi


def report(name, labs, X, series, detectors):
    lab = labs[0]
    ids = [c for c in np.unique(lab) if c >= 0]
    sizes = np.array([(lab == c).sum() for c in ids])
    none = X.max(1) < FOUND
    stab = np.mean([ari(a, b) for a, b in combinations(labs, 2)])
    p, split, multi = purity(lab, series)
    rng = np.random.default_rng(0)
    p0 = np.mean([purity(rng.permutation(lab), series)[0] for _ in range(100)])
    print(f'\n=== {name} ===')
    print(f'profiles: {len(ids)}   other: {(lab == -1).sum()} anomalies   profiles per seed: {[len(set(l[l >= 0])) for l in labs]}')
    print(f'stability (mean pairwise ARI, {len(labs)} seeds): {stab:.2f}')
    print(f'series with 2+ anomalies: {multi}, split over 2+ profiles: {split}   purity {p:.2f} (shuffled {p0:.2f})')
    print(f'top-2 profiles cover {np.sort(sizes)[::-1][:2].sum() / len(lab):.0%}')
    print(f'{"#":>3} {"n":>5} {"share":>6} {"series":>6} {"mean":>5} {"best":>5} {"none":>4}  top detectors (centre)')
    for c in list(ids) + ([-1] if (lab == -1).any() else []):
        m = lab == c
        centre = X[m].mean(0)
        top = ', '.join(f'{detectors[j]} {centre[j]:.2f}' for j in np.argsort(centre)[::-1][:3])
        print(f'{("other" if c < 0 else c):>3} {m.sum():>5} {m.mean():>6.1%} {len(set(series[m])):>6} {X[m].mean():>5.2f} '
              f'{X[m].max(1).mean():>5.2f} {(m & none).sum():>4}  {top}')
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='tsb-ad')
    ap.add_argument('--dataset', default='YAHOO')
    a = ap.parse_args()
    X, series, _, _, detectors, dropped, left_out = load(a.benchmark, a.dataset)
    print(f'{a.dataset}: {len(X)} anomalies, {len(set(series))} series, {X.shape[1]} detectors; '
          f'{dropped} anomalies dropped (a detector did not run), detectors left out: {left_out}; found by none (< {FOUND}): {(X.max(1) < FOUND).sum()}')

    e = edges_pyxida(X)
    lp = report('pyxida (centred covariance)', [louvain(len(X), e, s) for s in SEEDS], X, series, detectors)
    e = edges_dist(X)
    ld = report('dist (raw scores, Euclidean)', [louvain(len(X), e, s) for s in SEEDS], X, series, detectors)

    dm = cdist(X, X)                                      # sklearn's own distance pass segfaults in this env
    sil = {k: silhouette_score(dm, KMeans(k, n_init=10, random_state=0).fit_predict(X), metric='precomputed') for k in range(2, 16)}
    k = max(sil, key=sil.get)
    print('\nk-means silhouette per k: ' + '  '.join(f'{kk}:{v:.2f}' for kk, v in sil.items()))
    lk = report(f'kmeans (raw scores, k = {k})', [KMeans(k, n_init=10, random_state=s).fit_predict(X) for s in SEEDS], X, series, detectors)

    print(f'\nagreement (ARI): pyxida/dist {ari(lp, ld):.2f}   pyxida/kmeans {ari(lp, lk):.2f}   dist/kmeans {ari(ld, lk):.2f}')


if __name__ == '__main__':
    main()
