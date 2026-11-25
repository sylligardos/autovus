"""
Every demo table that depends on the gear, for one gear, written to app/data/<benchmark>/L<L>/.
A gear = the maximum buffer L (20 or 128), the same for VUS and AutoVUS. AutoVUS is per side.

    nice -n 19 conda run -n autovus python app/export_gear.py --L 20 --workers 1

Inputs (experiments/local_benchmark/, not in git):
    series_ps<L>/<series>.npz   AutoVUS per side at L: per-anomaly area and random baseline per buffer,
                                global AutoVUS (src/experiments/local_eval/export_local_ps.py)
    series/<series>.npz         fast128_vus, the global VUS-PR at every L up to 128
    global_adjusted.npz         VUS-PR and Adjusted VUS at every L up to 128
app/data/<benchmark>/scores.json  AUC-PR

Outputs, the same layout as the tables they replace:
    series/<series>.json   anomalies, local AutoVUS@L (detector x anomaly), global AutoVUS@L,
                           global VUS-PR at L = 0..128              (was local/series/, app/export_local.py)
    pairs.json             per (series, detector): K, global AutoVUS, mean and min local   (was local/pairs.json)
    cover.json             per series: the detector x anomaly matrix (x100, -999 = no run)   (was local/cover.json)
    scores.json            the Rankings table: AUC-PR, VUS-PR@L, Adjusted VUS@L, AutoVUS@L per (series,
                           detector), Oracle (best AUC-PR) and Stitched score          (was scores20.json,
                           app/export_rankings20.py). The stitched rows are the only new computation.
    stitched.json          AutoVUS@L of the stitched score per series                  (was local/stitched.json)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS, AdjustedVUS, AutoVUS  # noqa: E402
from tsad_parser import TSADParser  # noqa: E402
from bench import load_series, load_score  # noqa: E402
import oracle as orc  # noqa: E402

EXP = ROOT / "experiments" / "local_benchmark"
GEARS = (20, 128)
STITCHED = {"Stitched score": "median"}
L = 20                      # set by --L in main and in every worker


def setup(L_):
    global L
    L = L_


def r2(x, d=2):
    return [round(float(v), d) for v in np.asarray(x).ravel()]


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def local_of(z):
    """Local AutoVUS@L (D, K) from the per-side export: chance-adjusted mean over the buffers 0..L."""
    return orc._local_adjusted(z[f"ps{L}_area"], z[f"ps{L}_random"], L)


def local_tables(out, detectors):
    """series/*.json, pairs.json and cover.json in one pass over the per-side export."""
    g = np.load(EXP / "global_adjusted.npz")
    gi = {(s, d): i for i, (s, d) in enumerate(zip(g["series"], g["detector"]))}
    di = {d: i for i, d in enumerate(detectors)}
    pairs = {"L": L, "detectors": detectors, "series": [], "s": [], "d": [], "K": [], "a": [], "m": [], "n": []}
    cover = {"L": L, "detectors": detectors, "series": [], "K": [], "M": []}
    files = sorted((EXP / f"series_ps{L}").glob("*.npz"))
    for f in files:
        z = np.load(f)
        name = f.name[:-4]                                   # '<series>.csv'
        dets = [str(d) for d in z["detectors"]]
        loc, glob = local_of(z), z[f"ps{L}_adjusted"]
        K = loc.shape[1]
        vus = np.stack([g["vus"][gi[(name, d)]] for d in dets])          # (D, 129) plain VUS-PR
        # local at 4 decimals: the page picks the stitched winners from it, the export from full precision
        dump({"file": name, "L": L, "anomalies": [[int(a), int(b)] for a, b in zip(z["start"], z["end"])], "detectors": dets,
              "auto": [r2(row, 4) for row in loc], "autoG": r2(glob), "vusG": [r2(row, 3) for row in vus]},
             out / "series" / (name + ".json"))
        si = len(pairs["series"])
        pairs["series"].append(name)
        for j, d in enumerate(dets):
            pairs["s"].append(si); pairs["d"].append(di[d]); pairs["K"].append(K)
            pairs["a"].append(round(float(glob[j]), 3))
            pairs["m"].append(round(float(loc[j].mean()), 3)); pairs["n"].append(round(float(loc[j].min()), 3))
        M = np.full((len(detectors), K), -999, dtype=int)
        M[[di[d] for d in dets]] = np.round(loc * 100).astype(int)
        cover["series"].append(name); cover["K"].append(K); cover["M"].append(M.ravel().tolist())
    dump(pairs, out / "pairs.json")
    dump(cover, out / "cover.json")
    print(f"local: {len(files)} series, {len(pairs['s'])} pairs", flush=True)


def metrics(label, score):
    """AUC-PR, VUS-PR@L and Adjusted VUS@L (the L=128 cumulative curve read at L, as the reused
    numbers) and AutoVUS@L per side."""
    vus, rnd = AdjustedVUS(FastVUS(L=128, cumulative=True)).curves(label, score)
    auto = AutoVUS(FastVUS(L=L, cumulative=True), sides="per_side").compute(label, score)
    return {"auc_pr": float(average_precision_score(label, score)), "vus_pr": float(vus[L]),
            "skill_full": float(AdjustedVUS.adjust(vus[L], rnd[L])), "autovus": float(auto["adjusted"])}


def stitched(file):
    """The stitched row of one series, or {} when there is nothing to stitch."""
    z = np.load(EXP / f"series_ps{L}" / (file + ".npz"))
    if len(z["detectors"]) == 0 or len(z["start"]) == 0:
        return file, {}, []
    p = TSADParser()
    _, label = load_series(p, file)
    winners = [str(z["detectors"][i]) for i in orc.winners_from_local(local_of(z))]
    cache = {}

    def score_of(d):
        if d not in cache:
            cache[d] = load_score(p, file, d)
        return cache[d]

    rows = {}
    for name, norm in STITCHED.items():
        try:
            rows[name] = metrics(label, orc.stitch(label, score_of, winners, norm))
        except Exception as e:  # noqa: BLE001
            rows[name] = {"error": str(e)[:120]}
    return file, rows, winners


def scores_table(out, benchmark, workers):
    """scores.json (the Rankings table) and stitched.json."""
    t0 = time.time()
    with open(ROOT / "app" / "data" / benchmark / "scores.json") as f:
        old = json.load(f)
    g = np.load(EXP / "global_adjusted.npz")
    adj = {(str(s), str(d)): float(a) for s, d, a in zip(g["series"], g["detector"], g["adjusted"][:, L])}
    files, table = [], {}
    for file in sorted(old["files"]):
        path, vpath = EXP / f"series_ps{L}" / (file + ".npz"), EXP / "series" / (file + ".npz")
        if not path.exists() or not vpath.exists():
            continue
        z, v = np.load(path), np.load(vpath)
        vj = {str(d): j for j, d in enumerate(v["detectors"])}
        rows = {}
        for j, d in enumerate(map(str, z["detectors"])):
            ap = old["files"][file].get(d, {}).get("auc_pr")
            a = adj.get((file, d))
            if ap is None or a is None or d not in vj or not np.isfinite(z[f"ps{L}_adjusted"][j]):
                continue
            rows[d] = {"auc_pr": ap, "vus_pr": float(v["fast128_vus"][vj[d], L]), "skill_full": a,
                       "autovus": float(z[f"ps{L}_adjusted"][j])}
        if not rows:
            continue
        best = max(rows, key=lambda d: rows[d]["auc_pr"])
        rows["Oracle"] = dict(rows[best], pick=best)
        table[file] = rows
        files.append(file)
    print(f"scores: reused {len(files)} series, {time.time() - t0:.0f} s", flush=True)

    with Pool(workers, setup, (L,)) as pool:
        for i, (file, rows, winners) in enumerate(pool.imap_unordered(stitched, files), 1):
            for name, r in rows.items():
                table[file][name] = dict(r, winners=winners)
            if i % 100 == 0 or i == len(files):
                print(f"  stitched {i}/{len(files)}, {time.time() - t0:.0f} s", flush=True)

    dump({"L": L, "variants": {"": ["paper", "per_side"]}, "detectors": old["detectors"],
          "references": ["Oracle", *STITCHED], "files": table}, out / "scores.json")
    score = {name: round(float(rows["Stitched score"]["autovus"]), 3) for name, rows in table.items()
             if "autovus" in rows.get("Stitched score", {})}
    dump({"L": L, "score": score}, out / "stitched.json")
    print(f"scores: {len(table)} series, {len(score)} stitched, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True, choices=GEARS)
    ap.add_argument("--benchmark", default="tsb-ad")
    ap.add_argument("--workers", type=int, default=1, help="for the stitched rows")
    ap.add_argument("--only", nargs="*", default=None, help="local scores (default: both)")
    a = ap.parse_args()
    setup(a.L)
    out = ROOT / "app" / "data" / a.benchmark / f"L{a.L}"
    parts = set(a.only or ["local", "scores"])
    if "local" in parts:
        local_tables(out, sorted(str(d) for d in TSADParser().get_detectors()))
    if "scores" in parts:
        scores_table(out, a.benchmark, a.workers)


if __name__ == "__main__":
    main()
