"""Local C score (VUS-PR, L=20, AutoVUS-style random baseline) for good,
mediocre and random detectors on 10 real TSB-AD series.

    conda run -n autovus python detectors_check.py
good     = best real detector by global VUS-PR on the series
mediocre = median-ranked real detector
random   = 20 uniform random scores
smooth   = 20 random scores smoothed with a 20-point moving average
Output: experiments/local_eval/detectors_check.csv (one row per anomaly and detector)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app" / "api"))
from autovus import FastVUS
from tsad_parser import TSADParser
from bench import load_series, load_score
from vus_local import local_vus

L, R = 20, 20
SERIES = ["772_YAHOO_id_222_WebService_tr_500_1st_85.csv",
          "682_YAHOO_id_132_WebService_tr_500_1st_354.csv",
          "704_YAHOO_id_154_WebService_tr_500_1st_541.csv",
          "282_NEK_id_6_WebService_tr_500_1st_64.csv",
          "278_NEK_id_2_WebService_tr_500_1st_64.csv",
          "010_NAB_id_10_WebService_tr_500_1st_271.csv",
          "001_NAB_id_1_Facility_tr_1007_1st_2014.csv",
          "141_MSL_id_2_Sensor_tr_500_1st_550.csv",
          "269_IOPS_id_10_WebService_tr_1216_1st_1728.csv",
          "295_TODS_id_9_Synthetic_tr_1250_1st_2046.csv"]


def main():
    p = TSADParser()
    dets = list(p.get_detectors())
    rng = np.random.default_rng(0)
    rows = []
    for f in SERIES:
        _, y = load_series(p, f)
        S = {d: load_score(p, f, d) for d in dets}
        S = {d: s for d, s in S.items() if s is not None and len(s) == len(y)}
        glob = {d: FastVUS(L=L).compute(y, s) for d, s in S.items()}
        ranked = sorted(S, key=lambda d: -glob[d])
        runs = [("good", ranked[0], S[ranked[0]]), ("mediocre", ranked[len(ranked) // 2], S[ranked[len(ranked) // 2]])]
        for i in range(R):
            u = rng.random(len(y))
            runs.append(("random", f"rand{i}", u))
            runs.append(("smooth", f"smooth{i}", np.convolve(u, np.ones(20) / 20, mode="same")))
        for cls, name, s in runs:
            r = local_vus(y, s, L)
            for k, v in enumerate(r["C_adj"]):
                rows.append({"series": f.split("_WebService")[0].split("_Sensor")[0].split("_Facility")[0].split("_Synthetic")[0],
                             "T": len(y), "K": len(r["C_adj"]), "class": cls, "detector": name,
                             "vus": r["vus_global"], "k": k, "C_adj": v, "random_k": r["random_k"][k]})
        print(f"done {f} good={ranked[0]} ({glob[ranked[0]]:.2f}) mediocre={ranked[len(ranked)//2]} ({glob[ranked[len(ranked)//2]]:.2f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "experiments" / "local_eval" / "detectors_check.csv", index=False)


if __name__ == "__main__":
    main()
