"""Example files for "Upload your data" on the Explore page of the demo: series that are NOT in TSB-AD,
one per domain, with the detector scores their benchmark ships.

    conda run -n autovus python src/experiments/make_upload_examples.py

Reads ~/Documents/tsad-benchmarks (TimeEval and TSB-UAD in the TSB-AD format) and writes one CSV per
series to experiments/upload_examples/: the series, the label, then one score column per detector,
the format the page reads (at most 4 MB).

The TSB-UAD scores are shipped rounded to two decimals; they are written as they are. The ECG record
holds 229,899 points, over the 4 MB limit with 12 detectors, so only its first 40,000 points are kept.
"""
import os
from pathlib import Path

import numpy as np

BENCH = Path.home() / "Documents" / "tsad-benchmarks"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "upload_examples"
# output name, benchmark, series file, points kept (None = all)
EXAMPLES = [
    ("traffic_Dodgers", "timeeval", "00001_Dodgers_id_1_Traffic_tr_0_1st_0", None),
    ("medical_ECG_first40000", "tsb-uad", "0087_ECG_id_41_Medical_tr_0_1st_2607", 40_000),
    ("environment_SensorScope", "tsb-uad", "1592_SensorScope_id_10_Environment_tr_0_1st_2496", None),
    ("industrial_Genesis", "tsb-uad", "0226_Genesis_id_1_Sensor_tr_0_1st_15537", None),
    ("building_Occupancy", "tsb-uad", "1177_Occupancy_id_1_Facility_tr_0_1st_0", None),
]


def anomalies(label):
    d = np.diff(np.concatenate([[0], label, [0]]))
    return np.where(d > 0)[0], np.where(d < 0)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, bench, file, keep in EXAMPLES:
        raw = np.loadtxt(BENCH / bench / "raw" / "uni" / f"{file}.csv", delimiter=",", skiprows=1)
        n = len(raw)
        scores = {}
        for d in sorted(os.listdir(BENCH / bench / "scores" / "uni")):
            p = BENCH / bench / "scores" / "uni" / d / f"{file}.npy"
            if p.exists():
                s = np.load(p).astype(float).ravel()
                if len(s) == n and np.all(np.isfinite(s)):
                    scores[d] = s
        m = keep or n
        cols = [raw[:m, 0], raw[:m, 1].astype(int)] + [s[:m] for s in scores.values()]
        path = OUT / f"{name}.csv"
        with open(path, "w") as f:
            f.write("value,label," + ",".join(scores) + "\n")
            for row in zip(*cols):
                f.write(f"{row[0]:.6g},{row[1]}," + ",".join(f"{v:.4g}" for v in row[2:]) + "\n")
        start, end = anomalies(cols[1])
        print(f"{path.name}: {m:,} of {n:,} points, {len(start)} anomalies of {int((end - start).min())} to {int((end - start).max())} points, "
              f"{len(scores)} detectors ({', '.join(scores)}), {path.stat().st_size / 1e6:.2f} MB, from {bench}/{file}")


if __name__ == "__main__":
    main()
