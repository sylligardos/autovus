"""Summary of detectors_check.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
df = pd.read_csv(ROOT / "experiments" / "local_eval" / "detectors_check.csv")
pd.set_option("display.width", 200)

print("Per series: mean C_adj over anomalies (random/smooth: mean over the 20 runs)")
t = df.groupby(["series", "K", "class"])["C_adj"].mean().unstack("class")[["good", "mediocre", "random", "smooth"]]
g = df[df["class"].isin(["good", "mediocre"])].groupby(["series", "class"])["vus"].first().unstack("class")
t = t.reset_index().merge(g.rename(columns={"good": "VUS good", "mediocre": "VUS mediocre"}).reset_index(), on="series")
print(t.round(3).to_string())

print("\nRandom detectors, per anomaly over the 20 runs (all series pooled):")
for cls in ["random", "smooth"]:
    r = df[df["class"] == cls]
    per = r.groupby(["series", "k"])["C_adj"]
    m = per.mean()
    print(f"  {cls:7s} mean of means {m.mean():+.3f}  range of means [{m.min():+.3f}, {m.max():+.3f}]  "
          f"p95 {r['C_adj'].quantile(.95):+.3f}  max {r['C_adj'].max():+.3f}  share > 0.2 {(r['C_adj'] > 0.2).mean():.3f}")

print("\nPer anomaly ordering (good vs mediocre vs mean random):")
w = df[df["class"].isin(["good", "mediocre"])].pivot_table(index=["series", "k"], columns="class", values="C_adj")
w["random"] = df[df["class"] == "random"].groupby(["series", "k"])["C_adj"].mean()
n = len(w)
print(f"  anomalies {n}; good >= mediocre {(w.good >= w.mediocre).mean():.2f}; "
      f"mediocre > random {(w.mediocre > w.random).mean():.2f}; good > random {(w.good > w.random).mean():.2f}")
print(f"  good: mean {w.good.mean():.3f}, anomalies < 0.1: {(w.good < 0.1).sum()}; "
      f"mediocre: mean {w.mediocre.mean():.3f}, anomalies < 0.1: {(w.mediocre < 0.1).sum()}")
