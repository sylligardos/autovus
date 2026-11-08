"""Monte Carlo: mean local scores of uniform random detectors (should be ~0 for *_adj)."""
import numpy as np
from examples import examples
from local import local_scores

rng = np.random.default_rng(1)
for name, x, y, A, dets in examples():
    if name[0] not in "256":
        continue
    rs = [local_scores(y, rng.random(len(y))) for _ in range(200)]
    print(name, "check_M max", max(r["check_M"] for r in rs))
    for k in ["A_adj", "B_adj", "roc_adj"]:
        m = np.mean([r[k] for r in rs], axis=0)
        print(f"   {k:8s}", " ".join(f"{v:+.3f}" for v in m))
