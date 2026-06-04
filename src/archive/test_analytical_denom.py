"""
Quick visual check: does E[AUC-PR] = g_b for a Bernoulli score,
and does it hold for both stepwise and linear interpolation?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

rng = np.random.default_rng(42)
T = 1000

# Fixed label: two anomalies, anomaly ratio = 0.12
label = np.zeros(T)
label[200:280] = 1   # length 80
label[500:540] = 1   # length 40
g_b = label.mean()   # = 0.12

print(f"g_b = {g_b:.4f}")

# -------------------------------------------------------------------
# AUC helpers — sklearn versions
# -------------------------------------------------------------------

def auc_linear(label, score):
    """precision_recall_curve + auc: trapezoidal (linear) interpolation."""
    precision, recall, _ = precision_recall_curve(label, score)
    return float(auc(recall, precision))

def auc_stepwise(label, score):
    """average_precision_score: stepwise (right-endpoint) interpolation."""
    return float(average_precision_score(label, score))

# -------------------------------------------------------------------
# Monte Carlo over p values
# -------------------------------------------------------------------

p_values = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
K = 300

sw_means, li_means = [], []

for p in p_values:
    sw_vals, li_vals = [], []
    for _ in range(K):
        score = rng.binomial(1, p, size=T).astype(float)
        sw_vals.append(auc_stepwise(label, score))
        li_vals.append(auc_linear(label, score))
    sw_means.append(np.mean(sw_vals))
    li_means.append(np.mean(li_vals))
    print(f"  p={p:.2f}  stepwise={sw_means[-1]:.4f}  linear={li_means[-1]:.4f}")

# Analytical prediction for linear: g_b + p*(1-g_b)/2
analytical_linear = [g_b + p * (1 - g_b) / 2 for p in p_values]

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(p_values, sw_means, 'o-', color='steelblue', label='E[AUC] stepwise (MC)')
axes[0].axhline(g_b, color='crimson', linestyle='--', label=f'g_b = {g_b:.3f}')
axes[0].set_xlabel('p (Bernoulli probability)')
axes[0].set_ylabel('E[AUC-PR]')
axes[0].set_title('Stepwise — expected to be flat at g_b')
axes[0].legend()
axes[0].set_ylim(0, 0.5)

axes[1].plot(p_values, li_means,        'o-',  color='darkorange', label='E[AUC] linear (MC)')
axes[1].plot(p_values, analytical_linear, '--', color='gray',       label='g_b + p(1–g_b)/2')
axes[1].axhline(g_b, color='crimson', linestyle=':', label=f'g_b = {g_b:.3f}')
axes[1].set_xlabel('p (Bernoulli probability)')
axes[1].set_ylabel('E[AUC-PR]')
axes[1].set_title('Linear — expected to grow with p')
axes[1].legend()
axes[1].set_ylim(0, 0.5)

plt.suptitle(f'E[AUC-PR] vs Bernoulli p   (T={T}, g_b={g_b:.3f}, K={K})', fontsize=11)
plt.tight_layout()
plt.show()
