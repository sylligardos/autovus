"""AdjustedVUS: VUS-PR adjusted for what a random detector gets on the same label.

    from autovus import FastVUS, AdjustedVUS     # or FastVUSGPU
    result = AdjustedVUS(FastVUS(L=128, cumulative=True)).compute(label, score)
    result['adjusted']                            # 0 = random, 1 = perfect

The engine computes the VUS-PR curve over the buffers 0..L once. The random
curve is analytical: a random detector's expected AUC-PR at buffer b is the
mass of the buffered label over T (the engine's label_mass), and the curve is
its running mean over buffers, like VUS. The adjusted value is
(vus - random) / (1 - random) at L.
"""
import numpy as np


class AdjustedVUS:
    def __init__(self, engine):
        """
        Args:
            engine: FastVUS or FastVUSGPU built with cumulative=True.
        Defined over VUS-PR only.
        """
        if not engine.cumulative:
            raise ValueError("AdjustedVUS needs an engine built with cumulative=True")
        self.engine = engine

    def curves(self, label, score, cap_left=None, cap_right=None):
        """Cumulative VUS-PR of the detector and of a random detector, per buffer."""
        vus = np.asarray(self.engine.compute(label, score, cap_left, cap_right, metric='pr'))
        mass = np.asarray(self.engine.label_mass) / len(label)
        random = np.cumsum(mass) / np.arange(1, len(mass) + 1)
        return vus, random

    @staticmethod
    def adjust(vus, random):
        """(vus - random) / (1 - random): 0 = random, 1 = perfect, below 0 = worse than random."""
        return (vus - random) / np.maximum(1.0 - random, 1e-10)

    def compute(self, label, score, cap_left=None, cap_right=None):
        """VUS-PR at L, the random baseline at L and the adjusted value, plus both curves."""
        vus, random = self.curves(label, score, cap_left, cap_right)
        return {
            'vus': float(vus[-1]),
            'random': float(random[-1]),
            'adjusted': float(self.adjust(vus[-1], random[-1])),
            'vus_curve': vus,
            'random_curve': random,
        }
