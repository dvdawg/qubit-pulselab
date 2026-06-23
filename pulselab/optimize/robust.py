import numpy as np
from .base import Problem


def robust_cost(problems, pulse):
    """Mean cost of a pulse over an ensemble of Problem variants."""
    return float(np.mean([p.cost_from_pulse(pulse) for p in problems]))


class EnsembleProblem:
    """Duck-types Problem: cost_from_pulse averages over an ensemble.

    Works with any optimizer that only calls cost_from_pulse (DRAG, CRAB).
    """

    def __init__(self, problems):
        self.problems = list(problems)

    def cost_from_pulse(self, pulse):
        return robust_cost(self.problems, pulse)


def detuning_ensemble(problem, offsets_ghz):
    """Problems identical to `problem` but with the drive frequency shifted by each offset."""
    return [
        Problem(problem.model, target=problem.target,
                drive_freq_ghz=problem.drive_freq_ghz + off,
                hardware=problem.hardware, leakage_weight=problem.leakage_weight)
        for off in offsets_ghz
    ]
