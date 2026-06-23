import numpy as np
from .fidelity import leakage


def simulate_readout(p_excited, n_shots, readout_fidelity=1.0, seed=None):
    """Measured excited fraction from n_shots projective shots.

    p_read = F*p + (1-F)*(1-p) folds in assignment (SPAM) error; the finite-shot
    estimate is binomial(n_shots, p_read)/n_shots.
    """
    rng = np.random.default_rng(seed)
    p_read = readout_fidelity * p_excited + (1.0 - readout_fidelity) * (1.0 - p_excited)
    return rng.binomial(n_shots, p_read) / n_shots


def measured_cost(problem, pulse, n_shots, readout_fidelity=1.0, seed=None):
    """Finite-shot measured X-gate-error proxy: (1 - measured P_excited) + leakage.

    Models what a single 'apply gate, read out' experiment measures under shot +
    assignment noise. Not differentiable -- use with DRAG/CRAB, not GRAPE.
    """
    U = problem.propagated(pulse)
    p1 = float(abs(U[1, 0]) ** 2)
    measured_p1 = simulate_readout(p1, n_shots, readout_fidelity, seed)
    return (1.0 - measured_p1) + problem.leakage_weight * leakage(U)


class MeasuredProblem:
    """Duck-types Problem: cost_from_pulse is the finite-shot measured cost.

    Lets the derivative-free optimizers (DRAG, CRAB) optimize against realistic
    measurement noise. Not differentiable -- do not use with GRAPE.
    """

    def __init__(self, base_problem, n_shots, readout_fidelity=1.0, seed=None):
        self.base = base_problem
        self.n_shots = n_shots
        self.readout_fidelity = readout_fidelity
        self.seed = seed

    def cost_from_pulse(self, pulse):
        return measured_cost(self.base, pulse, self.n_shots,
                             self.readout_fidelity, self.seed)
