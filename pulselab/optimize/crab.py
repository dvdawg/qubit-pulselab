import numpy as np
from scipy.optimize import minimize
from ..pulse.envelope import Pulse
from .base import Optimizer, OptResult


class CrabOptimizer(Optimizer):
    """Optimize sine-basis corrections to a base pulse's I and Q envelopes."""

    def __init__(self, base_pulse, n_harmonics):
        self.base = base_pulse
        self.n_harmonics = n_harmonics
        t = base_pulse.t
        T = t[-1] - t[0] + (t[1] - t[0])  # full window incl. last sample
        self._basis = np.stack(
            [np.sin(2 * np.pi * n * (t - t[0]) / T) for n in range(1, n_harmonics + 1)]
        )  # shape (n_harmonics, N)

    def _pulse(self, params):
        aI = params[: self.n_harmonics]
        aQ = params[self.n_harmonics :]
        I = self.base.I + aI @ self._basis
        Q = self.base.Q + aQ @ self._basis
        return Pulse(t=self.base.t.copy(), I=I, Q=Q)

    def run(self, problem):
        history = []

        def objective(params):
            c = problem.cost_from_pulse(self._pulse(params))
            history.append(c)
            return c

        x0 = np.zeros(2 * self.n_harmonics)
        res = minimize(objective, x0=x0, method="Nelder-Mead")
        return OptResult(
            best_pulse=self._pulse(res.x),
            best_cost=float(res.fun),
            history=history,
            n_iter=res.nit,
        )
