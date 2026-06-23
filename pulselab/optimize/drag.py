import numpy as np
from scipy.optimize import minimize
from ..pulse.envelope import gaussian_drag
from .base import Optimizer, OptResult


class DragOptimizer(Optimizer):
    """Optimize [amplitude, drag_coef] of a Gaussian-DRAG pulse."""

    def __init__(self, duration_ns, sigma_ns, anharmonicity_ghz, dt_ns=1.0):
        self.duration_ns = duration_ns
        self.sigma_ns = sigma_ns
        self.anharmonicity_ghz = anharmonicity_ghz
        self.dt_ns = dt_ns

    def _pulse(self, params):
        amp, coef = params
        return gaussian_drag(self.duration_ns, amp, self.sigma_ns, coef,
                             self.anharmonicity_ghz, self.dt_ns)

    def run(self, problem, init_amp, init_drag_coef=0.0):
        history = []

        def objective(params):
            c = problem.cost_from_pulse(self._pulse(params))
            history.append(c)
            return c

        res = minimize(objective, x0=[init_amp, init_drag_coef], method="Nelder-Mead")
        return OptResult(best_pulse=self._pulse(res.x), best_cost=float(res.fun),
                         history=history, n_iter=res.nit,
                         params={"amp": float(res.x[0]), "drag_coef": float(res.x[1])})
