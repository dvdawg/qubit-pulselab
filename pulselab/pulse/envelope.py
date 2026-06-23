from dataclasses import dataclass
import numpy as np
from ..units import ghz_to_radns


@dataclass
class Pulse:
    """Microwave drive envelope on a uniform time grid. I,Q in rad/ns; t in ns."""

    t: np.ndarray
    I: np.ndarray
    Q: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def area(self) -> float:
        return float(np.trapz(self.I, self.t))


def gaussian_drag(duration_ns, amp_radns, sigma_ns, drag_coef,
                  anharmonicity_ghz, dt_ns=1.0):
    """Gaussian I envelope with first-order DRAG Q correction."""
    n = int(round(duration_ns / dt_ns))
    t = np.arange(n) * dt_ns
    t0 = duration_ns / 2.0
    I = amp_radns * np.exp(-((t - t0) ** 2) / (2 * sigma_ns ** 2))
    anharm_radns = ghz_to_radns(anharmonicity_ghz)
    Q = drag_coef * (-np.gradient(I, t)) / anharm_radns
    return Pulse(t=t, I=I, Q=Q)
