from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from ..dynamics.propagator import propagate
from ..metrics.fidelity import avg_gate_fidelity, leakage
from ..pulse.hardware import IdentityStage


class Problem:
    """Bundles the model, hardware chain, target and weights into one cost."""

    def __init__(self, model, target, drive_freq_ghz, hardware=None, leakage_weight=0.0):
        self.model = model
        self.target = np.asarray(target, dtype=complex)
        self.drive_freq_ghz = drive_freq_ghz
        self.hardware = hardware if hardware is not None else IdentityStage()
        self.leakage_weight = leakage_weight

    def propagated(self, pulse):
        distorted = self.hardware.apply(pulse)
        return propagate(self.model, distorted, self.drive_freq_ghz)

    def cost_from_pulse(self, pulse):
        U = self.propagated(pulse)
        return (1.0 - avg_gate_fidelity(U, self.target)
                + self.leakage_weight * leakage(U))


@dataclass
class OptResult:
    best_pulse: object
    best_cost: float
    history: list = field(default_factory=list)
    n_iter: int = 0
    params: dict = None  # optimizer-specific scalar params (e.g. DRAG amp/drag_coef)


class Optimizer(ABC):
    @abstractmethod
    def run(self, problem, **kwargs) -> OptResult:
        ...
