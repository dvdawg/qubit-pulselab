"""Headless demo: DRAG suppresses leakage on the real Q1 transmon."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import leakage


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    for coef in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = gaussian_drag(40, amp, 8.0, coef, model.anharmonicity_ghz())
        U = propagate(model, p, model.f01_ghz())
        print(f"drag_coef={coef:.2f}  leakage={leakage(U):.3e}")


if __name__ == "__main__":
    main()
