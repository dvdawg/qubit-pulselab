"""Show how drive-line bandwidth limiting degrades a DRAG gate.

The drive line is modeled as a single-pole low-pass with time constant tau:
a LARGER tau means a slower response = narrower bandwidth = more distortion of
the (fast) gate envelope, so fidelity gets WORSE as tau grows.
"""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage

X = np.array([[0, 1], [1, 0]], dtype=complex)


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    pulse = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())
    U = propagate(model, pulse, model.f01_ghz())
    print(f"ideal       F={avg_gate_fidelity(U, X):.5f}  leak={leakage(U):.3e}")
    # Increasing tau = narrowing bandwidth = more distortion -> fidelity falls.
    for tau in [3.0, 5.0, 10.0, 20.0, 30.0]:
        d = Chain([TransferFunction.single_pole_lowpass(tau, 1.0)]).apply(pulse)
        Ud = propagate(model, d, model.f01_ghz())
        print(f"tau={tau:5.1f}ns F={avg_gate_fidelity(Ud, X):.5f}  leak={leakage(Ud):.3e}")


if __name__ == "__main__":
    main()
