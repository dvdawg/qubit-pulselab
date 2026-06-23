import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_bandwidth_limiting_degrades_an_otherwise_good_pulse():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    pulse = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())

    f_ideal = avg_gate_fidelity(propagate(model, pulse, model.f01_ghz()), X)
    # A NARROW-bandwidth low-pass (LARGE tau, comparable to the gate length)
    # smears the drive envelope and significantly degrades the gate. Larger tau
    # = slower line response = narrower bandwidth = more distortion.
    distorted = Chain([TransferFunction.single_pole_lowpass(25.0, 1.0)]).apply(pulse)
    f_distorted = avg_gate_fidelity(propagate(model, distorted, model.f01_ghz()), X)

    assert f_distorted < f_ideal - 0.05
