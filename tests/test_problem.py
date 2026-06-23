import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import IdentityStage
from pulselab.optimize.base import Problem

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _model():
    return ChargeBasisTransmon(DeviceParams.q1())


def test_cost_lower_for_better_pulse():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    bare = gaussian_drag(40, amp, 8.0, 0.0, model.anharmonicity_ghz())
    drag = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=10.0)
    assert prob.cost_from_pulse(drag) < prob.cost_from_pulse(bare)


def test_identity_hardware_default():
    model = _model()
    p = gaussian_drag(40, 0.1, 8.0, 0.0, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz())
    # default hardware is identity -> propagated equals direct propagate
    from pulselab.dynamics.propagator import propagate
    assert np.allclose(prob.propagated(p), propagate(model, p, model.f01_ghz()))
