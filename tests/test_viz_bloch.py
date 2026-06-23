import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.viz import bloch_trajectory


def _model():
    return ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))


def test_ground_state_is_north_pole():
    model = _model()
    zero = Pulse(t=np.arange(10) * 1.0, I=np.zeros(10), Q=np.zeros(10))
    _, bloch = bloch_trajectory(model, zero, model.f01_ghz())
    assert np.allclose(bloch[0], [0, 0, 1], atol=1e-9)


def test_x90_lands_on_equator():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    x90 = gaussian_drag(40, (np.pi / 2) / probe.area(), 8.0, 0.0, model.anharmonicity_ghz())
    _, bloch = bloch_trajectory(model, x90, model.f01_ghz())
    assert abs(bloch[-1, 2]) < 0.1   # z ~ 0 (on the equator)
    assert np.isclose(np.linalg.norm(bloch[-1]), 1.0, atol=1e-6)  # pure state
