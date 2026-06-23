import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.dynamics.propagator import propagate
from pulselab.viz import state_trajectory, population_trajectory


def _model():
    return ChargeBasisTransmon(DeviceParams.q1())


def test_final_state_matches_propagator():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    pulse = gaussian_drag(40, np.pi / probe.area(), 8.0, 0.5, model.anharmonicity_ghz())
    t_edges, states = state_trajectory(model, pulse, model.f01_ghz())
    psi0 = np.zeros(model.n_levels, dtype=complex); psi0[0] = 1.0
    U = propagate(model, pulse, model.f01_ghz())
    assert states.shape == (pulse.t.size + 1, model.n_levels)
    assert t_edges.size == pulse.t.size + 1
    assert np.allclose(states[-1], U @ psi0, atol=1e-8)


def test_populations_normalized_and_start_in_ground():
    model = _model()
    pulse = gaussian_drag(40, 0.1, 8.0, 0.0, model.anharmonicity_ghz())
    t_edges, pops = population_trajectory(model, pulse, model.f01_ghz())
    assert np.allclose(pops.sum(axis=1), 1.0, atol=1e-8)
    assert np.isclose(pops[0, 0], 1.0)  # starts in |0>
