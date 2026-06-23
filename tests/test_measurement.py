import numpy as np
from pulselab.metrics.measurement import simulate_readout, measured_cost
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.metrics.fidelity import leakage
from pulselab.optimize.base import Problem


def test_converges_to_truth_ideal_readout():
    est = simulate_readout(0.3, n_shots=200000, readout_fidelity=1.0, seed=0)
    assert np.isclose(est, 0.3, atol=0.01)


def test_assignment_fidelity_biases_toward_half():
    # F=0.8, true p=1.0 -> p_read = 0.8.
    est = simulate_readout(1.0, n_shots=200000, readout_fidelity=0.8, seed=1)
    assert np.isclose(est, 0.8, atol=0.01)


def test_shot_noise_variance_scales():
    p = 0.5
    n = 400
    ests = [simulate_readout(p, n_shots=n, seed=s) for s in range(400)]
    expected_std = np.sqrt(p * (1 - p) / n)
    assert np.isclose(np.std(ests), expected_std, rtol=0.2)


X = np.array([[0, 1], [1, 0]], dtype=complex)


def _setup():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    pulse = gaussian_drag(40, np.pi / probe.area(), 8.0, 0.5, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=10.0)
    return prob, pulse


def test_measured_cost_converges_to_truth_high_shots():
    prob, pulse = _setup()
    U = prob.propagated(pulse)
    p1 = abs(U[1, 0]) ** 2
    true_cost = (1 - p1) + prob.leakage_weight * leakage(U)
    est = measured_cost(prob, pulse, n_shots=500000, readout_fidelity=1.0, seed=0)
    assert np.isclose(est, true_cost, atol=0.01)


def test_measured_cost_is_noisy_at_low_shots():
    prob, pulse = _setup()
    ests = [measured_cost(prob, pulse, n_shots=100, seed=s) for s in range(50)]
    assert np.std(ests) > 0  # finite shots -> fluctuates
