import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import Pulse
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import cost_grad_distorted

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _setup():
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=3))
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=5.0)
    rng = np.random.default_rng(0)
    N = 12
    I = 0.1 * rng.normal(size=N)
    Q = 0.1 * rng.normal(size=N)
    return prob, I, Q, N


def test_cost_matches_problem():
    prob, I, Q, N = _setup()
    t = np.arange(N) * 1.0
    c, _ = cost_grad_distorted(prob, I, Q, dt=1.0)
    assert np.isclose(c, prob.cost_from_pulse(Pulse(t=t, I=I, Q=Q)))


def test_gradient_matches_finite_difference():
    prob, I, Q, N = _setup()
    t = np.arange(N) * 1.0
    _, grad = cost_grad_distorted(prob, I, Q, dt=1.0)
    eps = 1e-6
    fd = np.zeros(2 * N)
    x0 = np.concatenate([I, Q])
    for k in range(2 * N):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        cp = prob.cost_from_pulse(Pulse(t=t, I=xp[:N], Q=xp[N:]))
        cm = prob.cost_from_pulse(Pulse(t=t, I=xm[:N], Q=xm[N:]))
        fd[k] = (cp - cm) / (2 * eps)
    assert np.max(np.abs(grad - fd)) < 1e-5
