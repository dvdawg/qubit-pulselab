import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import IQImbalance


def _pulse(n=10):
    rng = np.random.default_rng(2)
    return Pulse(t=np.arange(n) * 1.0, I=rng.normal(size=n), Q=rng.normal(size=n))


def test_defaults_are_identity():
    p = _pulse()
    out = IQImbalance().apply(p)
    assert np.allclose(out.I, p.I) and np.allclose(out.Q, p.Q)


def test_gain_phase_and_dc():
    p = _pulse()
    stage = IQImbalance(gain_imbalance=0.1, phase_error_rad=0.05, dc_i=0.02, dc_q=-0.03)
    out = stage.apply(p)
    eps = 1.1
    assert np.allclose(out.I, p.I + 0.02)
    assert np.allclose(out.Q, eps * (np.sin(0.05) * p.I + np.cos(0.05) * p.Q) - 0.03)


def test_jacobian_is_linear_part():
    n = 10
    p = _pulse(n)
    stage = IQImbalance(gain_imbalance=0.1, phase_error_rad=0.05, dc_i=0.02, dc_q=-0.03)
    J = stage.jacobian(p)
    # Jacobian excludes the DC offset: difference of two inputs cancels it.
    rng = np.random.default_rng(3)
    dx = rng.normal(size=2 * n)
    p2 = Pulse(t=p.t, I=p.I + dx[:n], Q=p.Q + dx[n:])
    o1, o2 = stage.apply(p), stage.apply(p2)
    delta = np.concatenate([o2.I - o1.I, o2.Q - o1.Q])
    assert np.allclose(delta, J @ dx, atol=1e-9)
