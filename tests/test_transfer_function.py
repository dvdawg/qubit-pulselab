import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import TransferFunction


def _step(n=200, dt=1.0):
    t = np.arange(n) * dt
    return Pulse(t=t, I=np.ones(n), Q=np.zeros(n))


def test_lowpass_unit_dc_gain_and_risetime():
    tau, dt = 10.0, 1.0
    tf = TransferFunction.single_pole_lowpass(tau_ns=tau, dt_ns=dt)
    out = tf.apply(_step())
    # Settles to DC gain 1.
    assert np.isclose(out.I[-1], 1.0, atol=1e-3)
    # One time constant -> ~1 - 1/e of the way up.
    idx = int(round(tau / dt))
    assert np.isclose(out.I[idx], 1 - np.exp(-1), atol=0.05)
    # Q (zero input) stays zero.
    assert np.allclose(out.Q, 0.0)


def test_jacobian_matches_apply():
    tf = TransferFunction.single_pole_lowpass(tau_ns=10.0, dt_ns=1.0)
    n = 30
    p = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    J = tf.jacobian(p)
    assert J.shape == (2 * n, 2 * n)
    # Linear stage: apply(x) == J @ x for arbitrary x (offset is zero).
    rng = np.random.default_rng(0)
    x = rng.normal(size=2 * n)
    px = Pulse(t=p.t, I=x[:n], Q=x[n:])
    ox = tf.apply(px)
    assert np.allclose(np.concatenate([ox.I, ox.Q]), J @ x, atol=1e-9)
