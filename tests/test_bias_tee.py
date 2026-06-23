import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import BiasTeeDroop


def test_droop_of_constant_level():
    tau, dt, n = 100.0, 1.0, 400
    bt = BiasTeeDroop(tau_ns=tau, dt_ns=dt)
    p = Pulse(t=np.arange(n) * dt, I=np.ones(n), Q=np.zeros(n))
    out = bt.apply(p)
    assert np.isclose(out.I[0], 1.0, atol=1e-6)         # starts at full level
    assert out.I[-1] < out.I[0]                          # droops over time
    # After one time constant the level has decayed by ~1/e.
    idx = int(round(tau / dt))
    assert np.isclose(out.I[idx], np.exp(-1), atol=0.05)


def test_jacobian_matches_apply():
    bt = BiasTeeDroop(tau_ns=100.0, dt_ns=1.0)
    n = 40
    p = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    J = bt.jacobian(p)
    rng = np.random.default_rng(1)
    x = rng.normal(size=2 * n)
    px = Pulse(t=p.t, I=x[:n], Q=x[n:])
    ox = bt.apply(px)
    assert np.allclose(np.concatenate([ox.I, ox.Q]), J @ x, atol=1e-9)
