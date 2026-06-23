import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import Chain, TransferFunction, BiasTeeDroop, IQImbalance


def test_chain_jacobian_matches_finite_difference():
    n = 24
    rng = np.random.default_rng(5)
    p = Pulse(t=np.arange(n) * 1.0, I=rng.normal(size=n), Q=rng.normal(size=n))
    chain = Chain([
        TransferFunction.single_pole_lowpass(8.0, 1.0),
        BiasTeeDroop(200.0, 1.0),
        IQImbalance(gain_imbalance=0.05, phase_error_rad=0.02, dc_i=0.01, dc_q=0.0),
    ])
    J = chain.jacobian(p)
    assert J is not None and J.shape == (2 * n, 2 * n)
    # Finite-difference each input sample (offsets cancel in the difference).
    base = chain.apply(p)
    base_vec = np.concatenate([base.I, base.Q])
    x0 = np.concatenate([p.I, p.Q])
    eps = 1e-6
    for k in range(2 * n):
        xk = x0.copy(); xk[k] += eps
        pk = Pulse(t=p.t, I=xk[:n], Q=xk[n:])
        ok = chain.apply(pk)
        col = (np.concatenate([ok.I, ok.Q]) - base_vec) / eps
        assert np.allclose(col, J[:, k], atol=1e-5)
