import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import HardwareStage, IdentityStage, Chain


def _pulse(n=8):
    t = np.arange(n) * 1.0
    return Pulse(t=t, I=np.linspace(0, 1, n), Q=np.linspace(1, 0, n))


def test_identity_stage_roundtrip_and_jacobian():
    p = _pulse()
    out = IdentityStage().apply(p)
    assert np.allclose(out.I, p.I) and np.allclose(out.Q, p.Q)
    assert out is not p  # new object, no mutation
    J = IdentityStage().jacobian(p)
    assert J.shape == (16, 16)
    assert np.allclose(J, np.eye(16))


def test_chain_applies_in_order_and_composes_jacobian():
    p = _pulse()
    chain = Chain([IdentityStage(), IdentityStage()])
    out = chain.apply(p)
    assert np.allclose(out.I, p.I)
    assert np.allclose(chain.jacobian(p), np.eye(16))


def test_chain_jacobian_none_if_any_stage_nonlinear():
    class NL(HardwareStage):
        def apply(self, pulse):
            return pulse
        # inherits default jacobian -> None
    assert Chain([IdentityStage(), NL()]).jacobian(_pulse()) is None
