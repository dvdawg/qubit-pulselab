import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import ControlNoise


def _zero(n=4096):
    return Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))


def test_seed_reproducible_and_jacobian_none():
    p = _zero(64)
    a = ControlNoise(sigma=0.1, seed=42).apply(p)
    b = ControlNoise(sigma=0.1, seed=42).apply(p)
    assert np.allclose(a.I, b.I) and np.allclose(a.Q, b.Q)
    assert ControlNoise(sigma=0.1, seed=42).jacobian(p) is None


def test_white_noise_statistics():
    p = _zero()
    out = ControlNoise(sigma=0.2, kind="white", seed=0).apply(p)
    assert np.isclose(out.I.std(), 0.2, rtol=0.1)
    assert np.isclose(out.I.mean(), 0.0, atol=0.02)


def test_pink_noise_has_more_low_frequency_power_than_white():
    p = _zero()
    white = ControlNoise(sigma=1.0, kind="white", seed=1).apply(p)
    pink = ControlNoise(sigma=1.0, kind="pink", seed=1).apply(p)
    n = p.I.size
    # Compare low-frequency band power fraction (excluding DC bin).
    def low_frac(x):
        ps = np.abs(np.fft.rfft(x)) ** 2
        ps[0] = 0.0
        return ps[1:n // 16].sum() / ps[1:].sum()
    assert low_frac(pink.I) > low_frac(white.I)
