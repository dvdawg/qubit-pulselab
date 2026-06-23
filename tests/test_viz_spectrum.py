import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.viz import drive_spectrum


def test_single_tone_peaks_at_its_frequency():
    dt = 1.0
    N = 256
    t = np.arange(N) * dt
    f0 = 0.05  # GHz (cycles/ns)
    env = np.exp(2j * np.pi * f0 * t)  # complex tone at +f0
    p = Pulse(t=t, I=env.real, Q=env.imag)
    freqs, power = drive_spectrum(p)
    assert np.isclose(power.max(), 1.0)
    assert np.isclose(freqs[np.argmax(power)], f0, atol=1.5 / (N * dt))


def test_length_matches():
    t = np.arange(64) * 1.0
    p = Pulse(t=t, I=np.ones(64), Q=np.zeros(64))
    freqs, power = drive_spectrum(p)
    assert freqs.shape == (64,) and power.shape == (64,)
