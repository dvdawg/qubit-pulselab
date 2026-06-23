import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.dynamics.propagator import propagate


def test_zero_drive_is_identity_up_to_phase():
    m = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064))
    t = np.arange(40) * 1.0
    zero = Pulse(t=t, I=np.zeros(40), Q=np.zeros(40))
    U = propagate(m, zero, drive_freq_ghz=m.f01_ghz())
    # On resonance, |0> and |1> have zero rotating-frame energy -> identity block.
    assert np.isclose(abs(U[0, 0]), 1.0, atol=1e-6)
    assert np.isclose(abs(U[1, 1]), 1.0, atol=1e-6)


def test_resonant_pi_pulse_inverts_population():
    # Use a 2-level truncation so a resonant pi-area square pulse cleanly
    # inverts the population -- this isolates the Rabi/propagation physics
    # from multi-level leakage (which a hard square pulse would induce on the
    # full transmon). The 2-level case inverts to ~1 exactly.
    m = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    duration = 40.0
    amp = np.pi / duration
    t = np.arange(int(duration)) * 1.0
    sq = Pulse(t=t, I=amp * np.ones(t.size), Q=np.zeros(t.size))
    U = propagate(m, sq, drive_freq_ghz=m.f01_ghz())
    p1 = abs(U[1, 0]) ** 2  # population transferred |0> -> |1>
    assert p1 > 0.999
