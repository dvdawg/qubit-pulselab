import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import Pulse, gaussian_drag
from pulselab.dynamics.lindblad import liouvillian, lindblad_propagate, collapse_operators
from pulselab.dynamics.propagator import propagate


def test_liouvillian_zero_dissipation_preserves_trace_and_is_unitary_like():
    # With no collapse ops, a diagonal H just adds phases: populations unchanged.
    H = np.diag([0.0, 1.0]).astype(complex)
    L = liouvillian(H, [])
    rho0 = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
    from scipy.linalg import expm
    vec = expm(L * 0.5) @ rho0.flatten(order="F")
    rho = vec.reshape((2, 2), order="F")
    assert np.isclose(np.trace(rho).real, 1.0)               # trace preserved
    assert np.allclose(np.diag(rho).real, np.diag(rho0).real)  # populations fixed (diagonal H)


def test_zero_drive_no_dissipation_is_identity_on_populations():
    model = ChargeBasisTransmon(DeviceParams.q1())
    n = 50
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = np.zeros((model.n_levels, model.n_levels), dtype=complex)
    rho0[1, 1] = 1.0  # start in |1>
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops=[])
    assert np.isclose(rho[1, 1].real, 1.0, atol=1e-6)


def test_lindblad_without_dissipation_matches_coherent_propagator():
    # With no collapse operators, the master equation must reproduce unitary
    # evolution rho -> U rho U^dagger. A real (off-diagonal) drive generates
    # coherences, so this discriminates the column-stacking order="F" convention
    # from a row-stacking mismatch that population-only tests cannot catch.
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = (np.pi / 2) / probe.area()  # ~pi/2 rotation -> strong 0-1 coherence
    pulse = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())
    d = model.n_levels
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1.0  # start in |0>
    U = propagate(model, pulse, model.f01_ghz())
    rho_coherent = U @ rho0 @ U.conj().T
    rho_lindblad = lindblad_propagate(model, pulse, model.f01_ghz(), rho0, c_ops=[])
    assert np.allclose(rho_lindblad, rho_coherent, atol=1e-8)


def test_t1_decay_matches_exponential():
    # Short T1 so decay is visible over a few hundred ns; idle (zero drive).
    T1_us = 0.1  # 100 ns
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    c_ops = collapse_operators(2, T1_us=T1_us, Tphi_us=1e9, include_dephasing=False)
    t_total = 100.0  # one T1
    n = int(t_total)
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1>
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
    # After one T1, excited population ~ 1/e.
    assert np.isclose(rho[1, 1].real, np.exp(-1), atol=0.02)


def test_pure_dephasing_decays_coherence():
    Tphi_us = 0.1  # 100 ns
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    c_ops = collapse_operators(2, T1_us=1e9, Tphi_us=Tphi_us, include_relaxation=False)
    n = 100
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = 0.5 * np.ones((2, 2), dtype=complex)  # |+> : coherence = 0.5
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
    assert np.isclose(abs(rho[0, 1]), 0.5 * np.exp(-1), atol=0.02)
