import numpy as np
from scipy.linalg import expm
from .pulse.envelope import gaussian_drag, Pulse
from .dynamics.propagator import propagate


def state_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """State after each time slice. Returns (t_edges[N+1], states[N+1, d])."""
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    d = H0.shape[0]
    if psi0 is None:
        psi0 = np.zeros(d, dtype=complex)
        psi0[0] = 1.0
    dt = pulse.dt
    psi = np.asarray(psi0, dtype=complex)
    states = [psi.copy()]
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Uk = expm(-1j * (H0 + Ik * X_op + Qk * Y_op) * dt)
        psi = Uk @ psi
        states.append(psi.copy())
    t_edges = np.concatenate([pulse.t, [pulse.t[-1] + dt]])
    return t_edges, np.array(states)


def population_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """Level populations over time. Returns (t_edges[N+1], pops[N+1, d])."""
    t_edges, states = state_trajectory(model, pulse, drive_freq_ghz, psi0)
    return t_edges, np.abs(states) ** 2


def bloch_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """Qubit-subspace Bloch vector (x,y,z) over time. Returns (t_edges, bloch[N+1,3])."""
    t_edges, states = state_trajectory(model, pulse, drive_freq_ghz, psi0)
    a = states[:, 0]
    b = states[:, 1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return t_edges, np.stack([x, y, z], axis=1)


def drive_spectrum(pulse):
    """Normalized power spectrum of the complex drive envelope I + iQ.

    Returns (freqs_ghz, power) with power.max() == 1.0. freqs are the detuning
    from the drive frequency, in GHz (dt is in ns).
    """
    env = pulse.I + 1j * pulse.Q
    n = env.size
    spec = np.fft.fftshift(np.fft.fft(env))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=pulse.dt))
    power = np.abs(spec) ** 2
    peak = power.max()
    if peak > 0:
        power = power / peak
    return freqs, power


_ALLXY_PAIRS = [
    ("I", "I"), ("X180", "X180"), ("Y180", "Y180"), ("X180", "Y180"), ("Y180", "X180"),
    ("X90", "I"), ("Y90", "I"), ("X90", "Y90"), ("Y90", "X90"), ("X90", "Y180"),
    ("Y90", "X180"), ("X180", "Y90"), ("Y180", "X90"), ("X90", "X180"), ("X180", "X90"),
    ("Y90", "Y180"), ("Y180", "Y90"), ("X180", "I"), ("Y180", "I"), ("X90", "X90"),
    ("Y90", "Y90"),
]


def allxy_populations(model, drive_freq_ghz, duration_ns=40, sigma_ns=8.0, dt_ns=1.0,
                      amp=None, drag_coef=0.0):
    """Simulate the 21 standard AllXY gate pairs; return (labels, P1).

    By default the X180 amplitude is calibrated to area pi and no DRAG is used
    (the canonical staircase). Pass `amp`/`drag_coef` (e.g. an optimized DRAG
    calibration) to build the gates with those values instead; on a multi-level
    model the DRAG correction then suppresses the leakage-induced staircase
    deviations. A Y gate is an X gate with the drive phase rotated 90 degrees
    (Omega_Y = i*Omega_X), which carries the DRAG quadrature correctly.
    """
    anh = model.anharmonicity_ghz()
    n = int(round(duration_ns / dt_ns))
    zero = np.zeros(n)
    if amp is None:
        unit = gaussian_drag(duration_ns, 1.0, sigma_ns, 0.0, anh, dt_ns)
        amp = np.pi / float(np.trapz(unit.I, unit.t))

    def gate(name):
        if name == "I":
            return zero, zero
        a = amp if name.endswith("180") else amp / 2
        g = gaussian_drag(duration_ns, a, sigma_ns, drag_coef, anh, dt_ns)
        if name[0] == "X":
            return g.I, g.Q
        return -g.Q, g.I  # Y gate: Omega_Y = i*Omega_X

    t2 = np.arange(2 * n) * dt_ns
    labels, p1 = [], []
    for g1, g2 in _ALLXY_PAIRS:
        I1, Q1 = gate(g1)
        I2, Q2 = gate(g2)
        pulse = Pulse(t=t2, I=np.concatenate([I1, I2]), Q=np.concatenate([Q1, Q2]))
        U = propagate(model, pulse, drive_freq_ghz)
        labels.append(f"{g1}-{g2}")
        p1.append(float(abs(U[1, 0]) ** 2))
    return labels, p1
