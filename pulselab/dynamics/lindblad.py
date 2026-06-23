import numpy as np
from scipy.linalg import expm


def collapse_operators(n_levels, T1_us, Tphi_us,
                       include_relaxation=True, include_dephasing=True):
    """Lindblad collapse operators for a transmon ladder (rates in 1/ns).

    Parameters
    ----------
    n_levels : int
        Number of energy levels in the system.
    T1_us : float
        Relaxation time constant in microseconds.
    Tphi_us : float
        Dephasing time constant in microseconds.
    include_relaxation : bool, optional
        Include relaxation (T1) collapse operator. Default True.
    include_dephasing : bool, optional
        Include dephasing (Tphi) collapse operator. Default True.

    Returns
    -------
    list[np.ndarray]
        List of Lindblad collapse operators, each of shape (n_levels, n_levels)
        and dtype complex.
    """
    ops = []
    if include_relaxation:
        gamma1 = 1.0 / (T1_us * 1000.0)
        A = np.zeros((n_levels, n_levels), dtype=complex)
        for j in range(1, n_levels):
            A[j - 1, j] = np.sqrt(j)  # lowering ladder
        ops.append(np.sqrt(gamma1) * A)
    if include_dephasing:
        gamma_phi = 1.0 / (Tphi_us * 1000.0)
        N = np.diag(np.arange(n_levels)).astype(complex)
        ops.append(np.sqrt(2 * gamma_phi) * N)
    return ops


def liouvillian(H, c_ops):
    """Column-stacking Liouvillian superoperator (d^2, d^2)."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in c_ops:
        cdc = c.conj().T @ c
        L += (np.kron(c.conj(), c)
              - 0.5 * np.kron(I, cdc)
              - 0.5 * np.kron(cdc.T, I))
    return L


def lindblad_propagate(model, pulse, drive_freq_ghz, rho0, c_ops):
    """Piecewise-constant master-equation evolution; returns final rho (d,d)."""
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    dt = pulse.dt
    d = H0.shape[0]
    vec = rho0.astype(complex).flatten(order="F")
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Hk = H0 + Ik * X_op + Qk * Y_op
        vec = expm(liouvillian(Hk, c_ops) * dt) @ vec
    return vec.reshape((d, d), order="F")
