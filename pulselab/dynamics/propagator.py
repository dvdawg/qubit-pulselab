import numpy as np
from scipy.linalg import expm


def propagate(model, pulse, drive_freq_ghz):
    """Coherent piecewise-constant propagation in the rotating frame.

    Returns the (d, d) unitary U = prod_k expm(-i H_k dt).
    """
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    dt = pulse.dt
    d = H0.shape[0]
    U = np.eye(d, dtype=complex)
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Hk = H0 + Ik * X_op + Qk * Y_op
        U = expm(-1j * Hk * dt) @ U
    return U
