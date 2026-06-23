import numpy as np

TWO_PI = 2 * np.pi


def ghz_to_radns(f_ghz):
    """Linear frequency in GHz -> angular frequency in rad/ns."""
    return TWO_PI * np.asarray(f_ghz, dtype=float) if np.ndim(f_ghz) else TWO_PI * f_ghz


def radns_to_ghz(omega):
    """Angular frequency in rad/ns -> linear frequency in GHz."""
    return np.asarray(omega, dtype=float) / TWO_PI if np.ndim(omega) else omega / TWO_PI
