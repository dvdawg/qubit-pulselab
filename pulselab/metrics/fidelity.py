import numpy as np


def subspace_block(U):
    """Top-left 2x2 computational-subspace block of the propagator."""
    return U[:2, :2]


def leakage(U):
    """Population that leaves the {|0>,|1>} subspace, averaged over the two inputs."""
    M = subspace_block(U)
    retained = np.sum(np.abs(M) ** 2)
    return float(1.0 - retained / 2.0)


def avg_gate_fidelity(U, target):
    """Average gate fidelity of the 2x2 block vs a 2x2 target unitary."""
    M = subspace_block(U)
    d = 2
    t = np.trace(target.conj().T @ M)
    return float((np.abs(t) ** 2 + np.trace(M.conj().T @ M)).real / (d * (d + 1)))
