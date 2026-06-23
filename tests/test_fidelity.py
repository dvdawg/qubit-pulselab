import numpy as np
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage, subspace_block

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_identity_block_unit_fidelity_zero_leakage():
    U = np.eye(5, dtype=complex)
    assert np.isclose(avg_gate_fidelity(U, I2), 1.0)
    assert np.isclose(leakage(U), 0.0)


def test_perfect_x_gate():
    U = np.zeros((5, 5), dtype=complex)
    U[0, 1] = U[1, 0] = 1.0  # X on the qubit subspace
    U[2, 2] = U[3, 3] = U[4, 4] = 1.0
    assert np.isclose(avg_gate_fidelity(U, X), 1.0)
    assert np.isclose(leakage(U), 0.0)


def test_leakage_detected():
    # Half of |0> leaks to |2>.
    U = np.eye(5, dtype=complex)
    U[0, 0] = np.sqrt(0.5)
    U[2, 0] = np.sqrt(0.5)
    assert leakage(U) > 0.0
