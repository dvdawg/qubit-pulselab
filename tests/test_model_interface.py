import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon, TransmonModel


def test_is_transmon_model_and_n_levels():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=5))
    assert isinstance(m, TransmonModel)
    assert m.n_levels == 5


def test_rotating_frame_operators_properties():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=4))
    f01 = m.f01_ghz()
    H0, X, Y = m.rotating_frame_operators(drive_freq_ghz=f01)
    assert H0.shape == (4, 4) and X.shape == (4, 4) and Y.shape == (4, 4)
    # On resonance with |0>-|1>, the 0,1 diagonal entries of H0_rot are ~equal.
    assert np.isclose(H0[0, 0].real, H0[1, 1].real, atol=1e-9)
    # Hermiticity of drive operators.
    assert np.allclose(X, X.conj().T)
    assert np.allclose(Y, Y.conj().T)
    # X couples neighbors with g0=1 -> X[1,0] == 0.5.
    assert np.isclose(X[1, 0], 0.5)
