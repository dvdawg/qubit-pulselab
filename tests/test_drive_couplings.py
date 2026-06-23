import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_couplings_normalized_and_harmonic_limit():
    # Deep transmon (large EJ/EC) -> couplings approach sqrt(j+1).
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.2, EJ_ghz=50.0, n_levels=4))
    g = m.drive_couplings()
    assert g.shape == (3,)
    assert np.isclose(g[0], 1.0)
    expected = np.sqrt([2.0, 3.0])  # g[1]/g[0], g[2]/g[0] ~ sqrt(2), sqrt(3)
    assert np.allclose(g[1:], expected, rtol=0.05)
