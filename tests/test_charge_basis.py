import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_frequencies_shape_and_ground_zero():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.25, EJ_ghz=15.0, n_levels=5))
    w = m.frequencies()
    assert w.shape == (5,)
    assert w[0] == 0.0
    assert np.all(np.diff(w) > 0)  # ascending


def test_perturbative_f01_and_anharmonicity():
    # Transmon perturbative: f01 ~ sqrt(8*EC*EJ) - EC ; anharm ~ -EC.
    # Deep transmon regime (EJ/EC ~ 860) for accurate perturbative predictions.
    EC, EJ = 0.064, 55.0
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=EC, EJ_ghz=EJ))
    f01_approx = np.sqrt(8 * EC * EJ) - EC
    assert np.isclose(m.f01_ghz(), f01_approx, rtol=0.02)
    assert np.isclose(m.anharmonicity_ghz(), -EC, rtol=0.05)
