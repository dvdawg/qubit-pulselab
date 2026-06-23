import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_defaults_and_fields():
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0)
    assert p.n_levels == 5
    assert p.ncut == 25
    assert p.flux == 0.0
    assert np.isclose(p.EJ_effective_ghz(), 55.0)


def test_flux_tuning_at_half_quantum_symmetric():
    # Symmetric SQUID (asymmetry=0) at flux=0.5 -> EJ tunes to 0.
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, flux=0.5, asymmetry=0.0)
    assert np.isclose(p.EJ_effective_ghz(), 0.0, atol=1e-9)


def test_flux_tuning_asymmetry_floor():
    # Asymmetry sets a nonzero floor at flux=0.5.
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, flux=0.5, asymmetry=0.1)
    assert np.isclose(p.EJ_effective_ghz(), 55.0 * 0.1)


def test_from_spectrum_roundtrip():
    p = DeviceParams.from_spectrum(f01_ghz=5.252, anharmonicity_ghz=-0.064)
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
    assert np.isclose(m.anharmonicity_ghz(), -0.064, atol=1e-3)


def test_q1_preset():
    p = DeviceParams.q1()
    assert p.T1_us == 15.0
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
    assert np.isclose(m.anharmonicity_ghz(), -0.064, atol=1e-3)
