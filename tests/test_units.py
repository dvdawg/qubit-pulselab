import numpy as np
from pulselab.units import ghz_to_radns, radns_to_ghz


def test_ghz_to_radns_scalar():
    assert np.isclose(ghz_to_radns(1.0), 2 * np.pi)


def test_roundtrip_array():
    f = np.array([0.0, 5.252, -0.064])
    assert np.allclose(radns_to_ghz(ghz_to_radns(f)), f)
