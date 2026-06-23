import numpy as np
import pytest
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_from_spectrum_still_works():
    p = DeviceParams.from_spectrum(5.252, -0.064)
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)


def test_from_spectrum_raises_on_nonconvergence():
    # Physically impossible target (positive anharmonicity for a transmon)
    # drives the solver to fail; we want a clear error, not silent garbage.
    with pytest.raises(ValueError):
        DeviceParams.from_spectrum(5.252, +5.0)
