import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_eigensystem_is_cached():
    m = ChargeBasisTransmon(DeviceParams.q1())
    a = m._eigensystem()
    b = m._eigensystem()
    # Same cached arrays returned (identity), not recomputed.
    assert a[0] is b[0]
    assert a[1] is b[1]


def test_cached_values_still_correct():
    m = ChargeBasisTransmon(DeviceParams.q1())
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
