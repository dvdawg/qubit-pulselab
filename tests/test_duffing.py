import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import DuffingTransmon, ChargeBasisTransmon, TransmonModel


def test_duffing_is_model_and_couplings():
    m = DuffingTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=4))
    assert isinstance(m, TransmonModel)
    assert np.allclose(m.drive_couplings(), np.sqrt([1.0, 2.0, 3.0]))


def test_duffing_close_to_charge_basis_levels():
    p = DeviceParams.from_spectrum(5.252, -0.064)
    duff = DuffingTransmon(p)
    exact = ChargeBasisTransmon(p)
    # f01 and anharmonicity agree to a few MHz in the deep-transmon regime.
    assert np.isclose(duff.f01_ghz(), exact.f01_ghz(), atol=5e-3)
    assert np.isclose(duff.anharmonicity_ghz(), exact.anharmonicity_ghz(), atol=5e-3)
