import numpy as np
from pulselab.dynamics.lindblad import collapse_operators


def test_relaxation_and_dephasing_present_and_shaped():
    ops = collapse_operators(n_levels=3, T1_us=15.0, Tphi_us=30.0)
    assert len(ops) == 2
    for c in ops:
        assert c.shape == (3, 3)


def test_relaxation_rate_normalization():
    (c_relax,) = collapse_operators(3, T1_us=15.0, Tphi_us=30.0,
                                    include_dephasing=False)
    g1 = 1.0 / (15.0 * 1000)
    # |0><1| element is sqrt(gamma1)*sqrt(1).
    assert np.isclose(c_relax[0, 1], np.sqrt(g1))
    # |1><2| element is sqrt(gamma1)*sqrt(2).
    assert np.isclose(c_relax[1, 2], np.sqrt(g1) * np.sqrt(2))


def test_dephasing_is_number_operator():
    (c_phi,) = collapse_operators(4, T1_us=15.0, Tphi_us=30.0,
                                  include_relaxation=False)
    gphi = 1.0 / (30.0 * 1000)
    assert np.allclose(np.diag(c_phi), np.sqrt(2 * gphi) * np.arange(4))
