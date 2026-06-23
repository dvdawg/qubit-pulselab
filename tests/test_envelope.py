import numpy as np
from pulselab.pulse.envelope import Pulse, gaussian_drag
from pulselab.units import ghz_to_radns


def test_gaussian_no_drag_has_zero_Q():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=0.0, anharmonicity_ghz=-0.064)
    assert np.allclose(p.Q, 0.0)
    assert p.I.max() > 0
    assert np.isclose(p.dt, 1.0)


def test_drag_q_is_scaled_negative_derivative():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=1.0, anharmonicity_ghz=-0.064)
    anharm = ghz_to_radns(-0.064)
    expected_Q = -np.gradient(p.I, p.t) / anharm
    assert np.allclose(p.Q, expected_Q, atol=1e-6)


def test_area_matches_trapz():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=0.0, anharmonicity_ghz=-0.064)
    assert np.isclose(p.area(), np.trapz(p.I, p.t))
