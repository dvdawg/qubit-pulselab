import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.crab import CrabOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_crab_improves_on_base_pulse():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp0 = np.pi / probe.area()
    base = gaussian_drag(40, amp0, 8.0, 0.0, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    base_cost = prob.cost_from_pulse(base)
    opt = CrabOptimizer(base, n_harmonics=2)
    res = opt.run(prob)
    assert res.best_cost <= base_cost
    assert len(res.history) > 0
