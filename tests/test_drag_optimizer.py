import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_drag_optimizer_beats_bare_gaussian():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp0 = np.pi / probe.area()
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    bare_cost = prob.cost_from_pulse(
        gaussian_drag(40, amp0, 8.0, 0.0, model.anharmonicity_ghz()))
    opt = DragOptimizer(40, 8.0, model.anharmonicity_ghz())
    res = opt.run(prob, init_amp=amp0, init_drag_coef=0.0)
    assert res.best_cost < bare_cost
    assert len(res.history) > 0
