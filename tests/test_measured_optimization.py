import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.metrics.measurement import MeasuredProblem

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_drag_reduces_measured_cost():
    model = ChargeBasisTransmon(DeviceParams.q1())
    anh = model.anharmonicity_ghz()
    amp0 = np.pi / gaussian_drag(40, 1.0, 8.0, 0.0, anh).area()
    base = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    mp = MeasuredProblem(base, n_shots=50000, readout_fidelity=1.0, seed=7)

    bare = gaussian_drag(40, amp0, 8.0, 0.0, anh)
    bare_cost = mp.cost_from_pulse(bare)
    res = DragOptimizer(40, 8.0, anh).run(mp, init_amp=amp0, init_drag_coef=0.0)
    assert res.best_cost < bare_cost
