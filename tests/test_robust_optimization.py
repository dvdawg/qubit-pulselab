import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.robust import EnsembleProblem, detuning_ensemble, robust_cost

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_robust_optimization_lowers_ensemble_cost():
    model = ChargeBasisTransmon(DeviceParams.q1())
    anh = model.anharmonicity_ghz()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, anh)
    amp0 = np.pi / probe.area()
    base = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    ens = detuning_ensemble(base, [-0.004, -0.002, 0.0, 0.002, 0.004])

    nominal = gaussian_drag(40, amp0, 8.0, 0.5, anh)
    ep = EnsembleProblem(ens)
    res = DragOptimizer(40, 8.0, anh).run(ep, init_amp=amp0, init_drag_coef=0.5)

    assert robust_cost(ens, res.best_pulse) <= robust_cost(ens, nominal)
