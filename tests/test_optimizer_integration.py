import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import GrapeOptimizer
from pulselab.metrics.fidelity import avg_gate_fidelity

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_grape_predistortion_beats_naive_through_line():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([TransferFunction.single_pole_lowpass(15.0, 1.0)])
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                   hardware=chain, leakage_weight=20.0)
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    naive = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())

    f_naive = avg_gate_fidelity(prob.propagated(naive), X)
    res = GrapeOptimizer().run(prob, init_pulse=naive, maxiter=120)
    f_grape = avg_gate_fidelity(prob.propagated(res.best_pulse), X)
    assert f_grape > f_naive
