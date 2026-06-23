import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.pulse.hardware import Chain, TransferFunction, ControlNoise
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import GrapeOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _init_pulse(model):
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    return gaussian_drag(40, amp, 8.0, 0.0, model.anharmonicity_ghz())


def test_grape_reduces_cost_no_hardware():
    model = ChargeBasisTransmon(DeviceParams.q1())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    init = _init_pulse(model)
    c0 = prob.cost_from_pulse(init)
    res = GrapeOptimizer().run(prob, init_pulse=init, maxiter=60)
    assert res.best_cost < c0


def test_grape_predistorts_through_lowpass():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([TransferFunction.single_pole_lowpass(20.0, 1.0)])
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                   hardware=chain, leakage_weight=20.0)
    init = _init_pulse(model)
    c0 = prob.cost_from_pulse(init)  # naive pulse through the distorting line
    res = GrapeOptimizer().run(prob, init_pulse=init, maxiter=80)
    # GRAPE pre-distorts to substantially beat the naive pulse.
    assert res.best_cost < c0


def test_grape_numerical_fallback_runs_through_seeded_noise():
    # ControlNoise has no Jacobian, but with a fixed seed the cost is
    # deterministic, so GRAPE falls back to a finite-difference gradient and
    # still optimizes (rather than refusing to run).
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([ControlNoise(sigma=0.005, seed=0)])  # jacobian is None
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                   hardware=chain, leakage_weight=20.0)
    init = _init_pulse(model)
    c0 = prob.cost_from_pulse(init)
    res = GrapeOptimizer().run(prob, init_pulse=init, maxiter=40)
    assert res.best_cost < c0


def test_grape_raises_when_numerical_disabled():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([ControlNoise(sigma=0.01, seed=0)])  # jacobian is None
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), hardware=chain)
    init = _init_pulse(model)
    try:
        GrapeOptimizer().run(prob, init_pulse=init, maxiter=5, allow_numerical=False)
        raised = False
    except ValueError:
        raised = True
    assert raised
