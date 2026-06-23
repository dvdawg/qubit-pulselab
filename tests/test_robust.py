import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.robust import robust_cost, EnsembleProblem, detuning_ensemble

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _base():
    model = ChargeBasisTransmon(DeviceParams.q1())
    return Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)


def test_robust_cost_is_mean_over_ensemble():
    base = _base()
    pulse = gaussian_drag(40, 0.16, 8.0, 0.5, base.model.anharmonicity_ghz())
    ens = detuning_ensemble(base, [-0.003, 0.0, 0.003])
    expected = np.mean([p.cost_from_pulse(pulse) for p in ens])
    assert np.isclose(robust_cost(ens, pulse), expected)


def test_detuning_ensemble_shifts_drive_freq():
    base = _base()
    ens = detuning_ensemble(base, [-0.003, 0.0, 0.003])
    assert len(ens) == 3
    assert np.isclose(ens[0].drive_freq_ghz, base.drive_freq_ghz - 0.003)
    assert np.isclose(ens[2].drive_freq_ghz, base.drive_freq_ghz + 0.003)
    assert ens[1].leakage_weight == base.leakage_weight


def test_ensemble_problem_duck_types_cost():
    base = _base()
    pulse = gaussian_drag(40, 0.16, 8.0, 0.5, base.model.anharmonicity_ghz())
    ens = detuning_ensemble(base, [-0.003, 0.0, 0.003])
    ep = EnsembleProblem(ens)
    assert np.isclose(ep.cost_from_pulse(pulse), robust_cost(ens, pulse))
