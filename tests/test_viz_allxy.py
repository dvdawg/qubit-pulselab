import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.viz import allxy_populations

X = np.array([[0, 1], [1, 0]], dtype=complex)
EXPECTED = np.array([0.0] * 5 + [0.5] * 12 + [1.0] * 4)


def test_canonical_staircase():
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    labels, p1 = allxy_populations(model, model.f01_ghz())
    assert len(labels) == 21 and len(p1) == 21
    assert np.max(np.abs(np.array(p1) - EXPECTED)) < 0.02
    assert labels[0] == "I-I" and labels[-1] == "Y90-Y90"


def test_optimized_drag_params_clean_up_multilevel_staircase():
    # On a multi-level transmon, an uncalibrated gate (drag_coef=0) leaves
    # leakage-induced deviations in the AllXY staircase; the optimized DRAG
    # calibration (amp + drag_coef) reduces them.
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=4))
    fd, anh = model.f01_ghz(), model.anharmonicity_ghz()
    amp0 = np.pi / gaussian_drag(40, 1.0, 8.0, 0.0, anh).area()
    res = DragOptimizer(40, 8.0, anh).run(
        Problem(model, target=X, drive_freq_ghz=fd, leakage_weight=20.0),
        init_amp=amp0, init_drag_coef=0.0)
    assert "amp" in res.params and "drag_coef" in res.params

    _, bare = allxy_populations(model, fd, amp=amp0, drag_coef=0.0)
    _, opt = allxy_populations(model, fd, amp=res.params["amp"],
                               drag_coef=res.params["drag_coef"])
    dev_bare = np.max(np.abs(np.array(bare) - EXPECTED))
    dev_opt = np.max(np.abs(np.array(opt) - EXPECTED))
    assert dev_opt < dev_bare
