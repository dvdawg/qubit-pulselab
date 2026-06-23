import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import leakage


def _leakage_for(drag_coef, model, amp):
    p = gaussian_drag(40, amp_radns=amp, sigma_ns=8.0, drag_coef=drag_coef,
                      anharmonicity_ghz=model.anharmonicity_ghz())
    U = propagate(model, p, drive_freq_ghz=model.f01_ghz())
    return leakage(U)


def test_drag_reduces_leakage_vs_bare_gaussian():
    model = ChargeBasisTransmon(DeviceParams.q1())
    # Calibrate amp so the bare Gaussian is ~a pi-pulse (area ~ pi).
    # area = amp * integral(gaussian); solve amp by matching area to pi.
    from pulselab.pulse.envelope import gaussian_drag as gd
    probe = gd(40, amp_radns=1.0, sigma_ns=8.0, drag_coef=0.0,
               anharmonicity_ghz=model.anharmonicity_ghz())
    amp = np.pi / probe.area()

    bare = _leakage_for(0.0, model, amp)
    drag = _leakage_for(0.5, model, amp)
    assert drag < bare


def test_bare_gaussian_leakage_in_physical_range():
    # A smooth 40ns Gaussian pi-pulse on the weakly-anharmonic Q1 transmon
    # leaks a small but nonzero amount to |2>. This pins the multi-level
    # coupling magnitude: ~1.6e-3 for the committed model.
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, amp_radns=1.0, sigma_ns=8.0, drag_coef=0.0,
                          anharmonicity_ghz=model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    bare = _leakage_for(0.0, model, amp)
    assert 5e-4 < bare < 5e-3


def test_drag_cuts_leakage_at_least_twofold():
    # First-order DRAG (coef=1.0) should cut |2> leakage by at least ~2x
    # relative to the bare Gaussian, verifying the |1>-|2> coupling
    # participates correctly in the multi-level dynamics.
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, amp_radns=1.0, sigma_ns=8.0, drag_coef=0.0,
                          anharmonicity_ghz=model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    bare = _leakage_for(0.0, model, amp)
    strong = _leakage_for(1.0, model, amp)
    assert strong < 0.5 * bare
