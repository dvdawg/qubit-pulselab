import numpy as np
from pulselab.pulse.envelope import gaussian_drag
from pulselab.export.opx import to_opx_waveforms


def test_export_shapes_and_scaling():
    p = gaussian_drag(40, amp_radns=0.16, sigma_ns=8.0, drag_coef=0.5,
                      anharmonicity_ghz=-0.064)
    scale = 1.25  # DAC units per rad/ns
    wf = to_opx_waveforms(p, dac_per_radns=scale)
    assert wf["I_wf"].shape == p.t.shape
    assert wf["Q_wf"].shape == p.t.shape
    assert np.allclose(wf["I_wf"], p.I * scale)
    assert np.allclose(wf["Q_wf"], p.Q * scale)


def test_bare_gaussian_has_zero_q_waveform():
    p = gaussian_drag(40, 0.16, 8.0, 0.0, -0.064)
    wf = to_opx_waveforms(p, dac_per_radns=1.0)
    assert np.allclose(wf["Q_wf"], 0.0)
    assert wf["I_wf"].max() > 0
