import numpy as np


def to_opx_waveforms(pulse, dac_per_radns):
    """Convert a Pulse (I/Q in rad/ns) to OPX DAC-unit waveform arrays.

    dac_per_radns is the power-Rabi calibration: DAC amplitude per rad/ns of
    Rabi rate. The returned arrays drop into configuration.py as the `samples`
    of an arbitrary waveform.
    """
    return {
        "I_wf": np.asarray(pulse.I, dtype=float) * dac_per_radns,
        "Q_wf": np.asarray(pulse.Q, dtype=float) * dac_per_radns,
    }
