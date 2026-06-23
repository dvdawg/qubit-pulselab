# Qubit Pulse Lab

Run the lab:

    cd lab_pulse_opt
    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

The sidebar exposes every configurable parameter (device, pulse, hardware line,
optimizer), each with an inline explanation. The main panel shows, live:

- the drive envelope, ideal vs. distorted by the hardware line
- state populations vs. time (leakage to |2> and above)
- the drive spectrum
- the AllXY staircase
- X-gate fidelity and leakage metrics

Pick an optimizer (DRAG / CRAB / GRAPE) and click **Run optimizer** to shape the
pulse against the current device + hardware, watch it converge, and download the
optimized I/Q waveform for the OPX.

All computation lives in tested pure functions in `pulselab/viz.py` and the
`pulselab` package; the Streamlit file is only the UI shell.

## Measured readout & robust pulses

Toggle **Simulate measured readout** in the sidebar to see the X-gate excited
population as a real experiment would measure it — sampled over a finite number
of shots with a readout assignment-fidelity error, instead of the exact value.

For pulses that tolerate parameter spread (e.g. qubit-frequency detuning), build
an ensemble of `Problem`s with `pulselab.optimize.robust.detuning_ensemble`, wrap
it in an `EnsembleProblem`, and optimize it with the DRAG or CRAB optimizer (the
derivative-free optimizers — GRAPE needs a differentiable cost). See
`examples/robust_vs_nominal.py`.
