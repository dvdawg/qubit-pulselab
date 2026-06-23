# Single-Transmon Pulse Optimization Lab — Design

**Date:** 2026-06-20
**Location:** `lab_pulse_opt/`
**Status:** Approved design, ready for implementation planning

## 1. Purpose

An interactive lab for **microwave-drive pulse optimization on a single flux-tunable
transmon**. It spans two modes in one tool:

- **Explore** — tweak pulse and device parameters live, and *see* the effect of each
  (state evolution, leakage, spectrum, distortion).
- **Optimize** — run pluggable optimizers that shape pulses against an accurate
  Hamiltonian *as seen through a realistic hardware chain*.

The intent behind the project is to **modify existing optimization methods and prototype
new ones**, especially methods that account for hardware modifications to pulses
(transient/ramp behavior, line distortion, control and readout noise). The architecture
is therefore built so the Hamiltonian, the hardware chain, and the optimizer are each
swappable behind small interfaces.

### Ground truth: the real device (Q1)

Seeded from the OPX project `opx/20260501_iMET_v1_2_SQUID_Q1/OPX_project/configuration.py`:

- Qubit frequency ≈ **5.252 GHz** (LO 5.5 GHz − IF 247.79 MHz)
- **Anharmonicity α = −64 MHz** (weak → leakage is a *real* problem for short gates)
- **T1 ≈ 15 µs**
- Single-qubit gates: **Gaussian-DRAG I/Q pulses**, `x180` = 40 ns, σ = len/5, amp 0.2,
  `drag_coef = 0` currently (i.e. uncorrected Gaussian — leaks)
- Existing calibration loop on the device: DRAG calibration (Yale + Google), AllXY, RB,
  cryoscope + FIR/IIR flux distortion correction

The lab reproduces this device in simulation so optimized pulses are directly comparable
to what runs on the OPX, and can be **exported in OPX `configuration.py` I/Q-array form**
(matching `drag_gaussian_pulse_waveforms` conventions).

## 2. Scope

### In scope (v1)

- Exact flux-tunable charge-basis transmon Hamiltonian **and** a lighter Duffing/Kerr
  model, swappable.
- Microwave I/Q drive in the rotating frame; single-qubit rotation targets (X180, X90,
  Y variants, arbitrary single-qubit unitary).
- Coherent (Schrödinger) **and** Lindblad (T1, Tφ) dynamics, toggleable.
- A composable **hardware distortion chain**: drive-line transfer function (finite
  bandwidth / rise-time / ringing), bias-tee long-tail droop, IQ mixer imbalance + LO
  leakage + control noise, and a readout-chain noise wrapper on the measured cost.
- Pluggable optimizers: analytic DRAG, GRAPE (gradient), CRAB (basis-expansion), behind a
  common `Optimizer`/`Problem` interface for prototyping new methods.
- Metrics: average gate fidelity (computational subspace), leakage to |2+⟩, process
  fidelity; AllXY/RB-style evaluators; optional noisy "measured" cost.
- Streamlit app for live exploration + running optimizers + exporting pulses.

### Out of scope (v1)

- Flux-pulse (frequency-tuning) gate optimization — flux appears only as a device
  operating-point parameter, not as a control knob. (Possible later add-on.)
- Two-qubit gates / coupling.
- Full readout-resonator dispersive simulation (we inject readout *noise* into the cost,
  we do not simulate the resonator + amplifier chain dynamics).
- Gradient-free optimizers (Nelder-Mead/CMA) — easy to add later via the same interface.

## 3. Core pipeline

```
optimizer params
  → Pulse envelope I(t)+iQ(t)      [DRAG | piecewise-constant | CRAB basis]
  → Hardware chain (composable)    [transfer fn → bias-tee droop → IQ imbalance + LO leak → control noise]
  → distorted envelope the qubit actually experiences
  → Dynamics                       [coherent Schrödinger | Lindblad master equation]
  → Transmon model                 [exact charge-basis | light Duffing — swappable]
  → Metrics                        [avg gate fidelity, leakage, (+ readout-noise measurement)]
  → scalar cost
  → Optimizer                      [DRAG | GRAPE | CRAB | user-defined]  → updates params
```

Each stage is an independent, individually testable unit with a defined interface. The
linear hardware stages additionally expose a Jacobian so gradients can be backpropagated
through them.

## 4. Module architecture

```
lab_pulse_opt/
  pyproject.toml            # package: pulselab; deps numpy, scipy, streamlit, matplotlib/plotly, pytest
  pulselab/
    device/
      params.py             # DeviceParams dataclass; from_opx_config() seeded to Q1; E_C/E_J/d/n_g/flux/levels/T1/Tphi
      hamiltonian.py        # TransmonModel interface:
                            #   ChargeBasisTransmon  — exact: diagonalize 4E_C(n-n_g)^2 - E_J(Phi)cos(phi),
                            #                          SQUID flux tuning + junction asymmetry; true levels,
                            #                          real charge matrix elements, charge dispersion
                            #   DuffingTransmon      — light Kerr oscillator (pluggable for speed)
                            # Both expose: H0 (eigenbasis), drive operator(s), n_levels, subspace projector
    pulse/
      envelope.py           # Pulse: time grid + complex envelope; parametrizations:
                            #   GaussianDrag, PiecewiseConstant (GRAPE), BasisExpansion (CRAB)
      hardware.py           # HardwareStage interface + composable Chain:
                            #   TransferFunction (FIR/IIR), BiasTeeDroop (high-pass),
                            #   IQImbalance (gain/phase + LO leakage), ControlNoise (jitter + 1/f)
                            #   linear stages expose jacobian() for gradient backprop
    dynamics/
      propagator.py         # coherent piecewise-constant evolution -> U and/or state trajectory
      lindblad.py           # master equation; collapse operators from T1, Tphi
      simulate.py           # simulate(model, pulse, options) -> Result (U, states, observables)
    metrics/
      fidelity.py           # avg gate fidelity (subspace), leakage to |2+>, process fidelity
      measurement.py        # readout-noise wrapper -> noisy measured cost; AllXY / RB-style evaluators
    optimize/
      base.py               # Problem(model, hardware, target, metric) -> cost(params); Optimizer interface
      drag.py               # analytic DRAG baseline
      grape.py              # gradient-based, analytic grads through linear hardware stages
      crab.py               # basis-expansion / bandwidth-limited
    export/
      opx.py                # export optimized pulse to OPX configuration.py I/Q arrays
  app/
    streamlit_app.py        # UI entry point
    panels/                 # device panel, pulse panel, hardware panel, optimizer panel, plots
  examples/                 # headless scripts mirroring app workflows
  tests/                    # TDD; physics anchors + lab-number checks
  docs/
```

### Defaults (configurable)

- 5 transmon levels
- 1 GS/s AWG grid (1 ns step)
- Rotating frame at the drive frequency
- Complex envelope representation throughout

## 5. Key interfaces

**TransmonModel**
```
H0() -> ndarray                      # static Hamiltonian in eigenbasis
drive_operators() -> (n_op, ...)     # charge/drive operators in rotating frame
n_levels() -> int
subspace_projector() -> ndarray      # computational {|0>,|1>} subspace
```

**HardwareStage**
```
apply(pulse) -> pulse                # distorts the envelope
jacobian(pulse) -> linear-op | None  # for linear stages, enables gradient backprop; None if nonlinear
```

**Problem / Optimizer**
```
Problem(model, hardware_chain, target_unitary, metric).cost(params) -> float
Optimizer.run(problem, initial_params, options) -> OptResult (best params, history, diagnostics)
```

The `Problem.cost` surface is the single extension point for prototyping a new method:
a new optimizer is a class implementing `Optimizer.run` against this cost.

## 6. Optimizer & gradient strategy

- **DRAG** — closed-form DRAG-corrected Gaussian. The interpretable baseline to beat.
- **GRAPE** — piecewise-constant, analytic gradients via the standard auxiliary-matrix
  method, **backpropagated through the linear hardware stages** (transfer function,
  bias-tee) so the optimizer learns pre-distortion. Through nonlinear stages (IQ
  imbalance) or Lindblad evolution, fall back to finite-difference.
- **CRAB** — optimize coefficients of a smooth basis (Fourier/Chebyshev); naturally
  bandwidth-limited, friendly to the transfer-function constraint, and robust when
  gradients are unavailable (noisy/measured cost).
- **Targets:** X180, X90, ±Y90/Y180, arbitrary single-qubit unitary.
- **Cost:** `1 − avg_gate_fidelity + leakage_penalty`, with an optional **robust/noisy**
  evaluation: readout-noise measurement and/or ensemble averaging over hardware-parameter
  uncertainty.

## 7. UI design — "tweak a parameter, see its effect"

The guiding principle: **every actually-configurable parameter is exposed as a control
with an inline explanation and an immediately visible, isolatable effect.** The app is
organized so the user builds intuition for *each* parameter independently.

- **Grouped parameter panels** (device / pulse / hardware / dynamics / optimizer). Each
  control has a short tooltip stating what it physically is and what it changes.
- **Effect isolation:** every hardware stage has an on/off toggle and shows
  *ideal vs. distorted* envelope overlaid, so the contribution of each stage is visible
  in isolation. Same for decoherence (coherent vs. Lindblad toggle).
- **Live plots** that update on parameter change:
  - pulse envelope I/Q — ideal vs. distorted (post-hardware-chain)
  - state evolution — Bloch vector and/or level populations vs. time
  - **leakage** to |2+⟩ vs. time (the headline error)
  - drive spectrum (shows bandwidth / DRAG sideband structure)
  - optimizer convergence (cost + leakage + fidelity vs. iteration), live during a run
  - AllXY staircase for the current pulse
- **Compare mode:** snapshot a pulse (e.g. bare Gaussian) and overlay against another
  (e.g. optimized) on the same axes.
- **Export:** one click to OPX `configuration.py` I/Q-array form.

Performance note: live exploration defaults to the light Duffing model + coherent
dynamics for responsiveness; the exact charge-basis model + Lindblad are used for
evaluation/optimization and can be toggled on for exploration when desired.

**UI framework:** Streamlit.

## 8. Validation strategy

Test-driven throughout, with physics anchors:

- Charge-basis model reproduces **α = −64 MHz** from the device's E_J/E_C.
- Bare Gaussian (`drag_coef = 0`) **leaks**; DRAG suppresses leakage — reproduce the
  Yale/Google DRAG result the lab's scripts implement.
- AllXY produces the canonical staircase for an ideal pulse; known distortions produce the
  known AllXY error signatures.
- Linear hardware stages: unit response / impulse-response checks; bias-tee high-pass
  produces droop with the expected time constant.
- An optimized 40 ns pulse **beats the current uncorrected Gaussian** in fidelity/leakage.
- GRAPE analytic gradients agree with finite-difference on a small problem.

## 9. Build phasing (physics-core first)

1. **Physics core** — `device/` (params + both Hamiltonians) + `dynamics/` (coherent) +
   `metrics/fidelity.py`. Fully headless + tested. Reproduce α and basic Rabi/DRAG physics.
2. **Hardware chain + open system** — `pulse/hardware.py` (all four stages) +
   `dynamics/lindblad.py`.
3. **Optimizers** — `optimize/` DRAG → GRAPE → CRAB against the `Problem` interface;
   `export/opx.py`.
4. **Streamlit app** — wire everything to grouped parameter panels + live plots +
   compare mode + export, per Section 7.
5. **Readout noise + robust/measured cost** — `metrics/measurement.py` and robust
   evaluation in `optimize/`.

## 10. Open items / future work

- Flux-pulse control as a second control axis.
- Gradient-free optimizers (Nelder-Mead/CMA) via the same `Optimizer` interface.
- Full readout-resonator + amplifier-chain dynamical simulation (beyond injected noise).
- Robustness optimization over correlated 1/f noise ensembles.
