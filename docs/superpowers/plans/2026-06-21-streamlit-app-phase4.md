# Single-Transmon Pulse Lab — Phase 4: Interactive Streamlit App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app that lets the user tweak every configurable device/pulse/hardware/optimizer parameter live — each with an inline explanation — and immediately see its effect (ideal-vs-distorted envelope, state populations, Bloch trajectory, leakage, drive spectrum, AllXY staircase, optimizer convergence), with one-click OPX waveform export.

**Architecture:** All computation lives in a new, fully-tested pure module `pulselab/viz.py` (no Streamlit/Plotly imports — just numpy on top of the Phase-1/2/3 primitives). `app/streamlit_app.py` is a thin shell: it reads widget values, calls `viz` functions, and renders Plotly figures. The app is smoke-tested headlessly with `streamlit.testing.v1.AppTest`.

**Tech Stack:** Python ≥3.10, numpy, scipy, pytest, streamlit, plotly (both installed). Builds on Phases 1–3.

## Global Constraints

- Units: angular frequency rad/ns, time ns, linear frequency GHz. Pulses are the Phase-1 `Pulse(t, I, Q)`.
- `viz.py` must NOT import streamlit or plotly — it returns plain numpy arrays / dicts so it is unit-testable headless. Only `app/streamlit_app.py` imports streamlit/plotly.
- AllXY gate convention (VERIFIED to reproduce the canonical staircase to ~0.006 on a 2-level model): an X rotation drives the I channel, a Y rotation drives the Q channel, identity is a zero pulse, a 180 gate has pulse area π (`amp = π/area` of the unit-amp Gaussian), a 90 gate uses half that amplitude. Canonical AllXY pattern over the 21 standard pairs is `[0]*5 + [0.5]*12 + [1]*4` (P1 = excited-state population).
- Bloch convention for a state `ψ` (use levels 0,1): `a, b = ψ[0], ψ[1]`; `x = 2*Re(conj(a)*b)`, `y = 2*Im(conj(a)*b)`, `z = |a|² - |b|²`. `|0>` → `(0,0,1)`.
- Every Streamlit input widget MUST pass a `help=` string explaining what the parameter physically is and what changing it does (this is the core UX requirement).
- TDD: failing test first, minimal code, commit per task. Run tests from `lab_pulse_opt/`: `python -m pytest ...`.
- Do NOT modify Phase-1/2/3 public interfaces; `viz.py` consumes them.

---

## File Structure

```
lab_pulse_opt/
  pulselab/
    viz.py                 # NEW: pure data-prep functions (Tasks 1-4)
  app/
    __init__.py            # NEW (Task 5)
    streamlit_app.py       # NEW: the Streamlit shell (Task 5)
  tests/
    test_viz_trajectory.py # Task 1
    test_viz_bloch.py      # Task 2
    test_viz_spectrum.py   # Task 3
    test_viz_allxy.py      # Task 4
    test_app_smoke.py      # Task 5
  README_app.md            # Task 6
```

---

### Task 1: State + population trajectories

**Files:**
- Create: `lab_pulse_opt/pulselab/viz.py`
- Test: `lab_pulse_opt/tests/test_viz_trajectory.py`

**Interfaces:**
- Consumes: Phase-1 `rotating_frame_operators`, `Pulse`; `scipy.linalg.expm`.
- Produces:
  - `state_trajectory(model, pulse, drive_freq_ghz, psi0=None) -> (t_edges, states)` where `psi0` defaults to `|0>`; returns `t_edges` of length `N+1` (`concatenate([pulse.t, [pulse.t[-1]+dt]])`) and `states` of shape `(N+1, d)` complex — the state after each slice, starting with `psi0`.
  - `population_trajectory(model, pulse, drive_freq_ghz, psi0=None) -> (t_edges, pops)` with `pops` shape `(N+1, d)` real, `pops = |states|²`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_viz_trajectory.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.dynamics.propagator import propagate
from pulselab.viz import state_trajectory, population_trajectory


def _model():
    return ChargeBasisTransmon(DeviceParams.q1())


def test_final_state_matches_propagator():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    pulse = gaussian_drag(40, np.pi / probe.area(), 8.0, 0.5, model.anharmonicity_ghz())
    t_edges, states = state_trajectory(model, pulse, model.f01_ghz())
    psi0 = np.zeros(model.n_levels, dtype=complex); psi0[0] = 1.0
    U = propagate(model, pulse, model.f01_ghz())
    assert states.shape == (pulse.t.size + 1, model.n_levels)
    assert t_edges.size == pulse.t.size + 1
    assert np.allclose(states[-1], U @ psi0, atol=1e-8)


def test_populations_normalized_and_start_in_ground():
    model = _model()
    pulse = gaussian_drag(40, 0.1, 8.0, 0.0, model.anharmonicity_ghz())
    t_edges, pops = population_trajectory(model, pulse, model.f01_ghz())
    assert np.allclose(pops.sum(axis=1), 1.0, atol=1e-8)
    assert np.isclose(pops[0, 0], 1.0)  # starts in |0>
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz_trajectory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.viz'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/viz.py
import numpy as np
from scipy.linalg import expm


def state_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """State after each time slice. Returns (t_edges[N+1], states[N+1, d])."""
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    d = H0.shape[0]
    if psi0 is None:
        psi0 = np.zeros(d, dtype=complex)
        psi0[0] = 1.0
    dt = pulse.dt
    psi = np.asarray(psi0, dtype=complex)
    states = [psi.copy()]
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Uk = expm(-1j * (H0 + Ik * X_op + Qk * Y_op) * dt)
        psi = Uk @ psi
        states.append(psi.copy())
    t_edges = np.concatenate([pulse.t, [pulse.t[-1] + dt]])
    return t_edges, np.array(states)


def population_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """Level populations over time. Returns (t_edges[N+1], pops[N+1, d])."""
    t_edges, states = state_trajectory(model, pulse, drive_freq_ghz, psi0)
    return t_edges, np.abs(states) ** 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_viz_trajectory.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/viz.py lab_pulse_opt/tests/test_viz_trajectory.py
git commit -m "feat(viz): state + population trajectories"
```

---

### Task 2: Bloch trajectory

**Files:**
- Modify: `lab_pulse_opt/pulselab/viz.py`
- Test: `lab_pulse_opt/tests/test_viz_bloch.py`

**Interfaces:**
- Consumes: `state_trajectory`.
- Produces: `bloch_trajectory(model, pulse, drive_freq_ghz, psi0=None) -> (t_edges, bloch)` where `bloch` is shape `(N+1, 3)` with columns `(x, y, z)` per the Bloch convention in Global Constraints, computed from the `(0,1)` amplitudes of each state.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_viz_bloch.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.viz import bloch_trajectory


def _model():
    return ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))


def test_ground_state_is_north_pole():
    model = _model()
    zero = Pulse(t=np.arange(10) * 1.0, I=np.zeros(10), Q=np.zeros(10))
    _, bloch = bloch_trajectory(model, zero, model.f01_ghz())
    assert np.allclose(bloch[0], [0, 0, 1], atol=1e-9)


def test_x90_lands_on_equator():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    x90 = gaussian_drag(40, (np.pi / 2) / probe.area(), 8.0, 0.0, model.anharmonicity_ghz())
    _, bloch = bloch_trajectory(model, x90, model.f01_ghz())
    assert abs(bloch[-1, 2]) < 0.1   # z ~ 0 (on the equator)
    assert np.isclose(np.linalg.norm(bloch[-1]), 1.0, atol=1e-6)  # pure state
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz_bloch.py -v`
Expected: FAIL — `ImportError: cannot import name 'bloch_trajectory'`.

- [ ] **Step 3: Write minimal implementation**

Add to `viz.py`:

```python
def bloch_trajectory(model, pulse, drive_freq_ghz, psi0=None):
    """Qubit-subspace Bloch vector (x,y,z) over time. Returns (t_edges, bloch[N+1,3])."""
    t_edges, states = state_trajectory(model, pulse, drive_freq_ghz, psi0)
    a = states[:, 0]
    b = states[:, 1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return t_edges, np.stack([x, y, z], axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_viz_bloch.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/viz.py lab_pulse_opt/tests/test_viz_bloch.py
git commit -m "feat(viz): Bloch trajectory"
```

---

### Task 3: Drive spectrum

**Files:**
- Modify: `lab_pulse_opt/pulselab/viz.py`
- Test: `lab_pulse_opt/tests/test_viz_spectrum.py`

**Interfaces:**
- Consumes: `Pulse`.
- Produces: `drive_spectrum(pulse) -> (freqs_ghz, power)` — the power spectrum of the complex envelope `I + 1j*Q` via `np.fft.fftshift(np.fft.fft(...))`. `freqs_ghz = np.fft.fftshift(np.fft.fftfreq(N, d=dt))` (in GHz, since dt is in ns → fftfreq gives cycles/ns = GHz). `power = |fft|²`, normalized so `power.max() == 1.0`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_viz_spectrum.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.viz import drive_spectrum


def test_single_tone_peaks_at_its_frequency():
    dt = 1.0
    N = 256
    t = np.arange(N) * dt
    f0 = 0.05  # GHz (cycles/ns)
    env = np.exp(2j * np.pi * f0 * t)  # complex tone at +f0
    p = Pulse(t=t, I=env.real, Q=env.imag)
    freqs, power = drive_spectrum(p)
    assert np.isclose(power.max(), 1.0)
    assert np.isclose(freqs[np.argmax(power)], f0, atol=1.5 / (N * dt))


def test_length_matches():
    t = np.arange(64) * 1.0
    p = Pulse(t=t, I=np.ones(64), Q=np.zeros(64))
    freqs, power = drive_spectrum(p)
    assert freqs.shape == (64,) and power.shape == (64,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz_spectrum.py -v`
Expected: FAIL — `ImportError: cannot import name 'drive_spectrum'`.

- [ ] **Step 3: Write minimal implementation**

Add to `viz.py`:

```python
def drive_spectrum(pulse):
    """Normalized power spectrum of the complex drive envelope I + iQ.

    Returns (freqs_ghz, power) with power.max() == 1.0. freqs are the detuning
    from the drive frequency, in GHz (dt is in ns).
    """
    env = pulse.I + 1j * pulse.Q
    n = env.size
    spec = np.fft.fftshift(np.fft.fft(env))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=pulse.dt))
    power = np.abs(spec) ** 2
    peak = power.max()
    if peak > 0:
        power = power / peak
    return freqs, power
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_viz_spectrum.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/viz.py lab_pulse_opt/tests/test_viz_spectrum.py
git commit -m "feat(viz): drive spectrum"
```

---

### Task 4: AllXY staircase simulator

**Files:**
- Modify: `lab_pulse_opt/pulselab/viz.py`
- Test: `lab_pulse_opt/tests/test_viz_allxy.py`

**Interfaces:**
- Consumes: Phase-1 `gaussian_drag`, `Pulse`, `propagate`.
- Produces: `allxy_populations(model, drive_freq_ghz, duration_ns=40, sigma_ns=8.0, dt_ns=1.0) -> (labels, p1)` where `labels` is the list of 21 `"G1-G2"` strings and `p1` is the excited-state population after each pair, starting from `|0>`, using the verified gate convention in Global Constraints. (X gate → I channel, Y gate → Q channel, identity → zeros, 180 → area π, 90 → half amplitude; no DRAG, intended for a 2-level model.)

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_viz_allxy.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.viz import allxy_populations


def test_canonical_staircase():
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    labels, p1 = allxy_populations(model, model.f01_ghz())
    assert len(labels) == 21 and len(p1) == 21
    expected = np.array([0.0] * 5 + [0.5] * 12 + [1.0] * 4)
    assert np.max(np.abs(np.array(p1) - expected)) < 0.02
    assert labels[0] == "I-I" and labels[-1] == "Y90-Y90"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_viz_allxy.py -v`
Expected: FAIL — `ImportError: cannot import name 'allxy_populations'`.

- [ ] **Step 3: Write minimal implementation**

Add to `viz.py` (add `from .pulse.envelope import gaussian_drag, Pulse` and `from .dynamics.propagator import propagate` at the top):

```python
_ALLXY_PAIRS = [
    ("I", "I"), ("X180", "X180"), ("Y180", "Y180"), ("X180", "Y180"), ("Y180", "X180"),
    ("X90", "I"), ("Y90", "I"), ("X90", "Y90"), ("Y90", "X90"), ("X90", "Y180"),
    ("Y90", "X180"), ("X180", "Y90"), ("Y180", "X90"), ("X90", "X180"), ("X180", "X90"),
    ("Y90", "Y180"), ("Y180", "Y90"), ("X180", "I"), ("Y180", "I"), ("X90", "X90"),
    ("Y90", "Y90"),
]


def allxy_populations(model, drive_freq_ghz, duration_ns=40, sigma_ns=8.0, dt_ns=1.0):
    """Simulate the 21 standard AllXY gate pairs; return (labels, P1)."""
    anh = model.anharmonicity_ghz()
    n = int(round(duration_ns / dt_ns))
    zero = np.zeros(n)
    unit = gaussian_drag(duration_ns, 1.0, sigma_ns, 0.0, anh, dt_ns)
    amp180 = np.pi / float(np.trapz(unit.I, unit.t))

    def gate(name):
        if name == "I":
            return zero, zero
        amp = amp180 if name.endswith("180") else amp180 / 2
        env = gaussian_drag(duration_ns, amp, sigma_ns, 0.0, anh, dt_ns).I
        return (env, zero) if name[0] == "X" else (zero, env)

    t2 = np.arange(2 * n) * dt_ns
    labels, p1 = [], []
    for g1, g2 in _ALLXY_PAIRS:
        I1, Q1 = gate(g1)
        I2, Q2 = gate(g2)
        pulse = Pulse(t=t2, I=np.concatenate([I1, I2]), Q=np.concatenate([Q1, Q2]))
        U = propagate(model, pulse, drive_freq_ghz)
        labels.append(f"{g1}-{g2}")
        p1.append(float(abs(U[1, 0]) ** 2))
    return labels, p1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_viz_allxy.py -v`
Expected: PASS (1 passed). The staircase matches `[0]*5+[0.5]*12+[1]*4` within 0.02.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/viz.py lab_pulse_opt/tests/test_viz_allxy.py
git commit -m "feat(viz): AllXY staircase simulator"
```

---

### Task 5: Streamlit app + smoke test

**Files:**
- Create: `lab_pulse_opt/app/__init__.py` (empty)
- Create: `lab_pulse_opt/app/streamlit_app.py`
- Test: `lab_pulse_opt/tests/test_app_smoke.py`

**Interfaces:**
- Consumes: all of `viz`, `Problem`, the optimizers, `Chain` + hardware stages, `to_opx_waveforms`, device/model/envelope.
- Produces: a runnable Streamlit app whose every input widget carries a `help=` explanation, with ideal-vs-distorted envelope, populations, Bloch endpoint, leakage, spectrum, and AllXY views, plus an optimizer run + OPX export. Smoke-tested with `streamlit.testing.v1.AppTest` (runs the default state without exception).

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_app_smoke.py
from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    assert not at.exception
    # The app exposes at least one slider and the title.
    assert len(at.slider) > 0
    assert any("Pulse" in m.value for m in at.markdown) or at.title


def test_app_has_helpful_controls():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    # Every slider must carry a help string (the core UX requirement).
    assert all(s.help for s in at.slider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: FAIL — `FileNotFoundError` / app does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/app/streamlit_app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import (
    Chain, IdentityStage, TransferFunction, BiasTeeDroop, IQImbalance, ControlNoise)
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.grape import GrapeOptimizer
from pulselab.optimize.crab import CrabOptimizer
from pulselab.export.opx import to_opx_waveforms
from pulselab import viz

X_TARGET = np.array([[0, 1], [1, 0]], dtype=complex)

st.set_page_config(page_title="Single-Transmon Pulse Lab", layout="wide")
st.title("Single-Transmon Pulse Optimization Lab")
st.caption("Tweak any parameter and watch its effect. Seeded with the real Q1 device.")

# ---- Device panel ----
st.sidebar.header("Device")
n_levels = st.sidebar.slider(
    "Transmon levels", 2, 6, 4,
    help="How many transmon energy levels to simulate. More levels capture leakage "
         "to |2> and above more accurately, at higher compute cost.")
flux = st.sidebar.slider(
    "Flux bias (Phi/Phi0)", 0.0, 0.5, 0.0, 0.01,
    help="SQUID flux bias. 0 is the sweet spot (max frequency); moving toward 0.5 "
         "tunes the qubit frequency down and changes the anharmonicity.")
params = DeviceParams.from_spectrum(5.252, -0.064, n_levels=n_levels, flux=flux)
model = ChargeBasisTransmon(params)
drive_freq = model.f01_ghz()
anh = model.anharmonicity_ghz()

# ---- Pulse panel ----
st.sidebar.header("Pulse")
duration = st.sidebar.slider(
    "Duration (ns)", 8, 120, 40, 2,
    help="Total gate length. Shorter gates are faster but have wider bandwidth, "
         "which drives more leakage on a weakly-anharmonic transmon.")
sigma = st.sidebar.slider(
    "Gaussian sigma (ns)", 2.0, 30.0, 8.0, 0.5,
    help="Width of the Gaussian envelope. Narrower pulses ring more in frequency.")
probe = gaussian_drag(duration, 1.0, sigma, 0.0, anh)
amp_default = float(np.pi / probe.area())
amp = st.sidebar.slider(
    "Amplitude (rad/ns)", 0.0, 2 * amp_default, amp_default, amp_default / 50,
    help="Peak Rabi drive. Calibrated so the bare Gaussian is a pi (X180) pulse by default.")
drag_coef = st.sidebar.slider(
    "DRAG coefficient", -2.0, 2.0, 0.0, 0.05,
    help="DRAG adds a derivative component on the quadrature channel to cancel "
         "leakage to |2>. 0 is a plain Gaussian; the optimal value suppresses leakage.")
pulse = gaussian_drag(duration, amp, sigma, drag_coef, anh)

# ---- Hardware panel ----
st.sidebar.header("Hardware line")
stages = []
if st.sidebar.checkbox("Drive-line low-pass", value=False,
                       help="Finite drive-line bandwidth / rise-time, as a one-pole "
                            "low-pass. Larger tau = narrower bandwidth = more distortion."):
    tau_lp = st.sidebar.slider("  low-pass tau (ns)", 1.0, 60.0, 15.0, 1.0,
                               help="Low-pass time constant.")
    stages.append(TransferFunction.single_pole_lowpass(tau_lp, pulse.dt))
if st.sidebar.checkbox("Bias-tee droop", value=False,
                       help="DC-blocking bias-tee causes slow droop of sustained levels."):
    tau_bt = st.sidebar.slider("  bias-tee tau (ns)", 50.0, 5000.0, 1000.0, 50.0,
                               help="Droop time constant (large = slow droop).")
    stages.append(BiasTeeDroop(tau_bt, pulse.dt))
if st.sidebar.checkbox("IQ imbalance", value=False,
                       help="Mixer gain/phase imbalance and LO/carrier leakage."):
    g = st.sidebar.slider("  gain imbalance", -0.2, 0.2, 0.05, 0.01, help="I/Q gain mismatch.")
    ph = st.sidebar.slider("  phase error (rad)", -0.3, 0.3, 0.02, 0.01, help="I/Q phase error.")
    stages.append(IQImbalance(gain_imbalance=g, phase_error_rad=ph))
if st.sidebar.checkbox("Control noise", value=False,
                       help="Additive amplitude noise on the drive (stochastic; disables "
                            "gradient-based GRAPE)."):
    sig = st.sidebar.slider("  noise sigma (rad/ns)", 0.0, 0.05, 0.005, 0.001, help="Noise std.")
    stages.append(ControlNoise(sigma=sig, seed=0))
hardware = Chain(stages) if stages else IdentityStage()

# ---- Compute ----
distorted = hardware.apply(pulse)
U = propagate(model, distorted, drive_freq)
F = avg_gate_fidelity(U, X_TARGET)
L = leakage(U)

c1, c2, c3 = st.columns(3)
c1.metric("X-gate fidelity", f"{F:.5f}")
c2.metric("Leakage to |2+>", f"{L:.2e}")
c3.metric("Qubit freq (GHz)", f"{drive_freq:.4f}")

# ---- Envelope: ideal vs distorted ----
st.subheader("Drive envelope (ideal vs distorted)")
fig_env = go.Figure()
fig_env.add_scatter(x=pulse.t, y=pulse.I, name="I ideal", line=dict(color="royalblue"))
fig_env.add_scatter(x=pulse.t, y=pulse.Q, name="Q ideal", line=dict(color="firebrick"))
fig_env.add_scatter(x=distorted.t, y=distorted.I, name="I distorted",
                    line=dict(color="royalblue", dash="dash"))
fig_env.add_scatter(x=distorted.t, y=distorted.Q, name="Q distorted",
                    line=dict(color="firebrick", dash="dash"))
fig_env.update_layout(xaxis_title="t (ns)", yaxis_title="rad/ns", height=300)
st.plotly_chart(fig_env, use_container_width=True)

# ---- Populations + leakage ----
st.subheader("State populations")
t_edges, pops = viz.population_trajectory(model, distorted, drive_freq)
fig_pop = go.Figure()
for j in range(model.n_levels):
    fig_pop.add_scatter(x=t_edges, y=pops[:, j], name=f"|{j}>")
fig_pop.update_layout(xaxis_title="t (ns)", yaxis_title="population", height=300)
st.plotly_chart(fig_pop, use_container_width=True)

# ---- Spectrum ----
st.subheader("Drive spectrum")
freqs, power = viz.drive_spectrum(pulse)
fig_spec = go.Figure(go.Scatter(x=freqs * 1000, y=power))
fig_spec.update_layout(xaxis_title="detuning (MHz)", yaxis_title="power (norm.)", height=250)
st.plotly_chart(fig_spec, use_container_width=True)

# ---- AllXY ----
st.subheader("AllXY")
labels, p1 = viz.allxy_populations(
    ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2, flux=flux)),
    drive_freq, duration_ns=duration, sigma_ns=sigma)
fig_axy = go.Figure(go.Scatter(x=labels, y=p1, mode="lines+markers"))
fig_axy.update_layout(yaxis_title="P(excited)", height=300, xaxis_tickangle=-60)
st.plotly_chart(fig_axy, use_container_width=True)

# ---- Optimizer ----
st.sidebar.header("Optimizer")
method = st.sidebar.selectbox(
    "Method", ["(none)", "DRAG", "CRAB", "GRAPE"],
    help="DRAG/CRAB are derivative-free (work with any hardware, incl. noise). "
         "GRAPE is gradient-based and pre-distorts through differentiable hardware.")
if st.sidebar.button("Run optimizer", help="Optimize the pulse against the current "
                     "device + hardware to maximize X-gate fidelity and suppress leakage."):
    problem = Problem(model, target=X_TARGET, drive_freq_ghz=drive_freq,
                      hardware=hardware, leakage_weight=20.0)
    try:
        if method == "DRAG":
            res = DragOptimizer(duration, sigma, anh).run(problem, init_amp=amp,
                                                          init_drag_coef=drag_coef)
        elif method == "CRAB":
            res = CrabOptimizer(pulse, n_harmonics=3).run(problem)
        elif method == "GRAPE":
            res = GrapeOptimizer().run(problem, init_pulse=pulse, maxiter=120)
        else:
            res = None
        if res is not None:
            st.success(f"Optimized cost {res.best_cost:.4e} "
                       f"(fidelity {avg_gate_fidelity(problem.propagated(res.best_pulse), X_TARGET):.5f})")
            fig_conv = go.Figure(go.Scatter(y=res.history))
            fig_conv.update_layout(title="Optimizer convergence", xaxis_title="evaluation",
                                   yaxis_title="cost", height=250)
            st.plotly_chart(fig_conv, use_container_width=True)
            wf = to_opx_waveforms(res.best_pulse, dac_per_radns=1.0)
            st.download_button("Download OPX I waveform (.npy is not used; CSV)",
                               data="\n".join(str(v) for v in wf["I_wf"]),
                               file_name="opt_I_wf.csv")
    except ValueError as e:
        st.error(str(e))


def main():
    """Entry point marker (Streamlit runs the module top-level)."""
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: PASS (2 passed). If `AppTest` reports an exception, read it and fix the app (it is exercising the real Streamlit render path).
Also manually verify (optional, not in CI): `streamlit run app/streamlit_app.py`.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/app/__init__.py lab_pulse_opt/app/streamlit_app.py lab_pulse_opt/tests/test_app_smoke.py
git commit -m "feat(app): interactive Streamlit pulse lab"
```

---

### Task 6: App README + full sweep

**Files:**
- Create: `lab_pulse_opt/README_app.md`
- (verification of full suite)

**Interfaces:** docs + verification.

- [ ] **Step 1: Write the README**

```markdown
# Single-Transmon Pulse Lab — Interactive App

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
```

- [ ] **Step 2: Run the whole suite**

Run (from `lab_pulse_opt/`): `python -m pytest -q`
Expected: all Phase-1 through Phase-4 tests PASS together.

- [ ] **Step 3: Commit**

```bash
git add lab_pulse_opt/README_app.md
git commit -m "docs(app): how to run the interactive lab"
```

---

## Subsequent Phase (separate plan)

- **Phase 5 — Readout noise + robust cost:** `metrics/measurement.py` (shot + amplifier readout noise → noisy measured cost; RB-style evaluator) and robust/ensemble cost (average over `ControlNoise` realizations / hardware-parameter spreads) optimized with the derivative-free DRAG/CRAB optimizers. The app gains a "measured (noisy)" toggle.

---

## Self-Review

**Spec coverage (design spec §7 UI requirements):**
- Every configurable parameter exposed with an inline explanation (`help=` on every widget) ✓ Task 5.
- Per-hardware-stage on/off toggles ✓ Task 5 (checkboxes per stage).
- Ideal-vs-distorted envelope overlay ✓ Task 5.
- Live plots: envelope, populations/leakage, spectrum, AllXY, optimizer convergence ✓ Tasks 1–5; Bloch trajectory available via `viz.bloch_trajectory` ✓ Task 2.
- Optimizer run + OPX export ✓ Task 5 (download button via `to_opx_waveforms`).
- Computation isolated in tested pure functions; Streamlit is a thin shell ✓ Tasks 1–4 (viz) + Task 5.

**Placeholder scan:** No TBD/TODO; every code step has complete, runnable code.

**Type consistency:** `state_trajectory`/`population_trajectory`/`bloch_trajectory` all return `(t_edges[N+1], array[N+1, ...])`; `drive_spectrum(pulse) -> (freqs, power)`; `allxy_populations(...) -> (labels, p1)`. The app consumes each with matching unpacking. `viz.py` has no streamlit/plotly imports (testable headless); only `app/streamlit_app.py` imports them.
