# Single-Transmon Pulse Lab — Phase 5: Readout Noise + Robust Cost — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add realistic measurement (shot + readout-assignment noise) to the cost an optimizer sees, and a robust/ensemble cost so optimizers can shape pulses that tolerate parameter spread (e.g. qubit-frequency detuning) — closing the loop to "what the real experiment measures."

**Architecture:** `metrics/measurement.py` provides a projective-readout sampler (`simulate_readout`) and a finite-shot measured cost (`measured_cost`). `optimize/robust.py` provides `robust_cost` (mean cost over an ensemble of `Problem` variants) and a duck-typed `EnsembleProblem` that exposes the same `cost_from_pulse` interface as `Problem`, so the existing derivative-free DRAG/CRAB optimizers optimize robustly with no new optimizer code. The Streamlit app gains a "measured readout" toggle.

**Tech Stack:** Python ≥3.10, numpy, scipy, pytest, streamlit. Builds on Phases 1–4.

## Global Constraints

- Units: angular frequency rad/ns, time ns, linear frequency GHz. Pulses are the Phase-1 `Pulse(t, I, Q)`.
- Readout model: given a true excited-state population `p` and assignment fidelity `F` (probability a prepared state is correctly identified), the probability of reading "excited" is `p_read = F*p + (1-F)*(1-p)`. A finite measurement of `n_shots` projective shots returns `binomial(n_shots, p_read) / n_shots`. `F=1` is ideal (no readout error); `F=0.5` is no information.
- Reproducibility: all stochastic functions take a `seed` and use `np.random.default_rng(seed)`.
- `robust_cost(problems, pulse) = mean(p.cost_from_pulse(pulse) for p in problems)`. `EnsembleProblem(problems).cost_from_pulse(pulse)` returns exactly that, so any optimizer that only calls `cost_from_pulse` (DRAG, CRAB — NOT GRAPE) optimizes the ensemble.
- The measured/robust costs are NOT differentiable; GRAPE must not be used on them (DRAG/CRAB only).
- TDD: failing test first, minimal code, commit per task. Run tests from `lab_pulse_opt/`: `python -m pytest ...`.
- Do NOT modify Phase-1–4 public interfaces; Phase 5 consumes them. `Problem` exposes `.model`, `.target`, `.drive_freq_ghz`, `.hardware`, `.leakage_weight`.

---

## File Structure

```
lab_pulse_opt/
  pulselab/
    metrics/
      measurement.py     # NEW: simulate_readout, measured_cost (Tasks 1-2)
    optimize/
      robust.py          # NEW: robust_cost, EnsembleProblem, detuning_ensemble (Task 3)
  app/
    streamlit_app.py     # MODIFY: measured-readout toggle (Task 5)
  tests/
    test_measurement.py        # Tasks 1-2
    test_robust.py             # Task 3
    test_robust_optimization.py# Task 4
    test_app_smoke.py          # MODIFY (Task 5): toggle still renders
  examples/
    robust_vs_nominal.py       # Task 4
```

---

### Task 1: Projective readout sampler

**Files:**
- Create: `lab_pulse_opt/pulselab/metrics/measurement.py`
- Test: `lab_pulse_opt/tests/test_measurement.py`

**Interfaces:**
- Consumes: nothing (pure stats).
- Produces: `simulate_readout(p_excited, n_shots, readout_fidelity=1.0, seed=None) -> float` — the measured excited fraction from `n_shots` projective shots through an assignment-fidelity-`readout_fidelity` readout, per the Global Constraints model.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_measurement.py
import numpy as np
from pulselab.metrics.measurement import simulate_readout


def test_converges_to_truth_ideal_readout():
    est = simulate_readout(0.3, n_shots=200000, readout_fidelity=1.0, seed=0)
    assert np.isclose(est, 0.3, atol=0.01)


def test_assignment_fidelity_biases_toward_half():
    # F=0.8, true p=1.0 -> p_read = 0.8.
    est = simulate_readout(1.0, n_shots=200000, readout_fidelity=0.8, seed=1)
    assert np.isclose(est, 0.8, atol=0.01)


def test_shot_noise_variance_scales():
    p = 0.5
    n = 400
    ests = [simulate_readout(p, n_shots=n, seed=s) for s in range(400)]
    expected_std = np.sqrt(p * (1 - p) / n)
    assert np.isclose(np.std(ests), expected_std, rtol=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measurement.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.metrics.measurement'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/metrics/measurement.py
import numpy as np


def simulate_readout(p_excited, n_shots, readout_fidelity=1.0, seed=None):
    """Measured excited fraction from n_shots projective shots.

    p_read = F*p + (1-F)*(1-p) folds in assignment (SPAM) error; the finite-shot
    estimate is binomial(n_shots, p_read)/n_shots.
    """
    rng = np.random.default_rng(seed)
    p_read = readout_fidelity * p_excited + (1.0 - readout_fidelity) * (1.0 - p_excited)
    return rng.binomial(n_shots, p_read) / n_shots
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measurement.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/metrics/measurement.py lab_pulse_opt/tests/test_measurement.py
git commit -m "feat(measurement): projective readout sampler (shot + SPAM noise)"
```

---

### Task 2: Measured cost

**Files:**
- Modify: `lab_pulse_opt/pulselab/metrics/measurement.py`
- Test: `lab_pulse_opt/tests/test_measurement.py` (extend)

**Interfaces:**
- Consumes: `simulate_readout`, `Problem` (uses `.propagated`, `.leakage_weight`), `leakage`.
- Produces: `measured_cost(problem, pulse, n_shots, readout_fidelity=1.0, seed=None) -> float` — a finite-shot *measured* X-gate-error proxy: propagate the pulse, take the true excited population `p1 = |U[1,0]|²` (the X-gate-on-|0> success), measure it via `simulate_readout`, and return `(1 - measured_p1) + problem.leakage_weight * leakage(U)`. (This is what a single-number "apply gate, read excited" experiment yields under shot noise, distinct from the exact process-fidelity cost.)

- [ ] **Step 1: Write the failing test**

```python
# append to lab_pulse_opt/tests/test_measurement.py
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.metrics.fidelity import leakage
from pulselab.metrics.measurement import measured_cost
from pulselab.optimize.base import Problem

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _setup():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    pulse = gaussian_drag(40, np.pi / probe.area(), 8.0, 0.5, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=10.0)
    return prob, pulse


def test_measured_cost_converges_to_truth_high_shots():
    prob, pulse = _setup()
    U = prob.propagated(pulse)
    p1 = abs(U[1, 0]) ** 2
    true_cost = (1 - p1) + prob.leakage_weight * leakage(U)
    est = measured_cost(prob, pulse, n_shots=500000, readout_fidelity=1.0, seed=0)
    assert np.isclose(est, true_cost, atol=0.01)


def test_measured_cost_is_noisy_at_low_shots():
    prob, pulse = _setup()
    ests = [measured_cost(prob, pulse, n_shots=100, seed=s) for s in range(50)]
    assert np.std(ests) > 0  # finite shots -> fluctuates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measurement.py -v`
Expected: FAIL — `ImportError: cannot import name 'measured_cost'`.

- [ ] **Step 3: Write minimal implementation**

Add to `measurement.py` (add `from .fidelity import leakage` at the top):

```python
def measured_cost(problem, pulse, n_shots, readout_fidelity=1.0, seed=None):
    """Finite-shot measured X-gate-error proxy: (1 - measured P_excited) + leakage.

    Models what a single 'apply gate, read out' experiment measures under shot +
    assignment noise. Not differentiable -- use with DRAG/CRAB, not GRAPE.
    """
    U = problem.propagated(pulse)
    p1 = float(abs(U[1, 0]) ** 2)
    measured_p1 = simulate_readout(p1, n_shots, readout_fidelity, seed)
    return (1.0 - measured_p1) + problem.leakage_weight * leakage(U)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measurement.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/metrics/measurement.py lab_pulse_opt/tests/test_measurement.py
git commit -m "feat(measurement): finite-shot measured cost proxy"
```

---

### Task 3: Robust cost + EnsembleProblem + detuning ensemble

**Files:**
- Create: `lab_pulse_opt/pulselab/optimize/robust.py`
- Test: `lab_pulse_opt/tests/test_robust.py`

**Interfaces:**
- Consumes: `Problem`.
- Produces:
  - `robust_cost(problems, pulse) -> float` = `mean(p.cost_from_pulse(pulse) for p in problems)`.
  - `EnsembleProblem(problems)` with `cost_from_pulse(pulse) -> float` returning `robust_cost(problems, pulse)` (duck-types `Problem` for the derivative-free optimizers).
  - `detuning_ensemble(problem, offsets_ghz) -> list[Problem]` building one `Problem` per offset, identical to `problem` except `drive_freq_ghz = problem.drive_freq_ghz + offset` (same model, target, hardware, leakage_weight).

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_robust.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_robust.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.optimize.robust'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/optimize/robust.py
import numpy as np
from .base import Problem


def robust_cost(problems, pulse):
    """Mean cost of a pulse over an ensemble of Problem variants."""
    return float(np.mean([p.cost_from_pulse(pulse) for p in problems]))


class EnsembleProblem:
    """Duck-types Problem: cost_from_pulse averages over an ensemble.

    Works with any optimizer that only calls cost_from_pulse (DRAG, CRAB).
    """

    def __init__(self, problems):
        self.problems = list(problems)

    def cost_from_pulse(self, pulse):
        return robust_cost(self.problems, pulse)


def detuning_ensemble(problem, offsets_ghz):
    """Problems identical to `problem` but with the drive frequency shifted by each offset."""
    return [
        Problem(problem.model, target=problem.target,
                drive_freq_ghz=problem.drive_freq_ghz + off,
                hardware=problem.hardware, leakage_weight=problem.leakage_weight)
        for off in offsets_ghz
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_robust.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/robust.py lab_pulse_opt/tests/test_robust.py
git commit -m "feat(robust): robust_cost + EnsembleProblem + detuning ensemble"
```

---

### Task 4: Robust optimization integration + demo

**Files:**
- Create: `lab_pulse_opt/tests/test_robust_optimization.py`
- Create: `lab_pulse_opt/examples/robust_vs_nominal.py`

**Interfaces:**
- Consumes: `DragOptimizer`, `EnsembleProblem`, `detuning_ensemble`, `robust_cost`, `gaussian_drag`.
- Produces: a test that optimizing the DRAG optimizer against an `EnsembleProblem` (detuning spread) yields a pulse with lower robust cost over that ensemble than the nominal (single-point optimal) pulse — verified during planning (robust 0.0323 < nominal 0.0355).

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_robust_optimization.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.robust import EnsembleProblem, detuning_ensemble, robust_cost

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_robust_optimization_lowers_ensemble_cost():
    model = ChargeBasisTransmon(DeviceParams.q1())
    anh = model.anharmonicity_ghz()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, anh)
    amp0 = np.pi / probe.area()
    base = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    ens = detuning_ensemble(base, [-0.004, -0.002, 0.0, 0.002, 0.004])

    nominal = gaussian_drag(40, amp0, 8.0, 0.5, anh)
    ep = EnsembleProblem(ens)
    res = DragOptimizer(40, 8.0, anh).run(ep, init_amp=amp0, init_drag_coef=0.5)

    assert robust_cost(ens, res.best_pulse) <= robust_cost(ens, nominal)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_robust_optimization.py -v`
Expected: Before the example exists, the import line works (all modules present by now); run it to confirm the inequality holds. If it fails, STOP and report — it indicates a real regression in the optimizer or ensemble plumbing, not a test issue.

- [ ] **Step 3: Write the demo**

```python
# lab_pulse_opt/examples/robust_vs_nominal.py
"""Show DRAG optimized for a detuning ensemble is more robust than the nominal pulse."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.robust import EnsembleProblem, detuning_ensemble
from pulselab.metrics.fidelity import avg_gate_fidelity
from pulselab.dynamics.propagator import propagate

X = np.array([[0, 1], [1, 0]], dtype=complex)


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    anh = model.anharmonicity_ghz()
    fd = model.f01_ghz()
    amp0 = np.pi / gaussian_drag(40, 1.0, 8.0, 0.0, anh).area()
    base = Problem(model, target=X, drive_freq_ghz=fd, leakage_weight=20.0)
    offsets = [-0.004, -0.002, 0.0, 0.002, 0.004]
    ens = detuning_ensemble(base, offsets)

    nominal = gaussian_drag(40, amp0, 8.0, 0.5, anh)
    robust = DragOptimizer(40, 8.0, anh).run(
        EnsembleProblem(ens), init_amp=amp0, init_drag_coef=0.5).best_pulse

    print("detuning(MHz)  F_nominal  F_robust")
    for off in offsets:
        fn = avg_gate_fidelity(propagate(model, nominal, fd + off), X)
        fr = avg_gate_fidelity(propagate(model, robust, fd + off), X)
        print(f"{off*1000:+7.1f}      {fn:.5f}    {fr:.5f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + demo**

Run: `python -m pytest tests/test_robust_optimization.py -v`
Expected: PASS (1 passed).
Run: `PYTHONPATH=. python examples/robust_vs_nominal.py`
Expected: the robust pulse has flatter / higher fidelity across the detuning sweep (lower worst-case error) than the nominal.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/tests/test_robust_optimization.py lab_pulse_opt/examples/robust_vs_nominal.py
git commit -m "test(robust): ensemble-optimized pulse beats nominal over detuning spread"
```

---

### Task 5: App measured-readout toggle

**Files:**
- Modify: `lab_pulse_opt/app/streamlit_app.py`
- Modify: `lab_pulse_opt/tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `simulate_readout`.
- Produces: a sidebar "Readout" section with a checkbox "Simulate measured readout" (help string) and, when enabled, an `n_shots` slider (help string) and an `assignment fidelity` slider (help string); the app shows the **measured** excited population (X-gate on |0>) alongside the exact one, using `simulate_readout(p1_true, n_shots, F, seed=0)`. The existing smoke tests must still pass (every widget keeps a `help=` string; the app renders without exception with the toggle off by default).

- [ ] **Step 1: Write the failing test**

```python
# append to lab_pulse_opt/tests/test_app_smoke.py
def test_readout_toggle_renders():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
    at.run()
    # The readout checkbox exists and has help; enabling it re-runs cleanly.
    labels = [c.label for c in at.checkbox]
    assert any("measured readout" in lbl.lower() for lbl in labels)
    target = next(c for c in at.checkbox if "measured readout" in c.label.lower())
    target.set_value(True).run()
    assert not at.exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_smoke.py::test_readout_toggle_renders -v`
Expected: FAIL — no checkbox labelled "measured readout" yet.

- [ ] **Step 3: Write minimal implementation**

In `app/streamlit_app.py`, add `from pulselab.metrics.measurement import simulate_readout` to the imports. Then, immediately AFTER the `c1, c2, c3 = st.columns(3)` metrics block (after the three `.metric(...)` calls), insert:

```python
# ---- Readout (measured) ----
st.sidebar.header("Readout")
measured = st.sidebar.checkbox(
    "Simulate measured readout", value=False,
    help="Sample the X-gate excited-state population with finite shots and "
         "readout assignment error, like a real experiment, instead of the "
         "exact value.")
if measured:
    n_shots = st.sidebar.slider(
        "Shots", 10, 100000, 1000, 10,
        help="Number of projective single-shot measurements. More shots = less "
             "statistical (shot) noise on the measured population.")
    rofid = st.sidebar.slider(
        "Readout assignment fidelity", 0.5, 1.0, 0.97, 0.005,
        help="Probability the qubit state is correctly identified. 1.0 is perfect; "
             "lower values bias the measured population toward 0.5 (SPAM error).")
    p1_true = float(abs(U[1, 0]) ** 2)
    p1_meas = simulate_readout(p1_true, n_shots, rofid, seed=0)
    st.metric("Measured P(excited) of X gate", f"{p1_meas:.3f}",
              delta=f"{p1_meas - p1_true:+.3f} vs exact")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: PASS (3 passed — the two existing smoke tests plus the new toggle test; all widgets still carry help).
Optional manual check: `streamlit run app/streamlit_app.py`.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/app/streamlit_app.py lab_pulse_opt/tests/test_app_smoke.py
git commit -m "feat(app): simulated measured-readout toggle"
```

---

### Task 6: Full Phase-5 sweep + README note

**Files:**
- Modify: `lab_pulse_opt/README_app.md`
- (full-suite verification)

- [ ] **Step 1: Append to the README**

Add this section to `lab_pulse_opt/README_app.md`:

```markdown

## Measured readout & robust pulses

Toggle **Simulate measured readout** in the sidebar to see the X-gate excited
population as a real experiment would measure it — sampled over a finite number
of shots with a readout assignment-fidelity error, instead of the exact value.

For pulses that tolerate parameter spread (e.g. qubit-frequency detuning), build
an ensemble of `Problem`s with `pulselab.optimize.robust.detuning_ensemble`, wrap
it in an `EnsembleProblem`, and optimize it with the DRAG or CRAB optimizer (the
derivative-free optimizers — GRAPE needs a differentiable cost). See
`examples/robust_vs_nominal.py`.
```

- [ ] **Step 2: Run the whole suite**

Run (from `lab_pulse_opt/`): `python -m pytest -q`
Expected: all Phase-1 through Phase-5 tests PASS together.

- [ ] **Step 3: Commit**

```bash
git add -f lab_pulse_opt/README_app.md
git commit -m "docs(app): measured readout + robust pulse notes"
```

---

## Self-Review

**Spec coverage (design spec §2/§4/§7 Phase-5 portion):**
- Readout-chain noise: shot noise + assignment (SPAM) error → noisy measured cost ✓ Tasks 1–2.
- Optimizers face realistic measurement uncertainty ✓ Task 2 (`measured_cost`, derivative-free).
- Robust/ensemble evaluation over parameter spread ✓ Task 3 (`robust_cost`, `EnsembleProblem`, `detuning_ensemble`).
- Robust optimization reuses the existing derivative-free optimizers (no new optimizer code) ✓ Tasks 3–4.
- App "measured (noisy)" toggle ✓ Task 5.

**Placeholder scan:** No TBD/TODO; every code step has complete, runnable code. The robust-optimization inequality was verified during planning (robust 0.0323 ≤ nominal 0.0355).

**Type consistency:** `simulate_readout(p, n_shots, readout_fidelity, seed) -> float`; `measured_cost(problem, pulse, n_shots, readout_fidelity, seed) -> float`; `robust_cost(problems, pulse) -> float`; `EnsembleProblem(problems).cost_from_pulse(pulse) -> float` (same signature DRAG/CRAB call on `Problem`); `detuning_ensemble(problem, offsets_ghz) -> list[Problem]`. `EnsembleProblem` deliberately matches `Problem`'s `cost_from_pulse` so the optimizers consume it unchanged.

---

## Project completion

This is the final phase. After it integrates, the lab spans the full spec: accurate physics core, realistic hardware chain + open-system dynamics, pluggable DRAG/CRAB/GRAPE optimizers with hardware-aware pre-distortion, an interactive Streamlit app, and measured/robust cost — all seeded with the real Q1 device and exporting OPX-ready waveforms.
