# Single-Transmon Pulse Lab — Phase 2: Hardware Chain + Open System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a composable hardware-distortion chain (transfer function, bias-tee droop, IQ imbalance, control noise) that distorts the ideal pulse before it reaches the qubit, plus Lindblad open-system dynamics (T1/Tφ), so the optimizer in Phase 3 can be made to cope with realistic non-idealities.

**Architecture:** Hardware stages implement a small `HardwareStage` interface (`apply(pulse) -> pulse`, `jacobian(pulse) -> matrix | None`); a `Chain` composes them in order. Linear stages (transfer function, bias-tee, IQ-imbalance mixing) expose an exact Jacobian over the stacked `[I; Q]` sample vector so Phase-3 GRAPE can backpropagate through them; stochastic stages (control noise) return `None`. Open-system dynamics are a separate `dynamics/lindblad.py` module that vectorizes the density matrix and propagates it piecewise with the Liouvillian superoperator built from the same `H(t)` the coherent propagator uses.

**Tech Stack:** Python ≥3.10, numpy, scipy (`scipy.signal.lfilter`, `scipy.linalg.expm`), pytest. Builds on the Phase-1 `pulselab` package.

## Global Constraints

- Units everywhere: angular frequency rad/ns, time ns, linear frequency GHz. Decay rates in 1/ns; convert from microseconds via `gamma = 1 / (T_us * 1000)`.
- Hardware stages operate on a `Pulse` (Phase-1 dataclass: `t`, `I`, `Q` arrays in ns / rad·ns⁻¹). A stage returns a NEW `Pulse` on the same time grid; it must not mutate its input.
- The "stacked sample vector" for Jacobians is `x = concatenate([I, Q])`, length `2N`. A linear stage's `jacobian(pulse)` returns the `(2N, 2N)` real matrix `J` such that `concatenate([out.I, out.Q]) == J @ concatenate([in.I, in.Q]) + const`, where `const` is the stage's input-independent offset (zero for all stages except IQ-imbalance DC leakage).
- IIR transfer-function / bias-tee follow the OPX convention (`scipy.signal.lfilter(b, a, x)`), matching the real device's `feedforward`(b)/`feedback`(a) filter slots. The device's measured line time constants are τ₁ = 126.7 ns and τ₂ = 6546.0 ns (use as realistic defaults / test values).
- Lindblad uses the COLUMN-STACKING vec convention: `vec(A X B) = (B.T ⊗ A) vec(X)`. The Liouvillian is `L = -1j*(kron(I, H) - kron(H.T, I)) + sum_k D_k` with `D_k = kron(conj(c), c) - 0.5*kron(I, c.conj().T @ c) - 0.5*kron((c.conj().T @ c).T, I)`.
- TDD: failing test first, minimal code, commit per task. Run tests from `lab_pulse_opt/`: `python -m pytest ...`.
- Do NOT modify Phase-1 public interfaces (`TransmonModel`, `Pulse`, `propagate`, metrics) except the two explicitly-scoped cleanups in Task 1.

---

## File Structure

```
lab_pulse_opt/
  pulselab/
    device/
      params.py            # MODIFY (Task 1): harden from_spectrum convergence
      hamiltonian.py       # MODIFY (Task 1): memoize _eigensystem
    pulse/
      hardware.py          # NEW: HardwareStage, Chain, TransferFunction,
                           #      BiasTeeDroop, IQImbalance, ControlNoise (Tasks 2-6)
    dynamics/
      lindblad.py          # NEW: collapse operators + Liouvillian propagation (Tasks 8-9)
  tests/
    test_eigensystem_cache.py     # Task 1
    test_from_spectrum_guard.py   # Task 1
    test_hardware_interface.py    # Task 2
    test_transfer_function.py     # Task 3
    test_bias_tee.py              # Task 4
    test_iq_imbalance.py          # Task 5
    test_control_noise.py         # Task 6
    test_hardware_jacobian.py     # Task 7
    test_hardware_integration.py  # Task 7
    test_collapse_operators.py    # Task 8
    test_lindblad.py              # Task 9, 10
  examples/
    distortion_vs_ideal.py        # Task 7
    t1_decay.py                   # Task 10
```

---

### Task 1: Phase-1 cleanup — memoize eigensystem + guard the solver

**Files:**
- Modify: `lab_pulse_opt/pulselab/device/hamiltonian.py`
- Modify: `lab_pulse_opt/pulselab/device/params.py`
- Test: `lab_pulse_opt/tests/test_eigensystem_cache.py`, `lab_pulse_opt/tests/test_from_spectrum_guard.py`

**Interfaces:**
- Consumes: Phase-1 `ChargeBasisTransmon`, `DeviceParams`.
- Produces: `ChargeBasisTransmon._eigensystem` memoized (same return signature `(evals_ghz, evecs)`); `DeviceParams.from_spectrum` raises `ValueError` if `fsolve` fails to converge.

- [ ] **Step 1: Write the failing tests**

```python
# lab_pulse_opt/tests/test_eigensystem_cache.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_eigensystem_is_cached():
    m = ChargeBasisTransmon(DeviceParams.q1())
    a = m._eigensystem()
    b = m._eigensystem()
    # Same cached arrays returned (identity), not recomputed.
    assert a[0] is b[0]
    assert a[1] is b[1]


def test_cached_values_still_correct():
    m = ChargeBasisTransmon(DeviceParams.q1())
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
```

```python
# lab_pulse_opt/tests/test_from_spectrum_guard.py
import numpy as np
import pytest
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_from_spectrum_still_works():
    p = DeviceParams.from_spectrum(5.252, -0.064)
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)


def test_from_spectrum_raises_on_nonconvergence():
    # Physically impossible target (positive anharmonicity for a transmon)
    # drives the solver to fail; we want a clear error, not silent garbage.
    with pytest.raises(ValueError):
        DeviceParams.from_spectrum(5.252, +5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_eigensystem_cache.py tests/test_from_spectrum_guard.py -v`
Expected: `test_eigensystem_is_cached` FAILS (returns fresh arrays, `is` is False); `test_from_spectrum_raises_on_nonconvergence` FAILS (no ValueError raised).

- [ ] **Step 3: Implement memoization + convergence guard**

In `hamiltonian.py`, add `import functools` at the top, and replace `ChargeBasisTransmon._eigensystem` with a cached version:

```python
    @functools.cached_property
    def _cached_eigensystem(self):
        evals, evecs = np.linalg.eigh(self._hamiltonian_ghz())
        return evals, evecs

    def _eigensystem(self):
        return self._cached_eigensystem
```

(`cached_property` works because `DeviceParams` is frozen and the model is constructed per parameter set.)

In `params.py`, change the `fsolve` call inside `from_spectrum` to check convergence:

```python
        x, _info, ier, msg = fsolve(residual, [ec0, ej0], full_output=True)
        if ier != 1:
            raise ValueError(f"from_spectrum failed to converge: {msg}")
        ec, ej = x
        return cls(EC_ghz=float(ec), EJ_ghz=float(ej), **kwargs)
```

- [ ] **Step 4: Run tests + full suite**

Run: `python -m pytest tests/test_eigensystem_cache.py tests/test_from_spectrum_guard.py -v`
Expected: PASS.
Run: `python -m pytest -q`
Expected: all prior Phase-1 tests still pass.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/hamiltonian.py lab_pulse_opt/pulselab/device/params.py lab_pulse_opt/tests/test_eigensystem_cache.py lab_pulse_opt/tests/test_from_spectrum_guard.py
git commit -m "perf(device): memoize eigensystem; guard from_spectrum convergence"
```

---

### Task 2: HardwareStage interface + Chain

**Files:**
- Create: `lab_pulse_opt/pulselab/pulse/hardware.py`
- Test: `lab_pulse_opt/tests/test_hardware_interface.py`

**Interfaces:**
- Consumes: `pulselab.pulse.envelope.Pulse`.
- Produces:
  - ABC `HardwareStage` with abstract `apply(self, pulse: Pulse) -> Pulse` and concrete `jacobian(self, pulse: Pulse) -> np.ndarray | None` (default returns `None`).
  - `IdentityStage(HardwareStage)`: returns an equal copy; `jacobian` returns `np.eye(2*N)`.
  - `Chain(stages: list[HardwareStage])`: `apply` runs stages in order; `jacobian(pulse)` returns the composed matrix product `J_n @ ... @ J_1` if every stage returns a non-None Jacobian (evaluating each on the intermediate pulse), else `None`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_hardware_interface.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import HardwareStage, IdentityStage, Chain


def _pulse(n=8):
    t = np.arange(n) * 1.0
    return Pulse(t=t, I=np.linspace(0, 1, n), Q=np.linspace(1, 0, n))


def test_identity_stage_roundtrip_and_jacobian():
    p = _pulse()
    out = IdentityStage().apply(p)
    assert np.allclose(out.I, p.I) and np.allclose(out.Q, p.Q)
    assert out is not p  # new object, no mutation
    J = IdentityStage().jacobian(p)
    assert J.shape == (16, 16)
    assert np.allclose(J, np.eye(16))


def test_chain_applies_in_order_and_composes_jacobian():
    p = _pulse()
    chain = Chain([IdentityStage(), IdentityStage()])
    out = chain.apply(p)
    assert np.allclose(out.I, p.I)
    assert np.allclose(chain.jacobian(p), np.eye(16))


def test_chain_jacobian_none_if_any_stage_nonlinear():
    class NL(HardwareStage):
        def apply(self, pulse):
            return pulse
        # inherits default jacobian -> None
    assert Chain([IdentityStage(), NL()]).jacobian(_pulse()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hardware_interface.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.pulse.hardware'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/pulse/hardware.py
from abc import ABC, abstractmethod
import numpy as np
from .envelope import Pulse


class HardwareStage(ABC):
    """A distortion stage: maps an ideal Pulse to a distorted Pulse."""

    @abstractmethod
    def apply(self, pulse: Pulse) -> Pulse:
        ...

    def jacobian(self, pulse: Pulse):
        """(2N,2N) Jacobian d[out_I;out_Q]/d[in_I;in_Q] for linear stages.

        Returns None for nonlinear/stochastic stages (the default).
        """
        return None


class IdentityStage(HardwareStage):
    def apply(self, pulse: Pulse) -> Pulse:
        return Pulse(t=pulse.t.copy(), I=pulse.I.copy(), Q=pulse.Q.copy())

    def jacobian(self, pulse: Pulse):
        n = pulse.I.size
        return np.eye(2 * n)


class Chain(HardwareStage):
    """Compose hardware stages, applied left-to-right."""

    def __init__(self, stages):
        self.stages = list(stages)

    def apply(self, pulse: Pulse) -> Pulse:
        out = pulse
        for stage in self.stages:
            out = stage.apply(out)
        return out

    def jacobian(self, pulse: Pulse):
        n = pulse.I.size
        J = np.eye(2 * n)
        current = pulse
        for stage in self.stages:
            Js = stage.jacobian(current)
            if Js is None:
                return None
            J = Js @ J
            current = stage.apply(current)
        return J
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hardware_interface.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/hardware.py lab_pulse_opt/tests/test_hardware_interface.py
git commit -m "feat(hardware): HardwareStage interface + Chain composition"
```

---

### Task 3: TransferFunction stage (IIR / finite bandwidth / rise-time)

**Files:**
- Modify: `lab_pulse_opt/pulselab/pulse/hardware.py`
- Test: `lab_pulse_opt/tests/test_transfer_function.py`

**Interfaces:**
- Consumes: `HardwareStage`, `Pulse`.
- Produces: `TransferFunction(b, a)(HardwareStage)` applying `scipy.signal.lfilter(b, a, x)` to `I` and `Q` independently (same filter on both quadratures). Classmethod `TransferFunction.single_pole_lowpass(tau_ns, dt_ns)` returns a unit-DC-gain one-pole low-pass (`r = exp(-dt/tau)`, `b=[1-r]`, `a=[1, -r]`) modeling finite bandwidth / rise-time. `jacobian` returns the exact `(2N,2N)` block-diagonal linear map (two identical `(N,N)` lower-triangular convolution blocks) computed by filtering the `N` unit impulses.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_transfer_function.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import TransferFunction


def _step(n=200, dt=1.0):
    t = np.arange(n) * dt
    return Pulse(t=t, I=np.ones(n), Q=np.zeros(n))


def test_lowpass_unit_dc_gain_and_risetime():
    tau, dt = 10.0, 1.0
    tf = TransferFunction.single_pole_lowpass(tau_ns=tau, dt_ns=dt)
    out = tf.apply(_step())
    # Settles to DC gain 1.
    assert np.isclose(out.I[-1], 1.0, atol=1e-3)
    # One time constant -> ~1 - 1/e of the way up.
    idx = int(round(tau / dt))
    assert np.isclose(out.I[idx], 1 - np.exp(-1), atol=0.05)
    # Q (zero input) stays zero.
    assert np.allclose(out.Q, 0.0)


def test_jacobian_matches_apply():
    tf = TransferFunction.single_pole_lowpass(tau_ns=10.0, dt_ns=1.0)
    n = 30
    p = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    J = tf.jacobian(p)
    assert J.shape == (2 * n, 2 * n)
    # Linear stage: apply(x) == J @ x for arbitrary x (offset is zero).
    rng = np.random.default_rng(0)
    x = rng.normal(size=2 * n)
    px = Pulse(t=p.t, I=x[:n], Q=x[n:])
    ox = tf.apply(px)
    assert np.allclose(np.concatenate([ox.I, ox.Q]), J @ x, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transfer_function.py -v`
Expected: FAIL — `ImportError: cannot import name 'TransferFunction'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hardware.py` (add `from scipy.signal import lfilter` at the top):

```python
class TransferFunction(HardwareStage):
    """Linear time-invariant drive-line response as an IIR filter applied to
    each quadrature (OPX feedforward=b / feedback=a convention)."""

    def __init__(self, b, a):
        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a, dtype=float)

    @classmethod
    def single_pole_lowpass(cls, tau_ns, dt_ns):
        r = np.exp(-dt_ns / tau_ns)
        return cls(b=[1 - r], a=[1.0, -r])

    def apply(self, pulse: Pulse) -> Pulse:
        I = lfilter(self.b, self.a, pulse.I)
        Q = lfilter(self.b, self.a, pulse.Q)
        return Pulse(t=pulse.t.copy(), I=I, Q=Q)

    def _block(self, n):
        # Columns are the filter's response to each unit impulse (LTI/causal).
        block = np.zeros((n, n))
        for j in range(n):
            imp = np.zeros(n)
            imp[j] = 1.0
            block[:, j] = lfilter(self.b, self.a, imp)
        return block

    def jacobian(self, pulse: Pulse):
        n = pulse.I.size
        block = self._block(n)
        J = np.zeros((2 * n, 2 * n))
        J[:n, :n] = block
        J[n:, n:] = block
        return J
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transfer_function.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/hardware.py lab_pulse_opt/tests/test_transfer_function.py
git commit -m "feat(hardware): IIR TransferFunction stage with exact Jacobian"
```

---

### Task 4: BiasTeeDroop stage (DC-block high-pass)

**Files:**
- Modify: `lab_pulse_opt/pulselab/pulse/hardware.py`
- Test: `lab_pulse_opt/tests/test_bias_tee.py`

**Interfaces:**
- Consumes: `HardwareStage`, `Pulse`.
- Produces: `BiasTeeDroop(tau_ns, dt_ns)(HardwareStage)`: a one-pole high-pass `b=[1, -1]`, `a=[1, -r]` with `r=exp(-dt/tau)` applied to `I` and `Q`, causing slow droop of sustained levels (DC blocked). `jacobian` returns the exact `(2N,2N)` block-diagonal map (same impulse-response construction as Task 3).

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_bias_tee.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import BiasTeeDroop


def test_droop_of_constant_level():
    tau, dt, n = 100.0, 1.0, 400
    bt = BiasTeeDroop(tau_ns=tau, dt_ns=dt)
    p = Pulse(t=np.arange(n) * dt, I=np.ones(n), Q=np.zeros(n))
    out = bt.apply(p)
    assert np.isclose(out.I[0], 1.0, atol=1e-6)         # starts at full level
    assert out.I[-1] < out.I[0]                          # droops over time
    # After one time constant the level has decayed by ~1/e.
    idx = int(round(tau / dt))
    assert np.isclose(out.I[idx], np.exp(-1), atol=0.05)


def test_jacobian_matches_apply():
    bt = BiasTeeDroop(tau_ns=100.0, dt_ns=1.0)
    n = 40
    p = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    J = bt.jacobian(p)
    rng = np.random.default_rng(1)
    x = rng.normal(size=2 * n)
    px = Pulse(t=p.t, I=x[:n], Q=x[n:])
    ox = bt.apply(px)
    assert np.allclose(np.concatenate([ox.I, ox.Q]), J @ x, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bias_tee.py -v`
Expected: FAIL — `ImportError: cannot import name 'BiasTeeDroop'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hardware.py`:

```python
class BiasTeeDroop(HardwareStage):
    """Bias-tee / DC-block high-pass: blocks DC, so sustained levels droop."""

    def __init__(self, tau_ns, dt_ns):
        r = np.exp(-dt_ns / tau_ns)
        self._tf = TransferFunction(b=[1.0, -1.0], a=[1.0, -r])

    def apply(self, pulse: Pulse) -> Pulse:
        return self._tf.apply(pulse)

    def jacobian(self, pulse: Pulse):
        return self._tf.jacobian(pulse)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bias_tee.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/hardware.py lab_pulse_opt/tests/test_bias_tee.py
git commit -m "feat(hardware): BiasTeeDroop high-pass stage"
```

---

### Task 5: IQImbalance stage (gain/phase imbalance + LO leakage)

**Files:**
- Modify: `lab_pulse_opt/pulselab/pulse/hardware.py`
- Test: `lab_pulse_opt/tests/test_iq_imbalance.py`

**Interfaces:**
- Consumes: `HardwareStage`, `Pulse`.
- Produces: `IQImbalance(gain_imbalance=0.0, phase_error_rad=0.0, dc_i=0.0, dc_q=0.0)(HardwareStage)`. Model (identity at all defaults):
  - `epsilon = 1 + gain_imbalance`, `phi = phase_error_rad`.
  - `out_I = in_I + dc_i`
  - `out_Q = epsilon * (sin(phi) * in_I + cos(phi) * in_Q) + dc_q`
  `jacobian` returns the `(2N,2N)` map of the linear part (the DC offsets are input-independent and do not appear in the Jacobian): top-left `= I_N`, bottom-left `= epsilon*sin(phi)*I_N`, bottom-right `= epsilon*cos(phi)*I_N`, top-right `= 0`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_iq_imbalance.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import IQImbalance


def _pulse(n=10):
    rng = np.random.default_rng(2)
    return Pulse(t=np.arange(n) * 1.0, I=rng.normal(size=n), Q=rng.normal(size=n))


def test_defaults_are_identity():
    p = _pulse()
    out = IQImbalance().apply(p)
    assert np.allclose(out.I, p.I) and np.allclose(out.Q, p.Q)


def test_gain_phase_and_dc():
    p = _pulse()
    stage = IQImbalance(gain_imbalance=0.1, phase_error_rad=0.05, dc_i=0.02, dc_q=-0.03)
    out = stage.apply(p)
    eps = 1.1
    assert np.allclose(out.I, p.I + 0.02)
    assert np.allclose(out.Q, eps * (np.sin(0.05) * p.I + np.cos(0.05) * p.Q) - 0.03)


def test_jacobian_is_linear_part():
    n = 10
    p = _pulse(n)
    stage = IQImbalance(gain_imbalance=0.1, phase_error_rad=0.05, dc_i=0.02, dc_q=-0.03)
    J = stage.jacobian(p)
    # Jacobian excludes the DC offset: difference of two inputs cancels it.
    rng = np.random.default_rng(3)
    dx = rng.normal(size=2 * n)
    p2 = Pulse(t=p.t, I=p.I + dx[:n], Q=p.Q + dx[n:])
    o1, o2 = stage.apply(p), stage.apply(p2)
    delta = np.concatenate([o2.I - o1.I, o2.Q - o1.Q])
    assert np.allclose(delta, J @ dx, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_iq_imbalance.py -v`
Expected: FAIL — `ImportError: cannot import name 'IQImbalance'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hardware.py`:

```python
class IQImbalance(HardwareStage):
    """IQ mixer gain/phase imbalance plus DC (LO/carrier) leakage."""

    def __init__(self, gain_imbalance=0.0, phase_error_rad=0.0, dc_i=0.0, dc_q=0.0):
        self.epsilon = 1.0 + gain_imbalance
        self.phi = phase_error_rad
        self.dc_i = dc_i
        self.dc_q = dc_q

    def apply(self, pulse: Pulse) -> Pulse:
        I = pulse.I + self.dc_i
        Q = self.epsilon * (np.sin(self.phi) * pulse.I + np.cos(self.phi) * pulse.Q) + self.dc_q
        return Pulse(t=pulse.t.copy(), I=I, Q=Q)

    def jacobian(self, pulse: Pulse):
        n = pulse.I.size
        eye = np.eye(n)
        J = np.zeros((2 * n, 2 * n))
        J[:n, :n] = eye
        J[n:, :n] = self.epsilon * np.sin(self.phi) * eye
        J[n:, n:] = self.epsilon * np.cos(self.phi) * eye
        return J
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_iq_imbalance.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/hardware.py lab_pulse_opt/tests/test_iq_imbalance.py
git commit -m "feat(hardware): IQ imbalance + LO leakage stage"
```

---

### Task 6: ControlNoise stage (jitter + 1/f)

**Files:**
- Modify: `lab_pulse_opt/pulselab/pulse/hardware.py`
- Test: `lab_pulse_opt/tests/test_control_noise.py`

**Interfaces:**
- Consumes: `HardwareStage`, `Pulse`.
- Produces: `ControlNoise(sigma, kind="white", seed=None)(HardwareStage)`. `apply` adds noise of std `sigma` to both `I` and `Q`. `kind="white"` is i.i.d. Gaussian; `kind="pink"` is 1/f noise generated by scaling Gaussian spectrum by `1/sqrt(f)` (FFT method) then normalized to std `sigma`. Uses `np.random.default_rng(seed)` so runs are reproducible when seeded. `jacobian` returns `None` (stochastic — inherits default).

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_control_noise.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import ControlNoise


def _zero(n=4096):
    return Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))


def test_seed_reproducible_and_jacobian_none():
    p = _zero(64)
    a = ControlNoise(sigma=0.1, seed=42).apply(p)
    b = ControlNoise(sigma=0.1, seed=42).apply(p)
    assert np.allclose(a.I, b.I) and np.allclose(a.Q, b.Q)
    assert ControlNoise(sigma=0.1, seed=42).jacobian(p) is None


def test_white_noise_statistics():
    p = _zero()
    out = ControlNoise(sigma=0.2, kind="white", seed=0).apply(p)
    assert np.isclose(out.I.std(), 0.2, rtol=0.1)
    assert np.isclose(out.I.mean(), 0.0, atol=0.02)


def test_pink_noise_has_more_low_frequency_power_than_white():
    p = _zero()
    white = ControlNoise(sigma=1.0, kind="white", seed=1).apply(p)
    pink = ControlNoise(sigma=1.0, kind="pink", seed=1).apply(p)
    n = p.I.size
    # Compare low-frequency band power fraction (excluding DC bin).
    def low_frac(x):
        ps = np.abs(np.fft.rfft(x)) ** 2
        ps[0] = 0.0
        return ps[1:n // 16].sum() / ps[1:].sum()
    assert low_frac(pink.I) > low_frac(white.I)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_control_noise.py -v`
Expected: FAIL — `ImportError: cannot import name 'ControlNoise'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hardware.py`:

```python
class ControlNoise(HardwareStage):
    """Additive control-line noise (stochastic; no Jacobian)."""

    def __init__(self, sigma, kind="white", seed=None):
        self.sigma = sigma
        self.kind = kind
        self.seed = seed

    def _noise(self, n, rng):
        if self.kind == "white":
            return rng.normal(0.0, self.sigma, n)
        if self.kind == "pink":
            white = rng.normal(0.0, 1.0, n)
            spectrum = np.fft.rfft(white)
            f = np.arange(spectrum.size)
            scale = np.ones_like(f, dtype=float)
            scale[1:] = 1.0 / np.sqrt(f[1:])  # 1/sqrt(f) amplitude -> 1/f power
            shaped = np.fft.irfft(spectrum * scale, n)
            std = shaped.std()
            return shaped * (self.sigma / std) if std > 0 else shaped
        raise ValueError(f"unknown noise kind: {self.kind}")

    def apply(self, pulse: Pulse) -> Pulse:
        rng = np.random.default_rng(self.seed)
        n = pulse.I.size
        return Pulse(
            t=pulse.t.copy(),
            I=pulse.I + self._noise(n, rng),
            Q=pulse.Q + self._noise(n, rng),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_control_noise.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/hardware.py lab_pulse_opt/tests/test_control_noise.py
git commit -m "feat(hardware): ControlNoise stage (white + 1/f)"
```

---

### Task 7: Chain Jacobian verification + propagator integration

**Files:**
- Create: `lab_pulse_opt/tests/test_hardware_jacobian.py`
- Create: `lab_pulse_opt/tests/test_hardware_integration.py`
- Create: `lab_pulse_opt/examples/distortion_vs_ideal.py`

**Interfaces:**
- Consumes: `Chain`, `TransferFunction`, `BiasTeeDroop`, `IQImbalance`, Phase-1 `gaussian_drag`, `propagate`, `leakage`, `ChargeBasisTransmon`, `DeviceParams`.
- Produces: verification that (a) the composed `Chain` Jacobian of all-linear stages matches finite differences, and (b) a distorted pulse run through `propagate` changes the gate physics (lower fidelity / different leakage) vs the ideal — confirming the chain feeds the dynamics correctly.

- [ ] **Step 1: Write the failing tests**

```python
# lab_pulse_opt/tests/test_hardware_jacobian.py
import numpy as np
from pulselab.pulse.envelope import Pulse
from pulselab.pulse.hardware import Chain, TransferFunction, BiasTeeDroop, IQImbalance


def test_chain_jacobian_matches_finite_difference():
    n = 24
    rng = np.random.default_rng(5)
    p = Pulse(t=np.arange(n) * 1.0, I=rng.normal(size=n), Q=rng.normal(size=n))
    chain = Chain([
        TransferFunction.single_pole_lowpass(8.0, 1.0),
        BiasTeeDroop(200.0, 1.0),
        IQImbalance(gain_imbalance=0.05, phase_error_rad=0.02, dc_i=0.01, dc_q=0.0),
    ])
    J = chain.jacobian(p)
    assert J is not None and J.shape == (2 * n, 2 * n)
    # Finite-difference each input sample (offsets cancel in the difference).
    base = chain.apply(p)
    base_vec = np.concatenate([base.I, base.Q])
    x0 = np.concatenate([p.I, p.Q])
    eps = 1e-6
    for k in range(2 * n):
        xk = x0.copy(); xk[k] += eps
        pk = Pulse(t=p.t, I=xk[:n], Q=xk[n:])
        ok = chain.apply(pk)
        col = (np.concatenate([ok.I, ok.Q]) - base_vec) / eps
        assert np.allclose(col, J[:, k], atol=1e-5)
```

```python
# lab_pulse_opt/tests/test_hardware_integration.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_bandwidth_limiting_degrades_an_otherwise_good_pulse():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    pulse = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())

    f_ideal = avg_gate_fidelity(propagate(model, pulse, model.f01_ghz()), X)
    # Aggressive low-pass (short tau) distorts the envelope -> worse gate.
    distorted = Chain([TransferFunction.single_pole_lowpass(3.0, 1.0)]).apply(pulse)
    f_distorted = avg_gate_fidelity(propagate(model, distorted, model.f01_ghz()), X)

    assert f_distorted < f_ideal
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hardware_jacobian.py tests/test_hardware_integration.py -v`
Expected: `test_hardware_integration` may fail on import of `propagate`/`gaussian_drag` only if paths are wrong — they exist from Phase 1, so the real first-run failure is the assertion if the chain weren't wired; with the chain implemented these should pass. If `tests/test_hardware_jacobian.py` cannot import a class, that indicates a Task 2-5 gap — fix before proceeding. Confirm both fail before the example exists, then pass after Step 3.

- [ ] **Step 3: Write the example demo**

```python
# lab_pulse_opt/examples/distortion_vs_ideal.py
"""Show how drive-line bandwidth limiting degrades a DRAG gate."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage

X = np.array([[0, 1], [1, 0]], dtype=complex)


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    pulse = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())
    U = propagate(model, pulse, model.f01_ghz())
    print(f"ideal      F={avg_gate_fidelity(U, X):.5f}  leak={leakage(U):.3e}")
    for tau in [20.0, 10.0, 5.0, 3.0]:
        d = Chain([TransferFunction.single_pole_lowpass(tau, 1.0)]).apply(pulse)
        Ud = propagate(model, d, model.f01_ghz())
        print(f"tau={tau:5.1f}ns F={avg_gate_fidelity(Ud, X):.5f}  leak={leakage(Ud):.3e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + demo**

Run: `python -m pytest tests/test_hardware_jacobian.py tests/test_hardware_integration.py -v`
Expected: PASS (2 passed).
Run: `python examples/distortion_vs_ideal.py` (needs `PYTHONPATH=.` or editable install)
Expected: fidelity decreases as tau shrinks (tighter bandwidth = more distortion).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/tests/test_hardware_jacobian.py lab_pulse_opt/tests/test_hardware_integration.py lab_pulse_opt/examples/distortion_vs_ideal.py
git commit -m "test(hardware): chain Jacobian vs finite-diff + propagator integration"
```

---

### Task 8: Lindblad collapse operators

**Files:**
- Create: `lab_pulse_opt/pulselab/dynamics/lindblad.py`
- Test: `lab_pulse_opt/tests/test_collapse_operators.py`

**Interfaces:**
- Consumes: Phase-1 `TransmonModel` (for `n_levels`), `DeviceParams` (for `T1_us`, `Tphi_us`).
- Produces: `collapse_operators(n_levels, T1_us, Tphi_us, include_relaxation=True, include_dephasing=True) -> list[np.ndarray]`:
  - Relaxation: `sqrt(gamma1) * A` where `A[j-1, j] = sqrt(j)` (lowering ladder), `gamma1 = 1/(T1_us*1000)` per ns.
  - Dephasing: `sqrt(2*gamma_phi) * N` where `N = diag(0,1,...,n_levels-1)`, `gamma_phi = 1/(Tphi_us*1000)`.
  Each operator is `(n_levels, n_levels)` complex.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_collapse_operators.py
import numpy as np
from pulselab.dynamics.lindblad import collapse_operators


def test_relaxation_and_dephasing_present_and_shaped():
    ops = collapse_operators(n_levels=3, T1_us=15.0, Tphi_us=30.0)
    assert len(ops) == 2
    for c in ops:
        assert c.shape == (3, 3)


def test_relaxation_rate_normalization():
    (c_relax,) = collapse_operators(3, T1_us=15.0, Tphi_us=30.0,
                                    include_dephasing=False)
    g1 = 1.0 / (15.0 * 1000)
    # |0><1| element is sqrt(gamma1)*sqrt(1).
    assert np.isclose(c_relax[0, 1], np.sqrt(g1))
    # |1><2| element is sqrt(gamma1)*sqrt(2).
    assert np.isclose(c_relax[1, 2], np.sqrt(g1) * np.sqrt(2))


def test_dephasing_is_number_operator():
    (c_phi,) = collapse_operators(4, T1_us=15.0, Tphi_us=30.0,
                                  include_relaxation=False)
    gphi = 1.0 / (30.0 * 1000)
    assert np.allclose(np.diag(c_phi), np.sqrt(2 * gphi) * np.arange(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collapse_operators.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.dynamics.lindblad'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/dynamics/lindblad.py
import numpy as np


def collapse_operators(n_levels, T1_us, Tphi_us,
                       include_relaxation=True, include_dephasing=True):
    """Lindblad collapse operators for a transmon ladder (rates in 1/ns)."""
    ops = []
    if include_relaxation:
        gamma1 = 1.0 / (T1_us * 1000.0)
        A = np.zeros((n_levels, n_levels), dtype=complex)
        for j in range(1, n_levels):
            A[j - 1, j] = np.sqrt(j)  # lowering ladder
        ops.append(np.sqrt(gamma1) * A)
    if include_dephasing:
        gamma_phi = 1.0 / (Tphi_us * 1000.0)
        N = np.diag(np.arange(n_levels)).astype(complex)
        ops.append(np.sqrt(2 * gamma_phi) * N)
    return ops
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collapse_operators.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/dynamics/lindblad.py lab_pulse_opt/tests/test_collapse_operators.py
git commit -m "feat(lindblad): transmon collapse operators from T1/Tphi"
```

---

### Task 9: Liouvillian + density-matrix propagation

**Files:**
- Modify: `lab_pulse_opt/pulselab/dynamics/lindblad.py`
- Test: `lab_pulse_opt/tests/test_lindblad.py`

**Interfaces:**
- Consumes: `collapse_operators`, Phase-1 `TransmonModel.rotating_frame_operators`, `Pulse`.
- Produces:
  - `liouvillian(H, c_ops) -> ndarray`: the `(d², d²)` superoperator for Hamiltonian `H` (d×d) and collapse ops, using the column-stacking convention in Global Constraints.
  - `lindblad_propagate(model, pulse, drive_freq_ghz, rho0, c_ops) -> ndarray`: piecewise-constant evolution of density matrix `rho0` (d×d) returning the final `rho` (d×d). For each slice build `H_k = H0 + I_k*X_op + Q_k*Y_op`, `L_k = liouvillian(H_k, c_ops)`, advance `vec(rho)` by `expm(L_k*dt) @ vec(rho)`. Use `vec = rho.flatten(order="F")` (column stacking) and unstack with `.reshape((d,d), order="F")`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_lindblad.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import Pulse
from pulselab.dynamics.lindblad import liouvillian, lindblad_propagate, collapse_operators


def test_liouvillian_zero_dissipation_preserves_trace_and_is_unitary_like():
    # With no collapse ops, a diagonal H just adds phases: populations unchanged.
    H = np.diag([0.0, 1.0]).astype(complex)
    L = liouvillian(H, [])
    rho0 = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
    from scipy.linalg import expm
    vec = expm(L * 0.5) @ rho0.flatten(order="F")
    rho = vec.reshape((2, 2), order="F")
    assert np.isclose(np.trace(rho).real, 1.0)               # trace preserved
    assert np.allclose(np.diag(rho).real, np.diag(rho0).real)  # populations fixed (diagonal H)


def test_zero_drive_no_dissipation_is_identity_on_populations():
    model = ChargeBasisTransmon(DeviceParams.q1())
    n = 50
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = np.zeros((model.n_levels, model.n_levels), dtype=complex)
    rho0[1, 1] = 1.0  # start in |1>
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops=[])
    assert np.isclose(rho[1, 1].real, 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lindblad.py -v`
Expected: FAIL — `ImportError: cannot import name 'liouvillian'`.

- [ ] **Step 3: Write minimal implementation**

Add to `lindblad.py` (add `from scipy.linalg import expm` at top):

```python
def liouvillian(H, c_ops):
    """Column-stacking Liouvillian superoperator (d^2, d^2)."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for c in c_ops:
        cdc = c.conj().T @ c
        L += (np.kron(c.conj(), c)
              - 0.5 * np.kron(I, cdc)
              - 0.5 * np.kron(cdc.T, I))
    return L


def lindblad_propagate(model, pulse, drive_freq_ghz, rho0, c_ops):
    """Piecewise-constant master-equation evolution; returns final rho (d,d)."""
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    dt = pulse.dt
    d = H0.shape[0]
    vec = rho0.astype(complex).flatten(order="F")
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Hk = H0 + Ik * X_op + Qk * Y_op
        vec = expm(liouvillian(Hk, c_ops) * dt) @ vec
    return vec.reshape((d, d), order="F")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lindblad.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/dynamics/lindblad.py lab_pulse_opt/tests/test_lindblad.py
git commit -m "feat(lindblad): Liouvillian + density-matrix propagation"
```

---

### Task 10: Lindblad physics anchors — T1 decay + dephasing

**Files:**
- Modify: `lab_pulse_opt/tests/test_lindblad.py`
- Create: `lab_pulse_opt/examples/t1_decay.py`

**Interfaces:**
- Consumes: `lindblad_propagate`, `collapse_operators`, `ChargeBasisTransmon`, `DeviceParams`, `Pulse`.
- Produces: physics validation that excited-state population decays as `exp(-t/T1)` under relaxation, and that the |0>-|1> coherence decays as `exp(-t/T2)` under dephasing; plus a runnable demo.

- [ ] **Step 1: Write the failing tests**

```python
# append to lab_pulse_opt/tests/test_lindblad.py
def test_t1_decay_matches_exponential():
    # Short T1 so decay is visible over a few hundred ns; idle (zero drive).
    T1_us = 0.1  # 100 ns
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    c_ops = collapse_operators(2, T1_us=T1_us, Tphi_us=1e9, include_dephasing=False)
    t_total = 100.0  # one T1
    n = int(t_total)
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1>
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
    # After one T1, excited population ~ 1/e.
    assert np.isclose(rho[1, 1].real, np.exp(-1), atol=0.02)


def test_pure_dephasing_decays_coherence():
    Tphi_us = 0.1  # 100 ns
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    c_ops = collapse_operators(2, T1_us=1e9, Tphi_us=Tphi_us, include_relaxation=False)
    n = 100
    zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
    rho0 = 0.5 * np.ones((2, 2), dtype=complex)  # |+> : coherence = 0.5
    rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
    assert np.isclose(abs(rho[0, 1]), 0.5 * np.exp(-1), atol=0.02)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lindblad.py::test_t1_decay_matches_exponential tests/test_lindblad.py::test_pure_dephasing_decays_coherence -v`
Expected: These are new functions; before adding the example they should already PASS if Task 9 is correct (they exercise existing code). Run them now to confirm the physics; if either fails, STOP and treat it as a physics bug in Task 9 — do not loosen the tolerance.

- [ ] **Step 3: Write the demo**

```python
# lab_pulse_opt/examples/t1_decay.py
"""Show T1 relaxation of the excited-state population under the master equation."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import Pulse
from pulselab.dynamics.lindblad import lindblad_propagate, collapse_operators


def main():
    T1_us = 0.1
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2))
    c_ops = collapse_operators(2, T1_us=T1_us, Tphi_us=1e9, include_dephasing=False)
    rho0 = np.array([[0, 0], [0, 1]], dtype=complex)
    for t_total in [0.0, 50.0, 100.0, 200.0]:
        n = max(int(t_total), 1)
        zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
        rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
        print(f"t={t_total:6.1f}ns  P1={rho[1,1].real:.4f}  (exp={np.exp(-t_total/100):.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + demo**

Run: `python -m pytest tests/test_lindblad.py -v`
Expected: PASS (all 4 lindblad tests).
Run: `python examples/t1_decay.py` (needs `PYTHONPATH=.`)
Expected: `P1` follows `exp(-t/100ns)`.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/tests/test_lindblad.py lab_pulse_opt/examples/t1_decay.py
git commit -m "test(lindblad): T1 decay + dephasing physics anchors"
```

---

### Task 11: Full Phase-2 suite sweep

**Files:** none (verification task).

- [ ] **Step 1: Run the whole suite**

Run (from `lab_pulse_opt/`): `python -m pytest -q`
Expected: all Phase-1 + Phase-2 tests PASS together.

- [ ] **Step 2: Commit (only if fixups were needed)**

```bash
git add -A lab_pulse_opt
git commit -m "chore(pulselab): phase-2 hardware chain + open system green"
```

---

## Subsequent Phases (separate plans)

- **Phase 3 — Optimizers:** `optimize/base.py` (`Problem` bundling model+hardware-chain+target+metric → `cost`; `Optimizer` interface), `drag.py`, `grape.py` (analytic gradients backpropagated through the linear-stage Jacobians built here), `crab.py`; `export/opx.py`.
- **Phase 4 — Streamlit app:** parameter panels with inline explanations, per-stage on/off toggles, ideal-vs-distorted overlays, live plots, compare mode, export.
- **Phase 5 — Readout noise + robust cost:** `metrics/measurement.py` (shot + amplifier noise → noisy measured cost; AllXY/RB evaluators) and robust/ensemble evaluation using `ControlNoise` + hardware-parameter ensembles.

---

## Self-Review

**Spec coverage (Phase 2 portion of the design spec §2/§4):**
- Drive-line transfer function (finite bandwidth / rise-time / ringing) ✓ Task 3 (low-pass + general IIR for ringing).
- Bias-tee long-tail droop ✓ Task 4.
- IQ imbalance + LO leakage ✓ Task 5.
- Control noise (jitter + 1/f) ✓ Task 6.
- Composable chain; linear stages expose `jacobian()` ✓ Tasks 2,3,4,5,7.
- Lindblad master equation with collapse ops from T1/Tφ ✓ Tasks 8,9,10.
- Toggle coherent/Lindblad: both propagators now exist (`propagate`, `lindblad_propagate`) sharing `rotating_frame_operators` ✓.
- Reviewer-deferred cleanups (#2 from_spectrum guard, #3 eigensystem memoization) ✓ Task 1.

**Placeholder scan:** No TBD/TODO; every code step has complete runnable code.

**Type consistency:** `Pulse(t,I,Q)` used identically throughout; `jacobian` returns `(2N,2N)` or `None` consistently (Tasks 2-7); `collapse_operators(...) -> list[ndarray]` consumed by `liouvillian(H, c_ops)` and `lindblad_propagate(model, pulse, drive_freq_ghz, rho0, c_ops)` with matching argument names (Tasks 8-10); column-stacking convention (`flatten(order="F")`/`reshape(order="F")`) consistent with the `liouvillian` kron ordering.
