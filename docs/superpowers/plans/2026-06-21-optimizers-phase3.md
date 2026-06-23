# Single-Transmon Pulse Lab — Phase 3: Optimizers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable optimizer layer — a `Problem` cost surface plus analytic-DRAG, CRAB, and GRAPE optimizers — that shapes single-qubit pulses against the Phase-1 model seen through the Phase-2 hardware chain, including GRAPE gradients backpropagated through the linear hardware Jacobians, and an OPX waveform exporter.

**Architecture:** A `Problem` bundles model + hardware chain + target + drive frequency + leakage weight into one `cost_from_pulse(pulse) -> float`. Optimizers implement `Optimizer.run(problem, ...) -> OptResult`. DRAG and CRAB are derivative-free `scipy.optimize.minimize` searches over a few shape parameters. GRAPE optimizes piecewise-constant samples with an analytic gradient (matrix-exponential Fréchet derivatives + forward/backward propagator products) and chains that gradient through the hardware `Chain.jacobian` via `Jᵀ`. The `Problem.cost_from_pulse` surface is the single extension point a new optimizer subclasses.

**Tech Stack:** Python ≥3.10, numpy, scipy (`scipy.optimize.minimize`, `scipy.linalg.expm`, `scipy.linalg.expm_frechet`), pytest. Builds on Phase-1 (`propagate`, `avg_gate_fidelity`, `leakage`, `gaussian_drag`) and Phase-2 (`Chain` + `jacobian`).

## Global Constraints

- Units: angular frequency rad/ns, time ns, linear frequency GHz. Pulses are the Phase-1 `Pulse(t, I, Q)`.
- Cost convention everywhere: `cost = (1 - avg_gate_fidelity(U, target)) + leakage_weight * leakage(U)`, where `U = propagate(model, hardware.apply(pulse), drive_freq_ghz)` and `target` is a 2×2 unitary. Lower is better. `avg_gate_fidelity` denominator is `d(d+1)=6` (d=2 subspace).
- GRAPE gradient (VERIFIED against finite differences to ~1e-9, both with and without hardware): for the distorted samples the propagator sees, with `H_k = H0 + Id_k·X_op + Qd_k·Y_op`, `U_k = expm(-1j·H_k·dt)`, `U = U_N…U_1`:
  - `M = U[:2,:2]`, `a = trace(target.conj().T @ M)`.
  - `G = -(a*target + M)/6 - (leakage_weight/2)*M`  (this is ∂cost/∂M*).
  - Embed `Ghat` (d×d) with top-left 2×2 = `G`, rest 0.
  - With `Fkm1[k] = U_{k-1}…U_1` (Fkm1[0]=I) and `Bk[k] = U_N…U_{k+1}` (Bk[N-1]=I): `Pre_k = Fkm1[k] @ Ghat.conj().T @ Bk[k]`.
  - `grad_I[k] = 2·Re(trace(Pre_k @ L_I))`, `grad_Q[k] = 2·Re(trace(Pre_k @ L_Q))`, where `L_I = expm_frechet(-1j·H_k·dt, -1j·dt·X_op)[1]`, `L_Q = expm_frechet(-1j·H_k·dt, -1j·dt·Y_op)[1]`.
- Hardware backprop (VERIFIED): gradient w.r.t. IDEAL samples `= chain.jacobian(ideal_pulse).T @ grad_distorted`. If `chain.jacobian(...)` is `None` (stochastic/nonlinear stage), GRAPE must raise a clear error directing the user to a derivative-free optimizer.
- The stacked sample vector is `concatenate([I, Q])`, length `2N` — same convention as the Phase-2 Jacobians.
- TDD: failing test first, minimal code, commit per task. Run tests from `lab_pulse_opt/`: `python -m pytest ...`.
- Do NOT modify Phase-1/Phase-2 public interfaces. Optimizers consume them only.

---

## File Structure

```
lab_pulse_opt/
  pulselab/
    optimize/
      __init__.py        # NEW (Task 1)
      base.py            # Problem, OptResult, Optimizer ABC (Task 1)
      drag.py            # DragOptimizer (Task 2)
      crab.py            # CrabOptimizer (Task 3)
      grape.py           # cost+grad (Task 4) + GrapeOptimizer (Task 5)
    export/
      __init__.py        # NEW (Task 6)
      opx.py             # to_opx_waveforms (Task 6)
  tests/
    test_problem.py            # Task 1
    test_drag_optimizer.py     # Task 2
    test_crab_optimizer.py     # Task 3
    test_grape_gradient.py     # Task 4
    test_grape_optimizer.py    # Task 5
    test_opx_export.py         # Task 6
    test_optimizer_integration.py  # Task 7
  examples/
    optimize_x_gate.py         # Task 7
```

---

### Task 1: Problem + Optimizer interface

**Files:**
- Create: `lab_pulse_opt/pulselab/optimize/__init__.py` (empty)
- Create: `lab_pulse_opt/pulselab/optimize/base.py`
- Test: `lab_pulse_opt/tests/test_problem.py`

**Interfaces:**
- Consumes: Phase-1 `propagate`, `avg_gate_fidelity`, `leakage`; Phase-2 `IdentityStage`; `Pulse`.
- Produces:
  - `Problem(model, target, drive_freq_ghz, hardware=None, leakage_weight=0.0)`. If `hardware is None`, use `IdentityStage()`. Methods: `propagated(pulse) -> ndarray` (returns `U` after applying hardware), `cost_from_pulse(pulse) -> float` per the cost convention.
  - `OptResult` dataclass: `best_pulse: Pulse`, `best_cost: float`, `history: list[float]`, `n_iter: int`.
  - `Optimizer` ABC with abstract `run(self, problem, **kwargs) -> OptResult`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_problem.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import IdentityStage
from pulselab.optimize.base import Problem

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _model():
    return ChargeBasisTransmon(DeviceParams.q1())


def test_cost_lower_for_better_pulse():
    model = _model()
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    bare = gaussian_drag(40, amp, 8.0, 0.0, model.anharmonicity_ghz())
    drag = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=10.0)
    assert prob.cost_from_pulse(drag) < prob.cost_from_pulse(bare)


def test_identity_hardware_default():
    model = _model()
    p = gaussian_drag(40, 0.1, 8.0, 0.0, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz())
    # default hardware is identity -> propagated equals direct propagate
    from pulselab.dynamics.propagator import propagate
    assert np.allclose(prob.propagated(p), propagate(model, p, model.f01_ghz()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_problem.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.optimize'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/optimize/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from ..dynamics.propagator import propagate
from ..metrics.fidelity import avg_gate_fidelity, leakage
from ..pulse.hardware import IdentityStage


class Problem:
    """Bundles the model, hardware chain, target and weights into one cost."""

    def __init__(self, model, target, drive_freq_ghz, hardware=None, leakage_weight=0.0):
        self.model = model
        self.target = np.asarray(target, dtype=complex)
        self.drive_freq_ghz = drive_freq_ghz
        self.hardware = hardware if hardware is not None else IdentityStage()
        self.leakage_weight = leakage_weight

    def propagated(self, pulse):
        distorted = self.hardware.apply(pulse)
        return propagate(self.model, distorted, self.drive_freq_ghz)

    def cost_from_pulse(self, pulse):
        U = self.propagated(pulse)
        return (1.0 - avg_gate_fidelity(U, self.target)
                + self.leakage_weight * leakage(U))


@dataclass
class OptResult:
    best_pulse: object
    best_cost: float
    history: list = field(default_factory=list)
    n_iter: int = 0


class Optimizer(ABC):
    @abstractmethod
    def run(self, problem, **kwargs) -> OptResult:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_problem.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/__init__.py lab_pulse_opt/pulselab/optimize/base.py lab_pulse_opt/tests/test_problem.py
git commit -m "feat(optimize): Problem cost surface + Optimizer interface"
```

---

### Task 2: DRAG optimizer

**Files:**
- Create: `lab_pulse_opt/pulselab/optimize/drag.py`
- Test: `lab_pulse_opt/tests/test_drag_optimizer.py`

**Interfaces:**
- Consumes: `Problem`, `OptResult`, `Optimizer`, `gaussian_drag`, `scipy.optimize.minimize`.
- Produces: `DragOptimizer(duration_ns, sigma_ns, anharmonicity_ghz, dt_ns=1.0)`; `run(problem, init_amp, init_drag_coef=0.0) -> OptResult` optimizing `[amp, drag_coef]` with Nelder-Mead over `problem.cost_from_pulse(gaussian_drag(...))`. `OptResult.best_pulse` is the optimized Gaussian-DRAG pulse.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_drag_optimizer.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_drag_optimizer_beats_bare_gaussian():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp0 = np.pi / probe.area()
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    bare_cost = prob.cost_from_pulse(
        gaussian_drag(40, amp0, 8.0, 0.0, model.anharmonicity_ghz()))
    opt = DragOptimizer(40, 8.0, model.anharmonicity_ghz())
    res = opt.run(prob, init_amp=amp0, init_drag_coef=0.0)
    assert res.best_cost < bare_cost
    assert len(res.history) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drag_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.optimize.drag'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/optimize/drag.py
import numpy as np
from scipy.optimize import minimize
from ..pulse.envelope import gaussian_drag
from .base import Optimizer, OptResult


class DragOptimizer(Optimizer):
    """Optimize [amplitude, drag_coef] of a Gaussian-DRAG pulse."""

    def __init__(self, duration_ns, sigma_ns, anharmonicity_ghz, dt_ns=1.0):
        self.duration_ns = duration_ns
        self.sigma_ns = sigma_ns
        self.anharmonicity_ghz = anharmonicity_ghz
        self.dt_ns = dt_ns

    def _pulse(self, params):
        amp, coef = params
        return gaussian_drag(self.duration_ns, amp, self.sigma_ns, coef,
                             self.anharmonicity_ghz, self.dt_ns)

    def run(self, problem, init_amp, init_drag_coef=0.0):
        history = []

        def objective(params):
            c = problem.cost_from_pulse(self._pulse(params))
            history.append(c)
            return c

        res = minimize(objective, x0=[init_amp, init_drag_coef], method="Nelder-Mead")
        return OptResult(best_pulse=self._pulse(res.x), best_cost=float(res.fun),
                         history=history, n_iter=res.nit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drag_optimizer.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/drag.py lab_pulse_opt/tests/test_drag_optimizer.py
git commit -m "feat(optimize): analytic-DRAG optimizer"
```

---

### Task 3: CRAB optimizer

**Files:**
- Create: `lab_pulse_opt/pulselab/optimize/crab.py`
- Test: `lab_pulse_opt/tests/test_crab_optimizer.py`

**Interfaces:**
- Consumes: `Problem`, `OptResult`, `Optimizer`, `Pulse`, `scipy.optimize.minimize`.
- Produces: `CrabOptimizer(base_pulse, n_harmonics)`; `run(problem) -> OptResult`. Parametrization: each harmonic `n=1..n_harmonics` adds `aI_n·sin(2π n t/T)` to `I` and `aQ_n·sin(2π n t/T)` to `Q` (sin vanishes at the endpoints, keeping the correction zero at the boundaries). `params` length `2*n_harmonics`, init zeros (so the base pulse is the starting point). Optimized with Nelder-Mead.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_crab_optimizer.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.crab import CrabOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_crab_improves_on_base_pulse():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp0 = np.pi / probe.area()
    base = gaussian_drag(40, amp0, 8.0, 0.0, model.anharmonicity_ghz())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    base_cost = prob.cost_from_pulse(base)
    opt = CrabOptimizer(base, n_harmonics=2)
    res = opt.run(prob)
    assert res.best_cost <= base_cost
    assert len(res.history) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crab_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.optimize.crab'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/optimize/crab.py
import numpy as np
from scipy.optimize import minimize
from ..pulse.envelope import Pulse
from .base import Optimizer, OptResult


class CrabOptimizer(Optimizer):
    """Optimize sine-basis corrections to a base pulse's I and Q envelopes."""

    def __init__(self, base_pulse, n_harmonics):
        self.base = base_pulse
        self.n_harmonics = n_harmonics
        t = base_pulse.t
        T = t[-1] - t[0] + (t[1] - t[0])  # full window incl. last sample
        self._basis = np.stack(
            [np.sin(2 * np.pi * n * (t - t[0]) / T) for n in range(1, n_harmonics + 1)]
        )  # shape (n_harmonics, N)

    def _pulse(self, params):
        aI = params[: self.n_harmonics]
        aQ = params[self.n_harmonics:]
        I = self.base.I + aI @ self._basis
        Q = self.base.Q + aQ @ self._basis
        return Pulse(t=self.base.t.copy(), I=I, Q=Q)

    def run(self, problem):
        history = []

        def objective(params):
            c = problem.cost_from_pulse(self._pulse(params))
            history.append(c)
            return c

        x0 = np.zeros(2 * self.n_harmonics)
        res = minimize(objective, x0=x0, method="Nelder-Mead")
        return OptResult(best_pulse=self._pulse(res.x), best_cost=float(res.fun),
                         history=history, n_iter=res.nit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crab_optimizer.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/crab.py lab_pulse_opt/tests/test_crab_optimizer.py
git commit -m "feat(optimize): CRAB basis-expansion optimizer"
```

---

### Task 4: GRAPE analytic gradient (no hardware)

**Files:**
- Create: `lab_pulse_opt/pulselab/optimize/grape.py`
- Test: `lab_pulse_opt/tests/test_grape_gradient.py`

**Interfaces:**
- Consumes: `Problem`, Phase-1 `rotating_frame_operators`, `avg_gate_fidelity`, `leakage`; `scipy.linalg.expm`, `scipy.linalg.expm_frechet`.
- Produces: `cost_grad_distorted(problem, I, Q, dt) -> (cost: float, grad: ndarray[2N])` — the cost and its gradient w.r.t. the (already-distorted) piecewise-constant samples the propagator sees, using the verified Fréchet-derivative method in Global Constraints. `cost` must equal `problem.cost_from_pulse(Pulse(t, I, Q))` (with identity hardware) to numerical precision.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_grape_gradient.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import Pulse
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import cost_grad_distorted

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _setup():
    model = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=3))
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=5.0)
    rng = np.random.default_rng(0)
    N = 12
    I = 0.1 * rng.normal(size=N)
    Q = 0.1 * rng.normal(size=N)
    return prob, I, Q, N


def test_cost_matches_problem():
    prob, I, Q, N = _setup()
    t = np.arange(N) * 1.0
    c, _ = cost_grad_distorted(prob, I, Q, dt=1.0)
    assert np.isclose(c, prob.cost_from_pulse(Pulse(t=t, I=I, Q=Q)))


def test_gradient_matches_finite_difference():
    prob, I, Q, N = _setup()
    t = np.arange(N) * 1.0
    _, grad = cost_grad_distorted(prob, I, Q, dt=1.0)
    eps = 1e-6
    fd = np.zeros(2 * N)
    x0 = np.concatenate([I, Q])
    for k in range(2 * N):
        xp = x0.copy(); xp[k] += eps
        xm = x0.copy(); xm[k] -= eps
        cp = prob.cost_from_pulse(Pulse(t=t, I=xp[:N], Q=xp[N:]))
        cm = prob.cost_from_pulse(Pulse(t=t, I=xm[:N], Q=xm[N:]))
        fd[k] = (cp - cm) / (2 * eps)
    assert np.max(np.abs(grad - fd)) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grape_gradient.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.optimize.grape'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/optimize/grape.py
import numpy as np
from scipy.linalg import expm, expm_frechet
from ..metrics.fidelity import avg_gate_fidelity, leakage


def cost_grad_distorted(problem, I, Q, dt):
    """Cost and gradient w.r.t. the distorted piecewise-constant samples.

    Returns (cost, grad) with grad = concatenate([dC/dI, dC/dQ]), length 2N.
    """
    H0, X_op, Y_op = problem.model.rotating_frame_operators(problem.drive_freq_ghz)
    target = problem.target
    w = problem.leakage_weight
    d = H0.shape[0]
    N = len(I)

    Us = []
    U = np.eye(d, dtype=complex)
    for k in range(N):
        Hk = H0 + I[k] * X_op + Q[k] * Y_op
        Uk = expm(-1j * Hk * dt)
        Us.append(Uk)
        U = Uk @ U

    cost = (1.0 - avg_gate_fidelity(U, target) + w * leakage(U))

    # Forward Fkm1[k] = U_{k-1}...U_1 ; backward Bk[k] = U_N...U_{k+1}.
    Fkm1 = [None] * N
    accf = np.eye(d, dtype=complex)
    for k in range(N):
        Fkm1[k] = accf
        accf = Us[k] @ accf
    Bk = [None] * N
    accb = np.eye(d, dtype=complex)
    for k in range(N - 1, -1, -1):
        Bk[k] = accb
        accb = accb @ Us[k]

    M = U[:2, :2]
    a = np.trace(target.conj().T @ M)
    G = -(a * target + M) / 6.0 - (w / 2.0) * M
    Ghat = np.zeros((d, d), dtype=complex)
    Ghat[:2, :2] = G

    gI = np.zeros(N)
    gQ = np.zeros(N)
    for k in range(N):
        Ak = -1j * (H0 + I[k] * X_op + Q[k] * Y_op) * dt
        _, LI = expm_frechet(Ak, -1j * dt * X_op)
        _, LQ = expm_frechet(Ak, -1j * dt * Y_op)
        Pre = Fkm1[k] @ Ghat.conj().T @ Bk[k]
        gI[k] = 2 * np.real(np.trace(Pre @ LI))
        gQ[k] = 2 * np.real(np.trace(Pre @ LQ))

    return float(cost), np.concatenate([gI, gQ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grape_gradient.py -v`
Expected: PASS (2 passed). The gradient matches finite differences to < 1e-5.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/grape.py lab_pulse_opt/tests/test_grape_gradient.py
git commit -m "feat(optimize): GRAPE analytic gradient (Frechet-derivative method)"
```

---

### Task 5: GRAPE optimizer with hardware backprop

**Files:**
- Modify: `lab_pulse_opt/pulselab/optimize/grape.py`
- Test: `lab_pulse_opt/tests/test_grape_optimizer.py`

**Interfaces:**
- Consumes: `cost_grad_distorted`, `Problem`, `Pulse`, `scipy.optimize.minimize`.
- Produces: `GrapeOptimizer()(Optimizer)`; `run(problem, init_pulse, maxiter=200) -> OptResult`. It optimizes the IDEAL samples `concatenate([I, Q])`. At each evaluation: build the ideal `Pulse`; `distorted = problem.hardware.apply(ideal)`; `(cost, grad_dist) = cost_grad_distorted(problem, distorted.I, distorted.Q, dt)`; `J = problem.hardware.jacobian(ideal)`; if `J is None` raise `ValueError` (stochastic/nonlinear hardware — use a derivative-free optimizer); else `grad_ideal = J.T @ grad_dist`. Minimize with `method="L-BFGS-B", jac=True`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_grape_optimizer.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.pulse.hardware import Chain, TransferFunction, ControlNoise
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import GrapeOptimizer

X = np.array([[0, 1], [1, 0]], dtype=complex)


def _init_pulse(model):
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    return gaussian_drag(40, amp, 8.0, 0.0, model.anharmonicity_ghz())


def test_grape_reduces_cost_no_hardware():
    model = ChargeBasisTransmon(DeviceParams.q1())
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), leakage_weight=20.0)
    init = _init_pulse(model)
    c0 = prob.cost_from_pulse(init)
    res = GrapeOptimizer().run(prob, init_pulse=init, maxiter=60)
    assert res.best_cost < c0


def test_grape_predistorts_through_lowpass():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([TransferFunction.single_pole_lowpass(20.0, 1.0)])
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                   hardware=chain, leakage_weight=20.0)
    init = _init_pulse(model)
    c0 = prob.cost_from_pulse(init)  # naive pulse through the distorting line
    res = GrapeOptimizer().run(prob, init_pulse=init, maxiter=80)
    # GRAPE pre-distorts to substantially beat the naive pulse.
    assert res.best_cost < c0


def test_grape_raises_on_nondifferentiable_hardware():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([ControlNoise(sigma=0.01, seed=0)])  # jacobian is None
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(), hardware=chain)
    init = _init_pulse(model)
    try:
        GrapeOptimizer().run(prob, init_pulse=init, maxiter=5)
        raised = False
    except ValueError:
        raised = True
    assert raised
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grape_optimizer.py -v`
Expected: FAIL — `ImportError: cannot import name 'GrapeOptimizer'`.

- [ ] **Step 3: Write minimal implementation**

Add to `grape.py` (add imports `from scipy.optimize import minimize`, `from ..pulse.envelope import Pulse`, `from .base import Optimizer, OptResult`):

```python
class GrapeOptimizer(Optimizer):
    """Gradient-based piecewise-constant optimization with hardware backprop."""

    def run(self, problem, init_pulse, maxiter=200):
        t = init_pulse.t
        dt = init_pulse.dt
        N = init_pulse.I.size
        history = []

        def cost_and_grad(x):
            ideal = Pulse(t=t, I=x[:N], Q=x[N:])
            distorted = problem.hardware.apply(ideal)
            cost, grad_dist = cost_grad_distorted(problem, distorted.I, distorted.Q, dt)
            J = problem.hardware.jacobian(ideal)
            if J is None:
                raise ValueError(
                    "GRAPE requires differentiable hardware (jacobian is None); "
                    "use a derivative-free optimizer (DRAG/CRAB) for this chain.")
            history.append(cost)
            return cost, J.T @ grad_dist

        x0 = np.concatenate([init_pulse.I, init_pulse.Q])
        res = minimize(cost_and_grad, x0=x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        best = Pulse(t=t, I=res.x[:N], Q=res.x[N:])
        return OptResult(best_pulse=best, best_cost=float(res.fun),
                         history=history, n_iter=res.nit)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_grape_optimizer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/optimize/grape.py lab_pulse_opt/tests/test_grape_optimizer.py
git commit -m "feat(optimize): GRAPE optimizer with hardware Jacobian backprop"
```

---

### Task 6: OPX waveform export

**Files:**
- Create: `lab_pulse_opt/pulselab/export/__init__.py` (empty)
- Create: `lab_pulse_opt/pulselab/export/opx.py`
- Test: `lab_pulse_opt/tests/test_opx_export.py`

**Interfaces:**
- Consumes: `Pulse`.
- Produces: `to_opx_waveforms(pulse, dac_per_radns) -> dict` returning `{"I_wf": ndarray, "Q_wf": ndarray}` where each array is the per-sample waveform in OPX DAC units: `I_wf = pulse.I * dac_per_radns`, `Q_wf = pulse.Q * dac_per_radns`. `dac_per_radns` is the user's power-Rabi calibration (DAC amplitude per rad/ns of Rabi rate). Arrays are plain Python-friendly numpy float arrays of length `len(pulse.t)`, ready to drop into `configuration.py` as arbitrary waveform `samples`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_opx_export.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_opx_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.export'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/export/opx.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_opx_export.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/export/__init__.py lab_pulse_opt/pulselab/export/opx.py lab_pulse_opt/tests/test_opx_export.py
git commit -m "feat(export): OPX I/Q waveform export"
```

---

### Task 7: Optimizer integration + demo

**Files:**
- Create: `lab_pulse_opt/tests/test_optimizer_integration.py`
- Create: `lab_pulse_opt/examples/optimize_x_gate.py`

**Interfaces:**
- Consumes: all optimizers, `Problem`, `Chain`, `TransferFunction`, metrics.
- Produces: a test that, through a bandwidth-limiting line, GRAPE's pre-distorted pulse achieves higher gate fidelity than the same naive DRAG pulse pushed through that line; plus a demo comparing DRAG / CRAB / GRAPE.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_optimizer_integration.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.optimize.base import Problem
from pulselab.optimize.grape import GrapeOptimizer
from pulselab.metrics.fidelity import avg_gate_fidelity

X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_grape_predistortion_beats_naive_through_line():
    model = ChargeBasisTransmon(DeviceParams.q1())
    chain = Chain([TransferFunction.single_pole_lowpass(15.0, 1.0)])
    prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                   hardware=chain, leakage_weight=20.0)
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    naive = gaussian_drag(40, amp, 8.0, 0.5, model.anharmonicity_ghz())

    f_naive = avg_gate_fidelity(prob.propagated(naive), X)
    res = GrapeOptimizer().run(prob, init_pulse=naive, maxiter=120)
    f_grape = avg_gate_fidelity(prob.propagated(res.best_pulse), X)
    assert f_grape > f_naive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimizer_integration.py -v`
Expected: before the example exists it still imports fine (all modules exist by now); the test should PASS once Tasks 1–5 are in. Run it to confirm the end-to-end behavior; if it fails, STOP and report (it indicates a real optimizer bug, not a test issue).

- [ ] **Step 3: Write the demo**

```python
# lab_pulse_opt/examples/optimize_x_gate.py
"""Compare DRAG / CRAB / GRAPE on an X gate, with and without a distorting line."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import Chain, TransferFunction
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.crab import CrabOptimizer
from pulselab.optimize.grape import GrapeOptimizer
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage

X = np.array([[0, 1], [1, 0]], dtype=complex)


def report(name, prob, pulse):
    U = prob.propagated(pulse)
    print(f"{name:24s} F={avg_gate_fidelity(U, X):.5f}  leak={leakage(U):.3e}")


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    base = gaussian_drag(40, amp, 8.0, 0.0, model.anharmonicity_ghz())

    for label, hw in [("ideal line", None),
                      ("low-pass tau=15ns", Chain([TransferFunction.single_pole_lowpass(15.0, 1.0)]))]:
        prob = Problem(model, target=X, drive_freq_ghz=model.f01_ghz(),
                       hardware=hw, leakage_weight=20.0)
        print(f"\n=== {label} ===")
        report("bare gaussian", prob, base)
        report("DRAG", prob, DragOptimizer(40, 8.0, model.anharmonicity_ghz())
               .run(prob, init_amp=amp).best_pulse)
        report("CRAB", prob, CrabOptimizer(base, n_harmonics=3).run(prob).best_pulse)
        if hw is None or hw.jacobian(base) is not None:
            report("GRAPE", prob, GrapeOptimizer().run(prob, init_pulse=base, maxiter=120).best_pulse)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + demo**

Run: `python -m pytest tests/test_optimizer_integration.py -v`
Expected: PASS (1 passed).
Run: `PYTHONPATH=. python examples/optimize_x_gate.py`
Expected: GRAPE achieves the highest fidelity / lowest leakage in both line conditions; under the low-pass, GRAPE clearly beats the bare gaussian and DRAG.

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/tests/test_optimizer_integration.py lab_pulse_opt/examples/optimize_x_gate.py
git commit -m "test(optimize): GRAPE pre-distortion integration + comparison demo"
```

---

### Task 8: Full Phase-3 suite sweep

**Files:** none (verification task).

- [ ] **Step 1: Run the whole suite**

Run (from `lab_pulse_opt/`): `python -m pytest -q`
Expected: all Phase-1 + Phase-2 + Phase-3 tests PASS together.

- [ ] **Step 2: Commit (only if fixups were needed)**

```bash
git add -A lab_pulse_opt
git commit -m "chore(pulselab): phase-3 optimizers green"
```

---

## Subsequent Phases (separate plans)

- **Phase 4 — Streamlit app:** parameter panels with inline explanations, per-stage hardware on/off toggles, ideal-vs-distorted overlays, live plots (envelope, populations/Bloch, leakage, spectrum, optimizer convergence, AllXY), compare mode, one-click `to_opx_waveforms` export. Consumes the `Problem` + optimizers built here.
- **Phase 5 — Readout noise + robust cost:** `metrics/measurement.py` (shot + amplifier noise → noisy measured cost; AllXY/RB evaluators) and robust/ensemble evaluation (optimize over `ControlNoise` + hardware-parameter ensembles), reusing the derivative-free optimizers for the non-differentiable noisy cost.

---

## Self-Review

**Spec coverage (design spec §2/§4/§6 optimizer portion):**
- `Problem` bundling model + hardware + target + metric → cost ✓ Task 1.
- Analytic DRAG optimizer ✓ Task 2.
- CRAB basis-expansion optimizer ✓ Task 3.
- GRAPE with analytic gradients backpropagated through linear hardware stages ✓ Tasks 4–5 (gradient verified to ~1e-9 vs finite difference, both with and without hardware, during planning).
- Pluggable `Optimizer` interface (the extension point for new methods) ✓ Task 1.
- Targets: X gate used throughout; arbitrary 2×2 target supported by `Problem.target` and the gradient (`a = tr(target†M)`).
- Cost = 1 − avg gate fidelity + leakage penalty ✓ Global Constraints, used identically in `Problem` and `cost_grad_distorted`.
- OPX export of optimized I/Q waveforms ✓ Task 6.
- GRAPE correctly refuses non-differentiable (stochastic) hardware and points to derivative-free optimizers ✓ Task 5.

**Placeholder scan:** No TBD/TODO; every code step has complete, runnable code. The GRAPE gradient code is the exact implementation validated against finite differences during planning.

**Type consistency:** `Problem.cost_from_pulse(pulse) -> float` and `Problem.propagated(pulse) -> ndarray` used consistently; `cost_grad_distorted(problem, I, Q, dt) -> (float, ndarray[2N])` consumed by `GrapeOptimizer.run`; `OptResult(best_pulse, best_cost, history, n_iter)` returned by every optimizer; `to_opx_waveforms(pulse, dac_per_radns) -> {"I_wf","Q_wf"}`. The `2N` stacked-sample convention and the `chain.jacobian(...) -> (2N,2N) | None` contract match Phase 2.
