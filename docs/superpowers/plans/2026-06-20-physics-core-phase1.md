# Single-Transmon Pulse Lab — Phase 1: Physics Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, fully-tested physics core that simulates a single flux-tunable transmon driven by a microwave I/Q pulse, and reproduces the "DRAG suppresses leakage" result on the real Q1 device parameters.

**Architecture:** A `pulselab` Python package with four independent units — `device` (parameters + Hamiltonian models), `pulse` (envelope parametrizations), `dynamics` (coherent time evolution), `metrics` (fidelity + leakage). Two interchangeable Hamiltonian models implement one `TransmonModel` interface: an exact charge-basis (Cooper-pair-box) model and a light Duffing model. Everything works in the rotating frame under RWA, in angular-frequency units (rad/ns) and time in ns.

**Tech Stack:** Python ≥3.10, numpy, scipy, pytest. (Streamlit/plotly arrive in Phase 4.)

## Global Constraints

- Package directory: `lab_pulse_opt/`, importable package name `pulselab`.
- Units everywhere: **angular frequency in rad/ns, time in ns, linear frequency stored in GHz**. Conversion: `omega_radns = 2*pi*f_GHz`.
- Device ground truth (Q1, from `opx/20260501_iMET_v1_2_SQUID_Q1/OPX_project/configuration.py`): **f01 = 5.252 GHz** (LO 5.5 − IF 0.24779), **anharmonicity = −0.064 GHz**, **T1 = 15 µs**, gate length **40 ns**, Gaussian σ = len/5.
- Drive convention: complex envelope `Omega(t) = I(t) + i*Q(t)` in rad/ns; a resonant I-only pulse rotates the qubit by angle `theta = ∫ I(t) dt` (π-pulse ⇔ area π).
- DRAG convention (matches `qualang_tools.drag_gaussian_pulse_waveforms`): `Q(t) = drag_coef * (-dI/dt) / anharmonicity_radns`.
- All angular-frequency arrays are level-indexed from the ground state: `omega[0] = 0`.
- TDD: every task is failing-test-first. Commit after each task. Run tests from the `lab_pulse_opt/` directory.

---

## File Structure

```
lab_pulse_opt/
  pyproject.toml                  # package config (Task 1)
  pulselab/
    __init__.py                   # (Task 1)
    units.py                      # GHz<->rad/ns helpers (Task 1)
    device/
      __init__.py
      params.py                   # DeviceParams dataclass + presets/solvers (Tasks 2, 6)
      hamiltonian.py              # TransmonModel ABC, ChargeBasisTransmon, DuffingTransmon (Tasks 3,4,5,7)
    pulse/
      __init__.py
      envelope.py                 # Pulse + GaussianDrag (Task 8)
    dynamics/
      __init__.py
      propagator.py               # coherent piecewise-constant evolution (Task 9)
    metrics/
      __init__.py
      fidelity.py                 # avg gate fidelity + leakage (Task 10)
  tests/
    test_units.py
    test_device_params.py
    test_charge_basis.py
    test_drive_couplings.py
    test_model_interface.py
    test_duffing.py
    test_envelope.py
    test_propagator.py
    test_fidelity.py
    test_drag_leakage.py          # integration anchor (Task 11)
  examples/
    drag_vs_gaussian.py           # headless demo (Task 11)
```

---

### Task 1: Package skeleton + unit helpers

**Files:**
- Create: `lab_pulse_opt/pyproject.toml`
- Create: `lab_pulse_opt/pulselab/__init__.py`
- Create: `lab_pulse_opt/pulselab/units.py`
- Create: `lab_pulse_opt/pulselab/device/__init__.py`, `pulse/__init__.py`, `dynamics/__init__.py`, `metrics/__init__.py` (empty)
- Test: `lab_pulse_opt/tests/test_units.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pulselab.units.ghz_to_radns(f_ghz) -> float|ndarray`, `pulselab.units.radns_to_ghz(w) -> float|ndarray`. Both accept scalars or numpy arrays.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_units.py
import numpy as np
from pulselab.units import ghz_to_radns, radns_to_ghz


def test_ghz_to_radns_scalar():
    assert np.isclose(ghz_to_radns(1.0), 2 * np.pi)


def test_roundtrip_array():
    f = np.array([0.0, 5.252, -0.064])
    assert np.allclose(radns_to_ghz(ghz_to_radns(f)), f)
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `lab_pulse_opt/`): `python -m pytest tests/test_units.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab'`.

- [ ] **Step 3: Write minimal implementation**

```toml
# lab_pulse_opt/pyproject.toml
[project]
name = "pulselab"
version = "0.1.0"
description = "Interactive single-transmon microwave pulse optimization lab."
requires-python = ">=3.10"
dependencies = ["numpy", "scipy"]

[project.optional-dependencies]
test = ["pytest"]
app = ["streamlit", "plotly"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```python
# lab_pulse_opt/pulselab/__init__.py
"""pulselab: single-transmon microwave pulse optimization lab."""
```

```python
# lab_pulse_opt/pulselab/units.py
import numpy as np

TWO_PI = 2 * np.pi


def ghz_to_radns(f_ghz):
    """Linear frequency in GHz -> angular frequency in rad/ns."""
    return TWO_PI * np.asarray(f_ghz, dtype=float) if np.ndim(f_ghz) else TWO_PI * f_ghz


def radns_to_ghz(omega):
    """Angular frequency in rad/ns -> linear frequency in GHz."""
    return np.asarray(omega, dtype=float) / TWO_PI if np.ndim(omega) else omega / TWO_PI
```

Create the empty `__init__.py` files:

```python
# lab_pulse_opt/pulselab/device/__init__.py
# lab_pulse_opt/pulselab/pulse/__init__.py
# lab_pulse_opt/pulselab/dynamics/__init__.py
# lab_pulse_opt/pulselab/metrics/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_units.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pyproject.toml lab_pulse_opt/pulselab lab_pulse_opt/tests/test_units.py
git commit -m "feat(pulselab): package skeleton + unit helpers"
```

---

### Task 2: DeviceParams dataclass

**Files:**
- Create: `lab_pulse_opt/pulselab/device/params.py`
- Test: `lab_pulse_opt/tests/test_device_params.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DeviceParams` frozen dataclass with fields `EC_ghz: float`, `EJ_ghz: float`, `ng: float = 0.0`, `asymmetry: float = 0.0`, `flux: float = 0.0` (in units of Φ0), `ncut: int = 25`, `n_levels: int = 5`, `T1_us: float = 15.0`, `Tphi_us: float = 30.0`. Method `EJ_effective_ghz() -> float` applies SQUID flux tuning: `EJ * sqrt(cos(pi*flux)^2 + asymmetry^2 * sin(pi*flux)^2)`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_device_params.py
import numpy as np
from pulselab.device.params import DeviceParams


def test_defaults_and_fields():
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0)
    assert p.n_levels == 5
    assert p.ncut == 25
    assert p.flux == 0.0
    assert np.isclose(p.EJ_effective_ghz(), 55.0)


def test_flux_tuning_at_half_quantum_symmetric():
    # Symmetric SQUID (asymmetry=0) at flux=0.5 -> EJ tunes to 0.
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, flux=0.5, asymmetry=0.0)
    assert np.isclose(p.EJ_effective_ghz(), 0.0, atol=1e-9)


def test_flux_tuning_asymmetry_floor():
    # Asymmetry sets a nonzero floor at flux=0.5.
    p = DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, flux=0.5, asymmetry=0.1)
    assert np.isclose(p.EJ_effective_ghz(), 55.0 * 0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_params.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/device/params.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DeviceParams:
    """Static parameters of a single flux-tunable transmon.

    Energies EC, EJ are stored as linear frequencies in GHz (i.e. E/h).
    flux is in units of the flux quantum Phi0.
    """

    EC_ghz: float
    EJ_ghz: float
    ng: float = 0.0
    asymmetry: float = 0.0
    flux: float = 0.0
    ncut: int = 25
    n_levels: int = 5
    T1_us: float = 15.0
    Tphi_us: float = 30.0

    def EJ_effective_ghz(self) -> float:
        """SQUID flux-tuned Josephson energy at the current flux bias."""
        phi = np.pi * self.flux
        d = self.asymmetry
        return self.EJ_ghz * np.sqrt(np.cos(phi) ** 2 + d ** 2 * np.sin(phi) ** 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_params.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/params.py lab_pulse_opt/tests/test_device_params.py
git commit -m "feat(device): DeviceParams dataclass with SQUID flux tuning"
```

---

### Task 3: Charge-basis Hamiltonian spectrum

**Files:**
- Create: `lab_pulse_opt/pulselab/device/hamiltonian.py`
- Test: `lab_pulse_opt/tests/test_charge_basis.py`

**Interfaces:**
- Consumes: `DeviceParams`.
- Produces: `ChargeBasisTransmon(params: DeviceParams)` with `_eigensystem() -> (evals_ghz: ndarray, evecs: ndarray)` returning ascending eigenvalues (GHz) and eigenvectors (columns) of the charge-basis Hamiltonian, and `frequencies() -> ndarray` returning the first `n_levels` transition angular frequencies relative to ground (rad/ns), `frequencies()[0] == 0`. Also `f01_ghz() -> float` and `anharmonicity_ghz() -> float`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_charge_basis.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_frequencies_shape_and_ground_zero():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.25, EJ_ghz=15.0, n_levels=5))
    w = m.frequencies()
    assert w.shape == (5,)
    assert w[0] == 0.0
    assert np.all(np.diff(w) > 0)  # ascending


def test_perturbative_f01_and_anharmonicity():
    # Transmon perturbative: f01 ~ sqrt(8*EC*EJ) - EC ; anharm ~ -EC.
    EC, EJ = 0.25, 20.0
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=EC, EJ_ghz=EJ))
    f01_approx = np.sqrt(8 * EC * EJ) - EC
    assert np.isclose(m.f01_ghz(), f01_approx, rtol=0.02)
    assert np.isclose(m.anharmonicity_ghz(), -EC, rtol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_charge_basis.py -v`
Expected: FAIL — `ImportError: cannot import name 'ChargeBasisTransmon'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/device/hamiltonian.py
import numpy as np
from ..units import ghz_to_radns
from .params import DeviceParams


class ChargeBasisTransmon:
    """Exact transmon: diagonalize H = 4 EC (n - ng)^2 - EJ cos(phi)
    in the charge basis |m>, m in [-ncut, ncut]."""

    def __init__(self, params: DeviceParams):
        self.params = params

    def _hamiltonian_ghz(self) -> np.ndarray:
        p = self.params
        ncut = p.ncut
        m = np.arange(-ncut, ncut + 1)
        dim = m.size
        H = np.zeros((dim, dim), dtype=float)
        # Charging term (diagonal).
        H[np.arange(dim), np.arange(dim)] = 4 * p.EC_ghz * (m - p.ng) ** 2
        # Josephson term: -EJ/2 on first off-diagonals (cos phi couples |m>,|m+1>).
        ej = p.EJ_effective_ghz()
        off = -ej / 2 * np.ones(dim - 1)
        H[np.arange(dim - 1), np.arange(1, dim)] = off
        H[np.arange(1, dim), np.arange(dim - 1)] = off
        return H

    def _eigensystem(self):
        evals, evecs = np.linalg.eigh(self._hamiltonian_ghz())
        return evals, evecs  # ascending (eigh guarantees)

    def frequencies(self) -> np.ndarray:
        evals, _ = self._eigensystem()
        levels = evals[: self.params.n_levels]
        return ghz_to_radns(levels - levels[0])

    def f01_ghz(self) -> float:
        evals, _ = self._eigensystem()
        return float(evals[1] - evals[0])

    def anharmonicity_ghz(self) -> float:
        evals, _ = self._eigensystem()
        return float((evals[2] - evals[1]) - (evals[1] - evals[0]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_charge_basis.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/hamiltonian.py lab_pulse_opt/tests/test_charge_basis.py
git commit -m "feat(device): exact charge-basis transmon spectrum"
```

---

### Task 4: Drive couplings (charge matrix elements)

**Files:**
- Modify: `lab_pulse_opt/pulselab/device/hamiltonian.py`
- Test: `lab_pulse_opt/tests/test_drive_couplings.py`

**Interfaces:**
- Consumes: `ChargeBasisTransmon._eigensystem`.
- Produces: `ChargeBasisTransmon.drive_couplings() -> ndarray` of length `n_levels - 1`, where entry `j` is the nearest-neighbor charge matrix element `<j+1|n|j>` normalized so `drive_couplings()[0] == 1.0`. For a deep transmon these approach `sqrt(j+1)`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_drive_couplings.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_couplings_normalized_and_harmonic_limit():
    # Deep transmon (large EJ/EC) -> couplings approach sqrt(j+1).
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.2, EJ_ghz=50.0, n_levels=4))
    g = m.drive_couplings()
    assert g.shape == (3,)
    assert np.isclose(g[0], 1.0)
    expected = np.sqrt([2.0, 3.0])  # g[1]/g[0], g[2]/g[0] ~ sqrt(2), sqrt(3)
    assert np.allclose(g[1:], expected, rtol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drive_couplings.py -v`
Expected: FAIL — `AttributeError: 'ChargeBasisTransmon' object has no attribute 'drive_couplings'`.

- [ ] **Step 3: Write minimal implementation**

Add to `ChargeBasisTransmon` in `hamiltonian.py`:

```python
    def drive_couplings(self) -> np.ndarray:
        """Nearest-neighbor charge matrix elements <j+1|n|j>, normalized to g0=1."""
        evals, evecs = self._eigensystem()
        ncut = self.params.ncut
        m = np.arange(-ncut, ncut + 1, dtype=float)
        n_levels = self.params.n_levels
        v = evecs[:, :n_levels]  # columns = eigenvectors
        # n is diagonal in charge basis: <i|n|j> = sum_m m * v_mi * v_mj.
        elems = np.array(
            [float((v[:, j] * m * v[:, j + 1]).sum()) for j in range(n_levels - 1)]
        )
        return elems / elems[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drive_couplings.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/hamiltonian.py lab_pulse_opt/tests/test_drive_couplings.py
git commit -m "feat(device): nearest-neighbor charge drive couplings"
```

---

### Task 5: TransmonModel interface + rotating-frame operators

**Files:**
- Modify: `lab_pulse_opt/pulselab/device/hamiltonian.py`
- Test: `lab_pulse_opt/tests/test_model_interface.py`

**Interfaces:**
- Consumes: `frequencies()`, `drive_couplings()`.
- Produces: an ABC `TransmonModel` with abstract `frequencies()` and `drive_couplings()` and `n_levels` property, plus a concrete mixin method `rotating_frame_operators(drive_freq_ghz) -> (H0_rot, X_op, Y_op)`:
  - `H0_rot = diag(omega_j - j * omega_d)` (rad/ns), `omega_d = ghz_to_radns(drive_freq_ghz)`.
  - With `A[j+1, j] = g_j` (lower-triangular raising operator): `X_op = (A + A.T)/2`, `Y_op = 1j*(A - A.T)/2`. All are `(n_levels, n_levels)` complex arrays; `X_op`, `Y_op` Hermitian.
  - `ChargeBasisTransmon` subclasses `TransmonModel`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_model_interface.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon, TransmonModel


def test_is_transmon_model_and_n_levels():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=5))
    assert isinstance(m, TransmonModel)
    assert m.n_levels == 5


def test_rotating_frame_operators_properties():
    m = ChargeBasisTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=4))
    f01 = m.f01_ghz()
    H0, X, Y = m.rotating_frame_operators(drive_freq_ghz=f01)
    assert H0.shape == (4, 4) and X.shape == (4, 4) and Y.shape == (4, 4)
    # On resonance with |0>-|1>, the 0,1 diagonal entries of H0_rot are ~equal.
    assert np.isclose(H0[0, 0].real, H0[1, 1].real, atol=1e-9)
    # Hermiticity of drive operators.
    assert np.allclose(X, X.conj().T)
    assert np.allclose(Y, Y.conj().T)
    # X couples neighbors with g0=1 -> X[1,0] == 0.5.
    assert np.isclose(X[1, 0], 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_interface.py -v`
Expected: FAIL — `ImportError: cannot import name 'TransmonModel'`.

- [ ] **Step 3: Write minimal implementation**

Add near the top of `hamiltonian.py` (after imports), and make `ChargeBasisTransmon` inherit it:

```python
from abc import ABC, abstractmethod


class TransmonModel(ABC):
    """Interface shared by all transmon Hamiltonian models."""

    @property
    @abstractmethod
    def n_levels(self) -> int: ...

    @abstractmethod
    def frequencies(self) -> np.ndarray:
        """Level angular frequencies (rad/ns) relative to ground; [0]==0."""

    @abstractmethod
    def drive_couplings(self) -> np.ndarray:
        """Nearest-neighbor couplings g_j, length n_levels-1, g0==1."""

    def rotating_frame_operators(self, drive_freq_ghz):
        """Return (H0_rot, X_op, Y_op) for H(t)=H0_rot + I(t)*X_op + Q(t)*Y_op."""
        omega = self.frequencies()
        d = self.n_levels
        levels = np.arange(d)
        omega_d = ghz_to_radns(drive_freq_ghz)
        H0_rot = np.diag(omega - levels * omega_d).astype(complex)
        g = self.drive_couplings()
        A = np.zeros((d, d), dtype=complex)
        A[np.arange(1, d), np.arange(d - 1)] = g  # A[j+1, j] = g_j
        X_op = (A + A.T) / 2
        Y_op = 1j * (A - A.T) / 2
        return H0_rot, X_op, Y_op
```

Change the class declaration to `class ChargeBasisTransmon(TransmonModel):` and add the `n_levels` property:

```python
    @property
    def n_levels(self) -> int:
        return self.params.n_levels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_interface.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/hamiltonian.py lab_pulse_opt/tests/test_model_interface.py
git commit -m "feat(device): TransmonModel interface + rotating-frame operators"
```

---

### Task 6: DeviceParams solver from device spectrum + Q1 preset

**Files:**
- Modify: `lab_pulse_opt/pulselab/device/params.py`
- Test: `lab_pulse_opt/tests/test_device_params.py` (extend)

**Interfaces:**
- Consumes: `ChargeBasisTransmon.f01_ghz`, `anharmonicity_ghz` (imported lazily inside the function to avoid a circular import).
- Produces: classmethod `DeviceParams.from_spectrum(f01_ghz, anharmonicity_ghz, **kwargs) -> DeviceParams` that solves for `EC_ghz, EJ_ghz` so the *exact* diagonalized spectrum matches the targets; and `DeviceParams.q1() -> DeviceParams` returning the Q1 preset (`from_spectrum(5.252, -0.064, T1_us=15.0)`).

- [ ] **Step 1: Write the failing test**

```python
# append to lab_pulse_opt/tests/test_device_params.py
from pulselab.device.hamiltonian import ChargeBasisTransmon


def test_from_spectrum_roundtrip():
    p = DeviceParams.from_spectrum(f01_ghz=5.252, anharmonicity_ghz=-0.064)
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
    assert np.isclose(m.anharmonicity_ghz(), -0.064, atol=1e-3)


def test_q1_preset():
    p = DeviceParams.q1()
    assert p.T1_us == 15.0
    m = ChargeBasisTransmon(p)
    assert np.isclose(m.f01_ghz(), 5.252, atol=1e-3)
    assert np.isclose(m.anharmonicity_ghz(), -0.064, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_device_params.py -v`
Expected: FAIL — `AttributeError: type object 'DeviceParams' has no attribute 'from_spectrum'`.

- [ ] **Step 3: Write minimal implementation**

Add to `DeviceParams` in `params.py` (import scipy at top: `from scipy.optimize import fsolve`):

```python
    @classmethod
    def from_spectrum(cls, f01_ghz, anharmonicity_ghz, **kwargs):
        """Solve for (EC, EJ) so the exact spectrum matches f01 and anharmonicity."""
        from .hamiltonian import ChargeBasisTransmon

        # Perturbative initial guess: anharm ~ -EC, f01 ~ sqrt(8 EC EJ) - EC.
        ec0 = -anharmonicity_ghz
        ej0 = (f01_ghz + ec0) ** 2 / (8 * ec0)

        def residual(x):
            ec, ej = x
            m = ChargeBasisTransmon(cls(EC_ghz=ec, EJ_ghz=ej, **kwargs))
            return [m.f01_ghz() - f01_ghz, m.anharmonicity_ghz() - anharmonicity_ghz]

        ec, ej = fsolve(residual, [ec0, ej0])
        return cls(EC_ghz=float(ec), EJ_ghz=float(ej), **kwargs)

    @classmethod
    def q1(cls):
        """Preset for the real device Q1 (iMET v1.2 SQUID)."""
        return cls.from_spectrum(f01_ghz=5.252, anharmonicity_ghz=-0.064, T1_us=15.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_device_params.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/params.py lab_pulse_opt/tests/test_device_params.py
git commit -m "feat(device): solve EC/EJ from spectrum + Q1 preset"
```

---

### Task 7: DuffingTransmon (light model)

**Files:**
- Modify: `lab_pulse_opt/pulselab/device/hamiltonian.py`
- Test: `lab_pulse_opt/tests/test_duffing.py`

**Interfaces:**
- Consumes: `TransmonModel`, `DeviceParams`.
- Produces: `DuffingTransmon(params)` subclassing `TransmonModel` with analytic `frequencies()[j] = ghz_to_radns(f01*j + 0.5*anharm*j*(j-1))` (using `f01 = sqrt(8 EC EJ_eff) - EC`, `anharm = -EC`) and `drive_couplings()[j] = sqrt(j+1)`. Same `rotating_frame_operators` via the inherited mixin.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_duffing.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import DuffingTransmon, ChargeBasisTransmon, TransmonModel


def test_duffing_is_model_and_couplings():
    m = DuffingTransmon(DeviceParams(EC_ghz=0.064, EJ_ghz=55.0, n_levels=4))
    assert isinstance(m, TransmonModel)
    assert np.allclose(m.drive_couplings(), np.sqrt([1.0, 2.0, 3.0]))


def test_duffing_close_to_charge_basis_levels():
    p = DeviceParams.from_spectrum(5.252, -0.064)
    duff = DuffingTransmon(p)
    exact = ChargeBasisTransmon(p)
    # f01 and anharmonicity agree to a few MHz in the deep-transmon regime.
    assert np.isclose(duff.f01_ghz(), exact.f01_ghz(), atol=5e-3)
    assert np.isclose(duff.anharmonicity_ghz(), exact.anharmonicity_ghz(), atol=5e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_duffing.py -v`
Expected: FAIL — `ImportError: cannot import name 'DuffingTransmon'`.

- [ ] **Step 3: Write minimal implementation**

Add to `hamiltonian.py`:

```python
class DuffingTransmon(TransmonModel):
    """Light anharmonic-oscillator (Kerr/Duffing) model. Fast; for live exploration."""

    def __init__(self, params: DeviceParams):
        self.params = params
        ec = params.EC_ghz
        ej = params.EJ_effective_ghz()
        self._f01 = np.sqrt(8 * ec * ej) - ec
        self._anharm = -ec

    @property
    def n_levels(self) -> int:
        return self.params.n_levels

    def f01_ghz(self) -> float:
        return float(self._f01)

    def anharmonicity_ghz(self) -> float:
        return float(self._anharm)

    def frequencies(self) -> np.ndarray:
        j = np.arange(self.n_levels)
        levels_ghz = self._f01 * j + 0.5 * self._anharm * j * (j - 1)
        return ghz_to_radns(levels_ghz)

    def drive_couplings(self) -> np.ndarray:
        j = np.arange(self.n_levels - 1)
        return np.sqrt(j + 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_duffing.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/device/hamiltonian.py lab_pulse_opt/tests/test_duffing.py
git commit -m "feat(device): light Duffing transmon model"
```

---

### Task 8: Pulse envelope + GaussianDrag parametrization

**Files:**
- Create: `lab_pulse_opt/pulselab/pulse/envelope.py`
- Test: `lab_pulse_opt/tests/test_envelope.py`

**Interfaces:**
- Consumes: nothing (uses an anharmonicity value passed in).
- Produces:
  - `Pulse` dataclass: `t: ndarray` (ns, length N), `I: ndarray` (rad/ns, length N), `Q: ndarray` (rad/ns, length N), with property `dt` (float, uniform step) and `area() -> float` returning `trapz(I, t)`.
  - `gaussian_drag(duration_ns, amp_radns, sigma_ns, drag_coef, anharmonicity_ghz, dt_ns=1.0) -> Pulse` where `I = amp*exp(-(t-t0)^2/(2 sigma^2))` (centered, `t0=duration/2`), and `Q = drag_coef * (-dI/dt) / anharm_radns`, `anharm_radns = ghz_to_radns(anharmonicity_ghz)`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_envelope.py
import numpy as np
from pulselab.pulse.envelope import Pulse, gaussian_drag
from pulselab.units import ghz_to_radns


def test_gaussian_no_drag_has_zero_Q():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=0.0, anharmonicity_ghz=-0.064)
    assert np.allclose(p.Q, 0.0)
    assert p.I.max() > 0
    assert np.isclose(p.dt, 1.0)


def test_drag_q_is_scaled_negative_derivative():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=1.0, anharmonicity_ghz=-0.064)
    anharm = ghz_to_radns(-0.064)
    expected_Q = -np.gradient(p.I, p.t) / anharm
    assert np.allclose(p.Q, expected_Q, atol=1e-6)


def test_area_matches_trapz():
    p = gaussian_drag(40, amp_radns=0.1, sigma_ns=8, drag_coef=0.0, anharmonicity_ghz=-0.064)
    assert np.isclose(p.area(), np.trapz(p.I, p.t))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_envelope.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.pulse.envelope'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/pulse/envelope.py
from dataclasses import dataclass
import numpy as np
from ..units import ghz_to_radns


@dataclass
class Pulse:
    """Microwave drive envelope on a uniform time grid. I,Q in rad/ns; t in ns."""

    t: np.ndarray
    I: np.ndarray
    Q: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def area(self) -> float:
        return float(np.trapz(self.I, self.t))


def gaussian_drag(duration_ns, amp_radns, sigma_ns, drag_coef,
                  anharmonicity_ghz, dt_ns=1.0):
    """Gaussian I envelope with first-order DRAG Q correction."""
    n = int(round(duration_ns / dt_ns))
    t = np.arange(n) * dt_ns
    t0 = duration_ns / 2.0
    I = amp_radns * np.exp(-((t - t0) ** 2) / (2 * sigma_ns ** 2))
    anharm_radns = ghz_to_radns(anharmonicity_ghz)
    Q = drag_coef * (-np.gradient(I, t)) / anharm_radns
    return Pulse(t=t, I=I, Q=Q)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_envelope.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/pulse/envelope.py lab_pulse_opt/tests/test_envelope.py
git commit -m "feat(pulse): Pulse envelope + Gaussian-DRAG parametrization"
```

---

### Task 9: Coherent propagator

**Files:**
- Create: `lab_pulse_opt/pulselab/dynamics/propagator.py`
- Test: `lab_pulse_opt/tests/test_propagator.py`

**Interfaces:**
- Consumes: `TransmonModel.rotating_frame_operators`, `Pulse`.
- Produces: `propagate(model, pulse, drive_freq_ghz) -> ndarray` returning the `(d, d)` complex unitary `U = prod_k expm(-1j * H_k * dt)` over time slices, with `H_k = H0_rot + I_k * X_op + Q_k * Y_op` (mid-point sample of I,Q). Uses `scipy.linalg.expm`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_propagator.py
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag, Pulse
from pulselab.dynamics.propagator import propagate


def test_zero_drive_is_identity_up_to_phase():
    m = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064))
    t = np.arange(40) * 1.0
    zero = Pulse(t=t, I=np.zeros(40), Q=np.zeros(40))
    U = propagate(m, zero, drive_freq_ghz=m.f01_ghz())
    # On resonance, |0> and |1> have zero rotating-frame energy -> identity block.
    assert np.isclose(abs(U[0, 0]), 1.0, atol=1e-6)
    assert np.isclose(abs(U[1, 1]), 1.0, atol=1e-6)


def test_resonant_pi_pulse_inverts_population():
    m = ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064))
    # Constant square I-pulse with area pi -> full |0>->|1> inversion.
    duration = 40.0
    amp = np.pi / duration
    t = np.arange(int(duration)) * 1.0
    sq = Pulse(t=t, I=amp * np.ones(t.size), Q=np.zeros(t.size))
    U = propagate(m, sq, drive_freq_ghz=m.f01_ghz())
    p1 = abs(U[1, 0]) ** 2  # population transferred |0> -> |1>
    assert p1 > 0.97
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_propagator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.dynamics.propagator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/dynamics/propagator.py
import numpy as np
from scipy.linalg import expm


def propagate(model, pulse, drive_freq_ghz):
    """Coherent piecewise-constant propagation in the rotating frame.

    Returns the (d, d) unitary U = prod_k expm(-i H_k dt).
    """
    H0, X_op, Y_op = model.rotating_frame_operators(drive_freq_ghz)
    dt = pulse.dt
    d = H0.shape[0]
    U = np.eye(d, dtype=complex)
    for Ik, Qk in zip(pulse.I, pulse.Q):
        Hk = H0 + Ik * X_op + Qk * Y_op
        U = expm(-1j * Hk * dt) @ U
    return U
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_propagator.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/dynamics/propagator.py lab_pulse_opt/tests/test_propagator.py
git commit -m "feat(dynamics): coherent piecewise-constant propagator"
```

---

### Task 10: Fidelity + leakage metrics

**Files:**
- Create: `lab_pulse_opt/pulselab/metrics/fidelity.py`
- Test: `lab_pulse_opt/tests/test_fidelity.py`

**Interfaces:**
- Consumes: a full propagator `U` (d×d) and a 2×2 target unitary.
- Produces:
  - `subspace_block(U) -> ndarray`: the top-left 2×2 block `U[:2, :2]`.
  - `leakage(U) -> float`: population leaving the computational subspace, `1 - (|U[0,0]|^2+|U[1,0]|^2+|U[0,1]|^2+|U[1,1]|^2)/2`.
  - `avg_gate_fidelity(U, target) -> float`: average gate fidelity of the 2×2 block vs `target`, using `F = (|tr(target^dag M)|^2 + tr(M^dag M)) / (d*(d+1))` with `d=2`, `M = U[:2,:2]`.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_fidelity.py
import numpy as np
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage, subspace_block

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_identity_block_unit_fidelity_zero_leakage():
    U = np.eye(5, dtype=complex)
    assert np.isclose(avg_gate_fidelity(U, I2), 1.0)
    assert np.isclose(leakage(U), 0.0)


def test_perfect_x_gate():
    U = np.zeros((5, 5), dtype=complex)
    U[0, 1] = U[1, 0] = 1.0  # X on the qubit subspace
    U[2, 2] = U[3, 3] = U[4, 4] = 1.0
    assert np.isclose(avg_gate_fidelity(U, X), 1.0)
    assert np.isclose(leakage(U), 0.0)


def test_leakage_detected():
    # Half of |0> leaks to |2>.
    U = np.eye(5, dtype=complex)
    U[0, 0] = np.sqrt(0.5)
    U[2, 0] = np.sqrt(0.5)
    assert leakage(U) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fidelity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pulselab.metrics.fidelity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/pulselab/metrics/fidelity.py
import numpy as np


def subspace_block(U):
    """Top-left 2x2 computational-subspace block of the propagator."""
    return U[:2, :2]


def leakage(U):
    """Population that leaves the {|0>,|1>} subspace, averaged over the two inputs."""
    M = subspace_block(U)
    retained = np.sum(np.abs(M) ** 2)
    return float(1.0 - retained / 2.0)


def avg_gate_fidelity(U, target):
    """Average gate fidelity of the 2x2 block vs a 2x2 target unitary."""
    M = subspace_block(U)
    d = 2
    t = np.trace(target.conj().T @ M)
    return float((np.abs(t) ** 2 + np.trace(M.conj().T @ M)).real / (d * (d + 1)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fidelity.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/pulselab/metrics/fidelity.py lab_pulse_opt/tests/test_fidelity.py
git commit -m "feat(metrics): average gate fidelity + leakage"
```

---

### Task 11: Integration anchor — DRAG suppresses leakage

**Files:**
- Create: `lab_pulse_opt/tests/test_drag_leakage.py`
- Create: `lab_pulse_opt/examples/drag_vs_gaussian.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the headline physics result — on the Q1 charge-basis model, a DRAG-corrected 40 ns π-pulse leaks less than the bare Gaussian, and a calibrated DRAG coefficient achieves low leakage. Also a runnable headless demo.

- [ ] **Step 1: Write the failing test**

```python
# lab_pulse_opt/tests/test_drag_leakage.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drag_leakage.py -v`
Expected: FAIL — `ModuleNotFoundError` (file not yet created) — confirm by creating the test first and running; it should fail to import until all prior tasks are committed, then assert-pass once implemented. (If prior tasks are done, the only failure mode is the assertion if physics is wrong.)

- [ ] **Step 3: Write minimal implementation**

```python
# lab_pulse_opt/examples/drag_vs_gaussian.py
"""Headless demo: DRAG suppresses leakage on the real Q1 transmon."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import leakage


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    probe = gaussian_drag(40, 1.0, 8.0, 0.0, model.anharmonicity_ghz())
    amp = np.pi / probe.area()
    for coef in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = gaussian_drag(40, amp, 8.0, coef, model.anharmonicity_ghz())
        U = propagate(model, p, model.f01_ghz())
        print(f"drag_coef={coef:.2f}  leakage={leakage(U):.3e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + demo to verify they pass**

Run: `python -m pytest tests/test_drag_leakage.py -v`
Expected: PASS (1 passed).
Run: `python examples/drag_vs_gaussian.py`
Expected: prints decreasing-then-minimum leakage as `drag_coef` increases (a clear minimum below the `drag_coef=0` value).

- [ ] **Step 5: Commit**

```bash
git add lab_pulse_opt/tests/test_drag_leakage.py lab_pulse_opt/examples/drag_vs_gaussian.py
git commit -m "test(physics): DRAG-suppresses-leakage integration anchor"
```

---

### Task 12: Full Phase-1 test sweep

**Files:** none (verification task).

- [ ] **Step 1: Run the whole suite**

Run (from `lab_pulse_opt/`): `python -m pytest -v`
Expected: all tests PASS (every task's tests green together).

- [ ] **Step 2: Commit (only if any fixups were needed)**

```bash
git add -A lab_pulse_opt
git commit -m "chore(pulselab): phase-1 physics core green"
```

---

## Subsequent Phases (separate plans, written when reached)

These are intentionally **not** expanded into bite-sized tasks yet — they depend on the
physics-core interfaces solidifying first, and each becomes its own plan via the
writing-plans skill.

- **Phase 2 — Hardware chain + open system:** `pulse/hardware.py` (TransferFunction
  FIR/IIR, BiasTeeDroop, IQImbalance + LO leakage, ControlNoise) with a composable
  `Chain`; linear stages expose `jacobian()`. `dynamics/lindblad.py` master-equation
  evolution with collapse operators from `T1_us`, `Tphi_us`.
- **Phase 3 — Optimizers:** `optimize/base.py` (`Problem`, `Optimizer`), `drag.py`,
  `grape.py` (analytic gradients through linear hardware stages), `crab.py`;
  `export/opx.py` exporting optimized I/Q arrays in OPX `configuration.py` form.
- **Phase 4 — Streamlit app:** grouped parameter panels with inline explanations,
  per-stage on/off toggles, ideal-vs-distorted overlays, live plots
  (envelope, populations/Bloch, leakage, spectrum, optimizer convergence, AllXY),
  compare mode, pulse export.
- **Phase 5 — Readout noise + robust cost:** `metrics/measurement.py` (shot +
  amplifier noise → noisy measured cost; AllXY/RB evaluators) and robust/ensemble
  evaluation in `optimize/`.

---

## Self-Review

**Spec coverage (Phase 1 portion):** exact charge-basis model ✓ (Tasks 3–6), light
Duffing model ✓ (Task 7), rotating-frame I/Q drive ✓ (Task 5), coherent dynamics ✓
(Task 9), fidelity + leakage metrics ✓ (Task 10), Q1 seeding ✓ (Task 6), DRAG-leakage
validation anchor ✓ (Task 11). Hardware chain, Lindblad, optimizers, UI, readout noise
are deferred to Phases 2–5 by design (physics-core-first, per approved spec §9).

**Placeholder scan:** No TBD/TODO; every code step contains complete, runnable code.

**Type consistency:** `frequencies()` returns rad/ns array `[0]==0` (Tasks 3,5,7);
`drive_couplings()` length `n_levels-1`, `[0]==1` (Tasks 4,7); `rotating_frame_operators`
returns `(H0_rot, X_op, Y_op)` consumed identically in Task 9; `Pulse` fields `t,I,Q`
with `dt`/`area()` used consistently in Tasks 8,9,11; fidelity functions operate on the
`U[:2,:2]` block consistently in Tasks 10,11.
