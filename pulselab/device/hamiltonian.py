import numpy as np
import functools
from abc import ABC, abstractmethod
from ..units import ghz_to_radns
from .params import DeviceParams


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


class ChargeBasisTransmon(TransmonModel):
    """Exact transmon: diagonalize H = 4 EC (n - ng)^2 - EJ cos(phi)
    in the charge basis |m>, m in [-ncut, ncut]."""

    def __init__(self, params: DeviceParams):
        self.params = params

    @property
    def n_levels(self) -> int:
        return self.params.n_levels

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

    @functools.cached_property
    def _cached_eigensystem(self):
        evals, evecs = np.linalg.eigh(self._hamiltonian_ghz())
        return evals, evecs

    def _eigensystem(self):
        return self._cached_eigensystem

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

    def drive_couplings(self) -> np.ndarray:
        """Magnitude of nearest-neighbor charge matrix elements |<j+1|n|j>|, normalized to g0=1."""
        evals, evecs = self._eigensystem()
        ncut = self.params.ncut
        m = np.arange(-ncut, ncut + 1, dtype=float)
        n_levels = self.params.n_levels
        v = evecs[:, :n_levels]  # columns = eigenvectors
        # n is diagonal in charge basis: <i|n|j> = sum_m m * v_mi * v_mj.
        elems = np.array(
            [float((v[:, j] * m * v[:, j + 1]).sum()) for j in range(n_levels - 1)]
        )
        # eigh fixes each eigenvector's sign arbitrarily, so raw <j+1|n|j> carry
        # spurious relative signs. abs() gauge-fixes them (a diagonal +/-1 gauge that
        # leaves the computational-subspace block and leakage invariant) so the
        # raising operator built in rotating_frame_operators is consistently signed.
        return np.abs(elems) / np.abs(elems[0])
