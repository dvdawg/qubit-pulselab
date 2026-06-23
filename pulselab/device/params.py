from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve


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

        x, _info, ier, msg = fsolve(residual, [ec0, ej0], full_output=True)
        if ier != 1:
            raise ValueError(f"from_spectrum failed to converge: {msg}")
        ec, ej = x
        if ec <= 0 or ej <= 0:
            raise ValueError(f"from_spectrum converged to unphysical parameters: EC={ec}, EJ={ej}")
        return cls(EC_ghz=float(ec), EJ_ghz=float(ej), **kwargs)

    @classmethod
    def q1(cls):
        """Preset for the real device Q1 (iMET v1.2 SQUID)."""
        return cls.from_spectrum(f01_ghz=5.252, anharmonicity_ghz=-0.064, T1_us=15.0)
