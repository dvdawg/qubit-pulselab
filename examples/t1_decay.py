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
        n = max(int(t_total), 2)
        zero = Pulse(t=np.arange(n) * 1.0, I=np.zeros(n), Q=np.zeros(n))
        rho = lindblad_propagate(model, zero, model.f01_ghz(), rho0, c_ops)
        print(f"t={t_total:6.1f}ns  P1={rho[1,1].real:.4f}  (exp={np.exp(-t_total/100):.4f})")


if __name__ == "__main__":
    main()
