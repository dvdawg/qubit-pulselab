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
