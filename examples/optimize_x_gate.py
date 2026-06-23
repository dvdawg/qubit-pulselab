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
