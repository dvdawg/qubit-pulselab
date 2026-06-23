"""Optimize DRAG against the noisy *measured* cost instead of the exact proxy."""
import numpy as np
from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.metrics.measurement import MeasuredProblem
from pulselab.metrics.fidelity import avg_gate_fidelity

X = np.array([[0, 1], [1, 0]], dtype=complex)


def main():
    model = ChargeBasisTransmon(DeviceParams.q1())
    anh = model.anharmonicity_ghz()
    fd = model.f01_ghz()
    amp0 = np.pi / gaussian_drag(40, 1.0, 8.0, 0.0, anh).area()

    base = Problem(model, target=X, drive_freq_ghz=fd, leakage_weight=20.0)
    mp = MeasuredProblem(base, n_shots=50000, readout_fidelity=1.0, seed=7)

    bare = gaussian_drag(40, amp0, 8.0, 0.0, anh)
    bare_cost = mp.cost_from_pulse(bare)

    res = DragOptimizer(40, 8.0, anh).run(mp, init_amp=amp0, init_drag_coef=0.0)
    optimized = res.best_pulse

    U_opt = base.propagated(optimized)
    fidelity_opt = avg_gate_fidelity(U_opt, X)

    print(f"Bare pulse measured cost:      {bare_cost:.5f}")
    print(f"Optimized pulse measured cost: {res.best_cost:.5f}")
    print(f"Optimized pulse exact fidelity: {fidelity_opt:.5f}")


if __name__ == "__main__":
    main()
