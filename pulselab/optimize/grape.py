import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from ..metrics.fidelity import avg_gate_fidelity, leakage
from ..pulse.envelope import Pulse
from .base import Optimizer, OptResult


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


class GrapeOptimizer(Optimizer):
    """Gradient-based piecewise-constant optimization with hardware backprop.

    When the hardware chain exposes an exact Jacobian (linear/affine stages), the
    analytic gradient is backpropagated through it. When it does not (e.g. a
    stochastic ControlNoise stage), GRAPE falls back to scipy's finite-difference
    gradient -- which is valid as long as the cost is deterministic (noise stages
    must be seeded). Set allow_numerical=False to instead raise on such chains.
    """

    def run(self, problem, init_pulse, maxiter=200, allow_numerical=True):
        t = init_pulse.t
        dt = init_pulse.dt
        N = init_pulse.I.size
        history = []
        differentiable = problem.hardware.jacobian(init_pulse) is not None

        if not differentiable and not allow_numerical:
            raise ValueError(
                "GRAPE requires differentiable hardware (jacobian is None); "
                "use a derivative-free optimizer (DRAG/CRAB), or allow_numerical=True.")

        x0 = np.concatenate([init_pulse.I, init_pulse.Q])

        if differentiable:
            def cost_and_grad(x):
                ideal = Pulse(t=t, I=x[:N], Q=x[N:])
                distorted = problem.hardware.apply(ideal)
                cost, grad_dist = cost_grad_distorted(problem, distorted.I, distorted.Q, dt)
                J = problem.hardware.jacobian(ideal)
                history.append(cost)
                return cost, J.T @ grad_dist

            res = minimize(cost_and_grad, x0=x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter})
        else:
            # Non-differentiable (e.g. seeded noise): let scipy approximate the
            # gradient by finite differences of the deterministic cost.
            def cost_only(x):
                cost = problem.cost_from_pulse(Pulse(t=t, I=x[:N], Q=x[N:]))
                history.append(cost)
                return cost

            res = minimize(cost_only, x0=x0, method="L-BFGS-B",
                           options={"maxiter": maxiter})

        best = Pulse(t=t, I=res.x[:N], Q=res.x[N:])
        return OptResult(best_pulse=best, best_cost=float(res.fun),
                         history=history, n_iter=res.nit)
