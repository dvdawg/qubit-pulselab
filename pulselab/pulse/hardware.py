from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import lfilter
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


class BiasTeeDroop(HardwareStage):
    """Bias-tee / DC-block high-pass: blocks DC, so sustained levels droop."""

    def __init__(self, tau_ns, dt_ns):
        r = np.exp(-dt_ns / tau_ns)
        self._tf = TransferFunction(b=[1.0, -1.0], a=[1.0, -r])

    def apply(self, pulse: Pulse) -> Pulse:
        return self._tf.apply(pulse)

    def jacobian(self, pulse: Pulse):
        return self._tf.jacobian(pulse)


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
