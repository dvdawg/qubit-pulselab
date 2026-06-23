import numpy as np
import streamlit as st
import plotly.graph_objects as go

from pulselab.device.params import DeviceParams
from pulselab.device.hamiltonian import ChargeBasisTransmon
from pulselab.pulse.envelope import gaussian_drag
from pulselab.pulse.hardware import (
    Chain, IdentityStage, TransferFunction, BiasTeeDroop, IQImbalance, ControlNoise)
from pulselab.dynamics.propagator import propagate
from pulselab.metrics.fidelity import avg_gate_fidelity, leakage
from pulselab.metrics.measurement import simulate_readout
from pulselab.optimize.base import Problem
from pulselab.optimize.drag import DragOptimizer
from pulselab.optimize.grape import GrapeOptimizer
from pulselab.optimize.crab import CrabOptimizer
from pulselab.export.opx import to_opx_waveforms
from pulselab import viz

X_TARGET = np.array([[0, 1], [1, 0]], dtype=complex)

st.set_page_config(page_title="Single-Transmon Pulse Lab", layout="wide")
st.title("Qubit Pulse Lab - Single Transmon")

# ---- Device panel ----
st.sidebar.header("Device")
n_levels = st.sidebar.slider(
    "Transmon levels", 2, 6, 4,
    help="How many transmon energy levels to simulate. More levels capture leakage "
         "to |2> and above more accurately, at higher compute cost.")
flux = st.sidebar.slider(
    "Flux bias (Phi/Phi0)", 0.0, 0.5, 0.0, 0.01,
    help="SQUID flux bias. 0 is the sweet spot (max frequency); moving toward 0.5 "
         "tunes the qubit frequency down and changes the anharmonicity.")
params = DeviceParams.from_spectrum(5.252, -0.064, n_levels=n_levels, flux=flux)
model = ChargeBasisTransmon(params)
drive_freq = model.f01_ghz()
anh = model.anharmonicity_ghz()

# ---- Pulse panel ----
st.sidebar.header("Pulse")
duration = st.sidebar.slider(
    "Duration (ns)", 8, 120, 40, 2,
    help="Total gate length. Shorter gates are faster but have wider bandwidth, "
         "which drives more leakage on a weakly-anharmonic transmon.")
sigma = st.sidebar.slider(
    "Gaussian sigma (ns)", 2.0, 30.0, 8.0, 0.5,
    help="Width of the Gaussian envelope. Narrower pulses ring more in frequency.")
probe = gaussian_drag(duration, 1.0, sigma, 0.0, anh)
amp_default = float(np.pi / probe.area())
amp = st.sidebar.slider(
    "Amplitude (rad/ns)", 0.0, 2 * amp_default, amp_default, amp_default / 50,
    help="Peak Rabi drive. Calibrated so the bare Gaussian is a pi (X180) pulse by default.")
drag_coef = st.sidebar.slider(
    "DRAG coefficient", -2.0, 2.0, 0.0, 0.05,
    help="DRAG adds a derivative component on the quadrature channel to cancel "
         "leakage to |2>. 0 is a plain Gaussian; the optimal value suppresses leakage.")
pulse = gaussian_drag(duration, amp, sigma, drag_coef, anh)

# ---- Hardware panel ----
st.sidebar.header("Hardware line")
stages = []
if st.sidebar.checkbox("Drive-line low-pass", value=False,
                       help="Finite drive-line bandwidth / rise-time, as a one-pole "
                            "low-pass. Larger tau = narrower bandwidth = more distortion."):
    tau_lp = st.sidebar.slider("  low-pass tau (ns)", 1.0, 60.0, 15.0, 1.0,
                               help="Low-pass time constant.")
    stages.append(TransferFunction.single_pole_lowpass(tau_lp, pulse.dt))
if st.sidebar.checkbox("Bias-tee droop", value=False,
                       help="DC-blocking bias-tee causes slow droop of sustained levels."):
    tau_bt = st.sidebar.slider("  bias-tee tau (ns)", 50.0, 5000.0, 1000.0, 50.0,
                               help="Droop time constant (large = slow droop).")
    stages.append(BiasTeeDroop(tau_bt, pulse.dt))
if st.sidebar.checkbox("IQ imbalance", value=False,
                       help="Mixer gain/phase imbalance and LO/carrier leakage."):
    g = st.sidebar.slider("  gain imbalance", -0.2, 0.2, 0.05, 0.01, help="I/Q gain mismatch.")
    ph = st.sidebar.slider("  phase error (rad)", -0.3, 0.3, 0.02, 0.01, help="I/Q phase error.")
    stages.append(IQImbalance(gain_imbalance=g, phase_error_rad=ph))
if st.sidebar.checkbox("Control noise", value=False,
                       help="Additive amplitude noise on the drive (stochastic; disables "
                            "gradient-based GRAPE)."):
    sig = st.sidebar.slider("  noise sigma (rad/ns)", 0.0, 0.05, 0.005, 0.001, help="Noise std.")
    stages.append(ControlNoise(sigma=sig, seed=0))
hardware = Chain(stages) if stages else IdentityStage()

# ---- Compute ----
distorted = hardware.apply(pulse)
U = propagate(model, distorted, drive_freq)
F = avg_gate_fidelity(U, X_TARGET)
L = leakage(U)

c1, c2, c3 = st.columns(3)
c1.metric("X-gate fidelity", f"{F:.5f}")
c2.metric("Leakage to |2+>", f"{L:.2e}")
c3.metric("Qubit freq (GHz)", f"{drive_freq:.4f}")

# ---- Readout (measured) ----
st.sidebar.header("Readout")
measured = st.sidebar.checkbox(
    "Simulate measured readout", value=False,
    help="Sample the X-gate excited-state population with finite shots and "
         "readout assignment error, like a real experiment, instead of the "
         "exact value.")
if measured:
    n_shots = st.sidebar.slider(
        "Shots", 10, 100000, 1000, 10,
        help="Number of projective single-shot measurements. More shots = less "
             "statistical (shot) noise on the measured population.")
    rofid = st.sidebar.slider(
        "Readout assignment fidelity", 0.5, 1.0, 0.97, 0.005,
        help="Probability the qubit state is correctly identified. 1.0 is perfect; "
             "lower values bias the measured population toward 0.5 (SPAM error).")
    p1_true = float(abs(U[1, 0]) ** 2)
    p1_meas = simulate_readout(p1_true, n_shots, rofid, seed=None)
    st.metric("Measured P(excited) of X gate", f"{p1_meas:.3f}",
              delta=f"{p1_meas - p1_true:+.3f} vs exact")

# ---- Envelope: ideal vs distorted ----
st.subheader("Drive envelope (ideal vs distorted)")
fig_env = go.Figure()
fig_env.add_scatter(x=pulse.t, y=pulse.I, name="I ideal", line=dict(color="royalblue"))
fig_env.add_scatter(x=pulse.t, y=pulse.Q, name="Q ideal", line=dict(color="firebrick"))
fig_env.add_scatter(x=distorted.t, y=distorted.I, name="I distorted",
                    line=dict(color="royalblue", dash="dash"))
fig_env.add_scatter(x=distorted.t, y=distorted.Q, name="Q distorted",
                    line=dict(color="firebrick", dash="dash"))
fig_env.update_layout(xaxis_title="t (ns)", yaxis_title="rad/ns", height=300)
st.plotly_chart(fig_env, use_container_width=True)

# ---- Populations + leakage ----
st.subheader("State populations")
t_edges, pops = viz.population_trajectory(model, distorted, drive_freq)
fig_pop = go.Figure()
for j in range(model.n_levels):
    fig_pop.add_scatter(x=t_edges, y=pops[:, j], name=f"|{j}>")
fig_pop.update_layout(xaxis_title="t (ns)", yaxis_title="population", height=300)
st.plotly_chart(fig_pop, use_container_width=True)

# ---- Spectrum ----
st.subheader("Drive spectrum")
freqs, power = viz.drive_spectrum(pulse)
fig_spec = go.Figure(go.Scatter(x=freqs * 1000, y=power))
fig_spec.update_layout(xaxis_title="detuning (MHz)", yaxis_title="power (norm.)", height=250)
st.plotly_chart(fig_spec, use_container_width=True)

# ---- AllXY ----
st.subheader("AllXY")
labels, p1 = viz.allxy_populations(
    ChargeBasisTransmon(DeviceParams.from_spectrum(5.252, -0.064, n_levels=2, flux=flux)),
    drive_freq, duration_ns=duration, sigma_ns=sigma)
fig_axy = go.Figure(go.Scatter(x=labels, y=p1, mode="lines+markers"))
fig_axy.update_layout(yaxis_title="P(excited)", height=300, xaxis_tickangle=-60)
st.plotly_chart(fig_axy, use_container_width=True)

# ---- Optimizer ----
st.sidebar.header("Optimizer")
method = st.sidebar.selectbox(
    "Method", ["(none)", "DRAG", "CRAB", "GRAPE"],
    help="DRAG/CRAB are derivative-free (work with any hardware, incl. noise). "
         "GRAPE is gradient-based; through differentiable hardware it backprops the "
         "exact gradient, and through non-differentiable (seeded-noise) chains it "
         "falls back to a slower numerical gradient.")
if method == "GRAPE" and hardware.jacobian(pulse) is None:
    st.sidebar.caption("⚠️ This hardware chain isn't differentiable; GRAPE will use a "
                       "slower numerical gradient.")
if st.sidebar.button("Run optimizer", help="Optimize the pulse against the current "
                     "device + hardware to maximize X-gate fidelity and suppress leakage. "
                     "The before/after comparison appears below the main panels."):
    problem = Problem(model, target=X_TARGET, drive_freq_ghz=drive_freq,
                      hardware=hardware, leakage_weight=20.0)
    try:
        if method == "DRAG":
            res = DragOptimizer(duration, sigma, anh).run(problem, init_amp=amp,
                                                          init_drag_coef=drag_coef)
        elif method == "CRAB":
            res = CrabOptimizer(pulse, n_harmonics=3).run(problem)
        elif method == "GRAPE":
            res = GrapeOptimizer().run(problem, init_pulse=pulse, maxiter=120)
        else:
            res = None
        if res is None:
            st.sidebar.warning("Pick an optimizer method first.")
        else:
            # Snapshot everything the comparison needs, so it survives reruns and
            # stays self-consistent even if the sidebar changes afterwards.
            st.session_state["opt"] = {
                "method": method, "history": list(res.history),
                "before": pulse, "after": res.best_pulse,
                "model": model, "hardware": hardware, "drive_freq": drive_freq,
                "duration": duration, "sigma": sigma, "flux": flux,
                "before_params": {"amp": amp, "drag_coef": drag_coef},
                "after_params": getattr(res, "params", None)}
    except ValueError as e:
        st.sidebar.error(str(e))


# ---- Optimization result: before vs after ----
def _descriptors(p):
    return [float(np.max(np.abs(p.I))), float(np.trapz(p.I, p.t)),
            float(np.max(np.abs(p.Q))), float(np.trapz(p.I ** 2 + p.Q ** 2, p.t))]


if "opt" in st.session_state:
    o = st.session_state["opt"]
    omodel, ohw, ofd = o["model"], o["hardware"], o["drive_freq"]
    before, after = o["before"], o["after"]
    before_d, after_d = ohw.apply(before), ohw.apply(after)   # what the qubit sees
    Ub, Ua = propagate(omodel, before_d, ofd), propagate(omodel, after_d, ofd)
    Fb, Fa = avg_gate_fidelity(Ub, X_TARGET), avg_gate_fidelity(Ua, X_TARGET)
    Lb, La = leakage(Ub), leakage(Ua)

    st.markdown("---")
    head, clear = st.columns([4, 1])
    head.subheader(f"Optimization result — {o['method']}")
    if clear.button("Clear result", help="Discard this optimization result."):
        del st.session_state["opt"]
        st.rerun()

    m1, m2 = st.columns(2)
    m1.metric("Fidelity (optimized)", f"{Fa:.5f}", delta=f"{Fa - Fb:+.5f}")
    m2.metric("Leakage (optimized)", f"{La:.2e}",
              delta=f"{La - Lb:+.1e}", delta_color="inverse")

    st.markdown("**Pulse shape descriptors (before → after)**")
    st.table({
        "descriptor": ["peak |I| (rad/ns)", "area ∫I dt",
                       "peak |Q| (rad/ns)", "energy ∫(I²+Q²) dt"],
        "before": [f"{v:.4f}" for v in _descriptors(before)],
        "after": [f"{v:.4f}" for v in _descriptors(after)],
    })

    st.markdown("**Drive envelope: before vs after**")
    fig_cmp = go.Figure()
    fig_cmp.add_scatter(x=before.t, y=before.I, name="I before", line=dict(color="royalblue"))
    fig_cmp.add_scatter(x=before.t, y=before.Q, name="Q before", line=dict(color="firebrick"))
    fig_cmp.add_scatter(x=after.t, y=after.I, name="I after",
                        line=dict(color="royalblue", dash="dash"))
    fig_cmp.add_scatter(x=after.t, y=after.Q, name="Q after",
                        line=dict(color="firebrick", dash="dash"))
    fig_cmp.update_layout(xaxis_title="t (ns)", yaxis_title="rad/ns", height=320)
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("**State populations: before vs after**")
    pc = st.columns(2)
    for col, lbl, dp in [(pc[0], "before", before_d), (pc[1], "after", after_d)]:
        te, pops = viz.population_trajectory(omodel, dp, ofd)
        fp = go.Figure()
        for j in range(omodel.n_levels):
            fp.add_scatter(x=te, y=pops[:, j], name=f"|{j}>")
        fp.update_layout(title=lbl, xaxis_title="t (ns)", yaxis_title="population", height=260)
        col.plotly_chart(fp, use_container_width=True)

    st.markdown("**Drive spectrum: before vs after**")
    fb_, pb_ = viz.drive_spectrum(before)
    fa_, pa_ = viz.drive_spectrum(after)
    fig_sp = go.Figure()
    fig_sp.add_scatter(x=fb_ * 1000, y=pb_, name="before")
    fig_sp.add_scatter(x=fa_ * 1000, y=pa_, name="after", line=dict(dash="dash"))
    fig_sp.update_layout(xaxis_title="detuning (MHz)", yaxis_title="power (norm.)", height=260)
    st.plotly_chart(fig_sp, use_container_width=True)

    _, bl_b = viz.bloch_trajectory(omodel, before_d, ofd)
    _, bl_a = viz.bloch_trajectory(omodel, after_d, ofd)
    st.markdown("**Bloch endpoint (X gate from |0>)**")
    st.table({
        "component": ["x", "y", "z"],
        "before": [f"{v:+.3f}" for v in bl_b[-1]],
        "after": [f"{v:+.3f}" for v in bl_a[-1]],
    })

    # AllXY before/after, only for DRAG (a scalar amp/drag_coef gate calibration).
    # Uses a 4-level model so leakage -- and DRAG's suppression of it -- is visible;
    # the canonical staircase is [0]*5,[0.5]*12,[1]*4.
    if o["method"] == "DRAG" and o.get("after_params"):
        st.markdown("**AllXY with the gate calibration: before vs after**")
        st.caption("Deviations from the staircase reveal gate miscalibration; the "
                   "optimized DRAG amp/drag_coef should flatten them. Computed on a "
                   "4-level transmon so leakage is visible.")
        axy_model = ChargeBasisTransmon(
            DeviceParams.from_spectrum(5.252, -0.064, n_levels=4, flux=o["flux"]))
        bp, ap = o["before_params"], o["after_params"]
        labels_b, p1_b = viz.allxy_populations(
            axy_model, ofd, duration_ns=o["duration"], sigma_ns=o["sigma"],
            amp=bp["amp"], drag_coef=bp["drag_coef"])
        _, p1_a = viz.allxy_populations(
            axy_model, ofd, duration_ns=o["duration"], sigma_ns=o["sigma"],
            amp=ap["amp"], drag_coef=ap["drag_coef"])
        canonical = [0.0] * 5 + [0.5] * 12 + [1.0] * 4
        fig_axy2 = go.Figure()
        fig_axy2.add_scatter(x=labels_b, y=canonical, name="ideal", mode="markers",
                             marker=dict(color="lightgray", size=9))
        fig_axy2.add_scatter(x=labels_b, y=p1_b, name="before", mode="lines+markers")
        fig_axy2.add_scatter(x=labels_b, y=p1_a, name="after", mode="lines+markers",
                             line=dict(dash="dash"))
        fig_axy2.update_layout(yaxis_title="P(excited)", height=300, xaxis_tickangle=-60)
        st.plotly_chart(fig_axy2, use_container_width=True)

    st.markdown("**Optimizer convergence**")
    fig_conv = go.Figure(go.Scatter(y=o["history"]))
    fig_conv.update_layout(xaxis_title="cost evaluation", yaxis_title="cost", height=240)
    st.plotly_chart(fig_conv, use_container_width=True)

    wf = to_opx_waveforms(after, dac_per_radns=1.0)
    csv = "I_wf,Q_wf\n" + "\n".join(f"{i},{q}" for i, q in zip(wf["I_wf"], wf["Q_wf"]))
    st.download_button("Download optimized I/Q waveforms (CSV)", data=csv,
                       file_name="optimized_waveforms.csv")


def main():
    """Entry point marker (Streamlit runs the module top-level)."""
    return None
