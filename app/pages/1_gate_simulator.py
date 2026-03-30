"""CZ Gate Simulator -- pulse design, waveform visualization, fidelity."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ryd_gate.ideal_cz import CZGateSimulator
from ryd_gate.blackman import blackman_pulse

st.set_page_config(page_title="CZ Gate Simulator", layout="wide", page_icon="\u269b")
st.title("CZ Gate Simulator")

# ── Known-good presets ──────────────────────────────────────────────────
PRESETS = {
    "Optimized Dark TO": {
        "strategy": "TO", "detuning_sign": 1,
        "x": [-0.699, 1.029, 0.376, 1.571, 1.445, 1.341],
    },
    "Optimized Bright TO": {
        "strategy": "TO", "detuning_sign": -1,
        "x": [0.625, 1.237, -0.471, 1.655, 3.420, 1.334],
    },
    "Custom": None,
}

TO_LABELS = [
    ("Amplitude A (rad)", -np.pi, np.pi, 0.0),
    ("\u03c9 / \u03a9_eff", -10.0, 10.0, 1.0),
    ("Phase \u03c6\u2080 (rad)", -np.pi, np.pi, 0.0),
    ("Chirp \u03b4 / \u03a9_eff", -2.0, 2.0, 0.0),
    ("Z-rotation \u03b8 (rad)", -np.pi, np.pi, 0.0),
    ("Gate time T / T_scale", 0.5, 3.0, 1.3),
]

AR_LABELS = [
    ("\u03c9 / \u03a9_eff", -10.0, 10.0, 1.0),
    ("Amp A\u2081 (rad)", -np.pi, np.pi, 0.0),
    ("Phase \u03c6\u2081 (rad)", -np.pi, np.pi, 0.0),
    ("Amp A\u2082 (rad)", -np.pi, np.pi, 0.0),
    ("Phase \u03c6\u2082 (rad)", -np.pi, np.pi, 0.0),
    ("Chirp \u03b4 / \u03a9_eff", -2.0, 2.0, 0.0),
    ("Gate time T / T_scale", 0.5, 3.0, 1.3),
    ("Z-rotation \u03b8 (rad)", -np.pi, np.pi, 0.0),
]


# ── Sidebar: system configuration ──────────────────────────────────────
st.sidebar.header("System Configuration")

param_set = st.sidebar.selectbox("Param set", ["our (Rb87 n=70)", "lukin (Harvard n=53)"])
param_set_key = "our" if "our" in param_set else "lukin"

strategy = st.sidebar.selectbox("Strategy", ["TO (Time-Optimal)", "AR (Amplitude-Robust)"])
strategy_key = "TO" if "TO" in strategy else "AR"

det_sign_label = st.sidebar.radio("Detuning", ["Dark (+1)", "Bright (-1)"], horizontal=True)
detuning_sign = 1 if "Dark" in det_sign_label else -1

blackmanflag = st.sidebar.checkbox("Blackman envelope", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Decay channels")
enable_rydberg_decay = st.sidebar.checkbox("Rydberg decay")
enable_intermediate_decay = st.sidebar.checkbox("Intermediate decay")
enable_polarization_leakage = st.sidebar.checkbox("Polarization leakage")


# ── Cached simulator creation ──────────────────────────────────────────
@st.cache_resource
def get_simulator(ps, strat, dsign, bf, erd, eid, epl):
    return CZGateSimulator(
        param_set=ps, strategy=strat, detuning_sign=dsign, blackmanflag=bf,
        enable_rydberg_decay=erd, enable_intermediate_decay=eid,
        enable_polarization_leakage=epl,
    )


sim = get_simulator(param_set_key, strategy_key, detuning_sign, blackmanflag,
                    enable_rydberg_decay, enable_intermediate_decay,
                    enable_polarization_leakage)


# ── Pulse parameters ───────────────────────────────────────────────────
col_params, col_wave, col_results = st.columns([1, 1.2, 1.5])

with col_params:
    st.subheader("Pulse Parameters")

    preset_name = st.selectbox("Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    labels = TO_LABELS if strategy_key == "TO" else AR_LABELS
    n_params = 6 if strategy_key == "TO" else 8

    x = []
    for i, (label, lo, hi, default) in enumerate(labels):
        val = preset["x"][i] if preset and i < len(preset.get("x", [])) else default
        x.append(st.slider(label, min_value=float(lo), max_value=float(hi),
                            value=float(val), step=0.001, key=f"p{i}"))

    # Show physical units
    t_scale_ns = sim.time_scale * 1e9
    t_gate_idx = 5 if strategy_key == "TO" else 6
    gate_time_ns = x[t_gate_idx] * t_scale_ns
    st.caption(f"T_scale = {t_scale_ns:.1f} ns | Gate time = {gate_time_ns:.1f} ns")
    st.caption(f"\u03a9_eff = {sim.rabi_eff / (2*np.pi*1e6):.1f} MHz")

# ── Waveform visualization ─────────────────────────────────────────────
with col_wave:
    st.subheader("Waveform")

    t_gate = x[t_gate_idx] * sim.time_scale
    t_arr = np.linspace(0, t_gate, 500)
    t_ns = t_arr * 1e9

    # Blackman envelope
    t_rise = sim._system.t_rise if hasattr(sim, '_system') else 20e-9
    if blackmanflag and t_gate >= 2 * t_rise:
        envelope = blackman_pulse(t_arr, t_rise, t_gate)
    else:
        envelope = np.ones_like(t_arr)

    # Phase modulation (simplified computation)
    if strategy_key == "TO":
        A, omega_norm, phi0, delta_norm = x[0], x[1], x[2], x[3]
        omega = omega_norm * sim.rabi_eff
        delta = delta_norm * sim.rabi_eff
        phase = A * np.cos(omega * t_arr + phi0) + delta * t_arr
    else:
        omega_norm, A1, phi1, A2, phi2, delta_norm = x[0], x[1], x[2], x[3], x[4], x[5]
        omega = omega_norm * sim.rabi_eff
        delta = delta_norm * sim.rabi_eff
        phase = A1 * np.sin(omega * t_arr + phi1) + A2 * np.sin(2 * omega * t_arr + phi2) + delta * t_arr

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t_ns, y=envelope, name="Envelope",
                                   line=dict(color="steelblue", width=2)))
    fig_wave.add_trace(go.Scatter(x=t_ns, y=np.cos(phase) * envelope,
                                   name="Re[exp(-i\u03c6)]",
                                   line=dict(color="coral", width=1.5)))
    fig_wave.update_layout(
        xaxis_title="Time (ns)", yaxis_title="Amplitude",
        height=350, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # Phase plot
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=t_ns, y=phase, name="\u03c6(t)",
                                    line=dict(color="mediumpurple", width=2)))
    fig_phase.update_layout(
        xaxis_title="Time (ns)", yaxis_title="Phase (rad)",
        height=250, margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# ── Simulation results ─────────────────────────────────────────────────
with col_results:
    st.subheader("Simulation Results")

    initial_state = st.selectbox("Initial state for diagnostics",
                                  ["11", "01", "10", "00"] + [f"SSS-{i}" for i in range(12)])

    run = st.button("Run Simulation", type="primary", use_container_width=True)

    if run:
        with st.spinner("Computing gate fidelity..."):
            infid_avg = sim.gate_fidelity(x, fid_type="average")
            infid_sss = sim.gate_fidelity(x, fid_type="sss")

        c1, c2 = st.columns(2)
        c1.metric("Average Infidelity", f"{infid_avg:.2e}")
        c2.metric("SSS Infidelity", f"{infid_sss:.2e}")

        # State fidelities bar chart
        with st.spinner("Computing state fidelities..."):
            comp_states = ["00", "01", "10", "11"]
            infids = [sim.state_infidelity(s, x) for s in comp_states]
            fids = [1 - inf for inf in infids]

        fig_bar = go.Figure(go.Bar(
            x=[f"|{s}\u27e9" for s in comp_states], y=fids,
            marker_color=["#2ecc71" if f > 0.99 else "#e74c3c" for f in fids],
            text=[f"{f:.4f}" for f in fids], textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis_title="State Fidelity", yaxis_range=[0, 1.05],
            height=280, margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Population evolution
        with st.spinner("Computing population evolution..."):
            mid, ryd, garb = sim.diagnose_run(x, initial_state)
            n_pts = len(mid)
            t_pop = np.linspace(0, gate_time_ns, n_pts)

        fig_pop = go.Figure()
        fig_pop.add_trace(go.Scatter(x=t_pop, y=mid, name="Intermediate",
                                      line=dict(color="dodgerblue", width=2)))
        fig_pop.add_trace(go.Scatter(x=t_pop, y=ryd, name="Rydberg |r\u27e9",
                                      line=dict(color="crimson", width=2)))
        fig_pop.add_trace(go.Scatter(x=t_pop, y=garb, name="Garbage |r'\u27e9",
                                      line=dict(color="gray", width=1.5, dash="dash")))
        fig_pop.update_layout(
            xaxis_title="Time (ns)", yaxis_title="Population",
            height=320, margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_pop, use_container_width=True)
