"""Error Analysis -- deterministic error budget and Monte Carlo noise."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ryd_gate.ideal_cz import CZGateSimulator

st.set_page_config(page_title="Error Analysis", layout="wide", page_icon="\u269b")
st.title("Error Analysis")

# ── Known pulse parameters ─────────────────────────────────────────────
X_DARK_TO = [-0.699, 1.029, 0.376, 1.571, 1.445, 1.341]
X_BRIGHT_TO = [0.625, 1.237, -0.471, 1.655, 3.420, 1.334]

SSS_STATES = [f"SSS-{i}" for i in range(12)]

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

config_label = st.sidebar.selectbox("Configuration", ["Dark TO (optimized)", "Bright TO (optimized)"])
if "Dark" in config_label:
    detuning_sign, x_params = 1, X_DARK_TO
else:
    detuning_sign, x_params = -1, X_BRIGHT_TO

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo")
n_mc = st.sidebar.slider("MC shots", 10, 500, 50, step=10)
sigma_det_khz = st.sidebar.slider("Dephasing \u03c3 (kHz)", 0, 500, 130, step=10)
sigma_pos_nm = st.sidebar.slider("Position \u03c3_xy (nm)", 0, 200, 70, step=10)
sigma_z_nm = st.sidebar.slider("Position \u03c3_z (nm)", 0, 300, 130, step=10)


# ── Deterministic Error Budget ─────────────────────────────────────────
st.subheader("Deterministic Error Budget")

run_det = st.button("Compute Error Budget", type="primary")

if run_det:
    with st.spinner("Computing Rydberg decay..."):
        sim_ryd = CZGateSimulator(
            param_set="our", strategy="TO", detuning_sign=detuning_sign,
            enable_rydberg_decay=True)
        infid_ryd = sim_ryd.gate_fidelity(x_params, fid_type="sss")
        budget_ryd = sim_ryd.error_budget(x_params, initial_states=SSS_STATES)

    with st.spinner("Computing intermediate decay..."):
        sim_mid = CZGateSimulator(
            param_set="our", strategy="TO", detuning_sign=detuning_sign,
            enable_intermediate_decay=True)
        infid_mid = sim_mid.gate_fidelity(x_params, fid_type="sss")
        budget_mid = sim_mid.error_budget(x_params, initial_states=SSS_STATES)

    with st.spinner("Computing polarization leakage..."):
        sim_pol = CZGateSimulator(
            param_set="our", strategy="TO", detuning_sign=detuning_sign,
            enable_polarization_leakage=True)
        infid_pol = sim_pol.gate_fidelity(x_params, fid_type="sss")
        budget_pol = sim_pol.error_budget(x_params, initial_states=SSS_STATES)

    # Build table
    sources = {
        "Rydberg decay": (infid_ryd, budget_ryd["rydberg_decay"]),
        "Intermediate decay": (infid_mid, budget_mid["intermediate_decay"]),
        "Polarization leakage": (infid_pol, budget_pol["polarization_leakage"]),
    }

    rows = []
    for name, (infid, b) in sources.items():
        rows.append({
            "Source": name,
            "Infidelity": f"{infid:.2e}",
            "XYZ (Pauli)": f"{b['XYZ']:.2e}" if b.get("XYZ") else "--",
            "AL (Atom Loss)": f"{b['AL']:.2e}" if b.get("AL") else "--",
            "LG (Leakage)": f"{b['LG']:.2e}" if b.get("LG") else "--",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Bar chart breakdown
    names = list(sources.keys())
    infids = [sources[n][0] for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=infids,
        marker_color=["#e74c3c", "#3498db", "#f39c12"],
        text=[f"{v:.2e}" for v in infids], textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="SSS Infidelity", height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Monte Carlo Analysis ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Monte Carlo Noise Simulation")

run_mc = st.button("Run Monte Carlo", type="primary")

if run_mc and sigma_det_khz > 0:
    sigma_det = sigma_det_khz * 1e3  # Hz
    sigma_pos = (sigma_pos_nm * 1e-9, sigma_pos_nm * 1e-9, sigma_z_nm * 1e-9)

    with st.spinner(f"Running {n_mc} MC shots..."):
        sim_mc = CZGateSimulator(
            param_set="our", strategy="TO", detuning_sign=detuning_sign,
            enable_rydberg_decay=True,
            enable_intermediate_decay=True,
            enable_polarization_leakage=True,
            enable_rydberg_dephasing=True,
            enable_position_error=True,
            sigma_detuning=sigma_det,
            sigma_pos_xyz=sigma_pos,
        )
        mc_result = sim_mc.run_monte_carlo_simulation(
            x_params, n_shots=n_mc,
            sigma_detuning=sigma_det, sigma_pos_xyz=sigma_pos,
            seed=42, compute_branching=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Infidelity", f"{mc_result.mean_infidelity:.2e}")
    c2.metric("Std Infidelity", f"{mc_result.std_infidelity:.2e}")
    c3.metric("Shots", str(mc_result.n_shots))

    # Fidelity histogram
    fig_hist = go.Figure(go.Histogram(
        x=1 - mc_result.fidelities, nbinsx=30,
        marker_color="steelblue", opacity=0.8,
    ))
    fig_hist.add_vline(x=mc_result.mean_infidelity, line_dash="dash",
                        line_color="red", annotation_text="mean")
    fig_hist.update_layout(
        xaxis_title="Infidelity", yaxis_title="Count",
        height=300, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Branching breakdown
    if mc_result.mean_branch_XYZ is not None:
        branch_names = ["XYZ (Pauli)", "AL (Atom Loss)", "LG (Leakage)", "Phase"]
        branch_vals = [
            mc_result.mean_branch_XYZ, mc_result.mean_branch_AL,
            mc_result.mean_branch_LG, mc_result.mean_branch_phase,
        ]
        branch_stds = [
            mc_result.std_branch_XYZ, mc_result.std_branch_AL,
            mc_result.std_branch_LG, mc_result.std_branch_phase,
        ]

        fig_branch = go.Figure(go.Bar(
            x=branch_names, y=branch_vals,
            error_y=dict(type="data", array=[s / np.sqrt(n_mc) for s in branch_stds]),
            marker_color=["#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"],
            text=[f"{v:.2e}" for v in branch_vals], textposition="outside",
        ))
        fig_branch.update_layout(
            yaxis_title="Mean Error Contribution", height=300,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_branch, use_container_width=True)
