"""Lattice Simulator -- many-body Rydberg dynamics on a square lattice."""

import time as _time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ryd_gate.lattice import (
    build_hamiltonian,
    build_operators,
    domain_config,
    evolve_constant_H,
    evolve_sweep,
    is_in_domain,
    make_square_lattice,
    measure_from_states,
    precompute_bit_masks,
    product_state,
)

st.set_page_config(page_title="Lattice Simulator", layout="wide", page_icon="\u269b")
st.title("Many-Body Lattice Simulator")

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.header("Lattice Configuration")
Lx = st.sidebar.slider("Lattice width (Lx)", 2, 5, 3)
Ly = st.sidebar.slider("Lattice height (Ly)", 2, 5, 3)
V_nn = st.sidebar.slider("V_nn / \u03a9", 1.0, 50.0, 24.0, step=1.0)

N = Lx * Ly
dim = 2 ** N
st.sidebar.caption(f"N = {N} atoms, dim = {dim:,}")
if N > 12:
    st.sidebar.warning("Large Hilbert space -- simulation may be slow.")

experiment = st.sidebar.selectbox("Experiment", ["Domain Shrinking", "Higgs Mode"])

st.sidebar.markdown("---")
st.sidebar.subheader("Sweep Parameters")
Delta_start = st.sidebar.slider("\u0394_start / \u03a9", -5.0, 0.0, -3.0, step=0.5)
n_sweep = st.sidebar.slider("Sweep steps", 10, 80, 30)
t_sweep = st.sidebar.slider("Sweep time (1/\u03a9)", 10.0, 100.0, 55.0, step=5.0)
delta_pin = st.sidebar.slider("Pinning strength |\u03b4| / \u03a9", 1.0, 10.0, 4.0, step=0.5)


# ── Cached operators ───────────────────────────────────────────────────
@st.cache_resource
def setup(lx, ly, v):
    lattice = make_square_lattice(lx, ly)
    ops = build_operators(lattice.N, lattice.vdw_pairs, v, verbose=False)
    masks = precompute_bit_masks(lattice.N)
    return lattice, ops, masks


lattice, ops, bit_masks = setup(Lx, Ly, V_nn)
coords, sublattice = lattice.coords, lattice.sublattice


# ── Lattice preview ────────────────────────────────────────────────────
col_lattice, col_results = st.columns([1, 2])

with col_lattice:
    st.subheader("Lattice Preview")
    colors = ["steelblue" if s > 0 else "coral" for s in sublattice]
    fig_lat = go.Figure(go.Scatter(
        x=coords[:, 1], y=coords[:, 0], mode="markers+text",
        marker=dict(size=22, color=colors, line=dict(width=1, color="black")),
        text=[str(i) for i in range(N)], textposition="middle center",
        textfont=dict(size=9, color="white"),
    ))
    fig_lat.update_layout(
        xaxis_title="y", yaxis_title="x", height=300,
        margin=dict(l=40, r=20, t=10, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_lat, use_container_width=True)
    st.caption("Blue = sublattice +1 (AF1 excited), Red = sublattice -1 (AF2 excited)")


# ── Domain Shrinking ───────────────────────────────────────────────────
if experiment == "Domain Shrinking":
    with col_results:
        st.subheader("Domain Shrinking")

        Delta_f = st.slider("Final \u0394 / \u03a9 (ordered phase)", 1.5, 5.0, 2.5, step=0.5)
        t_hold = st.slider("Hold time (1/\u03a9)", 1.0, 15.0, 6.0, step=1.0)
        n_hold = st.slider("Hold steps", 50, 300, 100)

        run = st.button("Run Domain Shrinking", type="primary")

    if run:
        cx, cy = (Lx - 1) / 2.0, (Ly - 1) / 2.0
        domain_radius = 0.8

        # Sweep
        with st.spinner("Phase 1: Adiabatic sweep with pinning..."):
            psi0 = product_state([0] * N, N)
            target = domain_config(coords, sublattice, (cx, cy), domain_radius)
            pin_deltas = np.where(target == 0, -delta_pin, 0.0)

            t0 = _time.time()
            psi = evolve_sweep(psi0, Delta_start, Delta_f, t_sweep, n_sweep,
                               pin_deltas, ops, omega_ramp_frac=0.1)
            sweep_time = _time.time() - t0

        ms_sw, n_sw, _ = measure_from_states(psi, bit_masks, sublattice)
        st.info(f"Sweep: {sweep_time:.1f}s | m_s = {ms_sw:.4f} | <n> = {n_sw:.4f}")

        # Hold
        with st.spinner("Phase 2: Free evolution..."):
            H_hold = build_hamiltonian(1.0, Delta_f, np.zeros(N), ops)
            t0 = _time.time()
            hold_times, hold_states = evolve_constant_H(psi, H_hold, t_hold, n_hold)
            hold_time_elapsed = _time.time() - t0

        st.success(f"Hold evolution: {hold_time_elapsed:.1f}s")

        ms, n_mean, occ_all = measure_from_states(hold_states, bit_masks, sublattice)

        # Domain area
        domain_weight = np.zeros(N)
        for i, (ix, iy) in enumerate(coords):
            if is_in_domain(ix, iy, cx, cy, domain_radius):
                domain_weight[i] = 1.0 if sublattice[i] < 0 else -1.0
        domain_areas = occ_all @ domain_weight + np.sum(domain_weight < 0)

        # Plots
        # Snapshots
        snap_idx = [0, len(hold_times) // 4, len(hold_times) // 2, len(hold_times) - 1]
        fig_snap = make_subplots(rows=1, cols=len(snap_idx),
                                  subplot_titles=[f"t={hold_times[i]:.1f}" for i in snap_idx])
        for col, idx in enumerate(snap_idx):
            local_ms = sublattice * (2 * occ_all[idx] - 1)
            grid = local_ms.reshape(Lx, Ly)
            fig_snap.add_trace(
                go.Heatmap(z=grid, colorscale="RdBu", zmin=-1, zmax=1, showscale=(col == len(snap_idx) - 1)),
                row=1, col=col + 1,
            )
        fig_snap.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_snap, use_container_width=True)

        # Time series
        fig_ts = make_subplots(rows=1, cols=3,
                                subplot_titles=["Staggered mag.", "Domain area", "Rydberg fraction"])
        fig_ts.add_trace(go.Scatter(x=hold_times, y=ms, line=dict(color="blue")), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=hold_times, y=domain_areas, line=dict(color="red")), row=1, col=2)
        fig_ts.add_trace(go.Scatter(x=hold_times, y=n_mean, line=dict(color="green")), row=1, col=3)
        fig_ts.update_xaxes(title_text="Hold time (1/\u03a9)")
        fig_ts.update_layout(height=300, showlegend=False, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_ts, use_container_width=True)


# ── Higgs Mode ─────────────────────────────────────────────────────────
elif experiment == "Higgs Mode":
    with col_results:
        st.subheader("Higgs Mode Oscillations")

        Delta_values = st.multiselect(
            "\u0394 / \u03a9 values to probe",
            options=[0.0, 0.5, 1.1, 1.5, 2.0, 2.5, 3.0],
            default=[0.0, 1.1, 2.5],
        )
        t_hold = st.slider("Hold time (1/\u03a9)", 2.0, 20.0, 10.0, step=1.0)
        n_hold = st.slider("Hold steps", 50, 400, 200)

        run = st.button("Run Higgs Mode", type="primary")

    if run and Delta_values:
        pin_deltas = np.where(sublattice > 0, -delta_pin, 0.0)
        all_results = {}
        progress = st.progress(0.0)

        for k, Delta_f in enumerate(sorted(Delta_values)):
            progress.progress((k) / len(Delta_values), text=f"\u0394/\u03a9 = {Delta_f:.1f}")
            psi0 = product_state([0] * N, N)
            psi = evolve_sweep(psi0, Delta_start, Delta_f, t_sweep, n_sweep,
                               pin_deltas, ops, omega_ramp_frac=0.1)

            H_hold = build_hamiltonian(1.0, Delta_f, np.zeros(N), ops)
            hold_times, hold_states = evolve_constant_H(psi, H_hold, t_hold, n_hold)
            ms, n_mean, _ = measure_from_states(hold_states, bit_masks, sublattice)
            all_results[Delta_f] = {"times": hold_times, "ms": ms}

        progress.progress(1.0, text="Done!")

        # m_s(t) plot
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        fig_ms = go.Figure()
        for i, Delta_f in enumerate(sorted(all_results)):
            r = all_results[Delta_f]
            fig_ms.add_trace(go.Scatter(
                x=r["times"], y=r["ms"], name=f"\u0394/\u03a9 = {Delta_f:.1f}",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig_ms.update_layout(
            xaxis_title="Hold time (1/\u03a9)", yaxis_title="m_s",
            height=350, margin=dict(l=40, r=20, t=30, b=40),
        )
        fig_ms.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
        st.plotly_chart(fig_ms, use_container_width=True)

        # FFT spectrum
        fig_fft = go.Figure()
        for i, Delta_f in enumerate(sorted(all_results)):
            r = all_results[Delta_f]
            ms_c = r["ms"] - np.mean(r["ms"])
            dt = r["times"][1] - r["times"][0]
            freqs = np.fft.rfftfreq(len(ms_c), d=dt)
            power = np.abs(np.fft.rfft(ms_c * np.hanning(len(ms_c)))) ** 2
            power[0] = 0
            pmax = power.max()
            fig_fft.add_trace(go.Scatter(
                x=freqs, y=power / pmax if pmax > 0 else power,
                name=f"\u0394/\u03a9 = {Delta_f:.1f}",
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig_fft.update_layout(
            xaxis_title="Frequency (\u03a9/2\u03c0)", yaxis_title="Power (norm.)",
            xaxis_range=[0, 2.0],
            height=300, margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig_fft, use_container_width=True)
