"""Interactive energy level visualizer for 1-atom and 2-atom Rydberg systems.

Diagonalizes the effective Hamiltonians in real time as sliders change,
showing how pinning (delta_A) and Rydberg blockade (V_ryd) reshape the
adiabatic energy landscape.

Physics:
  1-atom: 2x2 effective Hamiltonian {|g>, |r>} — single avoided crossing
  2-atom: 4x4 effective Hamiltonian {|gg>, |gr>, |rg>, |rr>} — blockade-assisted pinning
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Energy Levels", layout="wide", page_icon="☢")

st.title("Rydberg Energy Level Visualizer")
st.markdown(
    "Instantaneous eigenenergies of the effective Hamiltonian during an "
    "adiabatic detuning sweep.  **Dashed** = diabatic (bare) states, "
    "**solid** = adiabatic (dressed) states."
)

# ── Sidebar controls ──────────────────────────────────────────────────

st.sidebar.header("Laser parameters")
Omega_420_MHz = st.sidebar.slider(
    r"Ω₄₂₀ / (2π)  [MHz]", 10.0, 500.0, 135.0, 5.0,
)
Omega_1013_MHz = st.sidebar.slider(
    r"Ω₁₀₁₃ / (2π)  [MHz]", 10.0, 500.0, 135.0, 5.0,
)
Delta_int_GHz = st.sidebar.slider(
    r"Δ_int / (2π)  [GHz]  (intermediate detuning)", 0.5, 15.0, 2.4, 0.1,
)
Omega_eff_MHz = Omega_420_MHz * Omega_1013_MHz / (2 * Delta_int_GHz * 1e3)
st.sidebar.metric(r"Ω_eff / (2π)", f"{Omega_eff_MHz:.2f} MHz")

st.sidebar.divider()
st.sidebar.header("Addressing & interaction")
delta_A_MHz = st.sidebar.slider(
    r"δ_A / (2π)  [MHz]  (pinning detuning)", 0.0, 80.0, 10.0, 0.5,
)
V_ryd_MHz = st.sidebar.slider(
    r"V_ryd / (2π)  [MHz]  (Rydberg interaction)", 0.0, 500.0, 213.0, 1.0,
)

st.sidebar.divider()
st.sidebar.header("Sweep")
sweep_half_MHz = st.sidebar.slider(
    "Sweep half-range  [MHz]", 5.0, 100.0, 40.0, 1.0,
)
current_Delta_MHz = st.sidebar.slider(
    r"Current Δ / (2π)  [MHz]", -sweep_half_MHz, sweep_half_MHz, 0.0, 0.5,
)

# ── Computation ───────────────────────────────────────────────────────

N_PTS = 500
Delta = np.linspace(-sweep_half_MHz, sweep_half_MHz * 1.5, N_PTS)

# Extend x-range to show E_rr crossing when V_ryd is modest
x_max = max(sweep_half_MHz * 1.5, delta_A_MHz + V_ryd_MHz + 20)
Delta_2atom = np.linspace(-sweep_half_MHz, x_max, N_PTS)

# --- 1-atom (analytical) ---
E_g_diab = np.zeros_like(Delta)
E_r_diab = -Delta + delta_A_MHz
disc_1 = np.sqrt((Delta - delta_A_MHz) ** 2 + Omega_eff_MHz ** 2)
E_minus = (-Delta + delta_A_MHz - disc_1) / 2
E_plus = (-Delta + delta_A_MHz + disc_1) / 2

# Marker on lowest adiabatic at current Delta
d_cur = current_Delta_MHz
disc_cur = np.sqrt((d_cur - delta_A_MHz) ** 2 + Omega_eff_MHz ** 2)
marker_1_y = (-d_cur + delta_A_MHz - disc_cur) / 2

# --- 2-atom (numerical eigvalsh) ---
E_gg_diab = np.zeros_like(Delta_2atom)
E_gr_diab = -Delta_2atom
E_rg_diab = -Delta_2atom + delta_A_MHz
E_rr_diab = -2 * Delta_2atom + delta_A_MHz + V_ryd_MHz

adiab_2 = np.empty((N_PTS, 4))
for i, d in enumerate(Delta_2atom):
    H = np.array(
        [
            [0, Omega_eff_MHz / 2, Omega_eff_MHz / 2, 0],
            [Omega_eff_MHz / 2, -d, 0, Omega_eff_MHz / 2],
            [Omega_eff_MHz / 2, 0, -d + delta_A_MHz, Omega_eff_MHz / 2],
            [0, Omega_eff_MHz / 2, Omega_eff_MHz / 2, -2 * d + delta_A_MHz + V_ryd_MHz],
        ]
    )
    adiab_2[i] = np.linalg.eigvalsh(H)

# Marker for 2-atom: lowest adiabatic at current Delta
H_cur = np.array(
    [
        [0, Omega_eff_MHz / 2, Omega_eff_MHz / 2, 0],
        [Omega_eff_MHz / 2, -d_cur, 0, Omega_eff_MHz / 2],
        [Omega_eff_MHz / 2, 0, -d_cur + delta_A_MHz, Omega_eff_MHz / 2],
        [0, Omega_eff_MHz / 2, Omega_eff_MHz / 2, -2 * d_cur + delta_A_MHz + V_ryd_MHz],
    ]
)
eigs_cur = np.linalg.eigvalsh(H_cur)
marker_2_y = eigs_cur[0]

# ── Plots ─────────────────────────────────────────────────────────────

DIAB_OPACITY = 0.5
DIAB_WIDTH = 1.5
ADIAB_WIDTH = 2.5

col1, col2 = st.columns(2)

# --- 1-Atom plot ---
with col1:
    fig1 = go.Figure()

    # Diabatic
    fig1.add_trace(go.Scatter(
        x=Delta, y=E_g_diab, mode="lines", name="|g⟩",
        line=dict(dash="dash", color="steelblue", width=DIAB_WIDTH),
        opacity=DIAB_OPACITY,
    ))
    fig1.add_trace(go.Scatter(
        x=Delta, y=E_r_diab, mode="lines", name="|r⟩",
        line=dict(dash="dash", color="crimson", width=DIAB_WIDTH),
        opacity=DIAB_OPACITY,
    ))

    # Adiabatic
    fig1.add_trace(go.Scatter(
        x=Delta, y=E_minus, mode="lines", name="E₋ (adiabatic)",
        line=dict(color="royalblue", width=ADIAB_WIDTH),
    ))
    fig1.add_trace(go.Scatter(
        x=Delta, y=E_plus, mode="lines", name="E₊ (adiabatic)",
        line=dict(color="orangered", width=ADIAB_WIDTH),
    ))

    # Sweep region
    fig1.add_vrect(
        x0=-sweep_half_MHz, x1=sweep_half_MHz,
        fillcolor="lightgray", opacity=0.15, line_width=0,
        annotation_text="sweep", annotation_position="top left",
    )

    # Current detuning marker
    fig1.add_trace(go.Scatter(
        x=[d_cur], y=[marker_1_y], mode="markers",
        marker=dict(size=14, color="black", symbol="circle",
                    line=dict(width=2, color="yellow")),
        name=f"Δ = {d_cur:.1f} MHz",
        showlegend=True,
    ))
    fig1.add_vline(x=d_cur, line_dash="dot", line_color="gray", opacity=0.6)

    # Crossing annotation
    fig1.add_annotation(
        x=delta_A_MHz, y=0,
        text=f"crossing at δ_A = {delta_A_MHz:.0f}",
        showarrow=True, arrowhead=2, ax=0, ay=-40,
        font=dict(size=10, color="gray"),
    )

    fig1.update_layout(
        title=dict(text="Single atom  (2×2)", font=dict(size=16)),
        xaxis_title="Detuning  Δ/(2π)  [MHz]",
        yaxis_title="Energy / (2π)  [MHz]",
        height=550,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    st.plotly_chart(fig1, use_container_width=True)

# --- 2-Atom plot ---
with col2:
    fig2 = go.Figure()

    # Diabatic
    diab_colors = {"|gg⟩": "steelblue", "|gr⟩": "seagreen",
                   "|rg⟩": "orange", "|rr⟩": "crimson"}
    for name, y, col in [
        ("|gg⟩", E_gg_diab, "steelblue"),
        ("|gr⟩", E_gr_diab, "seagreen"),
        ("|rg⟩", E_rg_diab, "orange"),
        ("|rr⟩", E_rr_diab, "crimson"),
    ]:
        fig2.add_trace(go.Scatter(
            x=Delta_2atom, y=y, mode="lines", name=name,
            line=dict(dash="dash", color=col, width=DIAB_WIDTH),
            opacity=DIAB_OPACITY,
        ))

    # Adiabatic
    adiab_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for j in range(4):
        fig2.add_trace(go.Scatter(
            x=Delta_2atom, y=adiab_2[:, j], mode="lines",
            name=f"E_{j} (adiabatic)",
            line=dict(color=adiab_colors[j], width=ADIAB_WIDTH),
        ))

    # Sweep region
    fig2.add_vrect(
        x0=-sweep_half_MHz, x1=sweep_half_MHz,
        fillcolor="lightgray", opacity=0.15, line_width=0,
        annotation_text="sweep", annotation_position="top left",
    )

    # Current detuning marker
    fig2.add_trace(go.Scatter(
        x=[d_cur], y=[marker_2_y], mode="markers",
        marker=dict(size=14, color="black", symbol="circle",
                    line=dict(width=2, color="yellow")),
        name=f"Δ = {d_cur:.1f} MHz",
        showlegend=True,
    ))
    fig2.add_vline(x=d_cur, line_dash="dot", line_color="gray", opacity=0.6)

    # Crossing annotations
    fig2.add_annotation(
        x=0, y=0, text="1st crossing (δ_A splits |gr⟩/|rg⟩)",
        showarrow=True, arrowhead=2, ax=60, ay=-35,
        font=dict(size=10, color="gray"),
    )
    rr_crossing = delta_A_MHz + V_ryd_MHz
    if rr_crossing <= x_max:
        fig2.add_annotation(
            x=rr_crossing, y=-rr_crossing + delta_A_MHz,
            text=f"|rr⟩ wall at Δ = δ_A + V = {rr_crossing:.0f}",
            showarrow=True, arrowhead=2, ax=-60, ay=-35,
            font=dict(size=10, color="crimson"),
        )

    fig2.update_layout(
        title=dict(text="Two atoms  (4×4, with blockade)", font=dict(size=16)),
        xaxis_title="Detuning  Δ/(2π)  [MHz]",
        yaxis_title="Energy / (2π)  [MHz]",
        height=550,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=20, t=50, b=50),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Physics summary ───────────────────────────────────────────────────

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Ω_eff / (2π)", f"{Omega_eff_MHz:.2f} MHz")
c2.metric("δ_A / Ω_eff", f"{delta_A_MHz / Omega_eff_MHz:.1f}")
c3.metric("|rr⟩ crossing", f"Δ = {delta_A_MHz + V_ryd_MHz:.0f} MHz",
          delta=f"{V_ryd_MHz:.0f} from V_ryd")

inside_sweep = abs(delta_A_MHz) < sweep_half_MHz
st.markdown("---")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Single atom")
    if inside_sweep:
        st.warning(
            f"**Resonance at Δ = {delta_A_MHz:.0f} MHz is INSIDE the sweep "
            f"[{-sweep_half_MHz:.0f}, +{sweep_half_MHz:.0f}] MHz.**  "
            "The crossing is fully adiabatic "
            f"(P_LZ = {np.exp(-np.pi * Omega_eff_MHz**2 / (2 * 2 * sweep_half_MHz / 4.5)):.1e}) "
            "— the atom WILL transition to |r⟩."
        )
    else:
        st.success(
            f"Resonance at Δ = {delta_A_MHz:.0f} MHz is OUTSIDE the sweep "
            f"[{-sweep_half_MHz:.0f}, +{sweep_half_MHz:.0f}] MHz.  "
            "The atom stays in |g⟩ (pinning works)."
        )

with col_b:
    st.subheader("Two atoms")
    st.info(
        f"**Step 1 — Switch (δ_A):** At Δ ≈ 0, δ_A = {delta_A_MHz:.0f} MHz "
        f"splits |gr⟩ and |rg⟩ by {delta_A_MHz:.0f} MHz.  "
        "The adiabatic path follows the lower branch → |gr⟩.  \n"
        f"**Step 2 — Wall (V_ryd):** The |rr⟩ crossing is at "
        f"Δ = δ_A + V_ryd = {delta_A_MHz + V_ryd_MHz:.0f} MHz, "
        f"far beyond the sweep endpoint (+{sweep_half_MHz:.0f} MHz).  "
        "Atom A is permanently protected."
    )
