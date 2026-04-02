"""Local Addressing Analysis -- wavelength optimization and protocol explorer.

Two tabs:
  1. Wavelength Physics  -- instant AC Stark shift / scattering / FOM vs wavelength
  2. Protocol Explorer   -- Rabi dynamics, adiabatic sweep, and AC Stark compensation
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.constants import pi

from ryd_gate.core.atomic_system import (
    LAMBDA_D1,
    LAMBDA_D2,
    LAMBDA_PAPER,
    POWER_REF_UW,
    build_product_state_map,
    compute_shift_scatter,
)
from ryd_gate import simulate
from ryd_gate.analysis.observable_metrics import measure_trajectory, norm_squared
from ryd_gate.core.models.analog_3level import Analog3LevelModel
from ryd_gate.protocols.sweep import SweepProtocol

# Sweep constants (matching scripts/run_local_sweep.py)
DELTA_START = -2 * pi * 40e6   # rad/s
DELTA_END = 2 * pi * 40e6      # rad/s
T_GATE = 1.5e-6                # s
RYDBERG_KEYS = ["gg", "gr", "rg", "rr"]

st.set_page_config(page_title="Local Addressing", layout="wide", page_icon="\u269b")
st.title("Local Addressing Analysis")

st.markdown(f"""
Analyze the local addressing laser for site-selective AC Stark shifts on Rb87.
The paper uses **{LAMBDA_PAPER:.0f} nm** with {POWER_REF_UW:.0f} \u03bcW/spot
(1 \u03bcm waist), achieving \u03b4\u2080 \u2248 \u221212.2 MHz differential
shift and ~35 Hz scattering.
""")


# ── Cached helpers ──────────────────────────────────────────────────────

@st.cache_resource
def _setup_model_cached(distance_um: float = 3.0):
    model = Analog3LevelModel.from_defaults(detuning_sign=1, distance_um=distance_um)
    initial_state = build_product_state_map(n_levels=3)["gg"]
    return model, initial_state


@st.cache_data
def _wavelength_scan(wl_min, wl_max, n_wl, power_scale):
    """Cached vectorized wavelength scan."""
    wavelengths = np.linspace(wl_min, wl_max, n_wl)
    shifts, scatters = compute_shift_scatter(wavelengths)
    return wavelengths, shifts * power_scale, scatters * power_scale


# ── Tabs ────────────────────────────────────────────────────────────────

tab_phys, tab_proto = st.tabs(["Wavelength Physics", "Protocol Explorer"])


# =====================================================================
# Tab 1: Wavelength Physics (instant)
# =====================================================================

with tab_phys:
    st.subheader("AC Stark Shift & Scattering vs Wavelength")

    col_cfg, col_plot = st.columns([1, 3])

    with col_cfg:
        wl_min = st.slider("Min wavelength (nm)", 780.5, 783.0, 780.5, step=0.1,
                            key="phys_wl_min")
        wl_max = st.slider("Max wavelength (nm)", 783.0, 790.0, 786.0, step=0.1,
                            key="phys_wl_max")
        n_wl = st.slider("Points", 50, 500, 200, key="phys_n")
        power_uw = st.number_input("Power per spot (\u03bcW)", value=POWER_REF_UW,
                                    step=10.0, key="phys_power")
        power_scale = power_uw / POWER_REF_UW
        st.markdown("---")
        st.caption(f"D2 line: {LAMBDA_D2:.3f} nm")
        st.caption(f"D1 line: {LAMBDA_D1:.3f} nm")

    wavelengths, shifts_Hz, scatters_Hz = _wavelength_scan(wl_min, wl_max, n_wl, power_scale)
    shifts_MHz = shifts_Hz / 1e6
    fom = np.abs(shifts_Hz) / np.maximum(scatters_Hz, 1e-10)

    with col_plot:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=["AC Stark Shift", "Scattering Rate",
                                             "Figure of Merit"])
        fig.add_trace(go.Scatter(x=wavelengths, y=shifts_MHz, name="Shift",
                                  line=dict(color="steelblue", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=wavelengths, y=scatters_Hz, name="Scatter",
                                  line=dict(color="crimson", width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=wavelengths, y=fom / 1e6, name="FOM",
                                  line=dict(color="mediumpurple", width=2)), row=3, col=1)
        for row in [1, 2, 3]:
            fig.add_vline(x=LAMBDA_D2, line_dash="dash", line_color="red",
                          opacity=0.4, row=row, col=1)
            fig.add_vline(x=LAMBDA_PAPER, line_dash="dot", line_color="green",
                          opacity=0.6, row=row, col=1)
        fig.update_yaxes(title_text="Shift (MHz)", row=1)
        fig.update_yaxes(title_text="Scatter (Hz)", type="log", row=2)
        fig.update_yaxes(title_text="|Shift|/Scatter (\u00d710\u2076)", row=3)
        fig.update_xaxes(title_text="Wavelength (nm)", row=3)
        fig.update_layout(height=700, showlegend=False,
                          margin=dict(l=60, r=20, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # Table at key wavelengths
    st.markdown("#### Values at Selected Wavelengths")
    key_lams = np.arange(max(781.0, np.ceil(wl_min)), wl_max + 0.1, 1.0)
    key_shifts, key_scatters = compute_shift_scatter(key_lams)
    key_shifts *= power_scale
    key_scatters *= power_scale
    key_fom = np.abs(key_shifts) / np.maximum(key_scatters, 1e-10)
    rows = [{"Wavelength (nm)": f"{lam:.1f}",
             "Shift (MHz)": f"{s/1e6:.2f}",
             "Scatter (Hz)": f"{sc:.1f}",
             "FOM (\u00d710\u2076)": f"{f/1e6:.2f}"}
            for lam, s, sc, f in zip(key_lams, key_shifts, key_scatters, key_fom)]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    opt_idx = np.argmax(fom)
    st.success(f"Optimal wavelength: **{wavelengths[opt_idx]:.1f} nm** "
               f"(FOM = {fom[opt_idx]/1e6:.2f}\u00d710\u2076, "
               f"shift = {shifts_MHz[opt_idx]:.2f} MHz, "
               f"scatter = {scatters_Hz[opt_idx]:.1f} Hz)")


# =====================================================================
# Tab 2: Protocol Explorer
# =====================================================================

with tab_proto:
    st.subheader("Protocol Response at Chosen Wavelength")

    st.markdown("""
    Pick a wavelength and power to compute the AC Stark shift and scattering
    rate, then run deterministic simulations showing Rabi dynamics, adiabatic
    sweep populations, and AC Stark feed-forward compensation.
    """)

    col_pcfg, col_pres = st.columns([1, 3])

    with col_pcfg:
        proto_wl = st.slider("Wavelength (nm)", 780.5, 786.0, 784.0, step=0.1,
                              key="proto_wl")
        proto_power = st.number_input("Power per spot (\u03bcW)", value=POWER_REF_UW,
                                       step=10.0, key="proto_power")
        proto_dist = st.slider("Atom distance (\u03bcm)", 1.0, 10.0, 3.0, step=0.1,
                                key="proto_dist")
        power_scale_p = proto_power / POWER_REF_UW

        shift_Hz_arr, scatter_Hz_arr = compute_shift_scatter(np.array([proto_wl]))
        shift_Hz_p = float(shift_Hz_arr[0]) * power_scale_p
        scatter_Hz_p = float(scatter_Hz_arr[0]) * power_scale_p
        local_detuning_p = 2 * pi * shift_Hz_p   # rad/s (negative for red-detuned)
        local_scatter_p = scatter_Hz_p            # Hz

        # V_ryd at chosen distance
        v_ryd_p = 2 * pi * 874e9 / proto_dist**6
        st.metric("AC Stark shift", f"{shift_Hz_p / 1e6:.2f} MHz")
        st.metric("Scattering rate", f"{scatter_Hz_p:.1f} Hz")
        st.metric("V_ryd", f"{v_ryd_p / (2 * pi) / 1e6:.1f} MHz")
        st.markdown("---")
        run_proto = st.button("Run Simulation", type="primary",
                               use_container_width=True, key="btn_proto")

    if run_proto:
        model, initial_state_m = _setup_model_cached(distance_um=proto_dist)
        system = model.system

        protocol_free = SweepProtocol()
        protocol_addr = SweepProtocol(
            addressing={0: local_detuning_p},
            scatter_rate=local_scatter_p)

        # --- A. Rabi dynamics (resonant driving) ---
        n_cycles = 5
        t_rabi = n_cycles * system.time_scale
        x_rabi = [0.0, 0.0, t_rabi / system.time_scale]
        n_pts = 500
        t_eval_rabi = np.linspace(0, t_rabi, n_pts)

        result_free_rabi = simulate(system, protocol_free, x_rabi,
                                    initial_state_m, t_eval=t_eval_rabi)
        result_addr_rabi = simulate(system, protocol_addr, x_rabi,
                                    initial_state_m, t_eval=t_eval_rabi)

        obs_names_rabi = ["pop_A_r", "pop_B_r"]
        obs_free_rabi = measure_trajectory(model, result_free_rabi.states, obs_names_rabi)
        obs_addr_rabi = measure_trajectory(model, result_addr_rabi.states, obs_names_rabi)

        # Norm for addressed case (column-major: dim x n_t)
        if result_addr_rabi.states.shape[0] == model.basis.total_dim:
            norm_addr = np.array([norm_squared(result_addr_rabi.states[:, k])
                                  for k in range(result_addr_rabi.states.shape[1])])
        else:
            norm_addr = np.array([norm_squared(result_addr_rabi.states[k])
                                  for k in range(result_addr_rabi.states.shape[0])])

        # --- B. Final populations (adiabatic sweep) ---
        x_sweep = [
            DELTA_START / system.rabi_eff,
            DELTA_END / system.rabi_eff,
            T_GATE / system.time_scale,
        ]
        res_sweep_free = simulate(system, protocol_free, x_sweep, initial_state_m)
        res_sweep_addr = simulate(system, protocol_addr, x_sweep, initial_state_m)

        pops_free = {k: model.observables.measure(f"pop_{k}", res_sweep_free.psi_final)
                     for k in RYDBERG_KEYS}
        pops_addr = {k: model.observables.measure(f"pop_{k}", res_sweep_addr.psi_final)
                     for k in RYDBERG_KEYS}

        # --- C. AC Stark feed-forward compensation ---
        ac_stark_peak = system.rabi_420 ** 2 / (4 * abs(system.Delta))
        t_gate_phys = x_sweep[2] * system.time_scale
        t_eval_sweep = np.linspace(0, t_gate_phys, n_pts)

        protocol_raw = SweepProtocol(ac_stark_shift=0.0)
        protocol_comp = SweepProtocol(ac_stark_shift=ac_stark_peak)

        result_raw = simulate(system, protocol_raw, x_sweep, initial_state_m,
                              t_eval=t_eval_sweep)
        result_comp = simulate(system, protocol_comp, x_sweep, initial_state_m,
                               t_eval=t_eval_sweep)

        obs_names_stark = ["pop_A_r", "pop_B_r", "pop_r"]
        obs_raw = measure_trajectory(model, result_raw.states, obs_names_stark)
        obs_comp = measure_trajectory(model, result_comp.states, obs_names_stark)

        pops_raw = {k: model.observables.measure(f"pop_{k}", result_raw.psi_final)
                    for k in RYDBERG_KEYS}
        pops_comp = {k: model.observables.measure(f"pop_{k}", result_comp.psi_final)
                     for k in RYDBERG_KEYS}

        st.session_state["proto_results"] = {
            "t_us_free": result_free_rabi.times * 1e6,
            "t_us_addr": result_addr_rabi.times * 1e6,
            "obs_free": obs_free_rabi,
            "obs_addr": obs_addr_rabi,
            "norm_addr": norm_addr,
            "pops_free": pops_free,
            "pops_addr": pops_addr,
            "t_us_stark": result_raw.times * 1e6,
            "obs_raw": obs_raw,
            "obs_comp": obs_comp,
            "pops_raw": pops_raw,
            "pops_comp": pops_comp,
            "ac_stark_peak_MHz": ac_stark_peak / (2 * pi) / 1e6,
            "omega_eff_MHz": system.rabi_eff / (2 * pi) / 1e6,
            "shift_MHz": shift_Hz_p / 1e6,
            "scatter_Hz": scatter_Hz_p,
            "wavelength": proto_wl,
        }

    with col_pres:
        if "proto_results" in st.session_state:
            r = st.session_state["proto_results"]

            # --- Figure 1: Rabi dynamics ---
            st.markdown("#### Rabi Dynamics (resonant driving)")
            fig_rabi = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=[
                    "(a) Global excitation \u2014 blockade active",
                    "(b) Atom A addressed \u2014 free oscillation"])

            fig_rabi.add_trace(go.Scatter(
                x=r["t_us_free"], y=r["obs_free"]["pop_A_r"],
                name="P_r(A)", line=dict(color="#2ca02c")), row=1, col=1)
            fig_rabi.add_trace(go.Scatter(
                x=r["t_us_free"], y=r["obs_free"]["pop_B_r"],
                name="P_r(B)", line=dict(color="#1f77b4", dash="dash")),
                row=1, col=1)

            fig_rabi.add_trace(go.Scatter(
                x=r["t_us_addr"], y=r["obs_addr"]["pop_A_r"],
                name="P_r(A)", line=dict(color="#2ca02c"),
                showlegend=False), row=2, col=1)
            fig_rabi.add_trace(go.Scatter(
                x=r["t_us_addr"], y=r["obs_addr"]["pop_B_r"],
                name="P_r(B)", line=dict(color="#1f77b4", dash="dash"),
                showlegend=False), row=2, col=1)
            fig_rabi.add_trace(go.Scatter(
                x=r["t_us_addr"], y=r["norm_addr"],
                name="Norm", line=dict(color="gray", dash="dot")),
                row=2, col=1)

            fig_rabi.update_yaxes(title_text="Population", range=[-0.05, 1.05], row=1)
            fig_rabi.update_yaxes(title_text="Population", range=[-0.05, 1.05], row=2)
            fig_rabi.update_xaxes(title_text="Time (\u03bcs)", row=2)
            fig_rabi.update_layout(height=550, margin=dict(l=60, r=20, t=40, b=40))
            st.plotly_chart(fig_rabi, use_container_width=True)

            # --- Figure 2: Final populations ---
            st.markdown("#### Final State Populations (adiabatic sweep)")
            bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
            tick_labels = [f"|{k}\u27e9" for k in RYDBERG_KEYS]

            fig_pops = make_subplots(
                rows=1, cols=2, subplot_titles=[
                    "(a) Global sweep \u2014 no pinning",
                    "(b) Atom A addressed \u2014 pinned to |g\u27e9"])

            vals_free = [r["pops_free"][k] for k in RYDBERG_KEYS]
            vals_addr = [r["pops_addr"][k] for k in RYDBERG_KEYS]

            fig_pops.add_trace(go.Bar(
                x=tick_labels, y=vals_free, marker_color=bar_colors,
                name="Global", text=[f"{v:.3f}" for v in vals_free],
                textposition="outside"), row=1, col=1)
            fig_pops.add_trace(go.Bar(
                x=tick_labels, y=vals_addr, marker_color=bar_colors,
                name="Addressed", text=[f"{v:.3f}" for v in vals_addr],
                textposition="outside"), row=1, col=2)

            fig_pops.update_yaxes(title_text="Population", range=[0, 1.15], row=1, col=1)
            fig_pops.update_yaxes(range=[0, 1.15], row=1, col=2)
            fig_pops.update_layout(height=400, showlegend=False,
                                    margin=dict(l=60, r=20, t=40, b=40))
            st.plotly_chart(fig_pops, use_container_width=True)

            # --- Figure 3: AC Stark compensation ---
            st.markdown(
                f"#### AC Stark Feed-Forward Compensation "
                f"(\u0394_AC^peak = {r['ac_stark_peak_MHz']:.1f} MHz, "
                f"\u03a9_eff = {r['omega_eff_MHz']:.1f} MHz)")

            fig_stark = make_subplots(
                rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.1,
                subplot_titles=[
                    "(a) Without compensation",
                    "(b) With feed-forward compensation",
                    "(c) Final pops \u2014 no compensation",
                    "(d) Final pops \u2014 compensated"])

            for col_idx, (obs_key, pops_key) in enumerate([
                ("obs_raw", "pops_raw"), ("obs_comp", "pops_comp")
            ], start=1):
                obs = r[obs_key]
                show_leg = col_idx == 1
                fig_stark.add_trace(go.Scatter(
                    x=r["t_us_stark"], y=obs["pop_A_r"],
                    name="P_r(A)", line=dict(color="#2ca02c"),
                    showlegend=show_leg, legendgroup="prA"),
                    row=1, col=col_idx)
                fig_stark.add_trace(go.Scatter(
                    x=r["t_us_stark"], y=obs["pop_B_r"],
                    name="P_r(B)", line=dict(color="#1f77b4", dash="dash"),
                    showlegend=show_leg, legendgroup="prB"),
                    row=1, col=col_idx)
                fig_stark.add_trace(go.Scatter(
                    x=r["t_us_stark"], y=np.array(obs["pop_r"]) / 2,
                    name="\u27e8n_r\u27e9/2", line=dict(color="gray", dash="dot"),
                    showlegend=show_leg, legendgroup="nr"),
                    row=1, col=col_idx)

                pops = r[pops_key]
                vals = [pops[k] for k in RYDBERG_KEYS]
                fig_stark.add_trace(go.Bar(
                    x=tick_labels, y=vals, marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                    showlegend=False), row=2, col=col_idx)

            for col_idx in [1, 2]:
                fig_stark.update_yaxes(title_text="Population",
                                        range=[-0.05, 1.05], row=1, col=col_idx)
                fig_stark.update_xaxes(title_text="Time (\u03bcs)", row=1, col=col_idx)
                fig_stark.update_yaxes(title_text="Population",
                                        range=[0, 1.15], row=2, col=col_idx)

            fig_stark.update_layout(height=700, margin=dict(l=60, r=20, t=40, b=40))
            st.plotly_chart(fig_stark, use_container_width=True)

            st.info(
                f"Wavelength: {r['wavelength']:.1f} nm | "
                f"Shift: {r['shift_MHz']:.2f} MHz | "
                f"Scatter: {r['scatter_Hz']:.1f} Hz")
        else:
            st.info("Select a wavelength and power, then click "
                    "**Run Simulation** to see Rabi dynamics, adiabatic "
                    "sweep populations, and AC Stark compensation.")
