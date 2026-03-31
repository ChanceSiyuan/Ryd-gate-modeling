"""Local Addressing Analysis -- wavelength optimization and noise sensitivity.

Three tabs:
  1. Wavelength Physics  -- instant AC Stark shift / scattering / FOM vs wavelength
  2. Wavelength MC Sim   -- MC addressing quality (pinning/crosstalk/leakage) vs wavelength
  3. Noise Sensitivity   -- individual + combined noise sweeps
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.constants import pi

from ryd_gate.analysis.local_addressing import (
    BASELINE_AMP,
    BASELINE_DETUNING_HZ,
    BASELINE_RIN,
    COMBINED_SCALE_MAX,
    DEFAULT_LOCAL_DETUNING,
    DEFAULT_LOCAL_SCATTER,
    default_sweep_x,
    evaluate_addressing,
)
from ryd_gate.core.atomic_system import (
    LAMBDA_D1,
    LAMBDA_D2,
    LAMBDA_PAPER,
    POWER_REF_UW,
    build_sss_state_map,
    compute_shift_scatter,
    create_analog_system,
)
from ryd_gate.protocols.sweep import SweepProtocol

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
def _setup_system_cached():
    system = create_analog_system(detuning_sign=1)
    initial_state = build_sss_state_map(n_levels=3)["00"]
    return system, initial_state


@st.cache_data
def _wavelength_scan(wl_min, wl_max, n_wl, power_scale):
    """Cached vectorized wavelength scan."""
    wavelengths = np.linspace(wl_min, wl_max, n_wl)
    shifts, scatters = compute_shift_scatter(wavelengths)
    return wavelengths, shifts * power_scale, scatters * power_scale


# ── Tabs ────────────────────────────────────────────────────────────────

tab_phys, tab_mc, tab_noise = st.tabs([
    "Wavelength Physics", "Wavelength MC Sim", "Noise Sensitivity"])


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
# Tab 2: Wavelength MC Sim
# =====================================================================

with tab_mc:
    st.subheader("Addressing Quality vs Wavelength (MC Simulation)")

    st.markdown("""
    At each wavelength, the AC Stark shift and scattering rate are fed into
    the full two-atom addressing MC simulation to measure **pinning error**,
    **crosstalk**, and **leakage**.
    """)

    col_mc_cfg, col_mc_res = st.columns([1, 3])

    with col_mc_cfg:
        mc_wl_start = st.number_input("Start wavelength (nm)", value=781.0, step=0.5,
                                       key="mc_wl_start")
        mc_wl_end = st.number_input("End wavelength (nm)", value=786.0, step=0.5,
                                     key="mc_wl_end")
        mc_wl_step = st.number_input("Step (nm)", value=1.0, step=0.5, min_value=0.5,
                                      key="mc_wl_step")
        mc_n_shots = st.slider("MC shots per wavelength", 5, 100, 20, step=5,
                                key="mc_n_shots")
        mc_sigma_det = st.number_input("Detuning noise \u03c3 (kHz)", value=130,
                                        step=10, key="mc_sigma_det")
        run_mc_wl = st.button("Run Wavelength MC", type="primary",
                               use_container_width=True, key="btn_mc_wl")

    if run_mc_wl:
        system, initial_state = _setup_system_cached()
        x = default_sweep_x(system)
        sample_lams = np.arange(mc_wl_start, mc_wl_end + 0.01, mc_wl_step)
        sample_shifts, sample_scatters = compute_shift_scatter(sample_lams)

        pin_errs, xtalk_errs, leak_errs = [], [], []
        progress = st.progress(0.0)
        for k, lam in enumerate(sample_lams):
            progress.progress((k + 1) / len(sample_lams),
                               text=f"Simulating {lam:.1f} nm...")
            protocol = SweepProtocol(
                addressing={0: 2 * pi * sample_shifts[k]},
                scatter_rate=sample_scatters[k],
            )
            pin, xtalk, leak = evaluate_addressing(
                system, initial_state, protocol, x,
                {"sigma_detuning": mc_sigma_det * 1e3},
                mc_n_shots, seed=42)
            pin_errs.append(pin)
            xtalk_errs.append(xtalk)
            leak_errs.append(leak)
        progress.progress(1.0, text="Done!")

        st.session_state["mc_wl_results"] = {
            "lams": sample_lams, "shifts": sample_shifts,
            "pin": pin_errs, "xtalk": xtalk_errs, "leak": leak_errs,
        }

    with col_mc_res:
        if "mc_wl_results" in st.session_state:
            r = st.session_state["mc_wl_results"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r["lams"], y=r["pin"], name="Pinning",
                                      mode="lines+markers", line=dict(color="red"),
                                      marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=r["lams"], y=r["xtalk"], name="Crosstalk",
                                      mode="lines+markers", line=dict(color="blue"),
                                      marker=dict(size=7)))
            fig.add_trace(go.Scatter(x=r["lams"], y=r["leak"], name="Leakage",
                                      mode="lines+markers", line=dict(color="orange"),
                                      marker=dict(size=7)))
            fig.add_vline(x=LAMBDA_PAPER, line_dash="dot", line_color="green",
                          opacity=0.6, annotation_text=f"{LAMBDA_PAPER:.0f} nm")
            fig.update_layout(
                xaxis_title="Wavelength (nm)", yaxis_title="Error",
                yaxis_type="log", height=450,
                margin=dict(l=60, r=20, t=30, b=40))
            st.plotly_chart(fig, use_container_width=True)

            tbl = [{"Wavelength (nm)": f"{lam:.1f}",
                    "Shift (MHz)": f"{s/1e6:.2f}",
                    "Pinning": f"{p:.4f}", "Crosstalk": f"{x:.4f}",
                    "Leakage": f"{l:.4f}"}
                   for lam, s, p, x, l in zip(
                       r["lams"], r["shifts"], r["pin"], r["xtalk"], r["leak"])]
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Configure wavelength range and MC shots, then click "
                    "**Run Wavelength MC**. Each shot takes ~2 min.")


# =====================================================================
# Tab 3: Noise Sensitivity
# =====================================================================

with tab_noise:
    st.subheader("Noise Impact on Addressing Quality")

    st.markdown("""
    Sweep each noise source independently (detuning, local RIN, amplitude)
    and optionally run a combined sweep with all sources scaled together.
    """)

    col_ncfg, col_nres = st.columns([1, 3])

    with col_ncfg:
        ns_n_mc = st.slider("MC shots per point", 5, 100, 20, step=5, key="ns_n_mc")
        ns_n_pts = st.slider("Scan points", 3, 12, 6, key="ns_n_pts")
        st.markdown("---")
        st.markdown("**Individual scan ranges**")
        ns_det_max = st.number_input("Max detuning noise (kHz)", value=500, step=50,
                                      key="ns_det_max")
        ns_rin_max = st.number_input("Max local RIN (%)", value=5.0, step=0.5,
                                      key="ns_rin_max")
        ns_amp_max = st.number_input("Max amplitude noise (%)", value=5.0, step=0.5,
                                      key="ns_amp_max")
        st.markdown("---")
        ns_combined = st.checkbox("Include combined sweep", value=True, key="ns_combined")
        run_noise = st.button("Run Noise Scan", type="primary",
                               use_container_width=True, key="btn_noise")

    if run_noise:
        system, initial_state = _setup_system_cached()
        protocol = SweepProtocol(
            addressing={0: DEFAULT_LOCAL_DETUNING},
            scatter_rate=DEFAULT_LOCAL_SCATTER,
        )
        x = default_sweep_x(system)

        scans = {
            "Global Detuning": {
                "param": "sigma_detuning",
                "values": np.linspace(0, ns_det_max * 1e3, ns_n_pts),
                "xlabel": "\u03c3_\u0394 (kHz)", "xscale": 1e-3,
            },
            "Local RIN": {
                "param": "sigma_local_rin",
                "values": np.linspace(0, ns_rin_max / 100, ns_n_pts),
                "xlabel": "\u03c3_RIN (%)", "xscale": 100,
            },
            "Amplitude Noise": {
                "param": "sigma_amplitude",
                "values": np.linspace(0, ns_amp_max / 100, ns_n_pts),
                "xlabel": "\u03c3_\u03a9 (%)", "xscale": 100,
            },
        }

        total_evals = sum(len(c["values"]) for c in scans.values())
        if ns_combined:
            total_evals += ns_n_pts
        progress = st.progress(0.0)
        done = 0

        noise_results = {}
        for name, cfg in scans.items():
            pin_errs, xtalk_errs, leak_errs = [], [], []
            for val in cfg["values"]:
                kwargs = {"sigma_detuning": 0.0, "sigma_local_rin": 0.0,
                          "sigma_amplitude": 0.0}
                kwargs[cfg["param"]] = val
                pin, xtalk, leak = evaluate_addressing(
                    system, initial_state, protocol, x, kwargs, ns_n_mc)
                pin_errs.append(pin)
                xtalk_errs.append(xtalk)
                leak_errs.append(leak)
                done += 1
                progress.progress(done / total_evals, text=f"{name}: {done}/{total_evals}")
            noise_results[name] = {
                "values": cfg["values"], "xlabel": cfg["xlabel"],
                "xscale": cfg["xscale"],
                "pin": pin_errs, "xtalk": xtalk_errs, "leak": leak_errs,
            }

        combined_result = None
        if ns_combined:
            scale_factors = np.linspace(0, COMBINED_SCALE_MAX, ns_n_pts)
            pin_c, xtalk_c, leak_c = [], [], []
            for scale in scale_factors:
                pin, xtalk, leak = evaluate_addressing(
                    system, initial_state, protocol, x,
                    {"sigma_detuning": BASELINE_DETUNING_HZ * scale,
                     "sigma_local_rin": BASELINE_RIN * scale,
                     "sigma_amplitude": BASELINE_AMP * scale},
                    ns_n_mc)
                pin_c.append(pin)
                xtalk_c.append(xtalk)
                leak_c.append(leak)
                done += 1
                progress.progress(done / total_evals, text=f"Combined: {done}/{total_evals}")
            combined_result = {"scales": scale_factors,
                               "pin": pin_c, "xtalk": xtalk_c, "leak": leak_c}

        progress.progress(1.0, text="Done!")
        st.session_state["noise_results"] = noise_results
        st.session_state["noise_combined"] = combined_result

    with col_nres:
        if "noise_results" in st.session_state:
            noise_results = st.session_state["noise_results"]
            combined_result = st.session_state.get("noise_combined")

            n_panels = 4 if combined_result else 3
            titles = list(noise_results.keys())
            if combined_result:
                titles.append("Combined (all sources)")
            fig = make_subplots(rows=2, cols=2, subplot_titles=titles,
                                 horizontal_spacing=0.1, vertical_spacing=0.15)

            for idx, (name, r) in enumerate(noise_results.items()):
                row, col = idx // 2 + 1, idx % 2 + 1
                show_leg = idx == 0
                xvals = np.array(r["values"]) * r["xscale"]
                for metric, color, lg in [("pin", "red", "pin"),
                                           ("xtalk", "blue", "xtalk"),
                                           ("leak", "orange", "leak")]:
                    fig.add_trace(go.Scatter(
                        x=xvals, y=r[metric],
                        name={"pin": "Pinning", "xtalk": "Crosstalk", "leak": "Leakage"}[metric],
                        mode="lines+markers", line=dict(color=color),
                        marker=dict(size=5), showlegend=show_leg, legendgroup=lg,
                    ), row=row, col=col)
                fig.update_xaxes(title_text=r["xlabel"], row=row, col=col)
                fig.update_yaxes(title_text="Error", row=row, col=col)

            if combined_result:
                for metric, color, lg in [("pin", "red", "pin"),
                                           ("xtalk", "blue", "xtalk"),
                                           ("leak", "orange", "leak")]:
                    fig.add_trace(go.Scatter(
                        x=combined_result["scales"], y=combined_result[metric],
                        name={"pin": "Pinning", "xtalk": "Crosstalk", "leak": "Leakage"}[metric],
                        mode="lines+markers", line=dict(color=color),
                        marker=dict(size=5), showlegend=False, legendgroup=lg,
                    ), row=2, col=2)
                fig.add_vline(x=1.0, line_dash="dot", line_color="green",
                              opacity=0.6, row=2, col=2, annotation_text="baseline")
                fig.update_xaxes(title_text="Noise scale factor", row=2, col=2)
                fig.update_yaxes(title_text="Error", row=2, col=2)
            else:
                # Hide empty 4th panel
                fig.update_xaxes(visible=False, row=2, col=2)
                fig.update_yaxes(visible=False, row=2, col=2)

            fig.update_layout(height=700, margin=dict(l=60, r=20, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"Each point averaged over {ns_n_mc} MC shots.")
        else:
            st.info("Configure noise ranges and click **Run Noise Scan**. "
                    "Each MC shot takes ~2 min \u2014 start with few shots and points.")
