"""Rydberg Gate Desktop -- interactive Rydberg atom gate simulation tool.

Launch:
    uv run streamlit run app/rydberg_desktop.py
"""

import streamlit as st

st.set_page_config(
    page_title="Rydberg Gate Desktop",
    layout="wide",
    page_icon="\u269b",
    initial_sidebar_state="expanded",
)

st.title("\u269b Rydberg Gate Desktop")

st.markdown("""
Interactive tool for simulating Rydberg atom entangling gates and
many-body dynamics. No code required -- configure parameters in the
sidebar, click **Run**, and see results instantly.

### Pages

- **CZ Gate Simulator** -- Design pulse waveforms for a two-qubit CZ gate,
  visualize the phase modulation, and compute gate fidelity with population
  evolution diagnostics.

- **Lattice Simulator** -- Set up a 2D square lattice of Rydberg atoms and
  simulate quantum coarsening (domain shrinking) or Higgs mode oscillations
  from the Manovitz et al. experiment.

- **Error Analysis** -- Compute deterministic error budgets (Rydberg decay,
  intermediate decay, polarization leakage) with XYZ/AL/LG decomposition,
  and run Monte Carlo noise simulations.

- **Local Addressing** -- Scan the local addressing laser wavelength
  (780--786 nm) to optimize the AC Stark shift vs scattering tradeoff,
  and study how detuning noise, intensity noise (RIN), and amplitude
  noise affect addressing quality.

---
*Built on the `ryd_gate` package. Use the sidebar to navigate between pages.*
""")
