"""analog_3 (physical g/e/r ladder) on the YASTN PEPS + TeNPy MPS backends.

Covers the spec/blocks plumbing, end-to-end smokes (with the unitary
population-conservation invariant), a quantitative exact-vs-TN agreement check on
a 2-atom chain driven by an uncompensated double-ARP pulse, and the dt guard.
"""

import numpy as np
import pytest

from ryd_gate.analysis.observables import precompute_trit_masks
from ryd_gate.backends.peps2d import YASTNPEPSBackend
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR, tn_lattice_spec_from_system
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.protocol_context import TNProtocolContext
from ryd_gate.core.system import RydbergSystem
from ryd_gate.gates import CZProtocol, phase_from_chirp
from ryd_gate.lattice import Register
from ryd_gate.simulate import simulate


def _analog3_spec():
    return create_tn_lattice_spec(1, 2, V_nn=0.0, interaction_mode="nn", level_structure="analog_3")


def _short_arp(t_gate, n_steps=64):
    """A short uncompensated ARP pulse as a plain CZProtocol (analog_3-compatible:
    only the 420 leg is driven; 1013 is the static analog coupling)."""
    delta_max, t_pulse, sigma = 2 * np.pi * 23e6, 0.5 * t_gate, 0.175 * t_gate
    offset = np.exp(-((t_pulse / 2) ** 4) / sigma**4)

    def env(t):
        tc = float(np.clip(t, 0.0, t_gate)); u = tc if tc < t_pulse else tc - t_pulse
        return (np.exp(-((u - t_pulse / 2) ** 4) / sigma**4) - offset) / (1 - offset)

    def sweep(t):
        tc = float(np.clip(t, 0.0, t_gate)); u = tc if tc < t_pulse else tc - t_pulse
        return delta_max * np.sin(np.pi * (u / t_pulse - 0.5))

    phi = phase_from_chirp(sweep, t_gate, 4 * n_steps + 1)
    return CZProtocol(
        t_gate=t_gate,
        A_420=lambda s: env(float(np.clip(s, 0.0, 1.0)) * t_gate),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * t_gate),
        n_steps=n_steps,
    )


def _ir(spec, proto):
    ctx = TNProtocolContext(spec)
    pulse = proto.build([], ctx) if hasattr(proto, "build") else proto
    params = pulse.unpack_params([], ctx)
    return TNEvolutionIR(spec=spec, protocol=pulse, params=params, method="peps_yastn")


def _analog3_chain(n=2, spacing_um=5.0, t_gate_scale=2.0):
    """A 2-atom analog_3 system bound to a short uncompensated ARP pulse builder."""
    sys0 = RydbergSystem.set_atom_level("analog_3", detuning_sign=1).set_atom_geom(
        Register.chain(n, spacing_um=spacing_um)
    ).build()
    ts = sys0.meta("time_scale")
    return sys0.with_protocol(_short_arp(t_gate_scale * ts, n_steps=200)), ts


def _exact_ger_occupations(system, T):
    """Per-site ``(n_e, n_r)`` from the exact 3^N state at time ``T``."""
    psi = np.asarray(
        simulate(system, [], "all_ground", backend="exact_dense", t_eval=np.array([T])).psi_final
    ).reshape(-1)
    masks = precompute_trit_masks(system.N)
    return masks["e"] @ np.abs(psi) ** 2, masks["r"] @ np.abs(psi) ** 2


def test_analog3_spec_carries_local_blocks():
    lb = _analog3_spec().local_blocks
    assert lb is not None
    assert lb.static.shape == (3, 3) and lb.hermitian and lb.rydberg_index == 2
    # large static intermediate detuning on |e>, static e-r coupling off-diagonal
    assert np.isclose(lb.static[1, 1], 2 * np.pi * 9.1e9)
    assert np.isclose(lb.static[2, 1], np.pi * 491e6)  # rabi_1013 / 2


def test_yastn_peps_analog3_smoke():
    pytest.importorskip("yastn")
    spec = _analog3_spec()
    ts = spec.local_blocks.time_scale
    res = YASTNPEPSBackend(chi_max=4, dt=ts / 20).evolve_ir(
        _ir(spec, _short_arp(2 * ts)), initial_state="all_ground",
        t_eval=np.array([0.0, 2 * ts]), observables=["n_g", "n_e", "n_r", "n_mean"],
    )
    assert res.metadata["level_structure"] == "analog_3"
    assert res.metadata["local_dim"] == 3
    obs = res.metadata["obs"]
    # initial all-ground: n_g = 1, n_e = n_r = 0
    np.testing.assert_allclose(obs["n_g"][0], [1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(obs["n_r"][0], [0.0, 0.0], atol=1e-10)
    # unitary evolution conserves per-site population
    total = obs["n_g"][1] + obs["n_e"][1] + obs["n_r"][1]
    np.testing.assert_allclose(total, [1.0, 1.0], atol=1e-8)
    assert np.all(np.isfinite(obs["n_r"]))


def test_yastn_peps_analog3_matches_exact():
    pytest.importorskip("yastn")
    system, ts = _analog3_chain()
    T = system.protocol.t_gate
    ne_ex, nr_ex = _exact_ger_occupations(system, T)
    assert nr_ex.max() > 1e-3  # the pulse actually moves population (non-vacuous)

    spec = tn_lattice_spec_from_system(system)
    res = YASTNPEPSBackend(chi_max=8, dt=ts / 80).evolve_ir(
        _ir(spec, system.protocol), initial_state="all_ground",
        t_eval=np.array([T]), observables=["n_e", "n_r"],
    )
    np.testing.assert_allclose(res.metadata["obs"]["n_e"][0], ne_ex, atol=5e-3)
    np.testing.assert_allclose(res.metadata["obs"]["n_r"][0], nr_ex, atol=5e-3)


def test_tenpy_mps_analog3_smoke():
    pytest.importorskip("tenpy")
    from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend

    spec = _analog3_spec()
    ts = spec.local_blocks.time_scale
    res = TenpyTDVPBackend(chi_max=4, dt=ts / 20).evolve_ir(
        _ir(spec, _short_arp(2 * ts)), initial_state="all_ground",
        t_eval=np.array([0.0, 2 * ts]), observables=["n_g", "n_e", "n_r"],
    )
    obs = res.metadata["obs"]
    np.testing.assert_allclose(obs["n_g"][0], [1.0, 1.0], atol=1e-10)
    total = obs["n_g"][1] + obs["n_e"][1] + obs["n_r"][1]
    np.testing.assert_allclose(total, [1.0, 1.0], atol=1e-8)


def test_tenpy_mps_analog3_matches_exact():
    pytest.importorskip("tenpy")
    from ryd_gate.backends.tenpy_mps.backends import TenpyTDVPBackend

    system, ts = _analog3_chain()
    T = system.protocol.t_gate
    ne_ex, nr_ex = _exact_ger_occupations(system, T)

    spec = tn_lattice_spec_from_system(system)
    res = TenpyTDVPBackend(chi_max=8, dt=ts / 80).evolve_ir(
        _ir(spec, system.protocol), initial_state="all_ground",
        t_eval=np.array([T]), observables=["n_e", "n_r"],
    )
    np.testing.assert_allclose(res.metadata["obs"]["n_e"][0], ne_ex, atol=5e-3)
    np.testing.assert_allclose(res.metadata["obs"]["n_r"][0], nr_ex, atol=5e-3)


def test_analog3_dt_guard():
    pytest.importorskip("yastn")
    spec = _analog3_spec()
    ts = spec.local_blocks.time_scale
    # a natural-unit dt (>= time_scale) is catastrophically coarse -> raise
    with pytest.raises(ValueError, match="analog_3 needs a small TN step"):
        YASTNPEPSBackend(chi_max=4, dt=ts).evolve_ir(
            _ir(spec, _short_arp(2 * ts)), t_eval=np.array([2 * ts])
        )
