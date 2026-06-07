"""Tests for the PEPSKit.jl iPEPS real-time backend.

Fast tests (always run) cover the dispatch wiring and the iPEPS payload builder /
drive reduction. The physics-validation tests run the real Julia kernel (slow cold
JIT) and are opt-in via ``RYD_RUN_JULIA_TESTS=1``.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from ryd_gate.backends.pepskit.backend import (
    PEPSKitDriveError,
    PEPSKitIPEPSBackend,
    _initial_state_payload,
    _reduce_sublattice,
    _reduce_uniform,
    build_pepskit_payload,
)
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import (
    _default_method_for_backend,
    _normalize_backend,
    _protocol_context,
)
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol

RUN_JULIA = os.environ.get("RYD_RUN_JULIA_TESTS") == "1"
julia_only = pytest.mark.skipif(
    not RUN_JULIA, reason="set RYD_RUN_JULIA_TESTS=1 to run the PEPSKit Julia kernel (slow cold JIT)"
)


def _make_ir(level_structure, V_nn, proto):
    spec = create_tn_lattice_spec(2, 2, level_structure=level_structure, V_nn=V_nn, interaction_mode="nn")
    params = proto.unpack_params([], _protocol_context(spec))
    return TNEvolutionIR(
        spec=spec, protocol=proto, params=params, method="pepskit_ipeps_su",
        metadata={"compiler": "tn", "tn_spec": spec, "backend": "pepskit", "n_sites": spec.N},
    )


def _payload(ir, **overrides):
    kw = dict(
        initial_state="all_ground", t_eval=np.array([0.0, 0.1, 0.2]), observables=["n_r", "n_1"],
        dt=0.05, unit_cell="uniform", bond_dim=2, env_dim=12, trotter_order=2,
        ctmrg_tol=1e-8, ctmrg_maxiter=200, su_trunc_atol=1e-10, init_noise=1e-3,
    )
    kw.update(overrides)
    return build_pepskit_payload(ir, **kw)


# ---------------------------------------------------------------------------
# Dispatch wiring + payload builder (fast, always run)
# ---------------------------------------------------------------------------
def test_normalize_and_default_method():
    assert _normalize_backend("pepskit") == "pepskit"
    assert _normalize_backend("ipeps") == "pepskit"
    assert _default_method_for_backend("pepskit", "tdvp") == "pepskit_ipeps_su"


def test_build_payload_1r():
    ir = _make_ir("1r", 2.0, TFIMQuenchProtocol(hx=1.0, t_gate=0.2))  # omega_R = 2*hx = 2.0
    p = _payload(ir)
    assert p["lattice"]["physical_dim"] == 2
    assert p["lattice"]["levels"] == ["1", "r"]
    assert p["lattice"]["V_nn"] == 2.0
    assert p["initial_state"] == {"pattern": "uniform", "label": "1"}
    assert set(p["schedule"][0]) == {"step", "t_mid", "omega_R", "omega_hf", "delta_R", "delta_hf"}
    assert p["schedule"][0]["omega_R"] == pytest.approx(2.0)
    assert p["schedule"][0]["omega_hf"] == 0.0
    assert 0 in p["record_steps"]


def test_build_payload_01r_carries_hyperfine():
    proto = DigitalAnalogProtocol.constant(omega_R=1.0, omega_hf=0.5, delta_R=0.3, delta_hf=0.2, t_gate=0.2, n_steps=4)
    ir = _make_ir("01r", 2.0, proto)
    p = _payload(ir)
    assert p["lattice"]["physical_dim"] == 3
    assert p["lattice"]["levels"] == ["0", "1", "r"]
    s0 = p["schedule"][0]
    assert s0["omega_R"] == pytest.approx(1.0)
    assert s0["omega_hf"] == pytest.approx(0.5)
    assert s0["delta_R"] == pytest.approx(0.3)
    assert s0["delta_hf"] == pytest.approx(0.2)


def test_sublattice_schedule_has_ab_blocks():
    ir = _make_ir("1r", 2.0, TFIMQuenchProtocol(hx=1.0, t_gate=0.2))
    p = _payload(ir, unit_cell="sublattice")
    s0 = p["schedule"][0]
    assert set(s0) == {"step", "t_mid", "A", "B"}
    assert set(s0["A"]) == {"omega_R", "omega_hf", "delta_R", "delta_hf"}


def test_reduce_uniform_rejects_site_dependence():
    assert _reduce_uniform(np.full(4, 1.5), "delta_R") == pytest.approx(1.5)
    with pytest.raises(PEPSKitDriveError):
        _reduce_uniform(np.array([1.0, 2.0, 1.0, 1.0]), "delta_R")


def test_reduce_sublattice_splits_and_guards():
    spec = SimpleNamespace(N=4, sublattice=np.array([1, -1, -1, 1]))
    a, b = _reduce_sublattice(np.array([0.5, 0.9, 0.9, 0.5]), spec, "delta_R")
    assert a == pytest.approx(0.5)
    assert b == pytest.approx(0.9)
    with pytest.raises(PEPSKitDriveError):
        _reduce_sublattice(np.array([0.5, 0.9, 0.8, 0.5]), spec, "delta_R")  # B not constant


def test_initial_state_uniform_requires_homogeneous():
    spec = create_tn_lattice_spec(2, 2, level_structure="1r", V_nn=1.0, interaction_mode="nn")
    assert _initial_state_payload(spec, "all_ground", "uniform") == {"pattern": "uniform", "label": "1"}
    with pytest.raises(PEPSKitDriveError):
        _initial_state_payload(spec, "af1", "uniform")
    af1 = _initial_state_payload(spec, "af1", "sublattice")
    assert af1["pattern"] == "sublattice"
    assert {af1["A"], af1["B"]} == {"r", "1"}


# ---------------------------------------------------------------------------
# Physics validation against analytics (opt-in: real Julia kernel)
# ---------------------------------------------------------------------------
def _run(ir, *, t_eval, observables, **opts):
    backend = PEPSKitIPEPSBackend(timeout=1800, **opts)
    return backend.evolve_ir(ir, initial_state="all_ground", t_eval=t_eval, observables=observables)


@julia_only
def test_resonant_rabi_1r_matches_sin2():
    Omega = 2 * np.pi  # = 2*hx
    ir = _make_ir("1r", 0.0, TFIMQuenchProtocol(hx=np.pi, hz=0.0, t_gate=0.4))
    t_eval = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    res = _run(ir, t_eval=t_eval, observables=["n_r", "n_1"], unit_cell="uniform", bond_dim=2, env_dim=12, dt=0.05)
    nr = np.asarray(res.metadata["obs"]["n_r"])
    n1 = np.asarray(res.metadata["obs"]["n_1"])
    times = np.asarray(res.times)
    assert np.max(np.abs(nr - np.sin(Omega * times / 2) ** 2)) < 1e-2
    assert np.allclose(nr + n1, 1.0, atol=1e-6)  # 1r: n_1 = 1 - n_r


@julia_only
def test_detuned_rabi_1r_matches_generalized():
    Omega, Delta = 2 * np.pi, np.pi  # Omega=2*hx, Delta=-2*hz
    ir = _make_ir("1r", 0.0, TFIMQuenchProtocol(hx=np.pi, hz=np.pi / 2, t_gate=0.4))
    t_eval = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    res = _run(ir, t_eval=t_eval, observables=["n_r"], unit_cell="uniform", bond_dim=2, env_dim=12, dt=0.04)
    nr = np.asarray(res.metadata["obs"]["n_r"])
    times = np.asarray(res.times)
    gen = (Omega**2 / (Omega**2 + Delta**2)) * np.sin(np.sqrt(Omega**2 + Delta**2) * times / 2) ** 2
    assert np.max(np.abs(nr - gen)) < 2e-2


@julia_only
def test_hyperfine_rabi_01r_matches_sin2():
    # Only the |0>-|1> hyperfine drive on; population flows |1> -> |0> as cos^2/sin^2.
    Omega_hf = 2 * np.pi
    ir = _make_ir("01r", 0.0, DigitalAnalogProtocol.constant(omega_hf=Omega_hf, t_gate=0.4, n_steps=8))
    t_eval = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    res = _run(ir, t_eval=t_eval, observables=["n_0", "n_1"], unit_cell="uniform", bond_dim=2, env_dim=12, dt=0.05)
    n0 = np.asarray(res.metadata["obs"]["n_0"])
    times = np.asarray(res.times)
    assert np.max(np.abs(n0 - np.sin(Omega_hf * times / 2) ** 2)) < 1e-2


@julia_only
def test_rydberg_rabi_01r_matches_sin2():
    # Only the |1>-|r> Rydberg drive on; |1> -> |r| Rabi.
    Omega_R = 2 * np.pi
    ir = _make_ir("01r", 0.0, DigitalAnalogProtocol.constant(omega_R=Omega_R, t_gate=0.4, n_steps=8))
    t_eval = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    res = _run(ir, t_eval=t_eval, observables=["n_r"], unit_cell="uniform", bond_dim=2, env_dim=12, dt=0.05)
    nr = np.asarray(res.metadata["obs"]["n_r"])
    times = np.asarray(res.times)
    assert np.max(np.abs(nr - np.sin(Omega_R * times / 2) ** 2)) < 1e-2
