"""Focused tests for scripts/max_leakage_ode_sweep.py.

Script-specific coverage: locked nested-axis values, the analytic chirp/phase,
the scattering-budget channel tables, plotting/refinement/export and the CLI —
plus the real rb87_7_mp model checks (ARC) with the expensive solver-equivalence
runs marked ``slow``.  The shared Store/Runner/CostModel machinery is exercised
(parameterized over key configs) in tests/test_sweeplib.py.
"""

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "max_leakage_ode_sweep", ROOT / "scripts" / "max_leakage_ode_sweep.py")
mls = importlib.util.module_from_spec(_spec)
sys.modules["max_leakage_ode_sweep"] = mls
_spec.loader.exec_module(mls)


# ── nested axes and canonical keys (locked ode axis values; generic axis/key
# machinery is covered in tests/test_sweeplib.py) ─────────────────────────────


def test_axis_values_match_locked_specification():
    om0 = [float(v) for v in mls.axis_values_mhz(mls.OMEGA_ANCHORS_MHZ, 0)]
    assert om0 == pytest.approx([200.0, 1400 / 3, 2200 / 3, 1000.0])
    om1 = [float(v) for v in mls.axis_values_mhz(mls.OMEGA_ANCHORS_MHZ, 1)]
    assert om1 == pytest.approx(
        [200.0, 1000 / 3, 1400 / 3, 600.0, 2200 / 3, 2600 / 3, 1000.0])
    dw1 = [float(v) for v in mls.axis_values_mhz(mls.DSWEEP_ANCHORS_MHZ, 1)]
    assert dw1 == pytest.approx([2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0])


def test_20mhz_is_a_node_of_the_dsweep_axis_at_every_level():
    for level in range(4):
        values = [float(v) for v in mls.axis_values_mhz(mls.DSWEEP_ANCHORS_MHZ, level)]
        assert 20.0 in values


def test_point_key_axis_values_are_locked_mhz():
    k = mls.make_key(3, 4, (3, 2), (3, 2))
    assert float(k.omega_mhz()) == 600.0
    assert float(k.dsweep_mhz()) == 15.0


def test_pilot_keys_dedup_and_reusability():
    pkeys = mls.pilot_keys()
    assert len(pkeys) == len(set(pkeys)) == 72 + 16
    level1 = set(mls.all_keys(1))
    assert all(k in level1 for k in pkeys[:72])       # centers are 7x7 nodes
    level0 = set(mls.all_keys(0))
    assert all(k in level0 for k in pkeys[72:])       # extremes are 4x4 nodes


# ── analytic pulse (script-specific chirp/phase/Stark; the shared quintic
# envelope is covered in tests/test_sweeplib.py) ──────────────────────────────


def test_phase_is_exact_integral_of_chirp():
    t_gate, d_sweep, drmd1, r = 1.3e-6, 2 * np.pi * 17e6, -2 * np.pi * 3.1e6, 0.15
    assert mls.phase_rad(0.0, t_gate, d_sweep, drmd1, r) == 0.0
    # dphi/dt == chirp: central differences at interior points, one-sided at the
    # ramp breakpoints s = r, 1 - r.
    for s in (0.03, 0.08, 0.3, 0.5, 0.7, 0.9, 0.97):
        t = s * t_gate
        dt = 1e-6 * t_gate
        dphi = (mls.phase_rad(t + dt, t_gate, d_sweep, drmd1, r)
                - mls.phase_rad(t - dt, t_gate, d_sweep, drmd1, r)) / (2 * dt)
        chirp = mls.chirp_rad_s(t, t_gate, d_sweep, drmd1, r)
        assert dphi == pytest.approx(float(chirp), rel=1e-6, abs=1e-3 * abs(d_sweep))
    for s0, side in ((r, +1), (r, -1), (1 - r, +1), (1 - r, -1)):
        t = s0 * t_gate
        dt = side * 1e-7 * t_gate
        dphi = (mls.phase_rad(t + dt, t_gate, d_sweep, drmd1, r)
                - mls.phase_rad(t, t_gate, d_sweep, drmd1, r)) / dt
        chirp = mls.chirp_rad_s(t + 0.5 * dt, t_gate, d_sweep, drmd1, r)
        assert dphi == pytest.approx(float(chirp), rel=1e-4, abs=1e-3 * abs(d_sweep))

    # Pure-cosine limit (Dr - D1 = 0): phase is exactly -(D T / 2 pi) sin(2 pi s),
    # and it is NOT wrapped mod 2 pi.
    tg2, d2 = 2.0e-6, 2 * np.pi * 20e6
    tp = 0.37 * tg2
    expect = -(d2 * tg2 / (2 * np.pi)) * np.sin(2 * np.pi * 0.37)
    assert float(mls.phase_rad(tp, tg2, d2, 0.0)) == pytest.approx(expect, rel=1e-12)
    assert abs(float(mls.phase_rad(tp, tg2, d2, 0.0))) > 2 * np.pi   # unwrapped

    # The compensating Stark shifts feeding the chirp are both negative.
    d1s, drs = mls.stark_coefficients(2 * np.pi * 600e6, 2 * np.pi * 489.6e6,
                                      2 * np.pi * 20e9)
    assert d1s < 0 and drs < 0


# ── toy swap-symmetric model shared by the scattering tests below (the block-max
# DOP853 norm and the kernel invariants are covered in tests/test_sweeplib.py) ─


def _toy_ops(v_pair=2 * np.pi * 5e6, w_hf=2 * np.pi * 40e6, w_e=2 * np.pi * 300e6):
    """Two-atom, three-level toy analogue of PanelOperators: levels {0,1,e},
    logical = {0,1}, 420 leg drives 1->e (+ a weak 0->e), 1013 leg drives 0->e.
    Swap-symmetric by construction (identical site embeddings)."""
    d = 3
    e_loc = np.array([-w_hf, 0.0, w_e])
    eye = np.eye(d)
    diag = np.add.outer(e_loc, e_loc).ravel()
    pair = np.zeros(d * d)
    pair[2 * d + 2] = v_pair  # both atoms in |e>
    diag = diag + pair

    b420_loc = np.zeros((d, d), complex)
    b420_loc[2, 1] = 0.5
    b420_loc[2, 0] = 0.1
    b1013_loc = np.zeros((d, d), complex)
    b1013_loc[2, 0] = 0.3
    b420 = np.kron(b420_loc, eye) + np.kron(eye, b420_loc)
    b1013 = np.kron(b1013_loc, eye) + np.kron(eye, b1013_loc)

    return mls.PanelOperators(
        delta_e_hz=1e9, delta_rad_s=2 * np.pi * 1e9,
        h_static_diag=diag,
        x420=b420 + b420.conj().T, y420=1j * (b420 - b420.conj().T),
        x1013=b1013 + b1013.conj().T, y1013=1j * (b1013 - b1013.conj().T),
        amplitude_scale=1.0,
        logical_indices=np.array([0, 1, 3, 4]),
        swap_perm=mls._swap_permutation(d),
        swap_symmetric=True,
    )


_TOY_T = 0.4e-6
_TOY_OM = np.array([2 * np.pi * 30e6])
_TOY_DW = np.array([2 * np.pi * 10e6])
_TOY_OM1013 = 2 * np.pi * 25e6


def _toy_solve(**kw):
    kw.setdefault("rtol", 1e-10)
    kw.setdefault("atol", 1e-13)
    return mls.integrate_batch(
        _toy_ops(), _TOY_T, kw.pop("om", _TOY_OM), kw.pop("dw", _TOY_DW),
        _TOY_OM1013, **kw)


# ── mini store fixtures (real ode config) for the plotting/export/scatter tests ─


def _mini_cfg():
    return mls.ScanConfig()


def _mini_store(tmp_path, model_hash="modelhash-1"):
    store = mls.Store(str(tmp_path / "scan"))
    store.ensure_dirs()
    cfg = _mini_cfg()
    manifest = store.init_or_validate_manifest(
        cfg, model_hash, "codehash", {},
        **mls._manifest_extras(cfg, 2 * np.pi * 489.6e6))
    return store, manifest


def _fake_result(n, dim=49, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal((n, 4, dim)) + 1j * rng.standard_normal((n, 4, dim))
    psi /= np.linalg.norm(psi, axis=2, keepdims=True)
    leak = rng.uniform(1e-6, 1e-2, size=(n, 4))
    return mls.BatchResult(
        psi_final=psi, leakage=leak, max_leakage=leak.max(axis=1),
        worst_input=[mls.LOGICAL_INPUTS[int(i)] for i in leak.argmax(axis=1)],
        return_prob=rng.uniform(0.9, 1.0, size=(n, 4)),
        norm_err=np.full((n, 4), 1e-13), nfev=1234, used_swap=True,
    )


def test_export_store_writes_merged_npz_and_csv(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), "production",
                             1e-9, 1e-12, "b", _fake_result(len(keys)), 60.0)
    merged, csv_path = mls.export_store(store)
    with np.load(merged, allow_pickle=False) as d:
        assert d["max_leakage"].shape == (16,)
        assert d["psi_final"].shape == (16, 4, 49)
    lines = Path(csv_path).read_text().strip().splitlines()
    assert len(lines) == 17 and lines[0].startswith("point_id,")


def test_effective_batch_size_falls_back_to_one_without_a_passed_gate(tmp_path):
    """_effective_batch_size gates the requested size on a recorded, enabled
    packing acceptance; absent/failed gates fall back to one point per solve."""
    store, _ = _mini_store(tmp_path)
    assert mls._effective_batch_size(store, Namespace(batch_size=1)) == 1
    assert mls._effective_batch_size(store, Namespace(batch_size=48)) == 1   # no pilot
    pilot = Path(store.reports_dir) / "pilot.json"
    pilot.write_text(json.dumps({"packing_gate": {"enabled": True}}))
    assert mls._effective_batch_size(store, Namespace(batch_size=48)) == 48
    pilot.write_text(json.dumps({"packing_gate": {"enabled": False}}))
    assert mls._effective_batch_size(store, Namespace(batch_size=48)) == 1


# ── refinement priorities ────────────────────────────────────────────────────


def test_refinement_prefers_decision_contour_and_is_deterministic():
    coords = mls.axis_coords(0)
    leak, worst = {}, {}
    for panel, base in (((0, 0), 1e-6), ((0, 1), 1e-6)):
        for i, om in enumerate(coords):
            for j, dw in enumerate(coords):
                k = mls.make_key(panel[0], panel[1], om, dw)
                # panel (0,0): flat 1e-6 (no refinement); panel (0,1): one cell
                # crosses 1e-3 between the last two omega rows
                val = base
                if panel == (0, 1) and i == 3 and j >= 2:
                    val = 1e-2
                leak[k] = val
                worst[k] = "11"
    cand = mls.refinement_candidates(leak, worst, level=0)
    assert cand, "the contour-crossing panel must be refined"
    assert all(k.panel == (0, 1) for k, _ in cand)
    assert all(k in set(mls.all_keys(1)) for k, _ in cand)   # exact level-1 nodes
    assert all(k not in leak for k, _ in cand)               # only missing nodes
    scores = [s for _, s in cand]
    assert scores == sorted(scores, reverse=True)
    assert cand == mls.refinement_candidates(leak, worst, level=0)  # deterministic
    # incomplete panels are skipped entirely
    del leak[mls.make_key(0, 1, coords[0], coords[0])]
    assert mls.refinement_candidates(leak, worst, level=0) == []


def test_credibility_floor_rule_and_fallback():
    keys = mls.panel_keys(0, 0, 0)

    def rec(k, tier, leak):
        return mls.PointRecord(
            key=k, tier=tier, rtol=1e-9, atol=1e-12, status="ok",
            max_leakage=leak, leakage=np.full(4, leak), worst_input="11",
            return_prob=np.ones(4), norm_err=np.zeros(4), psi_final=mls._NO_STATES,
            nfev=1, runtime_s=1.0, batch_id="b", batch_size=1, retry_count=0,
            priority_score=0.0, message="", chunk_file="c", used_swap=True)

    few = [rec(keys[0], "production", 1e-4), rec(keys[0], "audit", 1e-4 + 1e-13)]
    vmin, info = mls._credibility_floor(few)
    assert info["fallback"] and vmin == pytest.approx(1e-11)

    many = []
    for i, k in enumerate(keys[:10]):
        many.append(rec(k, "production", 1e-4))
        many.append(rec(k, "audit", 1e-4 + (i + 1) * 1e-13))
    vmin, info = mls._credibility_floor(many)
    assert not info["fallback"]
    assert vmin == pytest.approx(max(1e-12, 10 * np.percentile(
        np.arange(1, 11) * 1e-13, 95)))


# ── plotting smoke test (synthetic store) ────────────────────────────────────


def test_plot_and_status_smoke(tmp_path, capsys):
    store, manifest = _mini_store(tmp_path)
    rng = np.random.default_rng(3)
    for seq, panel in enumerate([(0, 0), (3, 4)], start=1):
        keys = mls.panel_keys(panel[0], panel[1], 1)   # full 7x7 grid
        res = _fake_result(len(keys), seed=seq)
        res.leakage = rng.uniform(1e-5, 1e-1, size=(len(keys), 4))
        res.max_leakage = res.leakage.max(axis=1)
        store.write_result_chunk(seq, manifest, keys, _mini_cfg(),
                                 "production", 1e-9, 1e-12, f"b{seq}", res, 60.0)
    args = Namespace(output=store.root, dpi=60, veil=True,
                     metric="max_leakage")
    mls.cmd_plot(args)
    assert (Path(store.plots_dir) / "max_leakage_8x9.png").exists()
    assert (Path(store.plots_dir) / "max_leakage_8x9.pdf").exists()
    assert not list(Path(store.plots_dir).glob("panel_*.png"))

    mls.cmd_status(Namespace(output=store.root))
    out = capsys.readouterr().out
    assert "records:" in out and "98 unique ok points" in out


def test_cli_parser_covers_subcommands_and_locked_invocation():
    parser = mls.build_parser()
    args = parser.parse_args(["run", "--dry-run", "--target-level", "13"])
    assert args.func is mls.cmd_run and args.dry_run and args.target_level == "13"
    assert parser.parse_args(
        ["audit", "--audit-point", "d0_t0_om0-1_dw0-1"]).func is mls.cmd_audit
    assert parser.parse_args(["status"]).func is mls.cmd_status
    args = parser.parse_args(["scatter", "--level", "7", "--workers", "auto",
                              "--batch-size", "auto"])
    assert args.func is mls.cmd_scatter and args.level == "7"
    assert parser.parse_args(["plot", "--metric", "p_loss_total"]).metric == "p_loss_total"
    assert parser.parse_args(["plot", "--metric", "total_error"]).metric == "total_error"

    # The handoff's locked production command must parse verbatim.
    args = parser.parse_args([
        "run", "--output", "results/max_leakage_ode",
        "--workers", "auto", "--batch-size", "auto", "--target-level", "auto",
        "--budget-hours", "24", "--reserve-hours", "2"])
    assert isinstance(args.workers, int) and 1 <= args.workers <= 40
    assert args.batch_size == 48
    assert args.target_level == "auto"


def test_spacing_flag_and_derived_output_default():
    parser = mls.build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.spacing_um == 3.0 and args.output is None
    assert mls._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_ode", "a3.0")
    args = parser.parse_args(["scatter", "--level", "13", "--spacing-um", "7"])
    assert args.spacing_um == 7.0
    assert mls._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_ode", "a7.0")
    args = parser.parse_args(
        ["status", "--output", "results/max_leakage_ode/legacy_c6-874"])
    assert args.output == "results/max_leakage_ode/legacy_c6-874"


def test_spacing_um_changes_physics_hash():
    assert (mls.ScanConfig(spacing_um=4.0).physics_hash()
            != mls.ScanConfig().physics_hash())


def test_pulse_hash_is_stable_and_recorded(tmp_path):
    """The pulse fingerprint is deterministic (64 hex chars) and is what the
    manifest records, so a later pulse edit is caught by the provenance guards
    (see tests/test_sweeplib.py for the refuse-to-merge arms)."""
    h1, h2 = mls.pulse_hash(), mls.pulse_hash()
    assert h1 == h2 and len(h1) == 64
    _, manifest = _mini_store(tmp_path)
    assert manifest["pulse_hash"] == h1


def test_holdout_residuals_cover_every_interior_node():
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    x, y = xs.ravel(), ys.ravel()
    z = np.zeros_like(x)
    assert np.all(mls._holdout_residuals(x, y, z) == 0.0)
    z2 = x + 2 * y            # linear field: every holdout estimate is exact
    assert np.max(mls._holdout_residuals(x, y, z2)) < 1e-12
    z3 = z2.copy()
    spike = np.flatnonzero((x == 2) & (y == 2))[0]
    z3[spike] += 1.0          # one bad node: it (and only its lines) flags
    resid = mls._holdout_residuals(x, y, z3)
    assert resid[spike] == pytest.approx(1.0)
    corner = np.flatnonzero((x == 0) & (y == 0))[0]
    assert resid[corner] == 0.0


# ── scattering supplement (channel tables + integrals; the shared Store scatter
# series round-trip is covered in tests/test_sweeplib.py) ─────────────────────


def test_scatter_weight_vectors_count_atoms_per_group():
    w = mls._scatter_weight_vectors(7)
    # |e2,e2> (index 3*7+3) has two mid-state atoms; |r,0> (5*7+0) one Rydberg
    assert w["p_mid"][3 * 7 + 3] == 2.0
    assert w["p_ryd"][5 * 7 + 0] == 1.0
    assert w["p_r_garb"][6 * 7 + 6] == 2.0
    assert w["p_mid"][0] == w["p_ryd"][0] == w["p_r_garb"][0] == 0.0
    for v in w.values():
        assert v.shape == (49,)


def test_scattering_integrals_constant_population():
    """A state parked in |e2,e2> for the whole window gives p = 2*Gamma*T."""
    T, n_t = 2.0e-6, 51
    times = np.linspace(0.0, T, n_t)
    states = np.zeros((n_t, 1, 4, 49), dtype=complex)
    states[:, 0, :, 3 * 7 + 3] = 1.0
    gammas = {"p_mid": 9.0e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}
    out = mls.scattering_integrals(times, states, gammas)
    assert out["p_mid"].shape == (1, 4)
    assert np.allclose(out["p_mid"], 2.0 * 9.0e6 * T, rtol=1e-12)
    assert np.allclose(out["p_ryd"], 0.0)


def test_scatter_integrals_on_a_solved_toy_trajectory():
    """Integrals from a real solved trajectory: finite, nonnegative, and the
    driven |e>-population channel (index 2 -> 'p_mid' group's only in-range
    level) integrates to a physically sensible nonzero value."""
    t_eval = np.linspace(0.0, _TOY_T, 61)
    res = _toy_solve(t_eval=t_eval)
    gammas = {"p_mid": 1.0e7, "p_ryd": 1.0e4, "p_r_garb": 1.0e4}
    out = mls.scattering_integrals(res.times, res.states, gammas)
    for ch in mls.SCATTER_CHANNELS:
        assert out[ch].shape == (1, 4)
        assert np.all(out[ch] >= 0.0)
        assert np.all(np.isfinite(out[ch]))
    assert out["p_mid"].max() > 0.0   # toy drive populates level 2
    assert np.all(out["p_ryd"] == 0.0)  # group (5,) is outside the toy basis


def test_model_decay_rates_maps_channels():
    """model_decay_rates maps the e1/r/r_garb model channels to the p_* groups."""
    stub = Namespace(level_structure=Namespace(
        decay_rates_per_s={"e1": {"total": 9.03e6}, "r": {"total": 6.6e3},
                           "r_garb": {"total": 6.6e3}}))
    gammas = mls.model_decay_rates(stub)
    assert gammas == {"p_mid": 9.03e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}


def test_plot_metric_values_prefers_tight_rtol_and_totals(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    gammas = {0: {"p_mid": 9.03e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}}
    ones = {ch: np.full((1, 4), v) for ch, v in
            zip(mls.SCATTER_CHANNELS, (1e-2, 1e-3, 1e-4))}
    store.write_scatter_chunk(1, manifest, keys, _mini_cfg(), gammas, 1e-6,
                              1e-9, "loose", ones, np.array([1e-4]), 1.0)
    tight = {ch: np.full((1, 4), v) for ch, v in
             zip(mls.SCATTER_CHANNELS, (2e-2, 2e-3, 2e-4))}
    store.write_scatter_chunk(2, manifest, keys, _mini_cfg(), gammas, 1e-9,
                              1e-12, "tight", tight, np.array([1e-4]), 1.0)
    values, vmin, vmax, label = mls._plot_metric_values(store, manifest, [], "p_mid")
    assert values[keys[0]] == pytest.approx(2e-2)     # tighter rtol wins
    values, *_ = mls._plot_metric_values(store, manifest, [], "p_loss_total")
    assert values[keys[0]] == pytest.approx(2e-2 + 2e-3 + 2e-4)


def test_total_error_sums_each_input_before_selecting_the_worst(tmp_path):
    store, manifest = _mini_store(tmp_path)
    key = mls.panel_keys(0, 0, 0)[0]
    record = mls.PointRecord(
        key=key, tier="production", rtol=1e-9, atol=1e-12, status="ok",
        max_leakage=0.4, leakage=np.array([0.4, 0.1, 0.2, 0.3]),
        worst_input="00", return_prob=np.ones(4), norm_err=np.zeros(4),
        psi_final=mls._NO_STATES, nfev=1, runtime_s=1.0, batch_id="main",
        batch_size=1, retry_count=0, priority_score=0.0, message="",
        chunk_file="chunk", used_swap=True,
    )
    scatter = {
        "p_mid": np.array([[0.0, 0.4, 0.1, 0.0]]),
        "p_ryd": np.array([[0.0, 0.0, 0.2, 0.0]]),
        "p_r_garb": np.array([[0.0, 0.0, 0.0, 0.2]]),
    }
    gammas = {0: {"p_mid": 9.03e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}}
    store.write_scatter_chunk(1, manifest, [key], _mini_cfg(), gammas, 1e-9,
                              1e-12, "scatter", scatter, np.array([0.4]), 1.0)

    values, *_ = mls._plot_metric_values(store, manifest, [record], "total_error")

    assert values[key] == pytest.approx(0.5)


# ── real rb87_7_mp model (ARC required) ──────────────────────────────────────


@pytest.fixture(scope="module")
def real_ops():
    cfg = mls.ScanConfig()
    delta_e_hz = 20e9
    system = mls.build_system(cfg, delta_e_hz)
    ops = mls.aggregate_operators(system, delta_e_hz)
    return cfg, system, ops


def test_omega_1013_convention_matches_recorded_reference():
    omega_1013 = mls.compute_omega_1013(mls.ScanConfig())
    assert omega_1013 == pytest.approx(mls.OMEGA_1013_REFERENCE_RAD_S, rel=1e-6)


def test_aggregated_hamiltonian_structure(real_ops):
    cfg, system, ops = real_ops
    assert ops.h_static_diag.shape == (49,)
    assert np.isrealobj(ops.h_static_diag)
    for m in (ops.x420, ops.y420, ops.x1013, ops.y1013):
        assert np.max(np.abs(m - m.conj().T)) == 0.0   # Hermitian blocks
    assert ops.swap_symmetric
    assert list(ops.logical_indices) == [0, 1, 7, 8]
    # Delta enters the e-manifold diagonal with the positive convention
    assert ops.delta_rad_s == pytest.approx(2 * np.pi * 20e9)
    # both atoms in |e2> (index 3, the F=2 level with no hyperfine offset)
    e2e2 = ops.h_static_diag[3 * 7 + 3]
    assert e2e2 == pytest.approx(2 * ops.delta_rad_s, rel=1e-12)
    # both atoms in |e1> (index 2) carries the -2pi*51 MHz offset twice
    e1e1 = ops.h_static_diag[2 * 7 + 2]
    assert e1e1 == pytest.approx(2 * (ops.delta_rad_s - 2 * np.pi * 51e6), rel=1e-12)


def test_grouped_hamiltonian_matches_repository_compiler(real_ops):
    cfg, system, ops = real_ops
    t_gate = 1.0e-6
    rng = np.random.default_rng(7)
    times = np.concatenate([
        [0.0, 0.15 * t_gate, 0.85 * t_gate, t_gate],
        rng.uniform(0, t_gate, 5),
    ])
    dev = mls.hamiltonian_equivalence_error(
        system, ops, t_gate,
        omega_420=2 * np.pi * 600e6, omega_1013=mls.OMEGA_1013_REFERENCE_RAD_S,
        d_sweep=2 * np.pi * 15e6, times=times)
    scale = float(np.max(np.abs(ops.h_static_diag)))
    assert dev / scale < 1e-12


@pytest.mark.slow
def test_kernel_matches_repository_exact_ode_backend(real_ops):
    """Optimized kernel (batch of one, three columns + swap) vs the repository
    original-frame exact_ode backend at strict tolerance, on a short pulse."""
    import ryd_gate as rg
    from ryd_gate.protocols import CZProtocol

    cfg, system, ops = real_ops
    t_gate = 0.02e-6
    omega_420 = 2 * np.pi * 600e6
    omega_1013 = mls.compute_omega_1013(cfg)
    d_sweep = 2 * np.pi * 15e6
    d1, dr = mls.stark_coefficients(omega_420, omega_1013, ops.delta_rad_s)
    drmd1 = dr - d1

    res = mls.integrate_batch(
        ops, t_gate, np.array([omega_420]), np.array([d_sweep]), omega_1013,
        rtol=1e-10, atol=1e-13)

    proto = CZProtocol(
        t_gate_s=t_gate,
        intermediate_detuning_rad_s=ops.delta_rad_s,
        omega_420_max_rad_s=omega_420, omega_1013_max_rad_s=omega_1013,
        envelope_420=lambda t: float(np.sqrt(mls.envelope(t / t_gate))),
        phase_420_rad=lambda t: float(mls.phase_rad(t, t_gate, d_sweep, drmd1)),
        envelope_1013=lambda t: float(np.sqrt(mls.envelope(t / t_gate))),
        phase_1013_rad=lambda t: 0.0,
    )
    bound = system.with_protocol(proto)
    ref = rg.simulate(bound, [list(s) for s in mls.LOGICAL_INPUTS],
                      backend="exact_ode",
                      backend_options={"rtol": 1e-10, "atol": 1e-13})
    # EvolutionResult's dense state is private (S-schema): reconstruct each
    # reference vector from its public per-basis amplitudes (site 0 most
    # significant), matching the kernel's product_index ordering.
    levels = system._basis.local_levels
    for j in range(4):
        ref_vec = np.array(
            [ref[j].amplitude([a, b]) for a in levels for b in levels])
        dev = np.max(np.abs(res.psi_final[0, j] - ref_vec))
        assert dev < 1e-7, f"input {mls.LOGICAL_INPUTS[j]}: max dev {dev:.2e}"


@pytest.mark.slow
def test_real_model_swap_and_batching_invariants(real_ops):
    cfg, system, ops = real_ops
    t_gate = 0.02e-6
    omega_1013 = mls.compute_omega_1013(cfg)
    om = 2 * np.pi * np.array([400e6, 600e6])
    dw = 2 * np.pi * np.array([10e6, 20e6])
    batched = mls.integrate_batch(ops, t_gate, om, dw, omega_1013,
                                  rtol=1e-9, atol=1e-12)
    for i in range(2):
        alone = mls.integrate_batch(ops, t_gate, om[i:i + 1], dw[i:i + 1],
                                    omega_1013, rtol=1e-9, atol=1e-12)
        assert np.max(np.abs(batched.psi_final[i] - alone.psi_final[0])) < 1e-6
        assert np.max(np.abs(batched.leakage[i] - alone.leakage[0])) < 1e-8
    all4 = mls.integrate_batch(ops, t_gate, om[:1], dw[:1], omega_1013,
                               rtol=1e-9, atol=1e-12, use_swap=False)
    assert np.max(np.abs(batched.psi_final[0] - all4.psi_final[0])) < 1e-6
