"""Tests for the 297 nm single-photon leakage sweep script (fork of the
two-photon max_leakage_ode_sweep tests; same importlib loading pattern)."""

import importlib.util
import json
import math
import os
import sys
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "max_leakage_297_sweep", ROOT / "scripts" / "max_leakage_297_sweep.py")
mls297 = importlib.util.module_from_spec(_spec)
sys.modules["max_leakage_297_sweep"] = mls297
_spec.loader.exec_module(mls297)


def test_axes_are_the_locked_297_specification():
    assert mls297.RYD_N == (50, 53, 56, 60, 64, 68, 71, 73)
    assert mls297.T_GATE_US == (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
    assert mls297.OMEGA297_ANCHORS_MHZ == (
        Fraction(9), Fraction(12), Fraction(15), Fraction(18))
    assert mls297.DSWEEP_ANCHORS_MHZ == (
        Fraction(2), Fraction(10), Fraction(20), Fraction(30))
    assert mls297.DSWEEP_HW_LIMIT_MHZ == 20.0
    assert len(mls297.all_panels()) == 72
    assert len(mls297.all_keys(1)) == 72 * 49
    assert len(mls297.all_keys(3)) == 72 * 625


def test_20mhz_is_a_dsweep_node_and_grids_nest():
    for level in range(4):
        vals = mls297.axis_values_mhz(mls297.DSWEEP_ANCHORS_MHZ, level)
        assert Fraction(20) in vals
    coarse = set(mls297.axis_coords(1))
    assert coarse <= set(mls297.axis_coords(2)) <= set(mls297.axis_coords(3))


def test_point_key_id_uses_n_prefix_and_canonicalizes():
    k = mls297.make_key(2, 4, (2, 4), (6, 8))
    assert k == mls297.make_key(2, 4, (1, 2), (3, 4))
    assert k.id() == "n2_t4_om1-2_dw3-4"
    assert k.panel == (2, 4)
    assert float(k.omega_mhz()) == 10.5          # 9 + (12-9)*1/2  (om coord 1/2)
    assert float(k.dsweep_mhz()) == 8.0          # 2 + (10-2)*3/4  (dw coord 3/4)


def test_phase_is_exact_integral_of_chirp():
    t_gate, d_sweep = 1.7e-6, mls297.TAU * 13e6
    ts = np.linspace(0.0, t_gate, 4001)
    chirp = mls297.chirp_rad_s(ts, t_gate, d_sweep)
    phi_num = np.concatenate(
        [[0.0], np.cumsum((chirp[1:] + chirp[:-1]) * 0.5 * np.diff(ts))])
    phi = mls297.phase_rad(ts, t_gate, d_sweep)
    assert np.max(np.abs(phi - phi_num)) < 1e-6 * np.max(np.abs(phi))
    assert not hasattr(mls297, "stark_coefficients")


def test_physics_hash_covers_axes_and_spacing():
    base = mls297.ScanConfig().physics_hash()
    assert mls297.ScanConfig(spacing_um=4.0).physics_hash() != base
    assert mls297.ScanConfig(ryd_n=(50, 53)).physics_hash() != base


def test_swap_permutation_local_dim_4():
    perm = mls297._swap_permutation(4)
    a, b = np.divmod(np.arange(16), 4)
    assert np.array_equal(perm, b * 4 + a)


@pytest.mark.slow
def test_warm_and_build_single_panel_matches_repo_hamiltonian():
    cfg = mls297.ScanConfig(ryd_n=(53,))
    ops_by_n, model_hash, checks = mls297.warm_and_build(cfg)
    ops = ops_by_n[0]
    assert ops.h_static_diag.shape == (16,)
    assert checks["hamiltonian_equivalence_rel_dev"] < 1e-12
    assert checks["swap_symmetric"]
    assert set(checks["decay_rates_rad_s"][0]) == {"p_ryd", "p_r_garb"}
    assert len(model_hash) == 64


# ── block-max DOP853 error norm ──────────────────────────────────────────────


def test_error_norm_matches_installed_scipy():
    assert mls297.verify_scipy_error_norm() < mls297.ERR_NORM_REL_TOL


def test_block_solver_matches_stock_dop853_per_block():
    """Two decoupled 16-component blocks integrated together under the block-max
    norm agree with each block integrated alone by stock scipy DOP853."""
    import scipy.integrate

    w = [3.0, 41.0]

    def rhs_block(wi):
        return lambda t, y: -1j * wi * y

    def rhs_joint(t, y):
        out = np.empty_like(y)
        out[:16] = -1j * w[0] * y[:16]
        out[16:] = -1j * w[1] * y[16:]
        return out

    y0 = (np.arange(1, 33) + 1j) / 10.0
    cls = mls297.make_block_solver_class(16)
    solver = cls(rhs_joint, 0.0, y0.astype(complex), 2.0, rtol=1e-10, atol=1e-12)
    while solver.status == "running":
        solver.step()
    assert solver.status == "finished"
    for b in range(2):
        ref = scipy.integrate.solve_ivp(
            rhs_block(w[b]), (0.0, 2.0), y0[16 * b:16 * b + 16].astype(complex),
            method="DOP853", rtol=1e-10, atol=1e-12)
        exact = y0[16 * b:16 * b + 16] * np.exp(-1j * w[b] * 2.0)
        assert np.max(np.abs(solver.y[16 * b:16 * b + 16] - exact)) < 1e-8
        assert np.max(np.abs(ref.y[:, -1] - exact)) < 1e-8


# ── single-drive kernel on synthetic two-atom four-level models (no ARC) ──────


def _toy_ops(dim=16):
    """Random Hermitian x297/y297 kernel operators (not swap-symmetric); used to
    check the batched single-drive kernel against a dense reference."""
    rng = np.random.default_rng(7)
    b = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    x = b + b.conj().T
    y = 1j * (b - b.conj().T)
    diag = rng.normal(size=dim)
    perm = mls297._swap_permutation(4)
    return mls297.PanelOperators(
        ryd_n=53, h_static_diag=diag, x297=x, y297=y, amplitude_scale=1.0,
        logical_indices=np.array([0, 1, 4, 5]), swap_perm=perm,
        swap_symmetric=False)


def _toy_sym_ops(w_hf=2 * np.pi * 40e6, w_r=2 * np.pi * 300e6,
                 w_g=2 * np.pi * 305e6, v_pair=2 * np.pi * 5e6):
    """Two-atom, four-level swap-symmetric analogue of PanelOperators: levels
    {0, 1, r, r_garb}, logical = {0, 1}; the 297 leg drives 1->r (+ a weak 0->r)
    and a garbage branch 1->r_garb.  Swap-symmetric by construction (identical
    site embeddings)."""
    d = 4
    e_loc = np.array([-w_hf, 0.0, w_r, w_g])
    eye = np.eye(d)
    diag = np.add.outer(e_loc, e_loc).ravel()
    pair = np.zeros(d * d)
    pair[2 * d + 2] = v_pair  # both atoms in |r>
    diag = diag + pair

    b297_loc = np.zeros((d, d), complex)
    b297_loc[2, 1] = 0.5
    b297_loc[2, 0] = 0.1
    b297_loc[3, 1] = 0.3
    b297 = np.kron(b297_loc, eye) + np.kron(eye, b297_loc)

    return mls297.PanelOperators(
        ryd_n=53, h_static_diag=diag,
        x297=b297 + b297.conj().T, y297=1j * (b297 - b297.conj().T),
        amplitude_scale=1.0,
        logical_indices=np.array([0, 1, 4, 5]),
        swap_perm=mls297._swap_permutation(d),
        swap_symmetric=True)


_TOY_T = 0.4e-6
_TOY_OM = np.array([2 * np.pi * 30e6])
_TOY_DW = np.array([2 * np.pi * 10e6])


def _toy_solve(**kw):
    kw.setdefault("rtol", 1e-10)
    kw.setdefault("atol", 1e-13)
    return mls297.integrate_batch(
        _toy_sym_ops(), _TOY_T, kw.pop("om", _TOY_OM), kw.pop("dw", _TOY_DW), **kw)


def test_batched_kernel_matches_dense_reference():
    ops = _toy_ops()
    t_gate = 0.4e-6
    om = np.array([mls297.TAU * 12e6])
    dw = np.array([mls297.TAU * 15e6])
    # Real signature is (ops, t_gate, omega_297, d_sweep, ...) — no explicit
    # state-labels arg — and the container is `psi_final` (n_points, 4, dim)
    # indexed by LOGICAL_INPUTS (the brief called it `states_final`).
    res = mls297.integrate_batch(ops, t_gate, om, dw, rtol=1e-10, atol=1e-13)
    from scipy.integrate import solve_ivp

    def href(t):
        s = t / t_gate
        amp = math.sqrt(float(mls297.envelope(s)))
        phi = float(mls297.phase_rad(t, t_gate, float(dw[0])))
        c = float(om[0]) * amp * np.exp(-1j * phi)
        return (np.diag(ops.h_static_diag).astype(complex)
                + c.real * ops.x297 + c.imag * ops.y297)

    for label in ("00", "01", "11"):
        li = ("00", "01", "10", "11").index(label)
        psi0 = np.zeros(16, complex)
        psi0[ops.logical_indices[li]] = 1.0
        sol = solve_ivp(lambda t, y: -1j * (href(t) @ y), (0, t_gate), psi0,
                        rtol=1e-11, atol=1e-13, dense_output=False)
        ref = sol.y[:, -1]
        got = res.psi_final[0, li]
        assert np.max(np.abs(got - ref)) < 1e-7


def test_scatter_weights_count_r_and_garb_atoms():
    w = mls297._scatter_weight_vectors(4)
    assert set(w) == {"p_ryd", "p_r_garb"}
    idx_rr = 2 * 4 + 2
    assert w["p_ryd"][idx_rr] == 2.0 and w["p_r_garb"][idx_rr] == 0.0
    idx_1g = 1 * 4 + 3
    assert w["p_r_garb"][idx_1g] == 1.0


def test_swap_reconstruction_matches_direct_propagation():
    res_swap = _toy_solve(use_swap=True)
    res_all4 = _toy_solve(use_swap=False)
    assert res_swap.used_swap and not res_all4.used_swap
    assert np.max(np.abs(res_swap.psi_final - res_all4.psi_final)) < 1e-9
    # 01 and 10 must not be identical states (the permutation matters)
    assert np.max(np.abs(res_swap.psi_final[0, 1] - res_swap.psi_final[0, 2])) > 1e-3


def test_segmented_equals_unsegmented():
    res_seg = _toy_solve(segmented=True)
    res_one = _toy_solve(segmented=False)
    assert np.max(np.abs(res_seg.psi_final - res_one.psi_final)) < 1e-8


def test_batched_points_match_isolated_points():
    om = 2 * np.pi * np.array([20e6, 30e6, 45e6])
    dw = 2 * np.pi * np.array([8e6, 10e6, 14e6])
    batched = _toy_solve(om=om, dw=dw)
    for i in range(3):
        alone = _toy_solve(om=om[i:i + 1], dw=dw[i:i + 1])
        assert np.max(np.abs(batched.psi_final[i] - alone.psi_final[0])) < 1e-8
        assert abs(batched.max_leakage[i] - alone.max_leakage[0]) < 1e-10


def test_trajectory_sampling_and_time_dependent_restore():
    t_eval = np.linspace(0.0, _TOY_T, 41)
    res = _toy_solve(t_eval=t_eval)
    assert res.states is not None
    assert res.states.shape == (41, 1, 4, 16)
    np.testing.assert_allclose(res.times, t_eval, rtol=0, atol=1e-18)
    assert np.max(np.abs(res.states[-1] - res.psi_final)) < 1e-8
    # t = 0 sample must be the unrotated initial basis states
    for j, idx in enumerate([0, 1, 4, 5]):
        expect = np.zeros(16)
        expect[idx] = 1.0
        np.testing.assert_allclose(np.abs(res.states[0, 0, j]), expect, atol=1e-12)
        assert res.states[0, 0, j][idx] == pytest.approx(1.0)  # phase too


# ── persistence: manifest, chunks, resume, exports (dim 16, no omega_1013) ────


def _mini_cfg():
    return mls297.ScanConfig()


def _mini_store(tmp_path, model_hash="modelhash-1"):
    store = mls297.Store(str(tmp_path / "scan"))
    store.ensure_dirs()
    manifest = store.init_or_validate_manifest(
        _mini_cfg(), model_hash=model_hash, code_hash="codehash", run_meta={})
    return store, manifest


def _fake_result(n, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal((n, 4, dim)) + 1j * rng.standard_normal((n, 4, dim))
    psi /= np.linalg.norm(psi, axis=2, keepdims=True)
    leak = rng.uniform(1e-6, 1e-2, size=(n, 4))
    return mls297.BatchResult(
        psi_final=psi, leakage=leak, max_leakage=leak.max(axis=1),
        worst_input=[mls297.LOGICAL_INPUTS[int(i)] for i in leak.argmax(axis=1)],
        return_prob=rng.uniform(0.9, 1.0, size=(n, 4)),
        norm_err=np.full((n, 4), 1e-13), nfev=1234, used_swap=True,
    )


def test_manifest_has_297_axes_and_no_1013(tmp_path):
    _, manifest = _mini_store(tmp_path)
    assert manifest["axes"]["ryd_n"] == list(mls297.RYD_N)
    assert manifest["axes"]["omega297_anchors_mhz"] == ["9", "12", "15", "18"]
    assert not any("1013" in k for k in manifest)


def test_chunk_roundtrip_preserves_states_exactly(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:3]
    result = _fake_result(3)
    path = store.write_result_chunk(
        1, manifest, keys, _mini_cfg(), "production", 1e-9, 1e-12,
        "batch-a", result, runtime_s=30.0)
    with np.load(path, allow_pickle=False) as d:  # loads without pickle
        np.testing.assert_array_equal(d["psi_final"], result.psi_final)
        assert d["psi_final"].dtype == np.complex128
        assert list(d["status"]) == ["ok"] * 3
    records = store.load_records(manifest)
    assert len(records) == 3
    assert {r.key for r in records} == set(keys)
    np.testing.assert_array_equal(records[0].psi_final, result.psi_final[0])


def test_atomic_write_rejects_object_arrays_and_leaves_no_file(tmp_path):
    target = str(tmp_path / "chunk_000001.npz")
    with pytest.raises(TypeError):
        mls297._atomic_savez(target, bad=np.array([{"a": 1}], dtype=object))
    assert not any(tmp_path.iterdir())  # neither the file nor a temp survives


def test_stale_tmp_files_are_ignored_by_the_loader(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), "production",
                             1e-9, 1e-12, "b", _fake_result(1), 1.0)
    (Path(store.chunks_dir) / "chunk_000002.npz.tmp-dead").write_bytes(b"garbage")
    assert len(store.load_records(manifest)) == 1
    assert store.next_seq() == 2


@pytest.mark.parametrize("field, doctored, match", [
    ("physics_hash", "0" * 64, "physics_hash mismatch"),
    ("model_hash", "0" * 64, "model_hash mismatch"),
    ("pulse_hash", "0" * 64, "pulse_hash mismatch"),
])
def test_manifest_guard_rejects_mismatched_provenance(tmp_path, field, doctored, match):
    """init_or_validate_manifest refuses to resume when any recorded provenance
    hash disagrees with the live code/model.  (Omega_1013 is gone in the 297
    single-photon fork, so there is no detuning-guard arm.)"""
    store, _ = _mini_store(tmp_path, model_hash="modelhash-1")
    manifest_path = Path(store.manifest_path)
    doc = json.loads(manifest_path.read_text())
    doc[field] = doctored
    manifest_path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match=match):
        store.init_or_validate_manifest(
            _mini_cfg(), model_hash="modelhash-1", code_hash="codehash",
            run_meta={})


@pytest.mark.parametrize("field", ["physics_hash", "model_hash", "pulse_hash"])
def test_chunk_guard_refuses_to_merge_foreign_provenance(tmp_path, field):
    """load_records refuses to merge a chunk whose stamped hash differs from the
    manifest."""
    store, manifest = _mini_store(tmp_path)
    foreign = dict(manifest, **{field: "0" * 64})
    store.write_result_chunk(1, foreign, mls297.panel_keys(0, 0, 0)[:1], _mini_cfg(),
                             "production", 1e-9, 1e-12, "b", _fake_result(1), 1.0)
    with pytest.raises(RuntimeError, match=field):
        store.load_records(manifest)


def test_resume_dedup_and_tier_preference(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:2]
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), "production",
                             1e-9, 1e-12, "b1", _fake_result(2, seed=1), 10.0)
    store.write_result_chunk(2, manifest, keys[:1], _mini_cfg(), "audit",
                             1e-10, 1e-13, "b2", _fake_result(1, seed=2), 12.0)
    records = store.load_records(manifest)
    best = mls297.best_records(records)
    assert best[keys[0]].tier == "audit"          # tightest record wins exports
    assert best[keys[1]].tier == "production"
    done = mls297.completed_keys(records)
    missing = [k for k in mls297.panel_keys(0, 0, 0) if k not in done]
    assert len(missing) == 16 - 2                 # resume schedules only the rest
    pairs = mls297.audit_pairs(records)
    assert len(pairs) == 1 and pairs[0][0] == keys[0]


def test_failed_points_recorded_and_not_counted_done(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(
        1, manifest, keys, _mini_cfg(), "production", 1e-9, 1e-12,
        "bf", None, 5.0, statuses=["timeout"], message="timeout after 5s",
        retry_count=2)
    records = store.load_records(manifest)
    assert records[0].status == "timeout"
    assert records[0].retry_count == 2
    assert mls297.completed_keys(records) == set()
    assert mls297.best_records(records) == {}


def test_export_store_writes_merged_npz_and_csv(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), "production",
                             1e-9, 1e-12, "b", _fake_result(len(keys)), 60.0)
    merged, csv_path = mls297.export_store(store)
    with np.load(merged, allow_pickle=False) as d:
        assert d["max_leakage"].shape == (16,)
        assert d["psi_final"].shape == (16, 4, 16)
        assert "ryd_n" in d.files and "omega297_mhz" in d.files
    lines = Path(csv_path).read_text().strip().splitlines()
    assert len(lines) == 17 and lines[0].startswith("point_id,ryd_n,")


# ── scheduling helper ────────────────────────────────────────────────────────


def test_group_batches_stays_within_panel_and_orders_axes():
    keys = mls297.panel_keys(0, 0, 0) + mls297.panel_keys(1, 2, 0)
    batches = mls297.group_batches(keys, batch_size=6)
    assert all(len({k.panel for k in b.keys}) == 1 for b in batches)
    assert sum(len(b.keys) for b in batches) == 32
    first = batches[0].keys
    fracs = [(Fraction(k.om_num, k.om_den), Fraction(k.dw_num, k.dw_den)) for k in first]
    assert fracs == sorted(fracs)
    with pytest.raises(ValueError):
        mls297.Batch(keys=keys)  # crosses panels


# ── plotting / status / CLI smoke ────────────────────────────────────────────


def test_plot_and_status_smoke(tmp_path, capsys):
    store, manifest = _mini_store(tmp_path)
    rng = np.random.default_rng(3)
    gammas = {ni: {"p_ryd": 6.6e3, "p_r_garb": 6.6e3} for ni in (0, 3)}
    for seq, panel in enumerate([(0, 0), (3, 4)], start=1):
        keys = mls297.panel_keys(panel[0], panel[1], 1)   # full 7x7 grid
        res = _fake_result(len(keys), seed=seq)
        res.leakage = rng.uniform(1e-5, 1e-1, size=(len(keys), 4))
        res.max_leakage = res.leakage.max(axis=1)
        store.write_result_chunk(seq, manifest, keys, _mini_cfg(), "production",
                                 1e-9, 1e-12, f"b{seq}", res, 60.0)
        scatter = {ch: rng.uniform(1e-6, 1e-3, size=(len(keys), 4))
                   for ch in mls297.SCATTER_CHANNELS}
        store.write_scatter_chunk(seq, manifest, keys, _mini_cfg(), gammas, 1e-9,
                                  1e-12, f"s{seq}", scatter, res.max_leakage, 60.0)
    for metric in ("max_leakage", "p_ryd", "p_r_garb", "p_loss_total", "total_error"):
        mls297.cmd_plot(Namespace(output=store.root, dpi=60, veil=True, metric=metric))
        assert (Path(store.plots_dir) / f"{metric}_8x9.png").exists()
        assert (Path(store.plots_dir) / f"{metric}_8x9.pdf").exists()
    assert not list(Path(store.plots_dir).glob("panel_*.png"))

    mls297.cmd_status(Namespace(output=store.root))
    out = capsys.readouterr().out
    assert "records:" in out and "98 unique ok points" in out


def test_default_output_derivation():
    parser = mls297.build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.spacing_um == 3.0 and args.output is None
    assert mls297._default_output(3.0) == os.path.join(
        "results", "max_leakage_297", "a3.0")


def test_plot_metric_choices_are_the_five_297_metrics():
    parser = mls297.build_parser()
    for m in ("max_leakage", "p_ryd", "p_r_garb", "p_loss_total", "total_error"):
        assert parser.parse_args(["plot", "--metric", m]).metric == m
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--metric", "p_mid"])


def test_cli_parser_covers_subcommands_and_locked_invocation():
    parser = mls297.build_parser()
    args = parser.parse_args(["run", "--dry-run", "--target-level", "13"])
    assert args.func is mls297.cmd_run and args.dry_run and args.target_level == "13"
    assert parser.parse_args(
        ["audit", "--audit-point", "n0_t0_om0-1_dw0-1"]).func is mls297.cmd_audit
    assert parser.parse_args(["status"]).func is mls297.cmd_status
    args = parser.parse_args(["scatter", "--level", "7", "--workers", "auto",
                              "--batch-size", "auto"])
    assert args.func is mls297.cmd_scatter and args.level == "7"
    assert parser.parse_args(["plot", "--metric", "p_loss_total"]).metric == "p_loss_total"
    assert parser.parse_args(["plot", "--metric", "total_error"]).metric == "total_error"

    # No pinned two-photon store: --output defaults to None and the store dir is
    # derived from --spacing-um (see main()).
    args = parser.parse_args(["run", "--dry-run"])
    assert args.output is None and args.spacing_um == 3.0
    assert mls297._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_297", "a3.0")
    args = parser.parse_args(["scatter", "--level", "13", "--spacing-um", "7"])
    assert args.spacing_um == 7.0
    assert mls297._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_297", "a7.0")
