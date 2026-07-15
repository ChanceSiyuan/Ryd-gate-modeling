"""Focused tests for scripts/max_leakage_ode_sweep.py.

Grouped bottom-up like the implementation slices: nested axes/keys, analytic
pulse, block-max DOP853 error norm, integration kernel invariants (on a small
synthetic swap-symmetric model — no ARC needed), persistence/resume, scheduling
helpers, refinement priorities, plotting/export smoke — plus the real rb87_7_mp
model checks (ARC) with the expensive solver-equivalence runs marked ``slow``.
"""

import importlib.util
import json
import sys
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "max_leakage_ode_sweep", ROOT / "scripts" / "max_leakage_ode_sweep.py")
mls = importlib.util.module_from_spec(_spec)
sys.modules["max_leakage_ode_sweep"] = mls
_spec.loader.exec_module(mls)


# ── nested axes and canonical keys ───────────────────────────────────────────


def test_axis_sizes_and_cumulative_totals():
    for level, size in enumerate(mls.LEVEL_SIZES):
        assert len(mls.axis_coords(level)) == size
        assert len(mls.all_keys(level)) == 72 * size * size
    assert [len(mls.all_keys(lv)) for lv in range(4)] == [1152, 3528, 12168, 45000]


def test_axis_values_match_locked_specification():
    om0 = [float(v) for v in mls.axis_values_mhz(mls.OMEGA_ANCHORS_MHZ, 0)]
    assert om0 == pytest.approx([200.0, 1400 / 3, 2200 / 3, 1000.0])
    om1 = [float(v) for v in mls.axis_values_mhz(mls.OMEGA_ANCHORS_MHZ, 1)]
    assert om1 == pytest.approx(
        [200.0, 1000 / 3, 1400 / 3, 600.0, 2200 / 3, 2600 / 3, 1000.0])
    dw1 = [float(v) for v in mls.axis_values_mhz(mls.DSWEEP_ANCHORS_MHZ, 1)]
    assert dw1 == pytest.approx([2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0])


def test_finer_levels_nest_and_20mhz_is_a_node_everywhere():
    for level in range(3):
        coarse = set(mls.axis_coords(level))
        fine = set(mls.axis_coords(level + 1))
        assert coarse < fine  # midpoint insertion reuses every previous exact node
    for level in range(4):
        values = [float(v) for v in mls.axis_values_mhz(mls.DSWEEP_ANCHORS_MHZ, level)]
        assert 20.0 in values


def test_canonical_coordinates_reduce_to_identical_keys():
    # level-1 coordinate k=2/den=2 is the level-0 node k=1/den=1
    assert mls.canon_coord(2, 2) == (1, 1)
    k_a = mls.make_key(0, 0, (2, 2), (4, 4))
    k_b = mls.make_key(0, 0, (1, 1), (1, 1))
    assert k_a == k_b and k_a.id() == k_b.id()
    with pytest.raises(ValueError):
        mls.canon_coord(9, 2)  # outside [0, 3]


def test_point_key_values_and_level():
    k = mls.make_key(3, 4, (3, 2), (3, 2))
    assert float(k.omega_mhz()) == 600.0
    assert float(k.dsweep_mhz()) == 15.0
    assert k.level() == 1
    assert mls.make_key(0, 0, (1, 1), (0, 1)).level() == 0


def test_pilot_keys_dedup_and_reusability():
    pkeys = mls.pilot_keys()
    assert len(pkeys) == len(set(pkeys)) == 72 + 16
    level1 = set(mls.all_keys(1))
    assert all(k in level1 for k in pkeys[:72])       # centers are 7x7 nodes
    level0 = set(mls.all_keys(0))
    assert all(k in level0 for k in pkeys[72:])       # extremes are 4x4 nodes


# ── analytic pulse ───────────────────────────────────────────────────────────


def test_envelope_shape_and_symmetry():
    r = 0.15
    assert mls.envelope(0.0, r) == 0.0
    assert mls.envelope(0.075, r) == pytest.approx(0.5)   # q(1/2) = 0.5
    assert mls.envelope(0.5, r) == 1.0
    s = np.linspace(0, 1, 101)
    assert np.allclose(mls.envelope(s, r), mls.envelope(1.0 - s, r))


def test_envelope_integral_continuity_and_endpoints():
    r = 0.15
    assert mls.envelope_integral(0.0, r) == 0.0
    assert float(mls.envelope_integral(1.0, r)) == pytest.approx(1.0 - r)
    for s0 in (r, 1.0 - r):  # branch seams
        eps = 1e-9
        lo = float(mls.envelope_integral(s0 - eps, r))
        hi = float(mls.envelope_integral(s0 + eps, r))
        assert abs(hi - lo) < 1e-8
    # J' == E by quadrature on a coarse grid
    s = np.linspace(0.0, 1.0, 20001)
    j_num = np.concatenate([[0.0], np.cumsum(
        0.5 * (mls.envelope(s[1:], r) + mls.envelope(s[:-1], r)) * np.diff(s))])
    assert np.max(np.abs(j_num - mls.envelope_integral(s, r))) < 1e-8


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


def test_phase_units_pure_cosine_sweep():
    # With Dr - D1 = 0 the phase is exactly -(D T / 2 pi) sin(2 pi s).
    t_gate, d = 2.0e-6, 2 * np.pi * 20e6
    t = 0.37 * t_gate
    expect = -(d * t_gate / (2 * np.pi)) * np.sin(2 * np.pi * 0.37)
    assert float(mls.phase_rad(t, t_gate, d, 0.0)) == pytest.approx(expect, rel=1e-12)
    assert abs(float(mls.phase_rad(t, t_gate, d, 0.0))) > 2 * np.pi  # unwrapped


def test_stark_coefficients_signs_and_magnitude():
    om420, om1013, delta = 2 * np.pi * 600e6, 2 * np.pi * 489.6e6, 2 * np.pi * 20e9
    d1, dr = mls.stark_coefficients(om420, om1013, delta)
    assert d1 == pytest.approx(-(4 / 3) * om420 ** 2 / (4 * delta))
    assert dr == pytest.approx(-(om1013 ** 2) / (4 * delta))
    assert d1 < 0 and dr < 0


# ── block-max DOP853 error norm ──────────────────────────────────────────────


def test_error_norm_matches_installed_scipy():
    assert mls.verify_scipy_error_norm() < mls.ERR_NORM_REL_TOL


def test_block_solver_matches_stock_dop853_per_block():
    """Two decoupled oscillator blocks integrated together under the block-max
    norm agree with each block integrated alone by stock scipy DOP853."""
    import scipy.integrate

    w = [3.0, 41.0]

    def rhs_block(wi):
        return lambda t, y: -1j * wi * y

    def rhs_joint(t, y):
        out = np.empty_like(y)
        out[:4] = -1j * w[0] * y[:4]
        out[4:] = -1j * w[1] * y[4:]
        return out

    y0 = (np.arange(1, 9) + 1j) / 10.0
    cls = mls.make_block_solver_class(4)
    solver = cls(rhs_joint, 0.0, y0.astype(complex), 2.0, rtol=1e-10, atol=1e-12)
    while solver.status == "running":
        solver.step()
    assert solver.status == "finished"
    for b in range(2):
        ref = scipy.integrate.solve_ivp(
            rhs_block(w[b]), (0.0, 2.0), y0[4 * b:4 * b + 4].astype(complex),
            method="DOP853", rtol=1e-10, atol=1e-12)
        exact = y0[4 * b:4 * b + 4] * np.exp(-1j * w[b] * 2.0)
        assert np.max(np.abs(solver.y[4 * b:4 * b + 4] - exact)) < 1e-8
        assert np.max(np.abs(ref.y[:, -1] - exact)) < 1e-8


# ── kernel invariants on a synthetic swap-symmetric model (no ARC) ───────────


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


def test_kernel_norms_and_direct_leakage():
    res = _toy_solve()
    assert res.norm_err.max() < 1e-10
    # direct Q population equals 1 - P population to machine accuracy here
    pops = np.abs(res.psi_final) ** 2
    p_log = pops[:, :, [0, 1, 3, 4]].sum(axis=2)
    assert np.allclose(res.leakage, 1.0 - p_log, atol=1e-12)
    assert res.max_leakage[0] == res.leakage[0].max()
    assert res.worst_input[0] == mls.LOGICAL_INPUTS[int(np.argmax(res.leakage[0]))]
    assert 0 < res.max_leakage[0] < 1


def test_swap_reconstruction_matches_direct_propagation():
    res_swap = _toy_solve(use_swap=True)
    res_all4 = _toy_solve(use_swap=False)
    assert res_swap.used_swap and not res_all4.used_swap
    assert np.max(np.abs(res_swap.psi_final - res_all4.psi_final)) < 1e-9
    # 01 and 10 must not be identical states (the permutation matters)
    assert np.max(np.abs(res_swap.psi_final[0, 1] - res_swap.psi_final[0, 2])) > 1e-3


def test_scalar_shift_is_exact_global_phase():
    res_on = _toy_solve(use_shifts=True)
    res_off = _toy_solve(use_shifts=False)
    assert np.max(np.abs(res_on.psi_final - res_off.psi_final)) < 1e-8
    assert np.max(np.abs(res_on.leakage - res_off.leakage)) < 1e-11


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
    assert res.states.shape == (41, 1, 4, 9)
    np.testing.assert_allclose(res.times, t_eval, rtol=0, atol=1e-18)
    assert np.max(np.abs(res.states[-1] - res.psi_final)) < 1e-8
    # t = 0 sample must be the unrotated initial basis states
    for j, idx in enumerate([0, 1, 3, 4]):
        expect = np.zeros(9)
        expect[idx] = 1.0
        np.testing.assert_allclose(np.abs(res.states[0, 0, j]), expect, atol=1e-12)
        assert res.states[0, 0, j][idx] == pytest.approx(1.0)  # phase too


# ── persistence: manifest, chunks, resume, exports ───────────────────────────


def _mini_cfg():
    return mls.ScanConfig()


def _mini_store(tmp_path, model_hash="modelhash-1"):
    store = mls.Store(str(tmp_path / "scan"))
    store.ensure_dirs()
    manifest = store.init_or_validate_manifest(
        _mini_cfg(), omega_1013=2 * np.pi * 489.6e6, model_hash=model_hash,
        code_hash="codehash", run_meta={})
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


def test_chunk_roundtrip_preserves_states_exactly(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:3]
    result = _fake_result(3)
    path = store.write_result_chunk(
        1, manifest, keys, _mini_cfg(), 1.0, "production", 1e-9, 1e-12,
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
        mls._atomic_savez(target, bad=np.array([{"a": 1}], dtype=object))
    assert not any(tmp_path.iterdir())  # neither the file nor a temp survives


def test_stale_tmp_files_are_ignored_by_the_loader(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), 1.0, "production",
                             1e-9, 1e-12, "b", _fake_result(1), 1.0)
    (Path(store.chunks_dir) / "chunk_000002.npz.tmp-dead").write_bytes(b"garbage")
    assert len(store.load_records(manifest)) == 1
    assert store.next_seq() == 2


def test_incompatible_manifest_is_rejected(tmp_path):
    store, _ = _mini_store(tmp_path, model_hash="modelhash-1")
    with pytest.raises(RuntimeError, match="model_hash mismatch"):
        store.init_or_validate_manifest(
            _mini_cfg(), omega_1013=2 * np.pi * 489.6e6, model_hash="OTHER",
            code_hash="codehash", run_meta={})


def test_chunks_from_other_model_hash_refuse_to_merge(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    foreign = dict(manifest, model_hash="foreign")
    store.write_result_chunk(1, foreign, keys, _mini_cfg(), 1.0, "production",
                             1e-9, 1e-12, "b", _fake_result(1), 1.0)
    with pytest.raises(RuntimeError, match="model_hash"):
        store.load_records(manifest)


def test_resume_dedup_and_tier_preference(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:2]
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), 1.0, "production",
                             1e-9, 1e-12, "b1", _fake_result(2, seed=1), 10.0)
    store.write_result_chunk(2, manifest, keys[:1], _mini_cfg(), 1.0, "audit",
                             1e-10, 1e-13, "b2", _fake_result(1, seed=2), 12.0)
    records = store.load_records(manifest)
    best = mls.best_records(records)
    assert best[keys[0]].tier == "audit"          # tightest record wins exports
    assert best[keys[1]].tier == "production"
    done = mls.completed_keys(records)
    missing = [k for k in mls.panel_keys(0, 0, 0) if k not in done]
    assert len(missing) == 16 - 2                 # resume schedules only the rest
    pairs = mls.audit_pairs(records)
    assert len(pairs) == 1 and pairs[0][0] == keys[0]


def test_failed_points_recorded_and_not_counted_done(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(
        1, manifest, keys, _mini_cfg(), 1.0, "production", 1e-9, 1e-12,
        "bf", None, 5.0, statuses=["timeout"], message="timeout after 5s",
        retry_count=2)
    records = store.load_records(manifest)
    assert records[0].status == "timeout"
    assert records[0].retry_count == 2
    assert mls.completed_keys(records) == set()
    assert mls.best_records(records) == {}


def test_export_store_writes_merged_npz_and_csv(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), 1.0, "production",
                             1e-9, 1e-12, "b", _fake_result(len(keys)), 60.0)
    merged, csv_path = mls.export_store(store)
    with np.load(merged, allow_pickle=False) as d:
        assert d["max_leakage"].shape == (16,)
        assert d["psi_final"].shape == (16, 4, 49)
    lines = Path(csv_path).read_text().strip().splitlines()
    assert len(lines) == 17 and lines[0].startswith("point_id,")


# ── scheduling helpers ───────────────────────────────────────────────────────


def test_group_batches_stays_within_panel_and_orders_axes():
    keys = mls.panel_keys(0, 0, 0) + mls.panel_keys(1, 2, 0)
    batches = mls.group_batches(keys, batch_size=6)
    assert all(len({k.panel for k in b.keys}) == 1 for b in batches)
    assert sum(len(b.keys) for b in batches) == 32
    first = batches[0].keys
    fracs = [(Fraction(k.om_num, k.om_den), Fraction(k.dw_num, k.dw_den)) for k in first]
    assert fracs == sorted(fracs)
    with pytest.raises(ValueError):
        mls.Batch(keys=keys)  # crosses panels


def test_batch_split_isolates_points_and_counts_retries():
    batch = mls.Batch(keys=mls.panel_keys(0, 0, 0)[:5], priority_scores=[1, 2, 3, 4, 5])
    parts = mls.Runner._split(None, batch)
    assert [len(p.keys) for p in parts] == [2, 3]
    assert all(p.retry_count == 1 for p in parts)
    assert parts[0].priority_scores == [1, 2]
    single = mls.Batch(keys=batch.keys[:1])
    assert [len(p.keys) for p in mls.Runner._split(None, single)] == [1]


def test_budget_scheduler_defers_everything_past_the_deadline(tmp_path):
    """With the dispatch deadline already in the past, run_batches launches no
    work, defers every point, and leaves the reserve intact (no chunks)."""
    store, manifest = _mini_store(tmp_path)
    args = Namespace(workers=2, batch_size=4, budget_hours=0.0, reserve_hours=1.0,
                     point_timeout=60.0, rtol=1e-9, atol=1e-12,
                     audit_rtol=1e-10, audit_atol=1e-13)
    runner = mls.Runner(store, manifest, _mini_cfg(), 1.0, args, mls.CostModel(_mini_cfg()))
    try:
        batches = mls.group_batches(mls.panel_keys(0, 0, 0), batch_size=4)
        runner.run_batches(batches, "test-budget")
    finally:
        runner.shutdown()
    assert runner.deferred == 16
    assert runner.completed_points == 0
    assert store.load_records(manifest) == []


def test_store_lock_rejects_concurrent_writers(tmp_path):
    store, manifest = _mini_store(tmp_path)
    args = Namespace(workers=1, batch_size=1, budget_hours=1.0, reserve_hours=0.0,
                     point_timeout=60.0, rtol=1e-9, atol=1e-12,
                     audit_rtol=1e-10, audit_atol=1e-13)
    runner = mls.Runner(store, manifest, _mini_cfg(), 1.0, args, mls.CostModel(_mini_cfg()))
    try:
        import fcntl
        with open(Path(store.logs_dir) / "store.lock") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        runner.shutdown()


def test_cost_model_scales_with_gate_time_and_inflates():
    cfg = _mini_cfg()
    cost = mls.CostModel(cfg)
    assert cost.predict_point((0, 0)) == pytest.approx(150.0 * 1.0)
    cost.observe((0, 0), 100.0)                 # T = 1.0 us panel
    assert cost.predict_point((0, 0)) == 100.0
    assert cost.predict_point((0, 8)) == pytest.approx(100.0 * 4.5)  # same row, scaled
    assert cost.inflation_p90() == 1.5          # too few ratios -> conservative
    for r in (1.0, 1.1, 1.2, 1.0, 1.05, 1.3):
        cost.observe_batch(100.0, 100.0 * r)
    assert 1.0 <= cost.inflation_p90() <= 1.3
    eta = cost.eta_seconds(mls.panel_keys(0, 0, 0), n_workers=4)
    assert eta == pytest.approx(16 * 100.0 * cost.inflation_p90() / 4)


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
        store.write_result_chunk(seq, manifest, keys, _mini_cfg(), 1.0,
                                 "production", 1e-9, 1e-12, f"b{seq}", res, 60.0)
    args = Namespace(output=store.root, dpi=60, individual=False, veil=True,
                     metric="max_leakage")
    mls.cmd_plot(args)
    assert (Path(store.plots_dir) / "max_leakage_8x9.png").exists()
    assert (Path(store.plots_dir) / "max_leakage_8x9.pdf").exists()

    mls.cmd_status(Namespace(output=store.root))
    out = capsys.readouterr().out
    assert "records:" in out and "98 unique ok points" in out


def test_cli_parser_covers_subcommands():
    parser = mls.build_parser()
    args = parser.parse_args(["run", "--dry-run", "--target-level", "13"])
    assert args.func is mls.cmd_run and args.dry_run and args.target_level == "13"
    args = parser.parse_args(["audit", "--audit-point", "d0_t0_om0-1_dw0-1"])
    assert args.func is mls.cmd_audit
    args = parser.parse_args(["status"])
    assert args.func is mls.cmd_status


def test_cli_accepts_the_agreed_production_invocation():
    """The handoff's locked production command must parse verbatim."""
    args = mls.build_parser().parse_args([
        "run", "--output", "results/max_leakage_ode",
        "--workers", "auto", "--batch-size", "auto", "--target-level", "auto",
        "--budget-hours", "24", "--reserve-hours", "2"])
    assert isinstance(args.workers, int) and 1 <= args.workers <= 40
    assert args.batch_size == 48
    assert args.target_level == "auto"


def test_pulse_hash_is_stable_and_guards_resume(tmp_path):
    h1, h2 = mls.pulse_hash(), mls.pulse_hash()
    assert h1 == h2 and len(h1) == 64
    store, manifest = _mini_store(tmp_path)
    assert manifest["pulse_hash"] == h1
    # a manifest recorded under a different pulse must refuse to resume
    manifest_path = Path(store.manifest_path)
    doctored = json.loads(manifest_path.read_text())
    doctored["pulse_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(doctored))
    with pytest.raises(RuntimeError, match="pulse_hash mismatch"):
        store.init_or_validate_manifest(
            _mini_cfg(), omega_1013=2 * np.pi * 489.6e6, model_hash="modelhash-1",
            code_hash="codehash", run_meta={})
    # and chunks stamped with a foreign pulse hash must refuse to merge
    store2, manifest2 = _mini_store(tmp_path / "b")
    foreign = dict(manifest2, pulse_hash="0" * 64)
    store2.write_result_chunk(1, foreign, mls.panel_keys(0, 0, 0)[:1], _mini_cfg(),
                              1.0, "production", 1e-9, 1e-12, "b", _fake_result(1), 1.0)
    with pytest.raises(RuntimeError, match="pulse_hash"):
        store2.load_records(manifest2)


def test_best_records_prefers_tighter_rtol_over_tier(tmp_path):
    """A loosened ad-hoc audit must not displace a tighter production record."""
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, _mini_cfg(), 1.0, "production",
                             1e-9, 1e-12, "b1", _fake_result(1, seed=1), 1.0)
    store.write_result_chunk(2, manifest, keys, _mini_cfg(), 1.0, "audit",
                             1e-8, 1e-11, "b2", _fake_result(1, seed=2), 1.0)
    best = mls.best_records(store.load_records(manifest))
    assert best[keys[0]].tier == "production" and best[keys[0]].rtol == 1e-9


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


# ── scattering supplement ────────────────────────────────────────────────────


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


def test_model_decay_rates_channel_mapping():
    stub = Namespace(level_structure=Namespace(
        mid_state_decay_rate=9.0e6, ryd_state_decay_rate=6.6e3,
        ryd_garb_decay_rate=7.7e3))
    rates = mls.model_decay_rates(stub)
    assert rates == {"p_mid": 9.0e6, "p_ryd": 6.6e3, "p_r_garb": 7.7e3}


def test_scatter_chunk_roundtrip_resume_and_isolation(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:2]
    gammas = {"p_mid": 9.03e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}
    scatter = {ch: np.random.default_rng(0).uniform(1e-6, 1e-2, (2, 4))
               for ch in mls.SCATTER_CHANNELS}
    path = store.write_scatter_chunk(
        1, manifest, keys, _mini_cfg(), gammas, 1e-9, 1e-12, "sb1",
        scatter, np.array([1e-4, 2e-4]), runtime_s=10.0)
    assert "scatter" in path
    rows = store.load_scatter_records(manifest)
    assert len(rows) == 2 and rows[0]["status"] == "ok"
    np.testing.assert_array_equal(rows[1]["p_mid"], scatter["p_mid"][1])
    # resume: done keys are skipped
    done = {r["key"] for r in rows if r["status"] == "ok"}
    assert set(keys) == done
    # isolation: no main-series chunk was created
    assert store.load_records(manifest) == []
    assert store.next_seq() == 1 and store.next_scatter_seq() == 2
    # foreign pulse hash refuses to merge
    foreign = dict(manifest, pulse_hash="0" * 64)
    store.write_scatter_chunk(2, foreign, keys, _mini_cfg(), gammas, 1e-9,
                              1e-12, "sb2", scatter, np.array([1e-4, 2e-4]), 1.0)
    with pytest.raises(RuntimeError, match="pulse_hash"):
        store.load_scatter_records(manifest)


def test_plot_metric_values_prefers_tight_rtol_and_totals(tmp_path):
    store, manifest = _mini_store(tmp_path)
    keys = mls.panel_keys(0, 0, 0)[:1]
    gammas = {"p_mid": 9.03e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}
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


def test_cli_scatter_subcommand_parses():
    args = mls.build_parser().parse_args(
        ["scatter", "--level", "7", "--workers", "auto", "--batch-size", "auto"])
    assert args.func is mls.cmd_scatter and args.level == "7"
    args = mls.build_parser().parse_args(["plot", "--metric", "p_loss_total"])
    assert args.metric == "p_loss_total"


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
    from ryd_gate.protocols.gate_cz import cz_effective_rabi, cz_rabi_maxes

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

    o420, o1013 = cz_rabi_maxes(system, omega_420, omega_1013)
    _, time_scale = cz_effective_rabi(system, o420, o1013)
    proto = CZProtocol(
        duration_ratio=t_gate / time_scale,
        A_420=lambda s: float(np.sqrt(mls.envelope(s))),
        phi_420=lambda s: float(mls.phase_rad(s * t_gate, t_gate, d_sweep, drmd1)),
        A_1013=lambda s: float(np.sqrt(mls.envelope(s))),
        phi_1013=lambda s: 0.0,
        omega_420_max=omega_420, omega_1013_max=omega_1013,
    )
    bound = system.with_protocol(proto)
    ref = rg.simulate(bound, [list(s) for s in mls.LOGICAL_INPUTS],
                      backend="exact_ode",
                      backend_options={"rtol": 1e-10, "atol": 1e-13})
    for j in range(4):
        dev = np.max(np.abs(res.psi_final[0, j] - ref[j].final_state))
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
