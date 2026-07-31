"""Focused tests for scripts/max_leakage_297_sweep.py (297 nm single-photon fork).

Script-specific coverage: locked nested-axis values, the Stark-free chirp/phase,
the p_ryd/p_r_garb scattering-budget channel tables, plotting/export/gate and the
CLI — plus the real rb87_297_clock_4 model check (ARC) marked ``slow``.  The shared
Store/Runner/CostModel/kernel machinery is exercised (parameterized over key
configs, including an n_idx/n layout) in tests/test_sweeplib.py.
"""

import hashlib
import importlib.util
import json
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

# The Monte Carlo cross-check lives in its own script; load it the same way (and
# under its own name, so the fork pool's workers can unpickle its entry point).
_mc_spec = importlib.util.spec_from_file_location(
    "phase_noise_mc_check", ROOT / "scripts" / "phase_noise_mc_check.py")
mc_check = importlib.util.module_from_spec(_mc_spec)
sys.modules["phase_noise_mc_check"] = mc_check
_mc_spec.loader.exec_module(mc_check)


# ── nested axes and canonical keys (locked 297 axis values; generic axis/key
# machinery is covered in tests/test_sweeplib.py) ─────────────────────────────


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


def test_20mhz_is_a_node_of_the_dsweep_axis_at_every_level():
    for level in range(4):
        vals = [float(v) for v in mls297.axis_values_mhz(mls297.DSWEEP_ANCHORS_MHZ, level)]
        assert 20.0 in vals


def test_point_key_id_uses_n_prefix_and_canonicalizes():
    k = mls297.make_key(2, 4, (2, 4), (6, 8))
    assert k == mls297.make_key(2, 4, (1, 2), (3, 4))
    assert k.id() == "n2_t4_om1-2_dw3-4"
    assert k.panel == (2, 4)
    assert float(k.omega_mhz()) == 10.5          # 9 + (12-9)*1/2  (om coord 1/2)
    assert float(k.dsweep_mhz()) == 8.0          # 2 + (10-2)*3/4  (dw coord 3/4)


def test_pilot_keys_dedup_and_reusability():
    pkeys = mls297.pilot_keys()
    assert len(pkeys) == len(set(pkeys)) == 72 + 16
    level1 = set(mls297.all_keys(1))
    assert all(k in level1 for k in pkeys[:72])       # centers are 7x7 nodes
    level0 = set(mls297.all_keys(0))
    assert all(k in level0 for k in pkeys[72:])       # extremes are 4x4 nodes


# ── analytic pulse (297 Stark-free chirp/phase; the shared quintic envelope is
# covered in tests/test_sweeplib.py) ──────────────────────────────────────────


def test_phase_is_exact_integral_of_chirp():
    t_gate, d_sweep = 1.7e-6, mls297.TAU * 13e6
    ts = np.linspace(0.0, t_gate, 4001)
    chirp = mls297.chirp_rad_s(ts, t_gate, d_sweep)
    phi_num = np.concatenate(
        [[0.0], np.cumsum((chirp[1:] + chirp[:-1]) * 0.5 * np.diff(ts))])
    phi = mls297.phase_rad(ts, t_gate, d_sweep)
    assert np.max(np.abs(phi - phi_num)) < 1e-6 * np.max(np.abs(phi))
    # The pure-cosine phase is not wrapped mod 2 pi, and the single-photon fork has
    # no differential AC-Stark shift to compensate.
    assert abs(float(mls297.phase_rad(0.37 * t_gate, t_gate, d_sweep))) > 2 * np.pi
    assert not hasattr(mls297, "stark_coefficients")


# ── toy swap-symmetric model shared by the scattering tests below (the block-max
# DOP853 norm and the kernel invariants are covered in tests/test_sweeplib.py) ─


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
_TOY_OM = np.array([2 * np.pi * 12e6])
_TOY_DW = np.array([2 * np.pi * 10e6])


def _toy_solve(**kw):
    kw.setdefault("rtol", 1e-10)
    kw.setdefault("atol", 1e-13)
    return mls297.integrate_batch(
        _toy_sym_ops(), _TOY_T, kw.pop("om", _TOY_OM), kw.pop("dw", _TOY_DW), **kw)


def test_scatter_weights_count_r_and_garb_atoms():
    w = mls297._scatter_weight_vectors(4)
    assert set(w) == {"p_ryd", "p_r_garb"}
    idx_rr = 2 * 4 + 2
    assert w["p_ryd"][idx_rr] == 2.0 and w["p_r_garb"][idx_rr] == 0.0
    idx_1g = 1 * 4 + 3
    assert w["p_r_garb"][idx_1g] == 1.0
    for v in w.values():
        assert v.shape == (16,)


def test_scattering_integrals_constant_population():
    """A state parked in |r,r> for the whole window gives p_ryd = 2*Gamma*T."""
    T, n_t = 2.0e-6, 51
    times = np.linspace(0.0, T, n_t)
    states = np.zeros((n_t, 1, 4, 16), dtype=complex)
    states[:, 0, :, 2 * 4 + 2] = 1.0        # both atoms in |r> (local level 2)
    gammas = {"p_ryd": 6.6e3, "p_r_garb": 6.6e3}
    out = mls297.scattering_integrals(times, states, gammas)
    assert out["p_ryd"].shape == (1, 4)
    assert np.allclose(out["p_ryd"], 2.0 * 6.6e3 * T, rtol=1e-12)
    assert np.allclose(out["p_r_garb"], 0.0)


def test_scatter_integrals_on_a_solved_toy_trajectory():
    """Integrals from a real solved trajectory: finite, nonnegative, and the driven
    Rydberg channel (local level 2 -> 'p_ryd') integrates to a nonzero value."""
    t_eval = np.linspace(0.0, _TOY_T, 61)
    res = _toy_solve(t_eval=t_eval)
    gammas = {"p_ryd": 1.0e4, "p_r_garb": 1.0e4}
    out = mls297.scattering_integrals(res.times, res.states, gammas)
    for ch in mls297.SCATTER_CHANNELS:
        assert out[ch].shape == (1, 4)
        assert np.all(out[ch] >= 0.0)
        assert np.all(np.isfinite(out[ch]))
    assert out["p_ryd"].max() > 0.0   # toy 297 drive populates level 2 (r)


def test_model_decay_rates_maps_channels():
    """model_decay_rates maps the r/r_garb model channels to the p_* groups (no p_mid)."""
    stub = Namespace(level_structure=Namespace(
        decay_rates_per_s={"r": {"total": 6.6e3}, "r_garb": {"total": 5.5e3}}))
    gammas = mls297.model_decay_rates(stub)
    assert gammas == {"p_ryd": 6.6e3, "p_r_garb": 5.5e3}


# ── mini store fixtures (real 297 config) for the plotting/export/gate tests ──
# (dim 16, no omega_1013 provenance block).


def _mini_cfg():
    return mls297.ScanConfig()


def _mini_store(tmp_path, model_hash="modelhash-1"):
    store = mls297.Store(str(tmp_path / "scan"))
    store.ensure_dirs()
    cfg = _mini_cfg()
    manifest = store.init_or_validate_manifest(
        cfg, model_hash, "codehash", {}, **mls297._manifest_extras(cfg))
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


def test_effective_batch_size_falls_back_to_one_without_a_passed_gate(tmp_path):
    """_effective_batch_size gates the requested size on a recorded, enabled
    packing acceptance; absent/failed gates fall back to one point per solve."""
    store, _ = _mini_store(tmp_path)
    assert mls297._effective_batch_size(store, Namespace(batch_size=1)) == 1
    assert mls297._effective_batch_size(store, Namespace(batch_size=48)) == 1   # no pilot
    pilot = Path(store.reports_dir) / "pilot.json"
    pilot.write_text(json.dumps({"packing_gate": {"enabled": True}}))
    assert mls297._effective_batch_size(store, Namespace(batch_size=48)) == 48
    pilot.write_text(json.dumps({"packing_gate": {"enabled": False}}))
    assert mls297._effective_batch_size(store, Namespace(batch_size=48)) == 1


def test_ensure_scatter_gate_runs_once_and_skips_when_recorded_ok(tmp_path, monkeypatch):
    """The gate is a per-store setup step: it runs (and records
    reports/scatter_gate.json) when absent or not ok, and is skipped when a passed
    record already exists — so run/scatter share one gate per store."""
    store, _ = _mini_store(tmp_path)
    calls = {"n": 0}

    def fake_gate(runner, st):
        calls["n"] += 1
        return {"ok": True, "point_id": "n0_t0_om3-2_dw3-2", "max_abs_dev": 0.0}

    monkeypatch.setattr(mls297, "_scatter_equivalence_gate", fake_gate)
    gate_path = Path(store.reports_dir) / "scatter_gate.json"

    out = mls297._ensure_scatter_gate(None, store)          # absent -> runs, records
    assert out["ok"] and calls["n"] == 1 and gate_path.exists()
    assert json.loads(gate_path.read_text())["ok"] is True

    mls297._ensure_scatter_gate(None, store)                # present-and-ok -> skipped
    assert calls["n"] == 1

    gate_path.write_text(json.dumps({"ok": False, "reason": "moved"}))
    mls297._ensure_scatter_gate(None, store)                # recorded not-ok -> re-runs
    assert calls["n"] == 2


# ── plotting smoke test (synthetic store) ────────────────────────────────────
#
# The shared plot machinery (holdout residuals, credibility floor, metric
# selection, renderer) is unit-tested parameterized in tests/test_sweeplib.py;
# here the 297 plot renders end-to-end for every metric (no p_mid), including the
# total_error coherent+scatter join, and never emits per-panel PNGs.


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


# ── physics/pulse fingerprints and the local swap permutation ────────────────


def test_physics_hash_covers_axes_and_spacing():
    base = mls297.ScanConfig().physics_hash()
    assert mls297.ScanConfig(spacing_um=4.0).physics_hash() != base
    assert mls297.ScanConfig(ryd_n=(50, 53)).physics_hash() != base


def test_pulse_hash_is_stable_and_recorded(tmp_path):
    """The pulse fingerprint is deterministic (64 hex chars) and is what the
    manifest records, so a later pulse edit is caught by the provenance guards
    (see tests/test_sweeplib.py for the refuse-to-merge arms)."""
    h1, h2 = mls297.pulse_hash(), mls297.pulse_hash()
    assert h1 == h2 and len(h1) == 64
    _, manifest = _mini_store(tmp_path)
    assert manifest["pulse_hash"] == h1


def test_swap_permutation_local_dim_4():
    perm = mls297._swap_permutation(4)
    a, b = np.divmod(np.arange(16), 4)
    assert np.array_equal(perm, b * 4 + a)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_default_output_derivation():
    parser = mls297.build_parser()
    args = parser.parse_args(["run", "--dry-run"])
    assert args.spacing_um == 3.0 and args.output is None
    assert mls297._default_output(3.0) == os.path.join(
        "results", "max_leakage_297", "a3.0")
    args = parser.parse_args(["scatter", "--level", "13", "--spacing-um", "7"])
    assert args.spacing_um == 7.0
    assert mls297._default_output(args.spacing_um) == os.path.join(
        "results", "max_leakage_297", "a7.0")


def test_plot_metric_choices_are_the_297_metrics_and_the_noise_model_flags():
    parser = mls297.build_parser()
    for m in ("max_leakage", "p_ryd", "p_r_garb", "p_loss_total", "total_error",
              "eps_phase", "total_error_phase"):
        assert parser.parse_args(["plot", "--metric", m]).metric == m
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--metric", "p_mid"])

    # The extrapolation above the 1 MHz measurement edge is a bracket, so both
    # policies must be selectable and the conservative one is the default.
    args = parser.parse_args(["plot", "--metric", "eps_phase"])
    assert args.laser == "ECDL" and args.extrapolation == "flat"
    args = parser.parse_args(["plot", "--metric", "eps_phase", "--laser", "seed",
                              "--extrapolation", "power"])
    assert args.laser == "seed" and args.extrapolation == "power"
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--laser", "1013"])
    with pytest.raises(SystemExit):
        parser.parse_args(["plot", "--extrapolation", "servo"])

    # --f-min defaults to the stored kernels' own edge, so it cannot change any
    # render that does not ask for it.
    assert args.f_min == mls297.KERNEL_F_MIN_HZ
    assert parser.parse_args(["plot", "--f-min", "10"]).f_min == 10.0


def test_cli_parser_covers_subcommands_and_locked_invocation():
    parser = mls297.build_parser()
    args = parser.parse_args(["run", "--dry-run", "--target-level", "13"])
    assert args.func is mls297.cmd_run and args.dry_run and args.target_level == "13"
    assert parser.parse_args(["run", "--dry-run"]).target_level == "13"  # explicit default
    assert parser.parse_args(
        ["audit", "--audit-point", "n0_t0_om0-1_dw0-1"]).func is mls297.cmd_audit
    assert parser.parse_args(["status"]).func is mls297.cmd_status
    args = parser.parse_args(["scatter", "--level", "7", "--workers", "auto",
                              "--batch-size", "auto"])
    assert args.func is mls297.cmd_scatter and args.level == "7"
    assert parser.parse_args(["plot", "--metric", "p_loss_total"]).metric == "p_loss_total"
    assert parser.parse_args(["plot", "--metric", "total_error"]).metric == "total_error"

    # The locked driver invocations must keep parsing verbatim.
    assert parser.parse_args(
        ["pilot", "--spacing-um", "5", "--workers", "auto",
         "--batch-size", "auto"]).func is mls297.cmd_pilot
    args = parser.parse_args(
        ["run", "--spacing-um", "5", "--workers", "auto", "--batch-size", "auto",
         "--target-level", "13"])
    assert args.func is mls297.cmd_run and args.target_level == "13"
    assert isinstance(args.workers, int) and 1 <= args.workers <= 40
    assert args.batch_size == 48

    # No pinned two-photon store: --output defaults to None and the store dir is
    # derived from --spacing-um (see main()).
    args = parser.parse_args(["run", "--dry-run"])
    assert args.output is None and args.spacing_um == 3.0

    # The deleted machinery (target-level auto, wall-budget flags, worker bench)
    # no longer parses.
    for gone in (["run", "--target-level", "auto"], ["run", "--budget-hours", "24"],
                 ["run", "--reserve-hours", "2"], ["pilot", "--bench-workers", "20,40"]):
        with pytest.raises(SystemExit):
            parser.parse_args(gone)


# ── filter-function pass (real rb87_297_clock_4 model; ARC required) ─────────


def test_adjoint_leg_reproduces_the_forward_propagator():
    """<phi_q(t)|psi_s(t)> must equal <q|U(T,0)|s>, independent of t.

    The backward-integrated adjoints are the only new solver leg; this pins them
    against the existing, already-validated forward kernel.  ``q`` now runs over the
    complete basis, not the 12 nonlogical states: the fidelity-loss metric projects
    with Q = 1 - |psi_0(T)><psi_0(T)|, so ||G||^2 needs every component.
    """
    cfg = mls297.ScanConfig()
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    t_gate, omega, dsw = 1e-6, 2 * np.pi * 13.5e6, 2 * np.pi * 15e6

    fwd = mls297.integrate_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-10, atol=1e-13, use_swap=False)
    out = mls297.integrate_adjoint_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-10, atol=1e-13, ramp=cfg.ramp_frac, n_t=257, with_overlaps=True)

    assert out["components"].shape == (1, 4, 257, 16)
    for si in range(4):
        target = fwd.psi_final[0, si]           # all 16 components
        # overlaps <phi_q(t)|psi_s(t)> are t-independent and equal <q|psi_s(T)>
        got = out["overlaps"][0, si]            # (n_t, 16)
        assert np.allclose(got, target[None, :], atol=1e-8)


def test_kernel_projection_is_the_rydberg_population_of_the_noiseless_run():
    """``<psi_0(T)|A(t)> = <psi_0(t)|N_r|psi_0(t)>``, the term the kernel subtracts.

    The projection is contracted out of the adjoint components so that
    Cauchy-Schwarz survives in floating point, but it also has a closed form on the
    forward leg alone — real, non-negative, and needing no backward solve.  Checking
    the two against each other pins the enlarged backward leg *and* the contraction.
    """
    cfg = mls297.ScanConfig()
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    t_gate, omega, dsw = 1e-6, 2 * np.pi * 13.5e6, 2 * np.pi * 15e6
    n_t = 257

    out = mls297.integrate_adjoint_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-10, atol=1e-13, ramp=cfg.ramp_frac, n_t=n_t)
    fwd = mls297.integrate_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]), rtol=1e-10, atol=1e-13,
        use_swap=False, t_eval=out["times"])

    n_r = mls297._rydberg_number_diag(16)
    expected = np.einsum("tpsi,i,tpsi->pst", fwd.states.conj(), n_r,
                         fwd.states).real
    got = out["projection"]
    assert got.shape == (1, 4, n_t)
    assert expected.max() > 0.1                              # not trivially zero
    assert np.allclose(got, expected, atol=1e-8)
    assert np.max(np.abs(got.imag)) < 1e-8                   # real, as the identity says


def test_filter_kernel_is_converged_at_the_production_sampling():
    """Halving dt must not move eps by more than 1%, at both ends of the T axis.

    Gate the quantity that is actually consumed — ``error_from_kernel`` under the
    flat (white) extrapolation, the model that weights the high-frequency end most —
    rather than ``K_b / K_max``: a per-bin mask is set by where the kernel peaks
    (~20 MHz, the drive scale) and can hide the decade where the trapezoid transform
    is marginal.  T = 4.5 us is the worst case on the grid: dt = 1.1 ns leaves only
    4.5 samples per period at f_max = 200 MHz.
    """
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    cfg = mls297.ScanConfig()
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    psd = PhaseNoisePSD.white(1e4)
    f_bins, _df = mls297.kernel_frequency_bins()
    args = dict(rtol=1e-9, atol=1e-12, ramp=cfg.ramp_frac)
    for t_gate in (1e-6, 4.5e-6):
        k1, k2 = (mls297.filter_kernels(ops, t_gate, np.array([2 * np.pi * 13.5e6]),
                                        np.array([2 * np.pi * 15e6]), n_t=n, **args)
                  for n in (mls297.KERNEL_N_T, 2 * mls297.KERNEL_N_T))
        eps1, eps2 = ([error_from_kernel(psd, f_bins, k[0, s]) for s in range(4)]
                      for k in (k1, k2))
        # |00> never reaches the Rydberg manifold (|0> is a dark spectator), so
        # N_r psi_00 = 0 and its kernel vanishes identically at any sampling.
        assert eps1[0] == eps2[0] == 0.0
        assert max(abs(a - b) / b for a, b in zip(eps1[1:], eps2[1:])) < 0.01


def test_kernel_fine_grid_resolves_the_fringes_over_the_whole_t_axis():
    """The evaluation grid must give ``f_max ln10 / p <= 1/T`` at every panel column.

    ``tests/test_phase_noise.py`` shows what happens when it does not; this pins that
    the rule actually covers the locked T axis, and that it never drops below the
    library floor at the short end.
    """
    cfg = mls297.ScanConfig()
    for t_us in cfg.t_gate_us:
        t_gate = t_us * 1e-6
        p = mls297.kernel_fine_per_decade(t_gate)
        assert p >= mls297.KERNEL_FINE_MIN
        spacing = mls297.KERNEL_F_MAX_HZ * np.log(10.0) / p
        assert spacing <= 1.0 / t_gate
    assert mls297.kernel_fine_per_decade(1e-6) == 461
    assert mls297.kernel_fine_per_decade(4.5e-6) == 2073


def test_rydberg_number_counts_both_297_legs():
    """N_r counts atoms in r AND r_garb: one laser drives both 297 legs, so the
    noise operator is exactly the sum of the two scattering-channel weights."""
    n_r = mls297._rydberg_number_diag(16)
    w = mls297._scatter_weight_vectors(4)
    assert np.array_equal(n_r, w["p_ryd"] + w["p_r_garb"])
    assert n_r[2 * 4 + 3] == 2.0        # |r, r_garb>: both atoms Rydberg
    assert n_r[1 * 4 + 2] == 1.0        # |1, r>
    assert n_r[0] == 0.0                # |0, 0>


def test_filter_chunk_roundtrip_and_failure_rows(tmp_path):
    """The filter series round-trips one (4, n_bins) kernel per point and records a
    failed batch as NaN rows, without touching the coherent or scatter series."""
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:3]
    f_bins, _df = mls297.kernel_frequency_bins()
    kernels = np.arange(3 * 4 * f_bins.size, dtype=float).reshape(3, 4, f_bins.size)
    store.write_filter_chunk(1, manifest, keys, _mini_cfg(), 1e-9, 1e-12,
                             mls297.KERNEL_N_T, "b1", kernels, f_bins, 30.0)
    store.write_filter_chunk(2, manifest, keys[:1], _mini_cfg(), 1e-9, 1e-12,
                             mls297.KERNEL_N_T, "b2",
                             np.full((1, 4, f_bins.size), np.nan), f_bins, 1.0,
                             statuses=["timeout"], message="slow")

    rows = store.load_filter_records(manifest)
    assert len(rows) == 4
    ok = [r for r in rows if r["status"] == "ok"]
    assert [r["key"] for r in ok] == list(keys)
    assert ok[0]["kernel"].shape == (4, f_bins.size)
    assert np.array_equal(ok[0]["kernel"], kernels[0])
    assert np.array_equal(ok[0]["f_bins"], f_bins)
    assert ok[0]["runtime_s"] == 10.0 and ok[0]["rtol"] == 1e-9
    bad = [r for r in rows if r["status"] == "timeout"]
    assert len(bad) == 1 and np.all(np.isnan(bad[0]["kernel"]))
    assert store.load_records(manifest) == []
    assert store.load_scatter_records(manifest) == []


def test_filter_chunk_refuses_a_wrong_shape_and_a_mixed_frequency_grid(tmp_path):
    """The (n, 4, n_bins) shape and the store-global frequency grid are enforced.

    Neither is covered by the three provenance hashes, so a changed
    KERNEL_BINS_PER_DECADE/KERNEL_F_* would otherwise leave a silently mixed store
    whose rows cannot be summed against one PSD — and cmd_filter's resume set would
    never recompute them.
    """
    store, manifest = _mini_store(tmp_path)
    keys = mls297.panel_keys(0, 0, 0)[:2]
    f_bins, _df = mls297.kernel_frequency_bins()
    with pytest.raises(ValueError, match="kernels must have shape"):
        store.write_filter_chunk(1, manifest, keys, _mini_cfg(), 1e-9, 1e-12,
                                 mls297.KERNEL_N_T, "b", np.zeros((2, 4, 7)),
                                 f_bins, 1.0)

    store.write_filter_chunk(1, manifest, keys, _mini_cfg(), 1e-9, 1e-12,
                             mls297.KERNEL_N_T, "b1",
                             np.zeros((2, 4, f_bins.size)), f_bins, 1.0)
    store.write_filter_chunk(2, manifest, keys, _mini_cfg(), 1e-9, 1e-12,
                             mls297.KERNEL_N_T, "b2",
                             np.zeros((2, 4, f_bins.size - 1)), f_bins[:-1], 1.0)
    with pytest.raises(RuntimeError, match="different frequency grid"):
        store.load_filter_records(manifest)


def test_filter_subcommand_writes_a_resumable_series(tmp_path):
    """One panel, level 4: the filter series appears and resumes."""
    out = str(tmp_path / "store")
    argv = ["filter", "--output", out, "--level", "4", "--panels", "1,0",
            "--workers", "2", "--batch-size", "4"]
    mls297.main(argv)
    store = mls297.Store(out)
    manifest = store.load_manifest()
    rows = store.load_filter_records(manifest)
    assert len(rows) == 16 and all(r["status"] == "ok" for r in rows)
    assert rows[0]["kernel"].shape == (4, mls297.kernel_frequency_bins()[0].size)
    assert all(np.all(np.isfinite(r["kernel"])) and r["kernel"].min() >= 0.0
               for r in rows)
    # additive only: the coherent and scatter series stay untouched
    assert store.load_records(manifest) == []
    assert store.load_scatter_records(manifest) == []

    mls297.main(argv)                                  # resume: nothing new
    assert len(store.load_filter_records(manifest)) == 16


# ── power<->Rabi table (ARC once, then cached) ───────────────────────────────


def test_power_table_matches_arc_and_scales_as_one_over_rabi_squared():
    """The cached per-n table is the ARC target-leg Rabi at 1 W, and the lookup
    inverts Omega ~ sqrt(P/A)."""
    from ryd_gate.physics import rb87_297_clock_rabi_frequencies

    rows = mls297.power_table_rows(mls297.ScanConfig())
    i = list(rows["ryd_n"]).index(53)
    omega, _garb = rb87_297_clock_rabi_frequencies(
        1.0, mls297.POWER_BEAM_AREA_UM2, ryd_level=53)
    assert rows["omega_mhz_at_1w"][i] == pytest.approx(
        omega / (2 * np.pi * 1e6), rel=1e-9)
    # 18 MHz needs (18 / omega_at_1W)**2 watts at the atoms
    assert mls297.power_at_atoms_w(rows, 53, 18.0) == pytest.approx(
        (18.0 / rows["omega_mhz_at_1w"][i]) ** 2, rel=1e-9)

    # POWER_OPTICS_LOSS is a LOSS fraction, not a transmission (the convention
    # scripts/max_leakage_ode_sweep.py already uses), so nominal power is
    # at-atoms / (1 - loss).  Only the rendered cell applies that factor, and the
    # figures are read off it, so pin it against the independent record in
    # docs/superpowers/specs/2026-07-24-max-leakage-297-sweep-design.md: 1-3 W
    # nominal at this 0.8-loss / 420 um^2 optics is ~9.6-16.6 MHz on 53P.
    assert rows["omega_mhz_at_1w"][i] * np.sqrt(
        1.0 - mls297.POWER_OPTICS_LOSS) == pytest.approx(9.6, abs=0.1)
    cells = mls297._power_table(mls297.ScanConfig(), "caption")[2]
    at_atoms, nominal = (float(x) for x in cells[i][-1].split(" / "))
    assert (at_atoms, nominal) == pytest.approx((0.70, 3.52), abs=5e-3)


def test_power_table_cache_is_keyed_on_the_n_axis(tmp_path, monkeypatch):
    """The npz cache is reused only for the n axis it was written for.

    Nothing in the file records the axis but the ``ryd_n`` array itself, so a
    ScanConfig with a different ``ryd_n`` must re-enter ARC rather than silently
    return a table whose rows do not line up with the store's panels.
    """
    import ryd_gate.physics as physics

    monkeypatch.setattr(mls297, "_POWER_CACHE", str(tmp_path / "omega.npz"))
    calls = []

    def fake_rabi(power_w, area_um2, *, ryd_level):
        calls.append((power_w, area_um2, ryd_level))
        return float(ryd_level) * mls297.TAU * 1e6, 0.0

    monkeypatch.setattr(physics, "rb87_297_clock_rabi_frequencies", fake_rabi)

    rows = mls297.power_table_rows(mls297.ScanConfig(ryd_n=(50, 53)))
    assert rows["omega_mhz_at_1w"] == pytest.approx([50.0, 53.0])
    assert calls == [(1.0, mls297.POWER_BEAM_AREA_UM2, 50),
                     (1.0, mls297.POWER_BEAM_AREA_UM2, 53)]

    mls297.power_table_rows(mls297.ScanConfig(ryd_n=(50, 53)))     # cache hit
    assert len(calls) == 2

    rows = mls297.power_table_rows(mls297.ScanConfig(ryd_n=(60,)))  # axis changed
    assert len(calls) == 3 and list(rows["ryd_n"]) == [60]


# ── phase-noise plot metrics (synthetic filter store; no solver) ─────────────
#
# The kernels are synthetic, but the PSD, the reweighting and the plot plumbing are
# the production ones: what is under test is that a stored kernel plus one measured
# spectrum becomes an eps_phase map, that eps_phase composes with the coherent and
# scattering budgets per logical input, and that a phase-noise render can never
# land on top of a noise-free figure.

_PSD_PATH = os.path.join("results", "297_laser_noise", "psd_ECDL.csv")


def _phase_mini_store(tmp_path, boost=()):
    """Mini store with coherent + scatter + filter records over two full panels.

    Kernels are scaled so a typical point lands at eps_phase ~ 1e-3; ``boost``
    names key indices whose kernel is multiplied by 500 to push them past the 0.1
    perturbative ceiling.  ``|00>`` is dark and carries no 297 leg, so its kernel is
    exactly zero — the real store's kernels have that property too.
    """
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    store, manifest = _mini_store(tmp_path)
    psd = PhaseNoisePSD.from_csv(_PSD_PATH, harmonic=4, extrapolation="flat")
    f_bins, _df = mls297.kernel_frequency_bins()
    scale = 1e-3 / error_from_kernel(psd, f_bins, np.ones(f_bins.size))
    rng = np.random.default_rng(7)
    gammas = {ni: {"p_ryd": 6.6e3, "p_r_garb": 6.6e3} for ni in (0, 3)}
    for seq, panel in enumerate([(0, 0), (3, 4)], start=1):
        keys = mls297.panel_keys(panel[0], panel[1], 1)          # full 7x7 grid
        res = _fake_result(len(keys), seed=seq)
        res.leakage = rng.uniform(1e-5, 1e-3, size=(len(keys), 4))
        res.max_leakage = res.leakage.max(axis=1)
        store.write_result_chunk(seq, manifest, keys, _mini_cfg(), "production",
                                 1e-9, 1e-12, f"b{seq}", res, 60.0)
        scatter = {ch: rng.uniform(1e-6, 1e-4, size=(len(keys), 4))
                   for ch in mls297.SCATTER_CHANNELS}
        store.write_scatter_chunk(seq, manifest, keys, _mini_cfg(), gammas, 1e-9,
                                  1e-12, f"s{seq}", scatter, res.max_leakage, 60.0)
        kernels = scale * rng.uniform(0.5, 1.5, size=(len(keys), 4, f_bins.size))
        kernels[:, 0, :] = 0.0
        if seq == 1:
            for i in boost:
                kernels[i] *= 500.0
        store.write_filter_chunk(seq, manifest, keys, _mini_cfg(), 1e-9, 1e-12,
                                 mls297.KERNEL_N_T, f"f{seq}", kernels, f_bins, 60.0)
    return store, manifest


def _tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_phase_noise_values_reweight_the_stored_kernels_per_laser_model(tmp_path):
    """eps_phase per logical input is the PSD-weighted stored kernel, and the
    extrapolation above the 1 MHz measurement edge is a real bracket."""
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    store, manifest = _phase_mini_store(tmp_path)
    rows = {r["key"]: r for r in store.load_filter_records(manifest)}
    flat = mls297.phase_noise_values(store, manifest, "ECDL", "flat")
    assert set(flat) == set(rows)

    psd = PhaseNoisePSD.from_csv(_PSD_PATH, harmonic=4, extrapolation="flat")
    key = next(iter(rows))
    r = rows[key]
    assert flat[key] == pytest.approx(
        [error_from_kernel(psd, r["f_bins"], r["kernel"][s]) for s in range(4)])
    assert flat[key][0] == 0.0                     # |00> is dark: exactly zero

    # ECDL's ASD falls as f^-0.46, so continuing that power law above 1 MHz is
    # strictly optimistic against holding the edge value flat.  The gate is most
    # sensitive near Omega/2pi = 9-18 MHz, all of it extrapolated.
    power = mls297.phase_noise_values(store, manifest, "ECDL", "power")
    assert np.all(power[key][1:] < flat[key][1:])
    assert power[key][0] == 0.0


def test_phase_noise_f_min_recuts_the_integral_by_dropping_whole_stored_bins(tmp_path):
    """Raising f_min drops stored bins from the sum and does nothing else.

    ``S_dnu`` rises as ``f**-2.5`` while the gate's response to a static detuning is
    finite, so the error integral is infrared divergent and ``f_min`` is a modelling
    parameter (physically the inverse relock timescale) the campaign has to report a
    sensitivity for.  The kernel is stored bin-integrated, so this must be a pure
    reweighting: the dropped band has to account for the whole difference.
    """
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    store, manifest = _phase_mini_store(tmp_path)
    rows = {r["key"]: r for r in store.load_filter_records(manifest)}
    base = mls297.phase_noise_values(store, manifest, "ECDL", "flat")
    cut = mls297.phase_noise_values(store, manifest, "ECDL", "flat", 10.0)

    key = next(iter(rows))
    r = rows[key]
    assert np.all(cut[key][1:] < base[key][1:])      # there IS weight below 10 Hz
    assert cut[key][0] == base[key][0] == 0.0        # |00> is dark either way

    # The difference is exactly the dropped band: nothing double-counted, and no df
    # re-applied on top of the already-integrated bins.
    psd = PhaseNoisePSD.from_csv(_PSD_PATH, harmonic=4, extrapolation="flat")
    below = r["f_bins"] < 10.0
    assert base[key] - cut[key] == pytest.approx(
        [error_from_kernel(psd, r["f_bins"][below], r["kernel"][s][below])
         for s in range(4)])

    # Bins are dropped whole, on their centre.  The lowest stored centre is above
    # KERNEL_F_MIN_HZ, so the default keeps every bin and a cutoff below it is a
    # no-op — the new parameter cannot perturb the headline figures.
    assert r["f_bins"][0] > mls297.KERNEL_F_MIN_HZ
    assert mls297.phase_noise_values(
        store, manifest, "ECDL", "flat", 0.5)[key] == pytest.approx(base[key])


def test_phase_noise_values_keep_the_tightest_rtol_record_per_key(tmp_path):
    """A key present in two filter chunks resolves to the tighter-rtol kernel.

    A ``filter`` pass resumed or retried at a different ``--rtol`` is exactly how one
    key ends up with two records, and the campaign is where that happens.  The
    scatter path already selects on rtol; file order must not decide which tolerance
    the deliverable quotes, so both orders are checked.
    """
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    f_bins, _df = mls297.kernel_frequency_bins()
    loose = np.full((2, 4, f_bins.size), 1e-3)
    tight = 2.0 * loose
    psd = PhaseNoisePSD.from_csv(_PSD_PATH, harmonic=4, extrapolation="flat")
    expected = np.full(4, 2.0 * error_from_kernel(psd, f_bins, loose[0][0]))

    for tight_first in (False, True):
        store, manifest = _mini_store(tmp_path / f"tight_first_{tight_first}")
        keys = mls297.panel_keys(0, 0, 0)[:2]
        chunks = [(1e-11, tight), (1e-9, loose)]
        if not tight_first:
            chunks.reverse()
        for seq, (rtol, kernels) in enumerate(chunks, start=1):
            store.write_filter_chunk(seq, manifest, keys, _mini_cfg(), rtol, 1e-12,
                                     mls297.KERNEL_N_T, f"b{seq}", kernels,
                                     f_bins, 30.0)
        values = mls297.phase_noise_values(store, manifest, "ECDL", "flat")
        assert set(values) == set(keys)
        for v in values.values():
            assert v == pytest.approx(expected)


def test_eps_phase_and_total_error_phase_compose_per_logical_input(tmp_path):
    """eps_phase is the worst-input loss; total_error_phase adds it to the
    coherent leakage and the scattering channels BEFORE the worst input is taken.

    Summing after the maximum would pair three unrelated maxima and over-count, so
    the check uses a node whose worst total input is not its worst leakage input.
    """
    store, manifest = _phase_mini_store(tmp_path)
    records = store.load_records(manifest, include_states=False)
    extra = mls297.phase_noise_values(store, manifest, "ECDL", "flat")
    scatter = {r["key"]: sum(r[ch] for ch in mls297.SCATTER_CHANNELS)
               for r in store.load_scatter_records(manifest)}
    leakage = {k: r.leakage for k, r in mls297.best_records(records).items()}

    values, vmin, vmax, label = mls297.sweeplib.plot_metric_values(
        store, manifest, records, "eps_phase",
        scatter_channels=mls297.SCATTER_CHANNELS, extra_values=extra)
    assert values == pytest.approx({k: float(v.max()) for k, v in extra.items()})
    assert 0.0 < vmin <= vmax and "eps_phase" in label

    totals, *_ = mls297.sweeplib.plot_metric_values(
        store, manifest, records, "total_error_phase",
        scatter_channels=mls297.SCATTER_CHANNELS, extra_values=extra)
    per_input = {k: leakage[k] + scatter[k] + extra[k] for k in extra}
    assert totals == pytest.approx({k: float(v.max()) for k, v in per_input.items()})

    discriminating = [k for k in extra
                      if np.argmax(per_input[k]) != np.argmax(leakage[k])]
    assert discriminating, "fixture no longer discriminates sum-then-max"
    k = discriminating[0]
    max_then_sum = (leakage[k].max() + scatter[k].max() + extra[k].max())
    assert totals[k] < max_then_sum


def test_phase_noise_plot_emits_the_map_and_the_power_table(tmp_path, capsys,
                                                            monkeypatch):
    """A synthetic mini-store renders eps_phase with a table strip and a suffix.

    Every element of the noise model — laser, extrapolation and a non-default f_min —
    reaches BOTH the filename and the suptitle, so no two models of the same metric
    can collide on disk and no detached page is ambiguous about which it shows;
    plots/ is hashed before and after to prove nothing is ever overwritten.
    """
    import matplotlib.figure

    suptitles: list[str] = []
    real_suptitle = matplotlib.figure.Figure.suptitle

    def _record_suptitle(self, text, **kwargs):
        suptitles.append(text)
        return real_suptitle(self, text, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", _record_suptitle)

    store, manifest = _phase_mini_store(tmp_path, boost=(0, 5))
    plots = Path(store.plots_dir)
    mls297.cmd_plot(Namespace(output=store.root, dpi=60, veil=True,
                              metric="max_leakage"))
    before = _tree_digest(plots)
    assert before and "max_leakage_8x9.png" in before
    assert "laser-phase-noise model" not in suptitles[0]   # model-free render
    suptitles.clear()

    renders = {("eps_phase", "flat", 1.0): "eps_phase_8x9_ECDL_flat",
               ("eps_phase", "power", 1.0): "eps_phase_8x9_ECDL_power",
               ("total_error_phase", "flat", 1.0):
                   "total_error_phase_8x9_ECDL_flat",
               ("eps_phase", "flat", 10.0): "eps_phase_8x9_ECDL_flat_fmin10Hz"}
    for (metric, extrap, f_min), name in renders.items():
        mls297.cmd_plot(Namespace(output=store.root, dpi=60, veil=True,
                                  metric=metric, laser="ECDL",
                                  extrapolation=extrap, f_min=f_min))
        stem = plots / "phase_noise" / "ECDL" / name
        assert stem.with_suffix(".png").exists()
        assert stem.with_suffix(".pdf").exists()
        assert suptitles[-1].endswith(
            f"\nlaser-phase-noise model: ECDL PSD, '{extrap}' extrapolation above "
            f"the 1 MHz measurement edge, f_min = {f_min:g} Hz")
    after = _tree_digest(plots)
    assert {k: after[k] for k in before} == before      # nothing overwritten
    assert len(after) == len(before) + 2 * len(renders)

    # The boosted nodes are out of the perturbative regime and must be named, not
    # silently plotted (measured sigma_nu/Omega = 0.053; predictions above ~0.1 are
    # outside first-order perturbation theory).
    out = capsys.readouterr().out
    assert "out of the perturbative regime" in out
    assert "eps_phase > 0.1" in out and "n0_t0_" in out


# ── Monte Carlo cross-check of the filter kernels (scripts/phase_noise_mc_check.py)


def test_noisy_rhs_adds_two_pi_dnu_times_the_rydberg_number():
    """The Monte Carlo leg's only new physics is H -> H_0 + 2 pi dnu(t) N_r.

    Both the sign and the 2 pi are conventions that would survive an ensemble mean
    (which is even in dnu) without a trace, so pin them against the RHS the sweep
    already uses rather than against a solved trajectory.
    """
    ops = _toy_sym_ops()
    n_r = mls297._rydberg_number_diag(16)
    cols = {"omega_297": _TOY_OM, "d_sweep": _TOY_DW,
            "shift": np.array([ops.h_static_diag[1]])}
    dnu = 3.7e5

    class _ConstOffset:
        """A trace whose phase ramps at exactly 2 pi dnu, i.e. dnu(t) = dnu."""

        def derivative(self, t):
            return 2 * np.pi * dnu

    base = mls297._297_rhs_factory(ops, cols, _TOY_T, 0.15)
    noisy = mc_check._noisy_rhs_factory(_ConstOffset(), n_r)(ops, cols, _TOY_T, 0.15)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    for t in (0.0, 0.31 * _TOY_T, _TOY_T):
        added = noisy(t, y) - base(t, y)
        assert np.allclose(added, -1j * (2 * np.pi * dnu) * n_r * y, rtol=1e-12)


def test_adjoint_integral_is_the_gates_first_order_response_to_a_detuning():
    """``int_0^T A(t) dt`` is d(psi)/d(delta) for ``H -> H_0 + 2 pi delta N_r``.

    The kernel's entire content is ``G(f) = int A(t) exp(-2 pi i f t) dt``, and at
    ``f = 0`` that is the gate's exact linear response to a static frequency offset
    -- which a central difference of two solved trajectories measures directly, with
    no statistics.  This is the deterministic half of the Monte Carlo check: it pins
    ``G`` itself, separately from the error formula built on it -- and it is what
    established, when the first version of that formula failed at 13 of 20 points,
    that the defect was the metric and not the kernel.
    """
    import sweeplib

    cfg = mls297.ScanConfig()
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    t_gate, omega, dsw = 1e-6, 2 * np.pi * 13.5e6, 2 * np.pi * 15e6

    out = mls297.integrate_adjoint_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-11, atol=1e-14, ramp=cfg.ramp_frac, n_t=mls297.KERNEL_N_T)
    weights = np.gradient(out["times"])
    weights[[0, -1]] *= 0.5
    g0 = np.einsum("t,stq->sq", weights, out["components"][0])     # (4, 16)

    class _Static:
        """A trace whose dnu(t) is the constant ``delta``."""

        def __init__(self, delta):
            self.delta = 2 * np.pi * delta

        def derivative(self, t):
            return self.delta

    delta = 1.0

    def terminal(d):
        return sweeplib.integrate_batch(
            ops, t_gate, {"omega_297": np.array([omega]), "d_sweep": np.array([dsw])},
            mls297.LOGICAL_INPUTS,
            rhs_factory=mc_check._noisy_rhs_factory(
                _Static(d), mls297._rydberg_number_diag(16)),
            dim=16, rtol=1e-11, atol=1e-14, ramp=cfg.ramp_frac).psi_final[0]

    measured = (terminal(delta) - terminal(-delta)) / (2 * delta)   # (4, 16)
    assert np.linalg.norm(g0[1]) > 0.0                             # not trivially zero
    assert np.max(np.abs(measured + 2j * np.pi * g0)) < 1e-4 * np.max(np.abs(g0))


def test_monte_carlo_check_point_runs_end_to_end_on_one_point():
    """Pipeline smoke test for ``check_point``: n=53, T=1 us, 60 shots.

    Deliberately *not* an agreement test.  60 shots put four sigma at tens of percent
    of the prediction, so an assertion on ``passed`` would hold whether the error
    formula were right or wrong — which is exactly how the first version of that
    formula survived a test of this shape.  The agreement claim lives in the 20-point
    campaign (``reports/phase_noise_mc.json``, 200 shots); what this pins is that the
    two legs run, agree on shape, and produce finite non-negative numbers.

    ``|00>`` is the one exact statement available here: ``|0>`` carries no 297 leg,
    so ``N_r psi_00 = 0``, ``A(t)`` vanishes and the kernel is identically zero
    (the Monte Carlo leg only to solver tolerance, both runs stepping differently).
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    cfg = mls297.ScanConfig()
    key = mls297.make_key(*mc_check.MC_POINTS[0])
    assert (cfg.ryd_n[key.n_idx], cfg.t_gate_us[key.t_idx]) == (53, 1.0)
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    psd = mc_check.lnp.load_psds(
        os.path.join(mc_check.lnp.NOISE_DIR, "psd_ECDL.csv"))["flat"]

    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("fork")) as pool:
        rec = mc_check.check_point(ops, cfg, key, psd, shots=60, seed=1,
                                   rtol=1e-9, atol=1e-12, pool=pool)

    assert rec["n_shots"] == 60 and rec["point_id"] == key.id()
    assert rec["filter_prediction"][0] == 0.0 and abs(rec["mc_mean"][0]) < 1e-9
    for i in (1, 2, 3):
        assert rec["filter_prediction"][i] > 0.0
        assert np.isfinite(rec["mc_mean"][i]) and rec["mc_mean"][i] > 0.0
        assert np.isfinite(rec["mc_stderr"][i]) and rec["mc_stderr"][i] > 0.0
    assert isinstance(rec["passed"], bool)


# ── real rb87_297_clock_4 model (ARC required) ───────────────────────────────


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
