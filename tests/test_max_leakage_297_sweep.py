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
