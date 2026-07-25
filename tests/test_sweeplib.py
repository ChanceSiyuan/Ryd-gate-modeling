"""Tests for the shared sweeplib package (axes + solver machinery).

Covers the nested-grid axes and the ``make_pointkey_type`` factory — parameterized
over two key configurations (the ode ``delta_idx``/``d`` layout and a synthetic
``n_idx``/``n`` layout) to prove the factory generalizes — and the solver machinery
(quintic envelope, block-max DOP853 error norm, and the injected-RHS batched
integration kernel) on a small synthetic swap-symmetric model.  No script or ARC
imports: only sweeplib itself is exercised here.
"""

import dataclasses
import importlib.util
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = str(ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import sweeplib  # noqa: E402


# ── point-key factory (parameterized over two key configurations) ────────────

_ODE_OME = (Fraction(200), Fraction(1400, 3), Fraction(2200, 3), Fraction(1000))
_ODE_DSW = (Fraction(2), Fraction(10), Fraction(20), Fraction(30))
_SYN_OME = (Fraction(1), Fraction(2), Fraction(4), Fraction(8))
_SYN_DSW = (Fraction(3), Fraction(5), Fraction(7), Fraction(11))

_KEYCFGS = {
    "ode": dict(panel_field="delta_idx", id_prefix="d",
                omega_anchors=_ODE_OME, dsweep_anchors=_ODE_DSW,
                panel_len=8, n_t=9),
    "synthetic": dict(panel_field="n_idx", id_prefix="n",
                      omega_anchors=_SYN_OME, dsweep_anchors=_SYN_DSW,
                      panel_len=3, n_t=4),
}


@pytest.fixture(params=list(_KEYCFGS))
def keycfg(request):
    cfg = _KEYCFGS[request.param]
    PointKey, make_key, panel_keys, all_panels, all_keys = sweeplib.make_pointkey_type(**cfg)
    return SimpleNamespace(
        name=request.param, PointKey=PointKey, make_key=make_key,
        panel_keys=panel_keys, all_panels=all_panels, all_keys=all_keys, **cfg)


def test_factory_first_field_name_and_id_prefix(keycfg):
    """The dataclass's first field is NAMED for the script (serialized in the store)
    and .id() prefixes the panel index with the script's id prefix."""
    assert dataclasses.fields(keycfg.PointKey)[0].name == keycfg.panel_field
    k = keycfg.make_key(3, 4, (3, 2), (3, 2))
    assert k.id() == f"{keycfg.id_prefix}3_t4_om3-2_dw3-2"
    assert getattr(k, keycfg.panel_field) == 3
    assert k.panel == (3, 4)


def test_axis_sizes_and_cumulative_totals(keycfg):
    for level, size in enumerate(sweeplib.LEVEL_SIZES):
        assert len(sweeplib.axis_coords(level)) == size
        assert len(keycfg.all_keys(level)) == keycfg.panel_len * keycfg.n_t * size * size
    assert len(keycfg.all_panels()) == keycfg.panel_len * keycfg.n_t


def test_finer_levels_nest():
    """Midpoint insertion reuses every previous exact node (config-independent)."""
    for level in range(3):
        coarse = set(sweeplib.axis_coords(level))
        fine = set(sweeplib.axis_coords(level + 1))
        assert coarse < fine


def test_canonical_coordinates_reduce_to_identical_keys(keycfg):
    # level-1 coordinate k=2/den=2 is the level-0 node k=1/den=1
    assert sweeplib.canon_coord(2, 2) == (1, 1)
    k_a = keycfg.make_key(0, 0, (2, 2), (4, 4))
    k_b = keycfg.make_key(0, 0, (1, 1), (1, 1))
    assert k_a == k_b and k_a.id() == k_b.id()
    with pytest.raises(ValueError):
        sweeplib.canon_coord(9, 2)  # outside [0, 3]


def test_point_key_level_and_panel(keycfg):
    k = keycfg.make_key(2, 1, (3, 2), (3, 2))
    assert k.level() == 1
    assert k.panel == (2, 1)
    assert keycfg.make_key(0, 0, (1, 1), (0, 1)).level() == 0


def test_point_key_axis_values_use_the_configured_anchors(keycfg):
    k = keycfg.make_key(0, 0, (3, 2), (3, 2))
    assert k.omega_mhz() == sweeplib.coord_value_mhz(keycfg.omega_anchors, 3, 2)
    assert k.dsweep_mhz() == sweeplib.coord_value_mhz(keycfg.dsweep_anchors, 3, 2)


def test_keys_are_ordered_and_hashable(keycfg):
    a = keycfg.make_key(0, 0, (0, 1), (0, 1))
    b = keycfg.make_key(0, 0, (0, 1), (1, 1))
    assert a < b
    assert len({a, b, keycfg.make_key(0, 0, (0, 1), (0, 1))}) == 2


# ── analytic pulse envelope ──────────────────────────────────────────────────


def test_envelope_shape_integral_and_continuity():
    r = 0.15
    # shape + symmetry
    assert sweeplib.envelope(0.0, r) == 0.0
    assert sweeplib.envelope(0.075, r) == pytest.approx(0.5)   # q(1/2) = 0.5
    assert sweeplib.envelope(0.5, r) == 1.0
    s = np.linspace(0, 1, 101)
    assert np.allclose(sweeplib.envelope(s, r), sweeplib.envelope(1.0 - s, r))
    # integral endpoints + branch-seam continuity
    assert sweeplib.envelope_integral(0.0, r) == 0.0
    assert float(sweeplib.envelope_integral(1.0, r)) == pytest.approx(1.0 - r)
    for s0 in (r, 1.0 - r):  # branch seams
        eps = 1e-9
        lo = float(sweeplib.envelope_integral(s0 - eps, r))
        hi = float(sweeplib.envelope_integral(s0 + eps, r))
        assert abs(hi - lo) < 1e-8
    # J' == E by quadrature on a coarse grid
    s = np.linspace(0.0, 1.0, 20001)
    j_num = np.concatenate([[0.0], np.cumsum(
        0.5 * (sweeplib.envelope(s[1:], r) + sweeplib.envelope(s[:-1], r)) * np.diff(s))])
    assert np.max(np.abs(j_num - sweeplib.envelope_integral(s, r))) < 1e-8


# ── block-max DOP853 error norm ──────────────────────────────────────────────


def test_error_norm_matches_installed_scipy():
    assert sweeplib.verify_scipy_error_norm() < 1e-12


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
    cls = sweeplib.make_block_solver_class(4)
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


# ── kernel invariants on a synthetic swap-symmetric model + injected RHS ──────
#
# The kernel machinery (segmented restarts, per-column shifts, swap reconstruction,
# trajectory sampling) is model-agnostic; exercise it with a two-atom three-level
# toy analogue of PanelOperators and a representative two-drive+Stark rhs_factory.

TAU = 2.0 * np.pi


def _toy_ops(v_pair=TAU * 5e6, w_hf=TAU * 40e6, w_e=TAU * 300e6):
    """levels {0,1,e}, logical = {0,1}, 420 leg drives 1->e (+ a weak 0->e), 1013
    leg drives 0->e.  Swap-symmetric by construction (identical site embeddings)."""
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

    idx = np.arange(d * d)
    a, b = np.divmod(idx, d)
    swap_perm = b * d + a

    return SimpleNamespace(
        delta_rad_s=TAU * 1e9,
        h_static_diag=diag,
        x420=b420 + b420.conj().T, y420=1j * (b420 - b420.conj().T),
        x1013=b1013 + b1013.conj().T, y1013=1j * (b1013 - b1013.conj().T),
        amplitude_scale=1.0,
        logical_indices=np.array([0, 1, 3, 4]),
        swap_perm=swap_perm,
        swap_symmetric=True,
    )


def _toy_rhs_factory(omega_1013):
    """Representative two-drive (420/1013) + Stark-chirp RHS factory for the kernel."""

    def rhs_factory(ops, cols, t_gate, ramp):
        om_cols = cols["omega_420"]
        dsw_cols = cols["d_sweep"]
        d1 = -(4.0 / 3.0) * om_cols ** 2 / (4.0 * ops.delta_rad_s)
        dr = -(omega_1013 ** 2) / (4.0 * ops.delta_rad_s)
        drmd1_cols = dr - d1
        diag_row = ops.h_static_diag[None, :] - cols["shift"][:, None]
        x420_t = np.ascontiguousarray(ops.x420.T)
        y420_t = np.ascontiguousarray(ops.y420.T)
        x1013_t = np.ascontiguousarray(ops.x1013.T)
        ascale = ops.amplitude_scale
        sin_coef = -t_gate / TAU
        n_cols, dim = diag_row.shape

        def rhs(t, y):
            s = t / t_gate
            amp = math.sqrt(float(sweeplib.envelope(s, ramp)))
            phi = (sin_coef * math.sin(TAU * s)) * dsw_cols \
                + (t_gate * float(sweeplib.envelope_integral(s, ramp))) * drmd1_cols
            c420 = (ascale * amp) * om_cols * np.exp(-1j * phi)
            g1013 = ascale * omega_1013 * amp
            ym = y.reshape(n_cols, dim)
            out = diag_row * ym
            out += c420.real[:, None] * (ym @ x420_t)
            out += c420.imag[:, None] * (ym @ y420_t)
            out += g1013 * (ym @ x1013_t)
            return (-1j * out).ravel()

        return rhs

    return rhs_factory


_TOY_T = 0.4e-6
_TOY_OM = np.array([TAU * 30e6])
_TOY_DW = np.array([TAU * 10e6])
_TOY_OM1013 = TAU * 25e6


def _toy_solve(om=None, dw=None, use_swap=True, use_shifts=True, segmented=True,
               t_eval=None, rtol=1e-10, atol=1e-13):
    ops = _toy_ops()
    om = _TOY_OM if om is None else om
    dw = _TOY_DW if dw is None else dw
    state_labels = ("00", "01", "11") if use_swap else sweeplib.LOGICAL_INPUTS
    return sweeplib.integrate_batch(
        ops, _TOY_T,
        {"omega_420": np.asarray(om, float), "d_sweep": np.asarray(dw, float)},
        state_labels, rhs_factory=_toy_rhs_factory(_TOY_OM1013),
        dim=ops.h_static_diag.size, rtol=rtol, atol=atol,
        use_shifts=use_shifts, segmented=segmented, t_eval=t_eval)


def test_kernel_norms_and_direct_leakage():
    res = _toy_solve()
    assert res.norm_err.max() < 1e-10
    # direct Q population equals 1 - P population to machine accuracy here
    pops = np.abs(res.psi_final) ** 2
    p_log = pops[:, :, [0, 1, 3, 4]].sum(axis=2)
    assert np.allclose(res.leakage, 1.0 - p_log, atol=1e-12)
    assert res.max_leakage[0] == res.leakage[0].max()
    assert res.worst_input[0] == sweeplib.LOGICAL_INPUTS[int(np.argmax(res.leakage[0]))]
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
