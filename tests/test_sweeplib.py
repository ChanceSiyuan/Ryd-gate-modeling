"""Tests for the shared sweeplib package (axes + solver machinery).

Covers the nested-grid axes and the ``make_pointkey_type`` factory — parameterized
over two key configurations (the ode ``delta_idx``/``d`` layout and a synthetic
``n_idx``/``n`` layout) to prove the factory generalizes — and the solver machinery
(quintic envelope, block-max DOP853 error norm, and the injected-RHS batched
integration kernel) on a small synthetic swap-symmetric model.  No script or ARC
imports: only sweeplib itself is exercised here.
"""

import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
from argparse import Namespace
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
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


# ── Store / Runner machinery (parameterized over two script configs) ─────────
#
# The Store/Runner/CostModel/worker machinery is config-agnostic; exercise it
# under two "script configs" — an ode-shaped one (delta_idx / d, three scatter
# channels, dim 49, with the fixed-1013 manifest block + resume guard) and a
# synthetic one (n_idx / n, two channels, dim 16, no extra manifest block) — to
# prove the parameterization generalizes.  No script or ARC imports.

_OM1013 = 2 * np.pi * 489.6e6


@dataclass(frozen=True)
class _FakeScanConfig:
    """Minimal ScanConfig stand-in: only the fields Store/Runner/CostModel read."""

    tag: str
    t_gate_us: tuple
    panel_axis_name: str
    panel_axis: tuple
    ramp_frac: float = 0.15
    n_eval_trajectory: int = 301
    rtol_production: float = 1e-9
    atol_production: float = 1e-12
    rtol_audit: float = 1e-10
    atol_audit: float = 1e-13
    interp_space: str = "log10"
    credibility_floor_min: float = 1e-12

    def __post_init__(self):
        # expose the panel axis under its serialized name (delta_e_ghz / ryd_n)
        object.__setattr__(self, self.panel_axis_name, self.panel_axis)

    def physics_payload(self):
        return {"schema_version": 1, "tag": self.tag,
                "t_gate_us": list(self.t_gate_us),
                self.panel_axis_name: list(self.panel_axis)}

    def physics_hash(self):
        return hashlib.sha256(
            json.dumps(self.physics_payload(), sort_keys=True).encode()).hexdigest()


def _make_descriptor(panel_axis_name, panel_field, omega_col):
    def descriptor(cfg, keys):
        axis = getattr(cfg, panel_axis_name)
        return {
            panel_axis_name: np.asarray([axis[getattr(k, panel_field)] for k in keys]),
            "t_gate_us": np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
            omega_col: np.asarray([float(k.omega_mhz()) for k in keys]),
            "dsweep_mhz": np.asarray([float(k.dsweep_mhz()) for k in keys]),
        }
    return descriptor


def _make_result_extra(panel_axis_name, panel_field, omega_col, has_omega_1013):
    def result_extra(cfg, keys, manifest):
        axis = getattr(cfg, panel_axis_name)
        tg_us = np.asarray([cfg.t_gate_us[k.t_idx] for k in keys])
        om_mhz = np.asarray([float(k.omega_mhz()) for k in keys])
        dw_mhz = np.asarray([float(k.dsweep_mhz()) for k in keys])
        out = {
            "t_gate_s": tg_us * 1e-6,
            omega_col.replace("_mhz", "_rad_s"): om_mhz * 1e6 * 2 * np.pi,
            "dsweep_rad_s": dw_mhz * 1e6 * 2 * np.pi,
        }
        if has_omega_1013:
            axis_rad = np.asarray([axis[getattr(k, panel_field)] for k in keys]) * 1e9 * 2 * np.pi
            out = {panel_axis_name.replace("_ghz", "_rad_s"): axis_rad, **out,
                   "omega_1013_rad_s": np.full(len(keys), float(manifest["omega_1013_rad_s"]))}
        return out
    return result_extra


def _sweep_bundle(name):
    if name == "ode":
        panel_field, id_prefix = "delta_idx", "d"
        omega_anchors = (Fraction(200), Fraction(1400, 3), Fraction(2200, 3), Fraction(1000))
        dsweep_anchors = (Fraction(2), Fraction(10), Fraction(20), Fraction(30))
        panel_axis_name, panel_axis = "delta_e_ghz", (9.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
        t_gate_us = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
        omega_col = "omega_420_mhz"
        scatter_channels = ("p_mid", "p_ryd", "p_r_garb")
        scatter_rates = {"p_mid": 9.0e6, "p_ryd": 6.6e3, "p_r_garb": 6.6e3}
        dim = 49
        has_omega_1013 = True
        pulse_hash = "ode-pulse-hash"
    else:
        panel_field, id_prefix = "n_idx", "n"
        omega_anchors = (Fraction(1), Fraction(2), Fraction(4), Fraction(8))
        dsweep_anchors = (Fraction(3), Fraction(5), Fraction(7), Fraction(11))
        panel_axis_name, panel_axis = "ryd_n", (60, 70, 80)
        t_gate_us = (1.0, 2.0, 3.0, 4.0)
        omega_col = "omega297_mhz"
        scatter_channels = ("p_ryd", "p_r_garb")
        scatter_rates = {"p_ryd": 6.6e3, "p_r_garb": 6.6e3}
        dim = 16
        has_omega_1013 = False
        pulse_hash = "syn-pulse-hash"

    PointKey, make_key, panel_keys, all_panels, all_keys = sweeplib.make_pointkey_type(
        panel_field=panel_field, id_prefix=id_prefix,
        omega_anchors=omega_anchors, dsweep_anchors=dsweep_anchors,
        panel_len=len(panel_axis), n_t=len(t_gate_us))
    key_fields = (panel_field, "t_idx", "om_num", "om_den", "dw_num", "dw_den")
    cfg = _FakeScanConfig(tag=name, t_gate_us=t_gate_us,
                          panel_axis_name=panel_axis_name, panel_axis=panel_axis)
    provenance = sweeplib.ProvenanceColumns(
        scatter_channels=scatter_channels, default_dim=dim,
        descriptor=_make_descriptor(panel_axis_name, panel_field, omega_col),
        result_extra=_make_result_extra(panel_axis_name, panel_field, omega_col, has_omega_1013),
        schema_version=1)

    def manifest_extras():
        axes = {panel_axis_name: list(panel_axis), "t_gate_us": list(t_gate_us),
                "level_sizes": list(sweeplib.LEVEL_SIZES)}
        extra_fields = None
        extra_guard = None
        if has_omega_1013:
            extra_fields = {"omega_1013_rad_s": _OM1013,
                            "omega_1013_over_2pi_MHz": _OM1013 / (2 * np.pi) / 1e6}

            def extra_guard(existing):
                rec = float(existing.get("omega_1013_rad_s", 0.0))
                if abs(rec - _OM1013) > 1e-6 * abs(_OM1013):
                    raise RuntimeError(
                        f"Omega_1013 mismatch: manifest {rec!r} vs current {_OM1013!r}")
        return dict(pulse_hash=pulse_hash, axes=axes,
                    extra_fields=extra_fields, extra_guard=extra_guard)

    guard_cases = [("physics_hash", "0" * 64, "physics_hash mismatch"),
                   ("model_hash", "0" * 64, "model_hash mismatch"),
                   ("pulse_hash", "0" * 64, "pulse_hash mismatch")]
    if has_omega_1013:
        guard_cases.append(("omega_1013_rad_s", 1.0, "Omega_1013 mismatch"))

    return SimpleNamespace(
        name=name, PointKey=PointKey, make_key=make_key, panel_keys=panel_keys,
        all_panels=all_panels, all_keys=all_keys, key_fields=key_fields, cfg=cfg,
        provenance=provenance, manifest_extras=manifest_extras, dim=dim,
        scatter_channels=scatter_channels, scatter_rates=scatter_rates,
        guard_cases=guard_cases)


@pytest.fixture(params=["ode", "synthetic"])
def bundle(request):
    return _sweep_bundle(request.param)


def mini_store(bundle, tmp_path, name="scan", model_hash="modelhash-1"):
    store = sweeplib.Store(str(Path(tmp_path) / name), key_type=bundle.PointKey,
                           key_fields=bundle.key_fields,
                           provenance_columns=bundle.provenance)
    store.ensure_dirs()
    manifest = store.init_or_validate_manifest(
        bundle.cfg, model_hash, "codehash", {}, **bundle.manifest_extras())
    return store, manifest


def _fake_result(bundle, n, seed=0):
    dim = bundle.dim
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal((n, 4, dim)) + 1j * rng.standard_normal((n, 4, dim))
    psi /= np.linalg.norm(psi, axis=2, keepdims=True)
    leak = rng.uniform(1e-6, 1e-2, size=(n, 4))
    return sweeplib.BatchResult(
        psi_final=psi, leakage=leak, max_leakage=leak.max(axis=1),
        worst_input=[sweeplib.LOGICAL_INPUTS[int(i)] for i in leak.argmax(axis=1)],
        return_prob=rng.uniform(0.9, 1.0, size=(n, 4)),
        norm_err=np.full((n, 4), 1e-13), nfev=1234, used_swap=True)


# ── Synchronous fake executor for the Runner dispatch loop ───────────────────
#
# The Runner event loop (submit -> Future -> wait -> result/split/requeue) is the
# high-risk uncovered code.  A real ProcessPoolExecutor would fork and run the
# solver; instead ``runner_factory`` overrides Runner._make_executor with this
# inline stand-in.  A test supplies ``responder(spec) -> (kind, payload)``:
#   "result"       -> the future resolves to ``payload`` (an out dict)
#   "crash"        -> the future raises ``payload`` (an in-flight pool death)
#   "submit_error" -> submit() itself raises ``payload`` (a broken pool)


class _FakeExecutor:
    def __init__(self, responder):
        self._responder = responder
        self._processes = {}                 # Runner.shutdown() introspects this

    def submit(self, fn, spec):
        kind, payload = self._responder(spec)
        if kind == "submit_error":
            raise payload
        fut = Future()
        if kind == "crash":
            fut.set_exception(payload)
        else:
            fut.set_result(payload)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _runner_args(**over):
    base = dict(workers=1, batch_size=1, budget_hours=1.0, reserve_hours=0.0,
                point_timeout=60.0)
    base.update(over)
    return Namespace(**base)


@pytest.fixture
def runner_factory(monkeypatch):
    """Build Runners whose executor is the synchronous fake above.

    Runner._install_signals permanently replaces the process SIGINT/SIGTERM
    handlers and shutdown() never restores them, so the fixture saves/restores
    them and shuts down every Runner it builds (releasing the store flock)."""
    saved_int = signal.getsignal(signal.SIGINT)
    saved_term = signal.getsignal(signal.SIGTERM)
    built = []

    def _build(store, manifest, cfg, args, responder=None, cost=None):
        if responder is None:
            def responder(spec):
                raise AssertionError("executor.submit called unexpectedly")
        monkeypatch.setattr(sweeplib.Runner, "_make_executor",
                            lambda self: _FakeExecutor(responder))
        runner = sweeplib.Runner(store, manifest, cfg, args,
                                 cost or sweeplib.CostModel(cfg))
        built.append(runner)
        return runner

    try:
        yield _build
    finally:
        for r in built:
            try:
                r.shutdown()
            except Exception:
                pass
        signal.signal(signal.SIGINT, saved_int)
        signal.signal(signal.SIGTERM, saved_term)


# ── persistence: manifest, chunks, resume, exports ───────────────────────────


def test_chunk_roundtrip_preserves_states_exactly(bundle, tmp_path):
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:3]
    result = _fake_result(bundle, 3)
    path = store.write_result_chunk(
        1, manifest, keys, bundle.cfg, "production", 1e-9, 1e-12,
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
        sweeplib._atomic_savez(target, bad=np.array([{"a": 1}], dtype=object))
    assert not any(tmp_path.iterdir())  # neither the file nor a temp survives


def test_stale_tmp_files_are_ignored_by_the_loader(bundle, tmp_path):
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, bundle.cfg, "production",
                             1e-9, 1e-12, "b", _fake_result(bundle, 1), 1.0)
    (Path(store.chunks_dir) / "chunk_000002.npz.tmp-dead").write_bytes(b"garbage")
    assert len(store.load_records(manifest)) == 1
    assert store.next_seq() == 2


def test_manifest_guard_rejects_mismatched_provenance(bundle, tmp_path):
    """init_or_validate_manifest refuses to resume when any recorded provenance
    field (the three hashes or, for the ode config, Omega_1013) disagrees with the
    live code/model."""
    for field, doctored, match in bundle.guard_cases:
        store, _ = mini_store(bundle, tmp_path, name=field)
        manifest_path = Path(store.manifest_path)
        doc = json.loads(manifest_path.read_text())
        doc[field] = doctored
        manifest_path.write_text(json.dumps(doc))
        with pytest.raises(RuntimeError, match=match):
            store.init_or_validate_manifest(
                bundle.cfg, "modelhash-1", "codehash", {}, **bundle.manifest_extras())


def test_chunk_guard_refuses_to_merge_foreign_provenance(bundle, tmp_path):
    """load_records refuses to merge a chunk whose stamped hash differs from the
    manifest.  (Omega_1013 is manifest-only, so it has no chunk arm.)"""
    for field in ("physics_hash", "model_hash", "pulse_hash"):
        store, manifest = mini_store(bundle, tmp_path, name=field)
        foreign = dict(manifest, **{field: "0" * 64})
        store.write_result_chunk(1, foreign, bundle.panel_keys(0, 0, 0)[:1],
                                 bundle.cfg, "production", 1e-9, 1e-12, "b",
                                 _fake_result(bundle, 1), 1.0)
        with pytest.raises(RuntimeError, match=field):
            store.load_records(manifest)


def test_resume_dedup_and_tier_preference(bundle, tmp_path):
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:2]
    store.write_result_chunk(1, manifest, keys, bundle.cfg, "production",
                             1e-9, 1e-12, "b1", _fake_result(bundle, 2, seed=1), 10.0)
    store.write_result_chunk(2, manifest, keys[:1], bundle.cfg, "audit",
                             1e-10, 1e-13, "b2", _fake_result(bundle, 1, seed=2), 12.0)
    records = store.load_records(manifest)
    best = sweeplib.best_records(records)
    assert best[keys[0]].tier == "audit"          # tightest record wins exports
    assert best[keys[1]].tier == "production"
    done = sweeplib.completed_keys(records)
    missing = [k for k in bundle.panel_keys(0, 0, 0) if k not in done]
    assert len(missing) == 16 - 2                 # resume schedules only the rest
    pairs = sweeplib.audit_pairs(records)
    assert len(pairs) == 1 and pairs[0][0] == keys[0]


def test_failed_points_recorded_and_not_counted_done(bundle, tmp_path):
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(
        1, manifest, keys, bundle.cfg, "production", 1e-9, 1e-12,
        "bf", None, 5.0, statuses=["timeout"], message="timeout after 5s",
        retry_count=2)
    records = store.load_records(manifest)
    assert records[0].status == "timeout"
    assert records[0].retry_count == 2
    assert sweeplib.completed_keys(records) == set()
    assert sweeplib.best_records(records) == {}


def test_best_records_prefers_tighter_rtol_over_tier(bundle, tmp_path):
    """A loosened ad-hoc audit must not displace a tighter production record."""
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, bundle.cfg, "production",
                             1e-9, 1e-12, "b1", _fake_result(bundle, 1, seed=1), 1.0)
    store.write_result_chunk(2, manifest, keys, bundle.cfg, "audit",
                             1e-8, 1e-11, "b2", _fake_result(bundle, 1, seed=2), 1.0)
    best = sweeplib.best_records(store.load_records(manifest))
    assert best[keys[0]].tier == "production" and best[keys[0]].rtol == 1e-9


def test_next_seq_spans_chunk_and_trajectory_series(bundle, tmp_path):
    """Chunk and trajectory files share one seq counter, so next_seq must be past
    both — otherwise a later trajectory write would os.replace an existing one."""
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:1]
    store.write_result_chunk(1, manifest, keys, bundle.cfg, "production",
                             1e-9, 1e-12, "b", _fake_result(bundle, 1), 1.0)
    store.write_trajectory_chunk(2, manifest, keys[0], "production",
                                 np.linspace(0.0, 1.0, 3),
                                 np.zeros((3, 4, bundle.dim), dtype=complex))
    assert store.next_seq() == 3


def test_corrupted_chunk_raises_on_load(bundle, tmp_path):
    """A garbage chunk_NNNNNN.npz is picked up by the loader and raises ValueError
    mid-merge (np.load's pickle fallback under allow_pickle=False)."""
    store, manifest = mini_store(bundle, tmp_path)
    store.write_result_chunk(1, manifest, bundle.panel_keys(0, 0, 0)[:1], bundle.cfg,
                             "production", 1e-9, 1e-12, "b", _fake_result(bundle, 1), 1.0)
    (Path(store.chunks_dir) / "chunk_000002.npz").write_bytes(b"garbage, not a real npz")
    with pytest.raises(ValueError, match="pickled"):
        store.load_records(manifest)


def test_cross_level_resume_reuses_coarse_nodes(bundle, tmp_path):
    """End-to-end nesting invariant: level-0 records mark the coarse nodes done,
    so the level-1 missing set is exactly the newly inserted (33) nodes."""
    store, manifest = mini_store(bundle, tmp_path)
    coarse = bundle.panel_keys(0, 0, 0)             # 16 level-0 nodes of panel (0,0)
    store.write_result_chunk(1, manifest, coarse, bundle.cfg, "production",
                             1e-9, 1e-12, "b", _fake_result(bundle, len(coarse)), 60.0)
    done = sweeplib.completed_keys(store.load_records(manifest))
    assert set(coarse) <= done
    fine = bundle.panel_keys(0, 0, 1)               # 49 level-1 nodes, same panel
    missing = [k for k in fine if k not in done]
    assert len(missing) == 49 - 16
    assert set(coarse).isdisjoint(missing)


# ── scatter series ───────────────────────────────────────────────────────────


def test_scatter_chunk_roundtrip_resume_and_isolation(bundle, tmp_path):
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:2]
    gammas = {0: bundle.scatter_rates}
    scatter = {ch: np.random.default_rng(0).uniform(1e-6, 1e-2, (2, 4))
               for ch in bundle.scatter_channels}
    path = store.write_scatter_chunk(
        1, manifest, keys, bundle.cfg, gammas, 1e-9, 1e-12, "sb1",
        scatter, np.array([1e-4, 2e-4]), runtime_s=10.0)
    assert "scatter" in path
    rows = store.load_scatter_records(manifest)
    assert len(rows) == 2 and rows[0]["status"] == "ok"
    ch0 = bundle.scatter_channels[0]
    np.testing.assert_array_equal(rows[1][ch0], scatter[ch0][1])
    # resume: done keys are skipped
    done = {r["key"] for r in rows if r["status"] == "ok"}
    assert set(keys) == done
    # isolation: no main-series chunk was created
    assert store.load_records(manifest) == []
    assert store.next_seq() == 1 and store.next_scatter_seq() == 2
    # foreign pulse hash refuses to merge
    foreign = dict(manifest, pulse_hash="0" * 64)
    store.write_scatter_chunk(2, foreign, keys, bundle.cfg, gammas, 1e-9,
                              1e-12, "sb2", scatter, np.array([1e-4, 2e-4]), 1.0)
    with pytest.raises(RuntimeError, match="pulse_hash"):
        store.load_scatter_records(manifest)


# ── scheduling helpers ───────────────────────────────────────────────────────


def test_group_batches_stays_within_panel_and_orders_axes(bundle):
    keys = bundle.panel_keys(0, 0, 0) + bundle.panel_keys(1, 2, 0)
    batches = sweeplib.group_batches(keys, batch_size=6)
    assert all(len({k.panel for k in b.keys}) == 1 for b in batches)
    assert sum(len(b.keys) for b in batches) == 32
    first = batches[0].keys
    fracs = [(Fraction(k.om_num, k.om_den), Fraction(k.dw_num, k.dw_den)) for k in first]
    assert fracs == sorted(fracs)
    with pytest.raises(ValueError):
        sweeplib.Batch(keys=keys)  # crosses panels


def test_batch_split_isolates_points_and_counts_retries(bundle):
    batch = sweeplib.Batch(keys=bundle.panel_keys(0, 0, 0)[:5],
                           priority_scores=[1, 2, 3, 4, 5])
    parts = sweeplib.Runner._split(None, batch)
    assert [len(p.keys) for p in parts] == [2, 3]
    assert all(p.retry_count == 1 for p in parts)
    assert parts[0].priority_scores == [1, 2]
    single = sweeplib.Batch(keys=batch.keys[:1])
    assert [len(p.keys) for p in sweeplib.Runner._split(None, single)] == [1]


def test_cost_model_scales_with_gate_time_and_inflates(bundle):
    cfg = bundle.cfg
    cost = sweeplib.CostModel(cfg)
    assert cost.predict_point((0, 0)) == pytest.approx(150.0 * cfg.t_gate_us[0])
    cost.observe((0, 0), 100.0)                 # first panel column
    assert cost.predict_point((0, 0)) == 100.0
    last = len(cfg.t_gate_us) - 1
    assert cost.predict_point((0, last)) == pytest.approx(
        100.0 * cfg.t_gate_us[last] / cfg.t_gate_us[0])   # same row, scaled
    assert cost.inflation_p90() == 1.5          # too few ratios -> conservative
    for r in (1.0, 1.1, 1.2, 1.0, 1.05, 1.3):
        cost.observe_batch(100.0, 100.0 * r)
    assert 1.0 <= cost.inflation_p90() <= 1.3
    eta = cost.eta_seconds(bundle.panel_keys(0, 0, 0), n_workers=4)
    assert eta == pytest.approx(16 * 100.0 * cost.inflation_p90() / 4)


# ── Runner event loop: dispatch, retries, signals (synchronous fake executor) ─


def test_budget_scheduler_defers_everything_past_the_deadline(bundle, tmp_path, runner_factory, capsys):
    """With the dispatch deadline already in the past, run_batches launches no
    work (the default responder would assert if submit ran), defers every point,
    and records nothing — never-computed points must not be marked failed.  The
    deferral is announced loudly: a first-deferral line and an end-of-stage
    WARNING naming the total (so a driver log can't silently exit 0)."""
    store, manifest = mini_store(bundle, tmp_path)
    runner = runner_factory(store, manifest, bundle.cfg,
                            _runner_args(workers=2, batch_size=4,
                                         budget_hours=0.0, reserve_hours=1.0))
    runner.run_batches(sweeplib.group_batches(bundle.panel_keys(0, 0, 0), batch_size=4),
                       "test-budget")
    assert runner.deferred == 16
    assert runner.completed_points == 0
    assert store.load_records(manifest) == []
    out = capsys.readouterr().out
    assert "[deadline] deferring" in out                       # first-deferral heads-up
    assert "WARNING: 16 points deferred" in out                # end-of-stage total


def test_no_deferral_warning_when_nothing_deferred(bundle, tmp_path, runner_factory, capsys):
    """A stage that defers nothing prints neither the deferral heads-up nor the
    end-of-stage WARNING (the loud path is gated on deferred > 0)."""
    store, manifest = mini_store(bundle, tmp_path)

    def responder(spec):
        return "result", {"ok": True, "result": _fake_result(bundle, len(spec["keys"])),
                          "runtime_s": 1.0}

    runner = runner_factory(store, manifest, bundle.cfg,
                            _runner_args(workers=2, batch_size=4), responder=responder)
    runner.run_batches(sweeplib.group_batches(bundle.panel_keys(0, 0, 0), batch_size=4),
                       "no-defer")
    assert runner.deferred == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "deferring" not in out


def test_store_lock_rejects_concurrent_writers(bundle, tmp_path, runner_factory):
    store, manifest = mini_store(bundle, tmp_path)
    runner_factory(store, manifest, bundle.cfg, _runner_args())
    import fcntl
    with open(Path(store.logs_dir) / "store.lock") as fh:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failing_batch_splits_to_singletons_before_recording_failure(bundle, tmp_path, runner_factory):
    """A failed multi-point batch is halved (retry_count++) until size 1; only
    then is _write_failure called, so every recorded failure is a singleton."""
    store, manifest = mini_store(bundle, tmp_path)

    def responder(spec):
        return "result", {"ok": False, "reason": "failed",
                          "message": "boom", "runtime_s": 0.0}

    runner = runner_factory(store, manifest, bundle.cfg,
                            _runner_args(workers=2, batch_size=4), responder=responder)
    runner.run_batches(sweeplib.group_batches(bundle.panel_keys(0, 0, 0)[:4], 4), "split")
    records = store.load_records(manifest)
    assert len(records) == 4
    assert all(r.status == "failed" for r in records)
    assert all(r.batch_size == 1 for r in records)       # split down to singletons
    assert {r.retry_count for r in records} == {2}       # 4 -> 2 -> 1 escalations
    assert runner.completed_points == 0


def test_pool_crash_requeues_on_its_own_budget_then_fails(bundle, tmp_path, runner_factory):
    """An in-flight pool death requeues the innocent batch on the pool_retries
    budget (0..5); the 6th death exhausts it and records a single failure whose
    retry_count is still 0 (it was never a genuine failure-split)."""
    store, manifest = mini_store(bundle, tmp_path)
    calls = {"n": 0}

    def responder(spec):
        calls["n"] += 1
        return "crash", BrokenProcessPool("worker died")

    runner = runner_factory(store, manifest, bundle.cfg, _runner_args(), responder=responder)
    runner.run_batches([sweeplib.Batch(keys=bundle.panel_keys(0, 0, 0)[:1])], "crash")
    assert calls["n"] == 6                       # 5 requeues + the exhausting death
    records = store.load_records(manifest)
    assert len(records) == 1 and records[0].status == "failed"
    assert records[0].retry_count == 0


def test_broken_pool_at_submit_recreates_and_resubmits(bundle, tmp_path, runner_factory):
    """A BrokenProcessPool raised by submit() itself is caught in _submit, which
    recreates the pool and resubmits; the point then completes normally."""
    store, manifest = mini_store(bundle, tmp_path)
    calls = {"n": 0}

    def responder(spec):
        calls["n"] += 1
        if calls["n"] == 1:
            return "submit_error", BrokenProcessPool("pool broke")
        return "result", {"ok": True, "result": _fake_result(bundle, len(spec["keys"])),
                          "runtime_s": 1.0}

    runner = runner_factory(store, manifest, bundle.cfg, _runner_args(), responder=responder)
    runner.run_batches([sweeplib.Batch(keys=bundle.panel_keys(0, 0, 0)[:1])], "recreate")
    assert runner._pool_restarts == 1
    assert calls["n"] == 2
    records = store.load_records(manifest)
    assert len(records) == 1 and records[0].status == "ok"


def test_sigint_drain_then_hard_abort_then_lock_release(bundle, tmp_path, runner_factory):
    """SIGINT drain semantics: the first delivery requests a graceful stop (the
    submit loop closes, nothing is recorded); a duplicate inside the 2 s debounce
    is swallowed; a deliberate second Ctrl-C after the debounce escalates to a
    hard abort; shutdown() releases the flock so a fresh coordinator can start."""
    store, manifest = mini_store(bundle, tmp_path)
    runner = runner_factory(store, manifest, bundle.cfg, _runner_args())
    handler = signal.getsignal(signal.SIGINT)
    assert signal.getsignal(signal.SIGTERM) is handler   # both installed

    handler(signal.SIGINT, None)                 # first Ctrl-C: graceful stop
    assert runner.stop_requested and not runner.aborted
    handler(signal.SIGINT, None)                 # debounced duplicate
    assert not runner.aborted
    # submit loop is closed: nothing dispatches, nothing is recorded
    runner.run_batches(sweeplib.group_batches(bundle.panel_keys(0, 0, 0), 4), "drain")
    assert runner.completed_points == 0 and store.load_records(manifest) == []

    runner._signal_time -= 3.0                    # pretend >2 s have elapsed
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert runner.aborted

    runner.shutdown()                             # releases the flock
    fresh = runner_factory(store, manifest, bundle.cfg, _runner_args())
    assert fresh.stop_requested is False


def test_second_coordinator_rejected_without_truncating_pid(bundle, tmp_path, runner_factory):
    """A second Runner on a locked store exits; and because the lock file is
    opened append-mode and written only after the flock succeeds, the live
    coordinator's PID record survives the rejected launch."""
    store, manifest = mini_store(bundle, tmp_path)
    runner_factory(store, manifest, bundle.cfg, _runner_args())
    lock_path = Path(store.logs_dir) / "store.lock"
    live = lock_path.read_text()
    assert live.startswith(f"pid {os.getpid()} ")
    with pytest.raises(SystemExit):
        sweeplib.Runner(store, manifest, bundle.cfg, _runner_args(),
                        sweeplib.CostModel(bundle.cfg))
    assert lock_path.read_text() == live          # append-mode: not truncated


def test_scatter_write_failure_records_nan_rows(bundle, tmp_path, runner_factory):
    """A failed scatter batch writes NaN-filled rows into the scatter series with
    the failure status, and never touches the main coherent-leakage series."""
    store, manifest = mini_store(bundle, tmp_path)
    runner = runner_factory(store, manifest, bundle.cfg, _runner_args())
    runner.gammas = {0: bundle.scatter_rates}
    batch = sweeplib.Batch(keys=bundle.panel_keys(0, 0, 0)[:2], scatter=True)
    runner._write_failure(batch, {"reason": "timeout", "message": "slow",
                                  "runtime_s": 3.0})
    rows = store.load_scatter_records(manifest)
    assert len(rows) == 2 and all(r["status"] == "timeout" for r in rows)
    assert all(np.all(np.isnan(r[ch])) for r in rows for ch in bundle.scatter_channels)
    assert store.load_records(manifest) == []


def test_merged_run_writes_both_series_and_dedupes_scatter(bundle, tmp_path, runner_factory):
    """Single-pass merge: a production batch with write_both_series on writes the
    coherent chunk AND the scatter records in one completion step; keys already in
    the scatter series are skipped at write time (resume dedup), so a subsequent
    scatter-missing check over the computed keys reports zero missing."""
    store, manifest = mini_store(bundle, tmp_path)
    keys = bundle.panel_keys(0, 0, 0)[:4]
    # a prior scatter pass already covered the first two keys
    seed = {ch: np.full((2, 4), 0.5) for ch in bundle.scatter_channels}
    store.write_scatter_chunk(1, manifest, keys[:2], bundle.cfg,
                              {0: bundle.scatter_rates}, 1e-9, 1e-12, "seed",
                              seed, np.array([1e-4, 1e-4]), 1.0)

    def responder(spec):
        n = len(spec["keys"])
        assert spec["both"] is True                 # production run -> merged spec
        scatter = {ch: np.random.default_rng(1).uniform(1e-6, 1e-2, (n, 4))
                   for ch in bundle.scatter_channels}
        return "result", {"ok": True, "result": _fake_result(bundle, n),
                          "scatter": scatter, "runtime_s": 5.0}

    runner = runner_factory(store, manifest, bundle.cfg,
                            _runner_args(workers=2, batch_size=4), responder=responder)
    runner.gammas = {0: bundle.scatter_rates}
    runner.write_both_series = True
    runner.scatter_done = {r["key"] for r in store.load_scatter_records(manifest)
                           if r["status"] == "ok"}
    runner.run_batches(sweeplib.group_batches(keys, batch_size=4), "merged")

    # coherent series: every computed key present
    assert set(keys) <= sweeplib.completed_keys(store.load_records(manifest))
    # scatter series: every computed key present exactly once (the two seeded keys
    # were not re-written), so cmd_scatter's missing check would be empty
    from collections import Counter
    counts = Counter(r["key"] for r in store.load_scatter_records(manifest)
                     if r["status"] == "ok")
    assert set(counts) == set(keys) and all(v == 1 for v in counts.values())
    assert [k for k in keys if k not in counts] == []


# ── shared worker entry (injected solve + scatter, via the worker context) ────


def _toy_scattering_integrals(times, states, gammas):
    """Toy per-channel scattering integral: counts atoms in local level 2."""
    pops = np.abs(states) ** 2                     # (n_t, n_p, 4, 9)
    idx = np.arange(9)
    a, b = np.divmod(idx, 3)
    w = (a == 2).astype(float) + (b == 2).astype(float)
    return {"p_e": gammas["p_e"] * np.trapezoid(pops @ w, times, axis=0)}


def test_worker_entry_solves_and_scatters():
    """The shared _worker_run_batch calls the injected solve + scattering_integrals
    from the (fork-inherited) worker context and returns both series."""
    ops = _toy_ops()
    PointKey, make_key, _pk, _ap, _ak = sweeplib.make_pointkey_type(
        panel_field="delta_idx", id_prefix="d",
        omega_anchors=_ODE_OME, dsweep_anchors=_ODE_DSW, panel_len=1, n_t=1)
    key_fields = ("delta_idx", "t_idx", "om_num", "om_den", "dw_num", "dw_den")
    key = make_key(0, 0, (0, 1), (0, 1))          # 200 MHz, 2 MHz
    cfg = SimpleNamespace(t_gate_us=(0.05,), n_eval_trajectory=21, ramp_frac=0.15)

    def _solve(ops, t_gate, om, dw, *, rtol, atol, ramp, use_swap, t_eval):
        state_labels = ("00", "01", "11") if use_swap else sweeplib.LOGICAL_INPUTS
        return sweeplib.integrate_batch(
            ops, t_gate, {"omega_420": om, "d_sweep": dw}, state_labels,
            rhs_factory=_toy_rhs_factory(_TOY_OM1013), dim=ops.h_static_diag.size,
            rtol=rtol, atol=atol, ramp=ramp, use_shifts=True, segmented=True,
            t_eval=t_eval)

    sweeplib.set_worker_context(
        cfg, {0: ops}, use_swap=True, gammas={0: {"p_e": 1.0e7}},
        key_type=PointKey, solve=_solve, scattering_integrals=_toy_scattering_integrals)

    spec = dict(keys=[tuple(getattr(key, f) for f in key_fields)],
                panel_idx=0, t_idx=0, rtol=1e-9, atol=1e-12,
                save_traj=False, scatter=True, timeout_s=60)
    out = sweeplib._worker_run_batch(spec)
    assert out["ok"]
    assert isinstance(out["result"], sweeplib.BatchResult)
    assert out["result"].psi_final.shape == (1, 4, 9)
    assert np.all(np.isfinite(out["result"].psi_final.view(float)))
    assert out["scatter"]["p_e"].shape == (1, 4)
    assert np.all(out["scatter"]["p_e"] >= 0.0)
    # the trajectory was consumed, not returned, since save_traj is False
    assert out["result"].states is None


# ── shared plotting: interpolation, LOO veil, credibility floor, 8x9 renderer ─
#
# The plot machinery (holdout residuals, audit-derived floor, metric selection,
# 8x9 grid renderer) is model-agnostic given a scatter-channel table and a plot
# spec; the ``_plot_metric_values`` scatter/total_error logic and the renderer are
# exercised parameterized over both script configs.  The ode's own six-metric plot
# smoke (incl. total_error joins, asserts no panel_*.png) lives in its script test.


def test_holdout_residuals_cover_every_interior_node():
    xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
    x, y = xs.ravel(), ys.ravel()
    z = np.zeros_like(x)
    assert np.all(sweeplib.holdout_residuals(x, y, z) == 0.0)
    z2 = x + 2 * y            # linear field: every holdout estimate is exact
    assert np.max(sweeplib.holdout_residuals(x, y, z2)) < 1e-12
    z3 = z2.copy()
    spike = np.flatnonzero((x == 2) & (y == 2))[0]
    z3[spike] += 1.0          # one bad node: it (and only its lines) flags
    resid = sweeplib.holdout_residuals(x, y, z3)
    assert resid[spike] == pytest.approx(1.0)
    corner = np.flatnonzero((x == 0) & (y == 0))[0]
    assert resid[corner] == 0.0


def test_credibility_floor_rule_and_fallback(bundle):
    keys = bundle.panel_keys(0, 0, 0)

    def rec(k, tier, leak):
        return sweeplib.PointRecord(
            key=k, tier=tier, rtol=1e-9, atol=1e-12, status="ok",
            max_leakage=leak, leakage=np.full(4, leak), worst_input="11",
            return_prob=np.ones(4), norm_err=np.zeros(4),
            psi_final=sweeplib._NO_STATES, nfev=1, runtime_s=1.0, batch_id="b",
            batch_size=1, retry_count=0, priority_score=0.0, message="",
            chunk_file="c", used_swap=True)

    few = [rec(keys[0], "production", 1e-4), rec(keys[0], "audit", 1e-4 + 1e-13)]
    vmin, info = sweeplib.credibility_floor(few)
    assert info["fallback"] and vmin == pytest.approx(1e-11)

    many = []
    for i, k in enumerate(keys[:10]):
        many.append(rec(k, "production", 1e-4))
        many.append(rec(k, "audit", 1e-4 + (i + 1) * 1e-13))
    vmin, info = sweeplib.credibility_floor(many)
    assert not info["fallback"]
    assert vmin == pytest.approx(max(1e-12, 10 * np.percentile(
        np.arange(1, 11) * 1e-13, 95)))


def test_plot_metric_values_scatter_metrics_and_total_error(bundle, tmp_path):
    """A tighter rtol wins per key; p_loss_total sums the channels; total_error
    joins the coherent leakage with the summed scatter before selecting the worst
    input — parameterized over both channel tables."""
    store, manifest = mini_store(bundle, tmp_path)
    key = bundle.panel_keys(0, 0, 0)[0]
    gammas = {0: bundle.scatter_rates}
    channels = bundle.scatter_channels
    loose = {ch: np.full((1, 4), 1e-3 * (j + 1)) for j, ch in enumerate(channels)}
    tight = {ch: np.full((1, 4), 2e-3 * (j + 1)) for j, ch in enumerate(channels)}
    store.write_scatter_chunk(1, manifest, [key], bundle.cfg, gammas, 1e-6, 1e-9,
                              "loose", loose, np.array([1e-4]), 1.0)
    store.write_scatter_chunk(2, manifest, [key], bundle.cfg, gammas, 1e-9, 1e-12,
                              "tight", tight, np.array([1e-4]), 1.0)
    values, *_ = sweeplib.plot_metric_values(
        store, manifest, [], channels[0], scatter_channels=channels)
    assert values[key] == pytest.approx(2e-3)          # tighter rtol wins
    values, *_ = sweeplib.plot_metric_values(
        store, manifest, [], "p_loss_total", scatter_channels=channels)
    assert values[key] == pytest.approx(
        sum(2e-3 * (j + 1) for j in range(len(channels))))

    # total_error must sum coherent leakage + scatter PER INPUT and only then take
    # the worst input.  Use a distinct key whose scatter is asymmetric across the
    # four inputs so the worst-coherent input (0) is NOT the worst-total input:
    # this discriminates the correct per-input-sum-then-max from a buggy
    # max-then-sum (which would over-count by pairing the two unrelated maxima).
    key2 = bundle.panel_keys(0, 0, 0)[1]
    leakage = np.array([0.40, 0.10, 0.20, 0.30])         # worst coherent at input 0
    scat_sum = np.array([0.05, 0.40, 0.30, 0.20])        # worst scatter at input 1
    te_scatter = {ch: np.array([scat_sum / len(channels)]) for ch in channels}
    store.write_scatter_chunk(3, manifest, [key2], bundle.cfg, gammas, 1e-9, 1e-12,
                              "te", te_scatter, np.array([1e-4]), 1.0)
    record = sweeplib.PointRecord(
        key=key2, tier="production", rtol=1e-9, atol=1e-12, status="ok",
        max_leakage=float(leakage.max()), leakage=leakage, worst_input="00",
        return_prob=np.ones(4), norm_err=np.zeros(4), psi_final=sweeplib._NO_STATES,
        nfev=1, runtime_s=1.0, batch_id="m", batch_size=1, retry_count=0,
        priority_score=0.0, message="", chunk_file="c", used_swap=True)
    values, *_ = sweeplib.plot_metric_values(
        store, manifest, [record], "total_error", scatter_channels=channels)
    total = leakage + scat_sum                           # per-input totals
    assert np.argmax(total) != np.argmax(leakage)        # the worst input flips
    per_input_then_max = float(total.max())              # correct: 0.50
    max_then_sum = float(leakage.max() + scat_sum.max())  # buggy: 0.80
    assert per_input_then_max != pytest.approx(max_then_sum)
    assert values[key2] == pytest.approx(per_input_then_max)


def _bundle_plot_spec(bundle):
    return sweeplib.PlotSpec(
        scatter_channels=bundle.scatter_channels,
        row_axis_key=bundle.cfg.panel_axis_name,
        row_label=lambda v: f"row = {v:g}",
        xlabel=r"$\Omega/2\pi$ (MHz)",
        system_desc=f"two-atom {bundle.name} CZ")


def test_render_panel_grid_writes_frozen_files_and_no_panel_pngs(bundle, tmp_path):
    """The renderer emits only <metric>_8x9.{png,pdf}; total_error joins the
    coherent and scatter series; no per-panel PNGs are ever produced."""
    store, manifest = mini_store(bundle, tmp_path)
    rng = np.random.default_rng(5)
    gammas = {ri: bundle.scatter_rates for ri in range(len(bundle.cfg.panel_axis))}
    for seq, (ri, ti) in enumerate([(0, 0), (1, 1)], start=1):
        keys = bundle.panel_keys(ri, ti, 1)            # full 7x7 grid
        res = _fake_result(bundle, len(keys), seed=seq)
        res.leakage = rng.uniform(1e-5, 1e-1, size=(len(keys), 4))
        res.max_leakage = res.leakage.max(axis=1)
        store.write_result_chunk(seq, manifest, keys, bundle.cfg, "production",
                                 1e-9, 1e-12, f"b{seq}", res, 60.0)
        scatter = {ch: rng.uniform(1e-6, 1e-2, size=(len(keys), 4))
                   for ch in bundle.scatter_channels}
        store.write_scatter_chunk(seq, manifest, keys, bundle.cfg, gammas, 1e-9,
                                  1e-12, f"s{seq}", scatter, res.max_leakage, 60.0)
    records = store.load_records(manifest, include_states=False)
    spec = _bundle_plot_spec(bundle)
    for metric in ("max_leakage", "total_error"):
        png, pdf = sweeplib.render_panel_grid(
            store, manifest, records, metric, spec, veil=True, dpi=60)
        assert Path(png).exists() and Path(pdf).exists()
        assert Path(png).name == f"{metric}_8x9.png"
    assert not list(Path(store.plots_dir).glob("panel_*.png"))


# ── shared CLI scaffold (common args + derived-output resolution) ─────────────


def test_cli_scaffold_common_args_and_derived_output():
    """add_common_args builds the shared skeleton (workers/batch-size 'auto',
    tolerances, --panels) and default_output/resolve_output derive the store dir
    per store-family root — verified for both script families."""
    import argparse

    for family in ("max_leakage_ode", "max_leakage_297"):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        run = sub.add_parser("run")
        sweeplib.add_common_args(run, family, compute=True)
        args = p.parse_args(["run", "--spacing-um", "5", "--workers", "auto",
                             "--batch-size", "auto"])
        assert args.spacing_um == 5.0 and args.output is None
        assert isinstance(args.workers, int) and 1 <= args.workers <= 40
        assert args.batch_size == 48
        assert args.rtol == 1e-9 and args.atol == 1e-12
        assert args.audit_rtol == 1e-10 and args.audit_atol == 1e-13
        assert args.panels is None
        args = p.parse_args(["run", "--workers", "12", "--batch-size", "4",
                             "--panels", "0,0;1,2"])
        assert args.workers == 12 and args.batch_size == 4 and args.panels == "0,0;1,2"
        assert sweeplib.default_output(family, 3.0) == os.path.join(
            "results", family, "a3.0")
        args = p.parse_args(["run"])
        sweeplib.resolve_output(args, family)
        assert args.output == os.path.join("results", family, "a3.0")

    # a non-compute subparser omits the pool/tolerance/--panels flags
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    st = sub.add_parser("status")
    sweeplib.add_common_args(st, "max_leakage_ode", compute=False)
    parsed = p.parse_args(["status", "--spacing-um", "7"])
    assert parsed.spacing_um == 7.0
    assert not hasattr(parsed, "workers") and not hasattr(parsed, "panels")
