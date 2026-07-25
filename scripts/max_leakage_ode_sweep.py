#!/usr/bin/env python
"""Authoritative original-frame max-leakage sweep for the two-atom rb87_7_mp CZ gate.

Scans terminal coherent leakage ``max_s || Q psi_s(T) ||^2`` (Q = projector onto the
45-dim nonlogical subspace, s in {00, 01, 10, 11}) over an 8x9 panel family
(rows: Delta_e/2pi in {9..50} GHz, columns: T in {1..4.5} us).  Every panel scans
x = Omega_420,max/2pi in [200, 1000] MHz (direct peak physical Rabi) and
y = D_sweep/2pi in [2, 30] MHz (half-amplitude detuning-sweep convention), on
progressively nested exact grids 4x4 -> 7x7 -> 13x13 -> 25x25 whose coarse anchors
keep the 20 MHz hardware-limit line an exact node at every level.

Physics is the closed (decay-off, Hermitian) full seven-level model of
``scripts/notebooks/error_buget.ipynb``: rb87_7_mp at 3.0 um / B = 20 G /
n = 70 / positive intermediate detuning, quintic-smoothstep power envelope with
ramp fraction 0.15 on both lasers (field amplitudes sqrt(E)), and the
Stark-compensated cosine chirp

    chirp(t) = -D cos(2 pi t / T) + (Dr - D1) E(t / T),
    D1 = -(4/3) Omega_420^2 / (4 Delta),   Dr = -Omega_1013^2 / (4 Delta),

whose optical phase is integrated *analytically* (no interpolation).  Omega_1013 is
fixed by the notebook 1013-nm convention (100 W nominal, optics_loss 0.9, top-hat
beam area 7*20*3 um^2) and recorded in the manifest.

The only production solver is original-frame complex128 adaptive DOP853
(production rtol=1e-9/atol=1e-12; audit rtol=1e-10/atol=1e-13), with

  * one precompiled, channel-aggregated Hamiltonian per detuning row,
  * logical inputs 00/01/11 propagated together as matrix columns (10 obtained by
    the verified atom-swap symmetry), each column solved with its bare logical
    diagonal energy subtracted (exact global-phase shift, restored at T),
  * within-panel multi-point batching guarded by a per-(point, logical-input)-block
    maximum error norm that mirrors the installed SciPy DOP853 estimate exactly
    (acceptance-gated; falls back to one point per solve if unverifiable),
  * stepper restarts at the analytic envelope breakpoints t = 0.15 T and 0.85 T.

Results are append-only NPZ chunks under ``results/max_leakage_ode/`` with a
hash-validated manifest; interrupted scans resume without recomputing.  ``run``
is a single pass: every production solve samples the trajectory and writes BOTH
the coherent-leakage chunk and the scattering-budget records, staging pilot ->
full 4x4 -> full 7x7 -> the requested ``--target-level`` (13x13 by default).

Usage
-----
    # default store: results/max_leakage_ode/a{spacing:.1f} (spacing default 3.0)
    python scripts/max_leakage_ode_sweep.py status
    python scripts/max_leakage_ode_sweep.py pilot  --spacing-um 5 --workers 40
    python scripts/max_leakage_ode_sweep.py run    --spacing-um 5 --target-level 13
    python scripts/max_leakage_ode_sweep.py run    --spacing-um 5 --dry-run
    python scripts/max_leakage_ode_sweep.py scatter --spacing-um 5 --level 13
    python scripts/max_leakage_ode_sweep.py audit  --spacing-um 5
    python scripts/max_leakage_ode_sweep.py export --spacing-um 5
    python scripts/max_leakage_ode_sweep.py plot   --spacing-um 5
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np
import scipy

# The shared sweep machinery lives beside this script in scripts/sweeplib/; make it
# importable whether the script is run as ``python scripts/max_leakage_ode_sweep.py``
# (scripts/ on sys.path[0]) or loaded by tests via spec_from_file_location.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import sweeplib
from sweeplib import (
    LEVEL_DENS, LEVEL_SIZES, LEVEL_FROM_SIZE,
    canon_coord, axis_coords, axis_values_mhz, make_pointkey_type,
    envelope, envelope_integral, verify_scipy_error_norm, BatchResult,
    ProvenanceColumns, PointRecord, best_records, completed_keys,
    audit_pairs, Runner, CostModel, Batch, group_batches, set_worker_context,
)
from sweeplib.store import _atomic_savez, _NO_STATES
from sweeplib.runner import _worker_run_batch

TAU = 2.0 * math.pi
SCHEMA_VERSION = 1

# ── Locked scientific configuration ──────────────────────────────────────────

DELTA_E_GHZ = (9.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)      # panel rows
T_GATE_US = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)          # panel columns

# Nested-axis anchors (MHz, exact rationals).  Deliberately non-uniform so the
# 20 MHz hardware cap is an exact node at every level; do not replace with linspace.
OMEGA_ANCHORS_MHZ = (Fraction(200), Fraction(1400, 3), Fraction(2200, 3), Fraction(1000))
DSWEEP_ANCHORS_MHZ = (Fraction(2), Fraction(10), Fraction(20), Fraction(30))

# LEVEL_DENS/LEVEL_SIZES/LEVEL_FROM_SIZE are imported from sweeplib.axes.

DSWEEP_HW_LIMIT_MHZ = 20.0         # horizontal reference line in every panel

LOGICAL_INPUTS = ("00", "01", "10", "11")

# Reference value of the fixed 1013 Rabi under the notebook convention; the scan
# recomputes it from the live model and warns if it drifted (never hardcoded).
OMEGA_1013_REFERENCE_RAD_S = TAU * 489.623065836e6


@dataclass(frozen=True)
class ScanConfig:
    """Immutable physics/scan configuration (the manifest's scientific payload)."""

    spacing_um: float = 3.0
    magnetic_field_G: float = 20.0
    ryd_level: int = 70
    detuning_sign: int = 1
    p1013_nominal_w: float = 100.0
    optics_loss: float = 0.9
    beam_factor: float = 7 * 20            # beam_area_um2 = beam_factor * spacing_um
    ramp_frac: float = 0.15
    rtol_production: float = 1e-9
    atol_production: float = 1e-12
    rtol_audit: float = 1e-10
    atol_audit: float = 1e-13
    delta_e_ghz: tuple = DELTA_E_GHZ
    t_gate_us: tuple = T_GATE_US
    omega_anchors_mhz: tuple = tuple(str(a) for a in OMEGA_ANCHORS_MHZ)
    dsweep_anchors_mhz: tuple = tuple(str(a) for a in DSWEEP_ANCHORS_MHZ)
    credibility_floor_min: float = 1e-12
    interp_space: str = "log10"
    n_eval_trajectory: int = 301

    @property
    def beam_area_um2(self) -> float:
        return self.beam_factor * self.spacing_um

    def physics_payload(self) -> dict:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    # Solver tolerances are per-record provenance (chunks store rtol/atol/tier), not
    # part of the scientific identity — CLI tolerance overrides must not orphan data.
    _NON_PHYSICS_FIELDS = ("rtol_production", "atol_production", "rtol_audit",
                           "atol_audit", "n_eval_trajectory")

    def physics_hash(self) -> str:
        d = {k: v for k, v in self.physics_payload().items()
             if k not in self._NON_PHYSICS_FIELDS}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


# ── Nested rational axes and canonical point keys ────────────────────────────
#
# An axis is three anchor segments; a node is the piecewise-linear interpolation of
# the anchors at position p = num/den in "segment units" in [0, 3], with (num, den)
# in lowest terms and den a power of two (<= 8 through level 3).  Reduced (num, den)
# pairs are the canonical resume keys: they never depend on string-rounded floats,
# and a node inserted at level L+1 that coincides with a level-L node reduces to the
# identical pair.  The nested-grid math (canon_coord/coord_value_mhz/axis_coords)
# lives in sweeplib.axes; the PointKey type keeps its serialized ``delta_idx`` field
# and ``d`` id prefix via the factory below.

PointKey, make_key, panel_keys, all_panels, all_keys = make_pointkey_type(
    panel_field="delta_idx", id_prefix="d",
    omega_anchors=OMEGA_ANCHORS_MHZ, dsweep_anchors=DSWEEP_ANCHORS_MHZ,
    panel_len=len(DELTA_E_GHZ), n_t=len(T_GATE_US),
)


def pilot_keys() -> list[PointKey]:
    """Reusable pilot nodes: all 72 panel centers + 16 factorial extremes, deduped."""
    center = ((3, 2), (3, 2))  # (600 MHz, 15 MHz): level-1 node of both axes
    keys = [make_key(di, ti, *center) for di, ti in all_panels()]
    extremes = [
        make_key(di, ti, (om, 1), (dw, 1))
        for di in (0, len(DELTA_E_GHZ) - 1)
        for ti in (0, len(T_GATE_US) - 1)
        for om in (0, 3)
        for dw in (0, 3)
    ]
    seen: set[PointKey] = set()
    out: list[PointKey] = []
    for k in keys + extremes:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


# ── Analytic pulse: envelope, chirp, phase ───────────────────────────────────
#
# Normalized time s = t / T, ramp fraction r.  Power envelope E(s): quintic
# smoothstep rise on [0, r], flat 1 on [r, 1-r], symmetric fall on [1-r, 1].
# Both field amplitudes are sqrt(E).  The intended chirp (rad/s) is
#     chirp(t) = -D cos(2 pi t / T) + (Dr - D1) E(t / T)
# and its exact unwrapped integral is
#     phi(t) = -(D T / 2 pi) sin(2 pi s) + (Dr - D1) T J(s),
# with J(s) the closed-form integral of E below.  phi is *not* wrapped mod 2 pi.
# quintic/quintic_antideriv/envelope/envelope_integral are imported from sweeplib.solver.


def stark_coefficients(omega_420: float, omega_1013: float, delta: float) -> tuple[float, float]:
    """(D1, Dr): the compensated single-photon Stark shifts (all rad/s)."""
    d1 = -(4.0 / 3.0) * omega_420 ** 2 / (4.0 * delta)
    dr = -(omega_1013 ** 2) / (4.0 * delta)
    return d1, dr


def chirp_rad_s(t, t_gate: float, d_sweep: float, dr_minus_d1: float, ramp: float = 0.15):
    """Intended instantaneous chirp (rad/s) at time ``t``; scalar or ndarray."""
    s = np.asarray(t, dtype=float) / t_gate
    return -d_sweep * np.cos(TAU * s) + dr_minus_d1 * envelope(s, ramp)


def phase_rad(t, t_gate: float, d_sweep: float, dr_minus_d1: float, ramp: float = 0.15):
    """Exact unwrapped optical phase phi(t) = integral_0^t chirp; scalar or ndarray."""
    s = np.asarray(t, dtype=float) / t_gate
    return (-d_sweep * t_gate / TAU) * np.sin(TAU * s) \
        + dr_minus_d1 * t_gate * envelope_integral(s, ramp)


def pulse_hash() -> str:
    """Behavioral fingerprint of the analytic pulse family.

    Hashes the sampled values of envelope/J/chirp/phase/Stark coefficients on a
    fixed probe grid, so any edit that changes what the lasers actually do — even
    with identical ScanConfig — changes the hash.  Recorded in the manifest and
    every chunk; resume/merge refuses a mismatch (whitespace/comment edits are
    deliberately invisible, unlike a source-text hash).
    """
    s = np.linspace(0.0, 1.0, 257)
    t_gate, d_sweep, drmd1 = 1.7e-6, TAU * 13e6, -TAU * 2.3e6
    probe = np.concatenate([
        np.asarray(envelope(s), dtype=float),
        np.asarray(envelope_integral(s), dtype=float),
        np.asarray(chirp_rad_s(s * t_gate, t_gate, d_sweep, drmd1), dtype=float),
        np.asarray(phase_rad(s * t_gate, t_gate, d_sweep, drmd1), dtype=float),
        np.asarray(stark_coefficients(TAU * 555e6, TAU * 489e6, TAU * 21e9)),
    ])
    return hashlib.sha256(probe.tobytes()).hexdigest()


# ── Model compilation and Hamiltonian aggregation ────────────────────────────
#
# The rb87_7_mp two-atom model is built and compiled ONCE per detuning row in the
# parent process; workers inherit the immutable matrices copy-on-write over fork.
# The 12 primitive drive channels are aggregated into fixed forward 420/1013 blocks
# B; with X = B + B^dag and Y = i (B - B^dag) a complex laser coefficient c = a + ib
# contributes a X + b Y, so the RHS needs no per-channel loop.  The grouped
# Hamiltonian is verified against the repository compiler to complex128 precision
# (``hamiltonian_equivalence_error``) before any production run.


def _dense(operator) -> np.ndarray:
    if hasattr(operator, "toarray"):
        return np.asarray(operator.toarray())
    return np.asarray(operator)


@dataclass
class PanelOperators:
    """Immutable aggregated operators for one detuning row (T-independent)."""

    delta_e_hz: float
    delta_rad_s: float                  # signed Delta = detuning_sign * 2pi * Delta_e_Hz
    h_static_diag: np.ndarray           # (49,) float64 — verified real diagonal
    x420: np.ndarray                    # (49, 49) complex128, Hermitian
    y420: np.ndarray
    x1013: np.ndarray
    y1013: np.ndarray
    amplitude_scale: float
    logical_indices: np.ndarray         # basis indices of |00>,|01>,|10>,|11>
    swap_perm: np.ndarray               # atom-swap permutation of the 49 basis states
    swap_symmetric: bool                # swap commutes with every H constituent

    def hash_bytes(self) -> bytes:
        h = hashlib.sha256()
        for a in (self.h_static_diag, self.x420, self.y420, self.x1013, self.y1013,
                  np.asarray([self.delta_rad_s, self.amplitude_scale])):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.digest()


def build_system(cfg: ScanConfig, delta_e_hz: float):
    """The notebook's two-atom rb87_7_mp system with a placeholder CZ protocol bound.

    The placeholder protocol only supplies the channel set for IR compilation; the
    production kernel computes drive coefficients analytically.
    """
    import ryd_gate as rg
    from ryd_gate.protocols import CZProtocol
    from ryd_gate.lattice import Register

    delta_rad_s = cfg.detuning_sign * TAU * delta_e_hz
    proto = CZProtocol(
        t_gate_s=1e-6,                       # placeholder 1 us gate
        intermediate_detuning_rad_s=delta_rad_s,
        omega_420_max_rad_s=1.0, omega_1013_max_rad_s=1.0,
        envelope_420=lambda t: 1.0, phase_420_rad=lambda t: 0.0,
        envelope_1013=lambda t: 1.0, phase_1013_rad=lambda t: 0.0,
    )
    return rg.RydbergSystem(
        level_structure=rg.level_structure(
            "rb87_7_mp", ryd_level=cfg.ryd_level,
            magnetic_field_G=cfg.magnetic_field_G),
        register=Register.chain(2, spacing_um=cfg.spacing_um),
        protocol=proto,
    )


def compute_omega_1013(cfg: ScanConfig) -> float:
    """Fixed 1013 Rabi (rad/s) under the notebook 100 W / optics_loss / top-hat convention."""
    from ryd_gate.physics import rb87_7_mp_rabi_frequencies

    _, omega_1013 = rb87_7_mp_rabi_frequencies(
        1.0 * (1.0 - cfg.optics_loss),
        cfg.p1013_nominal_w * (1.0 - cfg.optics_loss),
        cfg.beam_area_um2,
        ryd_level=cfg.ryd_level,
    )
    return float(omega_1013)


def _swap_permutation(local_dim: int = 7) -> np.ndarray:
    idx = np.arange(local_dim * local_dim)
    a, b = divmod(idx, local_dim)
    return b * local_dim + a


def aggregate_operators(system, delta_e_hz: float) -> PanelOperators:
    """Compile the system and aggregate its channels into fixed dense blocks."""
    from ryd_gate.backends.exact.compiler import compile_exact
    from ryd_gate.core.states import product_index

    ham, _t_gate = compile_exact(system, hamiltonian_format="dense")
    dim = ham.dim

    # ham.h_static carries the atomic diagonal + the Rydberg pair interaction, but
    # not the intermediate detuning: the protocol now supplies Delta as constant
    # diagonal drive channels (E[e1,e1]/E[e2,e2]/E[e3,e3]).  Fold those constants
    # into the static diagonal, then split off the off-diagonal laser channels.
    h_static = np.array(ham.h_static, dtype=np.complex128)
    delta_rad_s = 0.0
    laser_channels = []
    for ch in ham.channels:
        if ch.is_diag:
            c = complex(ch.coeff(0.0))
            h_static += c * _dense(ch.sum_op)
            delta_rad_s = c.real
        else:
            laser_channels.append(ch)

    off_diag = h_static - np.diag(np.diag(h_static))
    if np.max(np.abs(off_diag)) > 0.0:
        raise RuntimeError("static Hamiltonian is not diagonal; kernel assumption broken")
    diag = np.diag(h_static)
    if np.max(np.abs(diag.imag)) > 0.0:
        raise RuntimeError(
            "static Hamiltonian has imaginary diagonal (decay enabled?); "
            "this scan requires closed Hermitian dynamics"
        )

    ratios = {"420": {}, "1013": {}}
    for leg in system.level_structure._laser_legs:
        if leg.group in ratios:
            ratios[leg.group][leg.channel] = leg.factor
    ops = {ch._channel: _dense(ch.sum_op) for ch in laser_channels}
    missing = (set(ratios["420"]) | set(ratios["1013"])) - set(ops)
    if missing:
        raise RuntimeError(f"drive channels missing from compiled Hamiltonian: {sorted(missing)}")
    for ch in laser_channels:
        if ch.sum_op_hc is None:
            raise RuntimeError(f"drive channel {ch._channel} lacks the h.c. leg")

    b420 = sum(ratios["420"][ch] * ops[ch] for ch in ratios["420"])
    b1013 = sum(ratios["1013"][ch] * ops[ch] for ch in ratios["1013"])
    x420 = b420 + b420.conj().T
    y420 = 1j * (b420 - b420.conj().T)
    x1013 = b1013 + b1013.conj().T
    y1013 = 1j * (b1013 - b1013.conj().T)

    logical = np.asarray(
        [product_index(list(s), system._basis) for s in LOGICAL_INPUTS]
    )
    perm = _swap_permutation(system._basis.local_dim)

    def _swap_invariant(mat: np.ndarray) -> bool:
        return bool(np.array_equal(mat[np.ix_(perm, perm)], mat))

    swap_ok = (
        bool(np.array_equal(diag.real[perm], diag.real))
        and all(_swap_invariant(m) for m in (x420, y420, x1013, y1013))
    )

    return PanelOperators(
        delta_e_hz=float(delta_e_hz),
        delta_rad_s=float(delta_rad_s),
        h_static_diag=np.ascontiguousarray(diag.real),
        x420=x420, y420=y420, x1013=x1013, y1013=y1013,
        amplitude_scale=1.0,
        logical_indices=logical,
        swap_perm=perm,
        swap_symmetric=swap_ok,
    )


def hamiltonian_equivalence_error(
    system,
    ops: PanelOperators,
    t_gate: float,
    omega_420: float,
    omega_1013: float,
    d_sweep: float,
    times: np.ndarray,
    ramp: float = 0.15,
) -> float:
    """Max |H_grouped - H_compiled| over ``times`` for one concrete pulse.

    Builds the repository Hamiltonian directly from a freshly compiled IR with the
    *same* analytic pulse bound as a real CZProtocol, then compares against the
    aggregated evaluation used by the production kernel.
    """
    from ryd_gate.backends.exact.compiler import compile_exact
    from ryd_gate.protocols import CZProtocol

    d1, dr = stark_coefficients(omega_420, omega_1013, ops.delta_rad_s)
    drmd1 = dr - d1

    proto = CZProtocol(
        t_gate_s=t_gate,
        intermediate_detuning_rad_s=ops.delta_rad_s,
        omega_420_max_rad_s=omega_420, omega_1013_max_rad_s=omega_1013,
        envelope_420=lambda t: float(np.sqrt(envelope(t / t_gate, ramp))),
        phase_420_rad=lambda t: float(phase_rad(t, t_gate, d_sweep, drmd1, ramp)),
        envelope_1013=lambda t: float(np.sqrt(envelope(t / t_gate, ramp))),
        phase_1013_rad=lambda t: 0.0,
    )
    bound = system.with_protocol(proto)
    ham, _ = compile_exact(bound, hamiltonian_format="dense")

    static = np.asarray(ham.h_static, dtype=np.complex128)
    drive = [(_dense(ch.sum_op), ch.coeff, ch.is_diag,
              None if ch.is_diag else _dense(ch.sum_op_hc))
             for ch in ham.channels]

    worst = 0.0
    for t in np.asarray(times, dtype=float):
        h_ref = static.copy()
        for op, coeff_fn, is_diag, op_hc in drive:
            c = complex(coeff_fn(t))
            h_ref += c * op
            if not is_diag:
                h_ref += np.conj(c) * op_hc

        s = t / t_gate
        amp = float(np.sqrt(envelope(s, ramp)))
        phi = float(phase_rad(t, t_gate, d_sweep, drmd1, ramp))
        c420 = ops.amplitude_scale * omega_420 * amp * np.exp(-1j * phi)
        c1013 = ops.amplitude_scale * omega_1013 * amp
        h_grp = (np.diag(ops.h_static_diag.astype(np.complex128))
                 + c420.real * ops.x420 + c420.imag * ops.y420
                 + c1013 * ops.x1013)  # c1013 is real: phi_1013 == 0 by construction
        worst = max(worst, float(np.max(np.abs(h_ref - h_grp))))
    return worst


# ── Batched original-frame integration kernel ────────────────────────────────
#
# The generic block-max DOP853 kernel (segmented restarts, per-column global-phase
# shifts, atom-swap reconstruction, t_eval trajectory sampling) and BatchResult live
# in sweeplib.solver; this script injects the two-drive (420/1013) + Stark-chirp RHS
# for the rb87_7_mp model.  Each column is solved with its bare logical diagonal
# energy subtracted (via cols["shift"]); sweeplib restores the exact global phase.


def _ode_rhs_factory(omega_1013: float):
    """Build the two-drive + Stark-chirp RHS factory consumed by sweeplib.integrate_batch."""

    def rhs_factory(ops, cols, t_gate, ramp):
        om_cols = cols["omega_420"]
        dsw_cols = cols["d_sweep"]
        d1, dr = stark_coefficients(om_cols, omega_1013, ops.delta_rad_s)
        drmd1_cols = dr - d1
        diag_row = ops.h_static_diag[None, :] - cols["shift"][:, None]   # (n_cols, dim) real
        x420_t = np.ascontiguousarray(ops.x420.T)
        y420_t = np.ascontiguousarray(ops.y420.T)
        x1013_t = np.ascontiguousarray(ops.x1013.T)
        ascale = ops.amplitude_scale
        sin_coef = -t_gate / TAU
        n_cols, dim = diag_row.shape

        def rhs(t, y):
            s = t / t_gate
            env = envelope(s, ramp)
            amp = math.sqrt(float(env))
            phi = (sin_coef * math.sin(TAU * s)) * dsw_cols \
                + (t_gate * float(envelope_integral(s, ramp))) * drmd1_cols
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


def integrate_batch(
    ops: PanelOperators,
    t_gate: float,
    omega_420: np.ndarray,
    d_sweep: np.ndarray,
    omega_1013: float,
    rtol: float,
    atol: float,
    ramp: float = 0.15,
    use_swap: bool | None = None,
    use_shifts: bool = True,
    segmented: bool = True,
    t_eval: np.ndarray | None = None,
) -> BatchResult:
    """Propagate all logical inputs of ``len(omega_420)`` panel points together.

    Columns are (point-major) the logical inputs 00/01/11 — 10 is reconstructed by
    the atom-swap permutation when the verified symmetry holds, else all four are
    propagated.  Thin wrapper that supplies the two-drive+Stark RHS to the shared
    sweeplib kernel.
    """
    omega_420 = np.asarray(omega_420, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    if omega_420.shape != d_sweep.shape or omega_420.ndim != 1:
        raise ValueError("omega_420 and d_sweep must be equal-length 1-D arrays")
    if use_swap is None:
        use_swap = ops.swap_symmetric
    state_labels = ("00", "01", "11") if use_swap else LOGICAL_INPUTS
    return sweeplib.integrate_batch(
        ops, t_gate,
        {"omega_420": omega_420, "d_sweep": d_sweep},
        state_labels,
        rhs_factory=_ode_rhs_factory(omega_1013),
        dim=ops.h_static_diag.size,
        rtol=rtol, atol=atol, ramp=ramp,
        use_shifts=use_shifts, segmented=segmented, t_eval=t_eval,
    )


# ── Append-only NPZ persistence ──────────────────────────────────────────────
#
# Layout (all under --output):
#   manifest.json                 immutable scientific config + hashes
#   chunks/chunk_NNNNNN.npz       append-only per-batch results (authoritative)
#   trajectories/traj_NNNNNN.npz  dense trajectories for pilot/extreme/audit points
#   logs/  reports/  exports/  plots/
#
# Chunks hold only fixed-shape numeric/str arrays (allow_pickle=False), are written
# to a temp file, fsynced, then atomically renamed; only the parent writes.  Chunks
# are authoritative — every index/export below is derived and regenerable.

_KEY_FIELDS = ("delta_idx", "t_idx", "om_num", "om_den", "dw_num", "dw_den")

# The Store, atomic NPZ writes, three-hash provenance gates, chunk/scatter series
# and the PointRecord loader live in sweeplib.store; this script supplies its
# serialized field name (delta_idx), the physical descriptor columns and the fixed
# 1013 Rabi via the ProvenanceColumns bundle below (formats frozen).


def _ode_descriptor(cfg: "ScanConfig", keys) -> dict:
    """Base physical-descriptor columns of one batch (delta_e / t_gate / drives)."""
    return {
        "delta_e_ghz": np.asarray([cfg.delta_e_ghz[k.delta_idx] for k in keys]),
        "t_gate_us": np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
        "omega_420_mhz": np.asarray([float(k.omega_mhz()) for k in keys]),
        "dsweep_mhz": np.asarray([float(k.dsweep_mhz()) for k in keys]),
    }


def _ode_result_extra(cfg: "ScanConfig", keys, manifest: dict) -> dict:
    """Extended coherent-chunk columns: rad/s conversions + the fixed 1013 Rabi."""
    de_ghz = np.asarray([cfg.delta_e_ghz[k.delta_idx] for k in keys])
    tg_us = np.asarray([cfg.t_gate_us[k.t_idx] for k in keys])
    om_mhz = np.asarray([float(k.omega_mhz()) for k in keys])
    dw_mhz = np.asarray([float(k.dsweep_mhz()) for k in keys])
    return {
        "delta_e_rad_s": de_ghz * 1e9 * TAU,
        "t_gate_s": tg_us * 1e-6,
        "omega_420_rad_s": om_mhz * 1e6 * TAU,
        "dsweep_rad_s": dw_mhz * 1e6 * TAU,
        "omega_1013_rad_s": np.full(len(keys), float(manifest["omega_1013_rad_s"])),
    }


class Store(sweeplib.Store):
    """The rb87_7_mp scan store: the shared sweeplib.Store bound to this script's
    serialized ``delta_idx`` field, physical descriptor columns and 1013 Rabi
    provenance.  Constructible from just the output directory (resume/status)."""

    def __init__(self, output_dir: str):
        super().__init__(
            output_dir, key_type=PointKey, key_fields=_KEY_FIELDS,
            provenance_columns=ProvenanceColumns(
                scatter_channels=SCATTER_CHANNELS, default_dim=49,
                descriptor=_ode_descriptor, result_extra=_ode_result_extra,
                schema_version=SCHEMA_VERSION))


def _manifest_extras(cfg: "ScanConfig", omega_1013: float) -> dict:
    """The ODE-specific manifest payload + resume guard for init_or_validate_manifest."""
    axes = {
        "delta_e_ghz": list(cfg.delta_e_ghz),
        "t_gate_us": list(cfg.t_gate_us),
        "omega_anchors_mhz": [str(a) for a in OMEGA_ANCHORS_MHZ],
        "dsweep_anchors_mhz": [str(a) for a in DSWEEP_ANCHORS_MHZ],
        "level_sizes": list(LEVEL_SIZES),
        "dsweep_hw_limit_mhz": DSWEEP_HW_LIMIT_MHZ,
    }
    extra_fields = {
        "omega_1013_rad_s": omega_1013,
        "omega_1013_over_2pi_MHz": omega_1013 / TAU / 1e6,
        "omega_1013_reference_rad_s": OMEGA_1013_REFERENCE_RAD_S,
    }

    def _guard(existing: dict) -> None:
        rec = float(existing.get("omega_1013_rad_s", 0.0))
        if abs(rec - omega_1013) > 1e-6 * abs(omega_1013):
            raise RuntimeError(
                f"Omega_1013 mismatch: manifest {rec!r} vs current {omega_1013!r}")

    return dict(pulse_hash=pulse_hash(), axes=axes, extra_fields=extra_fields,
                extra_guard=_guard)


def export_store(store: Store, records: list[PointRecord] | None = None) -> tuple[str, str]:
    """Regenerate exports/latest_merged.npz and exports/points.csv from chunks."""
    manifest = store.load_manifest()
    if manifest is None:
        raise RuntimeError(f"no manifest under {store.root}")
    if records is None:
        records = store.load_records(manifest)
    best = best_records(records)
    keys = sorted(best.keys())
    rows = [best[k] for k in keys]
    cfg = ScanConfig(**{k: tuple(v) if isinstance(v, list) else v
                        for k, v in manifest["physics"].items()
                        if k != "schema_version"})

    store.ensure_dirs()
    merged_path = os.path.join(store.exports_dir, "latest_merged.npz")
    n = len(rows)
    payload = dict(
        schema_version=np.int64(SCHEMA_VERSION),
        scan_uuid=str(manifest["scan_uuid"]),
        **store.keys_to_arrays(keys),
        delta_e_ghz=np.asarray([cfg.delta_e_ghz[k.delta_idx] for k in keys]),
        t_gate_us=np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
        omega_420_mhz=np.asarray([float(k.omega_mhz()) for k in keys]),
        dsweep_mhz=np.asarray([float(k.dsweep_mhz()) for k in keys]),
        max_leakage=np.asarray([r.max_leakage for r in rows]),
        leakage=np.asarray([r.leakage for r in rows]).reshape(n, 4),
        worst_input=np.asarray([r.worst_input for r in rows], dtype="U2"),
        return_prob=np.asarray([r.return_prob for r in rows]).reshape(n, 4),
        norm_err_max=np.asarray([float(np.max(r.norm_err)) for r in rows]),
        tier=np.asarray([r.tier for r in rows], dtype="U10"),
        rtol=np.asarray([r.rtol for r in rows]),
        atol=np.asarray([r.atol for r in rows]),
        psi_final=np.asarray([r.psi_final for r in rows]).reshape(n, 4, -1)
        if n else np.zeros((0, 4, 49), dtype=np.complex128),
    )
    _atomic_savez(merged_path, **payload)

    csv_path = os.path.join(store.exports_dir, "points.csv")
    cols = ["point_id", "delta_e_ghz", "t_gate_us", "omega_420_mhz", "dsweep_mhz",
            "max_leakage", "leak_00", "leak_01", "leak_10", "leak_11", "worst_input",
            "min_return_prob", "norm_err_max", "tier", "rtol", "atol", "nfev",
            "runtime_s", "batch_id"]
    tmp = csv_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for k, r in zip(keys, rows):
            fh.write(",".join(str(v) for v in (
                k.id(), cfg.delta_e_ghz[k.delta_idx], cfg.t_gate_us[k.t_idx],
                float(k.omega_mhz()), float(k.dsweep_mhz()),
                repr(r.max_leakage), repr(r.leakage[0]), repr(r.leakage[1]),
                repr(r.leakage[2]), repr(r.leakage[3]), r.worst_input,
                repr(float(np.min(r.return_prob))), repr(float(np.max(r.norm_err))),
                r.tier, r.rtol, r.atol, r.nfev, f"{r.runtime_s:.3f}", r.batch_id,
            )) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, csv_path)
    return merged_path, csv_path


# ── Scattering-budget integrals (supplemental `scatter` data) ────────────────
#
# p_ch = Gamma_ch * integral_0^T <n_ch(t)> dt for the intermediate (e1,e2,e3),
# Rydberg (r) and garbage-Rydberg (r_garb) manifolds, per logical input — the
# same trapezoid-on-301-samples convention as error_buget.ipynb and the stored
# pilot/audit trajectories.  These are written to a separate append-only
# ``scatter/`` chunk series and never touch the coherent-leakage chunks.

SCATTER_CHANNELS = ("p_mid", "p_ryd", "p_r_garb")
_SCATTER_LEVEL_GROUPS = {"p_mid": (2, 3, 4), "p_ryd": (5,), "p_r_garb": (6,)}


def _scatter_weight_vectors(local_dim: int = 7) -> dict[str, np.ndarray]:
    """Diagonal weights counting atoms of each decay group per two-atom index."""
    idx = np.arange(local_dim * local_dim)
    a, b = np.divmod(idx, local_dim)
    return {name: (np.isin(a, g).astype(float) + np.isin(b, g).astype(float))
            for name, g in _SCATTER_LEVEL_GROUPS.items()}


def scattering_integrals(times: np.ndarray, states: np.ndarray,
                         gammas: dict[str, float]) -> dict[str, np.ndarray]:
    """(n_points, 4) scattering probabilities per channel from sampled states.

    ``states`` is the (n_times, n_points, 4, dim) trajectory of
    :func:`integrate_batch`; the populations are diagonal expectations, so each
    channel is a weighted |psi|^2 sum integrated by the trapezoid rule.
    """
    pops = np.abs(states) ** 2                       # (n_t, n_p, 4, dim)
    local_dim = int(round(math.sqrt(states.shape[-1])))
    weights = _scatter_weight_vectors(local_dim)
    out = {}
    for name, w in weights.items():
        n_t = pops @ w                               # (n_t, n_p, 4)
        out[name] = gammas[name] * np.trapezoid(n_t, times, axis=0)
    return out


def model_decay_rates(system) -> dict[str, float]:
    """The rb87_7 decay rates (rad/s) used for the scattering integrals."""
    rates = system.level_structure.decay_rates_per_s
    return {
        "p_mid": float(rates["e1"]["total"]),
        "p_ryd": float(rates["r"]["total"]),
        "p_r_garb": float(rates["r_garb"]["total"]),
    }


def _per_panel_gammas(cfg: ScanConfig, decay_rates: dict[str, float]) -> dict[int, dict[str, float]]:
    """The shared worker/store API keys gammas by panel row; this model's decay
    rates are Delta_e-independent, so every row shares the one measured dict."""
    return {di: decay_rates for di in range(len(cfg.delta_e_ghz))}


# ── Startup: warm ARC, compile every row, run the mandatory verifications ────

HAM_EQUIV_REL_TOL = 1e-12
ERR_NORM_REL_TOL = 1e-12
PACK_GATE_STATE_TOL = 1e-6
PACK_GATE_LEAK_TOL = 1e-8


def _script_code_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def warm_and_build(cfg: ScanConfig) -> tuple[dict[int, PanelOperators], float, str, dict]:
    """Warm ARC in-parent, compile/aggregate all detuning rows, verify invariants.

    Returns ``(ops_by_delta, omega_1013, model_hash, checks)``.  Raises if the
    grouped Hamiltonian deviates from the repository compiler, if the static
    Hamiltonian is not closed/Hermitian, or if the SciPy error-norm seam moved.
    """
    t0 = time.time()
    omega_1013 = compute_omega_1013(cfg)   # first ARC touch happens here, pre-fork
    omega_dev = abs(omega_1013 - OMEGA_1013_REFERENCE_RAD_S) / OMEGA_1013_REFERENCE_RAD_S
    if omega_dev > 1e-6:
        print(f"WARNING: Omega_1013/2pi = {omega_1013 / TAU / 1e6:.6f} MHz deviates "
              f"from the recorded reference {OMEGA_1013_REFERENCE_RAD_S / TAU / 1e6:.6f} "
              f"MHz by {omega_dev:.2e} (relative) — the 1013 model changed; the "
              "manifest records the live value.", flush=True)

    ops_by_delta: dict[int, PanelOperators] = {}
    systems: dict[int, object] = {}
    for di, de_ghz in enumerate(cfg.delta_e_ghz):
        systems[di] = build_system(cfg, de_ghz * 1e9)
        ops_by_delta[di] = aggregate_operators(systems[di], de_ghz * 1e9)

    h = hashlib.sha256()
    for di in sorted(ops_by_delta):
        h.update(ops_by_delta[di].hash_bytes())
    h.update(np.asarray([omega_1013]).tobytes())
    model_hash = h.hexdigest()

    # Grouped-vs-compiled Hamiltonian equivalence on every row: ramp edges,
    # endpoints, and deterministic pseudo-random interior times.
    rng = np.random.default_rng(20260712)
    ham_dev_rel = 0.0
    t_gate = cfg.t_gate_us[0] * 1e-6
    probe_times = np.concatenate([
        [0.0, cfg.ramp_frac * t_gate, (1 - cfg.ramp_frac) * t_gate, t_gate],
        np.sort(rng.uniform(0.0, t_gate, 6)),
    ])
    for di in ops_by_delta:
        dev = hamiltonian_equivalence_error(
            systems[di], ops_by_delta[di], t_gate,
            omega_420=TAU * 600e6, omega_1013=omega_1013, d_sweep=TAU * 15e6,
            times=probe_times, ramp=cfg.ramp_frac)
        scale = float(np.max(np.abs(ops_by_delta[di].h_static_diag))
                      + 2 * TAU * 600e6 + 2 * omega_1013)
        ham_dev_rel = max(ham_dev_rel, dev / scale)
    if ham_dev_rel > HAM_EQUIV_REL_TOL:
        raise RuntimeError(
            f"grouped Hamiltonian deviates from the compiled IR by relative "
            f"{ham_dev_rel:.3e} (> {HAM_EQUIV_REL_TOL:g}); refusing to run")

    err_norm_dev = verify_scipy_error_norm()
    norm_ok = err_norm_dev <= ERR_NORM_REL_TOL
    if not norm_ok:
        print(f"WARNING: installed SciPy DOP853 error norm could not be reproduced "
              f"(max dev {err_norm_dev:.3e}); multi-point batching disabled.", flush=True)

    swap_ok = all(o.swap_symmetric for o in ops_by_delta.values())
    if not swap_ok:
        print("WARNING: atom-swap symmetry verification failed on some rows; "
              "all four logical inputs will be propagated.", flush=True)

    checks = {
        "omega_1013_rad_s": omega_1013,
        "omega_1013_rel_dev_from_reference": omega_dev,
        "hamiltonian_equivalence_rel_dev": ham_dev_rel,
        "error_norm_max_dev": err_norm_dev,
        "error_norm_verified": bool(norm_ok),
        "swap_symmetric": bool(swap_ok),
        "decay_rates_rad_s": model_decay_rates(systems[0]),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "build_seconds": time.time() - t0,
    }
    return ops_by_delta, omega_1013, model_hash, checks


def setup_run(args) -> tuple[Store, dict, ScanConfig, dict[int, PanelOperators], float, dict]:
    """Shared bring-up for pilot/run/audit: build, verify, manifest, worker context."""
    cfg = ScanConfig(
        spacing_um=args.spacing_um,
        rtol_production=args.rtol, atol_production=args.atol,
        rtol_audit=args.audit_rtol, atol_audit=args.audit_atol,
    )
    store = Store(args.output)
    store.ensure_dirs()
    ops, omega_1013, model_hash, checks = warm_and_build(cfg)
    manifest = store.init_or_validate_manifest(
        cfg, model_hash, _script_code_hash(),
        run_meta={
            "argv": sys.argv[1:], "workers": args.workers,
            "batch_size": args.batch_size,
            "budget_hours": getattr(args, "budget_hours", None),
            "reserve_hours": getattr(args, "reserve_hours", None),
        },
        **_manifest_extras(cfg, omega_1013))
    ver_path = os.path.join(store.reports_dir, "verification.json")
    with open(ver_path + ".tmp", "w") as fh:
        json.dump(checks, fh, indent=2)
    os.replace(ver_path + ".tmp", ver_path)

    # The shared worker context takes the script's solve wrapper (which closes over
    # the fixed 1013 Rabi) and its scattering_integrals; gammas are keyed by panel
    # row (this model's decay rates are row-independent, so the same dict for all).
    def _solve(ops, t_gate, omega_420, d_sweep, *, rtol, atol, ramp, use_swap, t_eval):
        return integrate_batch(ops, t_gate, omega_420, d_sweep, omega_1013,
                               rtol=rtol, atol=atol, ramp=ramp, use_swap=use_swap,
                               t_eval=t_eval)

    set_worker_context(
        cfg, ops, use_swap=checks["swap_symmetric"],
        gammas=_per_panel_gammas(cfg, checks["decay_rates_rad_s"]),
        key_type=PointKey, solve=_solve, scattering_integrals=scattering_integrals)
    print(f"[setup] Omega_1013/2pi = {omega_1013 / TAU / 1e6:.6f} MHz | "
          f"H equivalence rel dev {checks['hamiltonian_equivalence_rel_dev']:.2e} | "
          f"error-norm dev {checks['error_norm_max_dev']:.2e} | "
          f"swap {'ok' if checks['swap_symmetric'] else 'FAILED'}", flush=True)
    return store, manifest, cfg, ops, omega_1013, checks


# ── Pilot: reusable nodes, throughput, packing acceptance gate ───────────────

# Deterministic packing-gate panel and varied in-panel nodes (all level-0, reusable).
PACK_GATE_PANEL = (3, 0)          # Delta_e = 20 GHz, T = 1 us
PACK_GATE_COORDS = [((0, 1), (0, 1)), ((3, 1), (3, 1)), ((0, 1), (3, 1)),
                    ((3, 1), (0, 1)), ((1, 1), (2, 1)), ((2, 1), (1, 1))]


def run_packing_gate(runner: Runner, done: set[PointKey]) -> dict:
    """Acceptance gate for multi-point batching: packed vs isolated at *both* the
    production and audit tolerances (spec-mandated).  Isolated runs are saved as
    authoritative records; the packed probes are compared in memory and never
    persisted.  Any worker/pool error disables batching instead of crashing."""
    keys = [make_key(*PACK_GATE_PANEL, om, dw) for om, dw in PACK_GATE_COORDS]
    iso_out: dict[tuple[PointKey, str], BatchResult] = {}

    try:
        futures = {}
        for tier in ("production", "audit"):
            for k in keys:
                b = Batch(keys=[k], tier=tier)
                futures[runner._submit(b)] = b
        packed_futs = {tier: runner._submit(Batch(keys=keys, tier=tier))
                       for tier in ("production", "audit")}

        for fut, b in futures.items():
            out = fut.result()
            if not out["ok"]:
                return {"enabled": False, "reason": f"isolated gate run failed: "
                        f"{out.get('message', out.get('reason'))}"}
            if not (b.keys[0] in done and b.tier == "production"):
                runner._write_success(b, out)
            iso_out[(b.keys[0], b.tier)] = out["result"]
        packed_out = {tier: fut.result() for tier, fut in packed_futs.items()}
    except Exception as exc:  # pool breakage must not kill the whole run
        return {"enabled": False, "reason": f"gate execution error: {exc}"[:240]}

    devs = {}
    for tier, out in packed_out.items():
        if not out["ok"]:
            return {"enabled": False,
                    "reason": f"packed {tier} gate run failed: {out.get('message')}"}
        pres: BatchResult = out["result"]
        devs[tier] = {
            "max_state_dev": max(
                float(np.max(np.abs(pres.psi_final[i] - iso_out[(k, tier)].psi_final[0])))
                for i, k in enumerate(keys)),
            "max_leakage_dev": max(
                float(np.max(np.abs(pres.leakage[i] - iso_out[(k, tier)].leakage[0])))
                for i, k in enumerate(keys)),
        }
    enabled = all(d["max_state_dev"] < PACK_GATE_STATE_TOL
                  and d["max_leakage_dev"] < PACK_GATE_LEAK_TOL
                  for d in devs.values())
    return {
        "enabled": bool(enabled),
        "panel": PACK_GATE_PANEL,
        "n_points": len(keys),
        "tiers": devs,
        "max_state_dev": max(d["max_state_dev"] for d in devs.values()),
        "max_leakage_dev": max(d["max_leakage_dev"] for d in devs.values()),
        "state_tol": PACK_GATE_STATE_TOL,
        "leak_tol": PACK_GATE_LEAK_TOL,
    }


def stage_pilot(runner: Runner, panels: set[tuple[int, int]] | None = None) -> dict:
    """Stage 1: pilot nodes (+ initial audits + packing gate); returns pilot report."""
    records = runner.store.load_records(runner.manifest, include_states=False)
    done = completed_keys(records)
    audit_done = completed_keys(records, "audit")
    pkeys = _filter_panels(pilot_keys(), panels)
    missing = [k for k in pkeys if k not in done]
    if missing:
        runner.run_batches(
            [Batch(keys=[k], save_traj=True) for k in missing], "pilot")
        records = runner.store.load_records(runner.manifest, include_states=False)
        done = completed_keys(records)

    # Audit a diagonal spread of panel centers ((0,0),(1,1),...,(7,7)) so the
    # initial credibility-floor pairs sample both axes of the panel family.
    centers = [k for k in pkeys if (k.om_num, k.om_den) == (3, 2)]
    audit_keys = [k for k in centers[::10] if k not in audit_done and k in done]
    if audit_keys:
        runner.run_batches(
            [Batch(keys=[k], tier="audit", save_traj=True) for k in audit_keys],
            "pilot-audit")

    gate = {"enabled": False, "reason": "batching disabled (--batch-size 1)"}
    pilot_path = os.path.join(runner.store.reports_dir, "pilot.json")
    prior_gate = None
    if os.path.exists(pilot_path):
        try:
            with open(pilot_path) as fh:
                prior_gate = json.load(fh).get("packing_gate")
        except (OSError, json.JSONDecodeError):
            prior_gate = None
    if runner.args.batch_size > 1:
        norm_ok = json.load(open(os.path.join(runner.store.reports_dir,
                                              "verification.json")))["error_norm_verified"]
        if not norm_ok:
            gate = {"enabled": False,
                    "reason": "SciPy error-norm seam unverified; one point per solve"}
        elif prior_gate and "max_state_dev" in prior_gate:
            gate = prior_gate     # measured once per store; deterministic
        elif panels is not None and PACK_GATE_PANEL not in panels:
            gate = {"enabled": False,
                    "reason": f"--panels excludes gate panel {PACK_GATE_PANEL}"}
        elif not runner.stop_requested:
            gate = run_packing_gate(runner, done)

    records = runner.store.load_records(runner.manifest, include_states=False)
    pairs = audit_pairs(records)
    per_panel = {f"{p[0]},{p[1]}": float(np.median(v))
                 for p, v in sorted(runner.cost.samples.items())}
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_pilot_points": len(pkeys),
        "packing_gate": gate,
        "per_panel_median_point_s": per_panel,
        "inflation_p90": runner.cost.inflation_p90(),
        "audit_pairs": len(pairs),
        "eta_hours_full_levels": {
            str(LEVEL_SIZES[lv]): runner.cost.eta_seconds(
                all_keys(lv), runner.args.workers) / 3600.0
            for lv in range(len(LEVEL_SIZES))
        },
        "omega_1013_over_2pi_MHz": float(runner.manifest["omega_1013_rad_s"]) / TAU / 1e6,
    }
    path = os.path.join(runner.store.reports_dir, "pilot.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[pilot] packing gate: {gate}", flush=True)
    print(f"[pilot] full-level ETA estimates (h): "
          f"{report['eta_hours_full_levels']}", flush=True)
    return report


# ── Stage orchestration and CLI commands ─────────────────────────────────────


def _credibility_floor(records: list[PointRecord],
                       floor_min: float = 1e-12) -> tuple[float, dict]:
    """vmin = max(1e-12, 10 * P95(|L_prod - L_audit|)); documented fallback."""
    pairs = audit_pairs(records)
    if len(pairs) >= 8:
        diffs = np.abs([p - a for _, p, a in pairs])
        vmin = float(max(floor_min, 10.0 * np.percentile(diffs, 95)))
        info = {"rule": "10*P95(|L_prod - L_audit|)", "n_pairs": len(pairs),
                "p95_abs_diff": float(np.percentile(diffs, 95)), "vmin": vmin,
                "fallback": False}
    else:
        vmin = max(floor_min, 1e-11)
        info = {"rule": "fallback 1e-11 (fewer than 8 audit pairs)",
                "n_pairs": len(pairs), "vmin": vmin, "fallback": True}
    return vmin, info


def _run_level(runner: Runner, level: int, done: set[PointKey],
               batch_size: int, rerun_failures: bool,
               failed: set[PointKey],
               panels: set[tuple[int, int]] | None = None) -> None:
    missing = [k for k in _filter_panels(all_keys(level), panels)
               if k not in done and (rerun_failures or k not in failed)]
    if not missing:
        print(f"[level {LEVEL_SIZES[level]}] complete", flush=True)
        return
    batches = group_batches(missing, batch_size)
    runner.run_batches(batches, f"level-{LEVEL_SIZES[level]}")


def _parse_panels(args) -> set[tuple[int, int]] | None:
    """Optional panel restriction ``--panels "di,ti;di,ti"`` (smoke tests, reruns)."""
    spec = getattr(args, "panels", None)
    if not spec:
        return None
    panels = set()
    for part in spec.split(";"):
        di, ti = (int(v) for v in part.split(","))
        if not (0 <= di < len(DELTA_E_GHZ) and 0 <= ti < len(T_GATE_US)):
            raise SystemExit(f"--panels: ({di},{ti}) out of range")
        panels.add((di, ti))
    return panels


def _filter_panels(keys: Iterable[PointKey],
                   panels: set[tuple[int, int]] | None) -> list[PointKey]:
    return [k for k in keys if panels is None or k.panel in panels]


def cmd_pilot(args) -> None:
    store, manifest, cfg, ops, omega_1013, checks = setup_run(args)
    cost = CostModel(cfg)
    _feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    try:
        stage_pilot(runner, _parse_panels(args))
    except KeyboardInterrupt:
        print("[pilot] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status("pilot-aborted" if runner.aborted else "pilot-done")
        runner.shutdown()
    if not runner.aborted:
        export_store(store)


def _feed_cost_model(cost: CostModel, records: list[PointRecord]) -> None:
    for r in records:
        if r.status == "ok" and r.tier == "production":
            cost.observe(r.key.panel, r.runtime_s)


def _effective_batch_size(store: Store, args) -> int:
    """Requested batch size, gated by the recorded pilot packing acceptance."""
    if args.batch_size <= 1:
        return 1
    path = os.path.join(store.reports_dir, "pilot.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                gate = json.load(fh).get("packing_gate", {})
            if gate.get("enabled"):
                return args.batch_size
        except (OSError, json.JSONDecodeError):
            pass
    print("[run] packing gate not passed/recorded; using one point per solve",
          flush=True)
    return 1


def cmd_run(args) -> None:
    store, manifest, cfg, ops, omega_1013, checks = setup_run(args)
    panels = _parse_panels(args)
    records = store.load_records(manifest, include_states=False)
    done = completed_keys(records)
    failed = {r.key for r in records if r.status != "ok"} - done
    cost = CostModel(cfg)
    _feed_cost_model(cost, records)

    max_level = LEVEL_FROM_SIZE[int(args.target_level)]

    if args.dry_run:
        n_panels = len(panels) if panels is not None else len(all_panels())
        print(f"panels: {n_panels}  "
              f"(rows {list(cfg.delta_e_ghz)} GHz x cols {list(cfg.t_gate_us)} us)")
        for lv in range(len(LEVEL_SIZES)):
            keys = _filter_panels(all_keys(lv), panels)
            miss = [k for k in keys if k not in done]
            eta_h = cost.eta_seconds(miss, args.workers) / 3600.0
            eta_txt = (f"{eta_h:8.2f} h" if cost.samples else "unmeasured")
            print(f"level {LEVEL_SIZES[lv]:>2}x{LEVEL_SIZES[lv]:<2}: "
                  f"{len(keys) - len(miss):>6}/{len(keys):>6} done, "
                  f"{len(miss):>6} missing, predicted ETA {eta_txt} "
                  f"@ {args.workers} workers")
        pkeys = _filter_panels(pilot_keys(), panels)
        print(f"pilot: {sum(1 for k in pkeys if k in done)}/{len(pkeys)} done")
        print(f"failed points on record: {len(failed)}"
              + (" (will retry: --rerun-failures)" if args.rerun_failures else ""))
        return

    runner = Runner(store, manifest, cfg, args, cost)
    # Single-pass: every production level solve writes the coherent chunk AND the
    # scattering-budget records in one step.  Gammas are keyed by panel row (this
    # model's decay rates are row-independent), and already-scattered keys are
    # skipped at write time so a resumed run never duplicates a scatter record.
    runner.gammas = _per_panel_gammas(cfg, checks["decay_rates_rad_s"])
    runner.scatter_done = {r["key"] for r in store.load_scatter_records(manifest)
                           if r["status"] == "ok"}
    try:
        stage_pilot(runner, panels)
        # The trajectory-equivalence gate is a per-store setup step: it needs a
        # stored production trajectory (the pilot just produced one), runs once,
        # and is recorded/skipped thereafter.  Only enable the merged scatter
        # writes once it passes; otherwise the run still produces coherent data.
        gate = _ensure_scatter_gate(runner, store)
        if gate.get("ok"):
            runner.write_both_series = True
            print("[run] single-pass scatter enabled (equivalence gate ok)",
                  flush=True)
        else:
            print(f"[run] scatter-equivalence gate not ok "
                  f"({gate.get('reason')}); writing the coherent series only",
                  flush=True)
        batch_size = _effective_batch_size(store, args)
        print(f"[run] effective batch size: {batch_size}", flush=True)

        for level in range(len(LEVEL_SIZES)):
            if runner.stop_requested or level > max_level:
                break
            records = store.load_records(manifest, include_states=False)
            done = completed_keys(records)
            _run_level(runner, level, done, batch_size, args.rerun_failures,
                       failed, panels)
    except KeyboardInterrupt:
        print("[run] hard abort; in-flight batches were discarded (their points "
              "resume on the next run)", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status("run-aborted" if runner.aborted else "run-done")
        runner.shutdown()

    if not runner.aborted:
        export_store(store)
        write_summary_reports(store)
        print("[run] done; exports refreshed", flush=True)


def cmd_audit(args) -> None:
    store, manifest, cfg, ops, omega_1013, checks = setup_run(args)
    records = store.load_records(manifest, include_states=False)
    prod = sorted(completed_keys(records, "production"))
    audited = completed_keys(records, "audit")
    targets: list[PointKey] = []

    if args.audit_point:
        by_id = {k.id(): k for k in prod}
        if args.audit_point not in by_id:
            raise SystemExit(f"--audit-point {args.audit_point}: no successful "
                             "production record with that id")
        targets.append(by_id[args.audit_point])
    elif args.candidates:
        best = best_records(records)
        ranked = sorted((r.max_leakage, k) for k, r in best.items()
                        if r.tier == "production")
        targets = [k for _, k in ranked[:args.candidates] if k not in audited]
    else:
        pool = [k for k in prod if k not in audited]
        rng = np.random.default_rng(args.seed)
        n = min(args.n_points, len(pool))
        if n:
            targets = [pool[i] for i in
                       sorted(rng.choice(len(pool), size=n, replace=False))]

    if not targets:
        print("[audit] nothing to audit")
        return
    cost = CostModel(cfg)
    _feed_cost_model(cost, records)
    runner = Runner(store, manifest, cfg, args, cost)
    try:
        runner.run_batches(
            [Batch(keys=[k], tier="audit", save_traj=True) for k in targets],
            "audit", enforce_deadline=False)
    except KeyboardInterrupt:
        print("[audit] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.shutdown()
    if not runner.aborted:
        write_summary_reports(store)


def _ensure_scatter_gate(runner: Runner, store: Store) -> dict:
    """Run the trajectory-equivalence gate once per store (shared by run/scatter).

    Executed only when ``reports/scatter_gate.json`` is absent or not ok; the
    result is recorded there and a passed record is returned as-is on the next
    contact.  (The live a3.0 stores already carry a passed record, so merged code
    skips the gate on them; a store without one runs it once against its stored
    pilot trajectory.)
    """
    path = os.path.join(store.reports_dir, "scatter_gate.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                prev = json.load(fh)
            if prev.get("ok"):
                return prev
        except (OSError, json.JSONDecodeError):
            pass
    gate = _scatter_equivalence_gate(runner, store)
    with open(path + ".tmp", "w") as fh:
        json.dump(gate, fh, indent=2)
    os.replace(path + ".tmp", path)
    return gate


def _scatter_equivalence_gate(runner: Runner, store: Store) -> dict:
    """Validate the scatter pipeline against one stored production trajectory.

    Re-solves the cheapest trajectory point with the scatter path and compares
    its in-worker integrals against a reference computed with *independent*
    arithmetic: weights from the diagonal of the repository's
    ``build_occ_operator`` and decay rates re-read from a freshly built system's
    metadata — so a bug in the shared weight/Gamma plumbing cannot cancel out.
    """
    best_file, best_key, best_t = None, None, float("inf")
    for name in sorted(os.listdir(store.traj_dir)):
        if not (name.startswith("traj_") and name.endswith(".npz")):
            continue
        with np.load(os.path.join(store.traj_dir, name), allow_pickle=False) as d:
            if str(d["tier"][0]) != "production":
                continue
            key = store.arrays_to_keys(d)[0]
        t_gate = runner.cfg.t_gate_us[key.t_idx]
        if t_gate < best_t:
            best_file, best_key, best_t = name, key, t_gate
    if best_file is None:
        return {"ok": False, "reason": "no production trajectory to validate against"}

    with np.load(os.path.join(store.traj_dir, best_file), allow_pickle=False) as d:
        times, states = np.array(d["times"]), np.array(d["states"])

    # Independent reference leg (deliberately NOT scattering_integrals/gammas):
    # weights from the repository occupation operators, rates re-read from a
    # freshly built system, plain trapezoid.
    from ryd_gate.core.operators import build_occ_operator

    ref_sys = build_system(runner.cfg,
                           runner.cfg.delta_e_ghz[best_key.delta_idx] * 1e9)
    _ref_rates_raw = ref_sys.level_structure.decay_rates_per_s
    ref_rates = {"p_mid": float(_ref_rates_raw["e1"]["total"]),
                 "p_ryd": float(_ref_rates_raw["r"]["total"]),
                 "p_r_garb": float(_ref_rates_raw["r_garb"]["total"])}
    ref_levels = {"p_mid": (2, 3, 4), "p_ryd": (5,), "p_r_garb": (6,)}
    pops = np.abs(states) ** 2                       # (n_t, 4, dim)
    ref = {}
    for ch, levels in ref_levels.items():
        w = np.real(np.diag(sum(build_occ_operator(lv, 7) for lv in levels)))
        ref[ch] = ref_rates[ch] * np.trapezoid(pops @ w, times, axis=0)

    out = _worker_run_batch(runner._spec(Batch(keys=[best_key], scatter=True)))
    if not out.get("ok"):
        return {"ok": False, "reason": f"gate solve failed: {out.get('message')}"}
    dev = max(float(np.max(np.abs(out["scatter"][ch][0] - ref[ch])))
              for ch in SCATTER_CHANNELS)
    return {"ok": dev < 1e-8, "point_id": best_key.id(), "trajectory": best_file,
            "max_abs_dev": dev, "tol": 1e-8}


def cmd_scatter(args) -> None:
    """Supplemental scattering-budget pass: additive only (scatter/ series)."""
    store, manifest, cfg, ops, omega_1013, checks = setup_run(args)
    level = LEVEL_FROM_SIZE[int(args.level)]
    panels = _parse_panels(args)

    done_scatter = {r["key"] for r in store.load_scatter_records(manifest)
                    if r["status"] == "ok"}
    missing = [k for k in _filter_panels(all_keys(level), panels)
               if k not in done_scatter]
    print(f"[scatter] level {args.level}: {len(missing)} points to compute "
          f"({len(done_scatter)} already stored)", flush=True)
    if not missing:
        return

    cost = CostModel(cfg)
    _feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    runner.gammas = _per_panel_gammas(cfg, checks["decay_rates_rad_s"])
    gate_failed = False
    try:
        gate = _ensure_scatter_gate(runner, store)
        print(f"[scatter] trajectory-equivalence gate: {gate}", flush=True)
        if not gate.get("ok"):
            gate_failed = True
            raise SystemExit("[scatter] equivalence gate failed; not running")

        batch_size = _effective_batch_size(store, args)
        batches = group_batches(missing, batch_size)
        for b in batches:
            b.scatter = True
        runner.run_batches(batches, f"scatter-{args.level}")
    except KeyboardInterrupt:
        print("[scatter] hard abort", flush=True)
    finally:
        if gate_failed:
            # a refused run dispatched nothing: keep the previous failure report
            runner.write_status(f"scatter-{args.level}-gate-failed")
        else:
            runner.write_failure_report()
            runner.write_status(
                f"scatter-{args.level}-aborted" if runner.aborted
                else f"scatter-{args.level}-done")
        runner.shutdown()


def write_summary_reports(store: Store) -> None:
    """Regenerate reports/audit_summary.json and reports/candidates.json."""
    manifest = store.load_manifest()
    records = store.load_records(manifest, include_states=False)
    pairs = audit_pairs(records)
    vmin, floor_info = _credibility_floor(records)
    diffs = np.abs([p - a for _, p, a in pairs]) if pairs else np.zeros(0)
    audit_summary = {
        "n_pairs": len(pairs),
        "max_abs_leakage_diff": float(diffs.max()) if diffs.size else None,
        "p95_abs_leakage_diff": float(np.percentile(diffs, 95)) if diffs.size else None,
        "credibility_floor": floor_info,
        "worst_pairs": [
            {"point_id": k.id(), "L_production": p, "L_audit": a,
             "abs_diff": abs(p - a)}
            for k, p, a in sorted(pairs, key=lambda t: -abs(t[1] - t[2]))[:10]
        ],
    }
    path = os.path.join(store.reports_dir, "audit_summary.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(audit_summary, fh, indent=2)
    os.replace(path + ".tmp", path)

    best = best_records(records)
    cand: dict[str, dict] = {}
    for k, r in best.items():
        pk = f"{k.delta_idx},{k.t_idx}"
        if pk not in cand or r.max_leakage < cand[pk]["max_leakage"]:
            cand[pk] = {"point_id": k.id(), "max_leakage": r.max_leakage,
                        "omega_420_mhz": float(k.omega_mhz()),
                        "dsweep_mhz": float(k.dsweep_mhz()),
                        "worst_input": r.worst_input, "tier": r.tier}
    path = os.path.join(store.reports_dir, "candidates.json")
    with open(path + ".tmp", "w") as fh:
        json.dump({"note": "per-panel minima over exact ODE nodes only",
                   "panels": cand}, fh, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def cmd_status(args) -> None:
    store = Store(args.output)
    manifest = store.load_manifest()
    if manifest is None:
        print(f"no manifest under {store.root} (scan not initialized)")
        return
    records = store.load_records(manifest, include_states=False)
    done = completed_keys(records)
    failed = {r.key for r in records if r.status != "ok"} - done
    print(f"scan {manifest['scan_uuid'][:12]}  created {manifest['created_at']}")
    print(f"Omega_1013/2pi = {manifest['omega_1013_over_2pi_MHz']:.6f} MHz  "
          f"git {manifest['git']['commit'][:10]}"
          f"{' (dirty)' if manifest['git']['dirty'] else ''}")
    for lv, size in enumerate(LEVEL_SIZES):
        keys = all_keys(lv)
        n_done = sum(1 for k in keys if k in done)
        print(f"level {size:>2}x{size:<2}: {n_done:>6}/{len(keys):>6} nodes complete")
    pkeys = pilot_keys()
    print(f"pilot: {sum(1 for k in pkeys if k in done)}/{len(pkeys)} done")
    n_audit = len(completed_keys(records, 'audit'))
    pairs = audit_pairs(records)
    print(f"records: {len(records)} rows, {len(done)} unique ok points, "
          f"{n_audit} audit points, {len(pairs)} audit pairs, {len(failed)} failed")
    ok_prod = [r for r in records if r.status == "ok" and r.tier == "production"]
    if ok_prod:
        rt = np.asarray([r.runtime_s for r in ok_prod])
        print(f"per-point runtime (production): median {np.median(rt):.1f} s, "
              f"P90 {np.percentile(rt, 90):.1f} s, total {rt.sum() / 3600:.2f} core-h")
    status_path = os.path.join(store.reports_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path) as fh:
            print(f"last run status: {json.load(fh)}")


def cmd_export(args) -> None:
    store = Store(args.output)
    merged, csv_path = export_store(store)
    write_summary_reports(store)
    print(f"exports: {merged}\n         {csv_path}")


# ── Plotting ─────────────────────────────────────────────────────────────────
#
# Rasters are *visualization only*: piecewise-linear Delaunay interpolation of
# log10(leakage) over the exact nodes, never extrapolated outside their convex
# hull, with exact-node markers overlaid; every node whose axis-neighbor
# leave-one-out residual exceeds 0.2 dex is hatched (marking its surrounding
# cells as uncertain).  One global LogNorm/colorbar; values below the
# audit-derived credibility floor are shown as "below floor".

PLOT_LOO_MASK_DEX = 0.2
PLOT_RASTER_N = 81


def _panel_plot_data(values: dict[PointKey, float], panel: tuple[int, int],
                     vmin: float):
    """(x_mhz, y_mhz, z_log10) arrays of one panel's exact nodes, or None."""
    pts = [(float(k.omega_mhz()), float(k.dsweep_mhz()), v)
           for k, v in values.items() if k.panel == panel]
    if not pts:
        return None
    pts.sort(key=lambda t: (t[0], t[1]))
    x = np.asarray([p[0] for p in pts])
    y = np.asarray([p[1] for p in pts])
    z = np.log10(np.maximum([p[2] for p in pts], vmin / 10.0))
    return x, y, z


def _plot_metric_values(store: Store, manifest: dict, records, metric: str):
    """(values, vmin, vmax, colorbar_label) for a plot metric.

    ``max_leakage`` reads the coherent-leakage records (audit-derived floor);
    the ``p_*`` metrics read the supplemental scatter series. ``total_error``
    adds coherent leakage and every scattering contribution per logical input
    before selecting the worst input.
    """
    if metric == "max_leakage":
        best = best_records(records)
        values = {k: r.max_leakage for k, r in best.items()}
        vmin, floor_info = _credibility_floor(records)
        label = ("terminal max leakage  "
                 f"(floor {vmin:.1e}: "
                 f"{'audit-derived' if not floor_info['fallback'] else 'fallback'};"
                 " values at floor are below the numerical credibility floor)")
        return values, vmin, 1.0, label
    coherent = best_records(records) if metric == "total_error" else {}
    rows = [r for r in store.load_scatter_records(manifest) if r["status"] == "ok"]
    if not rows:
        raise SystemExit(f"no scatter records for --metric {metric}; "
                         "run the `scatter` subcommand first")
    per_key: dict[PointKey, tuple[float, float]] = {}
    for r in rows:
        scattering = r["p_mid"] + r["p_ryd"] + r["p_r_garb"]
        if metric == "total_error":
            if r["key"] not in coherent:
                continue
            v = float(np.max(coherent[r["key"]].leakage + scattering))
        elif metric == "p_loss_total":
            v = float(np.max(scattering))
        else:
            v = float(np.max(r[metric]))
        cur = per_key.get(r["key"])
        if cur is None or r["rtol"] < cur[1]:
            per_key[r["key"]] = (v, r["rtol"])
    values = {k: v for k, (v, _) in per_key.items()}
    if not values:
        raise SystemExit(f"no overlapping coherent and scatter records for --metric {metric}")
    pos = [v for v in values.values() if v > 0]
    vmin = max(1e-12, min(pos)) if pos else 1e-12
    vmax = max(max(values.values()), vmin * 10)
    if metric == "total_error":
        label = ("worst-input total error budget (terminal coherent leakage + "
                 "first-order scattering)")
    else:
        label = (f"worst-input {metric} (scattering-rate integral, "
                 "trapezoid over 301 samples)")
    return values, vmin, vmax, label


def _holdout_residuals(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Axis-neighbor leave-one-out residual (dex) for EVERY node.

    Each node is re-estimated by linear interpolation between its nearest present
    neighbors along each grid axis (holding it out); the residual is the worse of
    the two axes.  O(n log n), so no node is ever skipped — a node lacking both
    neighbors on both axes (panel edges) gets 0.
    """
    n = x.size
    resid = np.zeros(n)

    def _along(primary: np.ndarray, secondary: np.ndarray) -> None:
        for line in np.unique(secondary):
            idx = np.where(secondary == line)[0]
            if idx.size < 3:
                continue
            order = idx[np.argsort(primary[idx])]
            p, v = primary[order], z[order]
            est = v[:-2] + (v[2:] - v[:-2]) * (p[1:-1] - p[:-2]) / (p[2:] - p[:-2])
            np.maximum.at(resid, order[1:-1], np.abs(v[1:-1] - est))

    _along(x, y)
    _along(y, x)
    return resid


def _draw_panel(ax, x, y, z, vmin, vmax, cmap, veil: bool = True):
    """One panel: interpolated raster + uncertainty veil + nodes + hardware line.

    Regions whose nearest node has an axis-holdout LOO residual above
    ``PLOT_LOO_MASK_DEX`` are masked with a translucent white veil (the spec's
    "hatch or mask" rule) — a wash reads cleanly at any node density, where
    per-node hatched markers drown the map once grids reach 13x13/25x25.
    ``veil=False`` omits the overlay (raster is then pure interpolation —
    remember it is visualization only).
    """
    from matplotlib.colors import ListedColormap, LogNorm
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    xg = np.linspace(x.min(), x.max(), PLOT_RASTER_N)
    yg = np.linspace(y.min(), y.max(), PLOT_RASTER_N)
    XX, YY = np.meshgrid(xg, yg)
    resid = _holdout_residuals(x, y, z)
    bad = (resid > PLOT_LOO_MASK_DEX) if veil else np.zeros(x.size, dtype=bool)
    if x.size >= 4 and np.unique(x).size > 1 and np.unique(y).size > 1:
        interp = LinearNDInterpolator(np.column_stack([x, y]), z)
        ZZ = interp(XX, YY)                     # NaN outside the convex hull
        # rasterized: in vector (PDF) output each mesh quad is otherwise a
        # separate filled path, and viewers antialias the quad boundaries into
        # hairline white seams; rasterizing embeds the color field as one image
        # (axes/markers/text stay vector) and shrinks the file dramatically.
        mesh = ax.pcolormesh(XX, YY, np.ma.masked_invalid(10.0 ** ZZ),
                             cmap=cmap, norm=norm, shading="nearest",
                             rasterized=True)
        if np.any(bad):
            near_bad = NearestNDInterpolator(
                np.column_stack([x, y]), bad.astype(float))(XX, YY)
            veil = np.ma.masked_where(
                (near_bad < 0.5) | ~np.isfinite(ZZ), np.ones_like(near_bad))
            ax.pcolormesh(XX, YY, veil, cmap=ListedColormap([(1, 1, 1, 0.45)]),
                          vmin=0, vmax=1, shading="nearest", rasterized=True)
    else:
        mesh = ax.scatter(x, y, c=np.maximum(10.0 ** z, vmin), cmap=cmap,
                          norm=norm, s=14)
        if np.any(bad):
            ax.scatter(x[bad], y[bad], marker="s", s=40, facecolors="none",
                       edgecolors="w", linewidths=0.7)
    ax.plot(x, y, ".", color="k", ms=1.2, alpha=0.4)
    ax.axhline(DSWEEP_HW_LIMIT_MHZ, color="c", ls="--", lw=1.0, alpha=0.9)
    return mesh


def cmd_plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = Store(args.output)
    manifest = store.load_manifest()
    if manifest is None:
        raise SystemExit(f"no manifest under {store.root}")
    store.ensure_dirs()
    records = store.load_records(manifest, include_states=False)
    values, vmin, vmax, cb_label = _plot_metric_values(
        store, manifest, records, args.metric)
    if not values:
        raise SystemExit("no successful records to plot")
    cmap = "magma_r"
    de = manifest["axes"]["delta_e_ghz"]
    tg = manifest["axes"]["t_gate_us"]

    n_rows, n_cols = len(de), len(tg)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols + 1.6, 1.9 * n_rows + 1.2),
                             sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for di in range(n_rows):
        for ti in range(n_cols):
            ax = axes[di][ti]
            data = _panel_plot_data(values, (di, ti), vmin)
            if data is None:
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.4")
            else:
                mesh = _draw_panel(ax, *data, vmin, vmax, cmap,
                                   veil=args.veil) or mesh
            if di == 0:
                ax.set_title(f"T = {tg[ti]:g} us", fontsize=9)
            if di == n_rows - 1:
                ax.set_xlabel(r"$\Omega_{420}/2\pi$ (MHz)", fontsize=8)
            if ti == 0:
                ax.set_ylabel(f"$\\Delta_e/2\\pi$ = {de[di]:g} GHz\n"
                              r"$D_{\rm sweep}/2\pi$ (MHz)", fontsize=8)
            ax.tick_params(labelsize=7)
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, shrink=0.5, pad=0.01)
        cb.solids.set_rasterized(True)  # same PDF hairline-seam fix as the panels
        cb.set_label(cb_label, fontsize=9)
    if args.metric == "max_leakage":
        metric_title = "Coherent terminal leakage"
    elif args.metric == "total_error":
        metric_title = "Total first-order error budget (worst input)"
    else:
        metric_title = f"Scattering budget: {args.metric} (worst input)"
    dynamics_note = ("closed-dynamics trajectory + first-order scattering"
                     if args.metric == "total_error" else "closed dynamics")
    fig.suptitle(
        f"{metric_title}, two-atom rb87_7_mp CZ ({dynamics_note}, "
        "original-frame DOP853; rasters are log-linear interpolation between "
        "exact nodes — dots"
        + ("; white veil: interpolation untrusted, LOO residual > "
           f"{PLOT_LOO_MASK_DEX} dex)" if args.veil else
           "; NO uncertainty veil — raster is visualization only)"), fontsize=11)

    png = os.path.join(store.plots_dir, f"{args.metric}_8x9.png")
    pdf = os.path.join(store.plots_dir, f"{args.metric}_8x9.pdf")
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf, dpi=args.dpi)  # dpi applies to the rasterized mesh layers
    plt.close(fig)
    print(f"plots: {png}\n       {pdf}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _default_output(spacing_um: float) -> str:
    return os.path.join("results", "max_leakage_ode", f"a{spacing_um:.1f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="max_leakage_ode_sweep",
        description=__doc__.split("\n\n")[0],
    )
    sub = p.add_subparsers(dest="command", required=True)

    def int_or_auto(default):
        def parse(value):
            return default if value == "auto" else int(value)
        parse.__name__ = "int-or-auto"
        return parse

    def common(sp, compute: bool = False):
        sp.add_argument("--output", default=None,
                        help="scan store directory (default: "
                             "results/max_leakage_ode/a{spacing:.1f})")
        sp.add_argument("--spacing-um", type=float, default=3.0,
                        help="atom spacing in um (physics-hash relevant; also "
                             "selects the default store directory)")
        if compute:
            # "auto" = the pilot-benchmarked host default (40 of 40 logical CPUs
            # measured ~1.48x the 20-worker throughput) / the acceptance-gated
            # packing size, so the agreed production invocation parses verbatim.
            sp.add_argument("--workers", type=int_or_auto(min(40, os.cpu_count() or 40)),
                            default=40)
            sp.add_argument("--batch-size", type=int_or_auto(48), default=48,
                            help="max points packed per solve (acceptance-gated)")
            sp.add_argument("--point-timeout", type=float, default=3600.0,
                            help="wall-clock timeout per point (scaled by batch size)")
            sp.add_argument("--rtol", type=float, default=1e-9,
                            help="production relative tolerance")
            sp.add_argument("--atol", type=float, default=1e-12,
                            help="production absolute tolerance")
            sp.add_argument("--audit-rtol", type=float, default=1e-10)
            sp.add_argument("--audit-atol", type=float, default=1e-13)
            sp.add_argument("--panels", default=None, metavar="DI,TI[;DI,TI...]",
                            help="restrict to specific panels (smoke tests, reruns)")

    sp = sub.add_parser("status", help="summarize an existing scan store")
    common(sp)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("pilot", help="run the pilot stage only")
    common(sp, compute=True)
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("run", help="resumable staged single-pass scan")
    common(sp, compute=True)
    sp.add_argument("--target-level", default="13",
                    choices=["4", "7", "13", "25"],
                    help="finest grid level to fill (13x13 by default)")
    sp.add_argument("--dry-run", action="store_true",
                    help="print point counts, missing nodes and ETA; no simulation")
    sp.add_argument("--rerun-failures", action="store_true",
                    help="re-dispatch points whose only records are failures")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("audit", help="rerun selected points at audit tolerance")
    common(sp, compute=True)
    sp.add_argument("--n-points", type=int, default=24)
    sp.add_argument("--seed", type=int, default=0,
                    help="deterministic audit sampling seed")
    sp.add_argument("--audit-point", default=None, metavar="POINT_ID",
                    help="audit one specific point id (see exports/points.csv)")
    sp.add_argument("--candidates", type=int, default=0,
                    help="audit the N lowest-leakage exact nodes instead")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("scatter",
                        help="supplemental scattering-budget pass (additive: "
                             "writes only the scatter/ series)")
    common(sp, compute=True)
    sp.add_argument("--level", default="7", choices=["4", "7", "13", "25"],
                    help="grid level to cover (scattering maps are smooth; "
                         "7x7 is usually sufficient)")
    sp.set_defaults(func=cmd_scatter)

    sp = sub.add_parser("export", help="regenerate merged NPZ + CSV + reports")
    common(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("plot", help="render the 8x9 map family")
    common(sp)
    sp.add_argument("--dpi", type=int, default=170)
    sp.add_argument("--no-veil", dest="veil", action="store_false", default=True,
                    help="omit the uncertainty veil (raster is visualization only)")
    sp.add_argument("--metric", default="max_leakage",
                    choices=["max_leakage", "p_mid", "p_ryd", "p_r_garb",
                             "p_loss_total", "total_error"],
                    help="max_leakage from the main scan; p_* from the "
                         "supplemental scatter series; total_error combines both")
    sp.set_defaults(func=cmd_plot)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output is None:
        args.output = _default_output(args.spacing_um)
    args.func(args)


if __name__ == "__main__":
    main()
