#!/usr/bin/env python
"""Authoritative original-frame max-leakage sweep for the two-atom 297 nm single-photon CZ gate.

Scans terminal coherent leakage ``max_s || Q psi_s(T) ||^2`` (Q = projector onto the
nonlogical subspace, s in {00, 01, 10, 11}) over an 8x9 panel family
(rows: Rydberg principal quantum number n in {50, 53, 56, 60, 64, 68, 71, 73},
columns: T in {1..4.5} us).  Every panel scans x = Omega_297,max/2pi in [9, 18] MHz
(direct 297 nm peak physical Rabi) and y = D_sweep/2pi in [2, 30] MHz (half-amplitude
detuning-sweep convention), on progressively nested exact grids
4x4 -> 7x7 -> 13x13 -> 25x25 whose coarse anchors keep the 20 MHz hardware-limit line
an exact node at every level.

Physics is the closed (decay-off, Hermitian) single-photon 297 nm model: the ground
state is driven directly into the Rydberg manifold at 297 nm with no intermediate
state, at 3.0 um / B = 20 G, with a quintic-smoothstep power envelope of ramp
fraction 0.15 (field amplitude sqrt(E)) and the pure cosine detuning-sweep chirp

    chirp(t) = -D cos(2 pi t / T),

whose optical phase is integrated *analytically* (no interpolation).  A single-photon
drive has no intermediate state, so there is no differential AC-Stark shift to
compensate (no Dr - D1 term) and no intermediate-state scattering channel (no p_mid).

The only production solver is original-frame complex128 adaptive DOP853
(production rtol=1e-9/atol=1e-12; audit rtol=1e-10/atol=1e-13), with

  * one precompiled, channel-aggregated Hamiltonian per n row,
  * logical inputs 00/01/11 propagated together as matrix columns (10 obtained by
    the verified atom-swap symmetry), each column solved with its bare logical
    diagonal energy subtracted (exact global-phase shift, restored at T),
  * within-panel multi-point batching guarded by a per-(point, logical-input)-block
    maximum error norm that mirrors the installed SciPy DOP853 estimate exactly
    (acceptance-gated; falls back to one point per solve if unverifiable),
  * stepper restarts at the analytic envelope breakpoints t = 0.15 T and 0.85 T.

Results are append-only NPZ chunks under ``results/max_leakage_297/`` with a
hash-validated manifest; interrupted scans resume without recomputing.  ``run``
is a single pass: every production solve samples the trajectory and writes BOTH
the coherent-leakage chunk and the scattering-budget records, staging pilot ->
full 4x4 -> full 7x7 -> the requested ``--target-level`` (13x13 by default).

Usage
-----
    # default store: results/max_leakage_297/a{spacing:.1f} (spacing default 3.0)
    python scripts/max_leakage_297_sweep.py status
    python scripts/max_leakage_297_sweep.py pilot  --spacing-um 5 --workers 40
    python scripts/max_leakage_297_sweep.py run    --spacing-um 5 --target-level 13
    python scripts/max_leakage_297_sweep.py run    --spacing-um 5 --dry-run
    python scripts/max_leakage_297_sweep.py scatter --spacing-um 5 --level 13
    python scripts/max_leakage_297_sweep.py filter  --spacing-um 5 --level 13
    python scripts/max_leakage_297_sweep.py audit  --spacing-um 5
    python scripts/max_leakage_297_sweep.py export --spacing-um 5
    python scripts/max_leakage_297_sweep.py plot   --spacing-um 5
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
# importable whether the script is run as ``python scripts/max_leakage_297_sweep.py``
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
    cli, PlotSpec, render_panel_grid, credibility_floor as _credibility_floor,
)
from sweeplib.store import _atomic_savez
from sweeplib.runner import _worker_run_batch

TAU = 2.0 * math.pi
SCHEMA_VERSION = 1

# ── Locked scientific configuration ──────────────────────────────────────────

RYD_N = (50, 53, 56, 60, 64, 68, 71, 73)                          # panel rows (Rydberg n)
T_GATE_US = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)          # panel columns

# Nested-axis anchors (MHz, exact rationals).  Deliberately non-uniform so the
# 20 MHz hardware cap is an exact node at every level; do not replace with linspace.
OMEGA297_ANCHORS_MHZ = (Fraction(9), Fraction(12), Fraction(15), Fraction(18))
DSWEEP_ANCHORS_MHZ = (Fraction(2), Fraction(10), Fraction(20), Fraction(30))

# LEVEL_DENS/LEVEL_SIZES/LEVEL_FROM_SIZE are imported from sweeplib.axes.

DSWEEP_HW_LIMIT_MHZ = 20.0         # horizontal reference line in every panel

LOGICAL_INPUTS = ("00", "01", "10", "11")


@dataclass(frozen=True)
class ScanConfig:
    """Immutable physics/scan configuration (the manifest's scientific payload)."""

    spacing_um: float = 3.0
    magnetic_field_G: float = 20.0
    ramp_frac: float = 0.15
    rtol_production: float = 1e-9
    atol_production: float = 1e-12
    rtol_audit: float = 1e-10
    atol_audit: float = 1e-13
    ryd_n: tuple = RYD_N
    t_gate_us: tuple = T_GATE_US
    omega297_anchors_mhz: tuple = tuple(str(a) for a in OMEGA297_ANCHORS_MHZ)
    dsweep_anchors_mhz: tuple = tuple(str(a) for a in DSWEEP_ANCHORS_MHZ)
    credibility_floor_min: float = 1e-12
    interp_space: str = "log10"
    n_eval_trajectory: int = 301

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
# lives in sweeplib.axes; the PointKey type keeps its serialized ``n_idx`` field and
# ``n`` id prefix via the factory below.

PointKey, make_key, panel_keys, all_panels, all_keys = make_pointkey_type(
    panel_field="n_idx", id_prefix="n",
    omega_anchors=OMEGA297_ANCHORS_MHZ, dsweep_anchors=DSWEEP_ANCHORS_MHZ,
    panel_len=len(RYD_N), n_t=len(T_GATE_US),
)


def pilot_keys() -> list[PointKey]:
    """Reusable pilot nodes: all 72 panel centers + 16 factorial extremes, deduped."""
    center = ((3, 2), (3, 2))  # (13.5 MHz, 15 MHz): level-1 node of both axes
    keys = [make_key(di, ti, *center) for di, ti in all_panels()]
    extremes = [
        make_key(di, ti, (om, 1), (dw, 1))
        for di in (0, len(RYD_N) - 1)
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
# The 297 field amplitude is sqrt(E).  The chirp (rad/s) is the pure detuning sweep
#     chirp(t) = -D cos(2 pi t / T)
# and its exact unwrapped integral is
#     phi(t) = -(D T / 2 pi) sin(2 pi s),
# with J(s) the closed-form integral of E still used to shape the envelope.  phi is
# *not* wrapped mod 2 pi.  A single-photon drive has no intermediate state, hence no
# differential AC-Stark shift to compensate (no Dr - D1 term).
# quintic/quintic_antideriv/envelope/envelope_integral are imported from sweeplib.solver.


def chirp_rad_s(t, t_gate: float, d_sweep: float, ramp: float = 0.15):
    """Intended instantaneous chirp (rad/s) at time ``t``; scalar or ndarray."""
    s = np.asarray(t, dtype=float) / t_gate
    return -d_sweep * np.cos(TAU * s)


def phase_rad(t, t_gate: float, d_sweep: float, ramp: float = 0.15):
    """Exact unwrapped optical phase phi(t) = integral_0^t chirp; scalar or ndarray."""
    s = np.asarray(t, dtype=float) / t_gate
    return (-d_sweep * t_gate / TAU) * np.sin(TAU * s)


def pulse_hash() -> str:
    """Behavioral fingerprint of the analytic pulse family.

    Hashes the sampled values of envelope/J/chirp/phase on a fixed probe grid, so
    any edit that changes what the laser actually does — even with identical
    ScanConfig — changes the hash.  Recorded in the manifest and every chunk;
    resume/merge refuses a mismatch (whitespace/comment edits are deliberately
    invisible, unlike a source-text hash).
    """
    s = np.linspace(0.0, 1.0, 257)
    t_gate, d_sweep = 1.7e-6, TAU * 13e6
    probe = np.concatenate([
        np.asarray(envelope(s), dtype=float),
        np.asarray(envelope_integral(s), dtype=float),
        np.asarray(chirp_rad_s(s * t_gate, t_gate, d_sweep), dtype=float),
        np.asarray(phase_rad(s * t_gate, t_gate, d_sweep), dtype=float),
    ])
    return hashlib.sha256(probe.tobytes()).hexdigest()


# ── Model compilation and Hamiltonian aggregation ────────────────────────────
#
# The rb87_297_clock_4 two-atom model is built and compiled ONCE per Rydberg-n row
# in the parent process; workers inherit the immutable matrices copy-on-write over
# fork.  The primitive 297 drive channels (target |1>-|r> plus the garbage branch)
# are aggregated into a single fixed forward block B; with X = B + B^dag and
# Y = i (B - B^dag) a complex laser coefficient c = a + ib contributes a X + b Y,
# so the RHS needs no per-channel loop.  The grouped Hamiltonian is verified against
# the repository compiler to complex128 precision (``hamiltonian_equivalence_error``)
# before any production run.


def _dense(operator) -> np.ndarray:
    if hasattr(operator, "toarray"):
        return np.asarray(operator.toarray())
    return np.asarray(operator)


@dataclass
class PanelOperators:
    """Immutable aggregated operators for one Rydberg-n row (T-independent)."""

    ryd_n: int
    h_static_diag: np.ndarray           # (16,) float64 — verified real diagonal
    x297: np.ndarray                    # (16, 16) complex128, Hermitian
    y297: np.ndarray
    amplitude_scale: float
    logical_indices: np.ndarray         # basis indices of |00>,|01>,|10>,|11>
    swap_perm: np.ndarray               # atom-swap permutation of the 16 basis states
    swap_symmetric: bool                # swap commutes with every H constituent

    def hash_bytes(self) -> bytes:
        h = hashlib.sha256()
        for a in (self.h_static_diag, self.x297, self.y297,
                  np.asarray([float(self.ryd_n), self.amplitude_scale])):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.digest()


def build_system(cfg: ScanConfig, ryd_n: int):
    """Two-atom rb87_297_clock_4 system with a placeholder 297 protocol bound.

    The placeholder protocol only supplies the channel set for IR compilation; the
    production kernel computes drive coefficients analytically.
    """
    import ryd_gate as rg
    from ryd_gate.protocols import Direct297CZProtocol
    from ryd_gate.lattice import Register

    proto = Direct297CZProtocol(
        t_gate_s=1e-6, omega_297_max_rad_s=1.0,
        envelope_297=lambda t: 1.0, phase_297_rad=lambda t: 0.0,
    )
    return rg.RydbergSystem(
        level_structure=rg.level_structure(
            "rb87_297_clock_4", ryd_level=int(ryd_n),
            magnetic_field_G=cfg.magnetic_field_G),
        register=Register.chain(2, spacing_um=cfg.spacing_um),
        protocol=proto,
    )


def _swap_permutation(local_dim: int = 7) -> np.ndarray:
    idx = np.arange(local_dim * local_dim)
    a, b = divmod(idx, local_dim)
    return b * local_dim + a


def aggregate_operators(system, ryd_n: int) -> PanelOperators:
    """Compile the system and aggregate its 297 channels into fixed dense blocks."""
    from ryd_gate.backends.exact.compiler import compile_exact
    from ryd_gate.core.states import product_index

    ham, _t_gate = compile_exact(system, hamiltonian_format="dense")

    # ham.h_static carries the atomic diagonal + the Rydberg pair interaction.  The
    # single-photon 297 protocol supplies no constant diagonal drive channel, but
    # keep the diag-folding loop general: fold any diagonal channel into the static
    # diagonal (no intermediate detuning is retained here), then split off the
    # off-diagonal laser channels.
    h_static = np.array(ham.h_static, dtype=np.complex128)
    laser_channels = []
    for ch in ham.channels:
        if ch.is_diag:
            h_static += complex(ch.coeff(0.0)) * _dense(ch.sum_op)
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

    ratios = {"297": {}}
    for leg in system.level_structure._laser_legs:
        if leg.group in ratios:
            ratios[leg.group][leg.channel] = leg.factor
    ops = {ch._channel: _dense(ch.sum_op) for ch in laser_channels}
    missing = set(ratios["297"]) - set(ops)
    if missing:
        raise RuntimeError(f"drive channels missing from compiled Hamiltonian: {sorted(missing)}")
    for ch in laser_channels:
        if ch.sum_op_hc is None:
            raise RuntimeError(f"drive channel {ch._channel} lacks the h.c. leg")

    b297 = sum(ratios["297"][ch] * ops[ch] for ch in ratios["297"])
    x297 = b297 + b297.conj().T
    y297 = 1j * (b297 - b297.conj().T)

    logical = np.asarray(
        [product_index(list(s), system._basis) for s in LOGICAL_INPUTS]
    )
    perm = _swap_permutation(system._basis.local_dim)

    def _swap_invariant(mat: np.ndarray) -> bool:
        return bool(np.array_equal(mat[np.ix_(perm, perm)], mat))

    swap_ok = (
        bool(np.array_equal(diag.real[perm], diag.real))
        and all(_swap_invariant(m) for m in (x297, y297))
    )

    return PanelOperators(
        ryd_n=int(ryd_n),
        h_static_diag=np.ascontiguousarray(diag.real),
        x297=x297, y297=y297,
        amplitude_scale=1.0,
        logical_indices=logical,
        swap_perm=perm,
        swap_symmetric=swap_ok,
    )


def hamiltonian_equivalence_error(
    system,
    ops: PanelOperators,
    t_gate: float,
    omega_297: float,
    d_sweep: float,
    times: np.ndarray,
    ramp: float = 0.15,
) -> float:
    """Max |H_grouped - H_compiled| over ``times`` for one concrete 297 pulse.

    Builds the repository Hamiltonian directly from a freshly compiled IR with the
    *same* analytic pulse bound as a real Direct297CZProtocol, then compares against
    the aggregated evaluation used by the production kernel.
    """
    from ryd_gate.backends.exact.compiler import compile_exact
    from ryd_gate.protocols import Direct297CZProtocol

    proto = Direct297CZProtocol(
        t_gate_s=t_gate, omega_297_max_rad_s=omega_297,
        envelope_297=lambda t: float(np.sqrt(envelope(t / t_gate, ramp))),
        phase_297_rad=lambda t: float(phase_rad(t, t_gate, d_sweep, ramp)),
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
        phi = float(phase_rad(t, t_gate, d_sweep, ramp))
        c297 = ops.amplitude_scale * omega_297 * amp * np.exp(-1j * phi)
        h_grp = (np.diag(ops.h_static_diag.astype(np.complex128))
                 + c297.real * ops.x297 + c297.imag * ops.y297)
        worst = max(worst, float(np.max(np.abs(h_ref - h_grp))))
    return worst


# ── Batched original-frame integration kernel ────────────────────────────────
#
# The generic block-max DOP853 kernel (segmented restarts, per-column global-phase
# shifts, atom-swap reconstruction, t_eval trajectory sampling) and BatchResult live
# in sweeplib.solver; this script injects the single-drive (297, no Stark) RHS for
# the rb87_297_clock_4 model.  Each column is solved with its bare logical diagonal
# energy subtracted (via cols["shift"]); sweeplib restores the exact global phase.


def _297_rhs_factory(ops, cols, t_gate, ramp):
    """Single-drive (297, Stark-free) RHS consumed by sweeplib.integrate_batch.

    A single-photon drive has no intermediate state, so the drive coefficient is
    c297 = amplitude_scale * amp * Omega_297 * exp(-i phi) with the pure detuning
    sweep phi = -(D T / 2 pi) sin(2 pi s) — no differential AC-Stark term.
    """
    om_cols = cols["omega_297"]
    dsw_cols = cols["d_sweep"]
    diag_row = ops.h_static_diag[None, :] - cols["shift"][:, None]   # (n_cols, dim) real
    x297_t = np.ascontiguousarray(ops.x297.T)
    y297_t = np.ascontiguousarray(ops.y297.T)
    ascale = ops.amplitude_scale
    sin_coef = -t_gate / TAU
    n_cols, dim = diag_row.shape

    def rhs(t, y):
        s = t / t_gate
        env = envelope(s, ramp)
        amp = math.sqrt(float(env))
        phi = (sin_coef * math.sin(TAU * s)) * dsw_cols
        c297 = (ascale * amp) * om_cols * np.exp(-1j * phi)
        ym = y.reshape(n_cols, dim)
        out = diag_row * ym
        out += c297.real[:, None] * (ym @ x297_t)
        out += c297.imag[:, None] * (ym @ y297_t)
        return (-1j * out).ravel()

    return rhs


def integrate_batch(
    ops: PanelOperators,
    t_gate: float,
    omega_297: np.ndarray,
    d_sweep: np.ndarray,
    rtol: float,
    atol: float,
    ramp: float = 0.15,
    use_swap: bool | None = None,
    use_shifts: bool = True,
    segmented: bool = True,
    t_eval: np.ndarray | None = None,
) -> BatchResult:
    """Propagate all logical inputs of ``len(omega_297)`` panel points together.

    Columns are (point-major) the logical inputs 00/01/11 — 10 is reconstructed by
    the atom-swap permutation when the verified symmetry holds, else all four are
    propagated.  Thin wrapper that supplies the single-drive 297 RHS to the shared
    sweeplib kernel.
    """
    omega_297 = np.asarray(omega_297, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    if omega_297.shape != d_sweep.shape or omega_297.ndim != 1:
        raise ValueError("omega_297 and d_sweep must be equal-length 1-D arrays")
    if use_swap is None:
        use_swap = ops.swap_symmetric
    state_labels = ("00", "01", "11") if use_swap else LOGICAL_INPUTS
    return sweeplib.integrate_batch(
        ops, t_gate,
        {"omega_297": omega_297, "d_sweep": d_sweep},
        state_labels,
        rhs_factory=_297_rhs_factory,
        dim=ops.h_static_diag.size,
        rtol=rtol, atol=atol, ramp=ramp,
        use_shifts=use_shifts, segmented=segmented, t_eval=t_eval,
    )


# ── Filter-function pass: adjoint states and the frequency kernel ────────────
#
# The phase noise enters exactly as H -> H_0 + 2 pi dnu(t) N_r (V = exp(+i phi N_r)
# removes it from the drive), so first-order perturbation gives
#     <Delta L> = 2 pi^2 int S_dnu(|f|) ||Q G(f)||^2 df,
#     G(f) = int_0^T <q|U(T,t) N_r psi_s(t)> e^{-2 pi i f t} dt.
# Only the projected components are needed, so the propagator is never formed:
#     <q|A_s(t)> = <phi_q(t)| N_r |psi_s(t)>,   |phi_q(t)> = U(t,T)|q>.
# phi_q and psi_s obey the same equation and N_r is diagonal, so the exp(-i D_i t)
# factors cancel in the pointwise product and the sampled integrand carries only
# drive-scale structure — which is what makes n_t = 4096 enough despite the GHz
# pair interaction and the 6.8 GHz |0> hyperfine offset.

KERNEL_F_MIN_HZ = 1.0
KERNEL_F_MAX_HZ = 2.0e8
KERNEL_BINS_PER_DECADE = 30
KERNEL_N_T = 4096


def kernel_frequency_bins():
    """The fixed global storage bins shared by every point of the store."""
    from ryd_gate.phase_noise import log_frequency_bins

    return log_frequency_bins(KERNEL_F_MIN_HZ, KERNEL_F_MAX_HZ,
                              KERNEL_BINS_PER_DECADE)


def _297_adjoint_rhs_factory(ops, cols, t_gate, ramp):
    """Time-reversed RHS: dy/dtau = +i H(T - tau) y, same drive as the forward leg."""
    forward = _297_rhs_factory(ops, cols, t_gate, ramp)

    def rhs(tau, y):
        return -forward(t_gate - tau, y)

    return rhs


def integrate_adjoint_batch(ops, t_gate, omega_297, d_sweep, *,
                            rtol, atol, ramp=0.15, n_t=KERNEL_N_T):
    """Forward logical states + backward nonlogical adjoints, sampled together.

    Returns ``{"times": (n_t,), "components": (n_points, 4, n_t, 12),
    "overlaps": (n_points, 4, n_t, 12), "nfev": int}`` where ``components`` are
    ``<phi_q(t)|N_r|psi_s(t)>`` and ``overlaps`` are ``<phi_q(t)|psi_s(t)>`` (a
    conserved quantity, used as the correctness check).
    """
    omega_297 = np.asarray(omega_297, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    dim = ops.h_static_diag.size
    times = np.linspace(0.0, t_gate, n_t)

    fwd = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        LOGICAL_INPUTS, rhs_factory=_297_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times)

    nonlogical = np.setdiff1d(np.arange(dim), ops.logical_indices)
    adj = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        tuple(str(i) for i in nonlogical),
        rhs_factory=_297_adjoint_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times,
        initial_indices=nonlogical, reverse_time=True)

    # adj sampled in tau = T - t; flip back onto the forward time axis
    phi = adj.states[::-1]                       # (n_t, n_points, 12, dim)
    psi = fwd.states                             # (n_t, n_points, 4, dim)
    n_r = _rydberg_number_diag(dim)
    comp = np.einsum("tpqi,i,tpsi->pstq", phi.conj(), n_r, psi)
    over = np.einsum("tpqi,tpsi->pstq", phi.conj(), psi)
    return {"times": times, "components": comp, "overlaps": over,
            "nfev": fwd.nfev + adj.nfev}


def _rydberg_number_diag(dim: int, local_dim: int = 4) -> np.ndarray:
    """Diagonal of N_r: atoms in the Rydberg manifold (levels r and r_garb).

    One laser drives both 297 legs, so the noise operator counts both — the sum of
    the two scattering-channel weight vectors.
    """
    idx = np.arange(dim)
    a, b = np.divmod(idx, local_dim)
    return (np.isin(a, (2, 3)).astype(float) + np.isin(b, (2, 3)).astype(float))


def filter_kernels(ops, t_gate, omega_297, d_sweep, *,
                   rtol, atol, ramp=0.15, n_t=KERNEL_N_T) -> np.ndarray:
    """(n_points, 4, n_bins) binned filter kernels for one batch."""
    from ryd_gate.phase_noise import filter_kernel

    out = integrate_adjoint_batch(ops, t_gate, omega_297, d_sweep,
                                  rtol=rtol, atol=atol, ramp=ramp, n_t=n_t)
    f_bins, df_bins = kernel_frequency_bins()
    comp = out["components"]
    kernels = np.empty((comp.shape[0], 4, f_bins.size))
    for p in range(comp.shape[0]):
        for s in range(4):
            kernels[p, s] = filter_kernel(out["times"], comp[p, s], f_bins, df_bins)
    return kernels


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

_KEY_FIELDS = ("n_idx", "t_idx", "om_num", "om_den", "dw_num", "dw_den")

# The Store, atomic NPZ writes, three-hash provenance gates, chunk/scatter series
# and the PointRecord loader live in sweeplib.store; this script supplies its
# serialized field name (n_idx) and the physical descriptor columns via the
# ProvenanceColumns bundle below (formats frozen).


def _s297_descriptor(cfg: "ScanConfig", keys) -> dict:
    """Base physical-descriptor columns of one batch (ryd_n / t_gate / drives)."""
    return {
        "ryd_n": np.asarray([cfg.ryd_n[k.n_idx] for k in keys]),
        "t_gate_us": np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
        "omega297_mhz": np.asarray([float(k.omega_mhz()) for k in keys]),
        "dsweep_mhz": np.asarray([float(k.dsweep_mhz()) for k in keys]),
    }


def _s297_result_extra(cfg: "ScanConfig", keys, manifest: dict) -> dict:
    """Extended coherent-chunk columns: rad/s conversions of t_gate and the drives."""
    tg_us = np.asarray([cfg.t_gate_us[k.t_idx] for k in keys])
    om_mhz = np.asarray([float(k.omega_mhz()) for k in keys])
    dw_mhz = np.asarray([float(k.dsweep_mhz()) for k in keys])
    return {
        "t_gate_s": tg_us * 1e-6,
        "omega297_rad_s": om_mhz * 1e6 * TAU,
        "dsweep_rad_s": dw_mhz * 1e6 * TAU,
    }


class Store(sweeplib.Store):
    """The rb87_297_clock_4 scan store: the shared sweeplib.Store bound to this
    script's serialized ``n_idx`` field and physical descriptor columns.  A single-
    photon drive carries no fixed model constant (no 1013 Rabi) and hence no extra
    provenance.  Constructible from just the output directory (resume/status)."""

    def __init__(self, output_dir: str):
        super().__init__(
            output_dir, key_type=PointKey, key_fields=_KEY_FIELDS,
            provenance_columns=ProvenanceColumns(
                scatter_channels=SCATTER_CHANNELS, default_dim=16,
                descriptor=_s297_descriptor, result_extra=_s297_result_extra,
                schema_version=SCHEMA_VERSION))


def _manifest_extras(cfg: "ScanConfig") -> dict:
    """The 297-specific manifest payload for init_or_validate_manifest.

    A single-photon drive has no fixed model constant to record or guard on resume,
    so there is no extra_fields/extra_guard arm (unlike the ode 1013 block)."""
    axes = {
        "ryd_n": list(cfg.ryd_n),
        "t_gate_us": list(cfg.t_gate_us),
        "omega297_anchors_mhz": [str(a) for a in OMEGA297_ANCHORS_MHZ],
        "dsweep_anchors_mhz": [str(a) for a in DSWEEP_ANCHORS_MHZ],
        "level_sizes": list(LEVEL_SIZES),
        "dsweep_hw_limit_mhz": DSWEEP_HW_LIMIT_MHZ,
    }
    return dict(pulse_hash=pulse_hash(), axes=axes)


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
        ryd_n=np.asarray([cfg.ryd_n[k.n_idx] for k in keys]),
        t_gate_us=np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
        omega297_mhz=np.asarray([float(k.omega_mhz()) for k in keys]),
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
        if n else np.zeros((0, 4, 16), dtype=np.complex128),
    )
    _atomic_savez(merged_path, **payload)

    csv_path = os.path.join(store.exports_dir, "points.csv")
    cols = ["point_id", "ryd_n", "t_gate_us", "omega297_mhz", "dsweep_mhz",
            "max_leakage", "leak_00", "leak_01", "leak_10", "leak_11", "worst_input",
            "min_return_prob", "norm_err_max", "tier", "rtol", "atol", "nfev",
            "runtime_s", "batch_id"]
    tmp = csv_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for k, r in zip(keys, rows):
            fh.write(",".join(str(v) for v in (
                k.id(), cfg.ryd_n[k.n_idx], cfg.t_gate_us[k.t_idx],
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
# p_ch = Gamma_ch * integral_0^T <n_ch(t)> dt for the Rydberg (r) and
# garbage-Rydberg (r_garb) manifolds, per logical input — the same
# trapezoid-on-301-samples convention as error_buget.ipynb and the stored
# pilot/audit trajectories.  A single-photon 297 nm drive has no intermediate
# state, so there is no p_mid channel.  These are written to a separate
# append-only ``scatter/`` chunk series and never touch the coherent-leakage
# chunks.

SCATTER_CHANNELS = ("p_ryd", "p_r_garb")
_SCATTER_LEVEL_GROUPS = {"p_ryd": (2,), "p_r_garb": (3,)}


def _scatter_weight_vectors(local_dim: int = 4) -> dict[str, np.ndarray]:
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
    """The rb87_297_clock_4 Rydberg decay rates (rad/s) for the scattering integrals."""
    rates = system.level_structure.decay_rates_per_s
    return {
        "p_ryd": float(rates["r"]["total"]),
        "p_r_garb": float(rates["r_garb"]["total"]),
    }


# ── Startup: warm ARC, compile every row, run the mandatory verifications ────

HAM_EQUIV_REL_TOL = 1e-12
ERR_NORM_REL_TOL = 1e-12
PACK_GATE_STATE_TOL = 1e-6
PACK_GATE_LEAK_TOL = 1e-8


def _script_code_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def warm_and_build(cfg: ScanConfig) -> tuple[dict[int, PanelOperators], str, dict]:
    """Warm ARC in-parent, compile/aggregate all Rydberg-n rows, verify invariants.

    Returns ``(ops_by_n, model_hash, checks)``.  Raises if the grouped Hamiltonian
    deviates from the repository compiler, if the static Hamiltonian is not
    closed/Hermitian, or if the SciPy error-norm seam moved.
    """
    t0 = time.time()

    ops_by_n: dict[int, PanelOperators] = {}
    systems: dict[int, object] = {}
    decay_by_n: dict[int, dict[str, float]] = {}
    for n_idx, n in enumerate(cfg.ryd_n):
        systems[n_idx] = build_system(cfg, n)   # ARC touch (lifetimes/C6) happens here
        ops_by_n[n_idx] = aggregate_operators(systems[n_idx], n)
        decay_by_n[n_idx] = model_decay_rates(systems[n_idx])

    h = hashlib.sha256()
    for n_idx in sorted(ops_by_n):
        h.update(ops_by_n[n_idx].hash_bytes())
    model_hash = h.hexdigest()

    # Grouped-vs-compiled Hamiltonian equivalence on the middle panel (n=60 row in
    # the default 8-row config), across evenly spaced times spanning [0, T].
    mid = (len(cfg.ryd_n) - 1) // 2
    t_gate = 1.2e-6
    omega_297 = TAU * 13.5e6
    d_sweep = TAU * 15e6
    probe_times = np.linspace(0.0, t_gate, 41)
    dev = hamiltonian_equivalence_error(
        systems[mid], ops_by_n[mid], t_gate,
        omega_297=omega_297, d_sweep=d_sweep,
        times=probe_times, ramp=cfg.ramp_frac)
    scale = float(np.max(np.abs(ops_by_n[mid].h_static_diag)) + 2 * omega_297)
    ham_dev_rel = dev / scale
    if ham_dev_rel > HAM_EQUIV_REL_TOL:
        raise RuntimeError(
            f"grouped Hamiltonian deviates from the compiled IR by relative "
            f"{ham_dev_rel:.3e} (> {HAM_EQUIV_REL_TOL:g}); refusing to run")

    err_norm_dev = verify_scipy_error_norm()
    norm_ok = err_norm_dev <= ERR_NORM_REL_TOL
    if not norm_ok:
        print(f"WARNING: installed SciPy DOP853 error norm could not be reproduced "
              f"(max dev {err_norm_dev:.3e}); multi-point batching disabled.", flush=True)

    swap_ok = all(o.swap_symmetric for o in ops_by_n.values())
    if not swap_ok:
        print("WARNING: atom-swap symmetry verification failed on some rows; "
              "all four logical inputs will be propagated.", flush=True)

    checks = {
        "hamiltonian_equivalence_rel_dev": ham_dev_rel,
        "error_norm_max_dev": err_norm_dev,
        "error_norm_verified": bool(norm_ok),
        "swap_symmetric": bool(swap_ok),
        "decay_rates_rad_s": decay_by_n,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "build_seconds": time.time() - t0,
    }
    return ops_by_n, model_hash, checks


def setup_run(args) -> tuple[Store, dict, ScanConfig, dict[int, PanelOperators], dict]:
    """Shared bring-up for pilot/run/audit: build, verify, manifest, worker context."""
    cfg = ScanConfig(
        spacing_um=args.spacing_um,
        rtol_production=args.rtol, atol_production=args.atol,
        rtol_audit=args.audit_rtol, atol_audit=args.audit_atol,
    )
    store = Store(args.output)
    store.ensure_dirs()
    ops, model_hash, checks = warm_and_build(cfg)
    manifest = store.init_or_validate_manifest(
        cfg, model_hash, _script_code_hash(),
        run_meta={
            "argv": sys.argv[1:], "workers": args.workers,
            "batch_size": args.batch_size,
        },
        **_manifest_extras(cfg))
    ver_path = os.path.join(store.reports_dir, "verification.json")
    with open(ver_path + ".tmp", "w") as fh:
        json.dump(checks, fh, indent=2)
    os.replace(ver_path + ".tmp", ver_path)

    # The shared worker context takes the script's solve wrapper and its
    # scattering_integrals; gammas are keyed by panel row (this model's Rydberg
    # decay rates are n-dependent, so each row carries its own measured dict).
    def _solve(ops, t_gate, omega_297, d_sweep, *, rtol, atol, ramp, use_swap, t_eval):
        return integrate_batch(ops, t_gate, omega_297, d_sweep,
                               rtol=rtol, atol=atol, ramp=ramp, use_swap=use_swap,
                               t_eval=t_eval)

    def _filter_solve(ops, t_gate, omega_297, d_sweep, *, rtol, atol, ramp):
        return filter_kernels(ops, t_gate, omega_297, d_sweep,
                              rtol=rtol, atol=atol, ramp=ramp)

    set_worker_context(
        cfg, ops, use_swap=checks["swap_symmetric"],
        gammas=checks["decay_rates_rad_s"],
        key_type=PointKey, solve=_solve, scattering_integrals=scattering_integrals,
        filter_solve=_filter_solve)
    print(f"[setup] panels(n) = {len(cfg.ryd_n)} | "
          f"H equivalence rel dev {checks['hamiltonian_equivalence_rel_dev']:.2e} | "
          f"error-norm dev {checks['error_norm_max_dev']:.2e} | "
          f"swap {'ok' if checks['swap_symmetric'] else 'FAILED'}", flush=True)
    return store, manifest, cfg, ops, checks


# ── Pilot: reusable nodes, throughput, packing acceptance gate ───────────────

# Deterministic packing-gate panel and varied in-panel nodes (all level-0, reusable).
PACK_GATE_PANEL = (3, 0)          # n = RYD_N[3] = 60, T = 1 us
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
#
# The credibility floor (audit-derived vmin) lives in sweeplib.plotting and is
# imported above as ``_credibility_floor``.


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
        if not (0 <= di < len(RYD_N) and 0 <= ti < len(T_GATE_US)):
            raise SystemExit(f"--panels: ({di},{ti}) out of range")
        panels.add((di, ti))
    return panels


def _filter_panels(keys: Iterable[PointKey],
                   panels: set[tuple[int, int]] | None) -> list[PointKey]:
    return [k for k in keys if panels is None or k.panel in panels]


def cmd_pilot(args) -> None:
    store, manifest, cfg, ops, checks = setup_run(args)
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
    store, manifest, cfg, ops, checks = setup_run(args)
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
              f"(rows n={list(cfg.ryd_n)} x cols {list(cfg.t_gate_us)} us)")
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
    # model's Rydberg decay rates are n-dependent), and already-scattered keys are
    # skipped at write time so a resumed run never duplicates a scatter record.
    runner.gammas = checks["decay_rates_rad_s"]
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
    store, manifest, cfg, ops, checks = setup_run(args)
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
    contact.  (The live a3.0 store already carries a passed record, so merged code
    skips the gate on it; a store without one runs it once against its stored
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
    """Validate the scatter pipeline against an independent exact_ode reference.

    Picks the cheapest stored production trajectory point, re-solves it with the
    in-worker scatter path (the batched block-DOP853 kernel), and compares its
    integrals against a reference computed from a fully independent leg: the same
    pulse propagated through the repository's public ``exact_ode`` backend, with
    occupation weights from ``build_occ_operator`` and decay rates re-read from a
    freshly built system's metadata — so a bug in the shared weight/Gamma/kernel
    plumbing cannot cancel out.
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

    # Independent reference leg (deliberately NOT the block-DOP853 kernel): the
    # concrete pulse propagated through the public exact_ode backend, occupation
    # weights from the repository operators, rates from the model, plain trapezoid.
    from ryd_gate.backends.exact.compiler import compile_exact
    from ryd_gate.backends.exact.ode import evolve_states
    from ryd_gate.core.operators import build_occ_operator
    from ryd_gate.core.states import dense_product_state
    from ryd_gate.protocols import Direct297CZProtocol

    cfg = runner.cfg
    t_gate = cfg.t_gate_us[best_key.t_idx] * 1e-6
    omega_297 = float(best_key.omega_mhz()) * 1e6 * TAU
    d_sweep = float(best_key.dsweep_mhz()) * 1e6 * TAU
    ramp = cfg.ramp_frac
    proto = Direct297CZProtocol(
        t_gate_s=t_gate, omega_297_max_rad_s=omega_297,
        envelope_297=lambda t: float(np.sqrt(envelope(t / t_gate, ramp))),
        phase_297_rad=lambda t: float(phase_rad(t, t_gate, d_sweep, ramp)))
    ref_sys = build_system(cfg, cfg.ryd_n[best_key.n_idx]).with_protocol(proto)
    ref_rates = model_decay_rates(ref_sys)

    ham, _ = compile_exact(ref_sys, hamiltonian_format="dense")
    times = np.linspace(0.0, t_gate, cfg.n_eval_trajectory)
    psi0 = [dense_product_state(list(s), ref_sys._basis) for s in LOGICAL_INPUTS]
    ys = evolve_states(ham, t_gate, psi0, times,
                       rtol=cfg.rtol_audit, atol=cfg.atol_audit)
    states = np.stack([y.T for y in ys], axis=1)     # (n_t, 4, dim)

    local_dim = int(round(math.sqrt(states.shape[-1])))
    ref_levels = {"p_ryd": (2,), "p_r_garb": (3,)}
    pops = np.abs(states) ** 2                        # (n_t, 4, dim)
    ref = {}
    for ch, levels in ref_levels.items():
        w = np.real(np.diag(sum(build_occ_operator(lv, local_dim) for lv in levels)))
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
    store, manifest, cfg, ops, checks = setup_run(args)
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
    runner.gammas = checks["decay_rates_rad_s"]
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


def cmd_filter(args) -> None:
    """Filter-function pass: additive only (writes the filter/ series)."""
    store, manifest, cfg, ops, checks = setup_run(args)
    level = LEVEL_FROM_SIZE[int(args.level)]
    panels = _parse_panels(args)
    done = {r["key"] for r in store.load_filter_records(manifest)
            if r["status"] == "ok"}
    missing = [k for k in _filter_panels(all_keys(level), panels) if k not in done]
    print(f"[filter] level {args.level}: {len(missing)} points to compute "
          f"({len(done)} already stored)", flush=True)
    if not missing:
        return
    cost = CostModel(cfg)
    _feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    runner.filter_bins = kernel_frequency_bins()[0]
    runner.filter_n_t = KERNEL_N_T
    try:
        batches = group_batches(missing, _effective_batch_size(store, args))
        for b in batches:
            b.filter_pass = True
        runner.run_batches(batches, f"filter-{args.level}")
    except KeyboardInterrupt:
        print("[filter] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status(f"filter-{args.level}-aborted" if runner.aborted
                            else f"filter-{args.level}-done")
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
        pk = f"{k.n_idx},{k.t_idx}"
        if pk not in cand or r.max_leakage < cand[pk]["max_leakage"]:
            cand[pk] = {"point_id": k.id(), "max_leakage": r.max_leakage,
                        "omega297_mhz": float(k.omega_mhz()),
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
    print(f"git {manifest['git']['commit'][:10]}"
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
# The log-linear interpolation, LOO credibility veil, audit-derived floor and the
# 8x9 grid renderer live in sweeplib.plotting (rasters are visualization only; no
# per-panel PNGs).  This script binds the rb87_297_clock_4 scatter-channel table
# (no p_mid), the Rydberg-n row labeller, the 297-drive x-axis label and the
# system description.

_PLOT_SPEC = PlotSpec(
    scatter_channels=SCATTER_CHANNELS,
    row_axis_key="ryd_n",
    row_label=lambda v: f"$n$ = {v:g}",
    xlabel=r"$\Omega_{297}/2\pi$ (MHz)",
    system_desc="two-atom 297 nm single-photon CZ",
    hw_limit_mhz=DSWEEP_HW_LIMIT_MHZ,
)


def cmd_plot(args) -> None:
    store = Store(args.output)
    manifest = store.load_manifest()
    if manifest is None:
        raise SystemExit(f"no manifest under {store.root}")
    records = store.load_records(manifest, include_states=False)
    png, pdf = render_panel_grid(store, manifest, records, args.metric,
                                 _PLOT_SPEC, veil=args.veil, dpi=args.dpi)
    print(f"plots: {png}\n       {pdf}")


# ── CLI ──────────────────────────────────────────────────────────────────────
#
# The shared parser scaffold (--output/--spacing-um + the compute pool/tolerance/
# --panels flags) and the derived-output resolution live in sweeplib.cli; this
# script keeps its subcommand wiring, its metric list and its store-family root.

_FAMILY_ROOT = "max_leakage_297"


def _default_output(spacing_um: float) -> str:
    return cli.default_output(_FAMILY_ROOT, spacing_um)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="max_leakage_297_sweep",
        description=__doc__.split("\n\n")[0],
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, compute: bool = False):
        cli.add_common_args(sp, _FAMILY_ROOT, compute=compute)

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

    sp = sub.add_parser("filter",
                        help="filter-function pass (additive: writes only the "
                             "filter/ series; reusable across every PSD)")
    common(sp, compute=True)
    sp.add_argument("--level", default="13", choices=["4", "7", "13", "25"])
    sp.set_defaults(func=cmd_filter)

    sp = sub.add_parser("export", help="regenerate merged NPZ + CSV + reports")
    common(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("plot", help="render the 8x9 map family")
    common(sp)
    sp.add_argument("--dpi", type=int, default=170)
    sp.add_argument("--no-veil", dest="veil", action="store_false", default=True,
                    help="omit the uncertainty veil (raster is visualization only)")
    sp.add_argument("--metric", default="max_leakage",
                    choices=["max_leakage", "p_ryd", "p_r_garb",
                             "p_loss_total", "total_error"],
                    help="max_leakage from the main scan; p_* from the "
                         "supplemental scatter series; total_error combines both")
    sp.set_defaults(func=cmd_plot)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cli.resolve_output(args, _FAMILY_ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
