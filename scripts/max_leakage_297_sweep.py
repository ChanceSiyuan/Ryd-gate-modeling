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
    python scripts/max_leakage_297_sweep.py plot   --metric eps_phase --laser ECDL
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
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from typing import Sequence

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
    LEVEL_SIZES, LEVEL_FROM_SIZE, make_pointkey_type,
    envelope, envelope_integral, verify_scipy_error_norm, BatchResult,
    ProvenanceColumns, PointRecord, Runner, CostModel, Batch, group_batches,
    set_worker_context, cli, PlotSpec, render_panel_grid,
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
# removes it from the drive), so first-order perturbation gives the noise-induced
# fidelity loss against the noiseless final state,
#     eps = 1 - |<psi_0(T)|psi(T)>|^2
#         = 2 pi^2 int S_dnu(|f|) [ ||G(f)||^2 - |<psi_0(T)|G(f)>|^2 ] df,
#     G(f) = int_0^T <q|U(T,t) N_r psi_s(t)> e^{-2 pi i f t} dt.
# ||G||^2 runs over the COMPLETE basis, so the backward leg propagates all 16 basis
# states, not only the 12 nonlogical ones: the projector is Q = 1 - |psi_0><psi_0|,
# the only one that annihilates psi_0(T) and hence the only one for which this
# expression is second-order exact (see the design doc's "Why not the leakage
# increase").  The propagator is still never formed:
#     <q|A_s(t)> = <phi_q(t)| N_r |psi_s(t)>,   |phi_q(t)> = U(t,T)|q>.
# phi_q and psi_s obey the same equation and N_r is diagonal, so the exp(-i D_i t)
# factors cancel in the pointwise product and the sampled integrand carries only
# drive-scale structure — which is what makes n_t = 4096 enough despite the GHz
# pair interaction and the 6.8 GHz |0> hyperfine offset.
#
# The projection term costs nothing: <psi_s(T)|A_s(t)> = <psi_s(t)|N_r|psi_s(t)>,
# the Rydberg population along the noiseless trajectory.  It is contracted out of
# the same ``components`` array rather than computed from the forward leg alone, so
# Cauchy-Schwarz holds in floating point and the stored K_b cannot go negative.

KERNEL_F_MIN_HZ = 1.0
KERNEL_F_MAX_HZ = 2.0e8
KERNEL_BINS_PER_DECADE = 30
KERNEL_N_T = 4096
KERNEL_FINE_MIN = 200          # floor on filter_kernel's evaluation grid


def kernel_fine_per_decade(t_gate: float) -> int:
    """Evaluation points per decade for :func:`filter_kernel`, sized to ``t_gate``.

    ``||Q G(f)||^2`` carries sinc fringes of width ``1/T``, and a logarithmic grid of
    ``p`` points per decade has spacing ``f ln10 / p``, so resolving them across the
    whole band needs ``p >= ln10 * f_max * T``.  This is not a formality: at
    ``T = 4.5 us`` the library default of 200 misprices a pure tone at 50 MHz by +41%
    and one at 150 MHz by -91% (Parseval, ``tests/test_phase_noise.py``), and on the
    real gate it put the n=73 / 4.5 us corner 13% high.  The rule costs 2.3x the
    default quadrature at 1 us and 10x at 4.5 us; the solve is untouched.
    """
    return max(KERNEL_FINE_MIN,
               int(math.ceil(math.log(10.0) * KERNEL_F_MAX_HZ * float(t_gate))))


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
                            rtol, atol, ramp=0.15, n_t=KERNEL_N_T,
                            with_overlaps=False):
    """Forward logical states + the backward adjoint of every basis state.

    Returns ``{"times": (n_t,), "components": (n_points, 4, n_t, dim),
    "projection": (n_points, 4, n_t), "nfev": int}`` where ``components`` are
    ``<phi_q(t)|N_r|psi_s(t)>`` over the complete basis and ``projection`` is
    ``<psi_s(T)|A_s(t)>``, the piece :func:`ryd_gate.phase_noise.filter_kernel`
    subtracts.  ``with_overlaps`` adds ``"overlaps"`` of the shape of
    ``components``, the conserved ``<phi_q(t)|psi_s(t)>`` used as the correctness
    check — it costs a second einsum and array that size, so production leaves it
    off.
    """
    omega_297 = np.asarray(omega_297, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    dim = ops.h_static_diag.size
    times = np.linspace(0.0, t_gate, n_t)

    fwd = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        LOGICAL_INPUTS, rhs_factory=_297_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times)

    basis = np.arange(dim)
    adj = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        tuple(str(i) for i in basis),
        rhs_factory=_297_adjoint_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times,
        initial_indices=basis, reverse_time=True)

    # The tau -> t flip below is only the array reversal because ``times`` is a
    # symmetric linspace and both legs were sampled on exactly it; a segmented
    # solve that returned anything else would silently misalign the two legs.
    if not (np.array_equal(fwd.times, times) and np.array_equal(adj.times, times)):
        raise RuntimeError("integrate_batch did not sample the requested t_eval grid; "
                           "the tau = T - t reversal would misalign the two legs")

    # adj sampled in tau = T - t; flip back onto the forward time axis
    phi = adj.states[::-1]                       # (n_t, n_points, dim, dim)
    psi = fwd.states                             # (n_t, n_points, 4, dim)
    n_r = _rydberg_number_diag(dim)
    components = np.einsum("tpqi,i,tpsi->pstq", phi.conj(), n_r, psi)
    out = {"times": times,
           "components": components,
           "projection": np.einsum("psq,pstq->pst", fwd.psi_final.conj(), components),
           "nfev": fwd.nfev + adj.nfev}
    if with_overlaps:
        out["overlaps"] = np.einsum("tpqi,tpsi->pstq", phi.conj(), psi)
    return out


def _rydberg_number_diag(dim: int) -> np.ndarray:
    """Diagonal of N_r: atoms in the Rydberg manifold (levels r and r_garb).

    One laser drives both 297 legs, so the noise operator counts both — the sum of
    the two scattering-channel weight vectors, and the local dimension is derived
    from ``dim`` exactly as :func:`scattering_integrals` derives it, so the two can
    never disagree about what a Rydberg atom is.
    """
    local_dim = int(round(math.sqrt(dim)))
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
    comp, proj = out["components"], out["projection"]
    fine = kernel_fine_per_decade(t_gate)
    kernels = np.empty((comp.shape[0], 4, f_bins.size))
    for p in range(comp.shape[0]):
        for s in range(4):
            kernels[p, s] = filter_kernel(out["times"], comp[p, s], f_bins, df_bins,
                                          subtract=proj[p, s], fine_per_decade=fine)
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
    return sweeplib.campaign.export_store(store, ScanConfig, records)


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
    return sweeplib.campaign.run_packing_gate(
        runner, done, make_key=make_key, panel=PACK_GATE_PANEL,
        coords=PACK_GATE_COORDS, state_tol=PACK_GATE_STATE_TOL,
        leakage_tol=PACK_GATE_LEAK_TOL)


def stage_pilot(runner: Runner,
                panels: set[tuple[int, int]] | None = None) -> dict:
    return sweeplib.campaign.stage_pilot(
        runner, panels, pilot_keys=pilot_keys, packing_gate=run_packing_gate,
        packing_gate_panel=PACK_GATE_PANEL, all_keys=all_keys,
        level_sizes=LEVEL_SIZES)


# ── Stage orchestration and CLI commands ─────────────────────────────────────

def _parse_panels(args) -> set[tuple[int, int]] | None:
    return sweeplib.campaign.parse_panels(args, len(RYD_N), len(T_GATE_US))


def _campaign_gammas(cfg: ScanConfig, checks: dict) -> dict:
    return checks["decay_rates_rad_s"]


def _campaign_hooks() -> sweeplib.campaign.CampaignHooks:
    return sweeplib.campaign.CampaignHooks(
        setup_run=setup_run, parse_panels=_parse_panels, all_keys=all_keys,
        all_panels=all_panels, pilot_keys=pilot_keys, stage_pilot=stage_pilot,
        ensure_scatter_gate=_ensure_scatter_gate, gammas=_campaign_gammas,
        export_store=export_store, write_summary_reports=write_summary_reports,
        level_sizes=LEVEL_SIZES, level_from_size=LEVEL_FROM_SIZE,
        row_description=lambda cfg: (
            f"rows n={list(cfg.ryd_n)} x cols {list(cfg.t_gate_us)} us"))


def cmd_pilot(args) -> None:
    sweeplib.campaign.pilot_command(args, _campaign_hooks())


def cmd_run(args) -> None:
    sweeplib.campaign.run_command(args, _campaign_hooks())


def cmd_audit(args) -> None:
    sweeplib.campaign.audit_command(args, _campaign_hooks())


def _ensure_scatter_gate(runner: Runner, store: Store) -> dict:
    return sweeplib.campaign.ensure_scatter_gate(
        runner, store, _scatter_equivalence_gate)


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

    out = _worker_run_batch(runner._spec(Batch(keys=[best_key], mode="scatter")))
    if not out.get("ok"):
        return {"ok": False, "reason": f"gate solve failed: {out.get('message')}"}
    dev = max(float(np.max(np.abs(out["scatter"][ch][0] - ref[ch])))
              for ch in SCATTER_CHANNELS)
    return {"ok": dev < 1e-8, "point_id": best_key.id(), "trajectory": best_file,
            "max_abs_dev": dev, "tol": 1e-8}


def cmd_scatter(args) -> None:
    """Supplemental scattering-budget pass: additive only (scatter/ series)."""
    sweeplib.campaign.scatter_command(args, _campaign_hooks())


def cmd_filter(args) -> None:
    """Filter-function pass: additive only (writes the filter/ series)."""
    store, manifest, cfg, ops, checks = setup_run(args)
    level = LEVEL_FROM_SIZE[int(args.level)]
    panels = _parse_panels(args)
    done = {r["key"] for r in store.load_filter_records(manifest)
            if r["status"] == "ok"}
    missing = [
        key for key in sweeplib.campaign.filter_panels(all_keys(level), panels)
        if key not in done
    ]
    print(f"[filter] level {args.level}: {len(missing)} points to compute "
          f"({len(done)} already stored)", flush=True)
    if not missing:
        return
    cost = CostModel(cfg)
    sweeplib.campaign.feed_cost_model(
        cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    runner.filter_bins = kernel_frequency_bins()[0]
    runner.filter_n_t = KERNEL_N_T
    try:
        batches = group_batches(
            missing, sweeplib.campaign.effective_batch_size(store, args))
        for b in batches:
            b.mode = "filter"
        runner.run_batches(batches, f"filter-{args.level}")
    except KeyboardInterrupt:
        print("[filter] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status(f"filter-{args.level}-aborted" if runner.aborted
                            else f"filter-{args.level}-done")
        runner.shutdown()


def write_summary_reports(store: Store) -> None:
    sweeplib.campaign.write_summary_reports(store, "omega297_mhz")

def cmd_status(args) -> None:
    sweeplib.campaign.print_status(
        Store(args.output), all_keys=all_keys, pilot_keys=pilot_keys,
        level_sizes=LEVEL_SIZES)


def cmd_export(args) -> None:
    store = Store(args.output)
    merged, csv_path = export_store(store)
    write_summary_reports(store)
    print(f"exports: {merged}\\n         {csv_path}")


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


# ── power <-> Rabi table ─────────────────────────────────────────────────────
#
# Omega ~ sqrt(P / A) for a top-hat beam, so one ARC evaluation per n (the Rabi at
# 1 W over the nominal area) inverts to the power any Omega on the x axis costs.
# ARC is slow and is not needed to draw a figure, so the eight numbers are cached
# to npz; the cache is keyed on the n axis it was built for.

POWER_BEAM_AREA_UM2 = 420.0     # notebook nominal: 20 x spacing by 7 um top-hat
POWER_OPTICS_LOSS = 0.8         # 80% of the nominal power is lost before the atoms
POWER_TABLE_OMEGA_MHZ = (9.0, 11.0, 13.5, 15.0, 16.5, 18.0)
_LASER_NOISE_DIR = os.path.join("results", "297_laser_noise")
_POWER_CACHE = os.path.join(_LASER_NOISE_DIR, "omega_per_watt.npz")


def power_table_rows(cfg: "ScanConfig") -> dict:
    """Per-n target-leg Rabi at 1 W over POWER_BEAM_AREA_UM2 (cached; ARC once)."""
    if os.path.exists(_POWER_CACHE):
        with np.load(_POWER_CACHE, allow_pickle=False) as d:
            if list(d["ryd_n"]) == list(cfg.ryd_n):
                return {"ryd_n": d["ryd_n"], "omega_mhz_at_1w": d["omega_mhz_at_1w"]}
    import ryd_gate.physics as physics

    vals = np.asarray([
        physics.rb87_297_clock_rabi_frequencies(
            1.0, POWER_BEAM_AREA_UM2, ryd_level=int(n))[0] / (TAU * 1e6)
        for n in cfg.ryd_n])
    rows = {"ryd_n": np.asarray(cfg.ryd_n), "omega_mhz_at_1w": vals}
    _atomic_savez(_POWER_CACHE, **rows)
    return rows


def power_at_atoms_w(rows: dict, ryd_n: int, omega_mhz: float) -> float:
    """Power at the atoms (W) for ``omega_mhz`` at ``ryd_n``; Omega ~ sqrt(P/A)."""
    i = list(rows["ryd_n"]).index(int(ryd_n))
    return float((omega_mhz / rows["omega_mhz_at_1w"][i]) ** 2)


def _power_table(cfg: ScanConfig, caption: str) -> tuple:
    """(col_labels, row_labels, cells, caption) for PlotSpec.table."""
    rows = power_table_rows(cfg)
    cells = [[f"{power_at_atoms_w(rows, n, om):.2f} / "
              f"{power_at_atoms_w(rows, n, om) / (1.0 - POWER_OPTICS_LOSS):.2f}"
              for om in POWER_TABLE_OMEGA_MHZ] for n in rows["ryd_n"]]
    return ([f"{om:g} MHz" for om in POWER_TABLE_OMEGA_MHZ],
            [f"n = {n:g}" for n in rows["ryd_n"]], cells, caption)


# ── laser phase noise: the stored kernels reweighted by one measured PSD ─────
#
# The filter kernels do not depend on the spectrum, so each (laser x extrapolation)
# model is a reweighted sum over the stored bins and costs no solver time.  Both
# digitized spectra stop at 1 MHz while the gate is most sensitive at
# Omega/2pi = 9-18 MHz, so the extrapolation is an explicit bracket that both
# figures must carry, never a silent default: "flat" (hold the 1 MHz value) is the
# conservative headline and "power" (continue the fitted last-decade slope) the
# optimistic bound, and they differ by more than an order of magnitude.

PSD_HARMONIC = 4                 # 297 nm is the 4th harmonic of the measured 1180/1187
PHASE_NOISE_LASERS = ("ECDL", "seed")
PHASE_NOISE_EXTRAPOLATIONS = ("flat", "power")
PHASE_METRICS = ("eps_phase", "total_error_phase")

# The measured sigma_nu(1 Hz, 200 MHz)/Omega is 0.053, so the first-order filter
# function is a percent-level expansion parameter, not a part-per-thousand one:
# above this the prediction is outside its own regime and must be flagged.
EPS_PHASE_REGIME_MAX = 0.1


def phase_noise_values(store: Store, manifest: dict, laser: str,
                       extrapolation: str,
                       f_min: float = KERNEL_F_MIN_HZ) -> dict:
    """Per-point ``(4,)`` eps_phase from the stored kernels and one measured PSD.

    Each entry is the noise-induced fidelity loss of one logical input; ``|00>`` is
    dark and carries no 297 leg, so its kernel — and hence its loss — is exactly
    zero.  The bins come from the record rather than from the module constants: the
    store is what the kernel was integrated against, and it is the store's grid the
    spectrum has to be sampled on.

    ``f_min`` re-cuts the low-frequency edge of the integral.  It is a real modelling
    parameter, not a nicety: ``S_dnu`` rises as ``f**-2.5`` while the gate's response
    to a static detuning is finite, so the integral is infrared divergent and the
    cutoff is physically the inverse relock/calibration timescale.  Because the
    stored kernel is already bin-*integrated*, raising it is a pure reweighting of
    stored bins and needs no re-solve: bins whose **centre** falls below ``f_min``
    are dropped whole.  At 30 bins/decade a whole-bin decision places the true edge
    within 10**(1/60) = 3.9% of the requested one, and the default keeps every stored
    bin (the lowest centre is 1.039 Hz, above ``KERNEL_F_MIN_HZ``).

    One key can carry more than one record — a ``filter`` pass resumed or retried at
    a different ``--rtol`` writes a second chunk — so the tightest-``rtol`` row wins,
    exactly as the scatter path selects in ``sweeplib.plotting``.  Left to dict
    insertion order the deliverable would quote whichever tolerance sorted last.
    """
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    psd = PhaseNoisePSD.from_csv(
        os.path.join(_LASER_NOISE_DIR, f"psd_{laser}.csv"),
        harmonic=PSD_HARMONIC, extrapolation=extrapolation)
    best: dict = {}
    for r in store.load_filter_records(manifest):
        if r["status"] != "ok":
            continue
        cur = best.get(r["key"])
        if cur is None or r["rtol"] < cur["rtol"]:
            best[r["key"]] = r
    values = {}
    for key, r in best.items():
        band = r["f_bins"] >= f_min
        values[key] = np.asarray(
            [error_from_kernel(psd, r["f_bins"][band], r["kernel"][s][band])
             for s in range(len(LOGICAL_INPUTS))])
    return values


def _flag_out_of_regime(values: dict) -> str:
    """Name the nodes whose eps_phase leaves the perturbative regime.

    Prints them (so the campaign log names the cells) and returns a caption clause
    (so the figure carries the flag on its own).  A single top-decade *bin* is never
    quoted anywhere: only the integrated eps_phase is meaningful, since everything
    above 1 MHz is extrapolated.
    """
    bad = sorted(((float(np.max(v)), k) for k, v in values.items()
                  if float(np.max(v)) > EPS_PHASE_REGIME_MAX), reverse=True)
    if not bad:
        return ""
    shown = ", ".join(k.id() for _v, k in bad[:10])
    more = f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""
    print(f"[plot] WARNING: {len(bad)}/{len(values)} nodes have "
          f"eps_phase > {EPS_PHASE_REGIME_MAX} and are out of the perturbative "
          f"regime: {shown}{more}", flush=True)
    return (f"\nWARNING: {len(bad)} of {len(values)} nodes exceed "
            f"eps_phase = {EPS_PHASE_REGIME_MAX} (worst {bad[0][0]:.2f}) and are "
            "OUT OF THE PERTURBATIVE REGIME -- the first-order filter function "
            "does not apply there.")


def _phase_noise_caption(args, values: dict) -> str:
    """Table caption: the power conversion, the noise model, and the regime flag.

    Hard-wrapped: the strip's ``supxlabel`` does not reflow, and a single line of
    this length runs off both edges of the figure.
    """
    return (
        "Cell: 297 nm power at the atoms / nominal power (W) at the column's "
        f"Omega_297/2pi.  Beam area {POWER_BEAM_AREA_UM2:g} um^2 (P ~ A); "
        f"optics loss {POWER_OPTICS_LOSS:g}, so nominal = at-atoms / "
        f"{1.0 - POWER_OPTICS_LOSS:g}."
        f"\neps_phase: {args.laser} PSD x harmonic {PSD_HARMONIC}, "
        f"'{args.extrapolation}' extrapolation above the 1 MHz measurement edge, "
        f"f_min = {args.f_min:g} Hz."
        + _flag_out_of_regime(values))


def _phase_title_note(args) -> str:
    """Second suptitle line naming the noise model this figure was rendered under.

    The deliverable is a two-laser x two-extrapolation comparison read side by side
    and shared as detached pages, so the model may not live only in the filename and
    the table caption: a page has to say which laser and which extrapolation it is.
    """
    return (f"laser-phase-noise model: {args.laser} PSD, '{args.extrapolation}' "
            f"extrapolation above the 1 MHz measurement edge, "
            f"f_min = {args.f_min:g} Hz")


def cmd_plot(args) -> None:
    store = Store(args.output)
    manifest = store.load_manifest()
    if manifest is None:
        raise SystemExit(f"no manifest under {store.root}")
    records = store.load_records(manifest, include_states=False)
    spec, extra, suffix, subdir, title_note = _PLOT_SPEC, None, "", "", ""
    if args.metric in PHASE_METRICS:
        extra = phase_noise_values(store, manifest, args.laser, args.extrapolation,
                                   args.f_min)
        cfg = ScanConfig(ryd_n=tuple(manifest["axes"]["ryd_n"]))
        spec = replace(_PLOT_SPEC,
                       table=_power_table(cfg, _phase_noise_caption(args, extra)))
        title_note = _phase_title_note(args)
        # A non-default cutoff must reach the filename, or the f_min sensitivity
        # render lands on top of the headline one -- the same collision the
        # laser/extrapolation suffix exists to prevent.
        suffix = f"{args.laser}_{args.extrapolation}"
        if args.f_min != KERNEL_F_MIN_HZ:
            suffix += f"_fmin{args.f_min:g}Hz"
        subdir = os.path.join("phase_noise", args.laser)
    png, pdf = render_panel_grid(store, manifest, records, args.metric, spec,
                                 veil=args.veil, dpi=args.dpi,
                                 extra_values=extra, suffix=suffix, subdir=subdir,
                                 title_note=title_note)
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
                             "p_loss_total", "total_error",
                             "eps_phase", "total_error_phase"],
                    help="max_leakage from the main scan; p_* from the "
                         "supplemental scatter series; total_error combines both; "
                         "eps_phase reweights the filter series by --laser/"
                         "--extrapolation and total_error_phase adds it in")
    sp.add_argument("--laser", default="ECDL", choices=list(PHASE_NOISE_LASERS),
                    help="measured 297 nm phase-noise spectrum (eps_phase metrics)")
    sp.add_argument("--extrapolation", default="flat",
                    choices=list(PHASE_NOISE_EXTRAPOLATIONS),
                    help="S_dnu above the 1 MHz measurement edge: flat holds the "
                         "edge value (conservative), power continues the fitted "
                         "last-decade slope (optimistic).  Both bracket the answer")
    sp.add_argument("--f-min", type=float, default=KERNEL_F_MIN_HZ,
                    help="low-frequency cutoff (Hz) of the eps_phase integral; "
                         "stored bins whose centre falls below it are dropped, so "
                         "this is a reweighting and needs no re-solve.  The default "
                         "is the stored kernels' own edge; raising it models a lock "
                         "that removes the slow drift")
    sp.set_defaults(func=cmd_plot)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cli.resolve_output(args, _FAMILY_ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
