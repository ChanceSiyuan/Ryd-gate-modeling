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
hash-validated manifest; interrupted scans resume without recomputing.  Runs are
staged under a hard wall-time budget (default 24 h with a 2 h reserve): pilot ->
full 4x4 -> full 7x7 -> full 13x13 if the measured P90 ETA fits, else high-value
13-level nodes by a deterministic contour/residual priority.

Usage
-----
    # default store: results/max_leakage_297/a{spacing:.1f} (spacing default 3.0)
    python scripts/max_leakage_297_sweep.py status
    python scripts/max_leakage_297_sweep.py pilot  --spacing-um 5 --workers 40
    python scripts/max_leakage_297_sweep.py run    --spacing-um 5 --budget-hours 24 \
        --reserve-hours 2 --target-level auto
    python scripts/max_leakage_297_sweep.py run    --spacing-um 5 --dry-run
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
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Iterable, Sequence

import numpy as np
import scipy.integrate
from scipy.integrate._ivp.rk import DOP853 as _ScipyDOP853

TAU = 2.0 * math.pi
SCHEMA_VERSION = 1

# ── Locked scientific configuration ──────────────────────────────────────────

RYD_N = (50, 53, 56, 60, 64, 68, 71, 73)                          # panel rows (Rydberg n)
T_GATE_US = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)          # panel columns

# Nested-axis anchors (MHz, exact rationals).  Deliberately non-uniform so the
# 20 MHz hardware cap is an exact node at every level; do not replace with linspace.
OMEGA297_ANCHORS_MHZ = (Fraction(9), Fraction(12), Fraction(15), Fraction(18))
DSWEEP_ANCHORS_MHZ = (Fraction(2), Fraction(10), Fraction(20), Fraction(30))

LEVEL_DENS = (1, 2, 4, 8)          # nesting level 0..3 -> axis of 3*den+1 nodes
LEVEL_SIZES = (4, 7, 13, 25)       # nodes per axis at each level
LEVEL_FROM_SIZE = {4: 0, 7: 1, 13: 2, 25: 3}

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
# identical pair.


def canon_coord(num: int, den: int) -> tuple[int, int]:
    """Reduce a fractional axis coordinate ``num/den`` to lowest terms."""
    if den <= 0 or num < 0 or num > 3 * den:
        raise ValueError(f"coordinate {num}/{den} outside the axis range [0, 3]")
    g = math.gcd(num, den)
    return num // g, den // g


def coord_value_mhz(anchors: Sequence[Fraction], num: int, den: int) -> Fraction:
    """Exact axis value (MHz) of the reduced coordinate ``num/den``."""
    seg, rem = divmod(num, den)
    if rem == 0:
        return anchors[seg]
    return anchors[seg] + (anchors[seg + 1] - anchors[seg]) * Fraction(rem, den)


def axis_coords(level: int) -> list[tuple[int, int]]:
    """Canonical coordinates of every axis node at nesting ``level`` (0..3)."""
    den = LEVEL_DENS[level]
    return [canon_coord(k, den) for k in range(3 * den + 1)]


def axis_values_mhz(anchors: Sequence[Fraction], level: int) -> list[Fraction]:
    return [coord_value_mhz(anchors, n, d) for n, d in axis_coords(level)]


@dataclass(frozen=True, order=True)
class PointKey:
    """Canonical identity of one scan node: panel indices + reduced axis coords."""

    n_idx: int
    t_idx: int
    om_num: int
    om_den: int
    dw_num: int
    dw_den: int

    def id(self) -> str:
        return (f"n{self.n_idx}_t{self.t_idx}"
                f"_om{self.om_num}-{self.om_den}_dw{self.dw_num}-{self.dw_den}")

    @property
    def panel(self) -> tuple[int, int]:
        return (self.n_idx, self.t_idx)

    def omega_mhz(self) -> Fraction:
        return coord_value_mhz(OMEGA297_ANCHORS_MHZ, self.om_num, self.om_den)

    def dsweep_mhz(self) -> Fraction:
        return coord_value_mhz(DSWEEP_ANCHORS_MHZ, self.dw_num, self.dw_den)

    def level(self) -> int:
        """Finest nesting level this node first appears at."""
        den = max(self.om_den, self.dw_den)
        return LEVEL_DENS.index(den)


def make_key(n_idx: int, t_idx: int, om: tuple[int, int], dw: tuple[int, int]) -> PointKey:
    om = canon_coord(*om)
    dw = canon_coord(*dw)
    return PointKey(n_idx, t_idx, om[0], om[1], dw[0], dw[1])


def panel_keys(n_idx: int, t_idx: int, level: int) -> list[PointKey]:
    """All nodes of one panel at nesting ``level`` (row-major: omega outer, dw inner)."""
    coords = axis_coords(level)
    return [make_key(n_idx, t_idx, om, dw) for om in coords for dw in coords]


def all_panels() -> list[tuple[int, int]]:
    return [(di, ti) for di in range(len(RYD_N)) for ti in range(len(T_GATE_US))]


def all_keys(level: int) -> list[PointKey]:
    return [k for di, ti in all_panels() for k in panel_keys(di, ti, level)]


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


def quintic(u):
    """Quintic smoothstep q(u) = 10 u^3 - 15 u^4 + 6 u^5 on [0, 1] (clipped)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def quintic_antideriv(u):
    """Q(u) = integral_0^u q(v) dv = 2.5 u^4 - 3 u^5 + u^6 on [0, 1] (clipped)."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 4 * (2.5 + u * (-3.0 + u))


def envelope(s, ramp: float = 0.15):
    """Power envelope E(s) on s in [0, 1]; scalar or ndarray."""
    s = np.clip(s, 0.0, 1.0)
    return np.where(
        s < ramp,
        quintic(s / ramp),
        np.where(s > 1.0 - ramp, quintic((1.0 - s) / ramp), 1.0),
    )


def envelope_integral(s, ramp: float = 0.15):
    """J(s) = integral_0^s E(v) dv, closed form; scalar or ndarray."""
    s = np.clip(s, 0.0, 1.0)
    rise = ramp * quintic_antideriv(s / ramp)
    mid = s - 0.5 * ramp
    fall = 1.0 - ramp - ramp * quintic_antideriv((1.0 - s) / ramp)
    return np.where(s < ramp, rise, np.where(s > 1.0 - ramp, fall, mid))


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


# ── Block-max DOP853 solver ──────────────────────────────────────────────────
#
# scipy's DOP853 controls a single RMS error norm over the whole flattened vector,
# which would let one inaccurate point/state hide inside a large batch.  The
# subclass below reproduces the installed SciPy estimate *independently* for every
# 16-component (point, logical input) block and returns the maximum, so tolerance
# is enforced per block.  The private seam (E3/E5 + _estimate_error_norm) is pinned
# by ``verify_scipy_error_norm`` at startup and by unit tests.


class BlockMaxDOP853(_ScipyDOP853):
    """DOP853 whose error norm is the max of per-block SciPy DOP853 norms."""

    block_size: int = 16  # overridden per solve via a dynamic subclass

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale
        e5 = err5.reshape(-1, self.block_size)
        e3 = err3.reshape(-1, self.block_size)
        n5 = np.einsum("ij,ij->i", e5.real, e5.real) + np.einsum("ij,ij->i", e5.imag, e5.imag)
        n3 = np.einsum("ij,ij->i", e3.real, e3.real) + np.einsum("ij,ij->i", e3.imag, e3.imag)
        denom = n5 + 0.01 * n3
        out = np.zeros_like(n5)
        mask = denom > 0.0
        out[mask] = np.abs(h) * n5[mask] / np.sqrt(denom[mask] * self.block_size)
        return float(out.max())


def make_block_solver_class(block_size: int) -> type:
    """A BlockMaxDOP853 subclass bound to ``block_size`` (usable with solve_ivp)."""
    return type(f"BlockMaxDOP853_{block_size}", (BlockMaxDOP853,), {"block_size": block_size})


def verify_scipy_error_norm(n_blocks: int = 3, block: int = 49, seed: int = 12345) -> float:
    """Worst *relative* deviation of the custom norm from installed SciPy, random inputs.

    Verifies (1) the single-block custom norm reproduces the installed SciPy DOP853
    ``_estimate_error_norm`` and (2) the multi-block norm equals the max of the
    per-block SciPy norms.  Pins the private E3/E5/_estimate_error_norm seam.
    """
    rng = np.random.default_rng(seed)
    n_stages = _ScipyDOP853.n_stages

    class _Probe:
        E3 = _ScipyDOP853.E3
        E5 = _ScipyDOP853.E5

    def _custom(n: int, K, h, scale) -> float:
        solver = make_block_solver_class(block)(
            lambda t, y: 0 * y, 0.0, np.zeros(n, complex), 1.0)
        return BlockMaxDOP853._estimate_error_norm(solver, K, h, scale)

    def _rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b), 1e-300)

    worst = 0.0
    for _ in range(20):
        h = float(rng.uniform(1e-12, 1e-6))
        k1 = (rng.standard_normal((n_stages + 1, block))
              + 1j * rng.standard_normal((n_stages + 1, block)))
        scale1 = rng.uniform(1e-12, 1e-6, size=block)
        ref = _ScipyDOP853._estimate_error_norm(_Probe(), k1, h, scale1)
        worst = max(worst, _rel(ref, _custom(block, k1, h, scale1)))

        km = (rng.standard_normal((n_stages + 1, n_blocks * block))
              + 1j * rng.standard_normal((n_stages + 1, n_blocks * block)))
        scalem = rng.uniform(1e-12, 1e-6, size=n_blocks * block)
        per_block = max(
            _ScipyDOP853._estimate_error_norm(
                _Probe(), km[:, b * block:(b + 1) * block], h,
                scalem[b * block:(b + 1) * block])
            for b in range(n_blocks)
        )
        worst = max(worst, _rel(per_block, _custom(n_blocks * block, km, h, scalem)))
    return worst


# ── Batched original-frame integration kernel ────────────────────────────────


@dataclass
class BatchResult:
    """Terminal states and diagnostics for one integrated batch."""

    psi_final: np.ndarray        # (n_points, 4, dim) complex128, original frame
    leakage: np.ndarray          # (n_points, 4) direct nonlogical population
    max_leakage: np.ndarray      # (n_points,)
    worst_input: list[str]       # per point, the argmax logical input
    return_prob: np.ndarray      # (n_points, 4) |<s|psi_s>|^2
    norm_err: np.ndarray         # (n_points, 4) | ||psi||^2 - 1 |
    nfev: int
    used_swap: bool
    times: np.ndarray | None = None      # trajectory sample times (if requested)
    states: np.ndarray | None = None     # (n_times, n_points, 4, dim) (if requested)


def _integrate_segments(
    rhs: Callable,
    y0: np.ndarray,
    t_gate: float,
    ramp: float,
    rtol: float,
    atol: float,
    block_size: int,
    t_eval: np.ndarray | None,
):
    """Adaptive DOP853 over [0, rT], [rT, (1-r)T], [(1-r)T, T] with state carry-over.

    Returns ``(y_final, nfev, times, states)``; trajectory arrays are None unless
    ``t_eval`` is given (then states has shape (n_times, len(y0))).
    """
    cls = make_block_solver_class(block_size)
    breakpoints = [0.0, ramp * t_gate, (1.0 - ramp) * t_gate, t_gate]
    y = y0
    nfev = 0
    times_out: list[np.ndarray] = []
    states_out: list[np.ndarray] = []
    for t0, t1 in zip(breakpoints[:-1], breakpoints[1:]):
        if t_eval is None:
            solver = cls(rhs, t0, y, t1, rtol=rtol, atol=atol)
            while solver.status == "running":
                solver.step()
            if solver.status != "finished":
                raise RuntimeError(f"DOP853 failed on segment [{t0:g}, {t1:g}]")
            y = solver.y
            nfev += solver.nfev
        else:
            mask = (t_eval >= t0) & (t_eval < t1) if t1 < t_gate else \
                   (t_eval >= t0) & (t_eval <= t1)
            seg_eval = np.unique(np.concatenate([t_eval[mask], [t1]]))
            sol = scipy.integrate.solve_ivp(
                rhs, (t0, t1), y, method=cls, rtol=rtol, atol=atol, t_eval=seg_eval)
            if not sol.success:
                raise RuntimeError(f"DOP853 failed on segment [{t0:g}, {t1:g}]: {sol.message}")
            keep = np.isin(sol.t, t_eval[mask])
            times_out.append(sol.t[keep])
            states_out.append(sol.y.T[keep])
            y = sol.y[:, -1]
            nfev += sol.nfev
    if t_eval is None:
        return y, nfev, None, None
    times = np.concatenate(times_out) if times_out else np.empty(0)
    states = np.concatenate(states_out, axis=0) if states_out else None
    return y, nfev, times, states


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
    propagated.  Each column is solved with its bare logical diagonal energy
    subtracted (H - c_s I) and the exact global phase exp(-i c_s T) restored.
    """
    omega_297 = np.asarray(omega_297, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    if omega_297.shape != d_sweep.shape or omega_297.ndim != 1:
        raise ValueError("omega_297 and d_sweep must be equal-length 1-D arrays")
    n_points = omega_297.size
    dim = ops.h_static_diag.size

    if use_swap is None:
        use_swap = ops.swap_symmetric
    state_labels = ("00", "01", "11") if use_swap else LOGICAL_INPUTS
    state_cols = {s: i for i, s in enumerate(state_labels)}
    n_states = len(state_labels)
    n_cols = n_points * n_states

    col_of_point = np.repeat(np.arange(n_points), n_states)
    om_cols = omega_297[col_of_point]
    dsw_cols = d_sweep[col_of_point]

    logical_of_state = {s: ops.logical_indices[LOGICAL_INPUTS.index(s)] for s in state_labels}
    col_logical_idx = np.asarray([logical_of_state[s] for s in state_labels] * n_points)
    shifts = ops.h_static_diag[col_logical_idx] if use_shifts else np.zeros(n_cols)

    diag_row = ops.h_static_diag[None, :] - shifts[:, None]   # (n_cols, dim) real
    x297_t = np.ascontiguousarray(ops.x297.T)
    y297_t = np.ascontiguousarray(ops.y297.T)
    ascale = ops.amplitude_scale
    sin_coef = -t_gate / TAU

    def rhs(t, y):
        s = t / t_gate
        amp = math.sqrt(float(envelope(s, ramp)))
        phi = (sin_coef * math.sin(TAU * s)) * dsw_cols
        c297 = (ascale * amp) * om_cols * np.exp(-1j * phi)
        ym = y.reshape(n_cols, dim)
        out = diag_row * ym
        out += c297.real[:, None] * (ym @ x297_t)
        out += c297.imag[:, None] * (ym @ y297_t)
        return (-1j * out).ravel()

    y0 = np.zeros((n_cols, dim), dtype=np.complex128)
    for p in range(n_points):
        for j, s in enumerate(state_labels):
            y0[p * n_states + j, logical_of_state[s]] = 1.0

    if segmented:
        y_fin, nfev, times, traj = _integrate_segments(
            rhs, y0.ravel(), t_gate, ramp, rtol, atol, dim, t_eval)
    else:
        cls = make_block_solver_class(dim)
        solver = cls(rhs, 0.0, y0.ravel(), t_gate, rtol=rtol, atol=atol)
        while solver.status == "running":
            solver.step()
        if solver.status != "finished":
            raise RuntimeError("DOP853 failed (unsegmented)")
        y_fin, nfev, times, traj = solver.y, solver.nfev, None, None

    def assemble(y_flat: np.ndarray, t_at: float) -> np.ndarray:
        """(n_cols*dim,) chi at time ``t_at`` -> (n_points, 4, dim) restored psi."""
        chi = y_flat.reshape(n_cols, dim) * np.exp(-1j * shifts * t_at)[:, None]
        psi = np.empty((n_points, 4, dim), dtype=np.complex128)
        for p in range(n_points):
            for j, s in enumerate(LOGICAL_INPUTS):
                if s in state_cols:
                    psi[p, j] = chi[p * n_states + state_cols[s]]
                else:  # s == "10", reconstructed from 01 by the atom swap
                    psi[p, j] = chi[p * n_states + state_cols["01"]][ops.swap_perm]
        return psi

    psi_final = assemble(y_fin, t_gate)

    nonlogical = np.setdiff1d(np.arange(dim), ops.logical_indices)
    pops = np.abs(psi_final) ** 2
    leakage = pops[:, :, nonlogical].sum(axis=2)
    max_leakage = leakage.max(axis=1)
    worst_input = [LOGICAL_INPUTS[int(np.argmax(leakage[p]))] for p in range(n_points)]
    return_prob = np.stack(
        [pops[:, j, ops.logical_indices[j]] for j in range(4)], axis=1)
    norm_err = np.abs(pops.sum(axis=2) - 1.0)

    states = None
    if traj is not None:
        states = np.stack(
            [assemble(traj[i], float(times[i])) for i in range(traj.shape[0])], axis=0)

    return BatchResult(
        psi_final=psi_final, leakage=leakage, max_leakage=max_leakage,
        worst_input=worst_input, return_prob=return_prob, norm_err=norm_err,
        nfev=int(nfev), used_swap=bool(use_swap),
        times=times, states=states,
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

_KEY_FIELDS = ("n_idx", "t_idx", "om_num", "om_den", "dw_num", "dw_den")
TIER_RANK = {"production": 0, "audit": 1}
_NO_STATES = np.empty((4, 0), dtype=np.complex128)  # states-skipped sentinel


def _atomic_savez(path: str, **arrays) -> None:
    for name, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(
                f"array {name!r} has dtype=object; chunks must load with "
                "allow_pickle=False")
    tmp = f"{path}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    dir_fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _git_state(repo_root: str) -> dict:
    def _run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True,
                timeout=10).stdout.strip()
        except Exception:
            return ""
    return {
        "commit": _run("rev-parse", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def keys_to_arrays(keys: Sequence[PointKey]) -> dict[str, np.ndarray]:
    return {
        "n_idx": np.asarray([k.n_idx for k in keys], dtype=np.int16),
        "t_idx": np.asarray([k.t_idx for k in keys], dtype=np.int16),
        "om_num": np.asarray([k.om_num for k in keys], dtype=np.int32),
        "om_den": np.asarray([k.om_den for k in keys], dtype=np.int32),
        "dw_num": np.asarray([k.dw_num for k in keys], dtype=np.int32),
        "dw_den": np.asarray([k.dw_den for k in keys], dtype=np.int32),
    }


def arrays_to_keys(d) -> list[PointKey]:
    return [
        PointKey(int(a), int(b), int(c), int(e), int(f), int(g))
        for a, b, c, e, f, g in zip(
            d["n_idx"], d["t_idx"], d["om_num"], d["om_den"], d["dw_num"], d["dw_den"])
    ]


@dataclass
class PointRecord:
    """One per-point result row loaded back from a chunk."""

    key: PointKey
    tier: str
    rtol: float
    atol: float
    status: str                  # 'ok' | 'failed' | 'timeout'
    max_leakage: float
    leakage: np.ndarray          # (4,)
    worst_input: str
    return_prob: np.ndarray      # (4,)
    norm_err: np.ndarray         # (4,)
    psi_final: np.ndarray        # (4, dim)
    nfev: int
    runtime_s: float
    batch_id: str
    batch_size: int
    retry_count: int
    priority_score: float
    message: str
    chunk_file: str
    used_swap: bool


class Store:
    """The on-disk scan store: manifest + chunk read/write + derived indices."""

    def __init__(self, output_dir: str):
        self.root = output_dir
        self.chunks_dir = os.path.join(output_dir, "chunks")
        self.traj_dir = os.path.join(output_dir, "trajectories")
        self.scatter_dir = os.path.join(output_dir, "scatter")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.exports_dir = os.path.join(output_dir, "exports")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.manifest_path = os.path.join(output_dir, "manifest.json")

    def ensure_dirs(self) -> None:
        for d in (self.root, self.chunks_dir, self.traj_dir, self.scatter_dir,
                  self.logs_dir, self.reports_dir, self.exports_dir,
                  self.plots_dir):
            os.makedirs(d, exist_ok=True)

    # -- manifest ---------------------------------------------------------

    def load_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path) as fh:
            return json.load(fh)

    def init_or_validate_manifest(
        self,
        cfg: ScanConfig,
        model_hash: str,
        code_hash: str,
        run_meta: dict,
    ) -> dict:
        """Create the manifest on first run; on resume, refuse hash mismatches."""
        existing = self.load_manifest()
        if existing is not None:
            for name, val in (("physics_hash", cfg.physics_hash()),
                              ("model_hash", model_hash),
                              ("pulse_hash", pulse_hash())):
                if existing.get(name) != val:
                    raise RuntimeError(
                        f"{name} mismatch: manifest has {existing.get(name)!r}, current "
                        f"code/model gives {val!r}.  Refusing to mix data produced by "
                        "different physics/model/pulse code — use a fresh --output "
                        "directory."
                    )
            return existing
        self.ensure_dirs()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "scan_uuid": uuid.uuid4().hex,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": _git_state(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "physics": cfg.physics_payload(),
            "physics_hash": cfg.physics_hash(),
            "model_hash": model_hash,
            "pulse_hash": pulse_hash(),
            "code_hash": code_hash,
            "tolerances": {
                "production": {"rtol": cfg.rtol_production, "atol": cfg.atol_production},
                "audit": {"rtol": cfg.rtol_audit, "atol": cfg.atol_audit},
            },
            "axes": {
                "ryd_n": list(cfg.ryd_n),
                "t_gate_us": list(cfg.t_gate_us),
                "omega297_anchors_mhz": [str(a) for a in OMEGA297_ANCHORS_MHZ],
                "dsweep_anchors_mhz": [str(a) for a in DSWEEP_ANCHORS_MHZ],
                "level_sizes": list(LEVEL_SIZES),
                "dsweep_hw_limit_mhz": DSWEEP_HW_LIMIT_MHZ,
            },
            "policies": {
                "interp_space": cfg.interp_space,
                "credibility_floor": "vmin = max(1e-12, 10 * P95(|L_prod - L_audit|))",
                "refine_residual_dex": 0.25,
                "refine_contour_residual_dex": 0.10,
                "decision_contours": [1e-3, 1e-2, 1e-4],
            },
            "run_meta": run_meta,
        }
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.manifest_path)
        return manifest

    # -- chunk writing ------------------------------------------------------

    def next_seq(self) -> int:
        seqs = [0]
        if os.path.isdir(self.chunks_dir):
            for name in os.listdir(self.chunks_dir):
                if name.startswith("chunk_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("chunk_"):-len(".npz")]))
                    except ValueError:
                        pass
        if os.path.isdir(self.traj_dir):
            for name in os.listdir(self.traj_dir):
                if name.startswith("traj_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("traj_"):-len(".npz")]))
                    except ValueError:
                        pass
        return max(seqs) + 1

    def write_result_chunk(
        self,
        seq: int,
        manifest: dict,
        keys: Sequence[PointKey],
        cfg: ScanConfig,
        tier: str,
        rtol: float,
        atol: float,
        batch_id: str,
        result: BatchResult | None,
        runtime_s: float,
        statuses: Sequence[str] | None = None,
        message: str = "",
        retry_count: int = 0,
        priority_scores: Sequence[float] | None = None,
    ) -> str:
        """Persist one completed (or failed) batch as an append-only chunk."""
        n = len(keys)
        dim = 16
        if result is not None:
            psi = result.psi_final
            dim = psi.shape[2]
            leak, max_leak = result.leakage, result.max_leakage
            worst, ret = result.worst_input, result.return_prob
            nerr = result.norm_err
            nfev = np.full(n, result.nfev // max(n, 1), dtype=np.int64)
            used_swap = result.used_swap
        else:
            psi = np.full((n, 4, dim), np.nan, dtype=np.complex128)
            leak = np.full((n, 4), np.nan)
            max_leak = np.full(n, np.nan)
            worst = [""] * n
            ret = np.full((n, 4), np.nan)
            nerr = np.full((n, 4), np.nan)
            nfev = np.zeros(n, dtype=np.int64)
            used_swap = False
        statuses = list(statuses) if statuses is not None else ["ok"] * n
        scores = (np.asarray(priority_scores, dtype=float)
                  if priority_scores is not None else np.zeros(n))

        om_mhz = np.asarray([float(k.omega_mhz()) for k in keys])
        dw_mhz = np.asarray([float(k.dsweep_mhz()) for k in keys])
        ryd_n = np.asarray([cfg.ryd_n[k.n_idx] for k in keys])
        tg_us = np.asarray([cfg.t_gate_us[k.t_idx] for k in keys])

        payload = dict(
            schema_version=np.int64(SCHEMA_VERSION),
            scan_uuid=str(manifest["scan_uuid"]),
            physics_hash=str(manifest["physics_hash"]),
            model_hash=str(manifest["model_hash"]),
            pulse_hash=str(manifest["pulse_hash"]),
            **keys_to_arrays(keys),
            ryd_n=ryd_n,
            t_gate_us=tg_us,
            omega297_mhz=om_mhz,
            dsweep_mhz=dw_mhz,
            t_gate_s=tg_us * 1e-6,
            omega297_rad_s=om_mhz * 1e6 * TAU,
            dsweep_rad_s=dw_mhz * 1e6 * TAU,
            psi_final=psi,
            leakage=leak,
            max_leakage=max_leak,
            worst_input=np.asarray(worst, dtype="U2"),
            return_prob=ret,
            norm_err=nerr,
            all_finite=np.asarray([bool(np.all(np.isfinite(psi[i].view(float))))
                                   for i in range(n)]),
            solver=np.asarray(["DOP853"] * n, dtype="U8"),
            tier=np.asarray([tier] * n, dtype="U10"),
            rtol=np.full(n, rtol),
            atol=np.full(n, atol),
            used_swap=np.asarray([used_swap] * n),
            nfev=nfev,
            runtime_s=np.full(n, runtime_s / max(n, 1)),
            batch_id=np.asarray([batch_id] * n, dtype="U40"),
            batch_size=np.full(n, n, dtype=np.int32),
            status=np.asarray(statuses, dtype="U16"),
            message=np.asarray([message] * n, dtype="U240"),
            retry_count=np.full(n, retry_count, dtype=np.int32),
            priority_score=scores,
        )
        path = os.path.join(self.chunks_dir, f"chunk_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    def write_trajectory_chunk(
        self,
        seq: int,
        manifest: dict,
        key: PointKey,
        tier: str,
        times: np.ndarray,
        states: np.ndarray,
    ) -> str:
        payload = dict(
            schema_version=np.int64(SCHEMA_VERSION),
            scan_uuid=str(manifest["scan_uuid"]),
            **keys_to_arrays([key]),
            tier=np.asarray([tier], dtype="U10"),
            times=np.asarray(times, dtype=float),
            states=np.asarray(states, dtype=np.complex128),
        )
        path = os.path.join(self.traj_dir, f"traj_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    # -- scattering supplement (separate append-only series) ---------------

    def next_scatter_seq(self) -> int:
        seqs = [0]
        if os.path.isdir(self.scatter_dir):
            for name in os.listdir(self.scatter_dir):
                if name.startswith("scatter_") and name.endswith(".npz"):
                    try:
                        seqs.append(int(name[len("scatter_"):-len(".npz")]))
                    except ValueError:
                        pass
        return max(seqs) + 1

    def write_scatter_chunk(
        self,
        seq: int,
        manifest: dict,
        keys: Sequence[PointKey],
        cfg: ScanConfig,
        gammas: dict[int, dict[str, float]],
        rtol: float,
        atol: float,
        batch_id: str,
        scatter: dict[str, np.ndarray],
        max_leakage: np.ndarray,
        runtime_s: float,
        statuses: Sequence[str] | None = None,
        message: str = "",
    ) -> str:
        """Persist one scattering-supplement batch; never touches other series."""
        n = len(keys)
        statuses = list(statuses) if statuses is not None else ["ok"] * n
        # Gamma depends on the Rydberg n row; every key in a batch shares one panel.
        panel_gammas = gammas[keys[0].n_idx]
        payload = dict(
            schema_version=np.int64(SCHEMA_VERSION),
            scan_uuid=str(manifest["scan_uuid"]),
            physics_hash=str(manifest["physics_hash"]),
            model_hash=str(manifest["model_hash"]),
            pulse_hash=str(manifest["pulse_hash"]),
            **keys_to_arrays(keys),
            ryd_n=np.asarray([cfg.ryd_n[k.n_idx] for k in keys]),
            t_gate_us=np.asarray([cfg.t_gate_us[k.t_idx] for k in keys]),
            omega297_mhz=np.asarray([float(k.omega_mhz()) for k in keys]),
            dsweep_mhz=np.asarray([float(k.dsweep_mhz()) for k in keys]),
            **{name: np.asarray(scatter[name]).reshape(n, 4)
               for name in SCATTER_CHANNELS},
            max_leakage_check=np.asarray(max_leakage),
            gamma_ryd=np.full(n, panel_gammas["p_ryd"]),
            gamma_r_garb=np.full(n, panel_gammas["p_r_garb"]),
            n_eval=np.full(n, cfg.n_eval_trajectory, dtype=np.int32),
            rtol=np.full(n, rtol),
            atol=np.full(n, atol),
            batch_id=np.asarray([batch_id] * n, dtype="U40"),
            batch_size=np.full(n, n, dtype=np.int32),
            status=np.asarray(statuses, dtype="U16"),
            message=np.asarray([message] * n, dtype="U240"),
            runtime_s=np.full(n, runtime_s / max(n, 1)),
        )
        path = os.path.join(self.scatter_dir, f"scatter_{seq:06d}.npz")
        _atomic_savez(path, **payload)
        return path

    def load_scatter_records(self, manifest: dict | None = None) -> list[dict]:
        """Per-point scattering rows from every scatter chunk (hash-validated)."""
        if manifest is None:
            manifest = self.load_manifest()
        rows: list[dict] = []
        if not os.path.isdir(self.scatter_dir):
            return rows
        for name in sorted(os.listdir(self.scatter_dir)):
            if not (name.startswith("scatter_") and name.endswith(".npz")):
                continue
            with np.load(os.path.join(self.scatter_dir, name),
                         allow_pickle=False) as npz:
                d = {f: npz[f] for f in npz.files}
            if manifest is not None:
                for fieldname in ("physics_hash", "model_hash", "pulse_hash"):
                    if str(d[fieldname]) != manifest[fieldname]:
                        raise RuntimeError(
                            f"scatter chunk {name} has a different {fieldname}; "
                            "refusing to merge data from different model/pulse code")
            for i, key in enumerate(arrays_to_keys(d)):
                rows.append({
                    "key": key,
                    "status": str(d["status"][i]),
                    "rtol": float(d["rtol"][i]),
                    **{ch: np.array(d[ch][i]) for ch in SCATTER_CHANNELS},
                    "max_leakage_check": float(d["max_leakage_check"][i]),
                    "runtime_s": float(d["runtime_s"][i]),
                })
        return rows

    # -- loading / derived indices -----------------------------------------

    def load_records(self, manifest: dict | None = None,
                     include_states: bool = True) -> list[PointRecord]:
        """All per-point records from every chunk (hash-validated).

        ``include_states=False`` skips the (4, dim) final-state payload — use it
        for scheduling/status/plotting indices over large stores.
        """
        if manifest is None:
            manifest = self.load_manifest()
        records: list[PointRecord] = []
        if not os.path.isdir(self.chunks_dir):
            return records
        for name in sorted(os.listdir(self.chunks_dir)):
            if not (name.startswith("chunk_") and name.endswith(".npz")):
                continue
            path = os.path.join(self.chunks_dir, name)
            with np.load(path, allow_pickle=False) as npz:
                # NpzFile re-decompresses a member on every __getitem__; read once.
                d = {f: npz[f] for f in npz.files
                     if include_states or f != "psi_final"}
                if manifest is not None:
                    for fieldname in ("physics_hash", "model_hash", "pulse_hash"):
                        if str(d[fieldname]) != manifest[fieldname]:
                            raise RuntimeError(
                                f"chunk {name} has a different {fieldname}; refusing to "
                                "merge data from different model/pulse code")
                keys = arrays_to_keys(d)
                for i, key in enumerate(keys):
                    records.append(PointRecord(
                        key=key,
                        tier=str(d["tier"][i]),
                        rtol=float(d["rtol"][i]),
                        atol=float(d["atol"][i]),
                        status=str(d["status"][i]),
                        max_leakage=float(d["max_leakage"][i]),
                        leakage=np.array(d["leakage"][i]),
                        worst_input=str(d["worst_input"][i]),
                        return_prob=np.array(d["return_prob"][i]),
                        norm_err=np.array(d["norm_err"][i]),
                        psi_final=(np.array(d["psi_final"][i]) if include_states
                                   else _NO_STATES),
                        nfev=int(d["nfev"][i]),
                        runtime_s=float(d["runtime_s"][i]),
                        batch_id=str(d["batch_id"][i]),
                        batch_size=int(d["batch_size"][i]),
                        retry_count=int(d["retry_count"][i]),
                        priority_score=float(d["priority_score"][i]),
                        message=str(d["message"][i]),
                        chunk_file=name,
                        used_swap=bool(d["used_swap"][i]),
                    ))
        return records


def best_records(records: Iterable[PointRecord]) -> dict[PointKey, PointRecord]:
    """Tightest successful record per point: lowest rtol wins; the audit tier only
    breaks ties, so a loosened ad-hoc audit can never displace a tighter
    production record in exports."""
    best: dict[PointKey, PointRecord] = {}
    for r in records:
        if r.status != "ok":
            continue
        cur = best.get(r.key)
        if cur is None:
            best[r.key] = r
            continue
        a = (-r.rtol, TIER_RANK.get(r.tier, 0))
        b = (-cur.rtol, TIER_RANK.get(cur.tier, 0))
        if a > b:
            best[r.key] = r
    return best


def completed_keys(records: Iterable[PointRecord], tier: str | None = None) -> set[PointKey]:
    return {r.key for r in records
            if r.status == "ok" and (tier is None or r.tier == tier)}


def audit_pairs(records: Iterable[PointRecord]) -> list[tuple[PointKey, float, float]]:
    """(key, L_production, L_audit) for every point holding both successful tiers."""
    prod: dict[PointKey, float] = {}
    aud: dict[PointKey, float] = {}
    for r in records:
        if r.status != "ok":
            continue
        d = prod if r.tier == "production" else aud
        if r.key not in d:
            d[r.key] = r.max_leakage
    return [(k, prod[k], aud[k]) for k in sorted(prod.keys() & aud.keys())]


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
        **keys_to_arrays(keys),
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


# ── Worker process entry point ───────────────────────────────────────────────
#
# The parent warms ARC, compiles/aggregates every detuning row, and stores the
# immutable context in module globals BEFORE creating the fork pool, so workers
# share it copy-on-write and never touch ARC's SQLite database.

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


_WORKER_CTX: dict = {}


def _worker_process_init() -> None:
    """Fork-pool initializer: workers ignore SIGINT (the parent coordinates a
    graceful drain on Ctrl-C) but keep the default SIGTERM so a hard abort can
    still terminate them immediately."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _set_worker_context(cfg: ScanConfig,
                        panel_ops: dict[int, PanelOperators], use_swap: bool,
                        gammas: dict[int, dict[str, float]] | None = None) -> None:
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        cfg=cfg, panel_ops=panel_ops, use_swap=use_swap, gammas=gammas)


def _worker_run_batch(spec: dict) -> dict:
    """Integrate one batch in a worker.  Never raises: failures are reported."""
    start = time.time()

    def _on_alarm(signum, frame):
        raise TimeoutError("batch exceeded its wall-clock timeout")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(spec.get("timeout_s") or 0))
    try:
        cfg: ScanConfig = _WORKER_CTX["cfg"]
        ops: PanelOperators = _WORKER_CTX["panel_ops"][spec["n_idx"]]
        keys = [PointKey(*k) for k in spec["keys"]]
        t_gate = cfg.t_gate_us[spec["t_idx"]] * 1e-6
        omega_297 = np.asarray([float(k.omega_mhz()) for k in keys]) * 1e6 * TAU
        d_sweep = np.asarray([float(k.dsweep_mhz()) for k in keys]) * 1e6 * TAU
        scatter = bool(spec.get("scatter"))
        t_eval = (np.linspace(0.0, t_gate, cfg.n_eval_trajectory)
                  if (spec.get("save_traj") or scatter) else None)
        result = integrate_batch(
            ops, t_gate, omega_297, d_sweep,
            rtol=spec["rtol"], atol=spec["atol"], ramp=cfg.ramp_frac,
            use_swap=_WORKER_CTX["use_swap"] and ops.swap_symmetric,
            t_eval=t_eval,
        )
        if not np.all(np.isfinite(result.psi_final.view(float))):
            raise FloatingPointError("non-finite terminal state")
        out = {"ok": True, "result": result, "runtime_s": time.time() - start}
        if scatter:
            # Gamma depends on the Rydberg n row: every key in a batch shares one
            # panel, so index the nested per-panel gammas by this batch's n row.
            spec_n_idx = spec["n_idx"]
            assert all(k.n_idx == spec_n_idx for k in keys), \
                "a scatter batch must stay within one Rydberg-n panel"
            out["scatter"] = scattering_integrals(
                result.times, result.states, _WORKER_CTX["gammas"][spec_n_idx])
            if not spec.get("save_traj"):
                result.times = None      # keep the return payload small: the
                result.states = None     # trajectory is consumed, not persisted
        return out
    except TimeoutError:
        return {"ok": False, "reason": "timeout",
                "message": f"timeout after {time.time() - start:.0f}s",
                "runtime_s": time.time() - start}
    except Exception as exc:  # noqa: BLE001 — worker must never crash the pool
        import traceback
        return {"ok": False, "reason": "failed",
                "message": f"{type(exc).__name__}: {exc}"[:240],
                "traceback": traceback.format_exc()[-2000:],
                "runtime_s": time.time() - start}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# ── Cost model, batches, and the budget-aware runner ─────────────────────────


@dataclass
class Batch:
    keys: list[PointKey]
    tier: str = "production"
    save_traj: bool = False
    scatter: bool = False       # scattering-supplement solve -> scatter/ series
    retry_count: int = 0        # failure-split escalations (this batch's points)
    pool_retries: int = 0       # innocent requeues after pool crashes (separate budget)
    priority_scores: list[float] | None = None
    batch_id: str = ""

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = uuid.uuid4().hex[:12]
        panels = {k.panel for k in self.keys}
        if len(panels) != 1:
            raise ValueError("a batch must stay within one (n, T) panel")
        (self.n_idx, self.t_idx), = panels


class CostModel:
    """Measured per-point runtimes -> deterministic panel-level ETA prediction."""

    DEFAULT_POINT_S = 150.0

    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        self.samples: dict[tuple[int, int], list[float]] = {}
        self.ratios: list[float] = []       # actual / predicted per completed batch

    def observe(self, panel: tuple[int, int], per_point_s: float) -> None:
        if np.isfinite(per_point_s) and per_point_s > 0:
            self.samples.setdefault(panel, []).append(per_point_s)

    def observe_batch(self, predicted_s: float, actual_s: float) -> None:
        if predicted_s > 0 and actual_s > 0:
            self.ratios.append(actual_s / predicted_s)

    def predict_point(self, panel: tuple[int, int]) -> float:
        if panel in self.samples:
            return float(np.median(self.samples[panel]))
        t_here = self.cfg.t_gate_us[panel[1]]
        same_row = [(np.median(v), self.cfg.t_gate_us[p[1]])
                    for p, v in self.samples.items() if p[0] == panel[0]]
        if same_row:
            med, t_ref = min(same_row, key=lambda mt: abs(mt[1] - t_here))
            return float(med * t_here / t_ref)
        anywhere = [(np.median(v), self.cfg.t_gate_us[p[1]])
                    for p, v in self.samples.items()]
        if anywhere:
            med, t_ref = min(anywhere, key=lambda mt: abs(mt[1] - t_here))
            return float(med * t_here / t_ref)
        return self.DEFAULT_POINT_S * t_here

    def inflation_p90(self) -> float:
        if len(self.ratios) < 5:
            return 1.5
        return float(max(1.0, np.percentile(self.ratios, 90)))

    def predict_batch(self, batch: Batch) -> float:
        return self.predict_point((batch.n_idx, batch.t_idx)) * len(batch.keys)

    def eta_seconds(self, keys: Iterable[PointKey], n_workers: int) -> float:
        total = sum(self.predict_point(k.panel) for k in keys)
        return total * self.inflation_p90() / max(1, n_workers)


def group_batches(keys: Sequence[PointKey], batch_size: int, tier: str = "production",
                  save_traj: bool = False,
                  scores: dict[PointKey, float] | None = None) -> list[Batch]:
    """Group keys into within-panel batches of adjacent points (axis row-major)."""
    by_panel: dict[tuple[int, int], list[PointKey]] = {}
    for k in keys:
        by_panel.setdefault(k.panel, []).append(k)
    batches = []
    for panel in sorted(by_panel):
        pts = sorted(by_panel[panel],
                     key=lambda k: (Fraction(k.om_num, k.om_den), Fraction(k.dw_num, k.dw_den)))
        for i in range(0, len(pts), max(1, batch_size)):
            chunk = pts[i:i + max(1, batch_size)]
            batches.append(Batch(
                keys=chunk, tier=tier, save_traj=save_traj,
                priority_scores=[scores.get(k, 0.0) for k in chunk] if scores else None))
    return batches


class Runner:
    """Submits batches to a fork pool, handles retries/splits, writes all chunks."""

    def __init__(self, store: Store, manifest: dict, cfg: ScanConfig,
                 args, cost: CostModel):
        self.store = store
        self.manifest = manifest
        self.cfg = cfg
        self.args = args
        self.cost = cost
        self._acquire_store_lock()
        self.seq = store.next_seq()
        self.scatter_seq = store.next_scatter_seq()
        self.gammas: dict[int, dict[str, float]] | None = None   # set for scatter runs
        self.stop_requested = False
        self.start_time = time.time()
        self.dispatch_deadline = (self.start_time
                                  + (args.budget_hours - args.reserve_hours) * 3600.0)
        self.point_timeout_s = float(args.point_timeout)
        self.failures: list[dict] = []
        self.deferred: int = 0
        self.completed_points = 0
        self.aborted = False
        self._signal_time = 0.0
        self._pool_restarts = 0
        self._executor = self._make_executor()
        self._install_signals()

    def _make_executor(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(
            max_workers=self.args.workers, mp_context=mp.get_context("fork"),
            initializer=_worker_process_init)

    def _acquire_store_lock(self) -> None:
        """Single-coordinator lock: chunk sequence numbers must not race.

        Opened append-mode so a *rejected* second launch cannot truncate the live
        coordinator's PID record; the PID/run id is written only after the flock
        succeeds.
        """
        import fcntl

        self.store.ensure_dirs()
        self._lock_fh = open(os.path.join(self.store.logs_dir, "store.lock"), "a+")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_fh.close()
            raise SystemExit(
                f"another pilot/run/audit already holds {self.store.root}; "
                "wait for it or use a different --output")
        self._lock_fh.seek(0)
        self._lock_fh.truncate()
        self._lock_fh.write(
            f"pid {os.getpid()} scan {self.manifest['scan_uuid']} "
            f"started {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        self._lock_fh.flush()

    def _install_signals(self):
        def handler(signum, frame):
            now = time.monotonic()
            # Debounce duplicate deliveries: `timeout`/terminals signal the whole
            # process group AND `uv run` forwards to its child, so one Ctrl-C can
            # arrive several times within milliseconds.  A deliberate second
            # Ctrl-C (>2 s later) escalates to a hard abort.
            if self.stop_requested and now - self._signal_time > 2.0:
                self.aborted = True
                raise KeyboardInterrupt
            if not self.stop_requested:
                self.stop_requested = True
                self._signal_time = now
                print(f"\n[signal {signum}] stopping dispatch; in-flight batches "
                      "will finish and checkpoint (repeat to abort hard)", flush=True)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def shutdown(self):
        # On a hard abort the in-flight results are already lost — don't block on
        # the workers (they ignore SIGINT), SIGTERM them so interpreter shutdown
        # doesn't join them for up to a full solve; on a clean stop everything is
        # drained, so wait is instant.
        if self.aborted:
            procs = list(getattr(self._executor, "_processes", {}).values())
            self._executor.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
        else:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._lock_fh.close()

    def _submit(self, batch: Batch):
        """Submit a batch spec; a broken pool (worker died) is recreated once per
        incident so a single crash cannot burn the remaining wall budget."""
        from concurrent.futures.process import BrokenProcessPool

        spec = self._spec(batch)
        try:
            return self._executor.submit(_worker_run_batch, spec)
        except BrokenProcessPool:
            self._pool_restarts += 1
            print(f"[pool] worker pool broke (restart #{self._pool_restarts}); "
                  "recreating", flush=True)
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = self._make_executor()
            return self._executor.submit(_worker_run_batch, spec)

    def _tier_tols(self, tier: str) -> tuple[float, float]:
        if tier == "audit":
            return self.cfg.rtol_audit, self.cfg.atol_audit
        return self.cfg.rtol_production, self.cfg.atol_production

    def _spec(self, batch: Batch) -> dict:
        rtol, atol = self._tier_tols(batch.tier)
        return dict(
            keys=[tuple(getattr(k, f) for f in _KEY_FIELDS) for k in batch.keys],
            n_idx=batch.n_idx, t_idx=batch.t_idx,
            tier=batch.tier, rtol=rtol, atol=atol,
            save_traj=batch.save_traj,
            scatter=batch.scatter,
            timeout_s=self.point_timeout_s * len(batch.keys),
        )

    def _write_success(self, batch: Batch, out: dict) -> None:
        rtol, atol = self._tier_tols(batch.tier)
        result: BatchResult = out["result"]
        if batch.scatter:
            # Supplemental data: its own append-only series; the coherent-leakage
            # chunks and trajectories are never touched by a scatter batch.
            self.store.write_scatter_chunk(
                self.scatter_seq, self.manifest, batch.keys, self.cfg,
                self.gammas, rtol, atol, batch.batch_id, out["scatter"],
                result.max_leakage, out["runtime_s"])
            self.scatter_seq += 1
            self.cost.observe((batch.n_idx, batch.t_idx),
                              out["runtime_s"] / len(batch.keys))
            self.completed_points += len(batch.keys)
            return
        self.store.write_result_chunk(
            self.seq, self.manifest, batch.keys, self.cfg,
            batch.tier, rtol, atol, batch.batch_id, result, out["runtime_s"],
            retry_count=batch.retry_count,
            priority_scores=batch.priority_scores,
        )
        self.seq += 1
        if batch.save_traj and result.states is not None:
            for i, key in enumerate(batch.keys):
                self.store.write_trajectory_chunk(
                    self.seq, self.manifest, key, batch.tier,
                    result.times, result.states[:, i])
                self.seq += 1
        per_point = out["runtime_s"] / len(batch.keys)
        self.cost.observe((batch.n_idx, batch.t_idx), per_point)
        self.completed_points += len(batch.keys)

    def _write_failure(self, batch: Batch, out: dict) -> None:
        rtol, atol = self._tier_tols(batch.tier)
        if batch.scatter:
            n = len(batch.keys)
            self.store.write_scatter_chunk(
                self.scatter_seq, self.manifest, batch.keys, self.cfg,
                self.gammas, rtol, atol, batch.batch_id,
                {ch: np.full((n, 4), np.nan) for ch in SCATTER_CHANNELS},
                np.full(n, np.nan), out.get("runtime_s", 0.0),
                statuses=[out.get("reason", "failed")] * n,
                message=out.get("message", ""))
            self.scatter_seq += 1
        else:
            self.store.write_result_chunk(
                self.seq, self.manifest, batch.keys, self.cfg,
                batch.tier, rtol, atol, batch.batch_id, None,
                out.get("runtime_s", 0.0),
                statuses=[out.get("reason", "failed")] * len(batch.keys),
                message=out.get("message", ""), retry_count=batch.retry_count,
                priority_scores=batch.priority_scores,
            )
            self.seq += 1
        self.failures.append({
            "batch_id": batch.batch_id, "keys": [k.id() for k in batch.keys],
            "tier": batch.tier, "reason": out.get("reason", "failed"),
            "message": out.get("message", ""), "retry_count": batch.retry_count,
        })

    def _split(self, batch: Batch) -> list[Batch]:
        half = len(batch.keys) // 2
        parts = []
        for sl in (slice(0, half), slice(half, None)):
            keys = batch.keys[sl]
            if keys:
                parts.append(Batch(
                    keys=keys, tier=batch.tier, save_traj=batch.save_traj,
                    scatter=batch.scatter,
                    retry_count=batch.retry_count + 1,
                    priority_scores=(batch.priority_scores[sl]
                                     if batch.priority_scores else None)))
        return parts

    def run_batches(self, batches: list[Batch], phase: str,
                    enforce_deadline: bool = True,
                    preserve_order: bool = False) -> None:
        """Run batches (longest-predicted first unless ``preserve_order``); split
        failures recursively; write every chunk from this (parent) process."""
        from concurrent.futures import FIRST_COMPLETED, wait

        if preserve_order:
            pending = deque(batches)
        else:
            pending = deque(sorted(batches, key=self.cost.predict_batch, reverse=True))
        inflight: dict = {}
        # Keep the executor's internal queue shallow: the deadline check runs at
        # submit time, so a deep queue could carry a whole extra wave of batches
        # past the reserve boundary.
        max_outstanding = self.args.workers + max(2, self.args.workers // 8)
        n_total = sum(len(b.keys) for b in batches)
        n_done = 0
        t0 = time.time()
        print(f"[{phase}] {len(batches)} batches / {n_total} points", flush=True)

        while pending or inflight:
            while pending and len(inflight) < max_outstanding and not self.stop_requested:
                batch = pending.popleft()
                # A batch occupies one core; its predicted wall time is the P90-
                # inflated single-core estimate.  Anything that cannot finish
                # before the dispatch deadline is deferred, keeping the reserve.
                predicted = self.cost.predict_batch(batch) * self.cost.inflation_p90()
                if enforce_deadline and time.time() + predicted > self.dispatch_deadline:
                    self.deferred += len(batch.keys)
                    continue
                fut = self._submit(batch)
                inflight[fut] = (batch, self.cost.predict_batch(batch))
            if not inflight:
                break
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                batch, predicted = inflight.pop(fut)
                try:
                    out = fut.result()
                except Exception as exc:
                    # In-flight victims of a pool crash are innocent: requeue them
                    # on their own bounded budget (separate from failure splits) so
                    # repeated worker deaths cannot mark never-computed points as
                    # failed, while a deterministically pool-killing point is still
                    # isolated by splitting once its own requeues are exhausted.
                    if batch.pool_retries < 5:
                        requeued = Batch(
                            keys=batch.keys, tier=batch.tier,
                            save_traj=batch.save_traj, scatter=batch.scatter,
                            retry_count=batch.retry_count,
                            pool_retries=batch.pool_retries + 1,
                            priority_scores=batch.priority_scores)
                        pending.appendleft(requeued)
                        print(f"[{phase}] batch {batch.batch_id} lost to a pool "
                              f"error ({exc}); requeued "
                              f"({requeued.pool_retries}/5)", flush=True)
                        continue
                    out = {"ok": False, "reason": "failed",
                           "message": f"pool error: {exc}"[:240], "runtime_s": 0.0}
                if out["ok"]:
                    self._write_success(batch, out)
                    self.cost.observe_batch(predicted, out["runtime_s"])
                    n_done += len(batch.keys)
                elif len(batch.keys) > 1:
                    print(f"[{phase}] batch {batch.batch_id} "
                          f"{out.get('reason')}: splitting {len(batch.keys)} points",
                          flush=True)
                    for part in self._split(batch):
                        pending.appendleft(part)
                else:
                    self._write_failure(batch, out)
                    n_done += 1
                    print(f"[{phase}] point {batch.keys[0].id()} "
                          f"{out.get('reason')}: {out.get('message', '')}", flush=True)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 and n_done else 0.0
            eta = (n_total - n_done) / rate if rate > 0 else float("nan")
            print(f"[{phase}] {n_done}/{n_total} points "
                  f"({elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left)",
                  flush=True)
            self.write_status(phase)
            # when stop was requested, the submit loop stays closed and the outer
            # loop simply drains the in-flight futures before returning
        if self.stop_requested:
            print(f"[{phase}] stopped on request; "
                  f"{sum(len(b.keys) for b in pending)} points left unscheduled",
                  flush=True)

    def write_status(self, phase: str) -> None:
        status = {
            "phase": phase,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": time.time() - self.start_time,
            "dispatch_deadline_in_s": self.dispatch_deadline - time.time(),
            "completed_points_this_run": self.completed_points,
            "deferred_points": self.deferred,
            "failures": len(self.failures),
            "inflation_p90": self.cost.inflation_p90(),
            "workers": self.args.workers,
        }
        path = os.path.join(self.store.reports_dir, "status.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(status, fh, indent=2)
        os.replace(path + ".tmp", path)

    def write_failure_report(self) -> None:
        path = os.path.join(self.store.reports_dir, "failures.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(self.failures, fh, indent=2)
        os.replace(path + ".tmp", path)


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
            "budget_hours": getattr(args, "budget_hours", None),
            "reserve_hours": getattr(args, "reserve_hours", None),
        })
    ver_path = os.path.join(store.reports_dir, "verification.json")
    with open(ver_path + ".tmp", "w") as fh:
        json.dump(checks, fh, indent=2)
    os.replace(ver_path + ".tmp", ver_path)
    _set_worker_context(cfg, ops, use_swap=checks["swap_symmetric"],
                        gammas=checks["decay_rates_rad_s"])
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


def bench_worker_counts(runner: Runner, counts: list[int]) -> dict:
    """Measure *saturated* end-to-end throughput at several worker counts.

    Each configuration solves exactly ``count`` distinct, still-missing level-0
    nodes from the cheapest (T = 1 us) panel column — one full wave, so every
    worker is busy and the wall time reflects real scaling.  Points run isolated
    (batch size 1: solver throughput, not packing) and are persisted as
    authoritative production records, so the benchmark is useful scan work.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    records = runner.store.load_records(runner.manifest, include_states=False)
    done = completed_keys(records)
    # Order by in-panel position first so every configuration draws a similar
    # (Omega, D) cost mix; the T = 1 us column first, then T = 1.2 us as
    # overflow (the two cheapest columns together always cover 116 bench nodes).
    node_pool: deque[PointKey] = deque()
    for t_idx in (0, 1):
        col = [k for k in all_keys(0) if k.t_idx == t_idx and k not in done]
        col.sort(key=lambda k: (Fraction(k.om_num, k.om_den),
                                Fraction(k.dw_num, k.dw_den), k.n_idx))
        node_pool.extend(col)

    out: dict[str, dict] = {}
    for n in counts:
        take = [node_pool.popleft() for _ in range(min(n, len(node_pool)))]
        if len(take) < n:
            out[str(n)] = {"points": len(take),
                           "error": "not enough missing T=1us level-0 nodes"}
            continue
        ex = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork"),
                                 initializer=_worker_process_init)
        t0 = time.time()
        ok = 0
        try:
            futs = {ex.submit(_worker_run_batch, runner._spec(b)): b
                    for b in (Batch(keys=[k]) for k in take)}
            for fut, b in futs.items():
                res = fut.result()
                if res.get("ok"):
                    runner._write_success(b, res)
                    ok += 1
        except Exception as exc:  # a bench pool crash is a data point, not fatal
            out[str(n)] = {"points": len(take), "error": str(exc)[:200]}
            ex.shutdown(wait=False, cancel_futures=True)
            continue
        wall = time.time() - t0
        ex.shutdown()
        out[str(n)] = {"points": len(take), "ok": ok, "wall_s": wall,
                       "points_per_hour": ok / wall * 3600.0,
                       "point_ids": [k.id() for k in take]}
        print(f"[bench] {n} workers: {ok}/{len(take)} pts in {wall:.0f} s "
              f"({out[str(n)]['points_per_hour']:.0f} pts/h)", flush=True)
    return out


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

    bench = None
    if os.path.exists(pilot_path):
        try:
            with open(pilot_path) as fh:
                bench = json.load(fh).get("worker_benchmark")
        except (OSError, json.JSONDecodeError):
            bench = None
    if (getattr(runner.args, "bench_workers", None) and bench is None
            and not runner.stop_requested):
        counts = [int(v) for v in runner.args.bench_workers.split(",")]
        bench = bench_worker_counts(runner, counts)

    records = runner.store.load_records(runner.manifest, include_states=False)
    pairs = audit_pairs(records)
    per_panel = {f"{p[0]},{p[1]}": float(np.median(v))
                 for p, v in sorted(runner.cost.samples.items())}
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_pilot_points": len(pkeys),
        "packing_gate": gate,
        "worker_benchmark": bench,
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


# ── Adaptive refinement priorities ───────────────────────────────────────────

REFINE_RESIDUAL_DEX = 0.25
REFINE_CONTOUR_RESIDUAL_DEX = 0.10
DECISION_CONTOURS = (1e-3, 1e-2, 1e-4)


def _mid_coord(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    p = (Fraction(*a) + Fraction(*b)) / 2
    return canon_coord(p.numerator, p.denominator)


def refinement_candidates(
    leak_by_key: dict[PointKey, float],
    worst_by_key: dict[PointKey, str],
    level: int,
    floor: float = 1e-12,
) -> list[tuple[PointKey, float]]:
    """Prioritized missing level-(level+1) nodes from the completed level grid.

    Deterministic: cells are scored in ``z = log10(max(L, floor))`` by decision-
    contour crossings, worst-input identity changes, cell z-range, and an
    axis-neighbor leave-one-out residual; nodes inherit their best cell score.
    Panels whose level grid is incomplete are skipped (they are still owed their
    uniform pass).
    """
    if level + 1 >= len(LEVEL_DENS):
        return []
    coords = axis_coords(level)
    n = len(coords)
    scored: dict[PointKey, float] = {}

    for panel in all_panels():
        z = np.full((n, n), np.nan)
        worst = np.full((n, n), "", dtype="U2")
        for i, om in enumerate(coords):
            for j, dw in enumerate(coords):
                k = make_key(panel[0], panel[1], om, dw)
                if k in leak_by_key:
                    z[i, j] = math.log10(max(leak_by_key[k], floor))
                    worst[i, j] = worst_by_key.get(k, "")
        if not np.all(np.isfinite(z)):
            continue

        loo = np.zeros((n, n))
        loo[1:-1, :] = np.abs(z[1:-1, :] - 0.5 * (z[:-2, :] + z[2:, :]))
        loo[:, 1:-1] = np.maximum(
            loo[:, 1:-1], np.abs(z[:, 1:-1] - 0.5 * (z[:, :-2] + z[:, 2:])))

        for i in range(n - 1):
            for j in range(n - 1):
                zc = z[i:i + 2, j:j + 2]
                zmin, zmax = float(zc.min()), float(zc.max())
                dz = zmax - zmin
                resid = float(loo[i:i + 2, j:j + 2].max())
                crosses_decision = zmin <= math.log10(DECISION_CONTOURS[0]) <= zmax
                crosses_secondary = any(
                    zmin <= math.log10(c) <= zmax for c in DECISION_CONTOURS[1:])
                labels = set(worst[i:i + 2, j:j + 2].ravel())
                # Spec priority order: primary contour > holdout residual >
                # secondary contours > worst-input change > residual/gradient.
                score = (8.0 * crosses_decision
                         + min(6.0, 3.0 * resid / REFINE_RESIDUAL_DEX)
                         + 3.0 * crosses_secondary
                         + 2.0 * (len(labels) > 1) + min(dz, 2.0))
                needs = (crosses_decision or crosses_secondary
                         or dz > REFINE_RESIDUAL_DEX or resid > REFINE_RESIDUAL_DEX)
                if crosses_decision and (dz > REFINE_CONTOUR_RESIDUAL_DEX
                                         or resid > REFINE_CONTOUR_RESIDUAL_DEX):
                    needs = True
                if not needs:
                    continue
                om0, om1 = coords[i], coords[i + 1]
                dw0, dw1 = coords[j], coords[j + 1]
                omm, dwm = _mid_coord(om0, om1), _mid_coord(dw0, dw1)
                cell_nodes = [
                    make_key(panel[0], panel[1], omm, dwm),
                    make_key(panel[0], panel[1], omm, dw0),
                    make_key(panel[0], panel[1], omm, dw1),
                    make_key(panel[0], panel[1], om0, dwm),
                    make_key(panel[0], panel[1], om1, dwm),
                ]
                for nk in cell_nodes:
                    if nk not in leak_by_key:
                        scored[nk] = max(scored.get(nk, 0.0), score)

    return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))


# ── Stage orchestration and CLI commands ─────────────────────────────────────


def _leak_maps(records: list[PointRecord]) -> tuple[dict, dict]:
    best = best_records(records)
    return ({k: r.max_leakage for k, r in best.items()},
            {k: r.worst_input for k, r in best.items()})


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

    target = args.target_level
    max_level = (len(LEVEL_SIZES) - 1 if target == "auto"
                 else LEVEL_FROM_SIZE[int(target)])

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
    try:
        stage_pilot(runner, panels)
        batch_size = _effective_batch_size(store, args)
        print(f"[run] effective batch size: {batch_size}", flush=True)

        for level in (0, 1):
            if runner.stop_requested or level > max_level:
                break
            records = store.load_records(manifest, include_states=False)
            done = completed_keys(records)
            _run_level(runner, level, done, batch_size, args.rerun_failures,
                       failed, panels)

        # Stage 4/5: full 13x13 only if the measured P90 ETA fits the compute
        # deadline, else high-value 13-level nodes by deterministic priority.
        if not runner.stop_requested and max_level >= 2:
            records = store.load_records(manifest, include_states=False)
            done = completed_keys(records)
            missing2 = [k for k in _filter_panels(all_keys(2), panels)
                        if k not in done]
            eta2 = cost.eta_seconds(missing2, args.workers)
            fits = time.time() + eta2 <= runner.dispatch_deadline
            print(f"[run] level 13 decision: {len(missing2)} nodes, P90 ETA "
                  f"{eta2 / 3600:.2f} h, deadline in "
                  f"{(runner.dispatch_deadline - time.time()) / 3600:.2f} h -> "
                  f"{'full grid' if fits or target != 'auto' else 'high-value cells'}",
                  flush=True)
            if fits or target != "auto":
                _run_level(runner, 2, done, batch_size, args.rerun_failures,
                           failed, panels)
            else:
                leak, worst = _leak_maps(records)
                vmin, _ = _credibility_floor(records, cfg.credibility_floor_min)
                cand = refinement_candidates(leak, worst, level=1, floor=vmin)
                cand = [(k, s) for k, s in cand
                        if k not in done and (panels is None or k.panel in panels)]
                scores = dict(cand)
                batches = group_batches([k for k, _ in cand], batch_size,
                                        scores=scores)
                batches.sort(key=lambda b: -max(b.priority_scores or [0.0]))
                runner.run_batches(batches, "level-13-refine", preserve_order=True)

        # Bonus stage: adaptive level-25 nodes in high-value cells if the full
        # 13x13 grid finished and budget remains.
        if not runner.stop_requested and max_level >= 3:
            records = store.load_records(manifest, include_states=False)
            done = completed_keys(records)
            if all(k in done for k in _filter_panels(all_keys(2), panels)):
                leak, worst = _leak_maps(records)
                vmin, _ = _credibility_floor(records, cfg.credibility_floor_min)
                cand = [(k, s) for k, s in
                        refinement_candidates(leak, worst, level=2, floor=vmin)
                        if k not in done and (panels is None or k.panel in panels)]
                if target != "auto":
                    cand = [(k, 0.0)
                            for k in _filter_panels(all_keys(3), panels)
                            if k not in done]
                scores = dict(cand)
                batches = group_batches([k for k, _ in cand], batch_size,
                                        scores=scores)
                batches.sort(key=lambda b: -max(b.priority_scores or [0.0]))
                if batches:
                    runner.run_batches(batches, "level-25-refine", preserve_order=True)
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
            key = arrays_to_keys(d)[0]
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
        gate = _scatter_equivalence_gate(runner, store)
        path = os.path.join(store.reports_dir, "scatter_gate.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(gate, fh, indent=2)
        os.replace(path + ".tmp", path)
        print(f"[scatter] trajectory-equivalence gate: {gate}", flush=True)
        if not gate["ok"]:
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
        scattering = r["p_ryd"] + r["p_r_garb"]
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
    ryd_axis = manifest["axes"]["ryd_n"]
    tg = manifest["axes"]["t_gate_us"]

    n_rows, n_cols = len(ryd_axis), len(tg)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols + 1.6, 1.9 * n_rows + 1.2),
                             sharex=True, sharey=True, constrained_layout=True)
    mesh = None
    for ni in range(n_rows):
        for ti in range(n_cols):
            ax = axes[ni][ti]
            data = _panel_plot_data(values, (ni, ti), vmin)
            if data is None:
                ax.set_facecolor("0.92")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.4")
            else:
                mesh = _draw_panel(ax, *data, vmin, vmax, cmap,
                                   veil=args.veil) or mesh
            if ni == 0:
                ax.set_title(f"T = {tg[ti]:g} us", fontsize=9)
            if ni == n_rows - 1:
                ax.set_xlabel(r"$\Omega_{297}/2\pi$ (MHz)", fontsize=8)
            if ti == 0:
                ax.set_ylabel(f"$n$ = {ryd_axis[ni]:g}\n"
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
        f"{metric_title}, two-atom 297 nm single-photon CZ ({dynamics_note}, "
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
    return os.path.join("results", "max_leakage_297", f"a{spacing_um:.1f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="max_leakage_297_sweep",
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
                             "results/max_leakage_297/a{spacing:.1f})")
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
            sp.add_argument("--budget-hours", type=float, default=24.0)
            sp.add_argument("--reserve-hours", type=float, default=2.0)
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
    sp.add_argument("--bench-workers", default="20,24,32,40", metavar="N,N,...",
                    help="worker counts to benchmark during the pilot "
                         "(runs once per store; pass '' to skip)")
    sp.set_defaults(func=cmd_pilot)

    sp = sub.add_parser("run", help="resumable staged scan under a wall budget")
    common(sp, compute=True)
    sp.add_argument("--target-level", default="auto",
                    choices=["4", "7", "13", "25", "auto"])
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
                    choices=["max_leakage", "p_ryd", "p_r_garb",
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
