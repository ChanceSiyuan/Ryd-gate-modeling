"""The 7-level ⇄ {0,1,r} effective-theory map (2nd-order Schrieffer–Wolff).

This module is *the bridge* between the two levels at which a Rb87 CZ gate is
modeled — the full seven-level ``rb87_7`` ladder
``{0, 1, e1, e2, e3, r, r_garb}`` (driven by the 420 nm and 1013 nm lasers) and
the effective three-level ``{0, 1, r}`` model obtained by adiabatically
eliminating the far-detuned ``6P`` states.  It exists so the relationship is a
*readable map*, not a ``compensate_stark`` switch buried inside a protocol.

The physics is derived in ``Rydberg_sim.tex`` (Theorem 1 + Lemma 1) and validated
numerically in ``scripts/notebooks/find_phase.ipynb`` §4.  Public surface:

- :func:`lower_cz_to_effective_01r` — the complete public converter: an arbitrary
  rb87_7 CZ pulse (any ``CZProtocol``-style protocol) → an
  :class:`~ryd_gate.protocols.gate_cz.EffectiveCZProtocol` on the ``01r`` model.
  At each ``t`` it rebuilds ``H7(t)`` from the registered blocks and does the
  *two-stage* reduction (eliminate ``e1/e2/e3``, then eliminate ``r_garb/r'``),
  producing the full 3x3 ``{0,1,r}`` Hamiltonian incl. ``K0r``.
- :func:`schrieffer_wolff` — the 2nd-order (Löwdin) projection of a single-atom
  block onto the kept levels (one reduction stage).  This *is* find_phase §4.2.
- :func:`shift_coefficients` — the diagonal AC-Stark / light shifts the projection
  induces, read straight off the registered ``rb87_7`` blocks.
- :func:`reverse_amplitude_split` — the inverse direction: a target effective Rabi
  ``Ω_eff(t)`` and a 420/1013 power-split choice → the per-laser amplitudes.

The reduction is *exact* against the instantaneous two-stage Löwdin projection (the
matrix-level converter test pins this to machine precision), but only correct to
2nd order in ``Ω/Δ_e`` against the full 7-level dynamics; find_phase §4 measures the
residual (single-qubit phase to ~3e-3 rad, ZZ phase to ~2%).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

# |0>-|1> Rb87 clock hyperfine splitting; |0> sits at -EPS0 in H_const, so the
# off-resonant |0>->|e> 420 leg is detuned by Δ_e + EPS0 rather than Δ_e.  Kept
# here for documentation only — the formulas below read the actual energy from
# the H_const diagonal, so they carry detuning_sign / overrides for free.
EPS0 = 2 * np.pi * 6.835e9

# rb87_7 local-basis layout (the only level structure this map applies to).
_QUBIT_1 = 1
_QUBIT_0 = 0
_RYD = 5
_RYD_GARB = 6
_MID = (2, 3, 4)


def schrieffer_wolff(
    h_local: np.ndarray,
    keep_idx: Sequence[int],
    elim_idx: Sequence[int],
    bare_energies: Sequence[float] | None = None,
) -> np.ndarray:
    """Second-order symmetric (Löwdin) projection of a single-atom Hamiltonian.

    Eliminates ``elim_idx`` and returns the effective block on ``keep_idx``:

        H_eff[a,b] = H[a,b] + ½ Σ_q H[a,q] H[q,b] (1/(E_a-E_q) + 1/(E_b-E_q)).

    ``bare_energies`` are the unperturbed level energies entering the
    denominators; they default to ``Re(diag(h_local))`` (so a Hamiltonian built
    with any ``detuning_sign`` / laser override is handled automatically).  This
    is the construction of ``Rydberg_sim.tex`` Theorem 1 and find_phase §4.2.
    """
    h = np.asarray(h_local)
    keep = list(keep_idx)
    elim = list(elim_idx)
    energy = (
        np.real(np.diag(h)) if bare_energies is None else np.asarray(bare_energies, dtype=float)
    )
    eff = h[np.ix_(keep, keep)].astype(complex).copy()
    for ai, a in enumerate(keep):
        for bi, b in enumerate(keep):
            corr = 0j
            for q in elim:
                corr += h[a, q] * h[q, b] * 0.5 * (
                    1.0 / (energy[a] - energy[q]) + 1.0 / (energy[b] - energy[q])
                )
            eff[ai, bi] += corr
    return eff


def shift_coefficients(
    h_const: np.ndarray,
    h420: np.ndarray,
    h1013: np.ndarray,
) -> dict[str, float]:
    """Diagonal AC-Stark light shifts the eliminated ``6P`` manifold induces.

    These are the ``a == b`` entries of :func:`schrieffer_wolff` for the
    ``rb87_7`` layout, read directly off the registered blocks (``H_const``,
    ``drive_420``, the 1013 nm coupling).  Each is the shift on its level at the
    system's *nominal* laser amplitudes:

        D_a = Σ_e |coupling(a, e)|² / (E_a - E_e),   e ∈ {e1, e2, e3}.

    Scaling for the reverse map: ``D0, D1 ∝ Ω_420²`` and ``Dr, Dr_garb ∝
    Ω_1013²``.  (``D1`` and ``Dr`` are the old ``stark_1_per_amp2`` / ``stark_r``.)
    The ``|0>`` denominator carries the clock splitting via ``E_0 = -EPS0``.
    """
    hc = np.asarray(h_const)
    h420 = np.asarray(h420)
    h1013 = np.asarray(h1013)
    if hc.shape[0] < 7 or h420.shape[0] < 7 or h1013.shape[0] < 7:
        raise ValueError("shift_coefficients targets the rb87_7 (7-level) layout.")
    energy = np.real(np.diag(hc))
    if np.any(np.abs(energy[list(_MID)]) < 1e-15):
        raise ValueError("Intermediate-state energy denominator is zero.")

    def lightshift(level: int, coupling: np.ndarray) -> float:
        return float(
            np.real(sum(abs(coupling[e]) ** 2 / (energy[level] - energy[e]) for e in _MID))
        )

    return {
        "D0": lightshift(_QUBIT_0, h420[:, _QUBIT_0]),   # |0> 420 shift (∝ Ω_420²)
        "D1": lightshift(_QUBIT_1, h420[:, _QUBIT_1]),   # |1> 420 shift (∝ Ω_420²)
        "Dr": lightshift(_RYD, h1013[_RYD, :]),          # |r> 1013 shift (∝ Ω_1013²)
        "Dr_garb": lightshift(_RYD_GARB, h1013[_RYD_GARB, :]),  # |r'> 1013 shift (∝ Ω_1013²)
    }


def reverse_amplitude_split(
    omega_eff: Callable[[float], float],
    *,
    omega_eff_nom: float,
    hold: str = "1013",
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Reverse map: target ``Ω_eff(t)`` → per-laser amplitude envelopes.

    Returns ``(alpha, beta)`` callables, the dimensionless 420 / 1013 amplitudes
    relative to nominal: ``alpha = Ω_420(t)/Ω_420_nom``, ``beta =
    Ω_1013(t)/Ω_1013_nom``.  Since ``Ω_eff ∝ Ω_420·Ω_1013``, only the *product*
    ``alpha·beta = Ω_eff(t)/Ω_eff_nom`` is fixed; the split is one free knob:

    - ``hold="1013"`` (default, matches find_phase): ``beta≡1``, modulate 420.
    - ``hold="420"``: ``alpha≡1``, modulate 1013.
    - ``hold="balanced"``: ``alpha=beta=sqrt(ratio)`` — gentler light-shift swings.
    """
    if omega_eff_nom == 0.0:
        raise ValueError("omega_eff_nom must be non-zero.")

    def ratio(t: float) -> float:
        return float(omega_eff(t)) / float(omega_eff_nom)

    if hold == "1013":
        return (lambda t: ratio(t), lambda t: 1.0)
    if hold == "420":
        return (lambda t: 1.0, lambda t: ratio(t))
    if hold == "balanced":
        return (lambda t: float(np.sqrt(max(ratio(t), 0.0))),) * 2
    raise ValueError(f"hold must be '1013', '420', or 'balanced'; got {hold!r}.")


# rb87_7 -> {0,1,r} two-stage reduction indices (Rydberg_sim.tex Thm 1 + Lemma 1).
_SW_KEEP_E = [_QUBIT_0, _QUBIT_1, _RYD, _RYD_GARB]  # [0,1,r,r'] after eliminating |e>
_SW_ELIM_E = list(_MID)                              # eliminate {e1,e2,e3}
_SW_KEEP_RGARB = [0, 1, 2]                           # [0,1,r] in the 4-level layout
_SW_ELIM_RGARB = [3]                                 # eliminate r' (= r_garb)


def lower_cz_to_effective_01r(protocol, system7, *, n_steps: int | None = None):
    """Lower an rb87_7 CZ pulse to an effective ``{0,1,r}`` protocol.

    Maps *protocol* — any concrete rb87_7 laser protocol that exposes
    ``get_drive_coefficients`` (e.g. a :class:`~ryd_gate.protocols.gate_cz.CZProtocol`,
    including one from a ``TOProtocol``/``ARProtocol`` builder) — onto an
    :class:`~ryd_gate.protocols.gate_cz.EffectiveCZProtocol` that drives the full
    3x3 effective Hamiltonian on the ``01r`` model.

    At each time ``t`` it reads ``c420(t), c1013(t)`` off *protocol*, rebuilds the
    single-atom ``H7(t)`` from the registered ``H_const`` / ``drive_420`` /
    ``drive_1013`` blocks, and applies the two-stage Schrieffer–Wolff reduction:
    first eliminate the ``6P`` manifold ``{e1,e2,e3}``, then eliminate the garbage
    Rydberg ``r_garb`` (= ``r'``).  The result is the full ``{0,1,r}`` Hamiltonian
    ``D0,D1,Dr`` + ``K01,K0r,K1r`` (``K0r`` included; the ``r'`` 2nd-order
    corrections folded in by the second stage).

    Exact-backend only (the ``K0r`` / ``K01`` legs are not supported by the TN
    01r lowering).  The conversion is exact vs the instantaneous reduction; the
    resulting model matches the full 7-level only to 2nd order (see module doc).
    """
    if not hasattr(protocol, "get_drive_coefficients"):
        raise TypeError(
            f"{type(protocol).__name__} is a pulse *builder*, not a concrete pulse. "
            "Call protocol.build(system7) first and pass the resulting CZProtocol."
        )

    from ryd_gate.protocols.gate_cz import EffectiveCZProtocol

    blocks = system7.blocks
    hc = np.asarray(blocks.get("H_const").matrix)
    h420 = np.asarray(blocks.get("drive_420").matrix)
    h1013 = np.asarray(blocks.get("drive_1013").matrix)
    if hc.shape[0] < 7:
        raise ValueError("lower_cz_to_effective_01r targets the rb87_7 (7-level) layout.")
    h420_dag = h420.conj().T
    h1013_dag = h1013.conj().T

    params = protocol.unpack_params([], system7)
    scale = float(getattr(system7, "amplitude_scale", 1.0))

    def h7(t: float) -> np.ndarray:
        coeffs = protocol.get_drive_coefficients(float(t), params)
        c420 = scale * coeffs["drive_420"]
        c1013 = scale * coeffs.get("drive_1013", 0.0)
        return (
            hc
            + c420 * h420 + np.conjugate(c420) * h420_dag
            + c1013 * h1013 + np.conjugate(c1013) * h1013_dag
        )

    def h_eff(t: float) -> np.ndarray:
        h4 = schrieffer_wolff(h7(t), _SW_KEEP_E, _SW_ELIM_E)
        return schrieffer_wolff(h4, _SW_KEEP_RGARB, _SW_ELIM_RGARB)

    return EffectiveCZProtocol(
        t_gate=params["t_gate"],
        h_eff=h_eff,
        n_steps=n_steps if n_steps is not None else getattr(protocol, "n_steps", 200),
        has_K01=True,
        has_K0r=True,
    )
