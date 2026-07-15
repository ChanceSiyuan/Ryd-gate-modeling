"""The 7-level ⇄ {0,1,r} effective-theory map (resolvent block elimination).

This module is *the bridge* between the two levels at which a Rb87 CZ gate is
modeled — the full seven-level rb87 ladder (``rb87_7_mp`` / ``rb87_7_pm``)
``{0, 1, e1, e2, e3, r, r_garb}`` (driven by the 420 nm and 1013 nm lasers) and
the effective ``{0, 1, r}`` model obtained by eliminating the complementary
manifold.  It exists so the relationship is a *readable map*, not a
``compensate_stark`` switch buried inside a protocol.

The physics is derived in ``Rydberg_sim.tex`` (Theorems 1–2 + Lemma 1) and
validated numerically in ``scripts/notebooks/find_phase.ipynb`` §4.  The method
is the *one-step resolvent elimination*: keep ``P``, eliminate ``Q`` with the
exact block resolvent ``R_a = (H_Q - E_a)^{-1}``:

    D_a = -<v_a| R_a |v_a>,     K_ab = -1/2 <v_a| (R_a + R_b) |v_b>,

with ``|v_a> = Q H |a>``.  The ``e_F <-> r'`` couplings live *inside* ``H_Q``,
so virtual paths through ``r'`` are resummed by the resolvent rather than
treated in a separate (uncontrolled) local elimination.  The construction is
carried out in the tex rotating frame — real optical couplings, the laser
chirps moved onto the ``{e, r, r'}`` diagonal (``E_r = -Δ_add(t)``) — so the
denominators carry the instantaneous detunings exactly as in the tex.

Public surface:

- :func:`lower_cz_to_effective_01r` — single-atom converter (tex Theorem 1): an
  arbitrary rb87 seven-level CZ pulse (any ``CZProtocol``-style protocol) → a
  :class:`~ryd_gate.protocols.digital_analog.DigitalAnalogProtocol` on the
  ``01r`` model (full 3x3 incl. ``K0r``; tex-frame convention).  Reduction is through 4th
  order: :func:`resolvent_elimination` + :func:`vod4_s_terms`.  NOTE: the
  single-atom r' elimination is blind to the Rydberg interaction — in the
  ``|11>`` blockaded sector the r'-induced ``|1>`` light shift is suppressed by
  the ``v n_r n_r'`` blockade, so single-atom lowering carries a large ZZ-phase
  error at strong drive.  Use the pair converter for two-atom gate phases.
- :func:`vod4_s_terms` — the 4th-order ``V_od^4`` correction (tex Thm 1
  ``C^(4,S^2)/C^(4,S^3)``) that the 2nd-order resolvent misses.
- :func:`lower_cz_to_effective_pair` — two-atom converter (tex Theorem 2): the
  same one-step resolvent elimination applied to the interacting two-atom
  Hamiltonian (``P`` = the 9 two-qutrit states).  ``H_Q`` contains the
  interaction, so virtual states like ``|r'r>`` carry the blockade shift in
  their denominators — this is what fixes the ZZ phase.  Returns an
  :class:`EffectivePairModel` (a 9x9 ``h_eff(t)``; evolve it directly — the
  compile pipeline has no driven pair channels).
- :func:`resolvent_elimination` — the generic one-step reduction used by both.
- :func:`schrieffer_wolff` — the plain 2nd-order (Löwdin) projection with
  diagonal denominators; the closed-form limit of the resolvent method when
  ``H_Q`` is diagonal (kept for the Theorem-1 closed-form cross-checks).
- :func:`shift_coefficients` — the diagonal AC-Stark / light shifts of the
  ``6P`` elimination, read straight off the registered blocks.
- :func:`reverse_amplitude_split` — the inverse direction: a target effective
  Rabi ``Ω_eff(t)`` and a 420/1013 power-split choice → per-laser amplitudes.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ryd_gate.core.operators import parse_E

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


def resolvent_elimination(h, keep_idx: Sequence[int]) -> np.ndarray:
    """One-step block elimination with the exact ``Q``-block resolvent.

    Keeps ``keep_idx`` (``P``), eliminates the complement (``Q``), and returns

        H_eff[a,b] = H[a,b] - 1/2 <v_a| (R_a + R_b) |v_b>,

    with ``|v_a> = H[Q, a]``, ``E_a = Re H[a,a]``, and ``R_a = (H_Q - E_a)^{-1}``
    (``a == b`` gives the diagonal ``-<v_a|R_a|v_a>``).  This is the reduction of
    ``Rydberg_sim.tex`` Theorems 1 (single atom) and 2 (interacting pair): unlike
    :func:`schrieffer_wolff`, the internal couplings of the eliminated block
    (``e_F <-> r'``, and for a pair the blockade shifts of ``|r'r>``-type virtual
    states) enter through the resolvent instead of being dropped.  Reduces to
    :func:`schrieffer_wolff` exactly when ``H_Q`` is diagonal.
    """
    h = np.asarray(h, dtype=complex)
    keep = list(keep_idx)
    elim = [q for q in range(h.shape[0]) if q not in keep]
    hq = h[np.ix_(elim, elim)]
    v = h[np.ix_(elim, keep)]                  # column a is |v_a>
    energy = np.real(np.diag(h))
    ident = np.eye(len(elim))
    # R_a |v_a> for each kept state (one dense solve per kept state).
    rv = [
        np.linalg.solve(hq - energy[a] * ident, v[:, ai])
        for ai, a in enumerate(keep)
    ]
    eff = h[np.ix_(keep, keep)].astype(complex).copy()
    for ai in range(len(keep)):
        for bi in range(len(keep)):
            eff[ai, bi] -= 0.5 * (np.vdot(v[:, ai], rv[bi]) + np.vdot(rv[ai], v[:, bi]))
    return eff


def vod4_s_terms(h, keep_idx: Sequence[int], mid_idx: Sequence[int]) -> np.ndarray:
    """Fourth-order ``V_od^4`` (nested-commutator) correction of Rydberg_sim.tex Thm 1.

    :func:`resolvent_elimination` is 2nd order in the ``P``–``Q`` coupling
    ``V_od`` (it resums the block-diagonal ``V_d`` ladder — the ``e<->r'``
    couplings and the ``Δ_add`` shifts — to all orders through the resolvent, so
    it already carries ``C^(2) + C^(3) + C^(4,d)``).  The one piece it misses is
    the genuinely 4th-order-in-``V_od`` contribution from the ``S_1`` generator,

        δH^(4,S) = -1/6 C^(4,S^2) - 1/24 C^(4,S^3),

    Eqs. (C4S2)/(C4S3) of Theorem 1.  This function returns that ``keep×keep``
    matrix, built from the ``mid_idx`` (``6P`` manifold) block only: ``r_garb``
    (= ``r'``) does not enter the ``S`` terms (``d_a`` sums over the kept states
    and ``|v_a>``/``|χ_a>`` live in the ``e_F`` subspace).  Adding it to
    :func:`resolvent_elimination` gives the full Thm-1 reduction through 4th order.

        v_a = H[mid, a],   Λ_a = diag(1/(E_mid - E_a)),   χ_a = Λ_a v_a,
        d_a = -Σ_c [ χ_c (χ_c^† v_a) + 2 χ_c (v_c^† χ_a) + v_c (χ_c^† χ_a) ].
    """
    h = np.asarray(h, dtype=complex)
    keep = list(keep_idx)
    mid = list(mid_idx)
    energy = np.real(np.diag(h))
    e_mid = energy[mid]
    v = {a: h[np.ix_(mid, [a])].ravel() for a in keep}
    lam = {a: 1.0 / (e_mid - energy[a]) for a in keep}   # diagonal Λ_a
    chi = {a: lam[a] * v[a] for a in keep}
    d = {}
    for a in keep:
        acc = np.zeros(len(mid), dtype=complex)
        for c in keep:
            acc -= (
                chi[c] * np.vdot(chi[c], v[a])
                + 2.0 * chi[c] * np.vdot(v[c], chi[a])
                + v[c] * np.vdot(chi[c], chi[a])
            )
        d[a] = acc
    out = np.zeros((len(keep), len(keep)), dtype=complex)
    for ai, a in enumerate(keep):
        for bi, b in enumerate(keep):
            c4s2 = np.vdot(v[a], lam[b] * d[b]) + np.vdot(d[a], lam[a] * v[b])
            c4s3 = -(np.vdot(v[a], lam[a] * d[b]) + np.vdot(d[a], lam[b] * v[b]))
            out[ai, bi] = -c4s2 / 6.0 - c4s3 / 24.0
    return out


def shift_coefficients(
    h_const: np.ndarray,
    h420: np.ndarray,
    h1013: np.ndarray,
) -> dict[str, float]:
    """Diagonal AC-Stark light shifts the eliminated ``6P`` manifold induces.

    These are the ``a == b`` entries of :func:`schrieffer_wolff` for the
    rb87 seven-level layout, read directly off the single-atom parts (``h_const``
    diagonal, the 420 and 1013 couplings).  Each is the shift on its level at the
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


# rb87_7 -> {0,1,r} kept indices (Rydberg_sim.tex Thm 1: P3; Q3 = complement,
# i.e. {e1,e2,e3,r_garb} with the e<->r' couplings kept inside the Q block).
_KEEP_P3 = [_QUBIT_0, _QUBIT_1, _RYD]


def single_atom_hamiltonian_parts(system):
    """Single-atom ``(h_const, h420, h1013)`` 7x7 matrices for an rb87 system.

    Reconstructs the legacy dense matrices from the primitive operator model:
    ``h_const`` sums the static diagonal-energy terms (``E[a,a]`` coefficients,
    plus any h.c.-completed static coupling); ``h420`` / ``h1013`` are the
    unit-Rabi laser legs ``Σ ratio·|ket><bra|`` over the ``laser_channel_ratios``
    ``"420"`` / ``"1013"`` groups.
    """
    levels = system.basis.local_levels
    d = len(levels)
    idx = {lvl: i for i, lvl in enumerate(levels)}
    hc = np.zeros((d, d), dtype=complex)
    for term in system.static_hamiltonian_terms:
        if term.name == "H_pair":
            continue
        ket, bra = parse_E(term.name)
        hc[idx[ket], idx[bra]] += term.coefficient
        if term.add_hermitian_conjugate and ket != bra:
            hc[idx[bra], idx[ket]] += np.conjugate(term.coefficient)
    ratios = dict(system.level_structure.laser_channel_ratios or {})

    def _group(group: str) -> np.ndarray:
        m = np.zeros((d, d), dtype=complex)
        for chan, ratio in ratios.get(group, {}).items():
            ket, bra = parse_E(chan)
            m[idx[ket], idx[bra]] += ratio
        return m

    return hc, _group("420"), _group("1013")


def _tex_frame_h7_fn(protocol, system7):
    """Single-atom ``H7(t)`` in the tex rotating frame; returns ``(h7_tex, t_gate)``.

    Rebuilds the laser-frame ``H7(t)`` from the protocol's drive coefficients
    (static energies + Σ coeff·E[ket,bra] + h.c.), then gauge-rotates the
    ``{e1,e2,e3}`` rows by the instantaneous 420 phase and the ``{r, r_garb}``
    rows by the 420+1013 phase so all optical couplings are real, and moves the
    corresponding chirps onto the diagonal (Rydberg_sim.tex convention):

        E_e     += -dot_phi_420,
        E_r/r'  += -(dot_phi_420 + dot_phi_1013)  =  -Δ_add(t).

    Phase and chirp are read off one reference channel per laser group
    (largest CG ratio) with a wrap-safe arg-ratio finite difference.  Where a
    leg's amplitude vanishes the phase/chirp defaults to 0 — with no coupling
    the rotation is pure gauge on unoccupied levels.
    """
    if not hasattr(protocol, "get_drive_coefficients"):
        raise TypeError(
            f"{type(protocol).__name__} does not expose get_drive_coefficients; "
            "pass a concrete rb87_7 laser protocol (e.g. CZProtocol/TOProtocol)."
        )
    hc = single_atom_hamiltonian_parts(system7)[0]  # only the static part is needed here
    if hc.shape[0] < 7:
        raise ValueError("The effective-theory converters target the rb87_7 layout.")
    idx = {lvl: i for i, lvl in enumerate(system7.basis.local_levels)}

    params = protocol._resolve(system7)
    t_gate = float(params["t_gate"])
    scale = float(getattr(system7, "amplitude_scale", 1.0))

    # The driven channels and their (row, col) placement are constant across t,
    # so parse each E[ket,bra] name once rather than at every lowering grid point.
    channel_slots = []
    for chan in protocol.get_drive_coefficients(0.0, params):
        ket, bra = parse_E(chan)
        channel_slots.append((chan, idx[ket], idx[bra], ket != bra))

    # Reference channel per laser group for phase extraction (largest |ratio|).
    ratios = dict(system7.level_structure.laser_channel_ratios or {})
    refs = {
        group: max(chans, key=lambda ch: abs(chans[ch]))
        for group, chans in ratios.items()
        if chans
    }
    eps = 1e-7 * t_gate
    tol = 1e-12 * scale * max(
        abs(params.get("omega_420_max", 1.0)), abs(params.get("omega_1013_max", 1.0)), 1.0
    )

    def coeffs(t: float):
        return protocol.get_drive_coefficients(float(t), params)

    def _phase_chirp(group: str, t: float) -> tuple[float, float]:
        """(phi, dot_phi) of the group's laser at ``t``; coefficient = |.|e^{-i phi}."""
        chan = refs.get(group)
        if chan is None:
            return 0.0, 0.0
        c0 = coeffs(t).get(chan, 0j)
        cp = coeffs(t + eps).get(chan, 0j)
        cm = coeffs(t - eps).get(chan, 0j)
        if min(abs(c0), abs(cp), abs(cm)) < tol:
            return 0.0, 0.0
        # The constant CG-ratio sign adds a global pi to phi — a harmless basis
        # sign; the arg-ratio difference below is unaffected by it (and by wraps).
        return -float(np.angle(c0)), -float(np.angle(cp * np.conjugate(cm))) / (2.0 * eps)

    e_levels = list(_MID)
    ryd_levels = [_RYD, _RYD_GARB]

    def h7_tex(t: float) -> np.ndarray:
        H = hc.copy()
        cs = coeffs(t)
        for chan, i, j, off_diag in channel_slots:
            c = scale * cs[chan]
            H[i, j] += c
            if off_diag:
                H[j, i] += np.conjugate(c)
        phi420, dphi420 = _phase_chirp("420", t)
        phi1013, dphi1013 = _phase_chirp("1013", t)
        w = np.ones(H.shape[0], dtype=complex)
        w[e_levels] = np.exp(1j * phi420)
        w[ryd_levels] = np.exp(1j * (phi420 + phi1013))
        H = H * np.outer(w, np.conjugate(w))
        H[e_levels, e_levels] += -dphi420
        H[ryd_levels, ryd_levels] += -(dphi420 + dphi1013)
        return H

    return h7_tex, t_gate


def lower_cz_to_effective_01r(protocol, system7):
    """Lower an rb87_7 CZ pulse to a *single-atom* effective ``{0,1,r}`` protocol.

    Maps *protocol* — any concrete rb87_7 laser protocol that exposes
    ``get_drive_coefficients`` (e.g. a :class:`~ryd_gate.protocols.gate_cz.CZProtocol`,
    including the ``TOProtocol``/``ARProtocol`` phase families) — onto a public
    :class:`~ryd_gate.protocols.digital_analog.DigitalAnalogProtocol` that
    drives the full local 3x3 effective Hamiltonian on the ``01r`` model in
    direct matrix-element convention:

        coupling_r1(t) = H_eff[r, 1](t)      (K1r)
        coupling_10(t) = H_eff[1, 0](t)      (K01)
        coupling_r0(t) = H_eff[r, 0](t)      (K0r)
        energy_1(t)    = H_eff[1, 1] - H_eff[0, 0]
        energy_r(t)    = H_eff[r, r] - H_eff[0, 0]

    (the ``|0>`` energy is the gauge zero — an unobservable global phase).

    At each ``t`` it rebuilds the single-atom ``H7(t)`` in the tex rotating frame
    (:func:`_tex_frame_h7_fn`: real couplings, chirps on the diagonal) and applies
    the full 4th-order reduction of Rydberg_sim.tex Theorem 1:

        H_eff = resolvent_elimination(H7) + vod4_s_terms(H7).

    The resolvent (2nd order in ``V_od``) resums the ``e <-> r'`` couplings and the
    ``Δ_add`` ladder to all orders (``C^(2)+C^(3)+C^(4,d)``); :func:`vod4_s_terms`
    adds the genuinely 4th-order ``V_od^4`` nested-commutator piece the resolvent
    misses.  The returned protocol is in the tex frame — couplings real,
    ``energy_r`` carrying ``-Δ_add(t)`` — which changes no computational-basis
    observable.

    CAVEAT: single-atom lowering cannot know that the r' shift is
    blockade-suppressed when the partner atom is in ``|r>``; for two-atom gate
    phases (ZZ) use :func:`lower_cz_to_effective_pair`.
    """
    from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol

    h7_tex, t_gate = _tex_frame_h7_fn(protocol, system7)

    # The five channel functions all sample the same reduction at the same t;
    # memoize the last evaluation so the two SW reductions run once per t.
    cache: dict = {"t": None, "M": None}

    def h_eff(t: float) -> np.ndarray:
        if cache["t"] != t:
            m = h7_tex(t)
            cache["M"] = resolvent_elimination(m, _KEEP_P3) + vod4_s_terms(m, _KEEP_P3, _MID)
            cache["t"] = t
        return cache["M"]

    def _energy(row: int):
        def value(t: float) -> float:
            e = h_eff(t)[row, row] - h_eff(t)[0, 0]
            if abs(np.imag(e)) > 1e-9 * max(1.0, abs(e)):
                raise ValueError(
                    "effective diagonal energy is not real (decay-enabled source "
                    "system?); the lowered DigitalAnalogProtocol is coherent-only."
                )
            return float(np.real(e))

        return value

    return DigitalAnalogProtocol(
        t_gate=t_gate,
        coupling_10=lambda t: complex(h_eff(t)[1, 0]),
        coupling_r0=lambda t: complex(h_eff(t)[2, 0]),
        coupling_r1=lambda t: complex(h_eff(t)[2, 1]),
        energy_1=_energy(1),
        energy_r=_energy(2),
    )


class EffectivePairModel:
    """Two-atom direct effective Hamiltonian (Rydberg_sim.tex Theorem 2).

    ``h_eff(t)`` returns the dense 9x9 two-qutrit Hamiltonian in the product
    basis ``labels`` (atom 1 is the slow index).  The Rydberg pair interaction
    is already inside it (both through the kept ``V n_r n_r`` block and through
    the blockade-shifted denominators of the eliminated virtual states), so
    evolve it directly — e.g. a piecewise-constant ``expm`` loop with ``n_steps``
    steps over ``t_gate``.  It is *not* a compilable protocol: the compile
    pipeline has no driven pair channels.
    """

    labels = ("00", "01", "0r", "10", "11", "1r", "r0", "r1", "rr")

    def __init__(self, *, t_gate: float, n_steps: int, h_eff) -> None:
        self.t_gate = float(t_gate)
        self.n_steps = int(n_steps)
        self.h_eff = h_eff


def lower_cz_to_effective_pair(protocol, system7, *, n_steps: int | None = None):
    """Lower an rb87_7 CZ pulse to the *two-atom* effective model (tex Theorem 2).

    *system7* must be a two-atom rb87_7 system with its geometry bound (the pair
    strength ``V = C6/R^6`` is read from ``interaction_pairs``).  At each ``t``
    the interacting two-atom Hamiltonian ``H49(t) = H7(t)⊗1 + 1⊗H7(t) +
    V (n_r+n_r')⊗(n_r+n_r')`` is built in the tex frame and reduced in one step
    onto the 9 two-qutrit states with :func:`resolvent_elimination`.  Because the
    eliminated block contains the interaction, virtual states such as ``|r'r>``
    carry the blockade shift in their denominators — the correction that fixes
    the ZZ (entangling) phase which single-atom lowering gets wrong.

    Returns an :class:`EffectivePairModel`; evolve its ``h_eff(t)`` directly.
    """
    pairs = tuple(system7.interaction_pairs)
    if int(system7.N) != 2 or len(pairs) != 1:
        raise ValueError(
            "lower_cz_to_effective_pair needs a two-atom rb87_7 system with its "
            "geometry bound (exactly one interaction pair)."
        )
    v_nn = float(pairs[0][2])

    h7_tex, t_gate = _tex_frame_h7_fn(protocol, system7)
    dim = 7
    ident = np.eye(dim)
    n_ryd = np.zeros((dim, dim))
    n_ryd[_RYD, _RYD] = n_ryd[_RYD_GARB, _RYD_GARB] = 1.0  # both Rydberg states interact
    h_int = v_nn * np.kron(n_ryd, n_ryd)
    keep = [dim * i + j for i in _KEEP_P3 for j in _KEEP_P3]

    def h_eff(t: float) -> np.ndarray:
        h1 = h7_tex(t)
        return resolvent_elimination(
            np.kron(h1, ident) + np.kron(ident, h1) + h_int, keep
        )

    return EffectivePairModel(
        t_gate=t_gate,
        n_steps=n_steps if n_steps is not None else getattr(protocol, "n_steps", 200),
        h_eff=h_eff,
    )
