"""The 7-level ⇄ {0,1,r} effective-theory map (src/ryd_gate/core/effective_theory.py).

These lift the validation in scripts/notebooks/find_phase.ipynb §4 into tests:
the closed-form Lemma-1 coefficients vs the numeric reduction, the resolvent
method of manuscripts/chapters/Rydberg_sim.tex Theorems 1–2
(single-atom and interacting-pair one-step elimination), the blockade
suppression of the r' light shift that the pair theorem captures, and the
full-7-level vs effective phase comparison in the perturbative domain
(slow-marked).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.exact import compile_exact
from ryd_gate.core.effective_theory import (
    _single_pair_strength,
    _tex_frame_h7_fn,
    dressed_gap_ratio,
    lower_cz_to_effective_01r,
    lower_cz_to_effective_pair,
    resolvent_elimination,
    reverse_amplitude_split,
    schrieffer_wolff,
    shift_coefficients,
    single_atom_hamiltonian_parts,
    vod4_s_terms,
)
from ryd_gate.core.lowering import lower_drives
from ryd_gate.core.operators import parse_E
from ryd_gate.lattice import Register
from ryd_gate.physics import rb87_7_mp_rabi_frequencies
from ryd_gate.protocols import CZProtocol, DigitalAnalogProtocol, phase_from_chirp
from ryd_gate.protocols._resolved import _LaserDrive

# Canonical σ⁻/σ⁺ (rb87_7_mp) single-photon Rabis (rad/s) — used to restore the
# full Rabi onto the unit-normalized 420/1013 blocks in these reductions.
RABI_420_MP, RABI_1013_MP = 2 * np.pi * 491e6, 2 * np.pi * 185e6

# Intermediate detuning is now a PROTOCOL input (P19): the effective-theory
# helpers reconstruct Δ_e onto the |e> diagonals from the bound protocol, so the
# tests pass it to the CZProtocol and reuse the same number as "Delta".
DELTA_MP = 2 * np.pi * 9.1e9      # rb87_7_mp default intermediate detuning
DELTA_ADIA = 2 * np.pi * 40.1e9   # find_phase adiabatic-gate detuning

EPS0 = 2 * np.pi * 6.835e9
KEEP = [0, 1, 5]        # P3 = {0, 1, r}
ELIM = [2, 3, 4]        # {e1, e2, e3} only
ELIM_Q4 = [2, 3, 4, 6]  # Q4 = {e1, e2, e3, r'} — the direct 7->3 of tex Thm 1


def _block(system, *names):
    """Legacy block name -> single-atom matrix, via the primitive operator model.

    ``H_const`` -> static diagonal energies; ``drive_420`` -> the unit-Rabi 420
    leg; ``drive_1013`` / ``H_1013`` -> the unit-Rabi 1013 leg.
    """
    hc, h420, h1013 = single_atom_hamiltonian_parts(system)
    parts = {"H_const": hc, "drive_420": h420, "drive_1013": h1013, "H_1013": h1013}
    for name in names:
        if name in parts:
            return parts[name]
    raise KeyError(f"none of {names} known")


def _probe_cz_protocol(t_gate_s=0.5e-6, delta=DELTA_MP):
    """Minimal CZ pulse that only carries the intermediate detuning.

    single_atom_hamiltonian_parts folds ``delta`` back onto the |e> diagonals;
    the (constant unit) envelope/phase and the peak Rabis are irrelevant to the
    pure-matrix reductions, which restore the Rabi on the unit blocks by hand.
    """
    return CZProtocol(
        t_gate_s=t_gate_s,
        intermediate_detuning_rad_s=delta,
        omega_420_max_rad_s=RABI_420_MP,
        omega_1013_max_rad_s=RABI_1013_MP,
        envelope_420=lambda t: 1.0,
        phase_420_rad=lambda t: 0.0,
    )


def _one_atom_rb87_7(tag="rb87_7_mp", delta=DELTA_MP):
    return RydbergSystem(
        level_structure=level_structure(tag),
        register=Register.chain(1, spacing_um=3.0),
        protocol=_probe_cz_protocol(delta=delta),
    )


def _da_channels(protocol):
    """``channel -> coefficient(t)`` map exposed by a lowered protocol.

    The lowering seam is ``protocol._resolve(system) -> _ResolvedProtocol``; a
    DigitalAnalogProtocol ignores the system, so this reads the direct
    matrix-element coefficients the converters produce.
    """
    resolved = protocol._resolve(None)
    return {d.channel: d.coefficient for d in resolved.drives}


def test_shift_coefficients_match_lowdin_diagonal():
    """``shift_coefficients`` is exactly the diagonal of the Löwdin projection.

    Both are the same 2nd-order PT formula read off the same blocks, so they must
    agree to machine precision (this is the internal-consistency guarantee that
    is the internal-consistency guarantee behind the closed-form cross-checks)."""
    system = _one_atom_rb87_7()
    h_const = _block(system, "H_const")
    R420, R1013 = (RABI_420_MP, RABI_1013_MP)
    # Blocks are unit-normalized; at unit Rabi the diagonal shifts are ~1e-12 rad/s
    # and vanish under float cancellation against the ~4e10 clock energy on the
    # diagonal.  Restore the full Rabi so eff[a,a]-h_const[a,a] is resolvable.
    h420 = R420 * _block(system, "drive_420")
    h1013 = R1013 * _block(system, "drive_1013", "H_1013")

    # Full local H at the envelope plateau (full 420 drive, env = 1).
    h7 = h_const + h1013 + h1013.conj().T + h420 + h420.conj().T
    eff = schrieffer_wolff(h7, KEEP, ELIM)
    coeffs = shift_coefficients(h_const, h420, h1013)

    # diagonal correction = M[a,a] - H_const[a,a]
    assert np.real(eff[0, 0] - h_const[0, 0]) == pytest.approx(coeffs["D0"], rel=1e-9)
    assert np.real(eff[1, 1] - h_const[1, 1]) == pytest.approx(coeffs["D1"], rel=1e-9)
    assert np.real(eff[2, 2] - h_const[5, 5]) == pytest.approx(coeffs["Dr"], rel=1e-9)
    # Dr_garb (r' 1013 light shift) is otherwise unpinned: SW keeping only r'.
    eff_rg = schrieffer_wolff(h7, [6], ELIM)
    assert np.real(eff_rg[0, 0] - h_const[6, 6]) == pytest.approx(coeffs["Dr_garb"], rel=1e-9)


def test_closed_forms_match_resolvent():
    """Closed-form coefficients vs the one-step resolvent reduction (tex Thm 1).

    Also pins the structural relation between the two reducers: with the
    ``e <-> r'`` couplings removed the eliminated block is diagonal and
    :func:`resolvent_elimination` coincides with :func:`schrieffer_wolff`
    to machine precision; with them present the resolvent adds only the small
    r'-resummation on top.  The ~1% closed-form residual is the intermediate
    hyperfine spread η_F (the closed form puts all three |e_F> at Δ_e)."""
    system = _one_atom_rb87_7()
    h_const = _block(system, "H_const")
    De = DELTA_MP
    R420, R1013 = (RABI_420_MP, RABI_1013_MP)
    # The 420/1013 blocks are unit-normalized; restore the full Rabi so the numeric
    # shifts match the closed forms (which carry R420/R1013).
    h420 = R420 * _block(system, "drive_420")
    h1013 = R1013 * _block(system, "drive_1013", "H_1013")

    h7 = h_const + h1013 + h1013.conj().T + h420 + h420.conj().T
    eff = resolvent_elimination(h7, KEEP)

    # No e<->r' couplings -> H_Q diagonal -> resolvent == plain 2nd-order SW.
    h1013_ng = h1013.copy()
    h1013_ng[6, :] = 0.0
    h7_ng = h_const + h1013_ng + h1013_ng.conj().T + h420 + h420.conj().T
    scale = np.max(np.abs(h7))
    assert np.allclose(
        resolvent_elimination(h7_ng, KEEP),
        schrieffer_wolff(h7_ng, KEEP, ELIM_Q4),
        rtol=1e-12, atol=1e-9 * scale,
    )

    # numeric (resolvent) vs analytic (Clebsch-Gordan-summed closed forms), env = 1.
    checks = [
        ("D1", np.real(eff[1, 1] - h_const[1, 1]), -(4 / 3) * R420**2 / (4 * De), 2e-2),
        ("Dr", np.real(eff[2, 2] - h_const[5, 5]), -(R1013**2) / (4 * De), 2e-2),
        ("D0", np.real(eff[0, 0] - h_const[0, 0]), -(4 / 3) * R420**2 / (4 * (De + EPS0)), 2e-2),
        ("K1r", abs(eff[1, 2]), R420 * R1013 / (4 * De), 2e-2),
        ("K01", abs(eff[0, 1]), (2 / 3) * R420**2 * (1 / (8 * De) + 1 / (8 * (De + EPS0))), 5e-2),
        ("K0r", abs(eff[0, 2]), R420 * R1013 * (1 / (8 * De) + 1 / (8 * (De + EPS0))), 2e-2),
    ]
    for name, numeric, analytic, tol in checks:
        assert numeric == pytest.approx(analytic, rel=tol), name


def test_vod4_s_terms_hermitian_and_quartic():
    """The 4th-order V_od^4 correction (tex Thm 1) is Hermitian, scales as
    amplitude^4, and is (Ω/Δ_e)^2-small relative to the 2nd-order resolvent shift.

    r_garb (= r') does not enter the S-terms, so the correction is unchanged when
    the 1013 r_garb leg is zeroed — the diagnostic that it is a pure 6P (mid)
    manifold quantity."""
    system = _one_atom_rb87_7()
    h_const = _block(system, "H_const")
    De = DELTA_MP
    R420, R1013 = (RABI_420_MP, RABI_1013_MP)
    h420 = R420 * _block(system, "drive_420")
    h1013 = R1013 * _block(system, "drive_1013", "H_1013")
    h7 = h_const + h1013 + h1013.conj().T + h420 + h420.conj().T

    s = vod4_s_terms(h7, KEEP, [2, 3, 4])
    assert np.allclose(s, s.conj().T, atol=1e-6 * np.max(np.abs(s)))   # Hermitian

    # amplitude^4 scaling: halving both Rabis divides the correction by 16.
    h7_half = h_const + (h1013 + h1013.conj().T + h420 + h420.conj().T) / 2.0
    s_half = vod4_s_terms(h7_half, KEEP, [2, 3, 4])
    assert np.max(np.abs(s_half)) == pytest.approx(np.max(np.abs(s)) / 16.0, rel=1e-6)

    # (Ω/Δ_e)^2-small vs the 2nd-order resolvent light shift on |1>: the S-term
    # sits at ~(Ω/Δ_e)^2 * d1 up to an O(1) Clebsch factor.
    d1 = abs(np.real(resolvent_elimination(h7, KEEP)[1, 1] - h_const[1, 1]))
    ratio = np.max(np.abs(s)) / d1
    assert 0.05 * (R420 / De) ** 2 < ratio < 2.0 * (R420 / De) ** 2

    # r' does not enter the S-terms.
    h1013_ng = h1013.copy()
    h1013_ng[6, :] = 0.0
    h7_ng = h_const + h1013_ng + h1013_ng.conj().T + h420 + h420.conj().T
    assert np.allclose(vod4_s_terms(h7_ng, KEEP, [2, 3, 4]), s, atol=1e-9 * np.max(np.abs(s)))


def test_dressed_gap_ratio_matches_analytic_toys():
    """``η`` projects the coupling per dressed eigenstate: an uncoupled
    near-degenerate level contributes nothing (the failure mode of any
    norm-over-minimum-gap estimate), dressed mixing enters through the
    eigenbasis, and a coupled degeneracy is ``inf``."""
    g, delta = 0.3, 10.0

    # Diagonal H_Q with one uncoupled level almost degenerate with E_0 = 0.
    h = np.array([[0.0, g, 0.0], [g, delta, 0.0], [0.0, 0.0, 1e-9]], dtype=complex)
    assert np.isclose(dressed_gap_ratio(h, [0]), g / delta)

    # Internal H_Q mixing: eigenpairs (delta ± big, (1, ±1)/sqrt2), each seeing
    # |<q|v>| = g/sqrt2 against its own dressed gap.
    big = 4.0
    h_mix = np.array([[0.0, g, 0.0], [g, delta, big], [0.0, big, delta]], dtype=complex)
    expected = (g / np.sqrt(2.0)) / min(abs(delta - big), abs(delta + big))
    assert np.isclose(dressed_gap_ratio(h_mix, [0]), expected)

    # Coupled degeneracy: invalid reduction.
    h_deg = np.array([[0.0, g], [g, 0.0]], dtype=complex)
    assert dressed_gap_ratio(h_deg, [0]) == float("inf")

    with pytest.raises(ValueError, match="Hermitian"):
        dressed_gap_ratio(np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]), [0])


def test_lemma1_cg_product_sums():
    """Lemma-1 Clebsch–Gordan product sums, read off the registered rb87_7_mp
    blocks (unit Rabi, so each matrix entry is Ω_Fa/2):

        Σ|Ω_F1|² = Σ|Ω_F0|² = 4/3·Ω420², Σ|Ω_Fr|² = Ω1013²,
        Σ Ω_F1·Ω_Fr = Σ Ω_F0·Ω_Fr = Ω420·Ω1013, Σ Ω_F0·Ω_F1 = 2/3·Ω420²,
        Σ Ω_Fr·Ω_Fr' = 0  (why r' also drops out of the *diagonal-free* cross terms).
    """
    system = _one_atom_rb87_7()
    g420 = _block(system, "drive_420")      # g_Fa = Ω_Fa/2 at unit Ω_420
    g1013 = _block(system, "drive_1013")    # g_Fr = Ω_Fr/2 at unit Ω_1013
    o1 = 2.0 * g420[2:5, 1]
    o0 = 2.0 * g420[2:5, 0]
    o_r = 2.0 * g1013[5, 2:5]
    o_rg = 2.0 * g1013[6, 2:5]
    sums = {
        "sum |O_F1|^2 = 4/3": (np.sum(np.abs(o1) ** 2), 4.0 / 3.0),
        "sum |O_F0|^2 = 4/3": (np.sum(np.abs(o0) ** 2), 4.0 / 3.0),
        "sum |O_Fr|^2 = 1": (np.sum(np.abs(o_r) ** 2), 1.0),
        "sum O_F1 O_Fr = ±1": (abs(np.sum(o1 * o_r)), 1.0),
        "sum O_F0 O_Fr = ±1": (abs(np.sum(o0 * o_r)), 1.0),
        "sum O_F0 O_F1 = ±2/3": (abs(np.sum(o0 * o1)), 2.0 / 3.0),
    }
    for name, (numeric, analytic) in sums.items():
        assert float(np.real(numeric)) == pytest.approx(analytic, rel=1e-6), name
    assert abs(np.sum(o_r * o_rg)) < 1e-9  # r-r' 1013 cross sum vanishes


def test_cz_protocol_drives_420_and_1013():
    """The container always drives 420 AND 1013 (unit blocks need the protocol to
    restore the 1013 Rabi); each laser coefficient is ``Ω_max · env · e^{-iφ}``,
    and both legs (420 = E[e1,1], 1013 = E[r,e1]) lower to *driven* channels — the
    1013 leg is never a static term."""
    proto = CZProtocol(
        t_gate_s=1e-6,
        intermediate_detuning_rad_s=DELTA_MP,
        omega_420_max_rad_s=2.0,
        omega_1013_max_rad_s=3.0,
        envelope_420=lambda t: 0.7,
        phase_420_rad=lambda t: 0.0,
        envelope_1013=lambda t: 0.4,
        phase_1013_rad=lambda t: 0.5,
    )
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(1, spacing_um=3.0),
        protocol=proto,
    )
    lasers = {d.group: d for d in proto._resolve(system).drives if isinstance(d, _LaserDrive)}
    # both lasers are present as driven groups, never static
    assert set(lasers) == {"420", "1013"}
    assert lasers["420"].coefficient(0.3e-6) == pytest.approx(2.0 * 0.7)
    assert lasers["1013"].coefficient(0.3e-6) == pytest.approx(3.0 * 0.4 * np.exp(-1j * 0.5))

    # both legs lower to driven E[ket,bra] channels; the 1013 leg is not static.
    _, channels = lower_drives(system)
    driven = {ch.channel for ch in channels}
    static_names = {term.name for term in system._static_terms}
    assert {"E[e1,1]", "E[r,e1]"} <= driven
    assert "E[r,e1]" not in static_names


def _wrap(a):
    return float(np.angle(np.exp(1j * a)))


def _zz(p):
    """Gauge-invariant entangling phase."""
    return _wrap(p["11"] - p["01"] - p["10"] + p["00"])


def _theta1(p):
    """Gauge-invariant single-qubit phase."""
    return _wrap(p["01"] - p["00"])


def _h7_fn(proto7, sys7):
    """Single-atom lab-frame H7(t) rebuilt from the protocol's lowered drives,
    exactly as lower_cz_to_effective_01r does internally (before the tex rotation).

    ``single_atom_hamiltonian_parts`` already folds the constant intermediate
    detuning onto the |e> diagonals, so only the off-diagonal laser legs from the
    canonical lowering seam are added here."""
    h_const, _, _ = single_atom_hamiltonian_parts(sys7)
    idx = {lvl: i for i, lvl in enumerate(sys7._basis.local_levels)}
    _, channels = lower_drives(sys7)
    slots = []
    for ch in channels:
        ket, bra = parse_E(ch.channel)
        if ket != bra:   # diagonal intermediate detuning is already in h_const
            slots.append((ch.coefficient, idx[ket], idx[bra]))

    def h7(t):
        H = h_const.copy()
        for coeff, i, j in slots:
            c = complex(coeff(float(t)))
            H[i, j] += c
            H[j, i] += np.conj(c)
        return H

    return h7


def test_converter_matrix_level_reduction():
    """lower_cz_to_effective_01r == analytic tex-frame + (resolvent + S-terms).

    An independent in-test construction of the tex rotating frame (rotate the
    ``{e,r,r'}`` rows by the *known* chirp-integral 420 phase; put ``-chirp`` on
    their diagonal) followed by :func:`resolvent_elimination` **plus the 4th-order
    ``V_od^4``** correction (:func:`vod4_s_terms`) must reproduce the converter's
    lowered DigitalAnalog coefficients — compared through the gauge-invariant pieces
    (diagonal relative to |0>, coupling magnitudes, and the K01·K1r·K0r* phase
    triple), since the converter's internal phase extraction may differ by a
    constant basis sign per rotated row."""
    spacing, t_gate = 3.0, 0.5e-6
    o420, o1013 = (RABI_420_MP, RABI_1013_MP)
    d_amp = 2 * np.pi * 20e6
    chirp = lambda t: -d_amp * np.cos(2 * np.pi * t / t_gate)
    # Fine phase sampling: the converter differentiates the *interpolated* phase,
    # the reference below uses the analytic chirp — keep the two within ~1e-6.
    phi = phase_from_chirp(chirp, t_gate, n_samples=12001)
    proto7 = CZProtocol(
        t_gate_s=t_gate,
        intermediate_detuning_rad_s=DELTA_MP,
        omega_420_max_rad_s=o420,
        omega_1013_max_rad_s=o1013,
        envelope_420=lambda t: _smooth_env(t, t_gate),
        phase_420_rad=lambda t: phi(float(np.clip(t, 0.0, t_gate))),
    )
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=spacing),
        protocol=proto7,
    )
    proto_eff = lower_cz_to_effective_01r(proto7, sys7)
    co = _da_channels(proto_eff)
    assert set(co) == {"E[r,1]", "E[1,0]", "E[r,0]", "E[r,r]", "E[1,1]"}

    h7 = _h7_fn(proto7, sys7)
    rotated = [2, 3, 4, 5, 6]   # {e1,e2,e3,r,r'}; phi_1013 = 0 -> one common angle

    def h7_ref_tex(t):
        H = h7(t)
        w = np.ones(7, dtype=complex)
        w[rotated] = np.exp(1j * float(phi(t)))   # coefficients carry e^{-i phi_420}
        H = H * np.outer(w, np.conjugate(w))
        H[rotated, rotated] += -chirp(t)
        return H

    max_k0r = 0.0
    for t in np.linspace(0.05e-6, 0.45e-6, 7):
        t = float(t)
        ref = h7_ref_tex(t)
        h3 = resolvent_elimination(ref, KEEP) + vod4_s_terms(ref, KEEP, [2, 3, 4])
        recon = np.zeros((3, 3), dtype=complex)
        recon[1, 1] = co["E[1,1]"](t)
        recon[2, 2] = co["E[r,r]"](t)
        for (upper, lower), ch in (((2, 1), "E[r,1]"), ((1, 0), "E[1,0]"), ((2, 0), "E[r,0]")):
            recon[upper, lower], recon[lower, upper] = co[ch](t), np.conj(co[ch](t))
        ref3 = h3 - h3[0, 0] * np.eye(3)
        scale = 1 + np.max(np.abs(ref3))
        assert np.max(np.abs(np.diag(recon) - np.diag(ref3))) < 1e-6 * scale
        assert np.max(np.abs(np.abs(recon) - np.abs(ref3))) < 1e-6 * scale
        # phase triple is invariant under per-row basis signs
        tri = lambda M: np.angle(M[1, 0] * M[2, 1] * np.conj(M[2, 0]))
        assert abs(_wrap(tri(recon) - tri(ref3))) < 1e-6
        max_k0r = max(max_k0r, abs(h3[0, 2]))
    assert max_k0r > 0.0   # K0r is genuinely populated (not dropped)


def test_pair_diagonal_is_blockade_shifted_single_atom():
    """Tex Theorem-2 mechanism at matrix level: the pair model's ``|1r>``
    diagonal equals the single-atom sum *with the r' level blockade-shifted by
    V* — while the naive single-atom sum misses exactly the r'-induced |1>
    light shift (the term that corrupts the ZZ phase in single-atom lowering).
    """
    spacing, t_gate = 3.0, 1.0e-6
    o420, o1013 = rb87_7_mp_rabi_frequencies(1.2, 20.0, 7 * 20 * spacing, ryd_level=70)
    proto7 = _adia_cz_protocol(t_gate, 64, o420, o1013, DELTA_ADIA)
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=spacing),
        protocol=proto7,
    )
    h7_tex, _ = _tex_frame_h7_fn(proto7, sys7)
    v_nn = _single_pair_strength(sys7)
    pair = lower_cz_to_effective_pair(proto7, sys7)

    for s in (0.3, 0.5):
        t = s * t_gate
        h7 = h7_tex(t)
        h3 = resolvent_elimination(h7, KEEP)
        h7_blocked = h7.copy()
        h7_blocked[6, 6] += v_nn        # partner in |r> blockade-shifts this atom's r'
        h3_blocked = resolvent_elimination(h7_blocked, KEEP)
        rp_shift = float(np.real(h3[1, 1] - h3_blocked[1, 1]))   # r'-induced |1> shift

        e_1r = float(np.real(pair.h_eff(t)[5, 5]))               # |1r> pair diagonal
        err_naive = abs(e_1r - float(np.real(h3[1, 1] + h3[2, 2])))
        err_blocked = abs(e_1r - float(np.real(h3_blocked[1, 1] + h3[2, 2])))
        # the naive mismatch IS the r' shift; the blockaded residual is the much
        # smaller cross-atom dressing (~16 kHz vs ~0.5 MHz here)
        assert err_naive == pytest.approx(abs(rp_shift), rel=5e-2)
        assert err_blocked < 0.1 * err_naive
        assert abs(rp_shift) > 2 * np.pi * 0.2e6   # the effect is resolvable (>0.2 MHz)


_RAMP = 0.15


def _smooth_env(t, t_gate, ramp=_RAMP):
    s = float(np.clip(t / t_gate, 0.0, 1.0))
    q = lambda u: (lambda v: 10 * v**3 - 15 * v**4 + 6 * v**5)(np.clip(u, 0.0, 1.0))
    if s < ramp:
        return float(q(s / ramp))
    if s > 1.0 - ramp:
        return float(q((1.0 - s) / ramp))
    return 1.0


def _adia_cz_protocol(t_gate, n_steps, omega_420, omega_1013, delta):
    """The find_phase adiabatic waveform as a plain CZProtocol (chirp-integral
    420 phase, 0.15-ramped 420 envelope, 0.05-ramped 1013 envelope).

    Waveforms are physical-time callables (P24); ``delta`` is the signed
    intermediate detuning that used to live in the level structure (P19)."""
    d_amp = 2 * np.pi * 20e6
    phi = phase_from_chirp(
        lambda t: -d_amp * np.cos(2 * np.pi * t / t_gate), t_gate, n_samples=4 * n_steps + 1
    )
    return CZProtocol(
        t_gate_s=t_gate,
        intermediate_detuning_rad_s=delta,
        omega_420_max_rad_s=omega_420,
        omega_1013_max_rad_s=omega_1013,
        envelope_420=lambda t: _smooth_env(t, t_gate),
        phase_420_rad=lambda t: phi(float(np.clip(t, 0.0, t_gate))),
        envelope_1013=lambda t: _smooth_env(t, t_gate, ramp=0.05),
    )


_LABELS = ("00", "01", "10", "11")
_LOC = {"0": 0, "1": 1}


def _evolve_phases(h_local_fn, dim, ryd_int, v_nn, t_gate, n_steps):
    """Two-atom piecewise-constant evolution of a single-atom H(t); returns the
    per-basis-state return phases and the minimum return probability."""
    ident = np.eye(dim)
    h_int = v_nn * np.kron(ryd_int, ryd_int)
    dt = t_gate / n_steps
    basis = np.eye(dim)
    kets = {s: np.kron(basis[_LOC[s[0]]], basis[_LOC[s[1]]]).astype(complex) for s in _LABELS}
    psis = {s: kets[s].copy() for s in _LABELS}
    for k in range(n_steps):
        h_loc = h_local_fn((k + 0.5) * dt)
        u = expm(-1j * dt * (np.kron(h_loc, ident) + np.kron(ident, h_loc) + h_int))
        for s in _LABELS:
            psis[s] = u @ psis[s]
    ov = {s: np.vdot(kets[s], psis[s]) for s in _LABELS}
    return {s: float(np.angle(ov[s])) for s in _LABELS}, min(abs(ov[s]) ** 2 for s in _LABELS)


def _phases_via_converter(proto7, sys7, spacing, t_gate, n_steps):
    """find_phase-style path: converter -> effective-01r control functions,
    evolved with the midpoint piecewise-constant propagator (_evolve_phases).

    ``simulate()`` is deliberately not used here: the converter extracts each
    chirp by finite-differencing an interpolated phase, which leaves razor-thin
    glitch poles near the phase-grid nodes that stall the adaptive exact_ode
    solver (and the ~2π·6.8 GHz tex-frame diagonal makes an accurate adaptive
    solve impractically slow anyway).  The fixed midpoints step over the poles;
    this test pins the converter physics, not the backend.
    """
    proto_eff = lower_cz_to_effective_01r(proto7, sys7)
    sys_eff = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2, spacing_um=spacing),
        protocol=proto_eff,
    )
    co = _da_channels(proto_eff)

    def h3(t):
        """Lowered 3x3 local Hamiltonian (matrix-element convention H[upper, lower])."""
        h = np.zeros((3, 3), dtype=complex)
        h[1, 0] = co["E[1,0]"](t)
        h[2, 0] = co["E[r,0]"](t)
        h[2, 1] = co["E[r,1]"](t)
        h[1, 1] = co["E[1,1]"](t)
        h[2, 2] = co["E[r,r]"](t)
        return h + h.conj().T - np.diag(h.diagonal())

    v_nn = _single_pair_strength(sys_eff)
    phi, _ = _evolve_phases(h3, 3, np.diag([0.0, 0.0, 1.0]), v_nn, t_gate, n_steps)
    return phi


_PAIR_IDX = {"00": 0, "01": 1, "10": 3, "11": 4}


def _evolve_pair(pair, n_steps):
    """Piecewise-constant evolution of an EffectivePairModel's 9x9 h_eff(t)."""
    dt = pair.t_gate / n_steps
    basis = np.eye(9)
    kets = {s: basis[_PAIR_IDX[s]].astype(complex) for s in _LABELS}
    psis = {s: kets[s].copy() for s in _LABELS}
    for k in range(n_steps):
        u = expm(-1j * dt * pair.h_eff((k + 0.5) * dt))
        for s in _LABELS:
            psis[s] = u @ psis[s]
    return {s: float(np.angle(np.vdot(kets[s], psis[s]))) for s in _LABELS}


@pytest.mark.slow
def test_effective_matches_seven_level_phases():
    """find_phase §4: the resolvent effective theories (tex Thm 1 single-atom,
    Thm 2 pair) vs the gauge-invariant phases of the full 7-level CZ dynamics.

    In the ``weak`` domain (0.3x the find_phase drive) the truncation is
    controlled: both the single-atom converter (lowered control functions,
    midpoint propagator) and the pair model track theta1 AND ZZ to <0.03 rad.
    Needs fine steps to resolve the ~40 GHz |e> manifold (slow-marked)."""
    spacing, t_gate, n_steps = 3.0, 1.0e-6, 3000
    o420, o1013 = rb87_7_mp_rabi_frequencies(1.2, 20.0, 7 * 20 * spacing, ryd_level=70)
    o420, o1013, n_ref = 0.3 * o420, 0.3 * o1013, 32000
    proto7 = _adia_cz_protocol(t_gate, n_steps, o420, o1013, DELTA_ADIA)
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=spacing),
        protocol=proto7,
    )
    h7_local = _h7_fn(proto7, sys7)
    v_nn = _single_pair_strength(sys7)

    ryd7_int = np.diag([0, 0, 0, 0, 0, 1.0, 1.0])   # |r> + garbage |r'> interact
    phi7, ret7 = _evolve_phases(h7_local, 7, ryd7_int, v_nn, t_gate, n_ref)
    phi_s = _phases_via_converter(proto7, sys7, spacing, t_gate, n_steps=n_steps)
    d_th1_s = abs(_wrap(_theta1(phi_s) - _theta1(phi7)))
    d_zz_s = abs(_wrap(_zz(phi_s) - _zz(phi7)))
    print(f"\n[weak] return>={ret7:.4f}  single dth1={d_th1_s:.2e} dZZ={d_zz_s:.2e}")

    pair = lower_cz_to_effective_pair(proto7, sys7)
    phi_p = _evolve_pair(pair, 16000)
    d_th1_p = abs(_wrap(_theta1(phi_p) - _theta1(phi7)))
    d_zz_p = abs(_wrap(_zz(phi_p) - _zz(phi7)))
    print(f"[weak] pair dth1={d_th1_p:.2e} dZZ={d_zz_p:.2e}")

    assert ret7 > 0.5           # weak drive: partial adiabatic return is enough
    assert d_th1_s < 0.03
    assert d_zz_s < 0.03
    assert d_th1_p < 0.03
    assert d_zz_p < 0.03


def test_effective_01r_accepts_nonzero_k0r():
    """The 01r effective model supports the |0>-|r> (K0r) leg: a DigitalAnalog
    drive with a nonzero coupling_r0 lowers and compiles on the exact path
    (the old K0r guard is gone)."""
    proto = DigitalAnalogProtocol(
        t_gate_s=1e-6,
        coupling_r1_rad_s=lambda t: 2 * np.pi * 1e6,
        coupling_r0_rad_s=lambda t: 2 * np.pi * 0.3e6,
    )
    sys01r = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(1),
        protocol=proto,
    )
    _, channels = lower_drives(sys01r)
    names = {ch.channel for ch in channels}
    assert {"E[r,1]", "E[r,0]"} <= names           # both K1r and K0r are driven
    ham, _ = compile_exact(sys01r, hamiltonian_format="dense")
    assert ham.dim == 3                            # compiles on 01r without error


# ── reverse_amplitude_split (inverse map; consumed by notebooks 01_cz_gate / find_phase) ──


def test_reverse_amplitude_split_hold_modes():
    """The three power-split choices realize the same product alpha*beta = ratio."""
    nom = 2 * np.pi * 5e6
    omega_eff = lambda t: 0.25 * nom          # constant ratio 0.25
    a, b = reverse_amplitude_split(omega_eff, omega_eff_nom=nom, hold="1013")
    assert (a(0.0), b(0.0)) == pytest.approx((0.25, 1.0))          # modulate 420
    a, b = reverse_amplitude_split(omega_eff, omega_eff_nom=nom, hold="420")
    assert (a(0.0), b(0.0)) == pytest.approx((1.0, 0.25))          # modulate 1013
    a, b = reverse_amplitude_split(omega_eff, omega_eff_nom=nom, hold="balanced")
    assert a(0.0) == pytest.approx(np.sqrt(0.25))
    assert b(0.0) == pytest.approx(np.sqrt(0.25))
    assert a(0.0) * b(0.0) == pytest.approx(0.25)


def test_reverse_amplitude_split_balanced_clamps_negative_ratio():
    """A negative Ω_eff would give a negative ratio; the balanced split clamps to 0."""
    nom = 2 * np.pi * 5e6
    a, b = reverse_amplitude_split(lambda t: -nom, omega_eff_nom=nom, hold="balanced")
    assert a(0.0) == 0.0
    assert b(0.0) == 0.0


def test_reverse_amplitude_split_validation():
    with pytest.raises(ValueError, match="omega_eff_nom must be non-zero"):
        reverse_amplitude_split(lambda t: 1.0, omega_eff_nom=0.0)
    with pytest.raises(ValueError, match="hold must be"):
        reverse_amplitude_split(lambda t: 1.0, omega_eff_nom=1.0, hold="both")


# ── schrieffer_wolff / shift_coefficients / _single_pair_strength gap coverage ──


def test_schrieffer_wolff_uses_explicit_bare_energies():
    """Explicit ``bare_energies`` replace Re(diag) in the PT denominators
    (effective_theory.py:96-98)."""
    h = np.array(
        [[1.0, 0.0, 0.5],
         [0.0, 2.0, 0.3],
         [0.5, 0.3, 10.0]], dtype=complex
    )
    keep, elim = [0, 1], [2]
    eff = schrieffer_wolff(h, keep, elim, bare_energies=[0.0, 0.0, 10.0])
    assert np.real(eff[0, 0]) == pytest.approx(1.0 + 0.25 / (0.0 - 10.0))
    assert np.real(eff[1, 1]) == pytest.approx(2.0 + 0.09 / (0.0 - 10.0))
    # default (None) reads Re(diag): different denominators -> different shift.
    eff_default = schrieffer_wolff(h, keep, elim)
    assert np.real(eff_default[0, 0]) == pytest.approx(1.0 + 0.25 / (1.0 - 10.0))
    assert not np.isclose(eff[0, 0], eff_default[0, 0])


def test_shift_coefficients_error_branches():
    """shift_coefficients rejects a non-7-level layout and a zero |e> denominator
    (effective_theory.py:213-218)."""
    small = np.zeros((3, 3), dtype=complex)
    with pytest.raises(ValueError, match="rb87_7"):
        shift_coefficients(small, small, small)
    hc = np.diag([1.0, 2.0, 0.0, 0.0, 0.0, 5.0, 6.0]).astype(complex)  # E_mid == 0
    z7 = np.zeros((7, 7), dtype=complex)
    with pytest.raises(ValueError, match="denominator is zero"):
        shift_coefficients(hc, z7, z7)


def test_single_pair_strength_error_paths():
    """_single_pair_strength rejects N != 2, a missing pair term, and >1 pair
    (effective_theory.py:520-528)."""
    from types import SimpleNamespace

    from ryd_gate.core.operators import RydbergPairInteractionSpec

    with pytest.raises(ValueError, match="two-atom"):
        _single_pair_strength(SimpleNamespace(N=1))
    with pytest.raises(ValueError, match="no resolved Rydberg pair interaction"):
        _single_pair_strength(SimpleNamespace(N=2, _static_terms=()))
    multi = SimpleNamespace(
        name="H_pair",
        operator=RydbergPairInteractionSpec(
            pairs=((0, 1, 1.0), (0, 2, 2.0)), rydberg_levels=("r",)
        ),
    )
    with pytest.raises(ValueError, match="exactly one interaction pair"):
        _single_pair_strength(SimpleNamespace(N=2, _static_terms=(multi,)))


def test_lower_cz_to_effective_01r_rejects_non_real_energy(monkeypatch):
    """A decay-enabled source leaves a complex effective |1> energy; the converter's
    energy channels reject it (effective_theory.py:440-445).  The tex-frame H7 is
    stubbed with a complex |1> diagonal (off-diagonals zero -> eff[1,1] = 1 + 5j)."""
    m = np.diag([0.0, 1.0 + 5.0j, 100.0, 110.0, 120.0, 2.0, 3.0]).astype(complex)
    monkeypatch.setattr(
        "ryd_gate.core.effective_theory._tex_frame_h7_fn",
        lambda protocol, system7: ((lambda t: m), 1e-6),
    )
    proto_eff = lower_cz_to_effective_01r(None, None)
    co = _da_channels(proto_eff)
    with pytest.raises(ValueError, match="not real"):
        co["E[1,1]"](0.0)


def test_tex_frame_zero_amplitude_phase_default():
    """At an envelope zero the 420/1013 amplitudes vanish, so _tex_frame_h7_fn's
    phase/chirp read-off defaults to (0, 0) (effective_theory.py:360-361): the
    {e,r,r'} diagonal carries NO chirp, so H7(0) is exactly the static diagonal —
    even though the true chirp at t=0 is nonzero."""
    t_gate = 1.0e-6
    proto7 = _adia_cz_protocol(t_gate, 64, RABI_420_MP, RABI_1013_MP, DELTA_MP)
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(1, spacing_um=3.0),
        protocol=proto7,
    )
    h7_tex, _ = _tex_frame_h7_fn(proto7, sys7)
    h_const, _, _ = single_atom_hamiltonian_parts(sys7)

    # t=0: envelope 0 -> couplings vanish and the chirp default fires -> just static.
    assert np.allclose(h7_tex(0.0), h_const)
    # contrast: mid-gate the default does NOT fire and |r> carries -Δ_add(t).
    assert abs(h7_tex(0.5 * t_gate)[5, 5] - h_const[5, 5]) > 2 * np.pi * 1e6
