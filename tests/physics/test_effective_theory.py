"""The 7-level ⇄ {0,1,r} effective-theory map (src/ryd_gate/core/effective_theory.py).

These lift the validation in scripts/notebooks/find_phase.ipynb §4 into tests:
the closed-form Lemma-1 coefficients vs the numeric reduction, the resolvent
method of Rydberg_sim.tex Theorems 1–2 (single-atom and interacting-pair
one-step elimination), the blockade suppression of the r' light shift that the
pair theorem captures, and the full-7-level vs effective phase comparison in
the perturbative domain (slow-marked).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.exact import simulate
from ryd_gate.core.effective_theory import (
    _tex_frame_h7_fn,
    lower_cz_to_effective_01r,
    lower_cz_to_effective_pair,
    resolvent_elimination,
    schrieffer_wolff,
    shift_coefficients,
    single_atom_hamiltonian_parts,
    vod4_s_terms,
)
from ryd_gate.ir import compile_hamiltonian_ir
from ryd_gate.lattice import Register
from ryd_gate.physics import our_laser_rabis
from ryd_gate.protocols import CZProtocol, DigitalAnalogProtocol, TOProtocol
from ryd_gate.protocols.gate_cz import cz_effective_rabi, phase_from_chirp

# Canonical σ⁻/σ⁺ (rb87_7_mp) single-photon Rabis (rad/s) — used to restore the
# full Rabi onto the unit-normalized 420/1013 blocks in these reductions.
RABI_420_MP, RABI_1013_MP = 2 * np.pi * 491e6, 2 * np.pi * 185e6

X_TO_DARK = [
    -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
    1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
]

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


def _one_atom_rb87_7(tag="rb87_7_mp"):
    return RydbergSystem(
        level_structure=level_structure(tag),
        register=Register.chain(1, spacing_um=3.0),
    )


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
    De = system.level_structure.Delta
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
    De = system.level_structure.Delta
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


def test_cz_protocol_drives_1013_on_rb87():
    """rb87 builds *unit* 420/1013 blocks; the protocol supplies their Rabi, so
    both 420 AND 1013 are *driven* channels (not static)."""
    protocol = TOProtocol(
        phase_amplitude=X_TO_DARK[0], frequency_ratio=X_TO_DARK[1],
        phase_offset=X_TO_DARK[2], detuning_ratio=X_TO_DARK[3],
        duration_ratio=X_TO_DARK[5],
    )
    system = RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(2, spacing_um=3.0),
        protocol=protocol,
    )
    ir = compile_hamiltonian_ir(system)

    static_names = {term.name for term in ir.static_terms}
    driven = {term.channel for term in ir.drive_terms}
    # both the 420 (E[e_F,q]) and 1013 (E[r/r_garb,e_F]) legs are driven, not static.
    assert {"E[e1,1]", "E[r,e1]"} <= driven
    assert "E[r,e1]" not in static_names


def test_cz_protocol_drives_420_and_1013():
    """The container always drives 420 AND 1013 (unit blocks need the protocol to
    restore the 1013 Rabi); the four normalized functions of ``s`` set both."""
    from types import SimpleNamespace

    # Unit ratio per group so the expanded primitive coeff equals the laser coeff;
    # a duck-typed context supplies the ratio map and a unit time scale.
    context = SimpleNamespace(
        level_structure=SimpleNamespace(
            name="duck",
            laser_channel_ratios={"420": {"E[e1,1]": 1.0}, "1013": {"E[r,e1]": 1.0}},
            time_scale=1.0,
        )
    )
    proto = CZProtocol(
        duration_ratio=1e-6,
        A_420=lambda s: 0.7,
        phi_420=lambda s: 0.0,
        A_1013=lambda s: 0.4,
        phi_1013=lambda s: 0.5,
        omega_420_max=2.0,
        omega_1013_max=3.0,
    )
    assert {"E[e1,1]", "E[r,e1]"} <= proto.required_channels
    ctx = proto._resolve(context)
    assert ctx["t_gate"] == pytest.approx(1e-6)
    coeffs = proto.get_drive_coefficients(0.3e-6, ctx)
    # forward primitive channels only (the compiler adds each leg's h.c.)
    assert coeffs["E[e1,1]"] == pytest.approx(2.0 * 0.7)
    assert coeffs["E[r,e1]"] == pytest.approx(3.0 * 0.4 * np.exp(-1j * 0.5))


def _wrap(a):
    return float(np.angle(np.exp(1j * a)))


def _zz(p):
    """Gauge-invariant entangling phase."""
    return _wrap(p["11"] - p["01"] - p["10"] + p["00"])


def _theta1(p):
    """Gauge-invariant single-qubit phase."""
    return _wrap(p["01"] - p["00"])


def _h7_fn(proto7, sys7):
    """Single-atom H7(t) rebuilt from the protocol's drive coefficients,
    exactly as lower_cz_to_effective_01r does internally."""
    params = proto7._resolve(sys7)
    h_const, _, _ = single_atom_hamiltonian_parts(sys7)
    idx = {lvl: i for i, lvl in enumerate(sys7.basis.local_levels)}

    def h7(t):
        H = h_const.copy()
        for chan, c in proto7.get_drive_coefficients(float(t), params).items():
            ket, bra = chan[2:-1].split(",")
            i, j = idx[ket], idx[bra]
            H[i, j] += c
            if ket != bra:
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
    phi = phase_from_chirp(chirp, t_gate, 12001)
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp", detuning_sign=1),
        register=Register.chain(2, spacing_um=spacing),
    )
    _, time_scale = cz_effective_rabi(sys7, o420, o1013)
    proto7 = CZProtocol(
        duration_ratio=t_gate / time_scale,
        A_420=lambda s: _smooth_env(float(np.clip(s, 0.0, 1.0)) * t_gate, t_gate),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * t_gate),
        omega_420_max=o420, omega_1013_max=o1013,
    )
    proto_eff = lower_cz_to_effective_01r(proto7, sys7)
    assert proto_eff.required_channels == frozenset(
        {"E[r,1]", "E[1,0]", "E[r,0]", "E[r,r]", "E[1,1]"}
    )

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
        ref = h7_ref_tex(float(t))
        h3 = resolvent_elimination(ref, KEEP) + vod4_s_terms(ref, KEEP, [2, 3, 4])
        co = proto_eff.get_drive_coefficients(float(t), {"n_sites": 1})
        recon = np.zeros((3, 3), dtype=complex)
        recon[1, 1] = co["E[1,1]"]
        recon[2, 2] = co["E[r,r]"]
        for (upper, lower), ch in (((2, 1), "E[r,1]"), ((1, 0), "E[1,0]"), ((2, 0), "E[r,0]")):
            recon[upper, lower], recon[lower, upper] = co[ch], np.conj(co[ch])
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
    o420, o1013 = our_laser_rabis(
        p420_w=1.2, p1013_w=20.0, beam_area=7 * 20 * spacing, ryd_level=70
    )
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp", detuning_sign=1, Delta_Hz=40.1e9),
        register=Register.chain(2, spacing_um=spacing),
    )
    proto7 = _adia_cz_protocol(t_gate, 64, o420, o1013, sys7)
    h7_tex, _ = _tex_frame_h7_fn(proto7, sys7)
    v_nn = float(sys7.interaction_pairs[0][2])
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


def _adia_cz_protocol(t_gate, n_steps, omega_420, omega_1013, system):
    """The find_phase adiabatic waveform as a plain CZProtocol (chirp-integral
    420 phase, 0.15-ramped 420 envelope, 0.05-ramped 1013 envelope)."""
    d_amp = 2 * np.pi * 20e6
    phi = phase_from_chirp(
        lambda t: -d_amp * np.cos(2 * np.pi * t / t_gate), t_gate, 4 * n_steps + 1
    )
    _, time_scale = cz_effective_rabi(system, omega_420, omega_1013)
    return CZProtocol(
        duration_ratio=t_gate / time_scale,
        A_420=lambda s: _smooth_env(float(np.clip(s, 0.0, 1.0)) * t_gate, t_gate),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * t_gate),
        A_1013=lambda s: _smooth_env(
            float(np.clip(s, 0.0, 1.0)) * t_gate, t_gate, ramp=0.05
        ),
        omega_420_max=omega_420, omega_1013_max=omega_1013,
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
    ch = proto_eff._function_fields

    def h3(t):
        """Lowered 3x3 local Hamiltonian (matrix-element convention H[upper, lower])."""
        h = np.zeros((3, 3), dtype=complex)
        h[1, 0] = ch["coupling_10"](t)
        h[2, 0] = ch["coupling_r0"](t)
        h[2, 1] = ch["coupling_r1"](t)
        h[1, 1] = ch["energy_1"](t)
        h[2, 2] = ch["energy_r"](t)
        return h + h.conj().T - np.diag(h.diagonal())

    v_nn = float(sys_eff.interaction_pairs[0][2])
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
@pytest.mark.parametrize("case", ["weak", "moderate"])
def test_effective_matches_seven_level_phases(case):
    """find_phase §4: the resolvent effective theories (tex Thm 1 single-atom,
    Thm 2 pair) vs the gauge-invariant phases of the full 7-level CZ dynamics.

    ``weak`` (0.3x the find_phase drive) is the perturbative domain where the
    truncation is controlled: both the single-atom converter (lowered control
    functions, midpoint propagator) and the pair model track theta1 AND ZZ to
    <0.03 rad.
    ``moderate`` (the canonical test Rabis, Ω_420 ≈ 2π·2.6 GHz) keeps the ZZ
    guard; theta1 is NOT asserted there — at that drive the accumulated
    dynamical phase is ~100 rad, so the dropped 4th-order optical terms
    (~0.1% of H) already move theta1 by O(1 rad).  That is a property of any
    2nd-order effective theory at those parameters, not of the implementation.
    Needs fine steps to resolve the ~40 GHz |e> manifold (slow-marked)."""
    spacing, t_gate, n_steps = 3.0, 1.0e-6, 3000
    if case == "weak":
        o420, o1013 = our_laser_rabis(
            p420_w=1.2, p1013_w=20.0, beam_area=7 * 20 * spacing, ryd_level=70
        )
        o420, o1013, n_ref = 0.3 * o420, 0.3 * o1013, 32000
    else:
        o420, o1013 = our_laser_rabis(
            p420_w=0.641, p1013_w=10.0, beam_area=7 * 20 * spacing, ryd_level=70
        )
        n_ref = 16000
    sys7 = RydbergSystem(
        level_structure=level_structure("rb87_7_mp", detuning_sign=1, Delta_Hz=40.1e9),
        register=Register.chain(2, spacing_um=spacing),
    )
    proto7 = _adia_cz_protocol(t_gate, n_steps, o420, o1013, sys7)
    h7_local = _h7_fn(proto7, sys7)
    v_nn = float(sys7.interaction_pairs[0][2])

    ryd7_int = np.diag([0, 0, 0, 0, 0, 1.0, 1.0])   # |r> + garbage |r'> interact
    phi7, ret7 = _evolve_phases(h7_local, 7, ryd7_int, v_nn, t_gate, n_ref)
    phi_s = _phases_via_converter(proto7, sys7, spacing, t_gate, n_steps=n_steps)
    d_th1_s = abs(_wrap(_theta1(phi_s) - _theta1(phi7)))
    d_zz_s = abs(_wrap(_zz(phi_s) - _zz(phi7)))
    print(f"\n[{case}] return>={ret7:.4f}  single dth1={d_th1_s:.2e} dZZ={d_zz_s:.2e}")

    if case == "moderate":
        assert ret7 > 0.99      # the 7-level gate is a clean (adiabatic) return
        assert d_zz_s < 0.1
        return

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


def test_tn_path_accepts_nonzero_k0r():
    """The TN 01r lowering supports the |0>-|r> (K0r) leg: a driven E[r,0]
    coefficient lowers without error (the old guard is gone)."""
    from types import SimpleNamespace

    from ryd_gate.core.level_structures import (
        level_structure,
        three_level_profiles_from_coeffs,
    )

    spec = SimpleNamespace(level_spec=level_structure("01r"), N=1, level_structure="01r")
    three_level_profiles_from_coeffs({"E[r,1]": 1.0 + 0j}, spec)
    three_level_profiles_from_coeffs({"E[r,1]": 1.0 + 0j, "E[r,0]": 0.3 + 0j}, spec)


def test_digital_analog_effective_drive_runs_on_01r():
    """A matrix-element DigitalAnalog drive realizes (Omega_eff/2, phi) on the
    01r model: a resonant constant drive Rabi-flops |1> <-> |r>."""
    t_gate, omega = 1e-6, 2 * np.pi * 1e6
    proto = DigitalAnalogProtocol(
        t_gate=t_gate, coupling_r1=lambda t: 0.5 * omega
    )
    # Only the driven coupling exists — no K01/K0r channels, no energies.
    assert proto.required_channels == frozenset({"E[r,1]"})
    coeffs = proto.get_drive_coefficients(0.3e-6, {"t_gate": t_gate, "n_sites": 1})
    assert coeffs["E[r,1]"] == pytest.approx(0.5 * omega)
    assert set(coeffs) == {"E[r,1]"}

    sys01r = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(1),
        protocol=proto,
    )
    t_eval = np.linspace(0.0, t_gate, 101)
    result = simulate(
        sys01r, psi0=["1"], t_eval=t_eval,
        observables={"n_r_0": sys01r.observables.n("r", 0)},
    )
    n_r = np.real(result.expectation("n_r_0"))
    assert n_r.max() > 0.99   # full |1> -> |r> transfer at the pi-pulse
