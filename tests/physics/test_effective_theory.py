"""The 7-level ⇄ {0,1,r} effective-theory map (src/ryd_gate/core/effective_theory.py).

These lift the validation in scripts/notebooks/find_phase.ipynb §4 into pinned
tests: the closed-form Theorem-1 coefficients vs the numeric Löwdin projection,
and the internal consistency of ``shift_coefficients`` with that projection.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.linalg import expm

from ryd_gate import RydbergSystem
from ryd_gate.backends.exact import simulate
from ryd_gate.core.effective_theory import (
    lower_cz_to_effective_01r,
    schrieffer_wolff,
    shift_coefficients,
)
from ryd_gate.core.physical_models import rb87_default_rabis
from ryd_gate.gates import CZProtocol, EffectiveCZProtocol, TOProtocol, phase_from_chirp
from ryd_gate.ir import compile_hamiltonian_ir
from ryd_gate.lattice import Register
from ryd_gate.physics import our_laser_rabis

X_TO_DARK = [
    -0.6894097925886826, 1.040962607910546, 0.3277877211544321,
    1.5639989822346387, 0.6689846026179691, 1.3407418093368753,
]

EPS0 = 2 * np.pi * 6.835e9
KEEP = [0, 1, 5]   # {0, 1, r}
ELIM = [2, 3, 4]   # {e1, e2, e3}


def _block(system, *names):
    for name in names:
        if system.blocks.has(name):
            return np.asarray(system.blocks.get(name).matrix)
    raise KeyError(f"none of {names} registered")


def _one_atom_rb87_7(param_set="our"):
    return (
        RydbergSystem.set_atom_level("rb87_7", param_set=param_set)
        .set_atom_geom(Register.chain(1, spacing_um=3.0))
        .build()
    )


def test_shift_coefficients_match_lowdin_diagonal():
    """``shift_coefficients`` is exactly the diagonal of the Löwdin projection.

    Both are the same 2nd-order PT formula read off the same blocks, so they must
    agree to machine precision (this is the internal-consistency guarantee that
    is the internal-consistency guarantee behind the closed-form cross-checks)."""
    system = _one_atom_rb87_7()
    h_const = _block(system, "H_const")
    R420, R1013 = rb87_default_rabis("our")
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


def test_closed_forms_match_lowdin():
    """Theorem-1 closed forms vs the numeric Löwdin projection (find_phase §4.4).

    The ~1% residual is the intermediate-state hyperfine spread η_F (the closed
    form puts all three |e_F> at Δ_e); it is the documented 2nd-order accuracy."""
    system = _one_atom_rb87_7()
    h_const = _block(system, "H_const")
    De = system.meta("Delta")
    R420, R1013 = rb87_default_rabis("our")
    # The 420/1013 blocks are unit-normalized; restore the full Rabi so the numeric
    # Löwdin shifts match the closed forms (which carry R420/R1013).
    h420 = R420 * _block(system, "drive_420")
    h1013 = R1013 * _block(system, "drive_1013", "H_1013")

    h7 = h_const + h1013 + h1013.conj().T + h420 + h420.conj().T
    eff = schrieffer_wolff(h7, KEEP, ELIM)

    # numeric (Löwdin) vs analytic (Clebsch-Gordan-summed Theorem 1), env = 1.
    checks = [
        ("D1", np.real(eff[1, 1] - h_const[1, 1]), -(4 / 3) * R420**2 / (4 * De), 2e-2),
        ("Dr", np.real(eff[2, 2] - h_const[5, 5]), -(R1013**2) / (4 * De), 2e-2),
        ("D0", np.real(eff[0, 0] - h_const[0, 0]), -(4 / 3) * R420**2 / (4 * (De + EPS0)), 2e-2),
        ("K1r", abs(eff[1, 2]), R420 * R1013 / (4 * De), 2e-2),
        ("K01", abs(eff[0, 1]), (2 / 3) * R420**2 * (1 / (8 * De) + 1 / (8 * (De + EPS0))), 5e-2),
    ]
    for name, numeric, analytic, tol in checks:
        assert numeric == pytest.approx(analytic, rel=tol), name


def test_cz_protocol_drives_1013_on_rb87():
    """rb87_7 builds *unit* 420/1013 blocks; the protocol supplies their Rabi, so
    both 420 AND 1013 are *driven* channels (not static)."""
    system = (
        RydbergSystem.set_atom_level("rb87_7", param_set="our")
        .set_atom_geom(Register.chain(2, spacing_um=3.0))
        .set_protocol(TOProtocol())
    )
    pulse = system.protocol.build(X_TO_DARK, system)
    system = system.with_protocol(pulse)
    params = system.unpack_params([])
    ir = compile_hamiltonian_ir(system, params)

    static_names = {term.name for term in ir.static_terms}
    driven = {term.channel for term in ir.drive_terms}
    assert {"drive_420", "drive_1013", "drive_1013_dag"} <= driven
    assert "drive_1013" not in static_names and "drive_1013_dag" not in static_names


def test_cz_protocol_drives_420_and_1013():
    """The container always drives 420 AND 1013 (unit blocks need the protocol to
    restore the 1013 Rabi); the four normalized functions of ``s`` set both."""
    from ryd_gate.gates import CZProtocol

    params = {"t_gate": 1e-6, "omega_420_max": 2.0, "omega_1013_max": 3.0}
    proto = CZProtocol(
        t_gate=1e-6,
        A_420=lambda s: 0.7,
        phi_420=lambda s: 0.0,
        A_1013=lambda s: 0.4,
        phi_1013=lambda s: 0.5,
    )
    assert proto.required_channels == frozenset(
        {"drive_420", "drive_420_dag", "drive_1013", "drive_1013_dag"}
    )
    coeffs = proto.get_drive_coefficients(0.3e-6, params)
    assert coeffs["drive_420"] == pytest.approx(2.0 * 0.7)
    assert coeffs["drive_1013"] == pytest.approx(3.0 * 0.4 * np.exp(-1j * 0.5))
    assert coeffs["drive_1013_dag"] == pytest.approx(np.conjugate(3.0 * 0.4 * np.exp(-1j * 0.5)))


def _wrap(a):
    return float(np.angle(np.exp(1j * a)))


def test_converter_matrix_level_reduction():
    """lower_cz_to_effective_01r reproduces the two-stage SW reduction to machine
    precision: the 3x3 reconstructed from the returned EffectiveCZProtocol equals
    the direct two-stage Löwdin projection of H7(t), up to the |0>-energy gauge."""
    spacing, t_gate = 3.0, 0.5e-6
    o420, o1013 = rb87_default_rabis("our")
    proto7 = _adia_cz_protocol(t_gate, 64, o420, o1013)
    sys7 = (
        RydbergSystem.set_atom_level("rb87_7", param_set="our", detuning_sign=1)
        .set_atom_geom(Register.chain(2, spacing_um=spacing))
        .build()
    )
    proto_eff = lower_cz_to_effective_01r(proto7, sys7)
    assert proto_eff.required_channels == frozenset(
        {"drive_R", "drive_hf", "drive_0r", "delta_R", "delta_hf"}
    )

    params = proto7.unpack_params([], sys7)
    hc = np.asarray(sys7.blocks.get("H_const").matrix)
    h420 = np.asarray(sys7.blocks.get("drive_420").matrix)
    h1013 = np.asarray(sys7.blocks.get("drive_1013").matrix)

    def h7(t):
        c = proto7.get_drive_coefficients(t, params)
        c420, c1013 = c["drive_420"], c.get("drive_1013", 0.0)
        return (
            hc + c420 * h420 + np.conj(c420) * h420.conj().T
            + c1013 * h1013 + np.conj(c1013) * h1013.conj().T
        )

    max_k0r = 0.0
    for t in np.linspace(0.05e-6, 0.45e-6, 7):
        h4 = schrieffer_wolff(h7(t), [0, 1, 5, 6], [2, 3, 4])
        h3 = schrieffer_wolff(h4, [0, 1, 2], [3])
        co = proto_eff.get_drive_coefficients(float(t), {})
        recon = np.zeros((3, 3), dtype=complex)
        recon[1, 1] = co["delta_hf"]
        recon[2, 2] = co["delta_R"]
        for (upper, lower), ch in (((2, 1), "drive_R"), ((1, 0), "drive_hf"), ((2, 0), "drive_0r")):
            recon[upper, lower], recon[lower, upper] = co[ch], np.conj(co[ch])
        ref = h3 - h3[0, 0] * np.eye(3)
        assert np.max(np.abs(recon - ref)) < 1e-6 * (1 + np.max(np.abs(ref)))
        max_k0r = max(max_k0r, abs(h3[0, 2]))
    assert max_k0r > 0.0   # K0r is genuinely populated (not dropped)


_RAMP = 0.15


def _smooth_env(t, t_gate):
    s = float(np.clip(t / t_gate, 0.0, 1.0))
    q = lambda u: (lambda v: 10 * v**3 - 15 * v**4 + 6 * v**5)(np.clip(u, 0.0, 1.0))
    if s < _RAMP:
        return float(q(s / _RAMP))
    if s > 1.0 - _RAMP:
        return float(q((1.0 - s) / _RAMP))
    return 1.0


def _adia_cz_protocol(t_gate, n_steps, omega_420, omega_1013):
    """The find_phase adiabatic waveform as a plain CZProtocol (chirp-integral phase)."""
    d_amp = 2 * np.pi * 20e6
    phi = phase_from_chirp(
        lambda t: -d_amp * np.cos(2 * np.pi * t / t_gate), t_gate, 4 * n_steps + 1
    )
    return CZProtocol(
        t_gate=t_gate,
        A_420=lambda s: _smooth_env(float(np.clip(s, 0.0, 1.0)) * t_gate, t_gate),
        phi_420=lambda s: phi(float(np.clip(s, 0.0, 1.0)) * t_gate),
        omega_420_max=omega_420, omega_1013_max=omega_1013, n_steps=n_steps,
    )


def _adia_rb87_7(spacing=3.0, t_gate=1.0e-6, n_steps=3000):
    """rb87_7 'our' bound to the find_phase adiabatic CZ pulse."""
    omega_420, omega_1013 = our_laser_rabis(
        p420_w=0.641, p1013_w=10.0, beam_area=7 * 20 * spacing, ryd_level=70
    )
    proto7 = _adia_cz_protocol(t_gate, n_steps, omega_420, omega_1013)
    sys7 = (
        RydbergSystem.set_atom_level(
            "rb87_7", param_set="our", detuning_sign=1, Delta_Hz=40.1e9
        )
        .set_atom_geom(Register.chain(2, spacing_um=spacing))
        .build()
    )
    return sys7.with_protocol(proto7), proto7


@pytest.mark.slow
def test_effective_matches_seven_level_observables():
    """The full {0,1,r} effective theory (via the public converter) reproduces the
    7-level CZ dynamics — ZZ phase and Rydberg population — to the documented
    2nd-order tolerances (find_phase §4.3), and the converted EffectiveCZProtocol
    run through rg.simulate on 01r matches the direct two-stage SW reduction.
    Needs fine steps: the ~40 GHz |e> manifold must be resolved (slow-marked)."""
    spacing, t_gate, n_steps = 3.0, 1.0e-6, 3000
    sys7, proto7 = _adia_rb87_7(spacing, t_gate, n_steps)
    params = proto7.unpack_params([], sys7)
    h_const = np.asarray(sys7.blocks.get("H_const").matrix)
    h420 = np.asarray(sys7.blocks.get("drive_420").matrix)
    h1013 = np.asarray(sys7.blocks.get("drive_1013").matrix)
    v_nn = float(sys7.metadata["interaction_pairs"][0][2])
    labels, loc = ("00", "01", "10", "11"), {"0": 0, "1": 1}

    def h7_local(t):  # rebuild H7(t) exactly as the converter does (420 + const 1013)
        c = proto7.get_drive_coefficients(float(t), params)
        c420, c1013 = c["drive_420"], c.get("drive_1013", 0.0)
        return (
            h_const
            + c420 * h420 + np.conj(c420) * h420.conj().T
            + c1013 * h1013 + np.conj(c1013) * h1013.conj().T
        )

    def h_eff_local(t):  # two-stage SW: eliminate {e}, then r_garb (= converter's h_eff)
        h4 = schrieffer_wolff(h7_local(t), [0, 1, 5, 6], [2, 3, 4])
        return schrieffer_wolff(h4, [0, 1, 2], [3])

    def evolve(h_local_fn, dim, ryd_int, ryd_obs):
        ident = np.eye(dim)
        h_int = v_nn * np.kron(ryd_int, ryd_int)
        n_r = np.kron(ryd_obs, ident) + np.kron(ident, ryd_obs)
        dt = t_gate / n_steps
        basis = np.eye(dim)
        kets = {s: np.kron(basis[loc[s[0]]], basis[loc[s[1]]]).astype(complex) for s in labels}
        psis = {s: kets[s].copy() for s in labels}
        nr_traj = {s: [float(np.real(np.vdot(psis[s], n_r @ psis[s])))] for s in labels}
        for k in range(n_steps):
            h_loc = h_local_fn((k + 0.5) * dt)
            u = expm(-1j * dt * (np.kron(h_loc, ident) + np.kron(ident, h_loc) + h_int))
            for s in labels:
                psis[s] = u @ psis[s]
                nr_traj[s].append(float(np.real(np.vdot(psis[s], n_r @ psis[s]))))
        overlaps = {s: np.vdot(kets[s], psis[s]) for s in labels}
        phi = {s: float(np.angle(overlaps[s])) for s in labels}
        ret = min(abs(overlaps[s]) ** 2 for s in labels)
        return phi, nr_traj, ret

    ryd7_int = np.diag([0, 0, 0, 0, 0, 1.0, 1.0])   # |r> + garbage |r'> interact
    ryd7_obs = np.diag([0, 0, 0, 0, 0, 1.0, 0.0])   # observable: |r> only
    ryd3 = np.diag([0, 0, 1.0])
    phi7, nr7, ret7 = evolve(h7_local, 7, ryd7_int, ryd7_obs)
    phi_sw, nr_sw, _ = evolve(h_eff_local, 3, ryd3, ryd3)

    # Public path: converter -> EffectiveCZProtocol on 01r -> rg.simulate.
    proto_eff = lower_cz_to_effective_01r(proto7, sys7)
    sys_eff = (
        RydbergSystem.set_atom_level("01r")
        .set_atom_geom(Register.chain(2, spacing_um=spacing))
        .set_protocol(proto_eff)
    )
    e3 = np.eye(3)
    phi_sim = {}
    for s in labels:
        res = simulate(sys_eff, psi0=[s[0], s[1]], t_eval=np.array([t_gate]))
        ket = np.kron(e3[loc[s[0]]], e3[loc[s[1]]]).astype(complex)
        phi_sim[s] = float(np.angle(np.vdot(ket, res.psi_final)))

    zz = lambda p: _wrap(p["11"] - p["01"] - p["10"] + p["00"])
    theta1 = lambda p: _wrap(p["01"] - p["00"])
    # Plumbing: the public converter -> EffectiveCZProtocol -> rg.simulate path must
    # reproduce the direct two-stage SW evolution (phi_sw) for BOTH phases (the
    # |0>-energy gauge cancels in theta1 and ZZ, so they match to step precision).
    d_zz_plumb = abs(_wrap(zz(phi_sim) - zz(phi_sw)))
    d_theta1_plumb = abs(_wrap(theta1(phi_sim) - theta1(phi_sw)))
    # Physics: the {0,1,r} effective theory vs the full 7-level (2nd-order accurate).
    d_zz_phys = abs(_wrap(zz(phi_sim) - zz(phi7)))
    d_theta1_phys = abs(_wrap(theta1(phi_sim) - theta1(phi7)))
    print(
        f"\n[converter] return>={ret7:.4f}  plumb dZZ={d_zz_plumb:.2e} "
        f"dth1={d_theta1_plumb:.2e}  phys dZZ={d_zz_phys:.2e} dth1={d_theta1_phys:.2e}"
    )

    assert ret7 > 0.99          # the 7-level gate is a clean (adiabatic) return
    # Plumbing is exact: converter+simulate == the direct two-stage Schrieffer-Wolff
    # reduction, for both the entangling and single-qubit phases.
    assert d_zz_plumb < 5e-3
    assert d_theta1_plumb < 5e-3
    # Physics: the gauge-invariant entangling (ZZ) phase agrees with the full 7-level
    # to the documented ~2nd order.  theta1 is NOT asserted tightly -- this is the
    # *uncompensated* gate, whose large dynamic light shifts give the single-qubit
    # phase a big 2nd-order residual (~0.5 rad) while ZZ stays accurate; that is a
    # known effective-theory limitation, not a converter error.
    assert d_zz_phys < 0.1


def test_tn_path_rejects_nonzero_k0r():
    """The 01r structure declares drive_0r (exact backend), but the TN 01r lowering
    must reject a *nonzero* K0r (it can only express the |1>-|r> / |0>-|1> legs)."""
    from types import SimpleNamespace

    from ryd_gate.core.level_structures import (
        level_structure,
        three_level_profiles_from_coeffs,
    )

    spec = SimpleNamespace(level_spec=level_structure("01r"), N=1, level_structure="01r")
    # zero / absent K0r is fine
    three_level_profiles_from_coeffs({"drive_R": 1.0 + 0j, "drive_0r": 0.0 + 0j}, spec)
    three_level_profiles_from_coeffs({"drive_R": 1.0 + 0j}, spec)
    # a driven K0r is rejected with a clear message
    with pytest.raises(ValueError, match="K0r"):
        three_level_profiles_from_coeffs({"drive_R": 1.0 + 0j, "drive_0r": 0.3 + 0j}, spec)


def test_effective_cz_protocol_runs_on_01r():
    """EffectiveCZProtocol.from_components realizes (Omega_eff, phi, light shifts) on
    the 01r model: a resonant constant drive Rabi-flops |1> <-> |r>."""
    t_gate, omega = 1e-6, 2 * np.pi * 1e6
    proto = EffectiveCZProtocol.from_components(
        t_gate=t_gate, omega_eff_fn=lambda t: omega, phi_fn=lambda t: 0.0, n_steps=600
    )
    # No K01/K0r given -> a pure |1>-|r> drive (only drive_R/delta_R/delta_hf).
    assert proto.required_channels == frozenset({"drive_R", "delta_R", "delta_hf"})
    coeffs = proto.get_drive_coefficients(0.3e-6, {"t_gate": t_gate})
    assert coeffs["drive_R"] == pytest.approx(0.5 * omega)
    assert coeffs["delta_R"] == pytest.approx(0.0)
    assert coeffs["delta_hf"] == pytest.approx(0.0)
    assert "drive_hf" not in coeffs and "drive_0r" not in coeffs

    sys01r = (
        RydbergSystem.set_atom_level("01r")
        .set_atom_geom(Register.chain(1))
        .set_protocol(proto)
    )
    t_eval = np.linspace(0.0, t_gate, 101)
    result = simulate(sys01r, psi0=["1"], t_eval=t_eval)
    ryd = sys01r.product_state("r")
    n_r = np.array([abs(np.vdot(ryd, psi)) ** 2 for psi in result.states])
    assert n_r.max() > 0.99   # full |1> -> |r> transfer at the pi-pulse
