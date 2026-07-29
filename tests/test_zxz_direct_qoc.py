"""Pins for the ZXZ direct-QOC study (spec 2026-07-28 §4-§5)."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zxz_direct_qoc import (  # noqa: E402
    TAU,
    TAU_JEFF,
    build_model,
    build_target,
    build_zxz,
    fidelity,
    unitary_infidelity,
)

_Z = np.diag([1.0, -1.0]).astype(complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _lex_index():
    labels = [tuple(p) for p in product("1r", repeat=3)]
    return {lab: i for i, lab in enumerate(labels)}


def test_zxz_matrix_pin():
    # site 0 most significant in the lexicographic ordering -> Z (x) X (x) Z
    index = _lex_index()
    expected = np.kron(np.kron(_Z, _X), _Z)
    np.testing.assert_allclose(build_zxz(index), expected, atol=1e-14)
    target = build_target(index)
    np.testing.assert_allclose(target, expm(-1j * TAU_JEFF * expected), atol=1e-12)


def test_unitary_infidelity_gradient_convention():
    rng = np.random.default_rng(4)
    target = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    u = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))[0]
    value, g = unitary_infidelity(u, target)
    assert value == pytest.approx(1.0 - fidelity(u, target))
    h = 1e-7
    for i, j in ((0, 0), (2, 5)):
        e = np.zeros((8, 8), dtype=complex)
        e[i, j] = h
        fd_re = (unitary_infidelity(u + e, target)[0] - unitary_infidelity(u - e, target)[0]) / (2 * h)
        fd_im = (unitary_infidelity(u + 1j * e, target)[0] - unitary_infidelity(u - 1j * e, target)[0]) / (2 * h)
        assert abs(2.0 * g[i, j].real - fd_re) < 1e-6
        assert abs(2.0 * g[i, j].imag - fd_im) < 1e-6


@pytest.mark.slow
def test_model_pins_arc():
    from ryd_gate import Register, RydbergSystem, level_structure, simulate
    from ryd_gate.protocols import SweepProtocol

    model = build_model()
    index = model["index"]
    # bijective basis mapping
    assert sorted(index.values()) == list(range(8))
    # NN vdW: +C6/8.9^6 with ARC 70S C6 ~ 2pi x 862.7 GHz um^6 -> ~2pi x 1.736 MHz
    i_rr1 = index[("r", "r", "1")]
    i_r1r = index[("r", "1", "r")]
    v_nn = model["h0"][i_rr1, i_rr1].real
    v_nnn = model["h0"][i_r1r, i_r1r].real
    assert v_nn > 0.0, "repulsive S-state vdW expected; sign convention broke"
    assert abs(v_nn / TAU - 1.736) < 0.04
    assert abs(v_nnn - v_nn / 64.0) < 1e-6 * v_nn
    # constant-control parity: discrete chain vs exact_ode (ADR-0024 style)
    u_om, u_de, t_us = 1.0, 0.7, 0.1
    h = model["h0"] + u_om * model["controls"]["E[r,1]:x"] + u_de * model["controls"]["E[r,r]"]
    psi0 = np.zeros(8, dtype=complex)
    psi0[index[("1", "1", "1")]] = 1.0
    psi_disc = expm(-1j * t_us * h) @ psi0
    protocol = SweepProtocol(
        t_gate_s=t_us * 1e-6,
        omega_half_rad_s=lambda t: u_om * 1e6,
        detuning_rad_s=lambda t: -u_de * 1e6,
    )
    system = RydbergSystem(
        level_structure=level_structure("1r", ryd_level=70),
        register=Register.chain(3, spacing_um=8.9),
        protocol=protocol,
    )
    result = simulate(system, ["1", "1", "1"], backend="exact_ode")
    psi_ode = np.array(
        [result.amplitude(list(lab)) for lab, _ in sorted(index.items(), key=lambda kv: kv[1])]
    )
    overlap = abs(np.vdot(psi_disc, psi_ode))
    assert overlap > 1.0 - 1e-9
