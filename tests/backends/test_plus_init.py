"""Tests for the ``"plus"`` (per-site |+> = (|0>+|1>)/√2) initial state."""

from __future__ import annotations

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate import InteractionSpec, RydbergSystem
from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol

OMEGA_R = 2 * np.pi * 3.8e6


def _system_01r(Lx=2, Ly=2):
    proto = DigitalAnalogProtocol(
        t_gate=2e-8,
        omega_R_fn=lambda t: OMEGA_R,
        delta_R_fn=lambda t: 0.0,
    )
    return RydbergSystem.from_lattice(
        Register.rectangle(Lx, Ly, spacing_um=6.8),
        "01r",
        interaction=InteractionSpec(C6=2 * np.pi * 874e9, mode="nn"),
        protocol=proto,
    )


def test_product_superposition_state_normalized():
    v = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    psi = product_superposition_state(v, 3)
    assert psi.shape == (3**3,)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_plus_local_amplitudes_requires_0_and_1():
    assert np.allclose(plus_local_amplitudes(("0", "1", "r")), np.array([1, 1, 0]) / np.sqrt(2))
    with pytest.raises(ValueError, match="requires '0' and '1'"):
        plus_local_amplitudes(("1", "r"))


def test_plus_initial_state_exact():
    system = _system_01r()
    res = rg.simulate(system, [], "plus", backend="exact", t_eval=np.array([0.0]))
    psi0 = res.states[0]
    n0 = np.mean([system.expectation(f"n_0_{i}", psi0) for i in range(system.N)])
    n1 = np.mean([system.expectation(f"n_1_{i}", psi0) for i in range(system.N)])
    nr = np.mean([system.expectation(f"n_r_{i}", psi0) for i in range(system.N)])
    assert np.isclose(n0, 0.5, atol=1e-9)
    assert np.isclose(n1, 0.5, atol=1e-9)
    assert np.isclose(nr, 0.0, atol=1e-9)


@pytest.mark.slow
def test_plus_initial_state_mps():
    pytest.importorskip("tenpy")
    system = _system_01r()
    res = rg.simulate(
        system, [], "plus", backend="mps", t_eval=np.array([0.0]),
        observables=["n_0", "n_1", "n_r"],
        backend_options={"chi_max": 8, "dt": 0.2 / OMEGA_R, "svd_min": 1e-10},
    )
    obs = res.metadata["obs"]
    assert np.allclose(obs["n_0"][0], 0.5, atol=1e-6)
    assert np.allclose(obs["n_1"][0], 0.5, atol=1e-6)
    assert np.allclose(obs["n_r"][0], 0.0, atol=1e-6)
