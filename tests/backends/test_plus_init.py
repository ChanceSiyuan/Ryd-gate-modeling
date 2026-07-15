"""Tests for the ``"plus"`` (per-site |+> = (|0>+|1>)/√2) initial state."""

from __future__ import annotations

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate import InteractionSpec, RydbergSystem, level_structure
from ryd_gate.core.states import plus_local_amplitudes, product_superposition_state
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol

OMEGA_R = 2 * np.pi * 3.8e6


def _system_01r(Lx=2, Ly=2):
    proto = DigitalAnalogProtocol(
        t_gate=2e-8,
        coupling_r1=lambda t: 0.5 * OMEGA_R,
        energy_r=lambda t: 0.0,
    )
    return RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(Lx, Ly, spacing_um=6.8),
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
    obs = system.observables
    observables = {}
    for level in ("0", "1", "r"):
        observables.update(obs.site_populations(level))
    res = rg.simulate(system, "plus", t_eval=np.array([0.0]), observables=observables)
    n0 = np.mean([res.expectation(f"n_0_{i}")[0].real for i in range(system.N)])
    n1 = np.mean([res.expectation(f"n_1_{i}")[0].real for i in range(system.N)])
    nr = np.mean([res.expectation(f"n_r_{i}")[0].real for i in range(system.N)])
    assert np.isclose(n0, 0.5, atol=1e-9)
    assert np.isclose(n1, 0.5, atol=1e-9)
    assert np.isclose(nr, 0.0, atol=1e-9)


@pytest.mark.slow
def test_plus_initial_state_mps():
    pytest.importorskip("tenpy")
    system = _system_01r()
    obs = system.observables
    observables = {}
    for level in ("0", "1", "r"):
        observables.update(obs.site_populations(level))
    res = rg.simulate(
        system, "plus", backend="mps", t_eval=np.array([0.0]),
        observables=observables,
        backend_options={"chi_max": 8, "dt": 0.2 / OMEGA_R, "svd_min": 1e-10},
    )
    np.testing.assert_array_equal(res.times, [0.0])
    n0 = np.array([res.expectation(f"n_0_{i}")[0].real for i in range(system.N)])
    n1 = np.array([res.expectation(f"n_1_{i}")[0].real for i in range(system.N)])
    nr = np.array([res.expectation(f"n_r_{i}")[0].real for i in range(system.N)])
    assert np.allclose(n0, 0.5, atol=1e-6)
    assert np.allclose(n1, 0.5, atol=1e-6)
    assert np.allclose(nr, 0.0, atol=1e-6)
