"""The ``"plus"`` per-site initial state ``|+> = (|0>+|1>)/sqrt(2)`` on ``01r``.

``simulate(system, "plus", ...)`` prepares every site in the equal
``|0>``/``|1>`` superposition (no Rydberg amplitude), on the exact_ode backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import ryd_gate as rg
from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.states import _plus_local_amplitudes
from ryd_gate.protocols import DigitalAnalogProtocol

OMEGA_R = 2 * np.pi * 3.8e6


def _system_01r(rows=2, cols=1):
    return RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.rectangle(rows, cols, spacing_um=6.8),
        protocol=DigitalAnalogProtocol(
            t_gate_s=2e-8,
            coupling_r1_rad_s=lambda t: 0.5 * OMEGA_R,
            energy_r_rad_s=lambda t: 0.0,
        ),
    )


def test_plus_local_amplitudes_requires_0_and_1():
    np.testing.assert_allclose(
        _plus_local_amplitudes(("0", "1", "r")), np.array([1, 1, 0]) / np.sqrt(2)
    )
    with pytest.raises(ValueError, match="requires '0' and '1'"):
        _plus_local_amplitudes(("1", "r"))


def test_plus_initial_state_exact():
    """At t=0 the ``"plus"`` state has n0 = n1 = 1/2 and nr = 0 on every site."""
    system = _system_01r()  # 2x1 register: dim 3**2 = 9
    obs = system.observables
    observables = {
        f"n_{level}_{i}": obs.n(level, i)
        for level in ("0", "1", "r")
        for i in range(system.N)
    }
    res = rg.simulate(system, "plus", t_eval=np.array([0.0]), observables=observables)
    np.testing.assert_array_equal(res.times, [0.0])

    n0 = np.mean([res.expectation(f"n_0_{i}")[0] for i in range(system.N)])
    n1 = np.mean([res.expectation(f"n_1_{i}")[0] for i in range(system.N)])
    nr = np.mean([res.expectation(f"n_r_{i}")[0] for i in range(system.N)])
    assert np.isclose(n0, 0.5, atol=1e-9)
    assert np.isclose(n1, 0.5, atol=1e-9)
    assert np.isclose(nr, 0.0, atol=1e-9)
