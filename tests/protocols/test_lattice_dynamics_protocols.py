"""TFIM quench/anneal physics expressed as a SweepProtocol (P16).

The dedicated TFIM protocol classes are deleted. The transverse-field Ising
dynamics is written directly on the |1>-|r> channels of a ``1r`` system::

    H[r,1](t) = Omega/2 = h_x(t)        # transverse field
    H[r,r](t) = -Delta(t)               # longitudinal field / detuning
    H[r,r]_i += -local_detuning(t, i)   # optional per-site (boundary) pinning

These tests build such SweepProtocols and assert the compiled channel drives.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import SweepProtocol

MHZ = 2 * np.pi * 1e6


def _square_system(protocol, *, side=2, cutoff=None):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.square(side, spacing_um=9.0),
        protocol=protocol,
        interaction_cutoff_um=cutoff,
    )


def _channels_by_name(system):
    _t_gate, channels = lower_drives(system)
    return {c.channel: c for c in channels}


def test_tfim_quench_emits_constant_transverse_field_and_detuning():
    h_x = 1.0 * MHZ
    delta = 4.0 * MHZ
    t_gate = 0.5e-6
    quench = SweepProtocol(
        t_gate_s=t_gate,
        omega_half_rad_s=lambda t: h_x,
        detuning_rad_s=lambda t: delta,
    )
    system = _square_system(quench)
    by = _channels_by_name(system)

    assert system.t_gate == pytest.approx(t_gate)
    assert set(by) == {"E[r,1]", "E[r,r]"}
    t = 0.5 * t_gate
    assert complex(by["E[r,1]"].coefficient(t)) == pytest.approx(h_x)      # h_x = Omega/2
    assert float(by["E[r,r]"].coefficient(t)) == pytest.approx(-delta)     # -Delta
