"""Tests for the DigitalAnalogProtocol direct-matrix-element API (01r only).

Five optional, mutually independent controls, each a direct Hamiltonian matrix
element ``H[ket,bra](t)`` (the compiler adds only the Hermitian conjugate — no
extra ``1/2``). Each returns a scalar (site-uniform) or a length-N per-site
profile.
"""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure, simulate
from ryd_gate.core.lowering import lower_drives
from ryd_gate.protocols import DigitalAnalogProtocol


def _system(protocol, *, preset="01r", n=2, cutoff=None):
    return RydbergSystem(
        level_structure=level_structure(preset),
        register=Register.chain(n),
        protocol=protocol,
        interaction_cutoff_um=cutoff,
    )


def test_compatible_only_with_01r():
    assert DigitalAnalogProtocol._compatible_presets == frozenset({"01r"})
    proto = DigitalAnalogProtocol(t_gate_s=0.1e-6, coupling_r1_rad_s=lambda t: 1.0)
    # 01r binds; 1r (no |0> level) is rejected at binding.
    _system(proto)
    with pytest.raises(ValueError, match="not compatible"):
        _system(proto, preset="1r")


def test_requires_at_least_one_control():
    with pytest.raises(ValueError, match="at least one control"):
        DigitalAnalogProtocol(t_gate_s=0.1e-6)


def test_construction_validation():
    with pytest.raises(ValueError, match="t_gate_s"):
        DigitalAnalogProtocol(t_gate_s=-1.0, coupling_r1_rad_s=lambda t: 1.0)
    with pytest.raises(TypeError, match="coupling_r1_rad_s"):
        DigitalAnalogProtocol(t_gate_s=0.1e-6, coupling_r1_rad_s=5.0)
    with pytest.raises(TypeError, match="energy_r_rad_s"):
        DigitalAnalogProtocol(t_gate_s=0.1e-6, energy_r_rad_s=2.0)


def test_resolved_channel_mapping():
    proto = DigitalAnalogProtocol(
        t_gate_s=0.1e-6,
        coupling_10_rad_s=lambda t: 1.0,
        coupling_r0_rad_s=lambda t: 2.0 + 1.0j,
        coupling_r1_rad_s=lambda t: 3.0,
        energy_1_rad_s=lambda t: -0.25,
        energy_r_rad_s=lambda t: -0.5,
    )
    drives = {d.channel: d for d in proto._resolve(_system(proto)).drives}
    assert set(drives) == {"E[1,0]", "E[r,0]", "E[r,1]", "E[1,1]", "E[r,r]"}

    t = 0.05e-6
    assert complex(drives["E[1,0]"].coefficient(t)) == pytest.approx(1.0)
    assert complex(drives["E[r,0]"].coefficient(t)) == pytest.approx(2.0 + 1.0j)
    assert complex(drives["E[r,1]"].coefficient(t)) == pytest.approx(3.0)
    # diagonal energies stay real scalars
    assert float(drives["E[1,1]"].coefficient(t)) == pytest.approx(-0.25)
    assert float(drives["E[r,r]"].coefficient(t)) == pytest.approx(-0.5)


def test_energy_channel_rejects_complex_matrix_element():
    proto = DigitalAnalogProtocol(
        t_gate_s=0.1e-6, energy_r_rad_s=lambda t: 1.0 + 1.0j
    )
    drive = proto._resolve(_system(proto)).drives[0]
    with pytest.raises(ValueError, match="must be real"):
        drive.coefficient(0.0)


def test_per_site_profile_is_single_array_channel():
    proto = DigitalAnalogProtocol(
        t_gate_s=0.1e-6, coupling_r1_rad_s=lambda t: [1.0, 0.0]
    )
    _t_gate, channels = lower_drives(_system(proto, n=2))
    names = {c.channel for c in channels}
    # per-site profiles are one array-valued channel, NOT suffixed "_0"/"_1"
    assert names == {"E[r,1]"}
    coeff = channels[0].coefficient(0.05e-6)
    np.testing.assert_allclose(np.asarray(coeff), [1.0, 0.0])


def test_site_dependent_coupling_drives_one_site_only():
    # H[r,1] on site 0 = 0.5*omega => effective Rabi = omega; a pi/2 pulse gives
    # P_r ~ 0.5 on site 0 while site 1 (zero coupling) stays in |1>.
    omega = 2 * np.pi * 1e6
    t_pi2 = np.pi / (2 * omega)
    proto = DigitalAnalogProtocol(
        t_gate_s=t_pi2, coupling_r1_rad_s=lambda t: [0.5 * omega, 0.0]
    )
    system = _system(proto, n=2, cutoff=0.0)
    obs = system.observables
    result = simulate(
        system, ["1", "1"],
        observables={"n_r_0": obs.n("r", 0), "n_r_1": obs.n("r", 1)},
    )
    assert result.expectation("n_r_0")[0] == pytest.approx(0.5, abs=0.05)
    assert result.expectation("n_r_1")[0] < 0.05


def test_plot_draws_imaginary_trace_for_complex_coupling():
    # A complex coupling exercises the Im-trace branch of Protocol.plot
    # (base.py:57-58); the axis legend then carries an "Im ..." entry.
    import matplotlib

    matplotlib.use("Agg")
    proto = DigitalAnalogProtocol(
        t_gate_s=0.1e-6, coupling_r0_rad_s=lambda t: 1.0 + 1.0j
    )
    fig = proto.plot(None)  # DigitalAnalogProtocol ignores the system on resolve
    labels = [txt.get_text() for txt in fig.axes[0].get_legend().get_texts()]
    assert any(label.startswith("Im") for label in labels)
