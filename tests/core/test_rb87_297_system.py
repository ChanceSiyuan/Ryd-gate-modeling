"""Construction tests for the ``rb87_297_clock_4`` 297 nm single-photon model.

ARC-backed: resolving the preset pulls the σ⁻ branch dipole ratio, the nP₃/₂
lifetimes, and (for multi-atom registers) the channel-resolved, orientation-
dependent 53P C6. The system is protocol-bound (a Direct297 protocol) and the
resolved interaction is a :class:`ChannelResolvedPairSpec` selected via the
``quantization_axis``.
"""

import numpy as np
import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.core.operators import ChannelResolvedPairSpec
from ryd_gate.physics import arc_pair_c6_rad_s_um6, zeeman_shift_rad_s
from ryd_gate.protocols import Direct297PiProtocol

_CLOCK_HYPERFINE = 2 * np.pi * 6.835e9


def _proto():
    return Direct297PiProtocol(omega_297_max_rad_s=2 * np.pi * 1e6)


def _build(register=None, **kwargs):
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", **kwargs),
        register=register if register is not None else Register.chain(1),
        protocol=_proto(),
    )


def _static(system):
    return {t.name: t for t in system._static_terms}


def _pair_spec(system):
    return next(t.operator for t in system._static_terms if t.name == "H_pair")


def test_single_atom_public_fields_and_channel_resolved_interaction():
    s = _build()
    ls = s.level_structure

    assert s.N == 1
    assert ls.name == "rb87_297_clock_4"
    assert ls.levels == ("0", "1", "r", "r_garb")
    assert ls.ryd_level == 53
    assert ls.magnetic_field_G == 20.0
    assert ls.quantization_axis == pytest.approx((0.0, 0.0, 1.0))
    assert set(ls.decay_rates_per_s) == {"r", "r_garb"}
    assert dict(ls.branching_ratios) == {}

    # The interaction is channel-resolved (P-state), with no pairs for one atom.
    spec = _pair_spec(s)
    assert isinstance(spec, ChannelResolvedPairSpec)
    assert spec.channel_terms == ()

    # Dark spectator |0> carries the clock hyperfine energy (seven-level -w_hf).
    assert _static(s)["E[0,0]"].coefficient == pytest.approx(-_CLOCK_HYPERFINE)


def test_leg_ratios_carry_the_sigma_minus_branch_dipole_ratio():
    ls = _build().level_structure
    factors = {leg.channel: leg.factor for leg in ls._laser_legs}
    assert factors["E[r,1]"] == pytest.approx(0.5)
    # Garbage σ⁻ leg carries the 1/√3 Clebsch-Gordan branch ratio.
    assert factors["E[r_garb,1]"] == pytest.approx(0.5 / np.sqrt(3), rel=1e-6)


def test_garb_detuning_follows_zeeman_formula():
    B = 30.0
    s = _build(magnetic_field_G=B)
    expected = zeeman_shift_rad_s(B, l=1, j=1.5, delta_mj=1.0)
    assert expected > 0.0
    assert s.level_structure.magnetic_field_G == B
    assert _static(s)["E[r_garb,r_garb]"].coefficient == pytest.approx(expected)


def test_53p_decay_rates_positive_and_reasonable():
    rates = _build().level_structure.decay_rates_per_s["r"]
    total, radiative = rates["total"], rates["radiative"]
    # 53P3/2 total (300 K) lifetime is of order 100 us; don't pin the exact value.
    assert 1e3 < total < 1e5
    assert 0.0 < radiative < total


# The 53P θ=π/2 channel is not a dominant C6 eigenchannel (overlap ~0.46), so
# materialization warns by design; the warning is asserted elsewhere.
@pytest.mark.filterwarnings("ignore:arc_pair_c6_rad_s_um6:UserWarning")
def test_chain2_pair_uses_channel_resolved_arc_53p_c6_at_theta_pi_2():
    s = _build(Register.chain(2, spacing_um=3.0))
    spec = _pair_spec(s)
    assert isinstance(spec, ChannelResolvedPairSpec)

    # atoms along x, B along z -> pair axis at theta = pi/2, phi = 0.
    rr_terms = [
        V for i, j, a, b, V in spec.channel_terms if a == "r" and b == "r"
    ]
    assert len(rr_terms) == 1
    c6_rr = arc_pair_c6_rad_s_um6(n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.0)
    assert np.isfinite(rr_terms[0]) and rr_terms[0] != 0.0
    assert rr_terms[0] == pytest.approx(c6_rr / 3.0**6)
