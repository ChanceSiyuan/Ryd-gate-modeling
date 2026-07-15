"""Construction tests for the ``rb87_297_clock_4`` 297 nm single-photon model.

ARC-backed: materialization pulls the branch dipole ratio, nP3/2 lifetimes,
and (for multi-atom registers) the pair-dependent 53P C6.
"""

import numpy as np
import pytest

from ryd_gate import InteractionSpec, Register, RydbergSystem, level_structure
from ryd_gate.physics import arc_pair_c6_rad_s_um6, zeeman_shift_rad_s


def _build(register=None, *, interaction=None, **kwargs):
    return RydbergSystem(
        level_structure=level_structure("rb87_297_clock_4", **kwargs),
        register=register if register is not None else Register.chain(1),
        interaction=interaction,
    )


def test_single_atom_materializes():
    s = _build()
    assert s.N == 1
    assert s.basis.local_levels == ("0", "1", "r", "r_garb")
    assert "E[r,1]" in s.hamiltonian_channels
    assert "E[r_garb,1]" in s.hamiltonian_channels
    assert s.level_structure.physical_model == "rb87_297_clock_4"
    assert s.level_structure.name == "rb87_297_clock_4"
    assert s.basis.local_dim == 4
    assert s.level_structure.rydberg_indices == (2, 3)
    assert s.level_structure.ryd_level == 53
    assert s.interaction_pairs == ()
    ratios = s.level_structure.laser_channel_ratios["297"]
    assert ratios["E[r,1]"] == pytest.approx(0.5)
    # Unit-Rabi garbage leg carries the sigma- branch dipole ratio 1/sqrt(3).
    assert ratios["E[r_garb,1]"] == pytest.approx(0.5 / np.sqrt(3), rel=1e-6)
    # The dark spectator |0> carries the static clock hyperfine energy, matching
    # the seven-level h[0,0] = -w_hf convention.
    term = {t.name: t for t in s.static_hamiltonian_terms}["E[0,0]"]
    assert term.coefficient == pytest.approx(-2 * np.pi * 6.835e9)


def test_garb_detuning_follows_zeeman_formula():
    B = 30.0
    s = _build(magnetic_field_G=B)
    expected = zeeman_shift_rad_s(B, l=1, j=1.5, delta_mj=1.0)
    assert s.level_structure.garb_zeeman_detuning == pytest.approx(expected)
    assert s.level_structure.magnetic_field_G == B
    # The low-field clock state has a shared |F=2,mF=0> ground energy; positive
    # B leaves the garbage branch above the target by the excited-state splitting.
    assert expected > 0.0
    # The static |r_garb> diagonal carries the detuning (decay off by default).
    term = {t.name: t for t in s.static_hamiltonian_terms}["E[r_garb,r_garb]"]
    assert term.coefficient == pytest.approx(expected)


def test_53p_decay_rates_positive_and_reasonable():
    ls = _build().level_structure
    gamma = ls.ryd_state_decay_rate
    rd = ls.ryd_RD_rate
    bbr = ls.ryd_BBR_rate
    # 53P3/2 total (300 K) lifetime is of order 100 us; don't pin the exact value.
    assert 1e3 < gamma < 1e5
    assert 0.0 < rd < gamma
    assert bbr == pytest.approx(gamma - rd)


def test_decay_flag_controls_r_diagonal():
    names_off = {t.name for t in _build(enable_rydberg_decay=False).static_hamiltonian_terms}
    assert "E[r,r]" not in names_off  # resonant |r> has no static energy without decay
    s_on = _build(enable_rydberg_decay=True)
    term = {t.name: t for t in s_on.static_hamiltonian_terms}["E[r,r]"]
    assert term.coefficient == pytest.approx(
        -0.5j * s_on.level_structure.ryd_state_decay_rate
    )


# The 53P θ=π/2 channel is not a dominant C6 eigenchannel (overlap ~0.46), so
# materialization warns by design; the warning itself is asserted in
# test_direct_297_physics.py::test_arc_c6_53p_degenerate_finite_and_warns_on_weak_overlap.
@pytest.mark.filterwarnings("ignore:arc_pair_c6_rad_s_um6:UserWarning")
def test_chain2_pair_uses_arc_53p_c6_at_theta_pi_2():
    s = _build(Register.chain(2, spacing_um=3.0))
    pairs = s.interaction_pairs
    assert len(pairs) == 1
    _, _, V = pairs[0]
    c6 = arc_pair_c6_rad_s_um6(n1=53, l1=1, j1=1.5, mj1=-1.5, theta=np.pi / 2, phi=0.0)
    assert np.isfinite(V) and V != 0.0
    assert V == pytest.approx(c6 / 3.0**6)
    assert max(abs(v) for *_, v in pairs) == pytest.approx(abs(V))


@pytest.mark.filterwarnings("ignore:arc_pair_c6_rad_s_um6:UserWarning")
def test_2d_register_materializes_with_uniform_in_plane_c6():
    # B along z, atoms in the xy plane: every pair has theta = pi/2, and the C6
    # eigenchannel is phi-independent, so V_ij * R_ij^6 is the same for all pairs.
    s = _build(Register.square(2, spacing_um=4.0))
    pairs = s.interaction_pairs
    assert len(pairs) == 6
    coords = s.register.coords
    c6s = [V * float(np.linalg.norm(coords[i] - coords[j])) ** 6 for i, j, V in pairs]
    assert np.allclose(c6s, c6s[0], rtol=1e-9)


def test_explicit_interaction_spec_overrides_with_scalar_c6():
    C6 = 2 * np.pi * 100e9
    s = _build(Register.chain(2, spacing_um=3.0), interaction=InteractionSpec(C6=C6))
    ((_, _, V),) = s.interaction_pairs
    assert V == pytest.approx(C6 / 3.0**6)


def test_rejects_unknown_physical_kwargs():
    with pytest.raises(TypeError, match="does not accept physical parameter"):
        level_structure("rb87_297_clock_4", Delta_Hz=1e9)
