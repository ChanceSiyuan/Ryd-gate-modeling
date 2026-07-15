"""Tests for the primitive ``E[ket,bra]`` operator factory and the operator-model
surface of ``RydbergSystem`` (no public ``blocks``)."""

from __future__ import annotations

import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.core.model import BasisSpec
from ryd_gate.core.operators import (
    BasisOperatorFactory,
    LocalProjectorSpec,
    SumProjectorSpec,
    TransitionOperatorSpec,
    parse_E,
)
from ryd_gate.lattice import Register


def _factory():
    basis = BasisSpec(
        site_labels=("0", "1"),
        local_levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        local_dim=7,
        total_dim=49,
    )
    return BasisOperatorFactory(basis)


def test_parse_E_tolerates_underscore_labels():
    assert parse_E("E[r_garb,e1]") == ("r_garb", "e1")
    assert parse_E("E[r,r]") == ("r", "r")


def test_local_diagonal_is_projector():
    f = _factory()
    assert f.local("E[r,r]", 0) == LocalProjectorSpec("r", 0)
    assert f.sum("E[r,r]") == SumProjectorSpec("r")


def test_local_offdiagonal_is_transition():
    f = _factory()
    # E[ket,bra] = |ket><bra| -> TransitionOperatorSpec(lower=bra, upper=ket)
    assert f.local("E[r_garb,e1]", 0) == TransitionOperatorSpec(lower="e1", upper="r_garb", site=0)
    assert f.sum("E[e1,1]") == TransitionOperatorSpec(lower="1", upper="e1", site=None)


@pytest.mark.parametrize("bad", ["E[x,1]", "E[r,bogus]", "E[r]", "Erbra", "E[r,]", "E[,r]", "r_garb"])
def test_invalid_names_raise(bad):
    f = _factory()
    with pytest.raises(ValueError):
        f.local(bad, 0)


def test_is_hermitian():
    assert BasisOperatorFactory.is_hermitian("E[r,r]") is True
    assert BasisOperatorFactory.is_hermitian("E[r,e1]") is False
    assert BasisOperatorFactory.is_hermitian("E[r_garb,r_garb]") is True


def _chain2(name, spacing_um=4.0):
    return RydbergSystem(
        level_structure=level_structure(name),
        register=Register.chain(2, spacing_um=spacing_um),
    )


def test_system_has_no_public_blocks():
    s = _chain2("1r")
    assert not hasattr(s, "blocks")
    assert isinstance(s.operators, BasisOperatorFactory)


def test_symbolic_hamiltonian_channels_are_E_names():
    o1r = _chain2("01r")
    assert set(o1r.hamiltonian_channels) == {"E[r,1]", "E[1,0]", "E[r,0]", "E[r,r]", "E[1,1]"}
    one_r = _chain2("1r")
    assert set(one_r.hamiltonian_channels) == {"E[r,1]", "E[r,r]"}
    # every channel key is an E[...] name (no physical aliases)
    for chan in set(o1r.hamiltonian_channels) | set(one_r.hamiltonian_channels):
        assert chan.startswith("E[") and chan.endswith("]")


def test_observables_factory_lowers_to_local_projectors():
    """system.observables builds expressions whose lowered form is per-site
    |level><level| matrices on the declared basis."""
    import numpy as np

    s = _chain2("1r")
    n_r_0 = s.observables.n("r", 0)
    (term,) = n_r_0._terms
    assert term.coefficient == 1.0
    ((site, matrix),) = term.factors
    assert site == 0
    r_idx = s.basis.level_index("r")
    expected = np.zeros((2, 2), dtype=complex)
    expected[r_idx, r_idx] = 1.0
    np.testing.assert_array_equal(matrix, expected)
    # level_sum('r') = one such term per site
    total = s.observables.level_sum("r")
    assert [t.factors[0][0] for t in total._terms] == [0, 1]


@pytest.mark.parametrize("tag", ["rb87_7_mp", "rb87_7_pm"])
def test_rb87_primitive_channels(tag):
    s = RydbergSystem(level_structure=level_structure(tag), register=Register.chain(1))
    expected = {
        "E[e1,1]", "E[e2,1]", "E[e3,1]", "E[e1,0]", "E[e2,0]", "E[e3,0]",
        "E[r,e1]", "E[r,e2]", "E[r,e3]", "E[r_garb,e1]", "E[r_garb,e2]", "E[r_garb,e3]",
    }
    assert set(s.hamiltonian_channels) == expected
    # the diagonal energies live in static terms (E[a,a]); the pair term too.
    assert any(t.name == "H_pair" for t in s.static_hamiltonian_terms)
    assert any(t.name.startswith("E[") for t in s.static_hamiltonian_terms)


def test_no_laser_system_still_has_channels():
    # A no-protocol rb87 system still exposes its allowed primitive channels.
    s = _chain2("rb87_7_mp", spacing_um=3.0)
    assert s.protocol is None
    assert "E[e1,1]" in s.hamiltonian_channels
    assert "E[r,e1]" in s.hamiltonian_channels
