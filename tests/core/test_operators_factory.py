"""Tests for the primitive ``E[ket,bra]`` operator factory, the symbolic
interaction specs (isotropic :class:`RydbergPairInteractionSpec` and the
channel-resolved :class:`ChannelResolvedPairSpec`), and the ``ObservableFactory``
lowering — all built directly on a :class:`BasisSpec` (no RydbergSystem)."""

from __future__ import annotations

import numpy as np
import pytest

from ryd_gate.core.model import BasisSpec
from ryd_gate.core.observables import ObservableFactory
from ryd_gate.core.operators import (
    BasisOperatorFactory,
    ChannelResolvedPairSpec,
    LocalProjectorSpec,
    RydbergPairInteractionSpec,
    SumProjectorSpec,
    TransitionOperatorSpec,
    materialize_sparse_operator,
    parse_E,
)


def _factory():
    basis = BasisSpec(
        site_labels=("0", "1"),
        local_levels=("0", "1", "e1", "e2", "e3", "r", "r_garb"),
        local_dim=7,
        total_dim=49,
    )
    return BasisOperatorFactory(basis)


# ── E[ket,bra] name parsing & factory ────────────────────────────────────────


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


# ── isotropic Rydberg-pair interaction spec (I05) ────────────────────────────


def _basis_2site(levels):
    d = len(levels)
    return BasisSpec(
        site_labels=("0", "1"),
        local_levels=levels,
        local_dim=d,
        total_dim=d ** 2,
    )


def test_pair_projector_builds_isotropic_spec():
    f = BasisOperatorFactory(_basis_2site(("1", "r")))
    spec = f.pair_projector([(0, 1, 3.5)], ("r",))
    assert isinstance(spec, RydbergPairInteractionSpec)
    assert spec.pairs == ((0, 1, 3.5),)
    assert spec.rydberg_levels == ("r",)


def test_isotropic_pair_interaction_materializes_double_rydberg_diagonal():
    basis = _basis_2site(("1", "r"))  # index r == 1, site 0 most significant
    spec = RydbergPairInteractionSpec(((0, 1, 2.0),), ("r",))
    h = materialize_sparse_operator(spec, basis).toarray()
    diag = np.real(np.diag(h))
    # only |rr> (index 3) carries the pair energy V = 2.0
    np.testing.assert_allclose(diag, [0.0, 0.0, 0.0, 2.0])
    assert np.allclose(h, np.diag(np.diag(h)))  # purely diagonal


# ── channel-resolved Rydberg-pair interaction spec (I06) ─────────────────────


def test_channel_resolved_pair_materializes_per_channel_diagonal():
    # Two rydberg levels: index r == 0, r_garb == 1 (site 0 most significant).
    # Basis order: |rr>=0, |r rg>=1, |rg r>=2, |rg rg>=3.
    basis = _basis_2site(("r", "r_garb"))
    v_rr, v_gg, v_rg = 5.0, 7.0, 3.0
    spec = ChannelResolvedPairSpec((
        (0, 1, "r", "r", v_rr),
        (0, 1, "r_garb", "r_garb", v_gg),
        (0, 1, "r", "r_garb", v_rg),
        (0, 1, "r_garb", "r", v_rg),  # both orderings -> symmetric cross term
    ))
    h = materialize_sparse_operator(spec, basis).toarray()
    diag = np.real(np.diag(h))
    np.testing.assert_allclose(diag, [v_rr, v_rg, v_rg, v_gg])
    assert np.allclose(h, np.diag(np.diag(h)))


# ── ObservableFactory lowers to per-site local projectors ────────────────────


def test_observable_factory_lowers_to_local_projector_matrix():
    basis = _basis_2site(("1", "r"))
    obs = ObservableFactory(basis)
    n_r_0 = obs.n("r", 0)
    (term,) = n_r_0._terms
    assert term.coefficient == 1.0
    ((site, matrix),) = term.factors
    assert site == 0
    r_idx = basis.level_index("r")
    expected = np.zeros((2, 2), dtype=complex)
    expected[r_idx, r_idx] = 1.0
    np.testing.assert_array_equal(matrix, expected)
    # a total-Rydberg observable is one such term per site (via Python sum()).
    total = sum(obs.n("r", i) for i in range(basis.n_sites))
    assert [t.factors[0][0] for t in total._terms] == [0, 1]
