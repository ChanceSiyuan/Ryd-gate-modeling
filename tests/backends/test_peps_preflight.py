"""PEPS geometry/topology preflight + provenance (no YASTN import; PEPS28/29/31).

Only ``Register.chain``/``rectangle``/``square`` factory provenance reaches the
PEPS lattice; direct coordinates that happen to form a grid are still rejected
(no shape inference). Interactions must lie on Cartesian nearest-neighbour edges.
Also covers Register provenance privacy/survival and ``peps_evidence is None`` on
non-PEPS results.
"""

import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.peps._layout import peps_lattice_spec, validate_and_map_pairs
from ryd_gate.backends.peps._numerics import PEPSError
from ryd_gate.lattice import Register
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.results import EvolutionResult, GroundStateResult


class _FakeTerms:
    """Minimal stand-in for compiled ``TNTerms`` (only what _layout reads)."""

    def __init__(self, n_sites, pairs):
        self.n_sites = n_sites
        self.pairs = tuple(pairs)


class TestLatticeSpec:
    def test_chain_shape_mapping_edges(self):
        spec = peps_lattice_spec(Register.chain(4))
        assert spec.shape == (4, 1)
        assert spec.site_to_coord == ((0, 0), (1, 0), (2, 0), (3, 0))
        assert spec.allowed_edges == frozenset({(0, 1), (1, 2), (2, 3)})

    def test_rectangle_shape_mapping_edges(self):
        spec = peps_lattice_spec(Register.rectangle(2, 3))
        assert spec.shape == (2, 3)
        # row-major i = row*cols + col
        assert spec.site_to_coord == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))
        # horizontal (col neighbours) and vertical (row neighbours)
        assert (0, 1) in spec.allowed_edges  # (0,0)-(0,1)
        assert (0, 3) in spec.allowed_edges  # (0,0)-(1,0)
        assert (0, 4) not in spec.allowed_edges  # (0,0)-(1,1) diagonal

    def test_square_provenance(self):
        spec = peps_lattice_spec(Register.square(3))
        assert spec.shape == (3, 3)
        assert len(spec.allowed_edges) == 2 * 3 * (3 - 1)  # 2*side*(side-1)

    def test_custom_coords_identical_to_rectangle_rejected(self):
        rect = Register.rectangle(2, 2)
        custom = Register(rect.coords)  # same floats, no factory provenance
        with pytest.raises(PEPSError, match="Register.chain"):
            peps_lattice_spec(custom)

    def test_triangular_rejected(self):
        with pytest.raises(PEPSError, match="exact_ode.*mps|mps"):
            peps_lattice_spec(Register.triangular(2, 3))


class TestTopologyValidation:
    def test_nearest_neighbour_subset_mapped_with_coeffs(self):
        spec = peps_lattice_spec(Register.rectangle(2, 2))
        # sites: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1); NN edges (0,1),(0,2),(1,3),(2,3)
        terms = _FakeTerms(4, [(0, 1, 2.5), (2, 3, 1.5)])
        mapped = validate_and_map_pairs(spec, terms)
        assert mapped == (((0, 0), (0, 1), 2.5), ((1, 0), (1, 1), 1.5))

    def test_zero_coefficient_pair_ignored(self):
        spec = peps_lattice_spec(Register.rectangle(2, 2))
        terms = _FakeTerms(4, [(0, 1, 0.0), (0, 2, 1.0)])
        mapped = validate_and_map_pairs(spec, terms)
        assert mapped == (((0, 0), (1, 0), 1.0),)

    def test_diagonal_pair_rejected(self):
        spec = peps_lattice_spec(Register.rectangle(2, 2))
        terms = _FakeTerms(4, [(0, 3, 1.0)])  # (0,0)-(1,1) diagonal
        with pytest.raises(PEPSError, match="nearest-neighbour"):
            validate_and_map_pairs(spec, terms)

    def test_long_range_chain_pair_rejected(self):
        spec = peps_lattice_spec(Register.chain(4))
        terms = _FakeTerms(4, [(0, 2, 1.0)])
        with pytest.raises(PEPSError, match="nearest-neighbour"):
            validate_and_map_pairs(spec, terms)

    def test_self_pair_rejected(self):
        spec = peps_lattice_spec(Register.chain(3))
        with pytest.raises(PEPSError, match="0 <= i < j"):
            validate_and_map_pairs(spec, _FakeTerms(3, [(1, 1, 1.0)]))

    def test_out_of_range_pair_rejected(self):
        spec = peps_lattice_spec(Register.chain(3))
        with pytest.raises(PEPSError, match="0 <= i < j"):
            validate_and_map_pairs(spec, _FakeTerms(3, [(0, 5, 1.0)]))

    def test_nonfinite_coefficient_rejected(self):
        spec = peps_lattice_spec(Register.chain(3))
        with pytest.raises(PEPSError, match="non-finite"):
            validate_and_map_pairs(spec, _FakeTerms(3, [(0, 1, float("inf"))]))

    def test_site_count_mismatch_rejected(self):
        spec = peps_lattice_spec(Register.chain(3))
        with pytest.raises(PEPSError, match="n_sites"):
            validate_and_map_pairs(spec, _FakeTerms(4, []))


class TestProvenance:
    def test_no_public_provenance_attribute(self):
        reg = Register.rectangle(2, 3)
        assert "_origin" not in dir(reg) or not any(
            n in ("origin", "factory", "grid_shape", "provenance") for n in dir(reg)
        )
        # the only public geometry surface stays coords / N
        assert reg.coords.shape == (6, 2)
        assert reg.N == 6

    def test_provenance_survives_noise_realization(self):
        system = RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.rectangle(2, 2, spacing_um=8.0),
            protocol=SweepProtocol(
                t_gate_s=0.2e-6,
                omega_half_rad_s=lambda t: 0.0,
                detuning_rad_s=lambda t: 0.0,
            ),
            interaction_cutoff_um=0.0,
        )
        noisy = system._with_realization({"position_offsets_um": ((0.1, 0.0, 0.0),) * 4})
        # the nominal register (and thus its factory provenance) is reused unchanged
        spec = peps_lattice_spec(noisy.register)
        assert spec.shape == (2, 2)


class TestPepsEvidenceAbsentOnOtherBackends:
    class _PlainReader:
        def amplitude(self, labels):
            return 1 + 0j

        def sample(self, shots, seed):
            return {}

    def test_evolution_result_without_ledger_is_none(self):
        res = EvolutionResult(times=[1.0], expectations={}, reader=self._PlainReader())
        assert res.peps_evidence is None

    def test_ground_result_without_ledger_is_none(self):
        class _GSReader:
            def amplitude(self, labels, phase_reference):
                return 1 + 0j

            def sample(self, shots, seed):
                return {}

        res = GroundStateResult(expectations={"energy": -1.0}, reader=_GSReader())
        assert res.peps_evidence is None
