"""Tests for LevelStructureSpec as the Stage 1 atom-model API."""

import pytest

from ryd_gate import RydbergSystem
from ryd_gate.backends.exact.compiler import ExactSparseCompiler
from ryd_gate.core.level_structures import (
    InteractionSpec,
    LevelStructureSpec,
    TransitionSpec,
    level_structure,
)
from ryd_gate.lattice import Register


def _symbolic_ger_spec() -> LevelStructureSpec:
    """Hand-built symbolic three-level ladder (the custom-model escape hatch)."""
    return LevelStructureSpec(
        name="ger_symbolic",
        levels=("g", "e", "r"),
        rydberg_levels=("r",),
        transitions=(
            TransitionSpec("420", "g", "e", "drive_420"),
            TransitionSpec("1013", "e", "r", "H_1013"),
        ),
        detuning_levels={"delta_e": "e", "delta_R": "r"},
        initial_level="g",
    )


class _AnalogProtocol:
    """Minimal analog-style protocol: drives only the two-photon channels."""

    n_params = 0

    def validate_params(self, x):
        if x:
            raise ValueError("no params")

    def unpack_params(self, x, system):
        self.validate_params(x)
        return {"t_gate": 0.1, "pin_deltas": {}, "scatter_rates": {}, "static_overlays": []}

    def drive_channels(self, system):
        return frozenset({"drive_420", "drive_420_dag"})

    def get_drive_coefficients(self, t, params):
        return {"drive_420": 1.0, "drive_420_dag": 1.0}


class TestPresets:
    def test_01_preset(self):
        spec = level_structure("01")
        assert spec.levels == ("0", "1")
        assert spec.rydberg_levels == ()
        assert spec.initial_level == "0"
        assert spec.interaction_kind == "none"

    def test_initial_level_or_default(self):
        assert level_structure("1r").initial_level_or_default() == "1"
        assert level_structure("01r").initial_level_or_default() == "1"
        assert level_structure("analog_3").initial_level_or_default() == "g"
        assert level_structure("rb87_7").initial_level_or_default() == "0"
        # default rule: levels[0] when initial_level unset
        custom = LevelStructureSpec(name="x", levels=("a", "b"), rydberg_levels=())
        assert custom.initial_level_or_default() == "a"

    def test_physical_kwargs(self):
        assert level_structure("analog_3").physical_kwargs()["param_set"] == "analog_3"
        assert level_structure("rb87_7").physical_kwargs()["param_set"] == "our"
        for name in ("01", "1r", "01r"):
            assert level_structure(name).physical_kwargs() == {}

    def test_supports_backend_matrix(self):
        exact_ok = ("01", "1r", "01r", "analog_3", "rb87_7")
        analog_ok = ("1r", "01r", "analog_3")  # YASTN PEPS + TeNPy MPS lower analog_3
        for name in exact_ok:
            spec = level_structure(name)
            assert spec.supports_backend("exact")
            assert spec.supports_backend("mps") == (name in analog_ok)
            assert spec.supports_backend("peps") == (name in analog_ok)
            assert spec.supports_backend("stabilizer") == (name == "01")
            assert not spec.supports_backend("quantum_teleporter")

    def test_analog_3_keeps_channels(self):
        spec = level_structure("analog_3")
        channels = {t.channel for t in spec.transitions} | set(spec.detuning_levels)
        assert channels == {"drive_420", "H_1013", "delta_e", "delta_R"}

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown level-structure"):
            level_structure("1er")


class TestValidate:
    def test_duplicate_levels_flagged(self):
        spec = LevelStructureSpec(name="x", levels=("a", "a"), rydberg_levels=())
        codes = {issue.code for issue in spec.validate()}
        assert "level_structure.duplicate_levels" in codes

    def test_unknown_initial_level_flagged(self):
        spec = LevelStructureSpec(name="x", levels=("a", "b"), rydberg_levels=(), initial_level="z")
        codes = {issue.code for issue in spec.validate()}
        assert "level_structure.initial_level" in codes

    def test_rydberg_level_not_in_levels_flagged(self):
        spec = LevelStructureSpec(name="x", levels=("a", "b"), rydberg_levels=("r",))
        codes = {issue.code for issue in spec.validate()}
        assert "level_structure.rydberg_levels" in codes

    def test_valid_preset_has_no_issues(self):
        for name in ("01", "1r", "01r", "analog_3", "rb87_7"):
            assert level_structure(name).validate() == []


class TestAnalog3Semantics:
    def test_custom_symbolic_spec_never_infers_analog_3(self):
        """D11/D13: names carry semantics; param_set is numbers only."""
        physical = (
            RydbergSystem.set_atom_level(level_structure("analog_3"))
            .set_atom_geom(Register.chain(2), interaction=InteractionSpec(C6=0.0))
            .build()
        )
        symbolic = (
            RydbergSystem.set_atom_level(_symbolic_ger_spec(), param_set="analog_3")
            .set_atom_geom(Register.chain(2), interaction=InteractionSpec(C6=0.0))
            .build()
        )
        assert physical.basis.local_levels == symbolic.basis.local_levels == ("g", "e", "r")
        assert physical.metadata["physical_model"] == "analog_3"
        assert "physical_model" not in symbolic.metadata
        assert physical.blocks.has("H_const")
        assert not symbolic.blocks.has("H_const")

    def test_semantic_split_physical_vs_symbolic(self):
        analog = (
            RydbergSystem.set_atom_level(level_structure("analog_3"))
            .set_atom_geom(Register.chain(1), interaction=InteractionSpec(C6=0.0))
            .set_protocol(_AnalogProtocol())
        )
        symbolic = (
            RydbergSystem.set_atom_level(_symbolic_ger_spec())
            .set_atom_geom(Register.chain(1), interaction=InteractionSpec(C6=0.0))
            .build()
        )

        # physical blocks mounted only on analog_3
        assert analog.blocks.has("H_const")
        assert analog.blocks.has("H_1013_conj")
        assert "physical_model" not in symbolic.metadata
        assert not symbolic.blocks.has("H_const")
        assert not symbolic.blocks.has("H_1013_conj")

        # the e<->r coupling compiles as a *static* term for analog_3
        params = analog.unpack_params([])
        ir = ExactSparseCompiler().compile(analog, params)
        assert "H_1013" in {term.name for term in ir.static_terms}


class TestSerialization:
    def test_preset_roundtrip(self):
        spec = level_structure("analog_3")
        data = spec.to_dict()
        assert data["schema"] == "ryd-gate/level-structure/v1"
        assert LevelStructureSpec.from_dict(data) == spec

    def test_custom_roundtrip(self):
        spec = LevelStructureSpec(
            name="custom_gr",
            levels=("g", "r"),
            rydberg_levels=("r",),
            initial_level="g",
            interaction_kind="ising_c6",
            params={"note": "test"},
        )
        assert LevelStructureSpec.from_dict(spec.to_dict()) == spec

    def test_wrong_schema_raises(self):
        data = level_structure("1r").to_dict()
        data["schema"] = "ryd-gate/register/v1"
        with pytest.raises(ValueError, match="schema"):
            LevelStructureSpec.from_dict(data)
