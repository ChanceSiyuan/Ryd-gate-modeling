"""Tests for level_structure() presets as the atom-model API."""

import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.backends.exact.compiler import ExactCompiler
from ryd_gate.core.level_structures import InteractionSpec


class _AnalogProtocol:
    """Minimal analog-style protocol: drives only the two-photon channels."""

    def _resolve(self, system):
        return {"t_gate": 0.1}

    def drive_channels(self, system):
        return frozenset({"E[e,g]"})

    def get_drive_coefficients(self, t, ctx):
        return {"E[e,g]": 1.0}


class TestPresets:
    def test_01_preset_removed(self):
        with pytest.raises(ValueError, match="removed"):
            level_structure("01")

    def test_initial_level_or_default(self):
        assert level_structure("1r").initial_level_or_default() == "1"
        assert level_structure("01r").initial_level_or_default() == "1"
        assert level_structure("analog_3").initial_level_or_default() == "g"
        assert level_structure("rb87_7_mp").initial_level_or_default() == "0"

    def test_supports_backend_matrix(self):
        exact_ok = ("1r", "01r", "analog_3", "rb87_7_mp", "rb87_7_pm", "rb87_297_clock_4")
        analog_ok = ("1r", "01r", "analog_3")  # YASTN PEPS + TeNPy MPS lower analog_3
        for name in exact_ok:
            spec = level_structure(name)
            assert spec.supports_backend("exact_ode")
            assert spec.supports_backend("mps") == (name in analog_ok)
            assert spec.supports_backend("peps") == (name in analog_ok)
            assert not spec.supports_backend("exact")  # removed backend name
            assert not spec.supports_backend("stabilizer")  # capability removed
            assert not spec.supports_backend("quantum_teleporter")

    def test_analog_3_keeps_channels(self):
        spec = level_structure("analog_3")
        channels = {t.channel for t in spec.transitions} | set(spec.detuning_levels)
        assert channels == {"E[e,g]", "E[r,e]", "E[e,e]", "E[r,r]"}

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown level-structure"):
            level_structure("1er")


class TestAnalog3Semantics:
    def test_physical_blocks_mounted(self):
        analog = RydbergSystem(
            level_structure=level_structure("analog_3"),
            register=Register.chain(2),
            interaction=InteractionSpec(C6=0.0),
        )
        assert analog.basis.local_levels == ("g", "e", "r")
        assert analog.level_structure.physical_model == "analog_3"
        assert any(t.name.startswith("E[") for t in analog.static_hamiltonian_terms)

    def test_static_er_coupling_compiles_as_static_term(self):
        analog = RydbergSystem(
            level_structure=level_structure("analog_3"),
            register=Register.chain(1),
            interaction=InteractionSpec(C6=0.0),
            protocol=_AnalogProtocol(),
        )

        # the 1013 e<->r leg is a static (h.c.-completed) coupling on analog_3
        assert any(
            t.name == "E[r,e]" and t.add_hermitian_conjugate
            for t in analog.static_hamiltonian_terms
        )
        ir = ExactCompiler().compile(analog)
        assert "E[r,e]" in {term.name for term in ir.static_terms}
