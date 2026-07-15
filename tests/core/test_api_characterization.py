"""Behavioral characterization of the rewritten public API.

Complements ``test_init.py`` (which pins the four frozen namespaces) by locking
the constructor/factory contracts the rewrite established:

  * ``RydbergSystem`` requires a bound ``protocol`` (``TypeError`` without it).
  * ``level_structure`` accepts only each preset's declared physical kwargs and
    only the surviving preset names.
  * Names removed by the rewrite are gone from — or relocated out of — the
    namespaces callers used to import them from.

Kept small and ARC-free: only the effective ``1r`` preset is exercised.
"""

import pytest

from ryd_gate import Register, RydbergSystem, level_structure
from ryd_gate.protocols import SweepProtocol


def _sweep(t_gate_s: float = 1e-6) -> SweepProtocol:
    """A trivial constant global sweep compatible with the ``1r`` preset."""
    return SweepProtocol(
        t_gate_s=t_gate_s,
        omega_half_rad_s=lambda t: 0.5,
        detuning_rad_s=lambda t: 0.0,
    )


class TestProtocolIsMandatory:
    def test_missing_protocol_raises_type_error(self):
        """RydbergSystem without a protocol raises TypeError (missing kwarg)."""
        with pytest.raises(TypeError):
            RydbergSystem(
                level_structure=level_structure("1r"),
                register=Register.chain(2),
            )

    def test_explicit_none_protocol_raises_type_error(self):
        """protocol=None is rejected with TypeError."""
        with pytest.raises(TypeError):
            RydbergSystem(
                level_structure=level_structure("1r"),
                register=Register.chain(2),
                protocol=None,
            )

    def test_constructs_with_a_bound_protocol(self):
        """The mandatory-protocol path constructs and resolves t_gate."""
        system = RydbergSystem(
            level_structure=level_structure("1r"),
            register=Register.chain(2),
            protocol=_sweep(1e-6),
        )
        assert system.N == 2
        assert system.t_gate == pytest.approx(1e-6)


class TestLevelStructureKwargs:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"magnetic_field_G": 5.0},  # not accepted by 1r (rb87-only)
            {"t_rise": 1e-8},           # old pulse-shaping field, deleted
            {"Delta": 1e9},             # old static-energy field, deleted
            {"rabi_eff": 1e6},          # old Rabi scale, deleted
            {"time_scale": 1.0},        # old Rabi scale, deleted
            {"C6": 0.0},                # interaction now lives on the register
            {"initial_level": "1"},     # deleted field
        ],
    )
    def test_1r_rejects_removed_kwargs(self, kwargs):
        with pytest.raises(TypeError):
            level_structure("1r", **kwargs)

    def test_1r_accepts_ryd_level(self):
        ls = level_structure("1r", ryd_level=70)
        assert ls.ryd_level == 70

    @pytest.mark.parametrize("name", ["analog_3", "01", "rb87_7", "does_not_exist"])
    def test_deleted_or_unknown_presets_raise_value_error(self, name):
        with pytest.raises(ValueError):
            level_structure(name)


class TestRemovedAndRelocatedNames:
    def test_interaction_spec_removed_from_top_level(self):
        with pytest.raises(ImportError):
            from ryd_gate import InteractionSpec  # noqa: F401

    def test_default_c6_removed_from_top_level(self):
        with pytest.raises(ImportError):
            from ryd_gate import DEFAULT_C6  # noqa: F401

    def test_tfim_protocol_classes_deleted(self):
        with pytest.raises(ImportError):
            from ryd_gate.protocols import TFIMQuenchProtocol  # noqa: F401
        with pytest.raises(ImportError):
            from ryd_gate.protocols import TFIMAnnealProtocol  # noqa: F401

    def test_evolution_result_relocated_to_results(self):
        """EvolutionResult is no longer top-level (nor in ir); it lives in results."""
        with pytest.raises(ImportError):
            from ryd_gate import EvolutionResult  # noqa: F401
        with pytest.raises(ImportError):
            from ryd_gate.ir import EvolutionResult  # noqa: F401

        from ryd_gate.results import EvolutionResult

        assert EvolutionResult is not None
