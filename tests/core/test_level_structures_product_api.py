"""Tests for ``level_structure()`` presets as the atom-model API.

``level_structure(name, **physical_kwargs)`` is the only way to obtain a
:class:`LevelStructure`. Its public surface is exactly the seven read-only
characterization fields (S17); everything the compiler consumes stays private.
Unknown preset names, the deleted ``analog_3`` / ``01`` presets, and unknown
physical kwargs all raise.
"""

import dataclasses

import numpy as np
import pytest

from ryd_gate import level_structure

# The exact public (S17) field set.
_PUBLIC_FIELDS = {
    "name",
    "levels",
    "ryd_level",
    "magnetic_field_G",
    "quantization_axis",
    "decay_rates_per_s",
    "branching_ratios",
}

# Attributes that existed on the old level-structure object and are now gone.
_REMOVED_ATTRS = [
    "Delta", "t_rise", "rabi_eff", "time_scale", "default_c6", "initial_level",
    "initial_level_or_default", "supports_backend", "transitions",
    "detuning_levels", "physical_model", "rydberg_indices", "index",
    "local_dim", "laser_channel_ratios",
]


# ── effective presets (fast; a single cached ARC C6 call) ────────────────────


def test_1r_public_fields():
    ls = level_structure("1r")
    assert ls.name == "1r"
    assert ls.levels == ("1", "r")
    assert ls.ryd_level == 70  # default
    assert ls.magnetic_field_G is None
    assert ls.quantization_axis is None
    assert dict(ls.decay_rates_per_s) == {}
    assert dict(ls.branching_ratios) == {}


def test_01r_public_fields():
    # ryd_level is accepted (default 70, cached ARC C6); assert it flows through.
    ls = level_structure("01r", ryd_level=70)
    assert ls.name == "01r"
    assert ls.levels == ("0", "1", "r")
    assert ls.ryd_level == 70


def test_public_surface_is_exactly_s17():
    ls = level_structure("1r")
    field_names = {f.name for f in dataclasses.fields(ls) if not f.name.startswith("_")}
    assert field_names == _PUBLIC_FIELDS


@pytest.mark.parametrize("attr", _REMOVED_ATTRS)
def test_removed_attributes_absent(attr):
    ls = level_structure("1r")
    assert not hasattr(ls, attr)


def test_level_structure_is_immutable():
    ls = level_structure("1r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ls.name = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ls.ryd_level = 100


# ── preset / kwarg validation ────────────────────────────────────────────────


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure("1er")


@pytest.mark.parametrize("name", ["analog_3", "01", "rb87_7", "ger"])
def test_deleted_presets_raise(name):
    # analog_3 and '01' are deleted; 'rb87_7' and 'ger' were never resolvable
    # (rb87_7 split into rb87_7_mp / rb87_7_pm).
    with pytest.raises(ValueError, match="Unknown level-structure"):
        level_structure(name)


@pytest.mark.parametrize("name", ["1r", "01r"])
def test_symbolic_presets_reject_physical_kwargs(name):
    # 1r/01r accept only ryd_level (S07); an unknown kwarg is rejected.
    with pytest.raises(TypeError, match="does not accept physical parameter") as exc:
        level_structure(name, Delta_Hz=5.0e9)
    message = str(exc.value)
    assert "Delta_Hz" in message
    assert "Allowed parameters: ryd_level" in message


def test_effective_preset_rejects_magnetic_field_kwarg():
    # 1r/01r only accept ryd_level; magnetic_field_G belongs to the rb87 presets.
    with pytest.raises(TypeError, match="does not accept physical parameter"):
        level_structure("1r", magnetic_field_G=20.0)


# ── rb87 seven-level manifolds (ARC-backed; slow) ────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("name", ["rb87_7_mp", "rb87_7_pm"])
def test_rb87_7_public_fields(name):
    ls = level_structure(name)
    assert ls.name == name
    assert ls.levels == ("0", "1", "e1", "e2", "e3", "r", "r_garb")
    assert ls.magnetic_field_G == 20.0  # default
    assert ls.quantization_axis is None
    # decay / branching characterization is exposed for the decaying levels.
    assert set(ls.decay_rates_per_s) == {"r", "r_garb", "e1", "e2", "e3"}
    assert set(ls.branching_ratios) == {"r", "e1", "e2", "e3"}
    assert ls.decay_rates_per_s["r"]["total"] > 0.0


@pytest.mark.slow
def test_rb87_7_ryd_level_and_field_kwargs():
    ls = level_structure("rb87_7_mp", ryd_level=80, magnetic_field_G=30.0)
    assert ls.ryd_level == 80
    assert ls.magnetic_field_G == 30.0


@pytest.mark.slow
def test_rb87_7_rejects_unknown_physical_kwargs():
    # param_set (old manifold selector) and enable_polarization_leakage are gone.
    with pytest.raises(TypeError, match="param_set"):
        level_structure("rb87_7_mp", param_set="our")
    with pytest.raises(TypeError, match="enable_polarization_leakage"):
        level_structure("rb87_7_pm", enable_polarization_leakage=True)


# ── 297 nm single-photon four-level preset (ARC-backed; slow) ────────────────


@pytest.mark.slow
def test_rb87_297_clock_public_fields():
    ls = level_structure("rb87_297_clock_4")
    assert ls.name == "rb87_297_clock_4"
    assert ls.levels == ("0", "1", "r", "r_garb")
    assert ls.ryd_level == 53  # default
    assert ls.magnetic_field_G == 20.0
    assert ls.quantization_axis == pytest.approx((0.0, 0.0, 1.0))
    assert set(ls.decay_rates_per_s) == {"r", "r_garb"}
    assert dict(ls.branching_ratios) == {}


@pytest.mark.slow
def test_rb87_297_normalizes_quantization_axis():
    ls = level_structure("rb87_297_clock_4", quantization_axis=(0.0, 0.0, 2.0))
    assert np.allclose(ls.quantization_axis, (0.0, 0.0, 1.0))
