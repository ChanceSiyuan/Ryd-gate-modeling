"""Round-trip tests for the schema-tagged plain-dict serialization contract."""

import dataclasses
import json

import pytest

from ryd_gate import (
    LevelStructureSpec,
    Register,
    RegisterLayout,
    level_structure,
)


def _layout():
    return RegisterLayout(
        name="square_2x2",
        trap_coords_um=((0.0, 0.0), (0.0, 5.0), (5.0, 0.0), (5.0, 5.0)),
        kind="square",
    )


def _eq_objects():
    """Objects with scalar/tuple fields: round-trip must compare equal with ==."""
    return [
        ("register-layout", _layout(), RegisterLayout),
        ("level-structure", level_structure("analog_3"), LevelStructureSpec),
        (
            "level-structure",
            LevelStructureSpec(
                name="custom_gr", levels=("g", "r"), rydberg_levels=("r",),
                initial_level="g", params={"note": "test"},
            ),
            LevelStructureSpec,
        ),
    ]


class TestEqualityRoundTrips:
    @pytest.mark.parametrize(
        "kind,obj,cls", _eq_objects(), ids=lambda v: getattr(v, "__name__", str(v))[:40]
    )
    def test_json_and_roundtrip(self, kind, obj, cls):
        data = obj.to_dict()
        assert data["schema"] == f"ryd-gate/{kind}/v1"
        json.dumps(data)  # must be JSON-compatible
        assert cls.from_dict(data) == obj


class TestRegisterRoundTrip:
    @pytest.mark.parametrize("with_layout", [False, True])
    def test_fieldwise_roundtrip(self, with_layout):
        reg = Register.square(2, 5.0, prefix="a")
        if with_layout:
            reg = dataclasses.replace(reg, layout=_layout())
        data = reg.to_dict()
        assert data["schema"] == "ryd-gate/register/v1"
        json.dumps(data)
        back = Register.from_dict(data)
        assert back.ids == reg.ids
        assert back.coords_um == reg.coords_um
        assert back.sublattice.tolist() == reg.sublattice.tolist()
        assert back.spacing_um == reg.spacing_um
        assert back.layout == reg.layout
        assert dict(back.metadata) == dict(reg.metadata)


class TestFailureModes:
    def test_wrong_schema_tag_raises(self):
        data = _layout().to_dict()
        data["schema"] = "ryd-gate/register/v1"
        with pytest.raises(ValueError, match="schema"):
            RegisterLayout.from_dict(data)

    def test_invalid_payload_raises(self):
        data = Register.chain(2, 4.0).to_dict()
        data["ids"] = ["a", "a"]  # duplicate ids smuggled into the payload
        with pytest.raises(ValueError, match="unique"):
            Register.from_dict(data)

    def test_non_json_metadata_raises(self):
        reg = Register.chain(2, 4.0)
        bad = dataclasses.replace(reg, metadata={"obj": object()})
        with pytest.raises(ValueError, match="JSON"):
            bad.to_dict()
