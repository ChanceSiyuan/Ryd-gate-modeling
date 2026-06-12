"""Frozen v1 JSON Schemas validate every public to_dict() payload."""

import sys

import pytest

from ryd_gate import (
    NoiseModel,
    RegisterLayout,
    level_structure,
)
from ryd_gate.core.serialization import (
    load_json_schema,
    schema_path,
    validate_json_schema,
)
from ryd_gate.gates import CZGateReport


@pytest.fixture
def needs_jsonschema():
    pytest.importorskip("jsonschema")


def _layout():
    return RegisterLayout(
        name="demo", kind="custom",
        trap_coords_um=((0.0, 0.0), (0.0, 6.0), (6.0, 0.0), (6.0, 6.0)),
    )


def _report():
    return CZGateReport(
        protocol="TOProtocol",
        parameters=(0.1, 0.2),
        infidelity=1e-4,
        phase_error_rad=0.01,
        theta_rad=1.4,
        residuals={"ryd": 1e-5},
        error_budget={"rydberg_decay": {"total": 1e-4, "XYZ": 5e-5, "AL": 3e-5, "LG": 2e-5}},
        metadata={"note": "fixture"},
    )


PAYLOADS = {
    "register": lambda: _layout().define_register([0, 3], ["a", "b"]).to_dict(),
    "register-layout": lambda: _layout().to_dict(),
    "level-structure": lambda: level_structure("01r").to_dict(),
    "noise": lambda: NoiseModel(runs=8, detuning_sigma_rad_per_us=0.02,
                                position_sigma_um=(0.07, 0.07, 0.13)).to_dict(),
    "cz-gate-report": lambda: _report().to_dict(),
}


@pytest.mark.usefixtures("needs_jsonschema")
class TestSchemasValidatePayloads:
    @pytest.mark.parametrize("kind", sorted(PAYLOADS))
    def test_payload_validates(self, kind):
        issues = validate_json_schema(PAYLOADS[kind](), kind)
        assert issues == [], [issue.message for issue in issues]

    @pytest.mark.parametrize("kind", sorted(PAYLOADS))
    def test_missing_required_fields_fail(self, kind):
        schema = load_json_schema(kind)
        payload = PAYLOADS[kind]()
        for field in schema["required"]:
            broken = dict(payload)
            del broken[field]
            issues = validate_json_schema(broken, kind)
            assert issues, f"removing {field!r} from {kind} payload should fail"

    @pytest.mark.parametrize("kind", sorted(PAYLOADS))
    def test_wrong_schema_tag_fails(self, kind):
        payload = PAYLOADS[kind]()
        payload["schema"] = "ryd-gate/other/v1"
        assert validate_json_schema(payload, kind)


class TestSchemaLoading:
    def test_schemas_load_via_importlib_resources(self):
        for kind in PAYLOADS:
            schema = load_json_schema(kind)
            assert schema["title"] == f"ryd-gate/{kind}/v1"
            assert "required" in schema

    def test_schema_path_points_to_existing_file(self):
        path = schema_path("register")
        assert path.name == "register.v1.schema.json"
        assert path.exists()

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="no frozen schema"):
            load_json_schema("nonexistent")


class TestOptionalDependency:
    def test_missing_jsonschema_reports_typed_issue(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "jsonschema", None)
        issues = validate_json_schema({"schema": "ryd-gate/noise/v1"}, "noise")
        assert [issue.code for issue in issues] == ["serialization.jsonschema_missing"]
