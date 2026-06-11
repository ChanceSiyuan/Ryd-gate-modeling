"""Tests for the shared validation primitive (core/validation.py)."""

import pytest

from ryd_gate.core.validation import ValidationIssue, raise_for_errors


class TestValidationIssue:
    def test_valid_warning_constructs(self):
        issue = ValidationIssue("warning", "x.code", "message", ("a", "b"))
        assert issue.severity == "warning"
        assert issue.code == "x.code"
        assert issue.path == ("a", "b")

    def test_valid_error_constructs(self):
        issue = ValidationIssue("error", "x.code", "message")
        assert issue.severity == "error"
        assert issue.path == ()

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            ValidationIssue("fatal", "x.code", "message")

    def test_empty_code_raises(self):
        with pytest.raises(ValueError, match="code"):
            ValidationIssue("error", "", "message")

    def test_non_string_message_raises(self):
        with pytest.raises(ValueError, match="message"):
            ValidationIssue("error", "x.code", 123)

    def test_non_tuple_path_raises(self):
        with pytest.raises(ValueError, match="path"):
            ValidationIssue("error", "x.code", "message", ["a"])


class TestRaiseForErrors:
    def test_empty_list_does_not_raise(self):
        assert raise_for_errors([]) is None

    def test_warning_only_does_not_raise(self):
        issues = [ValidationIssue("warning", "w.one", "first warning")]
        assert raise_for_errors(issues) is None

    def test_one_error_raises_valueerror(self):
        issues = [ValidationIssue("error", "e.one", "boom")]
        with pytest.raises(ValueError, match="e.one"):
            raise_for_errors(issues)

    def test_multiple_errors_on_separate_lines(self):
        issues = [
            ValidationIssue("error", "e.one", "first"),
            ValidationIssue("warning", "w.skip", "not included"),
            ValidationIssue("error", "e.two", "second"),
        ]
        with pytest.raises(ValueError) as excinfo:
            raise_for_errors(issues)
        text = str(excinfo.value)
        lines = text.splitlines()
        assert "e.one: first" in lines
        assert "e.two: second" in lines
        assert not any("w.skip" in line for line in lines)
