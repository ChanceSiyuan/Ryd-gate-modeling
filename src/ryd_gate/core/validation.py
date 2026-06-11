"""Shared validation primitives for product API objects.

One vocabulary for "what is wrong": severity, stable machine-readable code,
human-readable message, and a path locating the offending field. Validation
methods accumulate :class:`ValidationIssue` objects without raising;
:func:`raise_for_errors` is the explicit raise boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ValidationSeverity = Literal["error", "warning"]

_VALID_SEVERITIES = ("error", "warning")


@dataclass(frozen=True)
class ValidationIssue:
    """One validation problem.

    Attributes
    ----------
    severity : {"error", "warning"}
        Errors block execution at ``raise_for_errors``; warnings never raise.
    code : str
        Stable machine-readable code (e.g. ``"register.min_distance"``).
        Codes are API: tests and downstream callers branch on them.
    message : str
        Human-readable description.
    path : tuple[str, ...]
        Location of the invalid field, e.g. ``("register", "coords")``.
    """

    severity: ValidationSeverity
    code: str
    message: str
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be 'error' or 'warning', got {self.severity!r}."
            )
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string.")
        if not isinstance(self.message, str):
            raise ValueError("message must be a string.")
        if not isinstance(self.path, tuple) or not all(
            isinstance(p, str) for p in self.path
        ):
            raise ValueError("path must be a tuple of strings.")


def raise_for_errors(issues: list[ValidationIssue]) -> None:
    """Raise ``ValueError`` if any issue has severity ``"error"``.

    Returns ``None`` when there are no errors (warnings never raise). The
    raised message lists each error's code and message on its own line.
    """
    errors = [issue for issue in issues if issue.severity == "error"]
    if not errors:
        return None
    lines = [f"{issue.code}: {issue.message}" for issue in errors]
    raise ValueError("validation failed:\n" + "\n".join(lines))
