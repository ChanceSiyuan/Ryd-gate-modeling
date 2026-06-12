"""Observable schedules for streaming measurement during TN evolution.

``ObservableConfig`` is validated, serializable data: which named observables
to record and when. The recording itself is the existing TeNPy TDVP
measurement path (Stage 3's ``measure_mps_observable``);
``simulate_sequence(..., backend="mps", observables=config)`` lowers the
schedule onto that path. The exact backend keeps its final-state handle
semantics and ignores the streaming schedule (documented Stage 6 rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ryd_gate.core.serialization import ValidationIssue, check_schema, schema_tag

__all__ = ["ObservableConfig"]


@dataclass(frozen=True)
class ObservableConfig:
    """Names + schedule for backend-native observable recording.

    Exactly one of ``times_ns`` (explicit nonnegative integer times) or
    ``every_ns`` (uniform stride) may be set; with neither, only the final
    state is recorded. Names are interpreted by the backend's existing
    observable resolver.
    """

    names: tuple[str, ...]
    times_ns: tuple[int, ...] | None = None
    every_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        if self.times_ns is not None:
            object.__setattr__(self, "times_ns", tuple(self.times_ns))

    def validate(self) -> list[ValidationIssue]:
        """Data validation; never raises."""
        issues: list[ValidationIssue] = []
        if not self.names or not all(
            isinstance(name, str) and name for name in self.names
        ):
            issues.append(ValidationIssue(
                "error", "observables.names",
                f"names must be a non-empty tuple of non-empty strings, got {self.names!r}.",
                ("observables", "names"),
            ))
        if self.times_ns is not None and self.every_ns is not None:
            issues.append(ValidationIssue(
                "error", "observables.schedule_conflict",
                "set either times_ns or every_ns, not both.",
                ("observables",),
            ))
        if self.times_ns is not None:
            for t in self.times_ns:
                if not isinstance(t, int) or isinstance(t, bool) or t < 0:
                    issues.append(ValidationIssue(
                        "error", "observables.times",
                        f"times_ns entries must be nonnegative integers, got {t!r}.",
                        ("observables", "times_ns"),
                    ))
                    break
        if self.every_ns is not None and (
            not isinstance(self.every_ns, int)
            or isinstance(self.every_ns, bool)
            or self.every_ns <= 0
        ):
            issues.append(ValidationIssue(
                "error", "observables.every",
                f"every_ns must be a positive integer, got {self.every_ns!r}.",
                ("observables", "every_ns"),
            ))
        return issues

    def schedule_times_ns(self, duration_ns: int) -> tuple[int, ...] | None:
        """Resolve the schedule against a sequence duration; ``None`` = final only."""
        if self.times_ns is not None:
            return self.times_ns
        if self.every_ns is not None:
            return tuple(range(0, int(duration_ns) + 1, self.every_ns))
        return None

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("observable-config"),
            "names": list(self.names),
            "times_ns": list(self.times_ns) if self.times_ns is not None else None,
            "every_ns": self.every_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservableConfig":
        check_schema(data, "observable-config")
        times = data.get("times_ns")
        return cls(
            names=tuple(data["names"]),
            times_ns=tuple(times) if times is not None else None,
            every_ns=data.get("every_ns"),
        )
