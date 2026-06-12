"""Declarative noise configuration for the exact Monte Carlo machinery.

``NoiseModel`` is the serializable, validated data layer that names what
noise is requested. It does no sampling itself: quasi-static noise is applied
by :class:`~ryd_gate.backends.exact.monte_carlo_runner.MonteCarloRunner`
through :func:`configure_monte_carlo_runner`, and non-Hermitian decay enters
at system construction time through :meth:`NoiseModel.physical_kwargs`.

Fields without runtime machinery in Stage 4 (``state_prep_error``,
``p_false_pos``, ``p_false_neg``, ``temperature_uK``, ``laser_waist_um``) are
accepted as serializable data but refused with typed validation errors when
applied to a runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ryd_gate.core.serialization import (
    ValidationIssue,
    check_schema,
    json_ready,
    raise_for_errors,
    schema_tag,
)

if TYPE_CHECKING:
    from ryd_gate.backends.exact.monte_carlo_runner import MonteCarloRunner

__all__ = ["NoiseModel", "configure_monte_carlo_runner"]

# Level structures whose local blocks carry the non-Hermitian decay terms.
_DECAY_CAPABLE = frozenset({"analog_3", "rb87_7"})

# Fields accepted as data but without a Stage 4 runtime path.
_RUNTIME_NOT_STAGE4 = ("state_prep", "readout", "temperature", "laser_waist")


@dataclass(frozen=True)
class NoiseModel:
    """Declarative noise request (data only; engines keep doing the work)."""

    runs: int = 1
    detuning_sigma_rad_per_us: float = 0.0
    amp_sigma: float = 0.0
    local_rin_sigma: float = 0.0
    position_sigma_um: float | tuple[float, float, float] = 0.0
    rydberg_decay: bool = False
    intermediate_decay: bool = False
    state_prep_error: float = 0.0
    p_false_pos: float = 0.0
    p_false_neg: float = 0.0
    temperature_uK: float | None = None
    laser_waist_um: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.position_sigma_um, (list, tuple)):
            object.__setattr__(self, "position_sigma_um", tuple(self.position_sigma_um))

    @property
    def n_trajectories(self) -> int:
        """Read-only alias for ``runs`` (Pulser terminology)."""
        return self.runs

    @property
    def noise_types(self) -> tuple[str, ...]:
        """Active noise type names, in a fixed stable order."""
        active = []
        if self.state_prep_error:
            active.append("state_prep")
        if self.p_false_pos or self.p_false_neg:
            active.append("readout")
        if self.detuning_sigma_rad_per_us:
            active.append("detuning")
        if self.amp_sigma:
            active.append("amplitude")
        if self.local_rin_sigma:
            active.append("local_rin")
        if self._position_active():
            active.append("position")
        if self.rydberg_decay:
            active.append("rydberg_decay")
        if self.intermediate_decay:
            active.append("intermediate_decay")
        if self.temperature_uK:
            active.append("temperature")
        if self.laser_waist_um:
            active.append("laser_waist")
        return tuple(active)

    def _position_active(self) -> bool:
        if isinstance(self.position_sigma_um, tuple):
            return any(bool(v) for v in self.position_sigma_um)
        return bool(self.position_sigma_um)

    def summary(self) -> str:
        """Deterministic plain-text description of the active noise types."""
        details = {
            "state_prep": f"state_prep: error={self.state_prep_error}",
            "readout": f"readout: p_false_pos={self.p_false_pos}, p_false_neg={self.p_false_neg}",
            "detuning": f"detuning: sigma={self.detuning_sigma_rad_per_us} rad/us",
            "amplitude": f"amplitude: sigma={self.amp_sigma} (fractional)",
            "local_rin": f"local_rin: sigma={self.local_rin_sigma} (fractional)",
            "position": f"position: sigma={self.position_sigma_um} um",
            "rydberg_decay": "rydberg_decay: enabled",
            "intermediate_decay": "intermediate_decay: enabled",
            "temperature": f"temperature: {self.temperature_uK} uK",
            "laser_waist": f"laser_waist: {self.laser_waist_um} um",
        }
        lines = [f"NoiseModel(runs={self.runs})"]
        for name in self.noise_types:
            lines.append("  " + details[name])
        if len(lines) == 1:
            lines.append("  no active noise")
        return "\n".join(lines)

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[ValidationIssue]:
        """Pure data validation; never raises."""
        issues: list[ValidationIssue] = []
        if not isinstance(self.runs, int) or isinstance(self.runs, bool) or self.runs < 1:
            issues.append(ValidationIssue(
                "error", "noise.runs",
                f"runs must be a positive integer, got {self.runs!r}.",
                ("noise", "runs"),
            ))
        for name in ("state_prep_error", "p_false_pos", "p_false_neg"):
            value = getattr(self, name)
            if not _is_number(value) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                issues.append(ValidationIssue(
                    "error", "noise.probability_range",
                    f"{name} must be a probability in [0, 1], got {value!r}.",
                    ("noise", name),
                ))
        for name in ("detuning_sigma_rad_per_us", "amp_sigma", "local_rin_sigma"):
            value = getattr(self, name)
            if not _is_number(value) or not np.isfinite(value) or value < 0:
                issues.append(ValidationIssue(
                    "error", "noise.nonnegative",
                    f"{name} must be finite and nonnegative, got {value!r}.",
                    ("noise", name),
                ))
        issues += self._validate_position_sigma()
        for name in ("temperature_uK", "laser_waist_um"):
            value = getattr(self, name)
            if value is None:
                continue
            if not _is_number(value) or not np.isfinite(value) or value < 0:
                issues.append(ValidationIssue(
                    "error", "noise.nonnegative",
                    f"{name} must be finite and nonnegative when set, got {value!r}.",
                    ("noise", name),
                ))
        try:
            json_ready(dict(self.metadata), "noise.metadata")
        except ValueError as exc:
            issues.append(ValidationIssue(
                "error", "noise.metadata_json", str(exc), ("noise", "metadata"),
            ))
        return issues

    def _validate_position_sigma(self) -> list[ValidationIssue]:
        value = self.position_sigma_um
        entries: tuple[float, ...]
        if isinstance(value, tuple):
            if len(value) != 3:
                return [ValidationIssue(
                    "error", "noise.position_sigma_shape",
                    f"position_sigma_um must be a scalar or a length-3 tuple, got length {len(value)}.",
                    ("noise", "position_sigma_um"),
                )]
            entries = value
        elif _is_number(value):
            entries = (value,)
        else:
            return [ValidationIssue(
                "error", "noise.position_sigma_shape",
                f"position_sigma_um must be a scalar or a length-3 tuple, got {value!r}.",
                ("noise", "position_sigma_um"),
            )]
        for entry in entries:
            if not _is_number(entry) or not np.isfinite(entry) or entry < 0:
                return [ValidationIssue(
                    "error", "noise.nonnegative",
                    f"position_sigma_um entries must be finite and nonnegative, got {entry!r}.",
                    ("noise", "position_sigma_um"),
                )]
        return []

    def validate_for(
        self,
        *,
        backend: str,
        level_structure=None,
        n_atoms: int | None = None,
    ) -> list[ValidationIssue]:
        """Runtime capability checks for applying this model on *backend*."""
        issues: list[ValidationIssue] = []
        active = self.noise_types
        if active and backend != "exact":
            issues.append(ValidationIssue(
                "error", "noise.backend_unsupported",
                f"noise types {active} have no runtime support on backend {backend!r}.",
                ("noise",),
            ))
        if (self.rydberg_decay or self.intermediate_decay) and level_structure is not None:
            name = level_structure if isinstance(level_structure, str) else level_structure.name
            if name not in _DECAY_CAPABLE:
                issues.append(ValidationIssue(
                    "error", "noise.decay_level_structure_unsupported",
                    f"decay needs physical local blocks ({sorted(_DECAY_CAPABLE)}); "
                    f"level structure is {name!r}.",
                    ("noise", "rydberg_decay"),
                ))
        if self._position_active() and n_atoms is not None and n_atoms != 2:
            issues.append(ValidationIssue(
                "error", "noise.position_two_atom_only",
                f"position noise uses the two-atom VdW perturbation path; system has {n_atoms} atoms.",
                ("noise", "position_sigma_um"),
            ))
        runtime_requested = [name for name in active if name in _RUNTIME_NOT_STAGE4]
        if runtime_requested:
            issues.append(ValidationIssue(
                "error", "noise.runtime_not_stage4",
                f"noise types {tuple(runtime_requested)} are serializable data only in Stage 4 "
                "and cannot be applied to a runtime path.",
                ("noise",),
            ))
        return issues

    # ── Runtime hand-off ─────────────────────────────────────────────────

    def physical_kwargs(self) -> dict[str, bool]:
        """Decay flags for ``RydbergSystem.from_lattice`` (construction time)."""
        return {
            "enable_rydberg_decay": self.rydberg_decay,
            "enable_intermediate_decay": self.intermediate_decay,
        }

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("noise"),
            "runs": self.runs,
            "detuning_sigma_rad_per_us": self.detuning_sigma_rad_per_us,
            "amp_sigma": self.amp_sigma,
            "local_rin_sigma": self.local_rin_sigma,
            "position_sigma_um": (
                list(self.position_sigma_um)
                if isinstance(self.position_sigma_um, tuple)
                else self.position_sigma_um
            ),
            "rydberg_decay": self.rydberg_decay,
            "intermediate_decay": self.intermediate_decay,
            "state_prep_error": self.state_prep_error,
            "p_false_pos": self.p_false_pos,
            "p_false_neg": self.p_false_neg,
            "temperature_uK": self.temperature_uK,
            "laser_waist_um": self.laser_waist_um,
            "metadata": json_ready(dict(self.metadata), "noise.metadata"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NoiseModel":
        check_schema(data, "noise")
        runs = data.get("runs", data.get("n_trajectories", 1))
        position = data.get("position_sigma_um", 0.0)
        return cls(
            runs=runs,
            detuning_sigma_rad_per_us=data.get("detuning_sigma_rad_per_us", 0.0),
            amp_sigma=data.get("amp_sigma", 0.0),
            local_rin_sigma=data.get("local_rin_sigma", 0.0),
            position_sigma_um=position,
            rydberg_decay=data.get("rydberg_decay", False),
            intermediate_decay=data.get("intermediate_decay", False),
            state_prep_error=data.get("state_prep_error", 0.0),
            p_false_pos=data.get("p_false_pos", 0.0),
            p_false_neg=data.get("p_false_neg", 0.0),
            temperature_uK=data.get("temperature_uK"),
            laser_waist_um=data.get("laser_waist_um"),
            metadata=dict(data.get("metadata", {})),
        )


def configure_monte_carlo_runner(runner: "MonteCarloRunner", noise: NoiseModel) -> "MonteCarloRunner":
    """Apply *noise* to an existing exact ``MonteCarloRunner`` and return it.

    Maps fields onto the runner's own ``setup_*`` methods with exact unit
    conversions (rad/us → Hz for detuning, um → m for position sigmas);
    zero-valued fields call nothing. Decay flags are construction-time
    physics (``NoiseModel.physical_kwargs()`` into
    ``RydbergSystem.from_lattice``) and are not applied here.
    """
    raise_for_errors(noise.validate())
    system = runner.system
    level_spec = system.meta("level_spec", None) if hasattr(system, "meta") else None
    n_atoms = system.meta("n_sites", None) if hasattr(system, "meta") else None
    raise_for_errors(noise.validate_for(
        backend="exact", level_structure=level_spec, n_atoms=n_atoms,
    ))

    if noise.detuning_sigma_rad_per_us:
        runner.setup_detuning_noise(noise.detuning_sigma_rad_per_us * 1e6 / (2 * np.pi))
    if noise.amp_sigma:
        runner.setup_amplitude_noise(noise.amp_sigma)
    if noise.local_rin_sigma:
        runner.setup_local_rin_noise(noise.local_rin_sigma)
    if noise._position_active():
        sigma = noise.position_sigma_um
        sigma3 = sigma if isinstance(sigma, tuple) else (sigma, sigma, sigma)
        sx, sy, sz = (float(s) * 1e-6 for s in sigma3)
        runner.setup_position_noise((sx, sy, sz))
    return runner


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
