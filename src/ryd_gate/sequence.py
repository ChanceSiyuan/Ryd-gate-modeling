"""Sequence: the user-level schedule of pulses on declared device channels.

A ``Sequence`` binds a :class:`Register`, a :class:`DeviceSpec`, and a level
structure, then records an append-only schedule of validated operations.
Backends never see it: it compiles to a kernel ``Protocol``
(:class:`~ryd_gate.protocols.sequence_protocol.SequenceProtocol`) and runs
through the existing ``RydbergSystem`` → ``HamiltonianIR`` → backend path.

Stage 2 scope: global Rydberg channels only; local addressing, hyperfine
driving, and phase-modulated pulses raise typed errors (gate-style phase
work stays on the gate-protocol path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ryd_gate.core.level_structures import LevelStructureSpec
from ryd_gate.core.level_structures import level_structure as level_structure_preset
from ryd_gate.core.serialization import (
    ValidationIssue,
    check_schema,
    raise_for_errors,
    schema_tag,
)
from ryd_gate.devices import DeviceSpec
from ryd_gate.lattice import Register
from ryd_gate.protocols.channels import ChannelSpec
from ryd_gate.pulse import Pulse

__all__ = ["DelayOp", "MeasureOp", "PulseOp", "Sequence", "TargetOp"]

_MEASURE_BASES = ("full-level", "rydberg", "computational")


@dataclass(frozen=True)
class PulseOp:
    """One scheduled pulse on a declared channel.

    ``targets`` is ``None`` on global channels; on local channels it is the
    atom-id tuple the channel was targeting when the pulse was added.
    """

    channel: str
    pulse: Pulse
    t_start_ns: int
    targets: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TargetOp:
    """Retarget a local channel onto a set of atom ids (Stage 8)."""

    channel: str
    targets: tuple[str, ...]
    t_start_ns: int


@dataclass(frozen=True)
class DelayOp:
    """A channel-local idle window."""

    channel: str
    duration_ns: int
    t_start_ns: int


@dataclass(frozen=True)
class MeasureOp:
    """The terminal measurement marker (basis consumed by result sampling)."""

    basis: Literal["full-level", "rydberg", "computational"]
    t_start_ns: int


class Sequence:
    """Append-only, device-validated pulse schedule (a builder, not frozen).

    Construction fails fast: the register and level structure are validated
    against the device immediately, so a constructed ``Sequence`` is always
    device-compatible. Every ``add`` validates the pulse against its channel.
    """

    def __init__(
        self,
        register: Register,
        device: DeviceSpec,
        level_structure: LevelStructureSpec | str | None = None,
    ) -> None:
        if level_structure is None:
            spec = level_structure_preset(device.default_level_structure)
        elif isinstance(level_structure, str):
            spec = level_structure_preset(level_structure)
        elif isinstance(level_structure, LevelStructureSpec):
            spec = level_structure
        else:
            raise TypeError(
                f"level_structure must be a LevelStructureSpec, a preset name, or None; "
                f"got {type(level_structure).__name__}."
            )
        raise_for_errors(device.validate_register(register))
        raise_for_errors(device.validate_level_structure(spec))
        self._register = register
        self._device = device
        self._level_structure = spec
        self._declared: dict[str, ChannelSpec] = {}
        self._compiler_names: dict[str, str] = {}  # compiler channel -> declared alias
        self._channel_end_ns: dict[str, int] = {}
        self._channel_targets: dict[str, tuple[str, ...] | None] = {}
        self._ops: list[PulseOp | DelayOp | MeasureOp | TargetOp] = []
        self._measured = False

    # ── Read-only views ─────────────────────────────────────────────────

    @property
    def register(self) -> Register:
        return self._register

    @property
    def device(self) -> DeviceSpec:
        return self._device

    @property
    def level_structure(self) -> LevelStructureSpec:
        return self._level_structure

    @property
    def declared_channels(self) -> Mapping[str, ChannelSpec]:
        return dict(self._declared)

    @property
    def operations(self) -> tuple[PulseOp | DelayOp | MeasureOp | TargetOp, ...]:
        return tuple(self._ops)

    @property
    def duration_ns(self) -> int:
        """Maximum end time across all declared channels (0 for an empty sequence)."""
        return max(self._channel_end_ns.values(), default=0)

    def is_measured(self) -> bool:
        return self._measured

    # ── Schedule construction ───────────────────────────────────────────

    def declare_channel(self, name: str, channel_id: str) -> None:
        """Bind a user alias to a device channel (fail-earliest Stage 2 gates)."""
        self._require_not_measured()
        if name in self._declared:
            raise ValueError(f"channel name {name!r} is already declared.")
        channel = self._device.channels.get(channel_id)
        if channel is None:
            raise ValueError(
                f"unknown channel id {channel_id!r} on device {self._device.name} "
                f"(available: {', '.join(self._device.channels) or 'none'})."
            )
        if channel.kind != "rydberg":
            raise NotImplementedError(
                "sequence.hyperfine_not_stage2: only Rydberg channels compile through "
                "the sequence path in Stage 2."
            )
        model = self._level_structure.name
        amp_channel = channel.amplitude_channels.get(model)
        det_channel = channel.detuning_channels.get(model)
        if amp_channel is None or det_channel is None:
            raise ValueError(
                f"sequence.channel_model_mismatch: channel {channel_id!r} has no compiler "
                f"mapping for level structure {model!r}."
            )
        for compiler_name in (amp_channel, det_channel):
            if compiler_name in self._compiler_names:
                raise ValueError(
                    f"sequence.compiler_channel_collision: compiler channel "
                    f"{compiler_name!r} is already driven by declared channel "
                    f"{self._compiler_names[compiler_name]!r}."
                )
        self._compiler_names[amp_channel] = name
        self._compiler_names[det_channel] = name
        self._declared[name] = channel
        self._channel_end_ns[name] = 0
        self._channel_targets[name] = None

    def target(self, atom_ids, channel: str) -> None:
        """Point a *local* channel at a set of atom ids (Stage 8).

        Subsequent pulses on the channel drive exactly these atoms. Consumes
        the channel's ``retarget_time_ns`` (when set) on the channel clock.
        """
        self._require_not_measured()
        channel_spec = self._require_declared(channel)
        if channel_spec.addressing != "local":
            raise ValueError(
                f"sequence.target_global: channel {channel!r} is global; "
                "target() applies to local channels only."
            )
        if isinstance(atom_ids, str):
            atom_ids = (atom_ids,)
        targets = tuple(str(atom_id) for atom_id in atom_ids)
        if not targets:
            raise ValueError("sequence.target_empty: target() needs at least one atom id.")
        if len(set(targets)) != len(targets):
            raise ValueError(f"sequence.target_duplicate: duplicate atom ids in {targets}.")
        assert self._register.ids is not None
        unknown = [atom_id for atom_id in targets if atom_id not in self._register.ids]
        if unknown:
            raise ValueError(
                f"sequence.target_unknown_atom: {unknown} not in register ids "
                f"{self._register.ids}."
            )
        if channel_spec.max_targets is not None and len(targets) > channel_spec.max_targets:
            raise ValueError(
                f"sequence.max_targets: {len(targets)} targets exceed channel limit "
                f"{channel_spec.max_targets}."
            )
        t_start = self._channel_end_ns[channel]
        self._ops.append(TargetOp(channel=channel, targets=targets, t_start_ns=t_start))
        if channel_spec.retarget_time_ns:
            self._channel_end_ns[channel] = t_start + channel_spec.retarget_time_ns
        self._channel_targets[channel] = targets

    def add(self, pulse: Pulse, channel: str) -> None:
        """Append *pulse* at the channel's current end time (hardware limits enforced)."""
        self._require_not_measured()
        channel_spec = self._require_declared(channel)
        raise_for_errors(pulse.validate(channel_spec))
        targets = None
        if channel_spec.addressing == "local":
            targets = self._channel_targets[channel]
            if targets is None:
                raise ValueError(
                    f"sequence.local_targets_missing: call target(...) on local channel "
                    f"{channel!r} before adding pulses."
                )
        t_start = self._channel_end_ns[channel]
        self._ops.append(
            PulseOp(channel=channel, pulse=pulse, t_start_ns=t_start, targets=targets)
        )
        self._channel_end_ns[channel] = t_start + pulse.duration_ns

    def delay(self, duration_ns: int, channel: str) -> None:
        """Append a channel-local idle window."""
        self._require_not_measured()
        self._require_declared(channel)
        if not isinstance(duration_ns, int) or isinstance(duration_ns, bool) or duration_ns <= 0:
            raise ValueError(f"delay duration_ns must be a positive integer, got {duration_ns!r}.")
        t_start = self._channel_end_ns[channel]
        self._ops.append(DelayOp(channel=channel, duration_ns=duration_ns, t_start_ns=t_start))
        self._channel_end_ns[channel] = t_start + duration_ns

    def measure(self, basis: Literal["full-level", "rydberg", "computational"] = "rydberg") -> None:
        """Append the terminal measurement marker; the sequence is locked afterwards."""
        self._require_not_measured()
        if basis not in _MEASURE_BASES:
            raise ValueError(f"basis must be one of {_MEASURE_BASES}, got {basis!r}.")
        self._ops.append(MeasureOp(basis=basis, t_start_ns=self.duration_ns))
        self._measured = True

    def _require_not_measured(self) -> None:
        if self._measured:
            raise ValueError("sequence.measured: the sequence is locked after measure().")

    def _require_declared(self, channel: str) -> ChannelSpec:
        spec = self._declared.get(channel)
        if spec is None:
            raise ValueError(
                f"channel {channel!r} is not declared "
                f"(declared: {', '.join(self._declared) or 'none'})."
            )
        return spec

    # ── Validation / serialization / drawing ───────────────────────────

    def validate(self) -> list[ValidationIssue]:
        """Re-run all device/pulse checks (for deserialized sequences); never raises."""
        issues = list(self._device.validate_register(self._register))
        issues += self._device.validate_level_structure(self._level_structure)
        for op in self._ops:
            if isinstance(op, PulseOp):
                issues += op.pulse.validate(self._declared[op.channel])
        return issues

    def to_dict(self) -> dict:
        return {
            "schema": schema_tag("sequence"),
            "register": self._register.to_dict(),
            "device": self._device.to_dict(),
            "level_structure": self._level_structure.to_dict(),
            "channels": {name: ch.channel_id for name, ch in self._declared.items()},
            "operations": [self._op_dict(op) for op in self._ops],
        }

    @staticmethod
    def _op_dict(op: PulseOp | DelayOp | MeasureOp | TargetOp) -> dict:
        if isinstance(op, PulseOp):
            return {
                "type": "pulse",
                "channel": op.channel,
                "pulse": op.pulse.to_dict(),
                "t_start_ns": op.t_start_ns,
            }
        if isinstance(op, DelayOp):
            return {
                "type": "delay",
                "channel": op.channel,
                "duration_ns": op.duration_ns,
                "t_start_ns": op.t_start_ns,
            }
        if isinstance(op, TargetOp):
            return {
                "type": "target",
                "channel": op.channel,
                "targets": list(op.targets),
                "t_start_ns": op.t_start_ns,
            }
        return {"type": "measure", "basis": op.basis, "t_start_ns": op.t_start_ns}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Sequence":
        """Rebuild by replaying declarations and operations through the public
        methods — one construction path, so all validation re-runs on load."""
        check_schema(data, "sequence")
        sequence = cls(
            Register.from_dict(data["register"]),
            DeviceSpec.from_dict(data["device"]),
            LevelStructureSpec.from_dict(data["level_structure"]),
        )
        for name, channel_id in data["channels"].items():
            sequence.declare_channel(name, channel_id)
        for op in data["operations"]:
            kind = op.get("type")
            if kind == "pulse":
                sequence.add(Pulse.from_dict(op["pulse"]), op["channel"])
            elif kind == "delay":
                sequence.delay(op["duration_ns"], op["channel"])
            elif kind == "target":
                sequence.target(op["targets"], op["channel"])
            elif kind == "measure":
                sequence.measure(op["basis"])
            else:
                raise ValueError(f"unknown operation type {kind!r}.")
        return sequence

    def draw(self, *, show: bool = True, **plot_kwargs):
        """Render the schedule via the kernel's generic ``Protocol.plot()``.

        No plotting code lives here: ``SequenceProtocol.pulse_traces`` feeds
        the existing protocol plotter. Returns ``(fig, ax)``.
        """
        from ryd_gate.protocols.sequence_protocol import SequenceProtocol

        protocol = SequenceProtocol(self)
        params = protocol.unpack_params([], None)
        defaults: dict[str, Any] = {
            "time_scale": 1e9,
            "time_label": "time (ns)",
            "unit_label": "rad/us",
        }
        defaults.update(plot_kwargs)
        return protocol.plot(params=params, show=show, **defaults)
