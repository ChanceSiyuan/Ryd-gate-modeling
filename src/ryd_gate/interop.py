"""Pulser abstract-representation bridge (narrow, typed subset).

Imports/exports Pulser's JSON abstract representation **without importing
Pulser**: payloads are plain dicts/JSON strings, and the bridge lowers them
into native Stage 1-2 objects (``Register``/``RegisterLayout``/``Waveform``/
``Pulse``/``Sequence``/``NoiseModel``). The supported subset is deliberately
narrow — registers (with optional layout trap mapping), one global Rydberg
channel, the five native waveform kinds, zero-phase pulses, delays, and the
ground-rydberg measurement. Everything else raises
:class:`PulserInteropError` naming the unsupported construct and its payload
path; nothing is silently dropped.

Stable error codes (plan-listed):
``pulser.eom_not_supported``, ``pulser.dmm_not_supported``,
``pulser.slm_not_supported``, ``pulser.xy_not_supported``,
``pulser.local_addressing_not_supported``, ``pulser.phase_not_supported``,
``pulser.measurement_not_supported``, ``pulser.waveform_not_supported``,
``pulser.channel_not_supported``, ``pulser.noise_field_not_supported``.
Extension codes for malformed/out-of-subset payloads:
``pulser.operation_not_supported``, ``pulser.layout_mismatch``,
``pulser.register_not_supported``.

Units match natively: Pulser durations are integer ns and amplitudes /
detunings are rad/us — the same conventions as :class:`~ryd_gate.pulse.Waveform`.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ryd_gate.devices import DeviceSpec
from ryd_gate.lattice import Register, RegisterLayout
from ryd_gate.noise import NoiseModel
from ryd_gate.pulse import Pulse, Waveform
from ryd_gate.sequence import DelayOp, MeasureOp, PulseOp, Sequence

__all__ = [
    "PulserInteropError",
    "from_pulser_abstract_repr",
    "noise_from_pulser_abstract_repr",
    "noise_to_pulser_abstract_repr",
    "to_pulser_abstract_repr",
]

# Native sequences import onto the two-level ground-rydberg model.
_IMPORT_LEVEL_STRUCTURE = "1r"
_SUPPORTED_CHANNEL_ID = "rydberg_global"
_COORD_MATCH_TOL_UM = 1e-9


class PulserInteropError(ValueError):
    """A Pulser construct outside the supported subset (typed, path-aware)."""

    def __init__(self, code: str, message: str, path: tuple[str, ...] = (), construct: str | None = None) -> None:
        self.code = code
        self.path = tuple(path)
        self.construct = construct
        location = f" at {'/'.join(self.path)}" if self.path else ""
        super().__init__(f"{code}: {message}{location}")


def _parse(data: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, Mapping):
        raise ValueError(f"expected a mapping or JSON string, got {type(data).__name__}.")
    return data


# ── Sequence import ──────────────────────────────────────────────────────────


def from_pulser_abstract_repr(data: Mapping[str, Any] | str) -> Sequence:
    """Lower a Pulser abstract-repr payload into a native :class:`Sequence`."""
    payload = _parse(data)
    _reject_top_level_constructs(payload)

    register = _register_from_ar(payload)
    sequence = Sequence(register, DeviceSpec.virtual_rb87(), _IMPORT_LEVEL_STRUCTURE)

    channels = payload.get("channels", {})
    for alias, channel_id in channels.items():
        _check_channel_supported(channel_id, ("channels", alias))
        sequence.declare_channel(alias, _SUPPORTED_CHANNEL_ID)

    for index, op in enumerate(payload.get("operations", [])):
        _apply_operation(sequence, op, ("operations", str(index)))

    measurement = payload.get("measurement")
    if measurement is not None:
        if measurement != "ground-rydberg":
            raise PulserInteropError(
                "pulser.measurement_not_supported",
                f"measurement basis {measurement!r} has no kernel counterpart",
                ("measurement",), construct=str(measurement),
            )
        sequence.measure("rydberg")
    return sequence


def _reject_top_level_constructs(payload: Mapping[str, Any]) -> None:
    if payload.get("slm_mask_targets"):
        raise PulserInteropError(
            "pulser.slm_not_supported", "SLM masks are not supported",
            ("slm_mask_targets",), construct="slm_mask",
        )
    if payload.get("magnetic_field") and any(float(b) != 0.0 for b in payload["magnetic_field"]):
        raise PulserInteropError(
            "pulser.xy_not_supported", "XY mode (magnetic field) is not supported",
            ("magnetic_field",), construct="magnetic_field",
        )
    for key in payload:
        if key.startswith("dmm"):
            raise PulserInteropError(
                "pulser.dmm_not_supported", "DMM channels are not supported",
                (key,), construct=key,
            )


def _register_from_ar(payload: Mapping[str, Any]) -> Register:
    entries = payload.get("register", [])
    if not entries:
        raise PulserInteropError(
            "pulser.register_not_supported", "payload has no register atoms", ("register",),
        )
    ids = [str(entry["name"]) for entry in entries]
    coords = [
        [float(entry["x"]), float(entry["y"])] + ([float(entry["z"])] if "z" in entry else [])
        for entry in entries
    ]
    dims = {len(c) for c in coords}
    if len(dims) != 1:
        raise PulserInteropError(
            "pulser.register_not_supported", "mixed 2D/3D register coordinates", ("register",),
        )

    layout_data = payload.get("layout")
    if layout_data is None:
        return Register.from_coordinates(coords, ids=ids, center=False)

    layout = RegisterLayout(
        name=str(layout_data.get("slug") or "pulser-layout"),
        trap_coords_um=tuple(tuple(float(x) for x in c) for c in layout_data["coordinates"]),
        kind="custom",
    )
    traps = np.asarray(layout.trap_coords_um, dtype=float)
    trap_ids = []
    for qubit_index, coord in enumerate(coords):
        distances = np.linalg.norm(traps - np.asarray(coord, dtype=float), axis=1)
        trap = int(np.argmin(distances))
        if distances[trap] > _COORD_MATCH_TOL_UM:
            raise PulserInteropError(
                "pulser.layout_mismatch",
                f"register atom {ids[qubit_index]!r} at {coord} matches no layout trap",
                ("register", str(qubit_index)),
            )
        trap_ids.append(trap)
    return layout.define_register(trap_ids, qubit_ids=ids, center=False)


def _check_channel_supported(channel_id: str, path: tuple[str, ...]) -> None:
    if channel_id == _SUPPORTED_CHANNEL_ID:
        return
    if "mw" in channel_id:
        raise PulserInteropError(
            "pulser.xy_not_supported", "XY/microwave channels are not supported",
            path, construct=channel_id,
        )
    if "local" in channel_id:
        raise PulserInteropError(
            "pulser.local_addressing_not_supported",
            "local addressing channels are not supported",
            path, construct=channel_id,
        )
    raise PulserInteropError(
        "pulser.channel_not_supported",
        f"channel {channel_id!r} is outside the supported subset",
        path, construct=channel_id,
    )


def _apply_operation(sequence: Sequence, op: Mapping[str, Any], path: tuple[str, ...]) -> None:
    kind = op.get("op")
    if kind == "pulse":
        if float(op.get("phase", 0.0)) != 0.0:
            raise PulserInteropError(
                "pulser.phase_not_supported",
                "pulses with nonzero phase are not supported",
                path + ("phase",), construct="phase",
            )
        if float(op.get("post_phase_shift", 0.0)) != 0.0:
            raise PulserInteropError(
                "pulser.phase_not_supported",
                "post phase shifts are not supported",
                path + ("post_phase_shift",), construct="post_phase_shift",
            )
        pulse = Pulse(
            amplitude=_waveform_from_ar(op["amplitude"], path + ("amplitude",)),
            detuning=_waveform_from_ar(op["detuning"], path + ("detuning",)),
        )
        sequence.add(pulse, op["channel"])
        return
    if kind == "delay":
        time_ns = int(op["time"])
        if time_ns > 0:
            sequence.delay(time_ns, op["channel"])
        return
    if kind == "target":
        raise PulserInteropError(
            "pulser.local_addressing_not_supported",
            "channel retargeting is local addressing",
            path, construct="target",
        )
    if kind in ("enable_eom_mode", "add_eom_pulse", "disable_eom_mode", "modify_eom_setpoint"):
        raise PulserInteropError(
            "pulser.eom_not_supported", "EOM mode is not supported",
            path, construct=str(kind),
        )
    raise PulserInteropError(
        "pulser.operation_not_supported",
        f"operation {kind!r} is outside the supported subset",
        path, construct=str(kind),
    )


def _waveform_from_ar(data: Mapping[str, Any], path: tuple[str, ...]) -> Waveform:
    kind = data.get("kind")
    if kind == "constant":
        return Waveform.constant(int(data["duration"]), float(data["value"]))
    if kind == "ramp":
        return Waveform.ramp(int(data["duration"]), float(data["start"]), float(data["stop"]))
    if kind == "blackman":
        return Waveform.blackman(int(data["duration"]), area=float(data["area"]))
    if kind == "blackman_max":
        return Waveform.blackman(int(data["duration"]), peak=float(data["max_val"]))
    if kind == "interpolated":
        duration = int(data["duration"])
        values = [float(v) for v in data["values"]]
        fractions = data.get("times")
        if fractions is None:
            fractions = np.linspace(0.0, 1.0, len(values))
        times_ns = [float(f) * duration for f in fractions]
        return Waveform.interpolated(duration, times_ns, values)
    if kind == "custom":
        return Waveform.custom([float(v) for v in data["samples"]], dt_ns=1)
    raise PulserInteropError(
        "pulser.waveform_not_supported",
        f"waveform kind {kind!r} is outside the supported subset",
        path, construct=str(kind),
    )


# ── Sequence export ──────────────────────────────────────────────────────────


def to_pulser_abstract_repr(sequence: Sequence) -> dict:
    """Export a native :class:`Sequence` to the supported Pulser subset.

    Raises :class:`PulserInteropError` instead of emitting a lossy payload
    when the sequence uses anything outside the subset.
    """
    register = sequence.register
    if register.dimensions != 2:
        raise PulserInteropError(
            "pulser.register_not_supported",
            f"only 2D registers export to the subset, got {register.dimensions}D",
            ("register",),
        )

    for alias, channel in sequence.declared_channels.items():
        if channel.addressing != "global" or channel.kind != "rydberg":
            code = (
                "pulser.local_addressing_not_supported"
                if channel.addressing == "local"
                else "pulser.channel_not_supported"
            )
            raise PulserInteropError(
                code, f"declared channel {alias!r} ({channel.channel_id}) is outside the subset",
                ("channels", alias), construct=channel.channel_id,
            )

    assert register.ids is not None  # normalized in Register.__post_init__
    payload: dict[str, Any] = {
        "version": "1",
        "name": "ryd-gate-sequence",
        "register": [
            {"name": atom_id, "x": float(x), "y": float(y)}
            for atom_id, (x, y) in zip(register.ids, register.coords_um)
        ],
        "channels": {
            alias: channel.channel_id
            for alias, channel in sequence.declared_channels.items()
        },
        "operations": [],
        "measurement": None,
    }
    if register.layout is not None:
        payload["layout"] = {
            "coordinates": [list(c) for c in register.layout.trap_coords_um],
            "slug": register.layout.name,
        }

    for index, op in enumerate(sequence.operations):
        path = ("operations", str(index))
        if isinstance(op, PulseOp):
            if op.pulse.phase_rad != 0.0 or op.pulse.post_phase_shift_rad != 0.0:
                raise PulserInteropError(
                    "pulser.phase_not_supported",
                    "pulses with nonzero phase are outside the subset",
                    path + ("phase",), construct="phase",
                )
            payload["operations"].append({
                "op": "pulse",
                "channel": op.channel,
                "protocol": "min-delay",
                "amplitude": _waveform_to_ar(op.pulse.amplitude, path + ("amplitude",)),
                "detuning": _waveform_to_ar(op.pulse.detuning, path + ("detuning",)),
                "phase": 0.0,
                "post_phase_shift": 0.0,
            })
        elif isinstance(op, DelayOp):
            payload["operations"].append({
                "op": "delay", "channel": op.channel, "time": op.duration_ns,
            })
        elif isinstance(op, MeasureOp):
            if op.basis != "rydberg":
                raise PulserInteropError(
                    "pulser.measurement_not_supported",
                    f"measurement basis {op.basis!r} has no Pulser counterpart in the subset",
                    ("measurement",), construct=op.basis,
                )
            payload["measurement"] = "ground-rydberg"
    return payload


def _waveform_to_ar(waveform: Waveform, path: tuple[str, ...]) -> dict:
    duration = waveform.duration_ns
    if waveform.kind == "constant":
        return {"kind": "constant", "duration": duration, "value": waveform.params["value"]}
    if waveform.kind == "ramp":
        return {
            "kind": "ramp", "duration": duration,
            "start": waveform.params["start"], "stop": waveform.params["stop"],
        }
    if waveform.kind == "blackman":
        # Exact inverse of Waveform.blackman(area=...): peak = area / (0.42 * T_us).
        area = waveform.params["peak"] * 0.42 * duration * 1e-3
        return {"kind": "blackman", "duration": duration, "area": area}
    if waveform.kind == "interpolated":
        return {
            "kind": "interpolated", "duration": duration,
            "times": [t / duration for t in waveform.params["times_ns"]],
            "values": list(waveform.params["values"]),
        }
    # custom
    assert waveform.samples is not None  # guaranteed by Waveform.custom
    if waveform.params["dt_ns"] != 1:
        raise PulserInteropError(
            "pulser.waveform_not_supported",
            "custom waveforms export only with dt_ns=1 (per-ns samples)",
            path, construct="custom",
        )
    return {"kind": "custom", "samples": list(waveform.samples)}


# ── NoiseModel bridge ────────────────────────────────────────────────────────

_NOISE_IMPORT_KEYS = {
    "runs", "n_trajectories", "state_prep_error", "p_false_pos", "p_false_neg",
    "amp_sigma", "detuning_sigma", "detuning_sigma_rad_per_us",
}
_NOISE_IGNORED_KEYS = {"noise_types", "samples_per_run"}


def noise_from_pulser_abstract_repr(data: Mapping[str, Any] | str) -> NoiseModel:
    """Map Pulser NoiseModel fields with matching semantics onto :class:`NoiseModel`."""
    payload = _parse(data)
    for key, value in payload.items():
        if key in _NOISE_IMPORT_KEYS:
            continue
        if key in _NOISE_IGNORED_KEYS:
            if key == "samples_per_run" and int(value or 1) != 1:
                raise PulserInteropError(
                    "pulser.noise_field_not_supported",
                    "samples_per_run > 1 has no native counterpart",
                    (key,), construct=key,
                )
            continue
        if value:  # nonzero / non-empty / True
            raise PulserInteropError(
                "pulser.noise_field_not_supported",
                f"noise field {key!r} has no matching native semantics",
                (key,), construct=key,
            )
    runs = int(payload.get("runs", payload.get("n_trajectories", 1)))
    detuning = payload.get(
        "detuning_sigma_rad_per_us", payload.get("detuning_sigma", 0.0)
    )
    return NoiseModel(
        runs=runs,
        detuning_sigma_rad_per_us=float(detuning),
        amp_sigma=float(payload.get("amp_sigma", 0.0)),
        state_prep_error=float(payload.get("state_prep_error", 0.0)),
        p_false_pos=float(payload.get("p_false_pos", 0.0)),
        p_false_neg=float(payload.get("p_false_neg", 0.0)),
    )


def noise_to_pulser_abstract_repr(noise: NoiseModel) -> dict:
    """Export the subset-compatible fields of a native :class:`NoiseModel`."""
    unsupported = [
        name for name, active in (
            ("local_rin_sigma", bool(noise.local_rin_sigma)),
            ("position_sigma_um", bool(noise.noise_types and "position" in noise.noise_types)),
            ("rydberg_decay", noise.rydberg_decay),
            ("intermediate_decay", noise.intermediate_decay),
            ("temperature_uK", bool(noise.temperature_uK)),
            ("laser_waist_um", bool(noise.laser_waist_um)),
        ) if active
    ]
    if unsupported:
        raise PulserInteropError(
            "pulser.noise_field_not_supported",
            f"native noise fields {unsupported} have no Pulser-subset counterpart",
            (unsupported[0],), construct=unsupported[0],
        )
    return {
        "runs": noise.runs,
        "state_prep_error": noise.state_prep_error,
        "p_false_pos": noise.p_false_pos,
        "p_false_neg": noise.p_false_neg,
        "amp_sigma": noise.amp_sigma,
        "detuning_sigma": noise.detuning_sigma_rad_per_us,
    }
