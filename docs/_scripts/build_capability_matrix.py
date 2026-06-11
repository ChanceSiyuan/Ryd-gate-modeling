"""Generate docs/capability_matrix.md from code (never hand-edited).

Every table row is derived from the runtime objects themselves —
``LevelStructureSpec.supports_backend``, the Stage 3 state-handle
``capabilities`` sets, ``NoiseModel.validate_for``, the ``simulate_sequence``
backend gate, and the interop module's supported-subset constants — so the
matrix cannot rot relative to the code.

Usage:
    uv run python docs/_scripts/build_capability_matrix.py        # rewrite
    uv run python docs/_scripts/build_capability_matrix.py --check  # CI diff
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PRESETS = ("01", "1r", "01r", "ger", "analog_3", "rb87_7")
BACKENDS = ("exact", "mps", "gputn", "peps", "stabilizer")
SEQUENCE_BACKENDS = BACKENDS
OUT_PATH = Path(__file__).resolve().parents[1] / "capability_matrix.md"

YES, NO = "yes", "—"


def _level_structure_table() -> list[str]:
    from ryd_gate import level_structure

    lines = [
        "| level structure | " + " | ".join(BACKENDS) + " |",
        "|---" * (len(BACKENDS) + 1) + "|",
    ]
    for name in PRESETS:
        spec = level_structure(name)
        cells = [YES if spec.supports_backend(b) else NO for b in BACKENDS]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return lines


def _handle_capability_table() -> list[str]:
    from ryd_gate import Register, level_structure
    from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
    from ryd_gate.results import ExactStateHandle, MPSStateHandle, UnsupportedStateHandle

    handles = {
        "exact (`statevector`)": ExactStateHandle(
            psi=np.zeros(2, dtype=complex), system=None,
            register=Register.chain(1), level_structure=level_structure("1r"),
        ),
        "mps (`mps`)": MPSStateHandle(
            mps=object(), spec=create_tn_lattice_spec(1, 1, level_structure="1r"),
            register_ids=("q0",),
        ),
        "gputn / peps (`unsupported`)": UnsupportedStateHandle(
            backend="gputn", reason_code="gputn.state_handle_not_implemented",
        ),
    }
    all_caps = sorted({cap for handle in handles.values() for cap in handle.capabilities})
    lines = [
        "| result handle | " + " | ".join(f"`{c}`" for c in all_caps) + " |",
        "|---" * (len(all_caps) + 1) + "|",
    ]
    for label, handle in handles.items():
        cells = [YES if cap in handle.capabilities else NO for cap in all_caps]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _noise_table() -> list[str]:
    from ryd_gate import NoiseModel

    probes = {
        "detuning / amplitude / local RIN / position (Monte Carlo)": NoiseModel(
            detuning_sigma_rad_per_us=0.01
        ),
        "rydberg / intermediate decay (construction-time)": NoiseModel(rydberg_decay=True),
        "state-prep / readout / temperature / waist (data only)": NoiseModel(
            state_prep_error=0.01
        ),
    }
    lines = [
        "| noise group | " + " | ".join(BACKENDS) + " |",
        "|---" * (len(BACKENDS) + 1) + "|",
    ]
    for label, model in probes.items():
        cells = []
        for backend in BACKENDS:
            issues = model.validate_for(backend=backend, level_structure="rb87_7", n_atoms=2)
            cells.append(YES if not issues else NO)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _sequence_table() -> list[str]:
    from ryd_gate import DeviceSpec, Pulse, Register, Sequence, simulate_sequence

    lines = [
        "| backend | `simulate_sequence` |",
        "|---|---|",
    ]
    for backend in SEQUENCE_BACKENDS:
        # Two atoms: the smallest register every sequence backend accepts
        # (the two-site TDVP engine cannot sweep a single site).
        seq = Sequence(Register.chain(2, 20.0), DeviceSpec.virtual_rb87(), "1r")
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(Pulse.constant(1000, 0.1, 0.0), "ryd")
        kwargs = {"backend_options": {"chi_max": 8, "dt": 2.5e-7}} if backend == "mps" else {}
        try:
            simulate_sequence(seq, backend=backend, **kwargs)
        except NotImplementedError:
            lines.append(f"| `{backend}` | {NO} (not on the sequence path) |")
        except ModuleNotFoundError:
            lines.append(f"| `{backend}` | {YES} (needs the optional backend extra) |")
        else:
            lines.append(f"| `{backend}` | {YES} |")
    return lines


def _interop_table() -> list[str]:
    from ryd_gate.interop import pulser as interop
    from ryd_gate.pulse import _WAVEFORM_KINDS

    noise_keys = sorted(interop._NOISE_IMPORT_KEYS)
    return [
        "| construct | supported subset |",
        "|---|---|",
        f"| channels | `{interop._SUPPORTED_CHANNEL_ID}` only |",
        f"| level structure on import | `{interop._IMPORT_LEVEL_STRUCTURE}` |",
        "| waveforms | " + ", ".join(f"`{k}`" for k in _WAVEFORM_KINDS) + " |",
        "| pulses | zero phase, zero post-phase-shift |",
        "| measurement | `ground-rydberg` |",
        "| layout | trap coordinates + trap → qubit mapping |",
        "| noise fields | " + ", ".join(f"`{k}`" for k in noise_keys) + " |",
    ]


def build_matrix() -> str:
    sections = [
        ("# Capability Matrix", [
            "",
            "Generated by `docs/_scripts/build_capability_matrix.py` — do not edit by hand.",
            "Regenerate with `uv run python docs/_scripts/build_capability_matrix.py`.",
            "",
        ]),
        ("## Level structure × backend", _level_structure_table()),
        ("## Result-handle capabilities by backend", _handle_capability_table()),
        ("## NoiseModel runtime support by backend", _noise_table()),
        ("## Sequence support by backend", _sequence_table()),
        ("## Pulser interop subset", _interop_table()),
    ]
    parts: list[str] = []
    for title, lines in sections:
        parts.append(title)
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    text = build_matrix()
    if "--check" in sys.argv:
        current = OUT_PATH.read_text() if OUT_PATH.exists() else ""
        if current != text:
            print(f"{OUT_PATH} is stale; regenerate it.", file=sys.stderr)
            return 1
        print("capability matrix is up to date")
        return 0
    OUT_PATH.write_text(text)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
