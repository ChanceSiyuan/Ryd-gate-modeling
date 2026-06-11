#!/usr/bin/env python3
"""Pulser interop demo: import a Pulser abstract-repr payload, run it, export it.

No Pulser installation required — the bridge works on the JSON abstract
representation directly. Bounded runtime (~seconds).

Usage:
    uv run python examples/demo_pulser_interop.py
"""

import json

import numpy as np

from ryd_gate import simulate_sequence
from ryd_gate.interop import (
    PulserInteropError,
    from_pulser_abstract_repr,
    to_pulser_abstract_repr,
)

PULSER_PAYLOAD = {
    "version": "1",
    "name": "afm-demo",
    "register": [
        {"name": "q0", "x": 0.0, "y": 0.0},
        {"name": "q1", "x": 20.0, "y": 0.0},
    ],
    "channels": {"ryd": "rydberg_global"},
    "operations": [
        {
            "op": "pulse",
            "channel": "ryd",
            "protocol": "min-delay",
            "amplitude": {"kind": "blackman", "duration": 1000, "area": float(np.pi)},
            "detuning": {"kind": "constant", "duration": 1000, "value": 0.0},
            "phase": 0.0,
            "post_phase_shift": 0.0,
        },
        {"op": "delay", "channel": "ryd", "time": 200},
    ],
    "measurement": "ground-rydberg",
}


def main() -> None:
    seq = from_pulser_abstract_repr(json.dumps(PULSER_PAYLOAD))
    print("imported atoms:", seq.register.ids, "| level structure:", seq.level_structure.name)

    result = simulate_sequence(seq)
    print("per-site |r> population:", np.round(result.populations("r"), 6))

    exported = to_pulser_abstract_repr(seq)
    print("export channels:", exported["channels"], "| measurement:", exported["measurement"])

    unsupported = dict(PULSER_PAYLOAD, operations=[{"op": "enable_eom_mode", "channel": "ryd"}])
    try:
        from_pulser_abstract_repr(unsupported)
    except PulserInteropError as err:
        print(f"typed refusal: {err.code} at {'/'.join(err.path)}")


if __name__ == "__main__":
    main()
