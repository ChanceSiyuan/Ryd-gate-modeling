#!/usr/bin/env python3
"""Minimal Sequence-API demo: a Blackman pi pulse on the exact backend.

Builds a device-validated sequence, runs it exactly, and queries the lazy
result handle. Bounded runtime (~seconds).

Usage:
    uv run python examples/demo_sequence_exact.py
"""

import numpy as np

from ryd_gate import DeviceSpec, Pulse, Register, Sequence, Waveform, simulate_sequence


def main() -> None:
    seq = Sequence(Register.chain(2, spacing_um=20.0), DeviceSpec.virtual_rb87(), "1r")
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "ryd")
    seq.measure("rydberg")

    result = simulate_sequence(seq)
    print("capabilities:", sorted(result.capabilities))
    print("per-site |r> population:", np.round(result.populations("r"), 6))
    print("total Rydberg expectation:", round(result.expectation("sum_nr"), 6))
    print("1000-shot sample:", result.sample(1000, basis="rydberg", seed=7))

    payload = seq.to_dict()
    print("serialized schema tag:", payload["schema"])
    rebuilt = Sequence.from_dict(payload)
    print("round-trip atoms:", rebuilt.register.ids)


if __name__ == "__main__":
    main()
