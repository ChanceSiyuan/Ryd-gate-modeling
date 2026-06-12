"""Stage 8: explicit lossy Protocol -> Sequence discretization bridge."""

import numpy as np
import pytest

from ryd_gate import (
    InteractionSpec,
    Register,
    RydbergSystem,
    SweepProtocol,
    sequence_from_protocol,
    simulate,
    simulate_sequence,
)

OMEGA = 2 * np.pi * 2e6  # rad/s
T_GATE = 1e-6


def _sweep_system(n_atoms=2):
    protocol = SweepProtocol(
        t_gate=T_GATE,
        omega_half_fn=lambda t: OMEGA / 2.0,
        delta_fn=lambda t: 2 * np.pi * 4e6 * (2.0 * t / T_GATE - 1.0),  # linear sweep
    )
    return RydbergSystem.from_lattice(
        Register.chain(n_atoms, spacing_um=20.0), "1r",
        interaction=InteractionSpec(C6=0.0), protocol=protocol,
    )


class TestBridge:
    def test_sweep_discretizes_and_matches_continuous(self):
        system = _sweep_system()
        seq = sequence_from_protocol(system, [], dt_ns=1)

        direct = simulate(system, [], "all_ground")
        bridged = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))

        n_direct = float(np.real(system.expectation("sum_nr", direct.psi_final)))
        assert bridged.expectation("sum_nr") == pytest.approx(n_direct, abs=1e-3)

    def test_waveforms_invert_the_lowering(self):
        seq = sequence_from_protocol(_sweep_system(), [], dt_ns=1)
        ops = [op for op in seq.operations if hasattr(op, "pulse")]
        pulse = ops[0].pulse
        # amp = Omega/2 rad/s -> waveform carries Omega in rad/us
        assert pulse.amplitude.value_at_ns(500) == pytest.approx(OMEGA / 1e6, rel=1e-9)
        # det = -Delta rad/s -> waveform carries Delta in rad/us (sweep midpoint = 0)
        assert pulse.detuning.value_at_ns(500) == pytest.approx(0.0, abs=1e-6)
        assert pulse.detuning.value_at_ns(0) == pytest.approx(-2 * np.pi * 4.0, rel=1e-6)

    def test_loss_metadata_stamped(self):
        seq = sequence_from_protocol(_sweep_system(), [], dt_ns=4)
        pulse = next(op.pulse for op in seq.operations if hasattr(op, "pulse"))
        assert pulse.metadata["discretized_from"] == "SweepProtocol"
        assert pulse.metadata["dt_ns"] == 4
        assert pulse.metadata["t_gate_s"] == pytest.approx(T_GATE)

    def test_round_trips_through_serialization(self):
        from ryd_gate import Sequence

        seq = sequence_from_protocol(_sweep_system(), [], dt_ns=2)
        rebuilt = Sequence.from_dict(seq.to_dict())
        assert rebuilt.duration_ns == seq.duration_ns

    def test_two_photon_protocol_refused(self):
        from ryd_gate.gates import TOProtocol

        system = RydbergSystem.from_lattice(
            Register.chain(2, spacing_um=3.0), "rb87_7", param_set="our",
            protocol=TOProtocol(),
        )
        x = [-0.699, 1.03, 0.376, 1.571, 1.445, 0.13]
        with pytest.raises(ValueError, match="discretize.channel_not_representable"):
            sequence_from_protocol(system, x)

    def test_protocol_missing_refused(self):
        system = RydbergSystem.from_lattice(
            Register.chain(2, spacing_um=20.0), "1r",
            interaction=InteractionSpec(C6=0.0),
        )
        with pytest.raises(ValueError, match="discretize.protocol_missing"):
            sequence_from_protocol(system, [])

    def test_local_addressing_protocol_refused(self):
        protocol = SweepProtocol(
            t_gate=T_GATE,
            omega_half_fn=lambda t: OMEGA / 2.0,
            delta_fn=lambda t: 0.0,
            address_fn=lambda t, i: 2 * np.pi * 1e6 if i == 0 else 0.0,
        )
        system = RydbergSystem.from_lattice(
            Register.chain(2, spacing_um=20.0), "1r",
            interaction=InteractionSpec(C6=0.0), protocol=protocol,
        )
        with pytest.raises(ValueError, match="discretize.channel_not_representable"):
            sequence_from_protocol(system, [])
