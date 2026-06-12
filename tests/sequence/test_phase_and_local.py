"""Stage 8: pulse phase (virtual-Z) and local-channel targeting."""

import numpy as np
import pytest

from ryd_gate import (
    DeviceSpec,
    InteractionSpec,
    Pulse,
    Register,
    Sequence,
    TargetOp,
    Waveform,
    simulate_sequence,
)


def _seq(n_atoms=1, spacing=20.0, model="1r"):
    return Sequence(Register.chain(n_atoms, spacing), DeviceSpec.virtual_rb87(), model)


def _half_pi(phase_rad=0.0, post_phase_shift_rad=0.0):
    return Pulse.constant_detuning(
        Waveform.blackman(1000, area=np.pi / 2), 0.0,
        phase_rad=phase_rad, post_phase_shift_rad=post_phase_shift_rad,
    )


class TestPhasePhysics:
    """Ramsey logic through the exact solver: phase is physical, not dropped."""

    def _run(self, second_phase=0.0, first_post_shift=0.0):
        seq = _seq()
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(_half_pi(post_phase_shift_rad=first_post_shift), "ryd")
        seq.add(_half_pi(phase_rad=second_phase), "ryd")
        result = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))
        return float(result.populations("r")[0])

    def test_in_phase_half_pis_make_a_pi(self):
        assert self._run(second_phase=0.0) == pytest.approx(1.0, abs=1e-3)

    def test_opposite_phase_echoes_back(self):
        assert self._run(second_phase=np.pi) == pytest.approx(0.0, abs=1e-3)

    def test_quadrature_phase_gives_half(self):
        assert self._run(second_phase=np.pi / 2) == pytest.approx(0.5, abs=1e-2)

    def test_post_phase_shift_acts_as_virtual_z(self):
        """post_phase_shift on pulse 1 == explicit phase on pulse 2."""
        via_shift = self._run(first_post_shift=np.pi)
        via_phase = self._run(second_phase=np.pi)
        assert via_shift == pytest.approx(via_phase, abs=1e-6)

    def test_phase_refused_on_tn_backends(self):
        seq = _seq(n_atoms=2)
        seq.declare_channel("ryd", "rydberg_global")
        seq.add(_half_pi(phase_rad=0.5), "ryd")
        with pytest.raises(ValueError, match="sequence.phase_backend_unsupported"):
            simulate_sequence(seq, backend="mps")


class TestLocalTargeting:
    def _local_pi_sequence(self, model="1r", targets=("q0",)):
        seq = _seq(n_atoms=2, model=model)
        seq.declare_channel("loc", "rydberg_local")
        seq.target(list(targets), "loc")
        seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "loc")
        return seq

    def test_local_pi_drives_only_the_target(self):
        result = simulate_sequence(
            self._local_pi_sequence(), interaction=InteractionSpec(C6=0.0)
        )
        populations = result.populations("r")
        assert populations[0] == pytest.approx(1.0, abs=1e-3)
        assert populations[1] == pytest.approx(0.0, abs=1e-6)

    def test_local_pi_on_01r_model(self):
        result = simulate_sequence(
            self._local_pi_sequence(model="01r", targets=("q1",)),
            interaction=InteractionSpec(C6=0.0),
        )
        populations = result.populations("r")
        assert populations[0] == pytest.approx(0.0, abs=1e-6)
        assert populations[1] == pytest.approx(1.0, abs=1e-3)

    def test_local_pi_on_mps_backend(self):
        pytest.importorskip("tenpy")
        result = simulate_sequence(
            self._local_pi_sequence(),
            backend="mps",
            interaction=InteractionSpec(C6=0.0),
            backend_options={"chi_max": 16, "dt": 2.5e-7},
        )
        populations = result.populations("r")
        assert populations[0] == pytest.approx(1.0, abs=2e-2)
        assert populations[1] == pytest.approx(0.0, abs=2e-2)

    def test_retarget_between_pulses(self):
        seq = _seq(n_atoms=2)
        seq.declare_channel("loc", "rydberg_local")
        seq.target("q0", "loc")
        seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "loc")
        seq.target("q1", "loc")
        seq.add(Pulse.constant_detuning(Waveform.blackman(1000, area=np.pi), 0.0), "loc")
        result = simulate_sequence(seq, interaction=InteractionSpec(C6=0.0))
        np.testing.assert_allclose(result.populations("r"), [1.0, 1.0], atol=1e-3)

    def test_serialization_replays_target_ops(self):
        seq = self._local_pi_sequence(targets=("q1",))
        data = seq.to_dict()
        target_ops = [op for op in data["operations"] if op["type"] == "target"]
        assert target_ops == [{"type": "target", "channel": "loc", "targets": ["q1"], "t_start_ns": 0}]
        rebuilt = Sequence.from_dict(data)
        ops = [op for op in rebuilt.operations if isinstance(op, TargetOp)]
        assert ops[0].targets == ("q1",)
        result = simulate_sequence(rebuilt, interaction=InteractionSpec(C6=0.0))
        assert result.populations("r")[1] == pytest.approx(1.0, abs=1e-3)

    def test_target_validation_rules(self):
        seq = _seq(n_atoms=2)
        seq.declare_channel("loc", "rydberg_local")
        with pytest.raises(ValueError, match="sequence.target_unknown_atom"):
            seq.target(["q9"], "loc")
        with pytest.raises(ValueError, match="sequence.target_duplicate"):
            seq.target(["q0", "q0"], "loc")
        with pytest.raises(ValueError, match="sequence.target_empty"):
            seq.target([], "loc")
        with pytest.raises(ValueError, match="sequence.local_targets_missing"):
            seq.add(Pulse.constant(100, 1.0, 0.0), "loc")

    def test_target_on_global_channel_refused(self):
        seq = _seq()
        seq.declare_channel("ryd", "rydberg_global")
        with pytest.raises(ValueError, match="sequence.target_global"):
            seq.target(["q0"], "ryd")

    def test_max_targets_enforced(self):
        from dataclasses import replace

        device = DeviceSpec.virtual_rb87()
        limited = replace(device.channels["rydberg_local"], max_targets=1)
        device = replace(device, channels={**device.channels, "rydberg_local": limited})
        seq = Sequence(Register.chain(2, 20.0), device, "1r")
        seq.declare_channel("loc", "rydberg_local")
        with pytest.raises(ValueError, match="sequence.max_targets"):
            seq.target(["q0", "q1"], "loc")

    def test_schema_validates_target_payload(self):
        pytest.importorskip("jsonschema")
        from ryd_gate.core.serialization import validate_json_schema

        data = self._local_pi_sequence().to_dict()
        assert validate_json_schema(data, "sequence") == []


class TestBackendDispatch:
    def test_gputn_and_peps_reach_the_kernel_dispatcher(self, monkeypatch):
        """No NotImplementedError gate; non-native states get UnsupportedStateHandle."""
        import sys

        import ryd_gate.simulate  # noqa: F401  (the package attr shadows the module)
        from ryd_gate.ir import EvolutionResult

        simulate_module = sys.modules["ryd_gate.simulate"]

        captured = {}

        def fake_simulate(system, x, psi0, *, backend, **kwargs):
            captured["backend"] = backend
            return EvolutionResult(psi_final=None, metadata={})

        monkeypatch.setattr(simulate_module, "simulate", fake_simulate)

        for backend in ("gputn", "peps"):
            seq = _seq(n_atoms=2)
            seq.declare_channel("ryd", "rydberg_global")
            seq.add(Pulse.constant(1000, 1.0, 0.0), "ryd")
            result = simulate_module.simulate_sequence(seq, backend=backend)
            assert captured["backend"] == backend
            assert result.state.kind == "unsupported"
            assert result.capabilities == frozenset()
            assert result.state.reason_code == f"{backend}.state_handle_not_implemented"
