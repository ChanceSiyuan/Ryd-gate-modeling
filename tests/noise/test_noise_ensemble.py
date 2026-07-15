"""NoiseModel + simulate_ensemble behavior (N01-N18).

Pins the quasi-static noise contract: a NoiseModel declaring zero-mean Gaussian
sigmas on named laser groups (amplitude / frequency) and on atom position (a 3D
per-atom jitter); mandatory integer ``shots``/``seed``; a shot-major realization
schema with exactly the three N16 keys, immutable and seed-reproducible
independent of backend/t_eval/observables (N17); the N14 preflight that rejects
named laser noise on channel-only protocols and accepts it on the matching laser
group; and position noise that supports arbitrary ``N`` with a fixed pair
topology. EnsembleResult exposes ``results``/``realizations``/``seed`` as tuples.
"""

import numpy as np
import pytest

from ryd_gate import (
    NoiseModel,
    Register,
    RydbergSystem,
    level_structure,
    simulate_ensemble,
)
from ryd_gate.protocols import CZProtocol, DigitalAnalogProtocol, SweepProtocol
from ryd_gate.results import EnsembleResult, EvolutionResult

MHZ = 2 * np.pi * 1e6
T_GATE = 0.05e-6


def _da_system(n=2, spacing_um=4.0, cutoff_um=None):
    """Fast 01r system driven only by effective channel drives (no physical laser)."""
    protocol = DigitalAnalogProtocol(
        t_gate_s=T_GATE,
        coupling_r1_rad_s=lambda t: 2.0 * MHZ,
    )
    return RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(n, spacing_um=spacing_um),
        protocol=protocol,
        interaction_cutoff_um=cutoff_um,
    )


def _sweep_system():
    protocol = SweepProtocol(
        t_gate_s=T_GATE,
        omega_half_rad_s=lambda t: 1.0 * MHZ,
        detuning_rad_s=lambda t: 0.0,
    )
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=4.0),
        protocol=protocol,
    )


def _cz_system(n=1):
    """rb87_7_mp + a CZ pulse: the only kind of protocol that drives named lasers.

    Modest MHz-scale detuning keeps the seven-level ODE fast (no GHz aliasing).
    """
    protocol = CZProtocol(
        t_gate_s=T_GATE,
        intermediate_detuning_rad_s=50.0 * MHZ,
        omega_420_max_rad_s=10.0 * MHZ,
        omega_1013_max_rad_s=10.0 * MHZ,
        envelope_420=lambda t: 1.0,
        phase_420_rad=lambda t: 0.0,
    )
    return RydbergSystem(
        level_structure=level_structure("rb87_7_mp"),
        register=Register.chain(n, spacing_um=4.0),
        protocol=protocol,
    )


def _n_r(system):
    obs = system.observables
    return {"n_r": sum(obs.n("r", i) for i in range(system.N))}


def _norm(rec):
    """Plain comparable snapshot of a realization record."""
    return (
        dict(rec["laser_amplitude_scales"]),
        dict(rec["laser_frequency_offsets_rad_s"]),
        rec["position_offsets_um"],
    )


class TestNoiseModelValidation:
    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="at least one nonzero sigma"):
            NoiseModel()

    def test_all_zero_is_rejected(self):
        with pytest.raises(ValueError, match="at least one nonzero sigma"):
            NoiseModel(laser_amplitude_sigma={"420": 0.0}, position_sigma_um=(0.0, 0.0, 0.0))

    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            NoiseModel(laser_amplitude_sigma={"420": -0.01})
        with pytest.raises(ValueError, match="finite and non-negative"):
            NoiseModel(laser_frequency_sigma_rad_s={"1013": -1.0})

    def test_nonfinite_sigma_rejected(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            NoiseModel(laser_amplitude_sigma={"420": np.inf})
        with pytest.raises(ValueError, match="finite and non-negative"):
            NoiseModel(laser_frequency_sigma_rad_s={"420": np.nan})

    def test_position_sigma_must_be_3tuple(self):
        with pytest.raises(ValueError, match=r"\(sx, sy, sz\)"):
            NoiseModel(position_sigma_um=(0.1, 0.1))
        with pytest.raises(ValueError, match=r"\(sx, sy, sz\)"):
            NoiseModel(position_sigma_um=(0.1, -0.1, 0.1))

    def test_nonstring_group_key_rejected(self):
        with pytest.raises(TypeError, match="laser-group strings"):
            NoiseModel(laser_amplitude_sigma={420: 0.01})

    def test_fields_and_immutability(self):
        noise = NoiseModel(
            laser_amplitude_sigma={"420": 0.02},
            laser_frequency_sigma_rad_s={"1013": 0.5 * MHZ},
            position_sigma_um=(0.05, 0.05, 0.1),
        )
        assert noise.laser_amplitude_sigma == {"420": 0.02}
        assert noise.laser_frequency_sigma_rad_s == {"1013": 0.5 * MHZ}
        assert noise.position_sigma_um == (0.05, 0.05, 0.1)
        with pytest.raises(AttributeError):
            noise.position_sigma_um = (0.0, 0.0, 0.0)
        with pytest.raises((TypeError, AttributeError)):
            noise.laser_amplitude_sigma["420"] = 1.0


class TestEnsemblePreflight:
    def test_shots_and_seed_are_mandatory_keywords(self):
        with pytest.raises(TypeError):
            simulate_ensemble(_da_system(), noise=NoiseModel(position_sigma_um=(0.05, 0.05, 0.05)), shots=2)
        with pytest.raises(TypeError):
            simulate_ensemble(_da_system(), noise=NoiseModel(position_sigma_um=(0.05, 0.05, 0.05)), seed=0)

    def test_seed_must_be_integer(self):
        with pytest.raises(TypeError, match="seed must be an integer"):
            simulate_ensemble(
                _da_system(), noise=NoiseModel(position_sigma_um=(0.05, 0.05, 0.05)), shots=2, seed=None
            )

    def test_shots_must_be_positive_integer(self):
        noise = NoiseModel(position_sigma_um=(0.05, 0.05, 0.05))
        with pytest.raises(ValueError, match="shots must be a positive integer"):
            simulate_ensemble(_da_system(), noise=noise, shots=0, seed=0)
        with pytest.raises(ValueError, match="shots must be a positive integer"):
            simulate_ensemble(_da_system(), noise=noise, shots=1.5, seed=0)

    def test_noise_must_be_noisemodel(self):
        with pytest.raises(TypeError, match="noise must be a NoiseModel"):
            simulate_ensemble(_da_system(), noise=None, shots=1, seed=0)

    def test_named_laser_noise_on_channel_protocol_rejected(self):
        # DigitalAnalog + Sweep produce only effective channel drives (N14).
        with pytest.raises(ValueError, match="not driven by this protocol"):
            simulate_ensemble(
                _da_system(), noise=NoiseModel(laser_amplitude_sigma={"420": 0.02}), shots=1, seed=0
            )
        with pytest.raises(ValueError, match="not driven by this protocol"):
            simulate_ensemble(
                _sweep_system(), noise=NoiseModel(laser_frequency_sigma_rad_s={"1013": 0.5 * MHZ}), shots=1, seed=0
            )

    def test_unknown_group_on_laser_protocol_rejected(self):
        with pytest.raises(ValueError, match="not driven by this protocol"):
            simulate_ensemble(
                _cz_system(), noise=NoiseModel(laser_amplitude_sigma={"999": 0.02}), shots=1, seed=0
            )


class TestEnsembleExecution:
    def test_matching_laser_group_accepted(self):
        """Named amplitude/frequency noise on the 420/1013 groups a CZ pulse drives (N14)."""
        noise = NoiseModel(
            laser_amplitude_sigma={"420": 0.02},
            laser_frequency_sigma_rad_s={"1013": 0.5 * MHZ},
        )
        ens = simulate_ensemble(_cz_system(), noise=noise, shots=2, seed=0)
        assert isinstance(ens, EnsembleResult)
        assert len(ens.results) == 2
        rec = ens.realizations[0]
        assert set(rec["laser_amplitude_scales"]) == {"420"}
        assert set(rec["laser_frequency_offsets_rad_s"]) == {"1013"}
        assert rec["position_offsets_um"] is None

    def test_container_types(self):
        ens = simulate_ensemble(
            _da_system(), noise=NoiseModel(position_sigma_um=(0.05, 0.05, 0.05)), shots=3, seed=7
        )
        assert isinstance(ens, EnsembleResult)
        assert isinstance(ens.results, tuple) and len(ens.results) == 3
        assert isinstance(ens.realizations, tuple) and len(ens.realizations) == 3
        assert ens.seed == 7
        with pytest.raises(AttributeError):
            ens.seed = 0

    def test_single_vs_batch_shot_major_shape(self):
        noise = NoiseModel(position_sigma_um=(0.05, 0.05, 0.05))
        single = simulate_ensemble(_da_system(), noise=noise, shots=2, seed=1)
        assert all(isinstance(shot, EvolutionResult) for shot in single.results)

        batch = simulate_ensemble(
            _da_system(), [["1", "1"], ["0", "1"]], noise=noise, shots=2, seed=1
        )
        assert len(batch.results) == 2  # shot-major
        for shot in batch.results:
            assert isinstance(shot, tuple) and len(shot) == 2
            assert all(isinstance(r, EvolutionResult) for r in shot)

    def test_realization_schema_and_immutability(self):
        noise = NoiseModel(position_sigma_um=(0.05, 0.05, 0.1))
        ens = simulate_ensemble(_da_system(), noise=noise, shots=1, seed=5)
        rec = ens.realizations[0]
        assert set(rec) == {
            "laser_amplitude_scales",
            "laser_frequency_offsets_rad_s",
            "position_offsets_um",
        }
        with pytest.raises((TypeError, AttributeError)):
            rec["position_offsets_um"] = None

    def test_same_seed_reproduces_independent_of_teval_and_observables(self):
        """Realizations depend only on (noise, N, shots, seed), not the readout (N17)."""
        noise = NoiseModel(position_sigma_um=(0.05, 0.05, 0.1))
        sys_a = _da_system()
        a = simulate_ensemble(sys_a, noise=noise, shots=3, seed=11)
        b = simulate_ensemble(
            _da_system(),
            noise=noise,
            shots=3,
            seed=11,
            t_eval=[0.5 * T_GATE, T_GATE],
            observables=_n_r(sys_a),
        )
        assert [_norm(r) for r in a.realizations] == [_norm(r) for r in b.realizations]

    def test_batch_shares_one_realization_per_shot(self):
        """Both states of a shot evolve under the same sampled system; shots differ."""
        sys = _da_system()
        noise = NoiseModel(position_sigma_um=(0.1, 0.1, 0.1))
        ens = simulate_ensemble(
            sys, [["1", "1"], ["1", "1"]], noise=noise, shots=2, seed=3, observables=_n_r(sys)
        )
        for shot in ens.results:
            np.testing.assert_allclose(shot[0].expectation("n_r"), shot[1].expectation("n_r"))
        off = [r["position_offsets_um"] for r in ens.realizations]
        assert off[0] != off[1]

    @pytest.mark.parametrize("n", [1, 4])
    def test_position_noise_supports_arbitrary_N(self, n):
        """Per-atom (N, 3) offsets for any N; nominal pair topology fixed by the cutoff."""
        noise = NoiseModel(position_sigma_um=(0.05, 0.05, 0.1))
        ens = simulate_ensemble(
            _da_system(n=n, spacing_um=4.0, cutoff_um=5.0), noise=noise, shots=1, seed=0
        )
        offsets = ens.realizations[0]["position_offsets_um"]
        assert len(offsets) == n
        assert all(len(row) == 3 for row in offsets)
