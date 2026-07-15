"""NoiseModel + simulate_ensemble behavior.

Pins the ensemble execution contract: mandatory integer ``shots``/``seed``,
exact-only capability, seed reproducibility, one shared realization per shot
across a batch of initial states, unit-bearing realization records, and
position noise derived from the actual register coordinates.
"""

import numpy as np
import pytest

from ryd_gate import (
    EnsembleResult,
    NoiseModel,
    Register,
    RydbergSystem,
    TFIMQuenchProtocol,
    level_structure,
    simulate_ensemble,
)
from ryd_gate.protocols import SweepProtocol

MHZ = 2 * np.pi * 1e6
T_GATE = 0.05e-6


def _system(spacing_um=4.0):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=spacing_um),
        protocol=TFIMQuenchProtocol(hx=2 * np.pi * 1e6, t_gate=T_GATE),
    )


def _addressed_system():
    protocol = SweepProtocol(
        t_gate=T_GATE,
        omega_half_fn=lambda t: 0.5 * 2.0 * MHZ,
        delta_fn=lambda t: 0.0,
        address_fn=lambda t, i: 1.0 * MHZ * i,
    )
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.chain(2, spacing_um=4.0),
        protocol=protocol,
    )


class TestNoiseModelValidation:
    def test_negative_sigma_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            NoiseModel(detuning_sigma_rad_s=-1.0)

    def test_position_sigma_shape(self):
        with pytest.raises(ValueError, match="scalar or"):
            NoiseModel(position_sigma_um=(0.1, 0.1))

    def test_scalar_position_sigma_broadcasts(self):
        noise = NoiseModel(position_sigma_um=0.1)
        assert noise.position_sigma_xyz_um == (0.1, 0.1, 0.1)

    def test_default_is_inactive(self):
        assert not NoiseModel().any_active
        assert NoiseModel(amplitude_sigma=0.01).any_active


class TestEnsemblePreflight:
    def test_shots_and_seed_are_mandatory_keywords(self):
        with pytest.raises(TypeError):
            simulate_ensemble(_system(), noise=NoiseModel(), shots=2)  # no seed
        with pytest.raises(TypeError):
            simulate_ensemble(_system(), noise=NoiseModel(), seed=0)  # no shots

    def test_seed_must_be_integer(self):
        with pytest.raises(TypeError, match="seed must be an integer"):
            simulate_ensemble(_system(), noise=NoiseModel(), shots=2, seed=None)

    def test_shots_must_be_positive(self):
        with pytest.raises(ValueError, match="shots must be a positive integer"):
            simulate_ensemble(_system(), noise=NoiseModel(), shots=0, seed=0)

    def test_tn_backend_capability_error(self):
        with pytest.raises(ValueError, match="exact backends"):
            simulate_ensemble(
                _system(), noise=NoiseModel(), shots=1, seed=0, backend="mps"
            )

    def test_local_rin_without_addressing_errors(self):
        with pytest.raises(ValueError, match="local-addressing"):
            simulate_ensemble(
                _system(),
                noise=NoiseModel(local_rin_sigma=0.05),
                shots=1,
                seed=0,
            )


class TestEnsembleExecution:
    def test_shot_major_shape_and_types(self):
        ens = simulate_ensemble(
            _system(),
            [["1", "1"], ["r", "1"]],
            noise=NoiseModel(detuning_sigma_rad_s=0.5 * MHZ),
            shots=3,
            seed=7,
        )
        assert isinstance(ens, EnsembleResult)
        assert len(ens.results) == 3
        assert all(len(shot) == 2 for shot in ens.results)
        assert len(ens.realizations) == 3
        assert ens.seed == 7

    def test_same_seed_reproduces(self):
        kwargs = dict(
            noise=NoiseModel(detuning_sigma_rad_s=0.5 * MHZ, amplitude_sigma=0.02),
            shots=3,
            seed=11,
        )
        a = simulate_ensemble(_system(), **kwargs)
        b = simulate_ensemble(_system(), **kwargs)
        assert a.realizations == b.realizations
        for shot_a, shot_b in zip(a.results, b.results):
            np.testing.assert_allclose(shot_a[0].final_state, shot_b[0].final_state)

    def test_batch_shares_realization_within_shot(self):
        """Both initial states of one shot see the same sampled Hamiltonian."""
        noise = NoiseModel(detuning_sigma_rad_s=2.0 * MHZ)
        ens = simulate_ensemble(
            _system(), [["1", "1"], ["1", "1"]], noise=noise, shots=2, seed=3
        )
        for shot in ens.results:
            np.testing.assert_allclose(shot[0].final_state, shot[1].final_state)
        # ... and different shots see different realizations.
        deltas = [r["detuning_rad_s"] for r in ens.realizations]
        assert deltas[0] != deltas[1]

    def test_realizations_are_unit_bearing_values(self):
        noise = NoiseModel(
            detuning_sigma_rad_s=0.5 * MHZ,
            amplitude_sigma=0.02,
            position_sigma_um=0.05,
        )
        ens = simulate_ensemble(_system(), noise=noise, shots=1, seed=5)
        rec = ens.realizations[0]
        assert set(rec) == {
            "detuning_rad_s",
            "amplitude_scale",
            "position_offsets_um",
            "pair_distances_um",
        }
        assert all(
            not callable(v) and not isinstance(v, np.ndarray) for v in rec.values()
        )

    def test_position_noise_uses_register_distance(self):
        """Nominal pair distance comes from the register (no hard-coded 3 um)."""
        noise = NoiseModel(position_sigma_um=1e-6)  # essentially frozen atoms
        ens = simulate_ensemble(
            _system(spacing_um=4.0), noise=noise, shots=1, seed=0
        )
        (d_new,) = ens.realizations[0]["pair_distances_um"].values()
        assert d_new == pytest.approx(4.0, abs=1e-3)

    def test_local_rin_scales_addressing(self):
        noise = NoiseModel(local_rin_sigma=0.1)
        ens = simulate_ensemble(_addressed_system(), noise=noise, shots=2, seed=9)
        for rec in ens.realizations:
            scales = rec["local_scales"]
            assert set(scales) == {0, 1}
            assert all(np.isfinite(s) and s > 0 for s in scales.values())

    def test_amplitude_noise_changes_evolution(self):
        clean = simulate_ensemble(
            _system(), noise=NoiseModel(), shots=1, seed=0
        ).results[0][0]
        noisy = simulate_ensemble(
            _system(), noise=NoiseModel(amplitude_sigma=0.2), shots=1, seed=12345
        ).results[0][0]
        assert not np.allclose(clean.final_state, noisy.final_state)
