"""Tests for the CZGateSimulator class in ideal_cz.py.

All tests reuse session-scoped fixtures from conftest.py to avoid
redundant ARC library initialisation (the main bottleneck).
"""

import numpy as np
import pytest

# Standard test parameters (mirrored from conftest.py fixtures)
X_TO = [0.1, 1.0, 0.0, 0.0, 0.0, 1.0]
X_AR = [1.0, 0.1, 0.0, 0.05, 0.0, 0.0, 1.0, 0.0]
X_TO_OUR_DARK = [
    -0.6989301339711643, 1.0296229082590798, 0.3759232324550267,
    1.5710180991068543, 1.4454279613697887, 1.3406239758422793,
]
X_TO_DECAY = [
    -0.9509172186259588, 1.105272315809505, 0.383911389220584,
    1.2848721417313045, 1.3035218398648376, 1.246566016566724,
]

# ==================================================================
# TESTS FOR INITIALIZATION
# ==================================================================


class TestCZGateSimulatorInit:
    """Tests for CZGateSimulator instantiation and parameter handling.

    These tests validate constructor behaviour and therefore reuse
    session-scoped fixtures instead of creating fresh instances.
    """

    def test_instantiation_our_TO(self, sim_to_our):
        """CZGateSimulator should instantiate with 'our' params and TO strategy."""
        assert sim_to_our.param_set == "our"
        assert sim_to_our.strategy == "TO"
        assert sim_to_our.ryd_level == 70

    def test_instantiation_our_AR(self, sim_ar_our):
        """CZGateSimulator should instantiate with 'our' params and AR strategy."""
        assert sim_ar_our.param_set == "our"
        assert sim_ar_our.strategy == "AR"

    def test_instantiation_lukin_TO(self, sim_to_lukin):
        """CZGateSimulator should instantiate with 'lukin' params."""
        assert sim_to_lukin.param_set == "lukin"
        assert sim_to_lukin.ryd_level == 53

    def test_instantiation_lukin_AR(self, sim_ar_our):
        """AR strategy attribute check (reuses session fixture)."""
        assert sim_ar_our.strategy == "AR"

    def test_instantiation_with_decay(self, sim_to_our_decay):
        """CZGateSimulator with decay enabled should have imaginary Hamiltonian parts."""
        assert np.any(np.imag(sim_to_our_decay.tq_ham_const) != 0)

    def test_instantiation_without_decay(self, sim_to_our):
        """CZGateSimulator with decay disabled should have real diagonal."""
        diagonal = np.diag(sim_to_our.tq_ham_const)
        assert np.allclose(np.imag(diagonal), 0)

    def test_invalid_param_set(self):
        """CZGateSimulator should raise ValueError for invalid param_set."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="Unknown parameter set"):
            CZGateSimulator(param_set="invalid")

    def test_invalid_strategy_in_optimize(self, sim_to_our):
        """optimize() should raise ValueError for invalid strategy."""
        old = sim_to_our.strategy
        sim_to_our.strategy = "INVALID"
        try:
            with pytest.raises(ValueError, match="Unknown strategy"):
                sim_to_our.optimize(X_TO)
        finally:
            sim_to_our.strategy = old

    def test_blackman_flag_true(self, sim_to_our_blackman):
        """CZGateSimulator should respect blackmanflag=True."""
        assert sim_to_our_blackman.blackmanflag is True

    def test_blackman_flag_default_true(self, sim_to_our):
        """CZGateSimulator should default to blackmanflag=True."""
        assert sim_to_our.blackmanflag is True


# ==================================================================
# TESTS FOR HAMILTONIAN CONSTRUCTION
# ==================================================================


class TestHamiltonianConstruction:
    """Tests for Hamiltonian matrix construction."""

    def test_ham_const_shape(self, sim_to_our):
        """Constant Hamiltonian should have shape (49, 49)."""
        assert sim_to_our.tq_ham_const.shape == (49, 49)

    def test_ham_420_shape(self, sim_to_our):
        """420nm Hamiltonian should have shape (49, 49)."""
        assert sim_to_our.tq_ham_420.shape == (49, 49)

    def test_ham_1013_shape(self, sim_to_our):
        """1013nm Hamiltonian should have shape (49, 49)."""
        assert sim_to_our.tq_ham_1013.shape == (49, 49)

    def test_ham_const_hermitian_no_decay(self, sim_to_our):
        """Constant Hamiltonian should be Hermitian when all decay flags are off."""
        assert np.allclose(sim_to_our.tq_ham_const, sim_to_our.tq_ham_const.conj().T)

    def test_ham_420_structure(self, sim_to_our):
        """420nm Hamiltonian should couple ground to intermediate states."""
        assert not np.allclose(sim_to_our.tq_ham_420, 0)

    def test_occ_operator_shape(self, sim_to_our):
        """Occupation operator should have shape (49, 49)."""
        occ_op = sim_to_our._occ_operator(0)
        assert occ_op.shape == (49, 49)

    def test_occ_operator_trace(self, sim_to_our):
        """Occupation operator trace should be 2*7=14 (both atoms, 7 levels each)."""
        occ_op = sim_to_our._occ_operator(0)
        assert np.isclose(np.trace(occ_op), 14)


# ==================================================================
# TESTS FOR FIDELITY CALCULATION
# ==================================================================


@pytest.mark.slow
class TestFidelityCalculation:
    """Tests for average fidelity calculation.

    Uses session-scoped fixtures to share ODE results across tests.
    """

    def test_fidelity_TO_returns_float(self, fid_to):
        """gate_fidelity with TO strategy should return a float."""
        assert isinstance(fid_to, (float, np.floating))

    def test_fidelity_AR_returns_float(self, fid_ar):
        """gate_fidelity with AR strategy should return a float."""
        assert isinstance(fid_ar, (float, np.floating))

    def test_fidelity_bounded_TO(self, fid_to):
        """Infidelity should be between 0 and 1 for TO strategy."""
        assert 0 <= fid_to <= 1

    def test_fidelity_bounded_AR(self, fid_ar):
        """Infidelity should be between 0 and 1 for AR strategy."""
        assert 0 <= fid_ar <= 1

    def test_fidelity_lukin_params(self, sim_to_lukin):
        """Fidelity calculation should work with lukin params."""
        infid = sim_to_lukin.gate_fidelity(X_TO)
        assert 0 <= infid <= 1


# ==================================================================
# TESTS FOR STATE EVOLUTION
# ==================================================================


@pytest.mark.slow
class TestStateEvolution:
    """Tests for quantum state evolution methods.

    Uses session-scoped fixtures to share the TO evolution result.
    """

    def test_get_gate_result_TO_shape(self, evo_to):
        """_get_gate_result_TO should return array of shape (49, 1000)."""
        assert evo_to.shape == (49, 1000)

    def test_get_gate_result_AR_shape(self, sim_ar_our):
        """_get_gate_result_AR should return array of shape (49, 1000)."""
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim_ar_our._get_gate_result_AR(
            omega=sim_ar_our.rabi_eff,
            phase_amp1=0.1,
            phase_init1=0.0,
            phase_amp2=0.05,
            phase_init2=0.0,
            delta=0.0,
            t_gate=sim_ar_our.time_scale,
            state_mat=ini_state,
            t_eval=np.linspace(0, sim_ar_our.time_scale, 1000),
        )
        assert result.shape == (49, 1000)

    def test_state_normalization_preserved(self, evo_to):
        """State norm should be preserved during evolution (no decay)."""
        for t_idx in [0, 250, 500, 750, 999]:
            norm = np.linalg.norm(evo_to[:, t_idx])
            assert np.isclose(norm, 1.0, rtol=1e-6)


# ==================================================================
# TESTS FOR DIAGNOSTIC METHODS
# ==================================================================


@pytest.mark.slow
class TestDiagnosticMethods:
    """Tests for diagnostic run methods.

    Uses session-scoped fixtures to share ODE results across tests.
    """

    def test_diagnose_run_TO_returns_three_arrays(self, diag_to_11):
        """diagnose_run with TO should return list of 3 arrays."""
        assert len(diag_to_11) == 3
        assert all(isinstance(arr, np.ndarray) for arr in diag_to_11)

    def test_diagnose_run_AR_returns_three_arrays(self, sim_ar_our):
        """diagnose_run with AR should return list of 3 arrays."""
        result = sim_ar_our.diagnose_run(X_AR, "11")
        assert len(result) == 3
        assert all(isinstance(arr, np.ndarray) for arr in result)

    def test_diagnose_run_array_shapes(self, diag_to_11):
        """diagnose_run arrays should have length 1000."""
        mid_pop, ryd_pop, ryd_garb_pop = diag_to_11
        assert len(mid_pop) == 1000
        assert len(ryd_pop) == 1000
        assert len(ryd_garb_pop) == 1000

    def test_diagnose_run_populations_positive(self, diag_to_11):
        """All population values should be non-negative."""
        mid_pop, ryd_pop, ryd_garb_pop = diag_to_11
        assert np.all(mid_pop >= 0)
        assert np.all(ryd_pop >= 0)
        assert np.all(ryd_garb_pop >= 0)

    def test_diagnose_run_invalid_initial_state(self, sim_to_our):
        """diagnose_run should raise ValueError for invalid initial state."""
        with pytest.raises(ValueError, match="Unsupported initial state"):
            sim_to_our.diagnose_run(X_TO, "invalid")

    def test_diagnose_run_all_initial_states(self, sim_to_our):
        """diagnose_run should work for all valid initial states."""
        for initial in ["00", "01", "10", "11"]:
            result = sim_to_our.diagnose_run(X_TO, initial)
            assert len(result) == 3


# ==================================================================
# TESTS FOR STORED-PARAMETER WORKFLOW
# ==================================================================


@pytest.mark.slow
class TestStoredParameterWorkflow:
    """Tests for the setup_protocol / stored-parameter API."""

    def test_setup_protocol_TO_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong TO parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        with pytest.raises(ValueError, match="6 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_setup_protocol_AR_wrong_length_raises(self):
        """setup_protocol should raise ValueError for wrong AR parameter count."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="AR")
        with pytest.raises(ValueError, match="8 elements"):
            sim.setup_protocol([1, 2, 3])

    def test_gate_fidelity_no_params_raises(self, sim_to_our):
        """gate_fidelity() with no stored params should raise ValueError."""
        # Ensure x_initial is None
        old = sim_to_our.x_initial
        sim_to_our.x_initial = None
        try:
            with pytest.raises(ValueError, match="No pulse parameters"):
                sim_to_our.gate_fidelity()
        finally:
            sim_to_our.x_initial = old

    def test_gate_fidelity_uses_stored_params(self, sim_to_our):
        """gate_fidelity() should use stored params and match explicit call."""
        old = sim_to_our.x_initial
        try:
            sim_to_our.setup_protocol(X_TO)
            infid_explicit = sim_to_our.gate_fidelity(X_TO)
            infid_stored = sim_to_our.gate_fidelity()
            assert infid_explicit == infid_stored
        finally:
            sim_to_our.x_initial = old

    def test_diagnose_run_stored_params(self, sim_to_our):
        """diagnose_run with stored params should work via keyword arg."""
        old = sim_to_our.x_initial
        try:
            sim_to_our.setup_protocol(X_TO)
            result = sim_to_our.diagnose_run(initial_state="11")
            assert len(result) == 3
        finally:
            sim_to_our.x_initial = old

    def test_diagnose_run_AR_sss_states(self, sim_ar_our):
        """diagnose_run with AR strategy should support SSS states."""
        for i in [0, 11]:
            result = sim_ar_our.diagnose_run(X_AR, f"SSS-{i}")
            assert len(result) == 3

    def test_gate_fidelity_explicit_does_not_mutate(self, sim_to_our):
        """Passing explicit x to gate_fidelity should not change x_initial."""
        old = sim_to_our.x_initial
        try:
            x_stored = list(X_TO)
            x_other = [0.2, 0.5, 0.1, 0.1, 0.1, 0.8]
            sim_to_our.setup_protocol(x_stored)
            sim_to_our.gate_fidelity(x_other)
            assert sim_to_our.x_initial == x_stored
        finally:
            sim_to_our.x_initial = old


# ==================================================================
# ==================================================================
# TESTS FOR MONTE CARLO SIMULATION
# ==================================================================


class TestMonteCarloSimulation:
    """Tests for quasi-static Monte Carlo simulation capabilities."""

    def test_monte_carlo_result_dataclass(self):
        """MonteCarloResult dataclass should be importable and have correct fields."""
        from ryd_gate.ideal_cz import MonteCarloResult

        result = MonteCarloResult(
            mean_fidelity=0.99,
            std_fidelity=0.01,
            mean_infidelity=0.01,
            std_infidelity=0.01,
            n_shots=100,
            fidelities=np.array([0.99] * 100),
        )
        assert result.mean_fidelity == 0.99
        assert result.n_shots == 100

    def test_constructor_requires_sigma_detuning_when_dephasing_enabled(self):
        """Constructor should raise if enable_rydberg_dephasing=True without sigma_detuning."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="sigma_detuning"):
            CZGateSimulator(
                param_set="our", strategy="TO",
                enable_rydberg_dephasing=True,
            )

    def test_constructor_requires_sigma_pos_xyz_when_position_enabled(self):
        """Constructor should raise if enable_position_error=True without sigma_pos_xyz."""
        from ryd_gate.ideal_cz import CZGateSimulator

        with pytest.raises(ValueError, match="sigma_pos_xyz"):
            CZGateSimulator(
                param_set="our", strategy="TO",
                enable_position_error=True,
            )

    @pytest.mark.slow
    def test_gate_fidelity_returns_tuple_when_mc_enabled(self):
        """gate_fidelity should return (mean, std) tuple when MC flags are on."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            enable_rydberg_dephasing=True,
            sigma_detuning=130e3,
            n_mc_shots=2,
            mc_seed=42,
        )
        result = sim.gate_fidelity(X_TO)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mean_inf, std_inf = result
        assert isinstance(mean_inf, float)
        assert isinstance(std_inf, float)
        assert mean_inf >= 0
        assert std_inf >= 0

    @pytest.mark.slow
    def test_3d_position_model_distances(self):
        """MC with 3D position model should produce positive distances."""
        from ryd_gate.ideal_cz import CZGateSimulator

        small_sigma = (0.05e-6, 0.05e-6, 0.05e-6)
        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            enable_position_error=True,
            sigma_pos_xyz=small_sigma,
        )
        result = sim.run_monte_carlo_simulation(
            X_TO, n_shots=2,
            sigma_pos_xyz=small_sigma,
            seed=42,
        )
        assert result.distance_samples is not None
        assert np.all(result.distance_samples > 0)
        assert np.all(result.distance_samples > 1.0)
        assert np.all(result.distance_samples < 10.0)

    def test_build_vdw_unit_operator(self, sim_to_our):
        """_build_vdw_unit_operator should return correct shape and structure."""
        op = sim_to_our._build_vdw_unit_operator()
        assert op.shape == (49, 49)
        assert np.all(np.diag(op) >= 0)



# ==================================================================
# TESTS FOR JAX-ACCELERATED MONTE CARLO
# ==================================================================


@pytest.mark.slow
class TestMonteCarloJax:
    """Tests for GPU-accelerated JAX Monte Carlo simulation."""

    def test_jax_mc_not_implemented_ar(self, sim_ar_our):
        """JAX MC should raise NotImplementedError for AR strategy."""
        with pytest.raises(NotImplementedError):
            sim_ar_our.run_monte_carlo_jax(X_AR, n_shots=5, seed=0)


# ==================================================================
# TESTS FOR INDEPENDENT ERROR SOURCE FLAGS
# ==================================================================


@pytest.mark.slow
class TestIndependentErrorFlags:
    """Tests for the independent enable_* error source flags."""

    def test_enable_rydberg_decay_only(self):
        """Rydberg diagonal should have imaginary part, mid-state should not."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_rydberg_decay=True)
        diag = np.diag(sim.tq_ham_const)
        ryd_idx_5 = 5 * 7 + 0
        ryd_idx_5b = 0 * 7 + 5
        assert np.imag(diag[ryd_idx_5]) != 0
        assert np.imag(diag[ryd_idx_5b]) != 0
        mid_idx = 2 * 7 + 0
        assert np.imag(diag[mid_idx]) == 0

    def test_enable_intermediate_decay_only(self):
        """Intermediate diagonal should have imaginary part, Rydberg should not."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_intermediate_decay=True)
        diag = np.diag(sim.tq_ham_const)
        mid_idx = 2 * 7 + 0
        assert np.imag(diag[mid_idx]) != 0
        ryd_idx = 5 * 7 + 0
        assert np.imag(diag[ryd_idx]) == 0

    def test_polarization_leakage_disabled(self, sim_to_our):
        """With leakage disabled, state 6 should be far-detuned."""
        assert sim_to_our.ryd_zeeman_shift > 2 * np.pi * 1e9
        assert np.allclose(sim_to_our.tq_ham_const, sim_to_our.tq_ham_const.conj().T)

    def test_polarization_leakage_enabled(self):
        """With leakage enabled, garbage Rabi freqs should be nonzero."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", enable_polarization_leakage=True)
        assert sim.rabi_420_garbage != 0.0
        assert sim.rabi_1013_garbage != 0.0

    def test_zero_state_no_coupling_in_h420(self, sim_to_our, sim_to_lukin):
        """420nm Hamiltonian should NOT directly couple |0⟩ to |eᵢ⟩."""
        for sim in (sim_to_our, sim_to_lukin):
            ham = sim.tq_ham_420
            for ei in (2, 3, 4):
                assert ham[ei, 0] == 0, f"|0⟩→|e{ei-1}⟩ should be zero in H_420"

    def test_zero_state_lightshift_scattering_gated_by_flag(self):
        """Scattering on |0⟩ should be gated by enable_intermediate_decay."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim_off = CZGateSimulator(param_set="our", enable_intermediate_decay=False)
        sim_on = CZGateSimulator(param_set="our", enable_intermediate_decay=True)
        diag_off = np.diag(sim_off.tq_ham_lightshift_zero)
        diag_on = np.diag(sim_on.tq_ham_lightshift_zero)

        assert np.any(diag_off.real != 0), "Should have nonzero real light shifts"
        assert np.allclose(diag_off.real, diag_on.real), "Real shifts should match"
        assert np.allclose(diag_off.imag, 0), "Flag off should have no scattering"
        s0_on = diag_on[0 * 7 + 1]
        assert s0_on.real < 0, "Expected negative |0⟩ light shift"
        assert s0_on.imag < 0, "Expected negative imaginary (scattering loss) on |0⟩"

    def test_all_flags_off_hermitian_hamiltonian(self, sim_to_our):
        """All flags off should produce a purely real-diagonal, Hermitian Hamiltonian."""
        diagonal = np.diag(sim_to_our.tq_ham_const)
        assert np.allclose(np.imag(diagonal), 0)
        assert np.allclose(sim_to_our.tq_ham_const, sim_to_our.tq_ham_const.conj().T)

    def test_all_flags_off_perfect_gate(self, sim_to_our_blackman):
        """All flags off should give near-perfect gate (includes always-on light shift)."""
        infidelity = sim_to_our_blackman.gate_fidelity(X_TO_OUR_DARK)
        assert infidelity < 1e-6, f"Infidelity {infidelity} too large for all-flags-off gate"

    def test_all_flags_off_norm_preserved(self, sim_to_our):
        """All flags off should preserve state normalization (unitary evolution)."""
        ini_state = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim_to_our._get_gate_result_TO(
            phase_amp=X_TO[0],
            omega=X_TO[1] * sim_to_our.rabi_eff,
            phase_init=X_TO[2],
            delta=X_TO[3] * sim_to_our.rabi_eff,
            t_gate=X_TO[5] * sim_to_our.time_scale,
            state_mat=ini_state,
        )
        final_norm = np.linalg.norm(result)
        assert np.isclose(final_norm, 1.0, rtol=1e-6)

    def test_dephasing_flag_gates_mc(self):
        """MC with enable_rydberg_dephasing=False should ignore sigma_detuning."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", enable_rydberg_dephasing=False
        )
        result = sim.run_monte_carlo_simulation(
            X_TO, n_shots=2, sigma_detuning=130e3, seed=42
        )
        assert result.detuning_samples is None
        assert result.std_fidelity == pytest.approx(0.0, abs=1e-10)

    def test_position_flag_gates_mc(self):
        """MC with enable_position_error=False should ignore sigma_pos_xyz."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", enable_position_error=False
        )
        result = sim.run_monte_carlo_simulation(
            X_TO, n_shots=2,
            sigma_pos_xyz=(70e-6, 70e-6, 170e-6),
            seed=42,
        )
        assert result.distance_samples is None
        assert result.std_fidelity == pytest.approx(0.0, abs=1e-10)


@pytest.mark.slow
class TestBranchingRatios:
    """Tests for branching ratio calculations and error budget."""

    def test_rydberg_branching_ratios_sum_to_one(self, sim_to_our):
        """Rydberg branching ratios should sum to 1."""
        br = sim_to_our._ryd_branch
        total = br["to_0"] + br["to_1"] + br["to_L0"] + br["to_L1"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_mid_branching_ratios_sum_to_one(self, sim_to_our):
        """Mid-state branching ratios should sum to 1 for each F."""
        for F in (1, 2, 3):
            br = sim_to_our._mid_branch[F]
            total = br["to_0"] + br["to_1"] + br["to_L0"] + br["to_L1"]
            assert total == pytest.approx(1.0, abs=1e-6), f"F={F} ratios sum to {total}"

    def test_error_budget_non_negative(self, budget_decay):
        """Error budget values should all be non-negative."""
        for source, errors in budget_decay.items():
            for etype, val in errors.items():
                assert val >= 0, f"{source}/{etype} = {val} is negative"

    def test_error_budget_xyz_al_lg_sum(self, budget_decay):
        """XYZ + AL + LG should approximately equal total for each source."""
        for source, errors in budget_decay.items():
            component_sum = errors["XYZ"] + errors["AL"] + errors["LG"]
            assert component_sum == pytest.approx(errors["total"], rel=1e-4), \
                f"{source}: XYZ+AL+LG={component_sum} != total={errors['total']}"

    def test_population_evolution_shapes(self, pops_01):
        """Population evolution should return correct shapes and valid values."""
        assert pops_01["t_list"].shape == (1000,)
        for key in ["e1", "e2", "e3", "ryd", "ryd_garb"]:
            assert pops_01[key].shape == (1000,), f"{key} has wrong shape"
            assert np.all(pops_01[key] >= -1e-10), f"{key} has negative values"
            assert np.all(pops_01[key] <= 1.0 + 1e-10), f"{key} exceeds 1"

    def test_error_budget_polarization_leakage_channel(self, sim_to_our_decay):
        """Polarization leakage channel should be non-negative and self-consistent."""
        budget = sim_to_our_decay.error_budget(X_TO_DECAY)
        assert "polarization_leakage" in budget
        pol = budget["polarization_leakage"]
        for etype, val in pol.items():
            assert val >= -1e-15, f"polarization_leakage/{etype} = {val} is negative"
        component_sum = pol["XYZ"] + pol["AL"] + pol["LG"]
        assert component_sum == pytest.approx(pol["total"], rel=1e-4)

    def test_error_budget_polarization_leakage_zero_when_disabled(self):
        """Polarization leakage channel should be ~0 when flag is off."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO", blackmanflag=True,
            enable_rydberg_decay=True, enable_intermediate_decay=True,
            enable_polarization_leakage=False,
        )
        budget = sim.error_budget(X_TO_DECAY)
        for etype in ["XYZ", "AL", "LG", "total"]:
            assert budget["polarization_leakage"][etype] == pytest.approx(
                0.0, abs=1e-12,
            ), f"polarization_leakage/{etype} should be ~0 when disabled"


# ==================================================================
# TESTS FOR MONTE CARLO BRANCHING
# ==================================================================


@pytest.mark.slow
class TestMonteCarloWithBranching:
    """Tests for MC branching decomposition (XYZ/AL/LG/phase).

    Uses a shared class-level MC result (2 shots) to avoid redundant ODE
    solves (~11 s each).
    """

    @pytest.fixture(scope="class")
    def mc_branching_result(self):
        """Shared MC result with branching for structural checks."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=1,
            enable_rydberg_dephasing=True,
            sigma_detuning=130e3,
        )
        return sim.run_monte_carlo_simulation(
            X_TO_OUR_DARK, n_shots=2, sigma_detuning=130e3,
            seed=42, compute_branching=True,
        )

    def test_mc_no_branching_by_default(self):
        """MC without compute_branching should return None for all branch fields."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=1,
            enable_rydberg_dephasing=True,
            sigma_detuning=130e3,
        )
        result = sim.run_monte_carlo_simulation(
            X_TO_OUR_DARK, n_shots=2, sigma_detuning=130e3, seed=42,
        )
        assert result.branch_XYZ is None
        assert result.mean_branch_XYZ is None

    def test_branching_sum_equals_infidelity(self, mc_branching_result):
        """XYZ + AL + LG + phase should equal total infidelity per shot."""
        result = mc_branching_result
        infidelities = 1 - result.fidelities
        branch_sum = (
            result.branch_XYZ + result.branch_AL
            + result.branch_LG + result.branch_phase
        )
        np.testing.assert_allclose(branch_sum, infidelities, rtol=1e-6)

    def test_branching_non_negative(self, mc_branching_result):
        """All branching components should be non-negative."""
        result = mc_branching_result
        assert np.all(result.branch_XYZ >= -1e-12)
        assert np.all(result.branch_AL >= -1e-12)
        assert np.all(result.branch_LG >= -1e-12)
        assert np.all(result.branch_phase >= -1e-6)

    def test_fidelity_avg_residuals(self, sim_to_our_blackman):
        """return_residuals=True should match normal call and have small residuals."""
        infid_new, residuals = sim_to_our_blackman._fidelity_avg(
            X_TO_OUR_DARK, return_residuals=True,
        )
        assert infid_new < 1e-4
        assert isinstance(residuals, dict)
        assert set(residuals.keys()) == {"e1", "e2", "e3", "ryd", "ryd_garb"}
        for key, val in residuals.items():
            assert val < 1e-4, f"Residual {key} = {val} too large for optimized gate"

    def test_save_load_roundtrip(self, tmp_path):
        """Saving and loading should reproduce the same MonteCarloResult."""
        from ryd_gate.ideal_cz import MonteCarloResult

        fids = np.array([0.995, 0.993, 0.997])
        bx = np.array([1e-3, 2e-3, 0.5e-3])
        ba = np.array([0.5e-3, 0.3e-3, 0.4e-3])
        bl = np.array([0.2e-3, 0.1e-3, 0.15e-3])
        bp = 1 - fids - bx - ba - bl
        result = MonteCarloResult(
            mean_fidelity=float(np.mean(fids)),
            std_fidelity=float(np.std(fids)),
            mean_infidelity=float(np.mean(1 - fids)),
            std_infidelity=float(np.std(1 - fids)),
            n_shots=3,
            fidelities=fids,
            detuning_samples=np.array([1e3, -2e3, 0.5e3]),
            branch_XYZ=bx, branch_AL=ba, branch_LG=bl, branch_phase=bp,
            mean_branch_XYZ=float(np.mean(bx)),
            std_branch_XYZ=float(np.std(bx)),
            mean_branch_AL=float(np.mean(ba)),
            std_branch_AL=float(np.std(ba)),
            mean_branch_LG=float(np.mean(bl)),
            std_branch_LG=float(np.std(bl)),
            mean_branch_phase=float(np.mean(bp)),
            std_branch_phase=float(np.std(bp)),
        )
        filepath = str(tmp_path / "mc_test.txt")
        result.save_to_file(filepath)
        loaded = MonteCarloResult.load_from_file(filepath)

        assert loaded.n_shots == result.n_shots
        np.testing.assert_allclose(loaded.fidelities, result.fidelities, rtol=1e-10)
        np.testing.assert_allclose(loaded.branch_XYZ, result.branch_XYZ, rtol=1e-10)
        np.testing.assert_allclose(loaded.branch_AL, result.branch_AL, rtol=1e-10)
        np.testing.assert_allclose(loaded.branch_LG, result.branch_LG, rtol=1e-10)
        np.testing.assert_allclose(loaded.branch_phase, result.branch_phase, rtol=1e-10)

    def test_residuals_to_branching_non_negative(self, sim_to_our_decay):
        """_residuals_to_branching should return non-negative values."""
        _, residuals = sim_to_our_decay._fidelity_avg(
            X_TO_OUR_DARK, return_residuals=True,
        )
        branching = sim_to_our_decay._residuals_to_branching(residuals)
        assert branching["XYZ"] >= 0
        assert branching["AL"] >= 0
        assert branching["LG"] >= 0


# ==================================================================
# COVERAGE-BOOSTING TESTS
# ==================================================================


@pytest.mark.slow
class TestStateInfidelity:
    """Tests for state_infidelity method (lines 1162-1237)."""

    def test_state_infidelity_string_label_00(self, sim_to_our):
        """state_infidelity should accept string label '00'."""
        infid = sim_to_our.state_infidelity("00", X_TO)
        assert 0 <= infid <= 1

    def test_state_infidelity_string_label_01(self, sim_to_our):
        """state_infidelity should accept string label '01'."""
        infid = sim_to_our.state_infidelity("01", X_TO)
        assert 0 <= infid <= 1

    def test_state_infidelity_string_label_11(self, sim_to_our):
        """state_infidelity should accept string label '11'."""
        infid = sim_to_our.state_infidelity("11", X_TO)
        assert 0 <= infid <= 1

    def test_state_infidelity_sss_label(self, sim_to_our):
        """state_infidelity should accept SSS-0 label."""
        infid = sim_to_our.state_infidelity("SSS-0", X_TO)
        assert 0 <= infid <= 1

    def test_state_infidelity_ndarray_input(self, sim_to_our):
        """state_infidelity should accept ndarray initial state."""
        ini = np.kron(
            [1 + 0j, 0, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        infid = sim_to_our.state_infidelity(ini, X_TO)
        assert 0 <= infid <= 1

    def test_state_infidelity_ar_strategy(self, sim_ar_our):
        """state_infidelity should work with AR strategy (theta = x[-1])."""
        infid = sim_ar_our.state_infidelity("00", X_AR)
        assert 0 <= infid <= 1

    def test_state_infidelity_invalid_label(self, sim_to_our):
        """state_infidelity should raise ValueError for invalid label."""
        with pytest.raises(ValueError, match="Unsupported initial state"):
            sim_to_our.state_infidelity("bad_label", X_TO)


@pytest.mark.slow
class TestGetGateResult:
    """Tests for get_gate_result public method (lines 1239-1290)."""

    def test_get_gate_result_to_final_state(self, sim_to_our):
        """get_gate_result TO with no t_eval returns shape (49,)."""
        ini = np.kron(
            [1 + 0j, 0, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]
        )
        result = sim_to_our.get_gate_result(ini, X_TO)
        assert result.shape == (49,)

    def test_get_gate_result_to_with_t_eval(self, sim_to_our):
        """get_gate_result TO with t_eval returns shape (49, N)."""
        ini = np.kron(
            [1 + 0j, 0, 0, 0, 0, 0, 0], [1 + 0j, 0, 0, 0, 0, 0, 0]
        )
        t_eval = np.linspace(0, sim_to_our.time_scale, 50)
        result = sim_to_our.get_gate_result(ini, X_TO, t_eval=t_eval)
        assert result.shape == (49, 50)

    def test_get_gate_result_ar_final_state(self, sim_ar_our):
        """get_gate_result AR with no t_eval returns shape (49,)."""
        ini = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        result = sim_ar_our.get_gate_result(ini, X_AR)
        assert result.shape == (49,)

    def test_get_gate_result_ar_with_t_eval(self, sim_ar_our):
        """get_gate_result AR with t_eval returns shape (49, N)."""
        ini = np.kron(
            [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
        )
        t_eval = np.linspace(0, sim_ar_our.time_scale, 50)
        result = sim_ar_our.get_gate_result(ini, X_AR, t_eval=t_eval)
        assert result.shape == (49, 50)


@pytest.mark.slow
class TestFidelityTypes:
    """Tests for fid_type='bell' and fid_type='sss' paths."""

    def test_fidelity_bell(self, sim_to_our):
        """gate_fidelity with fid_type='bell' should return bounded float."""
        infid = sim_to_our.gate_fidelity(X_TO, fid_type="bell")
        assert isinstance(infid, (float, np.floating))
        assert 0 <= infid <= 1

    def test_fidelity_sss(self, sim_to_our):
        """gate_fidelity with fid_type='sss' should return bounded float."""
        infid = sim_to_our.gate_fidelity(X_TO, fid_type="sss")
        assert isinstance(infid, (float, np.floating))
        assert 0 <= infid <= 1

    def test_ar_non_average_fid_type_supported(self, sim_ar_our):
        """AR strategy now supports all fid_types via unified implementation."""
        result = sim_ar_our._gate_infidelity_single(X_AR, fid_type="bell")
        assert isinstance(result, float)
        assert result >= 0

    def test_return_residuals_non_average_raises(self, sim_to_our):
        """return_residuals with non-'average' fid_type should raise ValueError."""
        with pytest.raises(ValueError, match="return_residuals is only supported"):
            sim_to_our._gate_infidelity_single(
                X_TO, fid_type="bell", return_residuals=True,
            )


@pytest.mark.slow
class TestARFidelityResiduals:
    """Tests for _avg_fidelity_AR with return_residuals=True."""

    def test_ar_return_residuals(self, sim_ar_our):
        """_avg_fidelity_AR with return_residuals should return tuple."""
        infid, residuals = sim_ar_our._avg_fidelity_AR(
            X_AR, return_residuals=True,
        )
        assert isinstance(infid, (float, np.floating))
        assert 0 <= infid <= 1
        assert isinstance(residuals, dict)
        assert set(residuals.keys()) == {"e1", "e2", "e3", "ryd", "ryd_garb"}
        for val in residuals.values():
            assert val >= 0

    def test_ar_gate_infidelity_single_average(self, sim_ar_our):
        """_gate_infidelity_single AR with fid_type='average' should work."""
        infid = sim_ar_our._gate_infidelity_single(X_AR, fid_type="average")
        assert 0 <= infid <= 1


class TestNominalDistance:
    """Tests for _get_nominal_distance method."""

    def test_nominal_distance_our(self, sim_to_our):
        """'our' param_set should return 3.0 μm."""
        assert sim_to_our._get_nominal_distance() == 3.0

    def test_nominal_distance_lukin(self, sim_to_lukin):
        """'lukin' param_set should return 3.0 μm."""
        assert sim_to_lukin._get_nominal_distance() == 3.0


class TestDiagnoseEdgeCases:
    """Tests for diagnose_run / diagnose_plot edge cases."""

    def test_diagnose_run_none_initial_state_raises(self, sim_to_our):
        """diagnose_run with initial_state=None should raise ValueError."""
        with pytest.raises(ValueError, match="initial_state is required"):
            sim_to_our.diagnose_run(X_TO, initial_state=None)

    def test_diagnose_plot_none_initial_state_raises(self, sim_to_our):
        """diagnose_plot with initial_state=None should raise ValueError."""
        with pytest.raises(ValueError, match="initial_state is required"):
            sim_to_our.diagnose_plot(X_TO, initial_state=None)


class TestMonteCarloResultSerialization:
    """Tests for MonteCarloResult save/load with distance_samples."""

    def test_save_load_with_distance_samples(self, tmp_path):
        """Save/load roundtrip should preserve distance_samples."""
        from ryd_gate.ideal_cz import MonteCarloResult

        fids = np.array([0.99, 0.98, 0.97])
        dists = np.array([3.01, 2.99, 3.02])
        result = MonteCarloResult(
            mean_fidelity=float(np.mean(fids)),
            std_fidelity=float(np.std(fids)),
            mean_infidelity=float(np.mean(1 - fids)),
            std_infidelity=float(np.std(1 - fids)),
            n_shots=3,
            fidelities=fids,
            distance_samples=dists,
        )
        filepath = str(tmp_path / "mc_dist.txt")
        result.save_to_file(filepath)
        loaded = MonteCarloResult.load_from_file(filepath)

        assert loaded.n_shots == 3
        np.testing.assert_allclose(loaded.fidelities, fids, rtol=1e-10)
        np.testing.assert_allclose(loaded.distance_samples, dists, rtol=1e-10)

    def test_save_load_without_branching(self, tmp_path):
        """Save/load roundtrip without branching data should work."""
        from ryd_gate.ideal_cz import MonteCarloResult

        fids = np.array([0.99, 0.98])
        result = MonteCarloResult(
            mean_fidelity=float(np.mean(fids)),
            std_fidelity=float(np.std(fids)),
            mean_infidelity=float(np.mean(1 - fids)),
            std_infidelity=float(np.std(1 - fids)),
            n_shots=2,
            fidelities=fids,
        )
        filepath = str(tmp_path / "mc_nobranch.txt")
        result.save_to_file(filepath)
        loaded = MonteCarloResult.load_from_file(filepath)

        assert loaded.n_shots == 2
        assert loaded.branch_XYZ is None
        assert loaded.distance_samples is None
        np.testing.assert_allclose(loaded.fidelities, fids, rtol=1e-10)


class TestBuildSSSStateMap:
    """Tests for _build_sss_state_map static method."""

    def test_sss_state_map_keys(self):
        """State map should contain 4 computational + 12 SSS states."""
        from ryd_gate.ideal_cz import CZGateSimulator

        smap = CZGateSimulator._build_sss_state_map()
        assert "00" in smap
        assert "01" in smap
        assert "10" in smap
        assert "11" in smap
        for i in range(12):
            assert f"SSS-{i}" in smap
        assert len(smap) == 16

    def test_sss_states_normalized(self):
        """All SSS states should be normalized."""
        from ryd_gate.ideal_cz import CZGateSimulator

        smap = CZGateSimulator._build_sss_state_map()
        for label, state in smap.items():
            assert state.shape == (49,), f"{label} has wrong shape"
            assert np.isclose(np.linalg.norm(state), 1.0, rtol=1e-10), \
                f"{label} is not normalized"


class TestOptimizeDispatch:
    """Tests for optimize() method dispatch."""

    def test_optimize_invalid_strategy_raises(self):
        """optimize() with invalid strategy should raise ValueError."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(param_set="our", strategy="TO")
        sim.strategy = "INVALID"
        with pytest.raises(ValueError, match="Unknown strategy"):
            sim.optimize(x_initial=[0.1, 1.0, 0.0, 0.0, 0.0, 1.0])

    def test_unknown_fid_type_raises(self, sim_to_our):
        """_gate_infidelity_single with unknown fid_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown fid_type"):
            sim_to_our._gate_infidelity_single(X_TO, fid_type="unknown")


class TestSetupProtocolAR:
    """Tests for _setup_protocol_AR print path."""

    def test_setup_protocol_ar_stores_params(self, sim_ar_our):
        """_setup_protocol_AR should store x_initial."""
        sim_ar_our.setup_protocol(X_AR)
        assert sim_ar_our.x_initial == X_AR


class TestPlotBlochDispatch:
    """Tests for plot_bloch dispatch."""

    def test_plot_bloch_ar_prints_message(self, sim_ar_our, capsys):
        """plot_bloch on AR strategy should print 'only implemented for TO'."""
        sim_ar_our.plot_bloch(X_AR)
        captured = capsys.readouterr()
        assert "only implemented" in captured.out


class TestDiagnosePlotDispatch:
    """Tests for diagnose_plot strategy dispatch."""

    def test_invalid_strategy_at_init_raises(self):
        """Invalid strategy at construction should raise ValueError."""
        from ryd_gate.ideal_cz import CZGateSimulator
        with pytest.raises(ValueError, match="Unknown strategy"):
            CZGateSimulator(param_set="our", strategy="INVALID")


@pytest.mark.slow
class TestMCProgressPrint:
    """Tests for MC progress indicator (lines 1380-1384)."""

    def test_mc_5_shots_prints_progress(self, capsys):
        """MC with n_shots >= 5 should print progress."""
        from ryd_gate.ideal_cz import CZGateSimulator

        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            enable_rydberg_dephasing=True,
            sigma_detuning=130e3,
        )
        sim.run_monte_carlo_simulation(
            X_TO, n_shots=5, sigma_detuning=130e3, seed=42,
        )
        captured = capsys.readouterr()
        assert "MC shot" in captured.out
