"""Shared fixtures for the test suite.

Session-scoped fixtures avoid re-creating CZGateSimulator instances
(which involve ARC library calculations) across test classes.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def sim_to_our():
    """Session-scoped TO simulator with 'our' params (no decay)."""
    from ryd_gate.ideal_cz import CZGateSimulator
    return CZGateSimulator(param_set="our", strategy="TO")


@pytest.fixture(scope="session")
def sim_ar_our():
    """Session-scoped AR simulator with 'our' params (no decay)."""
    from ryd_gate.ideal_cz import CZGateSimulator
    return CZGateSimulator(param_set="our", strategy="AR")


@pytest.fixture(scope="session")
def sim_to_lukin():
    """Session-scoped TO simulator with 'lukin' params (no decay)."""
    from ryd_gate.ideal_cz import CZGateSimulator
    return CZGateSimulator(param_set="lukin", strategy="TO")


@pytest.fixture(scope="session")
def sim_to_our_blackman():
    """Session-scoped TO simulator with Blackman pulse (no decay)."""
    from ryd_gate.ideal_cz import CZGateSimulator
    return CZGateSimulator(param_set="our", strategy="TO", blackmanflag=True)


@pytest.fixture(scope="session")
def sim_to_our_decay():
    """Session-scoped TO simulator with all decay flags on."""
    from ryd_gate.ideal_cz import CZGateSimulator
    return CZGateSimulator(
        param_set="our", strategy="TO", blackmanflag=True,
        enable_rydberg_decay=True, enable_intermediate_decay=True,
        enable_polarization_leakage=True,
    )


# Standard test parameters
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


@pytest.fixture(scope="session")
def fid_to(sim_to_our):
    """Shared TO fidelity result."""
    return sim_to_our.gate_fidelity(X_TO)


@pytest.fixture(scope="session")
def fid_ar(sim_ar_our):
    """Shared AR fidelity result."""
    return sim_ar_our.gate_fidelity(X_AR)


@pytest.fixture(scope="session")
def evo_to(sim_to_our):
    """Shared TO evolution result for |11⟩ with t_eval."""
    ini_state = np.kron(
        [0, 1 + 0j, 0, 0, 0, 0, 0], [0, 1 + 0j, 0, 0, 0, 0, 0]
    )
    return sim_to_our._get_gate_result_TO(
        phase_amp=0.1,
        omega=sim_to_our.rabi_eff,
        phase_init=0.0,
        delta=0.0,
        t_gate=sim_to_our.time_scale,
        state_mat=ini_state,
        t_eval=np.linspace(0, sim_to_our.time_scale, 1000),
    )


@pytest.fixture(scope="session")
def diag_to_11(sim_to_our):
    """Shared TO diagnose_run result for '11'."""
    return sim_to_our.diagnose_run(X_TO, "11")


@pytest.fixture(scope="session")
def budget_decay(sim_to_our_decay):
    """Shared error budget with decay."""
    return sim_to_our_decay.error_budget(X_TO_DECAY)


@pytest.fixture(scope="session")
def pops_01(sim_to_our_decay):
    """Shared population evolution for '01' with decay."""
    return sim_to_our_decay._population_evolution(X_TO_DECAY, "01")
