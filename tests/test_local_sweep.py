"""Tests for the 3-level local addressing sweep and Rydberg blockade physics.

Verifies:
1. Rydberg blockade under resonant Rabi driving -- the blockaded oscillation
   frequency is sqrt(2) x the single-atom frequency, and P_r maxima match
   theoretical bounds (~0.5 blockaded, ~1.0 free).
2. Adiabatic chirp sweep -- final populations match blockade predictions
   (symmetric |gr>+|rg> without pinning, deterministic |gr> with pinning).
"""

import numpy as np
import pytest
from scipy.constants import pi
from scipy.signal import find_peaks

from ryd_gate import simulate
from ryd_gate.analysis.observable_metrics import measure_trajectory
from ryd_gate.core.atomic_system import create_analog_system
from ryd_gate.core.models.analog_3level import Analog3LevelModel
from ryd_gate.protocols.sweep import SweepProtocol

N = 3

# Pinning parameters: |local_detuning| >> Omega_eff for clean pinning.
LOCAL_DETUNING = -2 * pi * 50e6
LOCAL_SCATTER = 150.0

# Sweep range: |delta| >> Omega_eff for adiabatic passage.
DELTA_START = -2 * pi * 40e6
DELTA_END = 2 * pi * 40e6
T_GATE = 1.5e-6


# -- Fixtures --

@pytest.fixture(scope="module")
def analog_model():
    return Analog3LevelModel.from_defaults(detuning_sign=1)


@pytest.fixture(scope="module")
def initial_state(analog_model):
    return analog_model.observables.get("pop_gg").operator @ np.zeros(9, dtype=complex) \
        if False else \
        np.zeros(9, dtype=complex)


@pytest.fixture(scope="module", autouse=True)
def _set_initial(initial_state):
    initial_state[0] = 1.0  # |gg>


# -- Helpers --

def _oscillation_period(t, signal):
    """Estimate oscillation period from peak spacing."""
    peaks, _ = find_peaks(signal, height=0.1)
    if len(peaks) < 2:
        return None
    spacings = np.diff(t[peaks])
    return float(np.median(spacings))


def _gg_state():
    """Return |gg> state vector for 3-level system."""
    psi = np.zeros(9, dtype=complex)
    psi[0] = 1.0
    return psi


# -- Tests: Rabi dynamics (resonant driving) --

@pytest.mark.slow
class TestRabiBlockade:
    """Resonant Rabi driving: blockade vs free oscillation."""

    @pytest.fixture(scope="class")
    def rabi_results(self, analog_model):
        system = analog_model.system
        n_cycles = 5
        t_rabi = n_cycles * system.time_scale
        x_rabi = [0.0, 0.0, t_rabi / system.time_scale]
        n_points = 500
        t_eval = np.linspace(0, t_rabi, n_points)

        proto_free = SweepProtocol()
        proto_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                    scatter_rate=LOCAL_SCATTER)
        psi0 = _gg_state()
        obs_names = ["pop_A_r", "pop_B_r"]

        r1 = simulate(system, proto_free, x_rabi, psi0, t_eval=t_eval)
        obs1 = measure_trajectory(analog_model, r1.states, obs_names)

        r2 = simulate(system, proto_addr, x_rabi, psi0, t_eval=t_eval)
        obs2 = measure_trajectory(analog_model, r2.states, obs_names)

        return {
            "t1": r1.times, "prA1": obs1["pop_A_r"], "prB1": obs1["pop_B_r"],
            "t2": r2.times, "prA2": obs2["pop_A_r"], "prB2": obs2["pop_B_r"],
            "time_scale": system.time_scale,
        }

    def test_blockade_atoms_overlap(self, rabi_results):
        """Both atoms should oscillate identically under blockade."""
        np.testing.assert_allclose(
            rabi_results["prA1"], rabi_results["prB1"], atol=1e-6,
        )

    def test_blockade_max_pr_bounded(self, rabi_results):
        """P_r max ~ 0.5 under blockade (two-atom sharing)."""
        max_pr = np.max(rabi_results["prA1"])
        assert max_pr == pytest.approx(0.5, abs=0.05)

    def test_free_max_pr_near_one(self, rabi_results):
        """Atom B reaches P_r ~ 1.0 when Atom A is pinned (no blockade)."""
        max_prB = np.max(rabi_results["prB2"])
        assert max_prB == pytest.approx(1.0, abs=0.05)

    def test_pinned_atom_stays_ground(self, rabi_results):
        """Atom A P_r stays near 0 when addressed."""
        max_prA = np.max(rabi_results["prA2"])
        assert max_prA < 0.05

    def test_sqrt2_frequency_ratio(self, rabi_results):
        """Blockaded frequency should be sqrt(2) x the free frequency."""
        T_blockade = _oscillation_period(
            rabi_results["t1"], rabi_results["prA1"],
        )
        T_free = _oscillation_period(
            rabi_results["t2"], rabi_results["prB2"],
        )
        assert T_blockade is not None and T_free is not None
        ratio = T_free / T_blockade
        assert ratio == pytest.approx(np.sqrt(2), abs=0.1)


# -- Tests: Adiabatic sweep final populations --

@pytest.mark.slow
class TestAdiabaticSweep:
    """Adiabatic chirp sweep: final state population distributions."""

    @pytest.fixture(scope="class")
    def sweep_results(self, analog_model):
        system = analog_model.system
        x_sweep = [
            DELTA_START / system.rabi_eff,
            DELTA_END / system.rabi_eff,
            T_GATE / system.time_scale,
        ]
        proto_free = SweepProtocol()
        proto_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                    scatter_rate=LOCAL_SCATTER)
        psi0 = _gg_state()
        pop_keys = ["pop_gg", "pop_gr", "pop_rg", "pop_rr"]

        r_free = simulate(system, proto_free, x_sweep, psi0)
        pops_free = {k.replace("pop_", ""): analog_model.observables.measure(k, r_free.psi_final)
                     for k in pop_keys}

        r_addr = simulate(system, proto_addr, x_sweep, psi0)
        pops_addr = {k.replace("pop_", ""): analog_model.observables.measure(k, r_addr.psi_final)
                     for k in pop_keys}

        return {"free": pops_free, "addr": pops_addr}

    def test_free_symmetric_entanglement(self, sweep_results):
        """|gr> and |rg> should be approximately equal (symmetric blockade)."""
        pops = sweep_results["free"]
        assert pops["gr"] == pytest.approx(pops["rg"], abs=0.05)

    def test_free_rydberg_transfer(self, sweep_results):
        """|gr> + |rg> should dominate (most population transferred)."""
        pops = sweep_results["free"]
        assert pops["gr"] + pops["rg"] > 0.85

    def test_free_no_double_rydberg(self, sweep_results):
        """|rr> ~ 0 -- blockade prevents double excitation."""
        assert sweep_results["free"]["rr"] < 0.01

    def test_addr_deterministic_gr(self, sweep_results):
        """|gr> dominates when Atom A is pinned."""
        assert sweep_results["addr"]["gr"] > 0.90

    def test_addr_no_rg(self, sweep_results):
        """|rg> ~ 0 -- Atom A never reaches |r>."""
        assert sweep_results["addr"]["rg"] < 0.05

    def test_addr_no_double_rydberg(self, sweep_results):
        """|rr> ~ 0 when one atom is pinned."""
        assert sweep_results["addr"]["rr"] < 0.01
