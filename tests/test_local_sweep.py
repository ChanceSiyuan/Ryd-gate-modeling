"""Tests for the 3-level local addressing sweep and Rydberg blockade physics.

Verifies:
1. Rydberg blockade under resonant Rabi driving — the blockaded oscillation
   frequency is √2 × the single-atom frequency, and P_r maxima match
   theoretical bounds (≈0.5 blockaded, ≈1.0 free).
2. Adiabatic chirp sweep — final populations match blockade predictions
   (symmetric |gr⟩+|rg⟩ without pinning, deterministic |gr⟩ with pinning).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import pi
from scipy.signal import find_peaks

# Allow importing helpers from the script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_local_sweep import _evolve, _joint_populations, _rydberg_pops  # noqa: E402

from ryd_gate.core.atomic_system import build_sss_state_map, create_analog_system
from ryd_gate.protocols.sweep import SweepProtocol
from ryd_gate.solvers.schrodinger import solve_gate

N = 3

# Pinning parameters: |local_detuning| >> Ω_eff for clean pinning.
LOCAL_DETUNING = -2 * pi * 50e6
LOCAL_SCATTER = 150.0

# Sweep range: |δ| >> Ω_eff for adiabatic passage.
DELTA_START = -2 * pi * 40e6
DELTA_END = 2 * pi * 40e6
T_GATE = 1.5e-6


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def analog_system():
    system = create_analog_system(detuning_sign=1)
    initial_state = build_sss_state_map(n_levels=3)["00"]
    return system, initial_state


# ── Helpers ───────────────────────────────────────────────────────────

def _oscillation_period(t, signal):
    """Estimate oscillation period from peak spacing."""
    peaks, _ = find_peaks(signal, height=0.1)
    if len(peaks) < 2:
        return None
    spacings = np.diff(t[peaks])
    return float(np.median(spacings))


def _solve_final(system, protocol, x, initial_state, ham_const_override=None):
    """Evolve and return only the final state (no trajectory storage)."""
    return solve_gate(system, protocol, x, initial_state,
                      ham_const_override=ham_const_override)


# ── Tests: Rabi dynamics (resonant driving) ───────────────────────────

@pytest.mark.slow
class TestRabiBlockade:
    """Resonant Rabi driving: blockade vs free oscillation."""

    @pytest.fixture(scope="class")
    def rabi_results(self, analog_system):
        system, initial_state = analog_system
        n_cycles = 5
        t_rabi = n_cycles * system.time_scale
        x_rabi = [0.0, 0.0, t_rabi / system.time_scale]

        proto_free = SweepProtocol()
        proto_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                    scatter_rate=LOCAL_SCATTER)

        t1, psi1 = _evolve(system, proto_free, x_rabi, initial_state)
        prA1, prB1 = _rydberg_pops(psi1)

        ham_addr = system.tq_ham_const + proto_addr.get_ham_const_additions()
        t2, psi2 = _evolve(system, proto_addr, x_rabi, initial_state,
                           ham_const_override=ham_addr)
        prA2, prB2 = _rydberg_pops(psi2)

        return {
            "t1": t1, "prA1": prA1, "prB1": prB1,
            "t2": t2, "prA2": prA2, "prB2": prB2,
            "time_scale": system.time_scale,
        }

    def test_blockade_atoms_overlap(self, rabi_results):
        """Both atoms should oscillate identically under blockade."""
        np.testing.assert_allclose(
            rabi_results["prA1"], rabi_results["prB1"], atol=1e-6,
        )

    def test_blockade_max_pr_bounded(self, rabi_results):
        """P_r max ≈ 0.5 under blockade (two-atom sharing)."""
        max_pr = np.max(rabi_results["prA1"])
        assert max_pr == pytest.approx(0.5, abs=0.05)

    def test_free_max_pr_near_one(self, rabi_results):
        """Atom B reaches P_r ≈ 1.0 when Atom A is pinned (no blockade)."""
        max_prB = np.max(rabi_results["prB2"])
        assert max_prB == pytest.approx(1.0, abs=0.05)

    def test_pinned_atom_stays_ground(self, rabi_results):
        """Atom A P_r stays near 0 when addressed."""
        max_prA = np.max(rabi_results["prA2"])
        assert max_prA < 0.05

    def test_sqrt2_frequency_ratio(self, rabi_results):
        """Blockaded frequency should be √2 × the free frequency."""
        T_blockade = _oscillation_period(
            rabi_results["t1"], rabi_results["prA1"],
        )
        T_free = _oscillation_period(
            rabi_results["t2"], rabi_results["prB2"],
        )
        assert T_blockade is not None and T_free is not None
        ratio = T_free / T_blockade
        assert ratio == pytest.approx(np.sqrt(2), abs=0.1)


# ── Tests: Adiabatic sweep final populations ──────────────────────────

@pytest.mark.slow
class TestAdiabaticSweep:
    """Adiabatic chirp sweep: final state population distributions."""

    @pytest.fixture(scope="class")
    def sweep_results(self, analog_system):
        system, initial_state = analog_system
        x_sweep = [
            DELTA_START / system.rabi_eff,
            DELTA_END / system.rabi_eff,
            T_GATE / system.time_scale,
        ]
        proto_free = SweepProtocol()
        proto_addr = SweepProtocol(addressing={0: LOCAL_DETUNING},
                                    scatter_rate=LOCAL_SCATTER)

        psi_free = _solve_final(system, proto_free, x_sweep, initial_state)
        pops_free = _joint_populations(psi_free)

        ham_addr = system.tq_ham_const + proto_addr.get_ham_const_additions()
        psi_addr = _solve_final(system, proto_addr, x_sweep, initial_state,
                                ham_const_override=ham_addr)
        pops_addr = _joint_populations(psi_addr)

        return {"free": pops_free, "addr": pops_addr}

    def test_free_symmetric_entanglement(self, sweep_results):
        """|gr⟩ and |rg⟩ should be approximately equal (symmetric blockade)."""
        pops = sweep_results["free"]
        assert pops["gr"] == pytest.approx(pops["rg"], abs=0.05)

    def test_free_rydberg_transfer(self, sweep_results):
        """|gr⟩ + |rg⟩ should dominate (most population transferred)."""
        pops = sweep_results["free"]
        assert pops["gr"] + pops["rg"] > 0.85

    def test_free_no_double_rydberg(self, sweep_results):
        """|rr⟩ ≈ 0 — blockade prevents double excitation."""
        assert sweep_results["free"]["rr"] < 0.01

    def test_addr_deterministic_gr(self, sweep_results):
        """|gr⟩ dominates when Atom A is pinned."""
        assert sweep_results["addr"]["gr"] > 0.90

    def test_addr_no_rg(self, sweep_results):
        """|rg⟩ ≈ 0 — Atom A never reaches |r⟩."""
        assert sweep_results["addr"]["rg"] < 0.05

    def test_addr_no_double_rydberg(self, sweep_results):
        """|rr⟩ ≈ 0 when one atom is pinned."""
        assert sweep_results["addr"]["rr"] < 0.01
