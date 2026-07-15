"""Tests for the shared TN lowering seam (compile_tn_terms).

The old ``TNLatticeSpec`` / snake-ordering machinery is gone; the TN backends now
consume a ``RydbergSystem`` directly. These tests cover the shared lowering that
both the MPS and PEPS engines build their Hamiltonians from.
"""

import numpy as np
import pytest

from ryd_gate import RydbergSystem, level_structure
from ryd_gate.backends.tn_common.compiler import (
    TNSegment,
    compile_tn_terms,
    plan_segments,
)
from ryd_gate.lattice import Register
from ryd_gate.protocols.digital_analog import DigitalAnalogProtocol
from ryd_gate.protocols.sweep import SweepProtocol

TWO_PI = 2 * np.pi


def _sweep_system(preset="1r", n=4, spacing=8.0, cutoff=9.0, local=None):
    proto = SweepProtocol(
        t_gate_s=0.3e-6,
        omega_half_rad_s=lambda t: TWO_PI * 2.0e6,
        detuning_rad_s=lambda t: TWO_PI * 1.0e6,
        local_detuning_rad_s=local,
    )
    return RydbergSystem(level_structure=level_structure(preset),
                         register=Register.chain(n, spacing_um=spacing),
                         protocol=proto, interaction_cutoff_um=cutoff)


class TestCompileTNTerms:
    def test_levels_and_channels_1r(self):
        terms = compile_tn_terms(_sweep_system("1r"))
        assert terms.levels == ("1", "r")
        assert terms.rydberg_levels == ("r",)
        channels = {(c.ket, c.bra) for c in terms.channels}
        assert ("r", "1") in channels  # E[r,1] drive
        assert ("r", "r") in channels  # E[r,r] detuning

    def test_nn_cutoff_limits_pairs(self):
        terms = compile_tn_terms(_sweep_system("1r", n=4, spacing=8.0, cutoff=9.0))
        assert len(terms.pairs) == 3  # nearest-neighbour chain bonds only
        for i, j, V in terms.pairs:
            assert i < j and V > 0.0

    def test_local_hamiltonians_are_hermitian(self):
        terms = compile_tn_terms(_sweep_system("01r"))
        h = terms.local_hamiltonians(0.1e-6)
        assert h.shape == (4, 3, 3)
        for i in range(4):
            np.testing.assert_allclose(h[i], h[i].conj().T, atol=1e-9)

    def test_per_site_local_detuning_is_site_resolved(self):
        addr = [0.0, TWO_PI * 1e6, -TWO_PI * 2e6, TWO_PI * 3e6]
        terms = compile_tn_terms(_sweep_system("1r", local=lambda t, i: addr[i]))
        h = terms.local_hamiltonians(0.1e-6)
        r = terms.levels.index("r")
        diag_r = np.array([h[i, r, r].real for i in range(4)])
        assert len(set(np.round(diag_r, 6))) == 4  # four distinct site energies

    def test_digital_analog_complex_channels(self):
        proto = DigitalAnalogProtocol(
            t_gate_s=0.3e-6,
            coupling_r1_rad_s=lambda t: 1.0 + 0.5j,
            coupling_10_rad_s=lambda t: 0.3j,
        )
        system = RydbergSystem(level_structure=level_structure("01r"),
                               register=Register.chain(2, spacing_um=8.0), protocol=proto)
        terms = compile_tn_terms(system)
        h = terms.local_hamiltonians(0.05e-6)
        r, one, zero = (terms.levels.index(x) for x in ("r", "1", "0"))
        # E[r,1] = 1+0.5j on the (r,1) element; conjugate on (1,r)
        np.testing.assert_allclose(h[0, r, one], 1.0 + 0.5j)
        np.testing.assert_allclose(h[0, one, r], 1.0 - 0.5j)
        np.testing.assert_allclose(h[0, one, zero], 0.3j)

    def test_non_tn_preset_rejected(self):
        from ryd_gate.protocols.gate_cz import CZProtocol
        from ryd_gate.protocols.pulses import blackman_pulse

        proto = CZProtocol(
            t_gate_s=0.2e-6, intermediate_detuning_rad_s=TWO_PI * 7.8e9,
            omega_420_max_rad_s=TWO_PI * 100e6, omega_1013_max_rad_s=TWO_PI * 100e6,
            envelope_420=lambda t: blackman_pulse(t, 20e-9, 0.2e-6),
            phase_420_rad=lambda t: 0.0,
        )
        system = RydbergSystem(
            level_structure=level_structure("rb87_7_mp", magnetic_field_G=20.0),
            register=Register.chain(2, spacing_um=5.0), protocol=proto)
        with pytest.raises(ValueError, match="TN-capable"):
            compile_tn_terms(system)


class TestPlanSegments:
    def test_anchor_is_exact_step_boundary(self):
        record_at_start, segs = plan_segments(1.0, np.array([0.5, 1.0]), 0.3)
        assert not record_at_start
        assert all(isinstance(s, TNSegment) for s in segs)
        assert segs[0].t1 == 0.5 and segs[-1].t1 == 1.0
        # 0.5 is not a multiple of 0.3: ceil(0.5/0.3)=2 equal steps hit it exactly
        assert segs[0].n_sub == 2
        np.testing.assert_allclose(segs[0].n_sub * segs[0].dt_sub, 0.5)

    def test_endpoint_only_appends_unrecorded_t_gate(self):
        record_at_start, segs = plan_segments(1.0, np.array([1.0]), 0.4)
        assert not record_at_start
        assert segs[-1].t1 == 1.0 and segs[-1].record is True

    def test_t0_recorded_when_zero_requested(self):
        record_at_start, segs = plan_segments(1.0, np.array([0.0, 1.0]), 0.5)
        assert record_at_start
