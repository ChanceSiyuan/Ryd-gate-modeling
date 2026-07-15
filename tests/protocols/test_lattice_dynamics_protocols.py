import numpy as np

from ryd_gate import (
    InteractionSpec,
    RydbergSystem,
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
    level_structure,
)
from ryd_gate.backends.exact.compiler import _compile_exact_ir
from ryd_gate.ir import compile_hamiltonian_ir
from ryd_gate.lattice import Register
from ryd_gate.protocols.base import Protocol
from ryd_gate.protocols.lattice_dynamics import tfim_to_rydberg_controls


def _nn_square_system(L=2):
    return RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.rectangle(L, L, spacing_um=1.0),
        interaction=InteractionSpec(C6=4.0, mode="nn"),
    )


def test_tfim_to_rydberg_controls_uniform_2x2_nn():
    system = _nn_square_system(2)

    controls = tfim_to_rydberg_controls(system, hx=0.5, hz=0.0)

    assert np.isclose(controls.Omega, 1.0)
    assert np.isclose(controls.Delta, 4.0)
    assert controls.pin_deltas == {}
    np.testing.assert_allclose(controls.interaction_shifts, np.full(4, 2.0))


def test_tfim_to_rydberg_controls_compensates_open_boundary_shifts():
    system = _nn_square_system(3)

    controls = tfim_to_rydberg_controls(system, hx=1.0, hz=0.25)
    effective_hz = controls.interaction_shifts - 0.5 * controls.delta_profile

    np.testing.assert_allclose(effective_hz, np.full(system.N, 0.25))
    assert controls.pin_deltas


def test_tfim_quench_protocol_emits_existing_lattice_channels():
    system = _nn_square_system(2).with_protocol(TFIMQuenchProtocol(hx=0.75, hz=0.0, t_gate=1.25))

    params = system.protocol._resolve(system)
    coeffs = system.protocol.get_drive_coefficients(0.5, params)

    assert params["t_gate"] == 1.25
    assert np.isclose(coeffs["E[r,1]"], 0.75)
    assert np.isclose(coeffs["E[r,r]"], -4.0)


def test_tfim_anneal_protocol_piecewise_schedule():
    system = _nn_square_system(2)
    proto = TFIMAnnealProtocol(
        hx_peak=3.0,
        hz_initial=-8.0,
        hz_final=0.0,
        t_rise=1.5,
        t_sweep=1.5,
        t_fall=1.5,
    )

    params = proto._resolve(system)

    assert np.isclose(proto.hx_at(0.0), 0.0)
    assert np.isclose(proto.hx_at(1.5), 3.0)
    assert np.isclose(proto.hz_at(1.5), -8.0)
    assert np.isclose(proto.hz_at(3.0), 0.0)
    assert np.isclose(params["t_gate"], 4.5)
    assert np.isclose(proto.get_drive_coefficients(1.5, params)["E[r,1]"], 3.0)


def test_exact_compiler_accepts_site_dependent_detuning_channels():
    class SiteDetuningProtocol(Protocol):
        @property
        def required_channels(self):
            return frozenset({"E[r,r]_0", "E[r,r]_1"})

        def drive_channels(self, system):
            return self.required_channels

        def _resolve(self, system):
            return {"t_gate": 1.0}

        def get_drive_coefficients(self, t, ctx):
            return {"E[r,r]_0": -1.0, "E[r,r]_1": -2.0}

    system = RydbergSystem(
        level_structure=level_structure("1r"),
        register=Register.rectangle(1, 2, spacing_um=1.0),
        interaction=InteractionSpec(C6=0.0, mode="nn"),
        protocol=SiteDetuningProtocol(),
    )

    ham = compile_hamiltonian_ir(system)
    ir = _compile_exact_ir(ham)

    assert {term.name for term in ir.drive_terms} == {"E[r,r]_0", "E[r,r]_1"}
