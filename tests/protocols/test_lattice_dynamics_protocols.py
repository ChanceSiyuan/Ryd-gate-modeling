import numpy as np

from ryd_gate import (
    InteractionSpec,
    RydbergSystem,
    TFIMAnnealProtocol,
    TFIMQuenchProtocol,
)
from ryd_gate.backends.exact.compiler import compile_expm_ir
from ryd_gate.ir import compile_hamiltonian_ir
from ryd_gate.lattice import Register
from ryd_gate.protocols import tfim_to_rydberg_controls
from ryd_gate.protocols.base import Protocol


def _nn_square_system(L=2):
    return RydbergSystem.set_atom_level("1r", Omega=1.0).set_atom_geom(
        Register.rectangle(L, L, spacing_um=1.0),
        interaction=InteractionSpec(C6=4.0, mode="nn"),
    ).build()


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

    params = system.unpack_params([])
    coeffs = system.protocol.get_drive_coefficients(0.5, params)

    assert params["t_gate"] == 1.25
    assert np.isclose(coeffs["global_X"], 0.75)
    assert np.isclose(coeffs["global_n"], -4.0)


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

    params = proto.unpack_params([], system)

    assert np.isclose(proto.hx_at(0.0), 0.0)
    assert np.isclose(proto.hx_at(1.5), 3.0)
    assert np.isclose(proto.hz_at(1.5), -8.0)
    assert np.isclose(proto.hz_at(3.0), 0.0)
    assert np.isclose(params["t_gate"], 4.5)
    assert np.isclose(proto.get_drive_coefficients(1.5, params)["global_X"], 3.0)


def test_exact_compiler_accepts_site_dependent_global_n_channels():
    class SiteDetuningProtocol(Protocol):
        @property
        def n_params(self):
            return 0

        @property
        def required_channels(self):
            return frozenset({"global_n_0", "global_n_1"})

        def drive_channels(self, system):
            return self.required_channels

        def validate_params(self, x):
            if x:
                raise ValueError

        def unpack_params(self, x, system):
            return {"t_gate": 1.0}

        def get_drive_coefficients(self, t, params):
            return {"global_n_0": -1.0, "global_n_1": -2.0}

    system = RydbergSystem.set_atom_level("1r").set_atom_geom(
        Register.rectangle(1, 2, spacing_um=1.0),
        interaction=InteractionSpec(C6=0.0, mode="nn"),
    ).set_protocol(SiteDetuningProtocol())

    ham = compile_hamiltonian_ir(system, system.unpack_params([]))
    ir = compile_expm_ir(ham)

    assert {term.name for term in ir.drive_terms} == {"global_n_0", "global_n_1"}
