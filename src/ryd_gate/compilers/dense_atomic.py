"""Compiler for dense two-atom systems (7-level CZ gates, 3-level analog).

Extracts the Hamiltonian assembly logic that was hardcoded in solve_gate()
into a declarative HamiltonianIR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryd_gate.compilers.base import Compiler
from ryd_gate.compilers.ir import HamiltonianIR, HamiltonianTerm

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import AtomicSystem
    from ryd_gate.protocols.base import Protocol


class DenseAtomicCompiler(Compiler):
    """Compiles AtomicSystem + Protocol into a dense HamiltonianIR.

    Maps protocol drive channels to AtomicSystem's precomputed matrices:
        "drive_420"       -> system.tq_ham_420
        "drive_420_dag"   -> system.tq_ham_420_conj
        "lightshift_zero" -> system.tq_ham_lightshift_zero

    Static terms: H_const + H_1013 + H_1013_conj + any protocol additions.
    """

    def __init__(self, amplitude_scale: float = 1.0) -> None:
        self.amplitude_scale = amplitude_scale

    def compile(
        self,
        system: AtomicSystem,
        protocol: Protocol,
        params: dict,
    ) -> HamiltonianIR:
        dim = system.n_levels ** system.n_atoms

        # Static terms (time-independent)
        static_terms = [
            HamiltonianTerm("H_const", system.tq_ham_const, 1.0),
            HamiltonianTerm("H_1013", system.tq_ham_1013, 1.0),
            HamiltonianTerm("H_1013_conj", system.tq_ham_1013_conj, 1.0),
        ]

        # Protocol-provided static additions (e.g. 784nm pinning)
        ham_additions = protocol.get_ham_const_additions(
            n_atoms=system.n_atoms, n_levels=system.n_levels,
        )
        if ham_additions is not None:
            static_terms[0] = HamiltonianTerm(
                "H_const",
                system.tq_ham_const + ham_additions,
                1.0,
            )

        # Channel-to-operator mapping
        channel_ops = {
            "drive_420": system.tq_ham_420,
            "drive_420_dag": system.tq_ham_420_conj,
            "lightshift_zero": system.tq_ham_lightshift_zero,
        }

        amp_scale = self.amplitude_scale

        def _make_coeff_fn(protocol, params, channel, amp_scale):
            """Create a closure that returns the coefficient for a channel at time t."""
            def coeff_fn(t):
                coeffs = protocol.get_drive_coefficients(t, params)
                return amp_scale * coeffs[channel] if channel != "lightshift_zero" else amp_scale ** 2 * coeffs[channel]
            return coeff_fn

        drive_terms = []
        for channel, op in channel_ops.items():
            coeff_fn = _make_coeff_fn(protocol, params, channel, amp_scale)
            drive_terms.append(HamiltonianTerm(channel, op, coeff_fn))

        return HamiltonianIR(
            static_terms=static_terms,
            drive_terms=drive_terms,
            dim=dim,
            is_sparse=False,
            metadata={
                "t_gate": params["t_gate"],
                "amplitude_scale": amp_scale,
                "param_set": system.param_set,
            },
        )
