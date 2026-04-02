"""Compiler for sparse N-atom lattice systems (2-level).

Maps SweepProtocol channels to LatticeSystem's precomputed sparse operators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryd_gate.compilers.base import Compiler
from ryd_gate.compilers.ir import HamiltonianIR, HamiltonianTerm

if TYPE_CHECKING:
    from ryd_gate.core.atomic_system import LatticeSystem
    from ryd_gate.protocols.base import Protocol


class SparseLatticeCompiler(Compiler):
    """Compiles LatticeSystem + SweepProtocol into a sparse HamiltonianIR.

    Maps protocol drive channels to LatticeSystem's sparse operators:
        "global_X" -> system.sum_X
        "global_n" -> system.sum_n

    Static terms: H_vdw + per-site pinning detunings.
    """

    def compile(
        self,
        system: LatticeSystem,
        protocol: Protocol,
        params: dict,
    ) -> HamiltonianIR:
        dim = 2 ** system.N

        # Static: VdW interaction + local pinning detunings
        H_static = system.H_vdw
        if hasattr(protocol, "get_pin_deltas"):
            import numpy as np
            pin_deltas = protocol.get_pin_deltas(system.N)
            for i, d_i in enumerate(pin_deltas):
                if abs(d_i) > 1e-15:
                    H_static = H_static - d_i * system.n_list[i]

        static_terms = [
            HamiltonianTerm("H_static", H_static, 1.0),
        ]

        # Channel-to-operator mapping
        channel_ops = {
            "global_X": system.sum_X,
            "global_n": system.sum_n,
        }

        def _make_coeff_fn(protocol, params, channel):
            def coeff_fn(t):
                coeffs = protocol.get_drive_coefficients(t, params)
                return coeffs[channel]
            return coeff_fn

        drive_terms = []
        for channel, op in channel_ops.items():
            coeff_fn = _make_coeff_fn(protocol, params, channel)
            drive_terms.append(HamiltonianTerm(channel, op, coeff_fn))

        return HamiltonianIR(
            static_terms=static_terms,
            drive_terms=drive_terms,
            dim=dim,
            is_sparse=True,
            metadata={
                "t_gate": params["t_gate"],
                "param_set": system.param_set,
                "N": system.N,
            },
        )
