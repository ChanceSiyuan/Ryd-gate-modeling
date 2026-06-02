"""Exact matrix compiler for state-vector sparse/dense backends.

This compiler materializes symbolic lattice operator specs into scipy sparse
matrices only when an exact state-vector simulation is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ryd_gate.model.operator_spec import is_operator_spec, materialize_sparse_operator
from ryd_gate.ir.matrix import HamiltonianIR, HamiltonianTerm


_HC_CHANNELS = frozenset({"global_X", "drive_R", "drive_hf"})
_HC_CHANNEL_PREFIXES = ("drive_R_", "drive_hf_", "global_X_")
_STATIC_BLOCKS = ("H_const", "H_vdw", "H_1013", "H_1013_conj")


@dataclass
class ExactSparseCompiler:
    """Compile a protocol-bound system into matrix Hamiltonian IR.

    Parameters
    ----------
    max_dim:
        Maximum Hilbert-space dimension allowed for exact state-vector
        matrix materialization. Use ``None`` to disable the guard.
    """

    max_dim: int | None = 2_000_000

    def compile(self, system, params: dict) -> HamiltonianIR:
        """Combine static blocks + protocol drives into a HamiltonianIR."""
        protocol = system._require_protocol()
        block_cache: dict[str, Any] = {}

        static_terms: list[HamiltonianTerm] = []
        for name in _STATIC_BLOCKS:
            if system.blocks.has(name):
                static_terms.append(
                    HamiltonianTerm(name, self.materialize_block(system, name, block_cache), 1.0)
                )

        overlay_dense, overlay_pin = self._static_overlays(system, params, block_cache)
        if overlay_dense is not None:
            if static_terms and static_terms[0].name == "H_const":
                static_terms[0] = HamiltonianTerm(
                    "H_const",
                    static_terms[0].operator + overlay_dense,
                    1.0,
                )
            else:
                static_terms.insert(0, HamiltonianTerm("H_const_addition", overlay_dense, 1.0))
        if overlay_pin is not None:
            static_terms.append(HamiltonianTerm("pinning", overlay_pin, 1.0))

        amplitude_scale = system.amplitude_scale
        drive_terms: list[HamiltonianTerm] = []
        drive_channel_set = protocol.drive_channels(system)
        for channel in drive_channel_set:
            block_name = self.block_name_for_drive_channel(system, channel)
            if block_name is None:
                continue

            def coeff_fn(t, channel=channel):
                coeffs = protocol.get_drive_coefficients(t, params)
                coeff = coeffs.get(channel, 0.0)
                if channel == "lightshift_zero":
                    return amplitude_scale**2 * coeff
                return amplitude_scale * coeff

            drive_terms.append(
                HamiltonianTerm(
                    channel,
                    self.materialize_block(system, block_name, block_cache),
                    coeff_fn,
                    add_hermitian_conjugate=self.channel_needs_hermitian_conjugate(channel),
                )
            )

        return HamiltonianIR(
            static_terms=static_terms,
            drive_terms=drive_terms,
            dim=system.basis.total_dim,
            is_sparse=system.is_sparse,
            metadata={
                "t_gate": params["t_gate"],
                "param_set": system.param_set,
                "n_sites": system.basis.n_sites,
                "local_dim": system.basis.local_dim,
            },
        )

    def materialize_block(self, system, name: str, cache: dict[str, Any] | None = None):
        """Return the exact matrix for a registered block."""
        cache = cache if cache is not None else {}
        if name in cache:
            return cache[name]
        operator = system.blocks.get(name)
        if is_operator_spec(operator):
            operator = materialize_sparse_operator(operator, system.basis, max_dim=self.max_dim)
        cache[name] = operator
        return operator

    def block_name_for_drive_channel(self, system, channel: str) -> str | None:
        """Map a protocol channel name to a registered block name."""
        if system.blocks.has(channel):
            return channel
        if channel.startswith("delta_R_"):
            block = f"n_r_{channel.rsplit('_', 1)[-1]}"
            return block if system.blocks.has(block) else None
        if channel.startswith("delta_hf_"):
            block = f"n_1_{channel.rsplit('_', 1)[-1]}"
            return block if system.blocks.has(block) else None
        return None

    @staticmethod
    def channel_needs_hermitian_conjugate(channel: str) -> bool:
        if channel in _HC_CHANNELS:
            return True
        return any(channel.startswith(prefix) for prefix in _HC_CHANNEL_PREFIXES)

    def _static_overlays(self, system, params: dict, block_cache: dict[str, Any]):
        """Return ``(dense_overlay, sparse_pinning)`` matrices from params dict."""
        n_sites = system.basis.n_sites
        n_levels = system.basis.local_dim
        ground_label = system.basis.local_levels[0]
        ryd_label = system.basis.local_levels[-1]

        dense_overlay = None
        sparse_pin = None
        pin_deltas = params.get("pin_deltas", {})
        scatter_rates = params.get("scatter_rates", {})
        static_overlays = params.get("static_overlays", [])

        for idx, delta in pin_deltas.items():
            if abs(delta) <= 1e-15 or idx >= n_sites:
                continue
            block_name = f"n_{ryd_label}_{idx}"
            if system.blocks.has(block_name):
                term = -delta * self.materialize_block(system, block_name, block_cache)
                sparse_pin = term if sparse_pin is None else sparse_pin + term
            else:
                from ryd_gate.model.operators import build_atom_projector

                proj = build_atom_projector(idx, n_levels - 1, n_sites, n_levels)
                term = -delta * proj
                dense_overlay = term if dense_overlay is None else dense_overlay + term

        for idx, rate in scatter_rates.items():
            if rate <= 0 or idx >= n_sites:
                continue
            block_name = f"n_{ground_label}_{idx}"
            if system.blocks.has(block_name):
                term = (-1j * rate / 2) * self.materialize_block(system, block_name, block_cache)
                sparse_pin = term if sparse_pin is None else sparse_pin + term
            else:
                from ryd_gate.model.operators import build_atom_projector

                proj = build_atom_projector(idx, 0, n_sites, n_levels)
                term = (-1j * rate / 2) * proj
                dense_overlay = term if dense_overlay is None else dense_overlay + term

        for block_name, coeff in static_overlays:
            if system.blocks.has(block_name):
                term = coeff * self.materialize_block(system, block_name, block_cache)
                if hasattr(term, "toarray"):
                    sparse_pin = term if sparse_pin is None else sparse_pin + term
                else:
                    dense_overlay = term if dense_overlay is None else dense_overlay + term

        return dense_overlay, sparse_pin

def compile_expm_ir(system, params: dict, *, max_dim: int | None = 2_000_000) -> HamiltonianIR:
    """Compile *system* into exact matrix IR for expm/ODE state-vector backends."""
    return ExactSparseCompiler(max_dim=max_dim).compile(system, params)
