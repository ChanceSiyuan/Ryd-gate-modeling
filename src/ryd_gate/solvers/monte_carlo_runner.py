"""New-architecture Monte Carlo runner using Compiler + SolverBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.solvers.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.core.system_model import SystemModel
    from ryd_gate.protocols.base import Protocol
    from ryd_gate.solvers.base import SolverBackend


class MonteCarloRunner:
    """Monte Carlo noise sampler using the new compiler + backend architecture.

    Unlike the legacy MonteCarloEngine, this runner:
    - Works with SystemModel (not raw AtomicSystem)
    - Uses Compiler + SolverBackend (not direct solve_gate calls)
    - Applies noise as perturbations via amplitude_scale in the compiler

    Supports quasi-static noise sources:
    - Detuning noise (T2* dephasing)
    - Amplitude noise (Rabi frequency fluctuations)
    """

    def __init__(
        self,
        model: SystemModel,
        x: list[float] | Protocol,
        x_legacy: list[float] | None = None,
        backend: SolverBackend | None = None,
    ) -> None:
        from ryd_gate.protocols.base import Protocol

        if isinstance(x, Protocol):
            import warnings as _w
            _w.warn(
                "MonteCarloRunner(model, protocol, x, ...) is deprecated. "
                "Pass a protocol-bound system: MonteCarloRunner(system, x, ...).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.model = model.with_protocol(x)
            self.x = x_legacy
        else:
            if model.protocol is None:
                raise ValueError(
                    "MonteCarloRunner expects a RydbergSystem with a bound protocol. "
                    "Use RydbergSystem(..., protocol=...) or .with_protocol(...)."
                )
            self.model = model
            self.x = x
        self.protocol = self.model.protocol

        # Default backend
        if backend is None:
            from ryd_gate.solvers.dense_ode import DenseODEBackend

            backend = DenseODEBackend()
        self.backend = backend

        # Noise parameters
        self._sigma_detuning_rad: float | None = None
        self._sigma_amplitude: float | None = None

    def setup_detuning_noise(self, sigma_detuning_hz: float) -> None:
        """Enable quasi-static detuning noise."""
        self._sigma_detuning_rad = 2 * np.pi * sigma_detuning_hz

    def setup_amplitude_noise(self, sigma_amplitude: float) -> None:
        """Enable Rabi frequency fluctuation noise (fractional)."""
        self._sigma_amplitude = sigma_amplitude

    def run_states(
        self,
        initial_states: list[Any],
        n_shots: int = 50,
        seed: int | None = None,
    ) -> list[list[EvolutionResult]]:
        """Evolve multiple initial states under noise.

        Returns list of [n_shots] lists, each containing EvolutionResults
        for each initial state.
        """
        rng = np.random.default_rng(seed)
        system = self.model.system
        params = self.protocol.unpack_params(self.x, system)

        all_results = []
        for shot in range(n_shots):
            # Sample noise
            amp_scale = 1.0
            ham_delta = None

            if self._sigma_amplitude is not None:
                amp_err = rng.normal(0, self._sigma_amplitude)
                amp_scale = 1.0 + amp_err

            if self._sigma_detuning_rad is not None:
                delta_err = rng.normal(0, self._sigma_detuning_rad)
                if self.model.blocks.has("sum_nr"):
                    occ_op = self.model.blocks.get("sum_nr")
                elif self.model.blocks.has("sum_n_r"):
                    occ_op = self.model.blocks.get("sum_n_r")
                else:
                    basis = self.model.basis
                    occ_op = np.zeros((basis.total_dim, basis.total_dim), dtype=np.complex128)
                    for level in [l for l in basis.local_levels if l.startswith("r")]:
                        block = f"sum_n_{level}"
                        if self.model.blocks.has(block):
                            occ_op = occ_op + self.model.blocks.get(block)
                ham_delta = delta_err * occ_op

            # Compile with noise
            ir = system.with_amplitude_scale(amp_scale).compile_ir(params)

            # Add detuning perturbation to static terms if needed
            if ham_delta is not None:
                from ryd_gate.solvers.ir import HamiltonianTerm

                ir.static_terms.append(HamiltonianTerm("detuning_noise", ham_delta, 1.0))

            # Evolve all initial states
            shot_results = []
            for psi0 in initial_states:
                result = self.backend.evolve(ir, psi0, params["t_gate"])
                shot_results.append(result)
            all_results.append(shot_results)

        return all_results
