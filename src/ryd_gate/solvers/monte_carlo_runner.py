"""New-architecture Monte Carlo runner using Compiler + SolverBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.compilers.dense_atomic import DenseAtomicCompiler
from ryd_gate.solvers.base import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.compilers.base import Compiler
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
        protocol: Protocol,
        x: list[float],
        backend: SolverBackend | None = None,
    ) -> None:
        self.model = model
        self.protocol = protocol
        self.x = x

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
                # Build detuning perturbation operator
                basis = self.model.basis
                n = basis.local_dim
                ryd_levels = [l for l in basis.local_levels if l.startswith("r")]
                occ_op = np.zeros((basis.total_dim, basis.total_dim), dtype=np.complex128)
                for level in ryd_levels:
                    idx = basis.level_index(level)
                    from ryd_gate.core.operators import build_occ_operator

                    occ_op += build_occ_operator(idx, n)
                ham_delta = delta_err * occ_op

            # Compile with noise
            compiler = DenseAtomicCompiler(amplitude_scale=amp_scale)
            ir = compiler.compile(system, self.protocol, params)

            # Add detuning perturbation to static terms if needed
            if ham_delta is not None:
                from ryd_gate.compilers.ir import HamiltonianTerm

                ir.static_terms.append(HamiltonianTerm("detuning_noise", ham_delta, 1.0))

            # Evolve all initial states
            shot_results = []
            for psi0 in initial_states:
                result = self.backend.evolve(ir, psi0, params["t_gate"])
                shot_results.append(result)
            all_results.append(shot_results)

        return all_results
