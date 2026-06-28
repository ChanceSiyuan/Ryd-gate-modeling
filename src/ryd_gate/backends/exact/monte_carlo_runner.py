"""Monte Carlo runner using compiler + backend architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ryd_gate.ir import EvolutionResult

if TYPE_CHECKING:
    from ryd_gate.backends.exact.compiler import SolverBackend
    from ryd_gate.core.system import RydbergSystem


@dataclass
class MonteCarloResult:
    """Summary statistics and per-shot samples from Monte Carlo runs."""

    mean_fidelity: float
    std_fidelity: float
    mean_infidelity: float
    std_infidelity: float
    n_shots: int
    fidelities: np.ndarray
    detuning_samples: np.ndarray | None = None
    distance_samples: np.ndarray | None = None
    branch_XYZ: np.ndarray | None = None
    branch_AL: np.ndarray | None = None
    branch_LG: np.ndarray | None = None
    branch_phase: np.ndarray | None = None
    mean_branch_XYZ: float | None = None
    std_branch_XYZ: float | None = None
    mean_branch_AL: float | None = None
    std_branch_AL: float | None = None
    mean_branch_LG: float | None = None
    std_branch_LG: float | None = None
    mean_branch_phase: float | None = None
    std_branch_phase: float | None = None

    def save_to_file(self, filepath: str) -> None:
        import datetime

        columns = ["fidelity"]
        data = [self.fidelities]
        if self.detuning_samples is not None:
            columns.append("detuning_sample")
            data.append(self.detuning_samples)
        if self.distance_samples is not None:
            columns.append("distance_sample")
            data.append(self.distance_samples)
        if self.branch_XYZ is not None:
            columns.extend(["branch_XYZ", "branch_AL", "branch_LG", "branch_phase"])
            data.extend([self.branch_XYZ, self.branch_AL, self.branch_LG, self.branch_phase])

        matrix = np.column_stack(data)
        header = [
            f"MonteCarloResult saved {datetime.datetime.now().isoformat()}",
            f"n_shots = {self.n_shots}",
            f"mean_fidelity = {self.mean_fidelity:.12e}",
            f"std_fidelity = {self.std_fidelity:.12e}",
            f"mean_infidelity = {self.mean_infidelity:.12e}",
            f"std_infidelity = {self.std_infidelity:.12e}",
            "columns = " + ",".join(columns),
        ]
        np.savetxt(filepath, matrix, header="\n".join(header), fmt="%.12e")

    @classmethod
    def load_from_file(cls, filepath: str) -> "MonteCarloResult":
        header: dict[str, str] = {}
        columns: list[str] | None = None
        with open(filepath) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "columns":
                        columns = [part.strip() for part in value.split(",")]
                    else:
                        header[key] = value

        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if columns is None:
            columns = ["fidelity"]
        idx = {name: i for i, name in enumerate(columns)}
        fidelities = data[:, idx["fidelity"]]
        infidelities = 1.0 - fidelities
        kwargs: dict[str, Any] = {
            "mean_fidelity": float(header.get("mean_fidelity", np.mean(fidelities))),
            "std_fidelity": float(header.get("std_fidelity", np.std(fidelities))),
            "mean_infidelity": float(header.get("mean_infidelity", np.mean(infidelities))),
            "std_infidelity": float(header.get("std_infidelity", np.std(infidelities))),
            "n_shots": int(header.get("n_shots", len(fidelities))),
            "fidelities": fidelities,
        }
        if "detuning_sample" in idx:
            kwargs["detuning_samples"] = data[:, idx["detuning_sample"]]
        if "distance_sample" in idx:
            kwargs["distance_samples"] = data[:, idx["distance_sample"]]
        if "branch_XYZ" in idx:
            bx = data[:, idx["branch_XYZ"]]
            ba = data[:, idx["branch_AL"]]
            bl = data[:, idx["branch_LG"]]
            bp = data[:, idx["branch_phase"]]
            kwargs.update(
                branch_XYZ=bx,
                branch_AL=ba,
                branch_LG=bl,
                branch_phase=bp,
                mean_branch_XYZ=float(np.mean(bx)),
                std_branch_XYZ=float(np.std(bx)),
                mean_branch_AL=float(np.mean(ba)),
                std_branch_AL=float(np.std(ba)),
                mean_branch_LG=float(np.mean(bl)),
                std_branch_LG=float(np.std(bl)),
                mean_branch_phase=float(np.mean(bp)),
                std_branch_phase=float(np.std(bp)),
            )
        return cls(**kwargs)


class MonteCarloRunner:
    """Quasi-static noise sampler for protocol-bound Rydberg systems."""

    def __init__(
        self,
        system: RydbergSystem,
        x: list[float],
        backend: SolverBackend | None = None,
    ) -> None:
        if system.protocol is None:
            raise ValueError(
                "MonteCarloRunner expects a RydbergSystem with a bound protocol. "
                "Use RydbergSystem(..., protocol=...) or .with_protocol(...)."
            )
        self.system = system
        self.x = x
        self.protocol = system.protocol

        if backend is None:
            from ryd_gate.backends.exact.dense_ode import DenseODEBackend

            backend = DenseODEBackend()
        self.backend = backend

        self._sigma_detuning_rad: float | None = None
        self._sigma_amplitude: float | None = None
        self._sigma_local_rin: float | None = None
        self._sigma_pos_um: tuple[float, float, float] | None = None
        self._d_nominal_um: float | None = None
        self._c6_vdw: float | None = None

    def setup_detuning_noise(self, sigma_detuning_hz: float) -> None:
        """Enable quasi-static detuning noise."""
        self._sigma_detuning_rad = 2 * np.pi * sigma_detuning_hz

    def setup_amplitude_noise(self, sigma_amplitude: float) -> None:
        """Enable fractional Rabi-frequency noise."""
        self._sigma_amplitude = sigma_amplitude

    def setup_local_rin_noise(self, sigma_local_rin: float) -> None:
        """Enable fractional local-addressing intensity noise."""
        self._sigma_local_rin = sigma_local_rin

    def setup_position_noise(self, sigma_pos_xyz: tuple[float, float, float]) -> None:
        """Enable two-atom interatomic-distance fluctuation noise."""
        from ryd_gate.core.operators import get_nominal_distance

        self._sigma_pos_um = tuple(float(s) * 1e6 for s in sigma_pos_xyz)
        self._d_nominal_um = get_nominal_distance(self.system.param_set)
        self._c6_vdw = self.system.meta("v_ryd") * self._d_nominal_um**6

    def run_states(
        self,
        initial_states: list[Any],
        n_shots: int = 50,
        seed: int | None = None,
    ) -> list[list[EvolutionResult]]:
        """Evolve multiple initial states under sampled quasi-static noise."""
        from ryd_gate.backends.exact.compiler import ExactSparseCompiler
        from ryd_gate.ir import HamiltonianTerm

        rng = np.random.default_rng(seed)
        compiler = ExactSparseCompiler()

        all_results = []
        for _ in range(n_shots):
            system = self._sample_local_addressing(rng)
            system, params, _ = self._bind_shot(system)
            amp_scale, terms, _ = self._sample_terms(rng, compiler, system)

            ir = compiler.compile(system.with_amplitude_scale(amp_scale), params)
            ir.static_terms.extend(HamiltonianTerm(name, op, 1.0) for name, op in terms)

            all_results.append([
                self.backend.evolve(ir, psi0, params["t_gate"])
                for psi0 in initial_states
            ])

        return all_results

    def run_gate_fidelity(
        self,
        n_shots: int = 1000,
        seed: int | None = None,
        compute_branching: bool = False,
    ) -> MonteCarloResult:
        """Compute CZ average gate fidelity for each sampled noise shot."""
        from ryd_gate.analysis.gate_metrics import residuals_to_branching
        from ryd_gate.backends.exact.compiler import ExactSparseCompiler
        from ryd_gate.ir import HamiltonianTerm

        rng = np.random.default_rng(seed)
        compiler = ExactSparseCompiler()
        states = self._cz_basis_states()

        fidelities = np.zeros(n_shots)
        detuning_samples = np.zeros(n_shots) if self._sigma_detuning_rad is not None else None
        distance_samples = np.zeros(n_shots) if self._sigma_pos_um is not None else None

        if compute_branching:
            branch_xyz = np.zeros(n_shots)
            branch_al = np.zeros(n_shots)
            branch_lg = np.zeros(n_shots)
            branch_phase = np.zeros(n_shots)

        for shot in range(n_shots):
            self._print_progress("MC shot", shot, n_shots)
            system = self._sample_local_addressing(rng)
            system, params, theta = self._bind_shot(system)
            amp_scale, terms, samples = self._sample_terms(rng, compiler, system)
            if detuning_samples is not None and "detuning" in samples:
                detuning_samples[shot] = samples["detuning"]
            if distance_samples is not None and "distance" in samples:
                distance_samples[shot] = samples["distance"]

            ir = compiler.compile(system.with_amplitude_scale(amp_scale), params)
            ir.static_terms.extend(HamiltonianTerm(name, op, 1.0) for name, op in terms)

            psi00 = self.backend.evolve(ir, states["00"], params["t_gate"]).psi_final
            psi01 = self.backend.evolve(ir, states["01"], params["t_gate"]).psi_final
            psi11 = self.backend.evolve(ir, states["11"], params["t_gate"]).psi_final

            a00 = np.vdot(states["00"], psi00)
            a01 = np.exp(-1.0j * theta) * np.vdot(states["01"], psi01)
            a11 = np.exp(-2.0j * theta - 1.0j * np.pi) * np.vdot(states["11"], psi11)
            avg_f = (1 / 20) * (
                abs(a00 + 2 * a01 + a11) ** 2
                + abs(a00) ** 2
                + 2 * abs(a01) ** 2
                + abs(a11) ** 2
            )
            infidelity = float(1.0 - avg_f)
            fidelities[shot] = 1.0 - infidelity

            if compute_branching:
                residuals = self._residuals(system, [psi00, psi01, psi11])
                branching = residuals_to_branching(system, residuals)
                branch_xyz[shot] = branching["XYZ"]
                branch_al[shot] = branching["AL"]
                branch_lg[shot] = branching["LG"]
                branch_phase[shot] = max(
                    infidelity - (branching["XYZ"] + branching["AL"] + branching["LG"]),
                    0.0,
                )

        if n_shots >= 5:
            print(f"\r  MC done: {n_shots}/{n_shots} (100%)    ")

        infidelities = 1.0 - fidelities
        kwargs: dict[str, Any] = {
            "mean_fidelity": float(np.mean(fidelities)),
            "std_fidelity": float(np.std(fidelities)),
            "mean_infidelity": float(np.mean(infidelities)),
            "std_infidelity": float(np.std(infidelities)),
            "n_shots": n_shots,
            "fidelities": fidelities,
            "detuning_samples": detuning_samples,
            "distance_samples": distance_samples,
        }
        if compute_branching:
            kwargs.update(
                branch_XYZ=branch_xyz,
                branch_AL=branch_al,
                branch_LG=branch_lg,
                branch_phase=branch_phase,
                mean_branch_XYZ=float(np.mean(branch_xyz)),
                std_branch_XYZ=float(np.std(branch_xyz)),
                mean_branch_AL=float(np.mean(branch_al)),
                std_branch_AL=float(np.std(branch_al)),
                mean_branch_LG=float(np.mean(branch_lg)),
                std_branch_LG=float(np.std(branch_lg)),
                mean_branch_phase=float(np.mean(branch_phase)),
                std_branch_phase=float(np.std(branch_phase)),
            )
        return MonteCarloResult(**kwargs)

    def _sample_local_addressing(self, rng: np.random.Generator) -> RydbergSystem:
        if self._sigma_local_rin is None or not hasattr(self.protocol, "addressing"):
            return self.system
        import copy

        protocol = copy.copy(self.protocol)
        protocol.addressing = {
            idx: delta * (1.0 + rng.normal(0, self._sigma_local_rin))
            for idx, delta in self.protocol.addressing.items()
        }
        protocol._stark_phase_table = None
        return self.system.with_protocol(protocol)

    def _bind_shot(self, system: RydbergSystem):
        """Per-shot: materialize a concrete pulse from ``self.x`` when a CZ builder
        (TO/AR, exposing ``build``) is bound, returning ``(system, params, theta)``.

        The build consumes no RNG and is inserted between the existing ``rng`` draws,
        so the seeded noise stream is unchanged.
        """
        proto = getattr(system, "protocol", None)
        if proto is not None and hasattr(proto, "build"):
            system = system.with_protocol(proto.build(list(self.x), system))
            ti = getattr(proto, "theta_index", None)
            theta = float(self.x[ti]) if ti is not None else 0.0
            return system, system.unpack_params(()), theta
        params = system.unpack_params(self.x)
        return system, params, float(params.get("theta", 0.0))

    @staticmethod
    def _print_progress(label: str, shot: int, n_shots: int) -> None:
        if n_shots < 5:
            return
        pct = (shot + 1) * 100 // n_shots
        prev_pct = shot * 100 // n_shots
        if shot == 0 or pct != prev_pct:
            print(f"\r  {label} {shot + 1}/{n_shots} ({pct}%)", end="", flush=True)

    def _sample_terms(self, rng: np.random.Generator, compiler, system: RydbergSystem):
        terms: list[tuple[str, Any]] = []
        samples: dict[str, float] = {}
        amp_scale = 1.0

        if self._sigma_detuning_rad is not None:
            delta_err = rng.normal(0, self._sigma_detuning_rad)
            terms.append(("detuning_noise", delta_err * self._rydberg_occupation_operator(compiler, system)))
            samples["detuning"] = delta_err

        if self._sigma_pos_um is not None:
            from ryd_gate.core.operators import build_vdw_unit_operator

            sigmas = np.array(self._sigma_pos_um * 2)
            dx1, dy1, dz1, dx2, dy2, dz2 = rng.normal(0, sigmas)
            d_new = np.sqrt(
                (self._d_nominal_um + dx1 - dx2) ** 2
                + (dy1 - dy2) ** 2
                + (dz1 - dz2) ** 2
            )
            d_new = max(float(d_new), 0.1)
            delta_v = self._c6_vdw / d_new**6 - system.meta("v_ryd")
            terms.append(("position_noise", delta_v * build_vdw_unit_operator(
                tuple(system.meta("rydberg_indices", (5, 6))),
                system.basis.local_dim,
            )))
            samples["distance"] = d_new

        if self._sigma_amplitude is not None:
            amp_err = rng.normal(0, self._sigma_amplitude)
            amp_scale = 1.0 + amp_err
            # ``amplitude_scale`` already scales every *driven* channel's coefficient
            # (the 420 drive and, on rb87_7, the now-driven 1013 — both carry their
            # Rabi in the protocol coefficient).  Only a *static* 1013 coupling
            # (analog_3's H_1013) needs an explicit additive intensity-noise term.
            driven = system.protocol.drive_channels(system)
            pair = next(
                (
                    (a, b)
                    for a, b in (("drive_1013", "drive_1013_dag"), ("H_1013", "H_1013_conj"))
                    if system.blocks.has(a) and system.blocks.has(b) and a not in driven
                ),
                None,
            )
            if pair is not None:
                h1013 = (
                    compiler.materialize_block(system, pair[0])
                    + compiler.materialize_block(system, pair[1])
                )
                terms.append(("amplitude_noise_1013", amp_err * h1013))
            samples["amplitude"] = amp_err

        return amp_scale, terms, samples

    def _cz_basis_states(self) -> dict[str, np.ndarray]:
        return {
            "00": self.system.product_state("00"),
            "01": self.system.product_state("01"),
            "10": self.system.product_state("10"),
            "11": self.system.product_state("11"),
        }

    @staticmethod
    def _residuals(system: RydbergSystem, states: list[np.ndarray]) -> dict[str, float]:
        residuals = {key: 0.0 for key in ("e1", "e2", "e3", "ryd", "ryd_garb")}
        obs_map = {
            "e1": "pop_e1",
            "e2": "pop_e2",
            "e3": "pop_e3",
            "ryd": "pop_r",
            "ryd_garb": "pop_r_garb",
        }
        for psi in states:
            for key, obs in obs_map.items():
                if system.observables.has(obs):
                    residuals[key] += system.observables.measure(obs, psi)
        return {key: float(value / len(states)) for key, value in residuals.items()}

    def _rydberg_occupation_operator(self, compiler, system: RydbergSystem):
        if system.blocks.has("sum_nr"):
            return compiler.materialize_block(system, "sum_nr")
        if system.blocks.has("sum_n_r"):
            return compiler.materialize_block(system, "sum_n_r")

        basis = system.basis
        occ_op = np.zeros((basis.total_dim, basis.total_dim), dtype=np.complex128)
        for level in [level for level in basis.local_levels if level.startswith("r")]:
            block = f"sum_n_{level}"
            if system.blocks.has(block):
                occ_op = occ_op + compiler.materialize_block(system, block)
        return occ_op
