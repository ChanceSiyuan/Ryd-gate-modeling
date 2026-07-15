"""Quasi-static noise description and the ensemble execution entry point.

:class:`NoiseModel` declares the *executable* quasi-static noise channels:
shot-to-shot static detuning offsets, global amplitude (Rabi) scale factors,
local-addressing intensity noise, and atom-position fluctuations.  Decay is
not noise configuration — it belongs to the level physical model
(``level_structure(..., enable_rydberg_decay=True)``).

:func:`simulate_ensemble` executes a protocol-bound system under sampled
realizations of a :class:`NoiseModel` and returns the raw shot-major
:class:`EnsembleResult`.  No statistics are computed here — mean/std/fidelity
aggregation belongs to the calling script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ryd_gate.ir import EvolutionResult, HamiltonianTerm

__all__ = ["NoiseModel", "EnsembleResult", "simulate_ensemble"]

_EXACT_BACKENDS = frozenset({"exact_ode"})


@dataclass(frozen=True)
class NoiseModel:
    """Declarative quasi-static noise channels (all standard deviations).

    Parameters
    ----------
    detuning_sigma_rad_s : float
        Static Rydberg-detuning offset per shot, sampled from ``N(0, sigma)``
        in rad/s and applied on the total target-``|r>`` occupation.
    amplitude_sigma : float
        Fractional global laser-amplitude (Rabi) noise: each shot scales every
        driven channel by ``1 + N(0, sigma)`` (static off-diagonal couplings —
        e.g. analog_3's static 1013 leg — receive the matching additive
        intensity term).
    local_rin_sigma : float
        Fractional local-addressing intensity noise: each shot scales each
        site's addressing shift by ``1 + N(0, sigma)``.  Requires a protocol
        with local-addressing capability (raises otherwise).
    position_sigma_um : float or (fx, fy, fz)
        Per-axis atom-position standard deviation in um.  Pair interaction
        shifts are computed from the actual register coordinates and each
        pair's nominal distance.
    """

    detuning_sigma_rad_s: float = 0.0
    amplitude_sigma: float = 0.0
    local_rin_sigma: float = 0.0
    position_sigma_um: float | tuple[float, float, float] = 0.0

    def __post_init__(self) -> None:
        sigma = self.position_sigma_um
        if isinstance(sigma, (list, tuple, np.ndarray)):
            sigma3 = tuple(float(s) for s in sigma)
            if len(sigma3) != 3:
                raise ValueError("position_sigma_um must be a scalar or (fx, fy, fz).")
            object.__setattr__(self, "position_sigma_um", sigma3)
        else:
            object.__setattr__(self, "position_sigma_um", float(sigma))
        for name in ("detuning_sigma_rad_s", "amplitude_sigma", "local_rin_sigma"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value}.")
            object.__setattr__(self, name, value)
        pos = self.position_sigma_um
        if any(s < 0.0 for s in (pos if isinstance(pos, tuple) else (pos,))):
            raise ValueError("position_sigma_um must be non-negative.")

    @property
    def position_sigma_xyz_um(self) -> tuple[float, float, float]:
        pos = self.position_sigma_um
        return pos if isinstance(pos, tuple) else (pos, pos, pos)

    @property
    def any_active(self) -> bool:
        return (
            self.detuning_sigma_rad_s > 0.0
            or self.amplitude_sigma > 0.0
            or self.local_rin_sigma > 0.0
            or any(s > 0.0 for s in self.position_sigma_xyz_um)
        )


@dataclass(frozen=True)
class EnsembleResult:
    """Raw shot-major ensemble output.

    ``results[shot][initial_state]`` are the per-shot evolution results (all
    initial states of a batch share the shot's realization);
    ``realizations[shot]`` records the actually applied, unit-bearing values
    (detuning offset in rad/s, amplitude scale, per-site local scales,
    position offsets in um and the resulting pair distances) — no matrices,
    no callables.  No statistics live here.
    """

    results: list[list[EvolutionResult]]
    realizations: list[dict]
    seed: int
    metadata: dict = field(default_factory=dict)


def simulate_ensemble(
    system,
    initial_state=None,
    *,
    noise: NoiseModel,
    shots: int,
    seed: int,
    backend: str = "exact_ode",
    t_eval=None,
    observables: dict | None = None,
    backend_options: dict | None = None,
) -> EnsembleResult:
    """Evolve *system* under ``shots`` sampled realizations of *noise*.

    ``shots`` and ``seed`` are mandatory integers (reproducibility is
    explicit: the same seed reproduces the same realizations).
    ``initial_state`` accepts the same label-based inputs as
    :func:`ryd_gate.simulate`; for a batch, every initial state within one
    shot shares that shot's realization.  ``observables`` is a
    ``dict[str, ObservableExpr]`` (built from ``system.observables``) recorded
    per shot exactly as in :func:`ryd_gate.simulate`.  Noise execution
    currently supports only the exact backends — tensor-network backends raise
    a capability error up front.
    """
    from ryd_gate.backends.exact.compiler import ExactCompiler
    from ryd_gate.backends.exact.ode import ExactODEBackend
    from ryd_gate.backends.exact.simulate import (
        _exact_initial_state,
        validated_exact_options,
    )
    from ryd_gate.core.states import normalize_initial_state

    if not isinstance(shots, (int, np.integer)) or isinstance(shots, bool) or shots < 1:
        raise ValueError(f"shots must be a positive integer; got {shots!r}.")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer; got {seed!r}.")
    if not isinstance(noise, NoiseModel):
        raise TypeError(f"noise must be a NoiseModel; got {type(noise).__name__}.")
    key = str(backend).lower()
    if key not in _EXACT_BACKENDS:
        raise ValueError(
            f"simulate_ensemble supports only the exact backends {sorted(_EXACT_BACKENDS)}; "
            f"got backend={backend!r} (tensor-network noise execution is not available)."
        )
    protocol = system._require_protocol()
    if noise.local_rin_sigma > 0.0 and not _has_local_addressing(protocol):
        raise ValueError(
            "local_rin_sigma requires a protocol with local-addressing "
            f"capability; {type(protocol).__name__} has none."
        )

    kind, state = normalize_initial_state(initial_state, system.basis.n_sites)
    state_list = state if kind == "batch" else [state]

    rng = np.random.default_rng(int(seed))
    compiler = ExactCompiler()
    opts = validated_exact_options(backend_options)

    results: list[list[EvolutionResult]] = []
    realizations: list[dict] = []
    for _ in range(int(shots)):
        shot_system, realization = _sample_local_rin(rng, system, protocol, noise)
        extra_terms, term_records = _sample_terms(rng, compiler, shot_system, noise)
        realization.update(term_records)
        if "amplitude_scale" in realization:
            shot_system = shot_system._with_amplitude_scale(realization["amplitude_scale"])

        ir = compiler.compile(shot_system)
        ir.static_terms.extend(HamiltonianTerm(name, op, 1.0) for name, op in extra_terms)
        solver = ExactODEBackend(**opts)
        t_gate = float(ir.metadata["t_gate"])

        shot_results = [
            solver.evolve(ir, _exact_initial_state(shot_system, s), t_gate, t_eval, observables)
            for s in state_list
        ]
        results.append(shot_results)
        realizations.append(realization)

    return EnsembleResult(
        results=results,
        realizations=realizations,
        seed=int(seed),
        metadata={
            "backend": key,
            "shots": int(shots),
            "n_initial_states": len(state_list),
            "noise": {
                "detuning_sigma_rad_s": noise.detuning_sigma_rad_s,
                "amplitude_sigma": noise.amplitude_sigma,
                "local_rin_sigma": noise.local_rin_sigma,
                "position_sigma_um": noise.position_sigma_xyz_um,
            },
        },
    )


def _has_local_addressing(protocol) -> bool:
    if getattr(protocol, "address_fn", None) is not None:
        return True
    return isinstance(getattr(protocol, "addressing", None), dict)


def _sample_local_rin(rng, system, protocol, noise: NoiseModel) -> tuple:
    """Sample the protocol-level part of a realization (local RIN)."""
    realization: dict = {}
    if noise.local_rin_sigma <= 0.0:
        return system, realization

    import copy

    if getattr(protocol, "address_fn", None) is not None:
        n_sites = system.basis.n_sites
        scales = 1.0 + rng.normal(0.0, noise.local_rin_sigma, size=n_sites)
        base_fn = protocol.address_fn
        shot_protocol = copy.copy(protocol)
        shot_protocol.address_fn = (
            lambda t, i, _base=base_fn, _s=scales: _s[i] * _base(t, i)
        )
        realization["local_scales"] = {i: float(s) for i, s in enumerate(scales)}
        return system.with_protocol(shot_protocol), realization

    # Duck-typed `.addressing` dict (site -> static addressing shift).
    shot_protocol = copy.copy(protocol)
    scales = {
        idx: 1.0 + float(rng.normal(0.0, noise.local_rin_sigma))
        for idx in protocol.addressing
    }
    shot_protocol.addressing = {
        idx: delta * scales[idx] for idx, delta in protocol.addressing.items()
    }
    if hasattr(shot_protocol, "_stark_phase_table"):
        shot_protocol._stark_phase_table = None
    realization["local_scales"] = scales
    return system.with_protocol(shot_protocol), realization


def _sample_terms(rng, compiler, system, noise: NoiseModel) -> tuple:
    """Sample the Hamiltonian-level part of a realization.

    Returns ``(extra_terms, records)``: additive static Hamiltonian terms and
    the unit-bearing sampled values.  The caller applies ``amplitude_scale``
    onto the system before compiling.
    """
    terms: list[tuple[str, object]] = []
    records: dict = {}

    if noise.detuning_sigma_rad_s > 0.0:
        delta_err = float(rng.normal(0.0, noise.detuning_sigma_rad_s))
        n_ryd = _ir_operator(
            compiler.materialize_operator(system, system.operators.sum("E[r,r]")), system
        )
        terms.append(("detuning_noise", delta_err * n_ryd))
        records["detuning_rad_s"] = delta_err

    sigma_xyz = noise.position_sigma_xyz_um
    if any(s > 0.0 for s in sigma_xyz):
        pair_terms, pair_records = _position_noise_terms(rng, system, sigma_xyz)
        terms.extend(pair_terms)
        records.update(pair_records)

    if noise.amplitude_sigma > 0.0:
        amp_err = float(rng.normal(0.0, noise.amplitude_sigma))
        records["amplitude_scale"] = 1.0 + amp_err
        # ``amplitude_scale`` already scales every *driven* channel's coefficient
        # (the 420 drive and, on rb87 seven-level, the now-driven 1013 — both
        # carry their Rabi in the protocol coefficient).  A *static* off-diagonal
        # coupling (analog_3's static 1013, E[r,e]) needs an explicit additive
        # intensity-noise term.
        driven = system.protocol.drive_channels(system)
        for term in system.static_hamiltonian_terms:
            if not term.add_hermitian_conjugate or term.name in driven:
                continue
            op = _ir_operator(compiler.materialize_operator(system, term.operator), system)
            coupling = term.coefficient * op + np.conj(term.coefficient) * op.conj().T
            terms.append(("amplitude_noise_static_coupling", amp_err * coupling))

    return terms, records


def _ir_operator(op, system):
    """Match a noise operator to the compiled IR's storage (sparse vs dense)."""
    if getattr(system, "is_sparse", True):
        return op
    return op.toarray() if hasattr(op, "toarray") else np.asarray(op)


def _position_noise_terms(rng, system, sigma_xyz):
    """Pairwise VdW fluctuation terms from actual register coordinates.

    Each atom gets an independent 3D Gaussian position offset; each
    Rydberg-interacting pair's nominal distance comes from the register
    coordinates (no hard-coded spacing), and the shot's interaction shift is
    ``ΔV = C6/d_new^6 - C6/d_nom^6`` with ``C6 = V_pair * d_nom^6``.
    """
    from ryd_gate.core.operators import build_vdw_unit_operator

    geometry = getattr(system, "geometry", None)
    coords = None if geometry is None else np.asarray(geometry.coords, dtype=float)
    if coords is None:
        raise ValueError("position noise requires a system with register geometry.")
    n_atoms = coords.shape[0]
    if n_atoms != 2:
        raise NotImplementedError(
            f"position noise currently supports two-atom registers (got {n_atoms} atoms)."
        )
    coords3 = np.zeros((n_atoms, 3))
    coords3[:, : coords.shape[1]] = coords  # register coords are in um

    pairs = [
        (int(i), int(j), float(v))
        for i, j, v in system.interaction_pairs
    ]
    if not pairs:
        raise ValueError(
            "position noise needs interaction pairs: the system resolved none "
            "(no Rydberg-interacting pair in range)."
        )

    offsets = rng.normal(0.0, np.asarray(sigma_xyz, dtype=float), size=(n_atoms, 3))
    terms = []

    records = {
        "position_offsets_um": {
            i: tuple(float(v) for v in offsets[i]) for i in range(n_atoms)
        },
        "pair_distances_um": {},
    }
    ryd_indices = system.level_structure.rydberg_indices
    if ryd_indices is None:
        labels = [str(level) for level in system.basis.local_levels]
        ryd_indices = tuple(
            i for i, label in enumerate(labels) if label in ("r", "r_garb")
        )
        if not ryd_indices:
            raise ValueError(
                "position noise needs Rydberg levels: the system declares no "
                "rydberg_indices metadata and no 'r' level."
            )
    ryd_indices = tuple(int(i) for i in ryd_indices)
    for i, j, v_pair in pairs:
        d_nom = float(np.linalg.norm(coords3[i] - coords3[j]))
        if d_nom <= 0.0:
            raise ValueError(f"pair ({i},{j}) has zero nominal distance in the register.")
        d_new = float(np.linalg.norm((coords3[i] + offsets[i]) - (coords3[j] + offsets[j])))
        d_new = max(d_new, 0.1)
        c6 = v_pair * d_nom**6
        delta_v = c6 / d_new**6 - v_pair
        vdw_op = build_vdw_unit_operator(ryd_indices, system.basis.local_dim)
        if getattr(system, "is_sparse", True):
            from scipy.sparse import csr_matrix

            vdw_op = csr_matrix(vdw_op)
        terms.append(("position_noise", delta_v * vdw_op))
        records["pair_distances_um"][(i, j)] = d_new
    return terms, records
