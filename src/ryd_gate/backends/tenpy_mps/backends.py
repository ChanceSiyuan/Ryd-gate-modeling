"""DMRG and TDVP backend wrappers for TeNPy, plus their typed options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ryd_gate.backends.tn_common._expressions import (
    MPS_MAX_TERM_SITES,
    preflight_tn_expressions,
    tn_basis,
)
from ryd_gate.backends.tn_common.lattice_spec import TNLatticeSpec
from ryd_gate.backends.tn_common.protocol_context import (
    analog3_dt_guard,
    merge_pin_deltas,
    pin_deltas_from_params,
    plan_tn_segments,
    require_tn_dt,
)
from ryd_gate.core.level_structures import (
    three_level_profiles_from_coeffs,
    two_level_drive_and_detuning_from_coeffs,
)
from ryd_gate.ir import EvolutionResult, validate_evolution_request


def _require_tenpy():
    try:
        import tenpy
        return tenpy
    except ImportError as exc:
        raise ImportError(
            "TeNPy is required for tensor network simulations. "
            "Install via: pip install physics-tenpy  "
            "or: pip install ryd-gate[tn]"
        ) from exc


class TenpyDMRGBackend:
    """DMRG ground-state solver wrapping TeNPy finite DMRG.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension.
    svd_min : float
        Truncation cutoff.
    n_sweeps : int
        Maximum number of DMRG sweeps.
    mixer : bool
        Whether to use a density-matrix mixer for convergence.
    """

    def __init__(
        self,
        chi_max: int = 256,
        svd_min: float = 1e-10,
        n_sweeps: int = 20,
        mixer: bool = True,
    ) -> None:
        self.chi_max = chi_max
        self.svd_min = svd_min
        self.n_sweeps = n_sweeps
        self.mixer = mixer

    def find_ground_state(
        self,
        spec: TNLatticeSpec,
        Delta: float,
        pin_deltas: np.ndarray | None = None,
        initial_state: str | np.ndarray = "all_ground",
    ) -> EvolutionResult:
        """Find ground state via finite DMRG.

        Parameters
        ----------
        spec : TNLatticeSpec
        Delta : float
            Global detuning.
        pin_deltas : ndarray or None
            Per-site local detunings.
        initial_state : str or ndarray
            Initial MPS configuration (see :func:`product_state_mps`).

        Returns
        -------
        EvolutionResult
            ``final_state`` is the TeNPy MPS ground state.  A ground-state
            search has no evolution clock, so the result carries the minimal
            time axis ``times == [0.0]`` and ``expectations == {}``; measure
            the returned MPS directly.  ``metadata`` contains ``energy``,
            ``chi``, ``n_sweeps``.
        """
        _require_tenpy()
        from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

        from .model import build_tenpy_model
        from .state import product_state_mps

        model = build_tenpy_model(spec, Delta, pin_deltas=pin_deltas)

        if hasattr(initial_state, "expectation_value"):
            psi = initial_state  # already an MPS
        else:
            psi = product_state_mps(spec, initial_state)

        dmrg_params = {
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
            },
            "mixer": self.mixer,
            "max_sweeps": self.n_sweeps,
        }

        eng = TwoSiteDMRGEngine(psi, model, dmrg_params)
        E0, _ = eng.run()

        metadata = {
            "backend": "mps",
            "engine_package": "tenpy",
            "energy": E0,
            "chi": max(psi.chi),
            "n_sweeps": eng.sweeps,
            "method": "dmrg",
        }
        return EvolutionResult(
            final_state=psi,
            times=np.array([0.0]),
            expectations={},
            metadata=metadata,
            basis=tn_basis(spec),
        )


class TenpyTDVPBackend:
    """Two-site TDVP time evolution using TeNPy.

    Parameters
    ----------
    chi_max : int
        Maximum bond dimension (paper uses 1200 for production).
    dt : float
        Time step (mandatory: ``backend_options={"dt": ...}``).  Between
        consecutive sampling anchors ``[a, b]`` (the requested ``t_eval``
        times plus ``t_gate``) the evolution takes ``ceil((b - a) / dt)``
        equal steps, hitting every anchor exactly.
    svd_min : float
        Truncation cutoff.
    """

    def __init__(
        self,
        chi_max: int = 256,
        dt: float | None = None,
        svd_min: float = 1e-10,
    ) -> None:
        self.chi_max = chi_max
        self.dt = dt
        self.svd_min = svd_min

    def evolve_ir(
        self,
        ir,
        initial_state: str | np.ndarray | object = "all_ground",
        t_eval: np.ndarray | None = None,
        observables=None,
    ) -> EvolutionResult:
        """Evolve a compiled TN IR with TeNPy two-site TDVP.

        Uses piecewise-constant Hamiltonian updates: at each TDVP step the protocol
        coefficients are evaluated at the interval midpoint and the MPO is rebuilt.

        Parameters
        ----------
        ir : TNEvolutionIR
            Carries the lattice spec, bound protocol, and the resolved context
            dict. The protocol supports ``SweepProtocol`` plus
            ``DigitalAnalogProtocol`` on either the effective ``1r`` TN
            subspace or explicit ``01r`` levels.
        initial_state : str, ndarray, or tenpy MPS
            Initial state; a string/array is built into a product-state MPS via
            :func:`product_state_mps`, while an existing MPS is used as-is
            (the MPS continuation seam).
        t_eval : ndarray or None
            Measurement times only — never the evolution endpoint.  ``None``
            measures at ``t_gate`` (result ``times == [t_gate]``); an explicit
            array is validated strictly and returned exactly as
            ``result.times``.
        observables : dict[str, ObservableExpr] or None
            Named observable expressions; each becomes a complex array in
            ``result.expectations`` (raw ``<psi|O|psi>``, one value per entry
            of ``result.times``).

        Returns
        -------
        EvolutionResult
            ``final_state`` is the MPS at the true ``t_gate`` endpoint.
        """
        spec = ir.spec
        protocol = ir.protocol
        params = ir.params
        t_gate = float(params["t_gate"])

        # Full preflight (times, expressions, step control) before any evolution.
        times, exprs = validate_evolution_request(t_gate, t_eval, observables)
        preflight_tn_expressions(exprs, spec, backend="mps", max_term_sites=MPS_MAX_TERM_SITES)
        dt = require_tn_dt(self.dt)
        analog3_dt_guard(spec, dt)

        from .observables import lower_expressions
        from .state import product_state_mps

        _require_tenpy()
        from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

        out_times = times if times is not None else np.array([t_gate])
        record_at_start, segments = plan_tn_segments(t_gate, out_times, dt)

        psi0 = (
            initial_state
            if hasattr(initial_state, "expectation_value")
            else product_state_mps(spec, initial_state)
        )
        psi = psi0.copy()

        plan = lower_expressions(exprs, spec) if exprs else None
        records: dict[str, list[complex]] = {label: [] for label in exprs}

        def record() -> None:
            if plan is None:
                return
            for label, value in plan.measure(psi).items():
                records[label].append(value)

        tdvp_params = {
            "trunc_params": {
                "chi_max": self.chi_max,
                "svd_min": self.svd_min,
            },
        }

        if record_at_start:
            record()
        n_steps = 0
        for segment in segments:
            for k in range(segment.n_sub):
                t_mid = segment.t0 + (k + 0.5) * segment.dt_sub
                model = _build_step_model(spec, protocol, params, t_mid)
                eng = TwoSiteTDVPEngine(psi, model, tdvp_params)
                eng.evolve(1, segment.dt_sub)
                n_steps += 1
            if segment.record:
                record()

        expectations = {
            label: np.asarray(values, dtype=complex) for label, values in records.items()
        }
        metadata_out = {
            **(ir.metadata or {}),
            "method": "mps_tdvp",
            "backend": "mps",
            "engine_package": "tenpy",
            "chi_max": self.chi_max,
            "dt": dt,
            "n_steps": n_steps,
        }
        return EvolutionResult(
            final_state=psi,
            times=out_times,
            expectations=expectations,
            metadata=metadata_out,
            basis=tn_basis(spec),
        )


def _build_step_model(spec: TNLatticeSpec, protocol, params: dict, t_mid: float):
    """The piecewise-constant TeNPy model for one TDVP step (coefficients at ``t_mid``)."""
    from .model import build_tenpy_model

    coeffs = protocol.get_drive_coefficients(t_mid, params)
    if spec.level_structure == "analog_3":
        # Under the TN context the protocol emits the unitless g-e envelope
        # on E[e,g]; the model re-applies the full Rabi via local_blocks.
        return build_tenpy_model(
            spec, Delta=0.0,
            drive_420_coeff=complex(coeffs.get("E[e,g]", 0.0)),
        )
    if spec.level_structure == "01r":
        profiles = three_level_profiles_from_coeffs(coeffs, spec)
        pin = pin_deltas_from_params(params, spec.N)
        if pin is not None:
            # A pin adds to the Rydberg detuning, i.e. subtracts energy.
            profiles["energy_r"] = profiles["energy_r"] - pin
        return build_tenpy_model(spec, Delta=0.0, **profiles)

    Omega_t, Delta_t, channel_pin_deltas = (
        two_level_drive_and_detuning_from_coeffs(coeffs, spec)
    )
    pin_deltas = merge_pin_deltas(
        pin_deltas_from_params(params, spec.N),
        channel_pin_deltas,
        n_sites=spec.N,
    )
    return build_tenpy_model(spec, Delta_t, Omega=Omega_t, pin_deltas=pin_deltas)


@dataclass(frozen=True)
class TenpyOptions:
    """Options for ``backend="mps"`` TDVP/DMRG evolution.

    ``None`` means "use the backend default". ``dt`` (mandatory step control
    for TDVP evolution) and ``chi_max`` apply to TDVP; ``n_sweeps`` and
    ``mixer`` apply to DMRG ground-state search.
    """

    chi_max: int | None = None
    dt: float | None = None
    svd_min: float | None = None
    n_sweeps: int | None = None
    mixer: bool | None = None
