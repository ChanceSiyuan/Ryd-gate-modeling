"""Thread-safe lazy readers: norm/coefficient caches, evidence ledgers, sampling seam.

No module-scope YASTN import. Each public amplitude call stages every new norm,
coefficient and evidence record and commits them atomically only after all
contraction/reference/phase checks succeed (PEPS §12.5): a failed call publishes
no norm error, coefficient entry, or evidence record. ``_peps_evidence_snapshot``
builds a fresh :class:`PEPSEvidence` from current ledger state and constructs no
environment, norm, amplitude, or sample.
"""

from __future__ import annotations

import threading

from ryd_gate.backends.peps._boundary import double_layer_norm, product_bra_coefficient
from ryd_gate.backends.peps._numerics import _VALIDITY_RTOL, PEPSError
from ryd_gate.results import PEPSAmplitudeEvidence, PEPSEvidence, PEPSSampleEvidence


class _RealTimeLedger:
    """Eager real-time evidence fields + append-only lazy records."""

    __slots__ = ("parameters", "max_ntu_truncation_error", "cumulative_ntu_truncation_error",
                 "env_residuals", "env_iterations", "norm_contraction_error", "amplitudes", "samples")

    def __init__(self, *, parameters, max_ntu_truncation_error, cumulative_ntu_truncation_error,
                 env_residuals, env_iterations) -> None:
        self.parameters = parameters
        self.max_ntu_truncation_error = max_ntu_truncation_error
        self.cumulative_ntu_truncation_error = cumulative_ntu_truncation_error
        self.env_residuals = env_residuals  # tuple[float | None, ...] over measurement times
        self.env_iterations = env_iterations  # tuple[int, ...] over measurement times
        self.norm_contraction_error: float | None = None
        self.amplitudes: list[PEPSAmplitudeEvidence] = []
        self.samples: list[PEPSSampleEvidence] = []

    def snapshot(self) -> PEPSEvidence:
        iterations = max(self.env_iterations) if self.env_iterations else None
        if not self.env_residuals or any(r is None for r in self.env_residuals):
            residual = None
        else:
            residual = max(self.env_residuals)
        return PEPSEvidence(
            parameters=self.parameters,
            hamiltonian_scale_rad_s=None,
            max_ntu_truncation_error=self.max_ntu_truncation_error,
            cumulative_ntu_truncation_error=self.cumulative_ntu_truncation_error,
            environment_residual=residual,
            environment_iterations=iterations,
            norm_contraction_error=self.norm_contraction_error,
            amplitudes=tuple(self.amplitudes),
            samples=tuple(self.samples),
        )


class _GroundLedger:
    """Eager ground-state evidence fields + append-only lazy records."""

    __slots__ = ("parameters", "hamiltonian_scale_rad_s", "max_ntu_truncation_error",
                 "env_residual", "env_iterations", "norm_contraction_error", "amplitudes", "samples")

    def __init__(self, *, parameters, hamiltonian_scale_rad_s, max_ntu_truncation_error,
                 env_residual, env_iterations) -> None:
        self.parameters = parameters
        self.hamiltonian_scale_rad_s = hamiltonian_scale_rad_s
        self.max_ntu_truncation_error = max_ntu_truncation_error
        self.env_residual = env_residual  # final CTM residual (float | None)
        self.env_iterations = env_iterations  # final CTM sweeps (int)
        self.norm_contraction_error: float | None = None
        self.amplitudes: list[PEPSAmplitudeEvidence] = []
        self.samples: list[PEPSSampleEvidence] = []

    def snapshot(self) -> PEPSEvidence:
        return PEPSEvidence(
            parameters=self.parameters,
            hamiltonian_scale_rad_s=self.hamiltonian_scale_rad_s,
            max_ntu_truncation_error=self.max_ntu_truncation_error,
            cumulative_ntu_truncation_error=None,
            environment_residual=self.env_residual,
            environment_iterations=self.env_iterations,
            norm_contraction_error=self.norm_contraction_error,
            amplitudes=tuple(self.amplitudes),
            samples=tuple(self.samples),
        )


class _BaseReader:
    """Shared lazy norm/coefficient cache + evidence ledger under one RLock."""

    __slots__ = ("_h", "_ops", "_psi", "_terms", "_site_to_coord", "_scales", "_controls",
                 "_options", "_ledger", "_lock", "_norm", "_coeff_cache")

    def __init__(self, h, ops, psi, terms, site_to_coord, scales, controls, options, ledger) -> None:
        self._h = h
        self._ops = ops
        self._psi = psi
        self._terms = terms
        self._site_to_coord = site_to_coord
        self._scales = scales
        self._controls = controls
        self._options = options
        self._ledger = ledger
        self._lock = threading.RLock()
        self._norm: tuple[float, float] | None = None  # (norm, contraction_error)
        self._coeff_cache: dict[tuple, complex] = {}

    def _peps_evidence_snapshot(self) -> PEPSEvidence:
        with self._lock:
            return self._ledger.snapshot()

    # ── staged transaction primitives (call under self._lock) ────────────────

    def _stage_norm(self, staged):
        """Return the (norm, error) pair, computing+staging it once if not yet cached."""
        if self._norm is not None:
            return self._norm
        if staged["norm"] is None:
            staged["norm"] = double_layer_norm(self._h, self._psi, self._scales, self._controls)
        return staged["norm"]

    def _stage_coefficient(self, labels, staged):
        """Return the normalized coefficient for ``labels``, staging a new record if fresh."""
        key = tuple(labels)
        if key in self._coeff_cache:
            return self._coeff_cache[key]
        if key in staged["coeffs"]:
            return staged["coeffs"][key]
        norm, _nerr = self._stage_norm(staged)
        raw, cerr = product_bra_coefficient(
            self._h, self._psi, self._scales, self._ops, self._site_to_coord, list(labels), self._controls
        )
        coeff = raw / (norm ** 0.5)
        staged["coeffs"][key] = coeff
        staged["records"].append(PEPSAmplitudeEvidence(labels=key, contraction_error=cerr))
        return coeff

    def _commit(self, staged) -> None:
        """Publish staged norm/coefficients/records atomically (call under lock)."""
        if staged["norm"] is not None and self._norm is None:
            self._norm = staged["norm"]
            self._ledger.norm_contraction_error = staged["norm"][1]
        self._coeff_cache.update(staged["coeffs"])
        self._ledger.amplitudes.extend(staged["records"])

    def _sample(self, shots: int, seed: int):
        from ryd_gate.backends.peps import _sampling

        with self._lock:
            counts = _sampling.sample(
                self._h, self._ops, self._psi, self._terms, self._site_to_coord,
                self._controls, self._options, self._ground_env, shots, seed,
            )
            self._ledger.samples.append(PEPSSampleEvidence(shots=shots, seed=seed))
            return counts

    _ground_env = None  # real-time sampling builds its own final-state environment lazily


class _RealTimeReader(_BaseReader):
    """Lazy amplitude/sample over the final real-time PEPS (ER04/ER07/ER09)."""

    __slots__ = ()

    def amplitude(self, labels) -> complex:
        from ryd_gate.backends.tn_common.initial_state import validate_labels

        labels = validate_labels(self._terms, labels)
        with self._lock:
            staged = {"norm": None, "coeffs": {}, "records": []}
            coeff = self._stage_coefficient(labels, staged)
            self._commit(staged)
            return coeff

    def sample(self, shots: int, seed: int):
        return self._sample(shots, seed)


class _GroundStateReader(_BaseReader):
    """Lazy phase-referenced amplitude/sample over the PEPS ground state (E16)."""

    __slots__ = ("_ground_env_held",)

    def __init__(self, h, ops, psi, terms, site_to_coord, scales, controls, options, ground_env, ledger) -> None:
        super().__init__(h, ops, psi, terms, site_to_coord, scales, controls, options, ledger)
        self._ground_env_held = ground_env  # retained final CTM for lazy sampling

    @property
    def _ground_env(self):
        return self._ground_env_held

    def amplitude(self, labels, phase_reference) -> complex:
        from ryd_gate.backends.tn_common.initial_state import validate_labels

        target = tuple(validate_labels(self._terms, labels))
        reference = tuple(validate_labels(self._terms, phase_reference))
        with self._lock:
            staged = {"norm": None, "coeffs": {}, "records": []}
            ref_coeff = self._stage_coefficient(reference, staged)
            if abs(ref_coeff) <= _VALIDITY_RTOL:
                raise PEPSError(
                    f"phase_reference amplitude is numerically zero ({abs(ref_coeff):.2e}); "
                    "choose a reference basis state with nonzero amplitude (E16)."
                )
            target_coeff = ref_coeff if target == reference else self._stage_coefficient(target, staged)
            result = target_coeff / (ref_coeff / abs(ref_coeff))
            self._commit(staged)
            return result

    def sample(self, shots: int, seed: int):
        return self._sample(shots, seed)
