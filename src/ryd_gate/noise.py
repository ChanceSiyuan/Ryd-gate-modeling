"""Quasi-static noise and the ensemble entry point (N01-N18).

:class:`NoiseModel` declares zero-mean Gaussian quasi-static noise on named laser
groups (amplitude / frequency) and on atom position. :func:`simulate_ensemble`
pre-samples every shot-major realization up front (deterministically, N17),
materializes a noisy fully specified system per shot, and delegates to the public
:func:`ryd_gate.simulate` — it never instantiates a compiler or solver (N08).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from ryd_gate.results import EnsembleResult

__all__ = ["NoiseModel", "simulate_ensemble"]


class NoiseModel:
    """Quasi-static zero-mean Gaussian noise (N01-N07).

    Parameters
    ----------
    laser_amplitude_sigma
        ``{group: sigma}`` relative amplitude noise ``(1 + epsilon_L)`` per laser.
    laser_frequency_sigma_rad_s
        ``{group: sigma}`` frequency-offset noise ``delta_L`` (rad/s) per laser.
    position_sigma_um
        ``(sigma_x, sigma_y, sigma_z)`` per-atom 3D Gaussian position jitter (um);
        ``sigma_z`` is out-of-plane thermal motion of the 2D array.
    """

    laser_amplitude_sigma: Mapping[str, float]
    laser_frequency_sigma_rad_s: Mapping[str, float]
    position_sigma_um: tuple[float, float, float] | None

    __slots__ = ("laser_amplitude_sigma", "laser_frequency_sigma_rad_s", "position_sigma_um")

    def __init__(
        self,
        laser_amplitude_sigma: Mapping[str, float] | None = None,
        laser_frequency_sigma_rad_s: Mapping[str, float] | None = None,
        position_sigma_um: tuple[float, float, float] | None = None,
    ) -> None:
        amp = _check_sigma_map(laser_amplitude_sigma, "laser_amplitude_sigma")
        freq = _check_sigma_map(laser_frequency_sigma_rad_s, "laser_frequency_sigma_rad_s")
        pos = _check_position_sigma(position_sigma_um)
        active = any(v > 0 for v in amp.values()) or any(v > 0 for v in freq.values()) or (
            pos is not None and any(s > 0 for s in pos)
        )
        if not active:
            raise ValueError(
                "NoiseModel must declare at least one nonzero sigma; use the "
                "deterministic simulate() for a noiseless run."
            )
        object.__setattr__(self, "laser_amplitude_sigma", MappingProxyType(dict(amp)))
        object.__setattr__(self, "laser_frequency_sigma_rad_s", MappingProxyType(dict(freq)))
        object.__setattr__(self, "position_sigma_um", pos)

    def __setattr__(self, name, value):
        raise AttributeError("NoiseModel is immutable.")


def _check_sigma_map(mapping, name) -> dict[str, float]:
    if mapping is None:
        return {}
    out: dict[str, float] = {}
    for key, sigma in dict(mapping).items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be laser-group strings; got {type(key).__name__}.")
        s = float(sigma)
        if not np.isfinite(s) or s < 0.0:
            raise ValueError(f"{name}[{key!r}] must be finite and non-negative; got {sigma!r}.")
        out[key] = s
    return out


def _check_position_sigma(position_sigma_um):
    if position_sigma_um is None:
        return None
    sig = tuple(float(s) for s in position_sigma_um)
    if len(sig) != 3 or any((not np.isfinite(s)) or s < 0.0 for s in sig):
        raise ValueError(
            f"position_sigma_um must be a finite non-negative (sx, sy, sz); got {position_sigma_um!r}."
        )
    return sig


def simulate_ensemble(
    system,
    initial_state=None,
    *,
    noise: NoiseModel,
    shots: int,
    seed: int,
    backend: str = "exact_ode",
    t_eval=None,
    observables=None,
    backend_options=None,
) -> EnsembleResult:
    """Sample ``shots`` noisy realizations and evolve each via ``simulate`` (N08)."""
    from ryd_gate.protocols._resolved import _LaserDrive
    from ryd_gate.simulate import simulate

    if not isinstance(noise, NoiseModel):
        raise TypeError("noise must be a NoiseModel.")
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError(f"shots must be a positive integer; got {shots!r}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer; got {seed!r}.")

    # N14 preflight: named laser noise must match active _LaserDrive groups.
    resolved = system.protocol._resolve(system)
    active_groups = {d.group for d in resolved.drives if isinstance(d, _LaserDrive)}
    named = set(noise.laser_amplitude_sigma) | set(noise.laser_frequency_sigma_rad_s)
    unknown = named - active_groups
    if unknown:
        raise ValueError(
            f"NoiseModel names laser group(s) {sorted(unknown)} not driven by this "
            f"protocol (active groups: {sorted(active_groups)}). Effective-control noise "
            "must be written into realization-specific coefficient functions (N13)."
        )

    realizations = _sample_realizations(noise, system.N, int(shots), int(seed))
    results = tuple(
        _simulate_shot(
            system, realization, initial_state, simulate,
            backend=backend, t_eval=t_eval, observables=observables, backend_options=backend_options,
        )
        for realization in realizations
    )
    return EnsembleResult(results=results, realizations=realizations, seed=int(seed))


def _sample_realizations(noise: NoiseModel, n_atoms: int, shots: int, seed: int) -> tuple:
    """Pre-sample all shot-major realizations in stable sorted-key order (N16/N17)."""
    rng = np.random.default_rng(seed)
    amp_keys = sorted(noise.laser_amplitude_sigma)
    freq_keys = sorted(noise.laser_frequency_sigma_rad_s)
    pos_sigma = noise.position_sigma_um

    realizations = []
    for _ in range(shots):
        amp = {k: 1.0 + noise.laser_amplitude_sigma[k] * float(rng.standard_normal()) for k in amp_keys}
        freq = {k: noise.laser_frequency_sigma_rad_s[k] * float(rng.standard_normal()) for k in freq_keys}
        if pos_sigma is not None and any(s > 0 for s in pos_sigma):
            draws = rng.standard_normal((n_atoms, 3)) * np.asarray(pos_sigma)
            positions = tuple(tuple(float(x) for x in row) for row in draws)
        else:
            positions = None
        realizations.append(
            MappingProxyType({
                "laser_amplitude_scales": MappingProxyType(amp),
                "laser_frequency_offsets_rad_s": MappingProxyType(freq),
                "position_offsets_um": positions,
            })
        )
    return tuple(realizations)


def _simulate_shot(system, realization, initial_state, simulate, *, backend, t_eval, observables, backend_options):
    noisy = system._with_realization(realization)
    return simulate(
        noisy, initial_state, backend=backend, t_eval=t_eval,
        observables=observables, backend_options=backend_options,
    )
