"""Public pulse-construction helpers (P18/P33).

Waveform/phase construction is not atomic physics, so these live in
``ryd_gate.protocols`` alongside the protocol classes, not in
``ryd_gate.physics``. Exactly two names are public:

    from ryd_gate.protocols import blackman_pulse, phase_from_chirp
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["blackman_pulse", "phase_from_chirp"]


def _blackman_window(t, rise_time_s):
    """Blackman rise kernel (protocol-internal; not exported)."""
    return (
        0.42
        - 0.5 * np.cos(2 * np.pi * t / (2 * rise_time_s))
        + 0.08 * np.cos(4 * np.pi * t / (2 * rise_time_s))
    )


def blackman_pulse(t_s, rise_time_s, t_gate_s):
    """Blackman-windowed flat-top envelope over ``[0, t_gate_s]`` (dimensionless).

    Blackman rise on ``[0, rise_time_s]``, unity flat top, symmetric Blackman
    fall. Accepts a scalar or a NumPy array ``t_s`` and preserves its shape.
    Square-pulse behaviour is deliberately not folded into this helper.

    ``t_gate_s`` and ``rise_time_s`` must be finite and positive with
    ``2 * rise_time_s <= t_gate_s``.
    """
    if not np.isfinite(t_gate_s) or t_gate_s <= 0.0:
        raise ValueError(f"t_gate_s must be finite and positive; got {t_gate_s!r}.")
    if not np.isfinite(rise_time_s) or rise_time_s <= 0.0:
        raise ValueError(f"rise_time_s must be finite and positive; got {rise_time_s!r}.")
    if 2 * rise_time_s > t_gate_s:
        raise ValueError(
            f"need 2*rise_time_s <= t_gate_s; got rise_time_s={rise_time_s!r}, "
            f"t_gate_s={t_gate_s!r}."
        )
    t = np.asarray(t_s, dtype=float)
    ret = (
        _blackman_window(t, rise_time_s) * np.heaviside(rise_time_s - t, 1)
        + np.heaviside(t - rise_time_s, 0) * np.heaviside(t_gate_s - t - rise_time_s, 0)
        + _blackman_window(t_gate_s - t, rise_time_s)
        * np.heaviside(rise_time_s - (t_gate_s - t), 1)
    )
    return ret if ret.ndim else float(ret)


def phase_from_chirp(
    chirp_rad_s: Callable[[float], float],
    t_gate_s: float,
    *,
    n_samples: int = 1001,
) -> Callable[[float], float]:
    """Integrate an instantaneous chirp into an optical phase over ``[0, t_gate_s]``.

    Returns a physical-time callable ``phase_rad(t_s) = ∫₀^{t_s} chirp(t') dt'``
    implemented as a cumulative trapezoid on a fixed grid of ``n_samples`` points
    followed by O(1) interpolation. ``chirp_rad_s(t_s)`` must return a finite
    real rad/s value at each sample.

    The returned callable's domain is strictly ``[0, t_gate_s]``; evaluating it
    outside raises rather than clipping, so a caller/backend time error is not
    hidden.
    """
    if not callable(chirp_rad_s):
        raise TypeError("chirp_rad_s must be a callable f(t_s) -> rad/s.")
    if not np.isfinite(t_gate_s) or t_gate_s <= 0.0:
        raise ValueError(f"t_gate_s must be finite and positive; got {t_gate_s!r}.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError(f"n_samples must be a non-bool integer >= 2; got {n_samples!r}.")

    grid = np.linspace(0.0, float(t_gate_s), int(n_samples))
    chirp = np.empty_like(grid)
    for i, t in enumerate(grid):
        v = float(chirp_rad_s(float(t)))
        if not np.isfinite(v):
            raise ValueError(f"chirp_rad_s returned a non-finite value at t={t}: {v!r}.")
        chirp[i] = v
    phase = np.zeros_like(grid)
    phase[1:] = np.cumsum(0.5 * (chirp[:-1] + chirp[1:]) * np.diff(grid))

    tg = float(t_gate_s)

    def phase_rad(t_s, _g=grid, _p=phase, _tg=tg):
        t = float(t_s)
        if not np.isfinite(t) or t < 0.0 or t > _tg:
            raise ValueError(
                f"phase_from_chirp callable is defined on finite [0, {_tg}]; got t_s={t_s!r}."
            )
        return float(np.interp(t, _g, _p))

    return phase_rad
