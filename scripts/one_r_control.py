"""Shared bounded spline control used by the two ``01r`` studies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

TWOPI = 2.0 * np.pi
MHz = TWOPI * 1e6

SPACING_UM = 4.0
RYD_LEVEL = 70
A_MAX = 17.0 * MHz
CHI_MAX = 20.0 * MHz
EDGE_CHIRP_MIN = 5.0 * MHz

SEED_SPECS = (
    {"branch": "phase_near_pi", "amplitude_MHz": 13.0,
     "edge_chirp_MHz": 15.0},
    {"branch": "negative_phase", "amplitude_MHz": 14.0,
     "edge_chirp_MHz": 12.0},
    {"branch": "positive_phase", "amplitude_MHz": 10.0,
     "edge_chirp_MHz": 13.0},
)
LOGICAL_LABELS = (["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"])


def power_envelope(s):
    """15% quintic rise, flat top, and symmetric fall."""
    s = np.asarray(s, dtype=float)
    out = np.ones_like(s)
    rise = 0.15

    left = s < rise
    x = np.clip(s[left] / rise, 0.0, 1.0)
    out[left] = 10 * x**3 - 15 * x**4 + 6 * x**5

    right = s > 1.0 - rise
    x = np.clip((1.0 - s[right]) / rise, 0.0, 1.0)
    out[right] = 10 * x**3 - 15 * x**4 + 6 * x**5
    return out.item() if out.ndim == 0 else out


@dataclass(frozen=True)
class ControlBasis:
    """Cubic B-spline coordinates for bounded amplitude and chirp."""

    n_coeffs: int = 8
    degree: int = 3

    def __post_init__(self):
        n_inner = self.n_coeffs - self.degree - 1
        inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        knots = np.r_[np.zeros(self.degree + 1), inner,
                      np.ones(self.degree + 1)]
        object.__setattr__(self, "knots", knots)
        object.__setattr__(
            self, "_spline",
            BSpline(knots, np.eye(self.n_coeffs), self.degree,
                    extrapolate=False),
        )

    @property
    def n_parameters(self):
        return 2 * self.n_coeffs + 1

    def matrix(self, s):
        return np.asarray(self._spline(np.asarray(s, dtype=float)))

    def seed(self, amplitude_MHz, edge_chirp_MHz):
        amplitude = np.full(self.n_coeffs, amplitude_MHz / (A_MAX / MHz))
        detuning = np.zeros(self.n_coeffs)
        eta = edge_chirp_MHz / (CHI_MAX / MHz)
        return np.r_[amplitude, detuning, eta]

    def bounds(self):
        return (
            [(0.0, 1.0)] * self.n_coeffs
            + [(-3.0, 3.0)] * self.n_coeffs
            + [(EDGE_CHIRP_MIN / CHI_MAX, 1.0)]
        )

    def controls(self, parameters, s, *, jacobian=False):
        parameters = np.asarray(parameters, dtype=float)
        scalar = np.ndim(s) == 0
        s = np.atleast_1d(np.asarray(s, dtype=float))
        basis = self.matrix(s)
        envelope = np.asarray(power_envelope(s))

        amplitude_coeffs = parameters[:self.n_coeffs]
        detuning_coeffs = parameters[self.n_coeffs:2 * self.n_coeffs]
        eta = parameters[-1]

        amplitude = A_MAX * envelope * (basis @ amplitude_coeffs)
        x = -eta * np.cos(TWOPI * s)
        v = np.tanh(envelope * (basis @ detuning_coeffs))
        denominator = 1.0 + x * v
        chirp = CHI_MAX * (x + v) / denominator

        if not jacobian:
            if scalar:
                return float(amplitude[0]), float(chirp[0])
            return amplitude, chirp

        d_amplitude = np.zeros((s.size, self.n_parameters))
        d_chirp = np.zeros((s.size, self.n_parameters))
        d_amplitude[:, :self.n_coeffs] = (
            A_MAX * envelope[:, None] * basis)
        chirp_factor = ((1.0 - x * x) * (1.0 - v * v)
                        / (denominator * denominator))
        d_chirp[:, self.n_coeffs:2 * self.n_coeffs] = (
            CHI_MAX * chirp_factor[:, None] * envelope[:, None] * basis)
        d_chirp[:, -1] = (
            -CHI_MAX * np.cos(TWOPI * s) * (1.0 - v * v)
            / (denominator * denominator))

        if scalar:
            return (float(amplitude[0]), float(chirp[0]),
                    d_amplitude[0], d_chirp[0])
        return amplitude, chirp, d_amplitude, d_chirp


BASIS = ControlBasis(n_coeffs=8)
