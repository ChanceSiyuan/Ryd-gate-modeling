"""Laser phase noise from a measured frequency-noise PSD (expert module).

Implements the noise model of X. Jiang, J. Scott, M. Friesen and M. Saffman,
*Sensitivity of quantum gate fidelity to laser phase and intensity noise*,
Phys. Rev. A **107**, 042611 (2023) (arXiv:2210.11007): a frequency-noise power
spectral density is turned either into random phase traces (their Eq. 104), which
plug straight into a protocol's ``phase_*_rad`` callback, or into a reusable filter
kernel whose reweighting by any PSD gives the ensemble-mean error to second order.

Like :mod:`ryd_gate.physics` this is an expert module, not a top-level export.

Conventions
-----------
All densities here are **one-sided**: ``sigma_nu**2 = int_0^inf S_dnu(f) df``, the
convention laboratory instruments report. The paper uses two-sided densities (its
Sec. II C), so its closed forms are a factor 2 away; conversions happen at the
comparison, never silently inside these functions.

``S_dnu(f) = f**2 S_phi(f)`` and ``d(phi)/dt = 2 pi dnu(t)``.

Frequency multiplication (297 nm is the fourth harmonic of a 1188 nm seed)
multiplies the optical phase by ``harmonic`` and hence ``S_dnu`` by
``harmonic**2``.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "PhaseNoisePSD",
]


class PhaseNoisePSD:
    """One-sided frequency-noise density, measured samples or analytic.

    Parameters
    ----------
    f_hz, s_dnu
        Measured samples of the *fundamental* density (Hz, Hz^2/Hz), strictly
        increasing in ``f_hz``. Empty arrays select the analytic branch.
    harmonic
        Frequency-multiplication factor; ``S_dnu`` is scaled by its square.
    extrapolation
        Behaviour above the highest measured frequency: ``"flat"`` holds the edge
        value, ``"power"`` continues the power law fitted over
        ``power_law_fit_hz``. Below the lowest measured frequency the lowest
        measured slope is continued.
    """

    def __init__(self, f_hz, s_dnu, *, harmonic: int = 1,
                 extrapolation: str = "flat",
                 power_law_fit_hz: tuple[float, float] = (1e5, 1e6),
                 white_h0: float = 0.0, servo_bump=None) -> None:
        if extrapolation not in ("flat", "power"):
            raise ValueError(f"extrapolation must be 'flat' or 'power'; got {extrapolation!r}")
        self.f_hz = np.asarray(f_hz, dtype=float)
        self.s_meas = np.asarray(s_dnu, dtype=float)
        if self.f_hz.size and np.any(np.diff(self.f_hz) <= 0.0):
            raise ValueError("f_hz must be strictly increasing")
        self.harmonic = int(harmonic)
        self.extrapolation = extrapolation
        self.power_law_fit_hz = power_law_fit_hz
        self.white_h0 = float(white_h0)
        self.servo_bump = servo_bump

    @classmethod
    def from_csv(cls, path, **kwargs) -> "PhaseNoisePSD":
        """Read a digitized ``results/297_laser_noise/psd_*.csv`` (ASD column)."""
        rows = np.genfromtxt(path, delimiter=",", comments="#", skip_header=1)
        rows = np.atleast_2d(rows)
        return cls(rows[:, 0], rows[:, 1] ** 2, **kwargs)

    @classmethod
    def white(cls, h0: float, *, servo_bump=None, harmonic: int = 1) -> "PhaseNoisePSD":
        """Analytic white floor plus an optional Gaussian servo bump.

        ``servo_bump = (s_g, f_g, sigma_g)`` with ``s_g`` the bump's dimensionless
        total phase-noise power (paper Eq. 41), so its one-sided ``S_dnu`` is
        ``s_g f_g**2 / (sqrt(2 pi) sigma_g) exp(-(f - f_g)**2 / (2 sigma_g**2))``.
        """
        return cls(np.empty(0), np.empty(0), harmonic=harmonic,
                   white_h0=h0, servo_bump=servo_bump)

    @property
    def power_law_exponent(self) -> float:
        """Least-squares ``p`` of ASD ~ f**-p over ``power_law_fit_hz``."""
        lo, hi = self.power_law_fit_hz
        m = (self.f_hz >= lo) & (self.f_hz <= hi)
        return -float(np.polyfit(np.log10(self.f_hz[m]),
                                 0.5 * np.log10(self.s_meas[m]), 1)[0])

    def s_dnu(self, f) -> np.ndarray:
        """One-sided frequency-noise density (Hz^2/Hz) at the working harmonic."""
        f = np.asarray(f, dtype=float)
        out = np.full(f.shape, self.white_h0, dtype=float)
        if self.servo_bump is not None:
            s_g, f_g, sig = self.servo_bump
            out = out + (s_g * f_g ** 2 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(
                -0.5 * ((f - f_g) / sig) ** 2)
        if self.f_hz.size:
            out = out + self._interpolated(f)
        return self.harmonic ** 2 * out

    def s_phi(self, f) -> np.ndarray:
        """One-sided phase-noise density (rad^2/Hz), ``S_dnu(f) / f**2``."""
        f = np.asarray(f, dtype=float)
        return self.s_dnu(f) / f ** 2

    def _interpolated(self, f: np.ndarray) -> np.ndarray:
        safe = np.clip(f, self.f_hz[0] * 1e-12, None)
        s = 10.0 ** np.interp(np.log10(safe), np.log10(self.f_hz),
                              np.log10(self.s_meas))
        hi = f > self.f_hz[-1]
        if np.any(hi):
            if self.extrapolation == "flat":
                s = np.where(hi, self.s_meas[-1], s)
            else:
                s = np.where(hi, self.s_meas[-1]
                             * (safe / self.f_hz[-1]) ** (-2.0 * self.power_law_exponent), s)
        lo = f < self.f_hz[0]
        if np.any(lo):
            slope = (np.log10(self.s_meas[1] / self.s_meas[0])
                     / np.log10(self.f_hz[1] / self.f_hz[0]))
            s = np.where(lo, self.s_meas[0] * (safe / self.f_hz[0]) ** slope, s)
        return s

    def sigma_nu(self, f_lo: float, f_hi: float, n: int = 20001) -> float:
        """RMS frequency deviation from the band ``[f_lo, f_hi]`` (Hz)."""
        lo = max(float(f_lo), 1e-12)
        f = np.logspace(np.log10(lo), np.log10(float(f_hi)), n)
        return float(np.sqrt(np.trapezoid(self.s_dnu(f), f)))
