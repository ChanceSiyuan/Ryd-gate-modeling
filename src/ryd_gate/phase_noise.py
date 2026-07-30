"""Laser phase noise from a measured frequency-noise PSD (expert module).

The noise model of X. Jiang, J. Scott, M. Friesen and M. Saffman, *Sensitivity of
quantum gate fidelity to laser phase and intensity noise*, Phys. Rev. A **107**,
042611 (2023) (arXiv:2210.11007): a measured or analytic frequency-noise power
spectral density, and the random realizations of the phase process ``phi_n(t)``
it generates (the paper's Eq. 104) for a time-domain solver to put on the drive.

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
from scipy.interpolate import CubicSpline

__all__ = [
    "PhaseNoisePSD",
    "PhaseTrace",
    "phase_trace",
    "log_frequency_bins",
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
        # np.interp holds the edge sample outside the measured range, which is already
        # the "flat" policy above the top sample; only "power" needs an override.
        s = 10.0 ** np.interp(np.log10(safe), np.log10(self.f_hz),
                              np.log10(self.s_meas))
        hi = f > self.f_hz[-1]
        if self.extrapolation == "power" and np.any(hi):
            s = np.where(hi, self.s_meas[-1]
                         * (safe / self.f_hz[-1]) ** (-2.0 * self.power_law_exponent), s)
        lo = f < self.f_hz[0]
        if np.any(lo):
            slope = (np.log10(self.s_meas[1] / self.s_meas[0])
                     / np.log10(self.f_hz[1] / self.f_hz[0]))
            s = np.where(lo, self.s_meas[0] * (safe / self.f_hz[0]) ** slope, s)
        return s

    def sigma_nu(self, f_lo: float, f_hi: float, n: int = 20001) -> float:
        """RMS frequency deviation from the band ``[f_lo, f_hi]`` (Hz).

        ``f_lo`` is the modelling parameter ``f_min``, physically the inverse of the
        calibration/relock timescale, and has no default: a measured ``S_dnu`` rises
        steeply towards DC and is extrapolated below the measurement edge, so an
        unbounded band would fabricate power rather than report it.
        """
        if float(f_lo) <= 0.0:
            raise ValueError(f"f_lo must be positive; got {f_lo!r}")
        f = np.logspace(np.log10(float(f_lo)), np.log10(float(f_hi)), n)
        return float(np.sqrt(np.trapezoid(self.s_dnu(f), f)))


def log_frequency_bins(f_lo: float, f_hi: float, points_per_decade: int):
    """Logarithmic bin centres and widths spanning ``[f_lo, f_hi]``.

    Edges are equally spaced in log10 so every bin's width is proportional to its
    centre; the centres are the geometric bin midpoints.

    An inverted span (``f_max`` below ``1/t_gate``, say) would otherwise return
    descending edges and negative widths, and ``sqrt`` of a negative amplitude turns
    the whole trace into NaN with nothing raised anywhere.
    """
    if not 0.0 < f_lo < f_hi:
        raise ValueError(f"need 0 < f_lo < f_hi; got f_lo={f_lo!r}, f_hi={f_hi!r}")
    n = max(1, int(round(np.log10(f_hi / f_lo) * points_per_decade)))
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n + 1)
    return np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)


class PhaseTrace:
    """One realization of ``phi_n(t)`` on ``[0, t_gate]`` (radians).

    ``values`` are the sampled phase, ``dnu_0`` the drawn quasi-static frequency
    offset (Hz) whose ``2 pi dnu_0 t`` ramp is already included, and
    ``f_grid``/``df_grid`` the explicitly summed band and its bin widths.
    """

    __slots__ = ("times", "values", "dnu_0", "f_grid", "df_grid", "_spline")

    def __init__(self, times, values, dnu_0, f_grid, df_grid):
        self.times = times
        self.values = values
        self.dnu_0 = float(dnu_0)
        self.f_grid = f_grid
        self.df_grid = df_grid
        self._spline = CubicSpline(times, values)

    def __call__(self, t):
        """Phase (rad) at ``t``, which must lie within ``[0, t_gate]``.

        Solver times reach the endpoint through accumulated arithmetic and can sit a
        rounding error outside it, so an excursion of a few ulp is clipped. A larger
        one is an error: clipping it would freeze the dominant ``2 pi dnu_0 t`` ramp
        at its final value instead of accumulating, which reads downstream as a
        plausible trace and a silently optimistic infidelity.
        """
        t = np.asarray(t, dtype=float)
        lo, hi = self.times[0], self.times[-1]
        tol = 4.0 * np.spacing(hi)
        if np.any(t < lo - tol) or np.any(t > hi + tol):
            raise ValueError(f"t must lie in [{lo}, {hi}]; got {t.min()} .. {t.max()}")
        return self._spline(np.clip(t, lo, hi))


def phase_trace(psd: PhaseNoisePSD, t_gate: float, *, seed: int,
                f_max: float, f_min: float = 1.0,
                points_per_decade: int = 40,
                n_samples: int = 4096) -> PhaseTrace:
    """A random phase realization for ``psd`` over ``[0, t_gate]`` (paper Eq. 104).

    The frequency axis is hybrid because a uniform grid from ``f_min`` to ``f_max``
    would need ~1e8 terms. Below ``1/t_gate`` the noise is frozen over the gate and
    is collapsed into one Gaussian quasi-static offset ``dnu_0`` of the correct
    band variance; above it a logarithmic grid is summed explicitly as

        phi(t) = 2 pi dnu_0 t + sum_j sqrt(2 S_phi(f_j) df_j) cos(2 pi f_j t + psi_j)

    which is the paper's ``2 sqrt(S^2s df)`` rewritten for one-sided densities.

    Only the phases ``psi_j`` are random, the amplitudes being fixed, so the ensemble
    reproduces the correct second moments but not full Gaussian marginals. The bin
    variances ``2 S_phi df`` fall as ``1/f`` for a white ``S_dnu``, so the
    participation ratio ``(sum a**2)**2 / sum a**4`` saturates near 35 tones however
    wide the band, and fewer for a steeper measured spectrum: near-Gaussian in the
    bulk, thinner in the tails. Averages of quantities nonlinear in the phase,
    infidelity among them, inherit that.
    """
    rng = np.random.default_rng(seed)
    f_split = 1.0 / t_gate
    f_grid, df_grid = log_frequency_bins(f_split, f_max, points_per_decade)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=f_grid.size)
    amp = np.sqrt(2.0 * psd.s_phi(f_grid) * df_grid)
    dnu_0 = psd.sigma_nu(f_min, f_split) * float(rng.standard_normal())

    times = np.linspace(0.0, t_gate, n_samples)
    values = (2.0 * np.pi * dnu_0) * times + (
        amp[None, :] * np.cos(2.0 * np.pi * np.outer(times, f_grid) + psi[None, :])
    ).sum(axis=1)
    return PhaseTrace(times, values, dnu_0, f_grid, df_grid)
