# Laser Phase Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate arbitrary laser phase noise from a measured PSD, and re-render the `results/max_leakage_297/a3.0` map family once per measured 297 nm laser with a power↔Rabi table, so the minimum required laser power can be read off.

**Architecture:** A new expert module `ryd_gate.phase_noise` turns a frequency-noise PSD into random phase traces (PRA 107, 042611 Eq. 104) that plug into the existing protocol `phase_*_rad` callbacks, and into a reusable *filter kernel* whose reweighting by any PSD gives the ensemble-mean error with no statistical noise. A new `filter` subcommand in `scripts/max_leakage_297_sweep.py` computes that kernel once over the whole grid; plotting then adds an `eps_phase` metric and a power table.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`solve_ivp`/DOP853), Matplotlib, pytest. ARC only for the power table (cached to npz).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-laser-phase-noise-design.md`. Read it before starting.
- **All spectral densities in this codebase are ONE-SIDED**: `sigma_nu**2 = int_0^inf S_dnu(f) df`. The paper's are two-sided (its Sec. II C). Every closed form quoted from the paper must be divided by 2 when compared against this repository's densities.
- `S_dnu(f) = f**2 * S_phi(f)`; `phi_dot = 2*pi*dnu`.
- **297 nm is the 4th harmonic** of the measured 1180/1187 nm fundamental: `S_dnu(297) = 16 * S_dnu(fundamental)`. The harmonic is a parameter, default 1.
- `f_min = 1.0` Hz everywhere by default (the measurement edge); it is a real modelling parameter, never hard-coded at a call site.
- Filter-kernel frequency grid, fixed and global: `f_min = 1.0` Hz, `f_max = 2.0e8` Hz, 200 fine points per decade integrated into 30 storage bins per decade.
- `ryd_gate.phase_noise` is an **expert module like `ryd_gate.physics`** — do NOT add it to `src/ryd_gate/__init__.py`'s `__all__`; that file documents its namespace as "exactly the seven names below".
- Existing code is not modified except where a task says so. The coherent (`chunks/`) and scatter (`scatter/`) series of the live store are never rewritten.
- Run tests on the DGX: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest -q <paths>'`. Pure-NumPy tests also run locally with `PYTHONPATH=src python3 -m pytest`.
- Match the surrounding style: module docstrings explaining *why*, no speculative configurability, no defensive handling of impossible states.

## File Structure

| File | Responsibility |
|---|---|
| `src/ryd_gate/phase_noise.py` (new) | `PhaseNoisePSD`, `phase_trace`/`PhaseTrace`, `log_frequency_bins`, `filter_kernel`, `error_from_kernel`. Model-independent; knows nothing about 297 nm or the sweep. |
| `tests/test_phase_noise.py` (new) | Unit tests for all of the above, including the PRA 107.042611 literature check. |
| `scripts/laser_noise_psd.py` (exists) | Digitization + the model figure. Gains a JSON emit so one source of truth feeds the sweep and the plots. |
| `scripts/max_leakage_297_sweep.py` (exists) | Gains `integrate_adjoint_batch`, the `filter` subcommand and its store series, the `eps_phase`/`total_error_phase` plot metrics, and the power table. |
| `scripts/sweeplib/store.py` (exists) | Gains the `filter/` append-only series (mirrors `scatter/`). |
| `scripts/sweeplib/plotting.py` (exists) | Gains an optional table strip under the panel grid. |
| `tests/test_max_leakage_297_sweep.py` (exists) | Gains coverage of the new subcommand, series and metrics. |

---

### Task 1: `PhaseNoisePSD` — the spectrum model

**Files:**
- Create: `src/ryd_gate/phase_noise.py`
- Create: `tests/test_phase_noise.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PhaseNoisePSD(f_hz, s_dnu, *, harmonic=1, extrapolation="flat", power_law_fit_hz=(1e5, 1e6))`
  - `PhaseNoisePSD.from_csv(path, **kwargs) -> PhaseNoisePSD` (reads the 4-column `results/297_laser_noise/psd_*.csv`, whose second column is the mean **ASD**, `sqrt(S_dnu)`)
  - `PhaseNoisePSD.white(h0, *, servo_bump=None) -> PhaseNoisePSD`, `servo_bump=(s_g, f_g, sigma_g)`
  - `.s_dnu(f) -> np.ndarray` (one-sided Hz^2/Hz, harmonic applied)
  - `.s_phi(f) -> np.ndarray`
  - `.sigma_nu(f_lo, f_hi) -> float`
  - `.power_law_exponent -> float` (property; ASD ~ f^-p over `power_law_fit_hz`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase_noise.py
"""PSD model + phase-trace generation (PRA 107, 042611).

One-sided densities throughout: sigma_nu**2 = int_0^inf S_dnu df. The paper uses
two-sided densities, so its closed forms are halved when compared here.
"""

import numpy as np
import pytest

from ryd_gate.phase_noise import PhaseNoisePSD


def test_white_psd_is_flat_and_s_phi_falls_as_f_squared():
    psd = PhaseNoisePSD.white(100.0)
    f = np.array([1.0, 1e3, 1e6])
    assert np.allclose(psd.s_dnu(f), 100.0)
    assert np.allclose(psd.s_phi(f), 100.0 / f**2)


def test_white_psd_sigma_nu_is_sqrt_h0_times_bandwidth():
    psd = PhaseNoisePSD.white(100.0)
    assert psd.sigma_nu(0.0, 1e4) == pytest.approx(np.sqrt(100.0 * 1e4), rel=1e-6)


def test_harmonic_scales_s_dnu_by_its_square():
    base = PhaseNoisePSD.white(100.0)
    quad = PhaseNoisePSD.white(100.0, harmonic=4)
    assert quad.s_dnu(np.array([1e6])) == pytest.approx(16 * base.s_dnu(np.array([1e6])))


def test_servo_bump_adds_a_gaussian_peak_of_the_requested_total_power():
    # s_g is the dimensionless total phase-noise power of the bump (paper Eq. 41);
    # the bump's S_dnu integral over positive f is therefore s_g * f_g**2.
    s_g, f_g, sigma_g = 1e-4, 1.0e6, 2.0e4
    psd = PhaseNoisePSD.white(0.0, servo_bump=(s_g, f_g, sigma_g))
    f = np.linspace(f_g - 6 * sigma_g, f_g + 6 * sigma_g, 20001)
    assert np.trapezoid(psd.s_dnu(f), f) == pytest.approx(s_g * f_g**2, rel=1e-3)


def test_measured_psd_interpolates_in_log_log_and_holds_flat_above_the_edge():
    f = np.array([1e3, 1e4, 1e5, 1e6])
    asd = np.array([1e3, 1e2, 1e1, 1e0])            # exactly ASD ~ f^-1
    psd = PhaseNoisePSD(f, asd**2, extrapolation="flat")
    assert psd.s_dnu(np.array([10**3.5]))[0] == pytest.approx((10**2.5) ** 2, rel=1e-9)
    assert psd.s_dnu(np.array([1e8]))[0] == pytest.approx(1.0, rel=1e-9)


def test_power_law_extrapolation_continues_the_fitted_slope():
    f = np.array([1e3, 1e4, 1e5, 1e6])
    asd = np.array([1e3, 1e2, 1e1, 1e0])            # p = 1 exactly
    psd = PhaseNoisePSD(f, asd**2, extrapolation="power",
                        power_law_fit_hz=(1e5, 1e6))
    assert psd.power_law_exponent == pytest.approx(1.0, abs=1e-9)
    assert psd.s_dnu(np.array([1e7]))[0] == pytest.approx(1e-2, rel=1e-9)


def test_from_csv_reads_the_digitized_measurement(tmp_path):
    p = tmp_path / "psd.csv"
    p.write_text("f_Hz,asd_mean_Hz_per_rtHz,asd_lo,asd_hi\n"
                 "# comment line\n"
                 "1e3,1e3,1,1\n1e6,1e0,1,1\n")
    psd = PhaseNoisePSD.from_csv(p, harmonic=4)
    assert psd.s_dnu(np.array([1e6]))[0] == pytest.approx(16.0, rel=1e-9)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ryd_gate.phase_noise'`

- [ ] **Step 3: Implement `PhaseNoisePSD`**

```python
# src/ryd_gate/phase_noise.py
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
    "PhaseTrace",
    "phase_trace",
    "log_frequency_bins",
    "filter_kernel",
    "error_from_kernel",
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Emit the model parameters from the digitizer**

In `scripts/laser_noise_psd.py`, add to `main()` right after the CSVs are written:

```python
    import json
    model = {
        laser: {
            "csv": f"psd_{laser}.csv",
            "harmonic": HARMONIC,
            "power_law_exponent": power_law_exponent(d),
            "s_dnu_edge_297": HARMONIC ** 2 * float(d[-1, 1]) ** 2,
            "f_edge_hz": float(d[-1, 0]),
        }
        for laser, d in datasets.items()
    }
    path = os.path.join(NOISE_DIR, "psd_model.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)
    print(f"wrote {path}")
```

- [ ] **Step 6: Run the digitizer and check the JSON**

Run: `python3 scripts/laser_noise_psd.py && cat results/297_laser_noise/psd_model.json`
Expected: both lasers present; `power_law_exponent` ≈ 0.463 (ECDL) and 0.554 (seed); `s_dnu_edge_297` ≈ 2444 and 103.

- [ ] **Step 7: Commit**

```bash
git add src/ryd_gate/phase_noise.py tests/test_phase_noise.py scripts/laser_noise_psd.py results/297_laser_noise/psd_model.json
git commit -m "Add PhaseNoisePSD: measured and analytic frequency-noise spectra"
```

---

### Task 2: `phase_trace` — the stochastic phase process

**Files:**
- Modify: `src/ryd_gate/phase_noise.py`
- Modify: `tests/test_phase_noise.py`

**Interfaces:**
- Consumes: `PhaseNoisePSD` from Task 1.
- Produces:
  - `phase_trace(psd, t_gate, *, seed, f_min=1.0, f_max, points_per_decade=40, n_samples=4096) -> PhaseTrace`
  - `PhaseTrace` with attributes `times` (n_samples,), `values` (n_samples,), `dnu_0` (float), `f_grid`, `df_grid`, and `__call__(t)` returning cubic-spline-interpolated radians.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_phase_noise.py`:

```python
from ryd_gate.phase_noise import phase_trace


def test_trace_is_reproducible_and_seed_dependent():
    psd = PhaseNoisePSD.white(1e4)
    a = phase_trace(psd, 1e-6, seed=7, f_max=1e8)
    b = phase_trace(psd, 1e-6, seed=7, f_max=1e8)
    c = phase_trace(psd, 1e-6, seed=8, f_max=1e8)
    assert np.array_equal(a.values, b.values)
    assert not np.allclose(a.values, c.values)


def test_trace_callable_matches_its_samples():
    psd = PhaseNoisePSD.white(1e4)
    tr = phase_trace(psd, 1e-6, seed=3, f_max=1e8)
    mid = 0.5 * (tr.times[:-1] + tr.times[1:])
    # the spline reproduces its own nodes exactly and stays close between them
    assert np.allclose(tr(tr.times), tr.values, atol=1e-12)
    assert np.max(np.abs(tr(mid) - np.interp(mid, tr.times, tr.values))) < 1e-3


def test_resolved_band_variance_matches_the_psd_integral():
    # var(phi) over the explicitly summed band equals int 2 S_phi df, since each
    # term sqrt(2 S_phi df) cos(...) contributes half its squared amplitude.
    psd = PhaseNoisePSD.white(1e6)
    t_gate, f_max = 1e-6, 1e8
    traces = [phase_trace(psd, t_gate, seed=s, f_max=f_max) for s in range(400)]
    resolved = np.asarray([tr.values - 2 * np.pi * tr.dnu_0 * tr.times for tr in traces])
    expected = float(np.sum(psd.s_phi(traces[0].f_grid) * traces[0].df_grid))
    assert np.var(resolved) == pytest.approx(expected, rel=0.15)


def test_quasi_static_offset_matches_the_unresolved_band():
    psd = PhaseNoisePSD.white(1e6)
    t_gate = 1e-6
    offsets = np.asarray([phase_trace(psd, t_gate, seed=s, f_max=1e8).dnu_0
                          for s in range(2000)])
    expected = psd.sigma_nu(1.0, 0.01 / t_gate)
    assert np.std(offsets) == pytest.approx(expected, rel=0.1)


def test_generated_traces_reproduce_the_input_spectrum():
    from scipy.signal import welch
    psd = PhaseNoisePSD.white(1e4)
    t_gate, n = 4e-6, 16384
    traces = [phase_trace(psd, t_gate, seed=s, f_max=2e8, n_samples=n)
              for s in range(60)]
    dt = traces[0].times[1] - traces[0].times[0]
    dnu = np.gradient(np.asarray([tr.values for tr in traces]), dt, axis=1) / (2 * np.pi)
    f, pxx = welch(dnu, fs=1.0 / dt, nperseg=n // 8, axis=1)
    band = (f > 5e6) & (f < 5e7)
    assert np.median(pxx.mean(axis=0)[band]) == pytest.approx(1e4, rel=0.35)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q -k trace`
Expected: FAIL — `ImportError: cannot import name 'phase_trace'`

- [ ] **Step 3: Implement `phase_trace`**

Append to `src/ryd_gate/phase_noise.py`:

```python
class PhaseTrace:
    """One realization of ``phi_n(t)`` on ``[0, t_gate]`` (radians).

    ``values`` are the sampled phase, ``dnu_0`` the drawn quasi-static frequency
    offset (Hz) whose ``2 pi dnu_0 t`` ramp is already included, and
    ``f_grid``/``df_grid`` the explicitly summed band and its bin widths.
    """

    __slots__ = ("times", "values", "dnu_0", "f_grid", "df_grid", "_spline")

    def __init__(self, times, values, dnu_0, f_grid, df_grid):
        from scipy.interpolate import CubicSpline

        self.times = times
        self.values = values
        self.dnu_0 = float(dnu_0)
        self.f_grid = f_grid
        self.df_grid = df_grid
        self._spline = CubicSpline(times, values)

    def __call__(self, t):
        return self._spline(np.clip(t, self.times[0], self.times[-1]))


def phase_trace(psd: PhaseNoisePSD, t_gate: float, *, seed: int,
                f_max: float, f_min: float = 1.0,
                points_per_decade: int = 40,
                n_samples: int = 4096) -> PhaseTrace:
    """A random phase realization for ``psd`` over ``[0, t_gate]`` (paper Eq. 104).

    The frequency axis is hybrid because a uniform grid from ``f_min`` to ``f_max``
    would need ~1e8 terms. Below ``0.01/t_gate`` the noise is frozen over the gate and
    is collapsed into one Gaussian quasi-static offset ``dnu_0`` of the correct
    band variance; above it a logarithmic grid is summed explicitly as

        phi(t) = 2 pi dnu_0 t + sum_j sqrt(2 S_phi(f_j) df_j) cos(2 pi f_j t + psi_j)

    which is the paper's ``2 sqrt(S^2s df)`` rewritten for one-sided densities.
    """
    rng = np.random.default_rng(seed)
    # two decades below 1/t_gate: collapsing a band substitutes |G(0)|^2 for the
    # true |G(f)|^2, which only holds for f << 1/t_gate.  At the boundary itself
    # this mismodels the band by +65% / -63% depending on rotation count.
    f_split = 0.01 / t_gate
    f_grid, df_grid = log_frequency_bins(f_split, f_max, points_per_decade)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=f_grid.size)
    amp = np.sqrt(2.0 * psd.s_phi(f_grid) * df_grid)
    dnu_0 = psd.sigma_nu(f_min, f_split) * float(rng.standard_normal())

    times = np.linspace(0.0, t_gate, n_samples)
    values = (2.0 * np.pi * dnu_0) * times + (
        amp[None, :] * np.cos(2.0 * np.pi * np.outer(times, f_grid) + psi[None, :])
    ).sum(axis=1)
    return PhaseTrace(times, values, dnu_0, f_grid, df_grid)
```

- [ ] **Step 4: Implement `log_frequency_bins` (used by both branches)**

Insert above `PhaseTrace`:

```python
def log_frequency_bins(f_lo: float, f_hi: float, points_per_decade: int):
    """Logarithmic bin centres and widths spanning ``[f_lo, f_hi]``.

    Edges are equally spaced in log10 so every bin's width is proportional to its
    centre; the centres are the geometric bin midpoints.
    """
    n = max(1, int(round(np.log10(f_hi / f_lo) * points_per_decade)))
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n + 1)
    return np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q`
Expected: PASS (12 tests). The Welch test takes ~20 s.

- [ ] **Step 6: Commit**

```bash
git add src/ryd_gate/phase_noise.py tests/test_phase_noise.py
git commit -m "Generate random phase traces from a PSD (PRA 107.042611 Eq. 104)"
```

---

### Task 3: Filter kernel + the literature validation

**Files:**
- Modify: `src/ryd_gate/phase_noise.py`
- Modify: `tests/test_phase_noise.py`

**Interfaces:**
- Consumes: `PhaseNoisePSD`, `phase_trace`, `log_frequency_bins` from Tasks 1–2.
- Produces:
  - `filter_kernel(times, components, f_bins, df_bins, *, fine_per_decade=200) -> np.ndarray` — `K_b`, shape `(n_bins,)`, where `components` is the `(n_t, n_comp)` array of `<q|A(t)>`. Returns `int_bin (||G(f)||**2 + ||G(-f)||**2) df`.
  - `error_from_kernel(psd, f_bins, kernel) -> float` — `2*pi**2 * sum_b S_dnu(f_b) * K_b`.

Note the kernel already carries the `df` integration, so `error_from_kernel` must
NOT multiply by `df_bins` again.

> **Superseded by Task 5 — the code below this line is a historical record, not
> the shipped interface.** Two changes landed in `d06e002`:
>
> 1. **The metric.** `<Delta L>` (the second-order increase of the *leakage*
>    observable) is provably wrong — it keeps `<||Q chi_1||**2>` but drops
>    `2 Re <Q psi_0 | Q chi_2>` of the same order; setting `Q = 1` makes the true
>    change identically zero while the formula returns a positive number. The
>    shipped metric is the noise-induced **fidelity loss**
>    `eps_phase = max_s [1 - |<psi_0^s(T)|psi^s(T)>|**2]`, second-order exact by
>    construction. `filter_kernel` therefore takes a `subtract` argument carrying
>    `<psi_0(T)|A(t)>`, and `||G||**2` runs over the **complete** 16-state basis
>    (not the 12 nonlogical ones).
> 2. **`fine_per_decade`.** The flat `200` under-resolves the `1/T` fringes: a log
>    grid of `p` per decade has spacing `f ln10 / p`, so `p >= ln10 * f_max * T` is
>    required (461 at `T = 1 us`, 2073 at 4.5 us; a flat 200 puts the 4.5 us kernel
>    ~13% high). The sizing rule is `kernel_fine_per_decade` in the sweep script.
>    The library default is still `200` — **callers at `T > ~0.43 us` must pass an
>    explicit value.**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_phase_noise.py`:

```python
from ryd_gate.phase_noise import error_from_kernel, filter_kernel, log_frequency_bins


def _rabi_pi_pulse_components(omega0, n_rot, n_t):
    """<perp|A(t)> for a resonant two-level Rabi drive of N rotations.

    A(t) = U(T,t) N_e psi(t) with N_e = |e><e|; the metric projects onto the
    single direction orthogonal to the ideal final state, so there is one
    component. Everything is available in closed form for a constant drive:
    U(t) = cos(omega0 t/2) I - i sin(omega0 t/2) sigma_x.
    """
    t_gate = 2.0 * np.pi * n_rot / omega0
    t = np.linspace(0.0, t_gate, n_t)

    def u(tau):
        c, s = np.cos(0.5 * omega0 * tau), np.sin(0.5 * omega0 * tau)
        return np.array([[c, -1j * s], [-1j * s, c]])

    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi_T = u(t_gate) @ psi0
    perp = np.array([-np.conj(psi_T[1]), np.conj(psi_T[0])])
    comp = np.empty((n_t, 1), dtype=complex)
    for k, tk in enumerate(t):
        psi_k = u(tk) @ psi0
        a_k = u(t_gate - tk) @ (np.array([0.0, 1.0]) * psi_k)   # N_e psi
        comp[k, 0] = np.vdot(perp, a_k)
    return t, comp, t_gate


def test_filter_kernel_reproduces_the_paper_white_noise_gate_error():
    # Paper Eq. 79 (initial state |0>): eps = pi**3 h0 N / Omega_0, with h0
    # TWO-SIDED. Against our one-sided h0 the target is half that.
    omega0 = 2 * np.pi * 1e6
    h0_onesided = 200.0
    for n_rot in (0.5, 1.0):
        t, comp, t_gate = _rabi_pi_pulse_components(omega0, n_rot, 200001)
        f_bins, df_bins = log_frequency_bins(1e2, 1e9, 60)
        kernel = filter_kernel(t, comp, f_bins, df_bins)
        eps = error_from_kernel(PhaseNoisePSD.white(h0_onesided), f_bins, kernel)
        expected = np.pi**3 * (h0_onesided / 2) * n_rot / omega0
        assert eps == pytest.approx(expected, rel=0.05)


def test_filter_kernel_agrees_with_direct_monte_carlo():
    """Same two-level pi pulse, integrated with real phase traces."""
    from scipy.integrate import solve_ivp

    omega0, h0 = 2 * np.pi * 1e6, 200.0
    n_rot = 0.5
    t_gate = 2.0 * np.pi * n_rot / omega0
    psd = PhaseNoisePSD.white(h0)

    def run(trace):
        def rhs(tt, y):
            c = 0.5 * omega0 * np.exp(-1j * trace(tt))
            return -1j * np.array([c * y[1], np.conj(c) * y[0]])
        sol = solve_ivp(rhs, (0.0, t_gate), np.array([1.0, 0.0], dtype=complex),
                        method="DOP853", rtol=1e-10, atol=1e-13)
        return sol.y[:, -1]

    ideal = run(_ZERO_TRACE(t_gate))
    errs = []
    for s in range(300):
        psi = run(phase_trace(psd, t_gate, seed=s, f_max=2e8, n_samples=8192))
        errs.append(1.0 - abs(np.vdot(ideal, psi)) ** 2)
    mc = float(np.mean(errs))

    t, comp, _ = _rabi_pi_pulse_components(omega0, n_rot, 200001)
    f_bins, df_bins = log_frequency_bins(1e2, 1e9, 60)
    predicted = error_from_kernel(psd, f_bins,
                                 filter_kernel(t, comp, f_bins, df_bins))
    stderr = float(np.std(errs) / np.sqrt(len(errs)))
    assert abs(mc - predicted) < 4 * stderr + 0.05 * predicted


class _ZERO_TRACE:
    """Noiseless stand-in with the PhaseTrace call signature."""

    def __init__(self, t_gate):
        self.t_gate = t_gate

    def __call__(self, t):
        return 0.0 * np.asarray(t, dtype=float)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q -k kernel`
Expected: FAIL — `ImportError: cannot import name 'filter_kernel'`

- [ ] **Step 3: Implement the kernel**

Append to `src/ryd_gate/phase_noise.py`:

```python
def filter_kernel(times, components, f_bins, df_bins, *,
                  fine_per_decade: int = 200) -> np.ndarray:
    """Bin-integrated response ``K_b`` of a gate to frequency noise.

    ``components`` is the ``(n_t, n_comp)`` array of ``<q|A(t)>`` with
    ``A(t) = U(T,t) N_r psi(t)``; ``<Delta L> = 2 pi**2 sum_b S_dnu(f_b) K_b`` with

        K_b = int_bin ( ||G(f)||**2 + ||G(-f)||**2 ) df ,
        G(f) = int_0^T A(t) exp(-2 pi i f t) dt .

    ``K`` carries fringe structure on the ``1/T`` scale, far finer than the storage
    bins, so it is evaluated on a ``fine_per_decade`` grid and integrated into the
    bins rather than point-sampled at their centres.
    """
    times = np.asarray(times, dtype=float)
    comp = np.asarray(components, dtype=np.complex128)
    edges = np.concatenate([f_bins - 0.5 * df_bins, [f_bins[-1] + 0.5 * df_bins[-1]]])
    fine, dfine = log_frequency_bins(max(edges[0], 1e-12), edges[-1],
                                     fine_per_decade)

    weights = np.gradient(times)
    kern = np.empty(fine.size)
    # (n_fine, n_t) @ (n_t, n_comp) -> (n_fine, n_comp), one BLAS call per sign.
    for sign in (-1.0, +1.0):
        phase = np.exp(-2j * np.pi * sign * np.outer(fine, times)) * weights[None, :]
        g = phase @ comp
        contrib = np.einsum("fc,fc->f", g.conj(), g).real
        kern = contrib if sign < 0 else kern + contrib

    idx = np.clip(np.searchsorted(edges, fine) - 1, 0, f_bins.size - 1)
    out = np.zeros(f_bins.size)
    np.add.at(out, idx, kern * dfine)
    return out


def error_from_kernel(psd: PhaseNoisePSD, f_bins, kernel) -> float:
    """``<Delta L> = 2 pi**2 sum_b S_dnu(f_b) K_b`` (K_b already carries its df)."""
    return float(2.0 * np.pi**2 * np.sum(psd.s_dnu(np.asarray(f_bins))
                                         * np.asarray(kernel)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python3 -m pytest tests/test_phase_noise.py -q`
Expected: PASS (14 tests). The Monte Carlo test takes ~2 min.

If the literature test fails by a clean factor (2, 4, …), the convention is wrong somewhere — fix the code, not the expected value. The three legs (closed form, kernel, Monte Carlo) are independent; two agreeing against one identifies the bug.

- [ ] **Step 5: Commit**

```bash
git add src/ryd_gate/phase_noise.py tests/test_phase_noise.py
git commit -m "Add the phase-noise filter kernel, validated against PRA 107.042611 Eq. 79"
```

---

### Task 4: `filter` subcommand in the 297 sweep

**Files:**
- Modify: `scripts/sweeplib/store.py` (add the `filter/` series)
- Modify: `scripts/sweeplib/runner.py:117-134` (`Batch` gains `filter_pass`), `:322-333` (`_spec`), `:335-369` (`_write_success`), `:390-416` (`_write_failure`), `:418-430` (`_split`)
- Modify: `scripts/max_leakage_297_sweep.py`
- Modify: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Consumes: `log_frequency_bins`, `filter_kernel` from Tasks 1–3.
- Produces:
  - `KERNEL_F_MIN_HZ = 1.0`, `KERNEL_F_MAX_HZ = 2.0e8`, `KERNEL_BINS_PER_DECADE = 30`, `KERNEL_N_T = 4096`
  - `kernel_frequency_bins() -> tuple[np.ndarray, np.ndarray]`
  - `integrate_adjoint_batch(ops, t_gate, omega_297, d_sweep, *, rtol, atol, ramp, n_t) -> dict` with keys `times` (n_t,), `components` (n_points, 4, n_t, 12)
  - `filter_kernels(ops, t_gate, omega_297, d_sweep, *, rtol, atol, ramp, n_t) -> np.ndarray` of shape `(n_points, 4, n_bins)`
  - `Store.next_filter_seq()`, `Store.write_filter_chunk(...)`, `Store.load_filter_records(manifest) -> list[dict]` with keys `key`, `status`, `kernel` (4, n_bins), `rtol`, `runtime_s`
  - `cmd_filter(args)` wired to a `filter` subparser with `--level` (default `13`)

- [ ] **Step 1: Write the failing test for the backward leg**

Add to `tests/test_max_leakage_297_sweep.py`:

```python
def test_adjoint_leg_reproduces_the_forward_propagator(mls297):
    """<phi_q(t)|psi_s(t)> must equal <q|U(T,0)|s>, independent of t.

    The backward-integrated adjoints are the only new solver leg; this pins them
    against the existing, already-validated forward kernel.
    """
    import numpy as np

    cfg = mls297.ScanConfig()
    system = mls297.build_system(cfg, 53)
    ops = mls297.aggregate_operators(system, 53)
    t_gate, omega, dsw = 1e-6, 2 * np.pi * 13.5e6, 2 * np.pi * 15e6

    fwd = mls297.integrate_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-10, atol=1e-13, use_swap=False)
    out = mls297.integrate_adjoint_batch(
        ops, t_gate, np.array([omega]), np.array([dsw]),
        rtol=1e-10, atol=1e-13, ramp=cfg.ramp_frac, n_t=257)

    nonlogical = np.setdiff1d(np.arange(16), ops.logical_indices)
    for si in range(4):
        target = fwd.psi_final[0, si][nonlogical]
        # overlaps <phi_q(t)|psi_s(t)> are t-independent and equal <q|psi_s(T)>
        got = out["overlaps"][0, si]            # (n_t, 12)
        assert np.allclose(got, target[None, :], atol=1e-8)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py -q -k adjoint'`
Expected: FAIL — `AttributeError: module has no attribute 'integrate_adjoint_batch'`

- [ ] **Step 3: Implement the backward leg and the components**

Add to `scripts/max_leakage_297_sweep.py` after `integrate_batch`:

```python
# ── Filter-function pass: adjoint states and the frequency kernel ────────────
#
# The phase noise enters exactly as H -> H_0 + 2 pi dnu(t) N_r (V = exp(+i phi N_r)
# removes it from the drive), so first-order perturbation gives
#     <Delta L> = 2 pi^2 int S_dnu(|f|) ||Q G(f)||^2 df,
#     G(f) = int_0^T <q|U(T,t) N_r psi_s(t)> e^{-2 pi i f t} dt.
# Only the projected components are needed, so the propagator is never formed:
#     <q|A_s(t)> = <phi_q(t)| N_r |psi_s(t)>,   |phi_q(t)> = U(t,T)|q>.
# phi_q and psi_s obey the same equation and N_r is diagonal, so the exp(-i D_i t)
# factors cancel in the pointwise product and the sampled integrand carries only
# drive-scale structure — which is what makes n_t = 4096 enough despite the GHz
# pair interaction and the 6.8 GHz |0> hyperfine offset.

KERNEL_F_MIN_HZ = 1.0
KERNEL_F_MAX_HZ = 2.0e8
KERNEL_BINS_PER_DECADE = 30
KERNEL_N_T = 4096


def kernel_frequency_bins():
    """The fixed global storage bins shared by every point of the store."""
    from ryd_gate.phase_noise import log_frequency_bins

    return log_frequency_bins(KERNEL_F_MIN_HZ, KERNEL_F_MAX_HZ,
                              KERNEL_BINS_PER_DECADE)


def _297_adjoint_rhs_factory(ops, cols, t_gate, ramp):
    """Time-reversed RHS: dy/dtau = +i H(T - tau) y, same drive as the forward leg."""
    forward = _297_rhs_factory(ops, cols, t_gate, ramp)

    def rhs(tau, y):
        return -forward(t_gate - tau, y)

    return rhs


def integrate_adjoint_batch(ops, t_gate, omega_297, d_sweep, *,
                            rtol, atol, ramp=0.15, n_t=KERNEL_N_T):
    """Forward logical states + backward nonlogical adjoints, sampled together.

    Returns ``{"times": (n_t,), "components": (n_points, 4, n_t, 12),
    "overlaps": (n_points, 4, n_t, 12), "nfev": int}`` where ``components`` are
    ``<phi_q(t)|N_r|psi_s(t)>`` and ``overlaps`` are ``<phi_q(t)|psi_s(t)>`` (a
    conserved quantity, used as the correctness check).
    """
    omega_297 = np.asarray(omega_297, dtype=float)
    d_sweep = np.asarray(d_sweep, dtype=float)
    n_points = omega_297.size
    dim = ops.h_static_diag.size
    times = np.linspace(0.0, t_gate, n_t)

    fwd = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        LOGICAL_INPUTS, rhs_factory=_297_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times)

    nonlogical = np.setdiff1d(np.arange(dim), ops.logical_indices)
    adj = sweeplib.integrate_batch(
        ops, t_gate, {"omega_297": omega_297, "d_sweep": d_sweep},
        tuple(str(i) for i in nonlogical),
        rhs_factory=_297_adjoint_rhs_factory, dim=dim,
        rtol=rtol, atol=atol, ramp=ramp, t_eval=times,
        initial_indices=nonlogical, use_swap=False)

    # adj sampled in tau = T - t; flip back onto the forward time axis
    phi = adj.states[::-1]                       # (n_t, n_points, 12, dim)
    psi = fwd.states                             # (n_t, n_points, 4, dim)
    n_r = _rydberg_number_diag(dim)
    comp = np.einsum("tpqi,i,tpsi->pstq", phi.conj(), n_r, psi)
    over = np.einsum("tpqi,tpsi->pstq", phi.conj(), psi)
    return {"times": times, "components": comp, "overlaps": over,
            "nfev": fwd.nfev + adj.nfev}


def _rydberg_number_diag(dim: int, local_dim: int = 4) -> np.ndarray:
    """Diagonal of N_r: atoms in the Rydberg manifold (levels r and r_garb)."""
    idx = np.arange(dim)
    a, b = np.divmod(idx, local_dim)
    return (np.isin(a, (2, 3)).astype(float) + np.isin(b, (2, 3)).astype(float))


def filter_kernels(ops, t_gate, omega_297, d_sweep, *,
                   rtol, atol, ramp=0.15, n_t=KERNEL_N_T) -> np.ndarray:
    """(n_points, 4, n_bins) binned filter kernels for one batch."""
    from ryd_gate.phase_noise import filter_kernel

    out = integrate_adjoint_batch(ops, t_gate, omega_297, d_sweep,
                                  rtol=rtol, atol=atol, ramp=ramp, n_t=n_t)
    f_bins, df_bins = kernel_frequency_bins()
    comp = out["components"]
    kernels = np.empty((comp.shape[0], 4, f_bins.size))
    for p in range(comp.shape[0]):
        for s in range(4):
            kernels[p, s] = filter_kernel(out["times"], comp[p, s], f_bins, df_bins)
    return kernels
```

**CORRECTION (found during implementation, verified twice):** the backward leg also
needs the **conjugate** phase restoration. `integrate_batch` solves each column with its
bare diagonal energy subtracted and restores `exp(-i c t)`; running the same machinery in
`tau = T - t` with a sign-flipped RHS gives `chi(tau) = exp(-i c tau) phi(T - tau)`, so the
restoration is `exp(+i c tau)`. Implemented as a `reverse_time` flag on `integrate_batch`
(forward path bit-identical). Without it the adjoint-overlap invariant misses by 2.5e-5
instead of 2.4e-11. The `use_swap=False` kwarg in the sketch below is also wrong —
`integrate_batch` has no such parameter; `initial_indices` is what bypasses the swap.

`sweeplib.integrate_batch` currently derives its initial conditions from
`ops.logical_indices`. Add an `initial_indices=None` keyword to
`scripts/sweeplib/solver.py:206` that overrides them, used only by the adjoint leg:

```python
    logical_of_state = (
        {s: int(i) for s, i in zip(state_labels, initial_indices)}
        if initial_indices is not None
        else {s: ops.logical_indices[LOGICAL_INPUTS.index(s)] for s in state_labels}
    )
```

and guard the atom-swap reconstruction in `assemble` with
`if initial_indices is not None: psi[p, j] = chi[p * n_states + j]` for all
`j < n_states`, returning an `(n_points, n_states, dim)` array in that case.

- [ ] **Step 4: Run the adjoint test to verify it passes**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py -q -k adjoint'`
Expected: PASS

- [ ] **Step 5: Write the failing test for the convergence check**

```python
def test_filter_kernel_is_converged_at_the_production_sampling(mls297):
    """Halving dt must not move the kernel by more than 1% in any populated bin."""
    import numpy as np

    cfg = mls297.ScanConfig()
    ops = mls297.aggregate_operators(mls297.build_system(cfg, 53), 53)
    args = dict(rtol=1e-9, atol=1e-12, ramp=cfg.ramp_frac)
    k1 = mls297.filter_kernels(ops, 1e-6, np.array([2 * np.pi * 13.5e6]),
                               np.array([2 * np.pi * 15e6]),
                               n_t=mls297.KERNEL_N_T, **args)
    k2 = mls297.filter_kernels(ops, 1e-6, np.array([2 * np.pi * 13.5e6]),
                               np.array([2 * np.pi * 15e6]),
                               n_t=2 * mls297.KERNEL_N_T, **args)
    big = k2 > 1e-3 * k2.max()
    assert np.max(np.abs(k1[big] - k2[big]) / k2[big]) < 0.01
```

- [ ] **Step 6: Run it**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py -q -k converged'`
Expected: PASS. If it fails, raise `KERNEL_N_T` to the smallest power of two that passes and re-run — do not loosen the tolerance.

- [ ] **Step 7: Add the `filter/` store series**

In `scripts/sweeplib/store.py`, beside `next_scatter_seq`/`write_scatter_chunk`/`load_scatter_records`, add the exact analogues `next_filter_seq`, `write_filter_chunk`, `load_filter_records` writing `filter/filter_NNNNNN.npz`. `write_filter_chunk` takes `kernels` of shape `(n, 4, n_bins)` plus `f_bins`, and carries the same five provenance fields (`schema_version`, `scan_uuid`, `physics_hash`, `model_hash`, `pulse_hash`), the key arrays, the descriptor columns, `rtol`/`atol`/`n_t`/`batch_id`/`batch_size`/`status`/`message`/`runtime_s`. Add `self.filter_dir` to `__init__` and to `ensure_dirs`.

`load_filter_records` returns `{"key", "status", "kernel" (4, n_bins), "f_bins", "rtol", "runtime_s"}` and raises on a hash mismatch exactly as `load_scatter_records` does.

- [ ] **Step 8: Wire the runner**

In `scripts/sweeplib/runner.py`: add `filter_pass: bool = False` to `Batch`; propagate it through `_spec` (`filter=batch.filter_pass`) and `_split`; in `_worker_run_batch`, when `spec["filter"]` is set, call `ctx["filter_solve"](ops, t_gate, omega, d_sweep, rtol=..., atol=..., ramp=cfg.ramp_frac)` and return `{"ok": True, "kernels": ..., "runtime_s": ...}` instead of a `BatchResult`; add `filter_solve` to `set_worker_context`; in `_write_success`/`_write_failure` branch on `batch.filter_pass` to the filter series (mirroring the `batch.scatter` branch), with `self.filter_seq` initialized from `store.next_filter_seq()`.

- [ ] **Step 9: Add `cmd_filter` and its subparser**

```python
def cmd_filter(args) -> None:
    """Filter-function pass: additive only (writes the filter/ series)."""
    store, manifest, cfg, ops, checks = setup_run(args)
    level = LEVEL_FROM_SIZE[int(args.level)]
    panels = _parse_panels(args)
    done = {r["key"] for r in store.load_filter_records(manifest)
            if r["status"] == "ok"}
    missing = [k for k in _filter_panels(all_keys(level), panels) if k not in done]
    print(f"[filter] level {args.level}: {len(missing)} points to compute "
          f"({len(done)} already stored)", flush=True)
    if not missing:
        return
    cost = CostModel(cfg)
    _feed_cost_model(cost, store.load_records(manifest, include_states=False))
    runner = Runner(store, manifest, cfg, args, cost)
    try:
        batches = group_batches(missing, _effective_batch_size(store, args))
        for b in batches:
            b.filter_pass = True
        runner.run_batches(batches, f"filter-{args.level}")
    except KeyboardInterrupt:
        print("[filter] hard abort", flush=True)
    finally:
        runner.write_failure_report()
        runner.write_status(f"filter-{args.level}-aborted" if runner.aborted
                            else f"filter-{args.level}-done")
        runner.shutdown()
```

In `setup_run`, add the filter solve to `set_worker_context`:

```python
    def _filter_solve(ops, t_gate, omega_297, d_sweep, *, rtol, atol, ramp):
        return filter_kernels(ops, t_gate, omega_297, d_sweep,
                              rtol=rtol, atol=atol, ramp=ramp)
```

In `build_parser`:

```python
    sp = sub.add_parser("filter",
                        help="filter-function pass (additive: writes only the "
                             "filter/ series; reusable across every PSD)")
    common(sp, compute=True)
    sp.add_argument("--level", default="13", choices=["4", "7", "13", "25"])
    sp.set_defaults(func=cmd_filter)
```

- [ ] **Step 10: Write the end-to-end smoke test**

```python
def test_filter_subcommand_writes_a_resumable_series(mls297, tmp_path):
    """One panel, level 4: the filter series appears, resumes, and reweights."""
    import numpy as np

    out = str(tmp_path / "store")
    argv = ["filter", "--output", out, "--level", "4", "--panels", "1,0",
            "--workers", "2", "--batch-size", "4"]
    mls297.main(argv)
    store = mls297.Store(out)
    manifest = store.load_manifest()
    rows = store.load_filter_records(manifest)
    assert len(rows) == 16 and all(r["status"] == "ok" for r in rows)
    assert rows[0]["kernel"].shape == (4, mls297.kernel_frequency_bins()[0].size)

    mls297.main(argv)                                  # resume: nothing new
    assert len(store.load_filter_records(manifest)) == 16
```

- [ ] **Step 11: Run the full sweep test file**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py tests/test_sweeplib.py tests/test_sweep_compat_locks.py -q'`
Expected: PASS, including the pre-existing compat-lock tests — **the stored chunk/scatter formats must not have moved.**

- [ ] **Step 12: Commit**

```bash
git add scripts/max_leakage_297_sweep.py scripts/sweeplib/ tests/test_max_leakage_297_sweep.py
git commit -m "Add the filter-function pass to the 297 nm leakage sweep"
```

---

### Task 5: Monte Carlo validation on the real gate

**Files:**
- Create: `scripts/phase_noise_mc_check.py`
- Modify: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `results/max_leakage_297/a3.0/reports/phase_noise_mc.json` with, per checked point, `point_id`, `mc_mean`, `mc_stderr`, `filter_prediction`, `passed`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Validate the stored filter kernels against direct Monte Carlo on the real gate.

Picks N points spread over the panel family, runs ``shots`` phase-noise
realizations of each through the same DOP853 kernel the sweep uses (the noise
entering as an added 2 pi dnu(t) N_r term), and compares the mean leakage increase
against ``error_from_kernel`` on the stored kernel.  Acceptance: within four
Monte Carlo standard errors or 10% of the prediction, whichever is looser.
"""
```

The Monte Carlo leg reuses `integrate_batch` with a noise-augmented RHS built by wrapping `_297_rhs_factory`'s output:

```python
def _noisy_rhs_factory(trace, n_r_diag):
    def make(ops, cols, t_gate, ramp):
        base = _297_rhs_factory(ops, cols, t_gate, ramp)
        n_cols = cols["shift"].size
        dphi = trace._spline.derivative()

        def rhs(t, y):
            out = base(t, y)
            ym = y.reshape(n_cols, n_r_diag.size)
            return out - 1j * (float(dphi(t)) * (n_r_diag[None, :] * ym)).ravel()
        return rhs
    return make
```

- [ ] **Step 2: Run it on 20 points**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run python scripts/phase_noise_mc_check.py --n-points 20 --shots 200 --laser ECDL --extrapolation flat'`
Expected: `passed: 20/20`. If points fail, the discrepancy is real — check the sign and factor conventions in `_noisy_rhs_factory` against the `H_0 + 2 pi dnu N_r` statement before touching the kernel.

- [ ] **Step 3: Pin one cheap point as a test**

```python
def test_filter_prediction_matches_monte_carlo_on_one_point(mls297):
    """One n=53, T=1 us point, 60 shots: the kernel prediction is inside 4 sigma."""
```

(Use `--n-points 1 --shots 60`; the test imports the script's `check_point` helper directly so it does not shell out.)

- [ ] **Step 4: Run the test**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py -q -k monte_carlo'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/phase_noise_mc_check.py tests/test_max_leakage_297_sweep.py results/max_leakage_297/a3.0/reports/phase_noise_mc.json
git commit -m "Validate the phase-noise filter kernels against direct Monte Carlo"
```

---

### Task 6: `eps_phase` metric and the power table

**Files:**
- Modify: `scripts/sweeplib/plotting.py:25-43` (`PlotSpec`), `:201-271` (`render_panel_grid`)
- Modify: `scripts/max_leakage_297_sweep.py` (`cmd_plot`, `_PLOT_SPEC`, power cache)
- Modify: `tests/test_max_leakage_297_sweep.py`

**Interfaces:**
- Consumes: `load_filter_records` (Task 4), `PhaseNoisePSD`/`error_from_kernel` (Tasks 1, 3).
- Produces:
  - `power_table_rows(cfg) -> dict` cached to `results/297_laser_noise/omega_per_watt.npz` with arrays `ryd_n` (8,) and `omega_mhz_at_1w` (8,) for `beam_area_um2 = 420`
  - `--laser {ECDL,seed}` and `--extrapolation {flat,power}` on the `plot` subcommand, plus metrics `eps_phase` and `total_error_phase`
  - `PlotSpec.table` — an optional `(col_labels, row_labels, cells, caption)` bundle drawn as a strip under the grid

- [ ] **Step 1: Write the failing test for the power cache**

```python
def test_power_table_matches_arc_and_scales_as_one_over_rabi_squared(mls297):
    import numpy as np
    from ryd_gate.physics import rb87_297_clock_rabi_frequencies

    rows = mls297.power_table_rows(mls297.ScanConfig())
    i = list(rows["ryd_n"]).index(53)
    omega, _ = rb87_297_clock_rabi_frequencies(1.0, 420.0, ryd_level=53)
    assert rows["omega_mhz_at_1w"][i] == pytest.approx(omega / (2 * np.pi * 1e6), rel=1e-9)
    # 18 MHz needs (18 / omega_at_1W)**2 watts at the atoms
    assert mls297.power_at_atoms_w(rows, 53, 18.0) == pytest.approx(
        (18.0 / rows["omega_mhz_at_1w"][i]) ** 2, rel=1e-9)
```

- [ ] **Step 2: Run it**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py -q -k power_table'`
Expected: FAIL — `AttributeError: power_table_rows`

- [ ] **Step 3: Implement the cache and the lookup**

```python
POWER_BEAM_AREA_UM2 = 420.0     # notebook nominal: 20 x spacing by 7 um top-hat
POWER_OPTICS_LOSS = 0.8         # 80% of the nominal power is lost before the atoms
POWER_TABLE_OMEGA_MHZ = (9.0, 11.0, 13.5, 15.0, 16.5, 18.0)
_POWER_CACHE = os.path.join("results", "297_laser_noise", "omega_per_watt.npz")


def power_table_rows(cfg: "ScanConfig") -> dict:
    """Per-n target-leg Rabi at 1 W over POWER_BEAM_AREA_UM2 (cached; ARC once)."""
    if os.path.exists(_POWER_CACHE):
        with np.load(_POWER_CACHE, allow_pickle=False) as d:
            if list(d["ryd_n"]) == list(cfg.ryd_n):
                return {"ryd_n": d["ryd_n"], "omega_mhz_at_1w": d["omega_mhz_at_1w"]}
    from ryd_gate.physics import rb87_297_clock_rabi_frequencies

    vals = np.asarray([
        rb87_297_clock_rabi_frequencies(1.0, POWER_BEAM_AREA_UM2, ryd_level=int(n))[0]
        / (TAU * 1e6) for n in cfg.ryd_n])
    rows = {"ryd_n": np.asarray(cfg.ryd_n), "omega_mhz_at_1w": vals}
    _atomic_savez(_POWER_CACHE, **rows)
    return rows


def power_at_atoms_w(rows: dict, ryd_n: int, omega_mhz: float) -> float:
    """Power at the atoms (W) for ``omega_mhz`` at ``ryd_n``; Omega ~ sqrt(P/A)."""
    i = list(rows["ryd_n"]).index(int(ryd_n))
    return float((omega_mhz / rows["omega_mhz_at_1w"][i]) ** 2)
```

- [ ] **Step 4: Run the test**

Run the Step 2 command.
Expected: PASS

- [ ] **Step 5: Add the metrics to `plot_metric_values`**

Extend `scripts/sweeplib/plotting.py:75` with an optional `extra_values: dict | None = None` parameter: when `metric == "eps_phase"` the values come straight from `extra_values`; when `metric == "total_error_phase"` they are added per logical input to the coherent leakage and the scattering channels before the worst-input maximum is taken. `cmd_plot` in the sweep script builds `extra_values`:

```python
    from ryd_gate.phase_noise import PhaseNoisePSD, error_from_kernel

    psd = PhaseNoisePSD.from_csv(
        os.path.join("results", "297_laser_noise", f"psd_{args.laser}.csv"),
        harmonic=4, extrapolation=args.extrapolation)
    f_bins, _ = kernel_frequency_bins()
    extra = {}
    for r in store.load_filter_records(manifest):
        if r["status"] == "ok":
            extra[r["key"]] = np.asarray(
                [error_from_kernel(psd, f_bins, r["kernel"][s]) for s in range(4)])
```

- [ ] **Step 6: Add the table strip to `render_panel_grid`**

Extend `PlotSpec` with `table: tuple | None = None`. When present, `render_panel_grid` reserves the bottom ~12% of the figure for a single `ax.table` (no axes, `fontsize=7`) and writes `spec.table[3]` as the caption below it. Output filenames gain the laser/extrapolation suffix: `f"{metric}_8x9_{suffix}.png"` when `suffix` is passed, so the existing noise-free figures are never overwritten.

- [ ] **Step 7: Write the plot smoke test**

```python
def test_phase_noise_plot_emits_the_map_and_the_power_table(mls297, tmp_path, mini_store):
    """A synthetic mini-store renders eps_phase with a table strip and a suffix."""
```

Build on the existing synthetic mini-store fixture in the file; assert that `plots/phase_noise/ECDL/eps_phase_8x9_ECDL_flat.png` exists and that no file in `plots/` was modified.

- [ ] **Step 8: Run the tests**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run --extra dev pytest tests/test_max_leakage_297_sweep.py tests/test_sweeplib.py -q'`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
# This checkout is SHARED and carries ~2460 unrelated staged deletions from
# another actor. `git add` only the exact paths you touched, and commit with an
# EXPLICIT pathspec so a bare `git commit` cannot sweep those deletions in.
git add scripts/max_leakage_297_sweep.py scripts/sweeplib/plotting.py \
        tests/test_max_leakage_297_sweep.py results/297_laser_noise/omega_per_watt.npz
git commit -m "Render eps_phase and total_error_phase maps with a power-Rabi table" -- \
        scripts/max_leakage_297_sweep.py scripts/sweeplib/plotting.py \
        tests/test_max_leakage_297_sweep.py results/297_laser_noise/omega_per_watt.npz
```

---

### Task 7: Run the campaign

**Files:**
- Create: `results/max_leakage_297/a3.0/plots/phase_noise/{ECDL,seed}/*`
- Create: `docs/superpowers/plans/2026-07-30-laser-phase-noise-results.md`

- [ ] **Step 1: Pilot the filter pass on one panel and check the ETA**

Run: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && uv run python scripts/max_leakage_297_sweep.py filter --level 4 --panels 3,0 --workers 20'`
Expected: 16 points; note the per-point runtime and confirm it is within ~2x of 5x the coherent pass.

- [ ] **Step 2: Run the full level-13 filter pass**

Run in the background: `ssh chance@172.20.4.137 'cd ~/Ryd-gate-modeling && export PATH=$HOME/.local/bin:$PATH && nohup uv run python scripts/max_leakage_297_sweep.py filter --level 13 --workers 20 > filter.log 2>&1 &'`
Expected: ~6 h; resumable, so a stop is harmless.

- [ ] **Step 3: Render all 14 figures**

```bash
for laser in ECDL seed; do
  for metric in max_leakage p_ryd p_r_garb eps_phase total_error_phase; do
    uv run python scripts/max_leakage_297_sweep.py plot --metric $metric \
        --laser $laser --extrapolation flat
  done
  for metric in eps_phase total_error_phase; do
    uv run python scripts/max_leakage_297_sweep.py plot --metric $metric \
        --laser $laser --extrapolation power
  done
done
```

- [ ] **Step 4: Write the results note**

Record, per laser and extrapolation: where `total_error_phase` is minimized on the `(n, T, Omega, D_sweep)` grid, the minimum required nominal power there, and how the optimum moved relative to the noise-free maps. State the `f_min = 1 Hz` sensitivity by re-rendering `eps_phase` once with `f_min = 10 Hz` and reporting the shift.

- [ ] **Step 5: Commit**

```bash
# Explicit pathspec — see the Task 6 note about this checkout's staged deletions.
git add results/max_leakage_297/a3.0/plots/phase_noise \
        docs/superpowers/plans/2026-07-30-laser-phase-noise-results.md
git commit -m "Render the phase-noise map family for both measured 297 nm lasers" -- \
        results/max_leakage_297/a3.0/plots/phase_noise \
        docs/superpowers/plans/2026-07-30-laser-phase-noise-results.md
```

---

## Self-Review

**Spec coverage:** `PhaseNoisePSD` incl. servo bump and both extrapolations → Task 1; `phase_trace` hybrid grid and `f_min` → Task 2; the filter function and the literature check → Task 3; the `filter/` store series and the adjoint formulation → Task 4; the grid-level Monte Carlo check → Task 5; `eps_phase`/`total_error_phase` and the power table → Task 6; the 14 figures and the `f_min` sensitivity → Task 7. The trace-statistics test the spec asks for is Task 2 Step 1 (Welch). Spec's 420/1013 coverage is satisfied by `phase_noise.py` being model-independent — no task implements a 420/1013 filter path, which the spec puts out of scope.

**Type consistency:** `filter_kernel(times, components, f_bins, df_bins)` returns bin-integrated `K_b`, so `error_from_kernel(psd, f_bins, kernel)` takes no `df` — stated at Task 3's interface block and honoured in Tasks 4 and 6. `load_filter_records` rows carry `kernel` of shape `(4, n_bins)`, matching `r["kernel"][s]` in Task 6. `kernel_frequency_bins()` returns `(f_bins, df_bins)` everywhere.

**Known risk:** Task 4 Step 3 changes `sweeplib/solver.py:integrate_batch`, which the compat-lock tests cover. The `initial_indices=None` default keeps the existing call sites byte-identical; Step 11 runs those locks explicitly to prove it.
