"""Function-defined global Rydberg sweep protocol.

``SweepProtocol`` is intentionally small: the schedule lives on the protocol,
so the parameter vector passed to ``simulate()`` is empty.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ryd_gate.protocols.base import Protocol

ScalarTimeFunction = Callable[[float], float]
AddressTimeFunction = Callable[[float, int], float]


class SweepProtocol(Protocol):
    """Two-level global Rydberg sweep with arbitrary time functions.

    Parameters
    ----------
    t_gate : float
        Total evolution time.
    omega_half_fn : callable
        Function ``t -> Omega(t)/2`` in the same angular-frequency units used
        by the Hamiltonian.
    delta_fn : callable
        Function ``t -> Delta(t)`` for the global Rydberg detuning.
    address_fn : callable, optional
        Function ``(t, i) -> Delta_addr_i(t)`` for local-addressing detuning
        shifts.  The total detuning on site ``i`` is
        ``Delta(t) + Delta_addr_i(t)``.
    n_steps : int
        Number of piecewise-constant time steps used by exact sparse evolution.

    Schedules emit ``global_X = Omega(t)/2``, ``global_n = -Delta(t)``, and,
    when ``address_fn`` is provided, ``global_n_i = -Delta_addr_i(t)``.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        omega_half_fn: ScalarTimeFunction,
        delta_fn: ScalarTimeFunction,
        address_fn: AddressTimeFunction | None = None,
        n_steps: int = 200,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        if not callable(omega_half_fn):
            raise TypeError("omega_half_fn must be callable.")
        if not callable(delta_fn):
            raise TypeError("delta_fn must be callable.")
        if address_fn is not None and not callable(address_fn):
            raise TypeError("address_fn must be callable when provided.")

        self.t_gate = float(t_gate)
        self.omega_half_fn = omega_half_fn
        self.delta_fn = delta_fn
        self.address_fn = address_fn
        self.n_steps = int(n_steps)
        self._phase_table: tuple[float, int, np.ndarray, np.ndarray] | None = None

    @property
    def n_params(self) -> int:
        return 0

    def validate_params(self, x) -> None:
        if len(x) != 0:
            raise ValueError(f"SweepProtocol takes no x parameters; got {len(x)}.")

    def unpack_params(self, x, system) -> dict:
        self.validate_params(x)
        n_sites = self._n_sites(system)
        return {
            "t_gate": self.t_gate,
            "Omega": 2.0 * self.omega_half_at(self.t_gate),
            "Delta": self.delta_at(self.t_gate),
            "n_sites": n_sites,
            "pin_deltas": {},
            "scatter_rates": {},
            "static_overlays": [],
            "_system_type": "lattice",
        }

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"global_X", "global_n"})

    def drive_channels(self, system) -> frozenset[str]:
        n_sites = self._n_sites(system)
        channels = {"global_X", "global_n"}
        if self.address_fn is not None:
            channels.update(f"global_n_{i}" for i in range(n_sites))
        return frozenset(channels)

    def omega_half_at(self, t: float) -> float:
        return float(self.omega_half_fn(self._clamp_time(t)))

    def delta_at(self, t: float) -> float:
        value = self.delta_fn(self._clamp_time(t))
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        raise ValueError("delta_fn(t) must return a scalar global detuning.")

    def address_at(self, t: float, n_sites: int) -> np.ndarray:
        if n_sites < 1:
            raise ValueError("n_sites must be positive.")
        if self.address_fn is None:
            return np.zeros(n_sites, dtype=float)
        t_clamped = self._clamp_time(t)
        return np.asarray([self.address_fn(t_clamped, i) for i in range(n_sites)], dtype=float)

    def total_delta_at(self, t: float, n_sites: int) -> np.ndarray:
        return self.delta_at(t) + self.address_at(t, n_sites)

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        coeffs: dict[str, complex] = {
            "global_X": self.omega_half_at(t),
            "global_n": -self.delta_at(t),
        }
        if self.address_fn is not None:
            n_sites = params.get("n_sites")
            if n_sites is None:
                raise ValueError("SweepProtocol with address_fn requires params['n_sites'].")
            coeffs.update(
                {f"global_n_{i}": -float(shift) for i, shift in enumerate(self.address_at(t, int(n_sites)))}
            )
        return coeffs

    def _phase_delta_at(self, t: float) -> float:
        return self.delta_at(t)

    def plot(
        self,
        system: Any | None = None,
        *,
        params: dict | None = None,
        n_sites: int | None = None,
        grid_shape: tuple[int, int] | None = None,
        n_points: int = 601,
        address_time: float | None = None,
        savefig: bool | str | Path = False,
        show: bool = True,
    ):
        """Plot ``Omega(t)/2``, global ``Delta(t)``, and address-shift map.

        The plotted frequency unit is MHz for angular frequencies, i.e. values
        are divided by ``2*pi*1e6``.  Time is shown in microseconds.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        if params is None:
            params = self.unpack_params([], system) if system is not None else {"t_gate": self.t_gate}
        if n_sites is None:
            n_sites = params.get("n_sites")
        if n_sites is None and system is not None:
            n_sites = self._n_sites(system)
        if n_sites is None:
            raise ValueError("plot() needs system, params['n_sites'], or n_sites for the address map.")
        n_sites = int(n_sites)

        import matplotlib.pyplot as plt

        t_shape_s = np.linspace(0.0, self.t_gate, n_points)
        t_shape_us = t_shape_s * 1e6
        omega_half_mhz = np.asarray([self.omega_half_at(t) / (2.0 * np.pi * 1e6) for t in t_shape_s])
        delta_mhz = np.asarray([self.delta_at(t) / (2.0 * np.pi * 1e6) for t in t_shape_s])

        fig_p, axes_p = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
        axes_p[0].plot(t_shape_us, omega_half_mhz, lw=2, color="tab:blue")
        axes_p[0].set_ylabel(r"$\Omega(t)/2$ (MHz)")
        axes_p[1].plot(t_shape_us, delta_mhz, lw=2, color="tab:blue", label="global")
        axes_p[1].legend(loc="best")
        axes_p[1].set_ylabel(r"$\Delta(t)/2\pi$ (MHz)")
        axes_p[1].set_xlabel(r"time ($\mu$s)")
        for ax_p in axes_p:
            ax_p.axhline(0.0, color="k", ls=":", lw=0.8)
            ax_p.grid(alpha=0.3)
            ax_p.set_xlim(0.0, self.t_gate * 1e6)
        fig_p.suptitle("SweepProtocol pulse-time shape", y=1.02)
        fig_p.tight_layout()

        t_addr = self.t_gate if address_time is None else self._clamp_time(address_time)
        address_shift_mhz = np.abs(self.address_at(t_addr, n_sites)) / (2.0 * np.pi * 1e6)
        address_grid = self._profile_to_grid(address_shift_mhz, system, grid_shape)
        shift_vmax = max(1.0, float(np.nanmax(address_grid)))

        fig_s, ax_s = plt.subplots(figsize=(3.6, 3.2))
        im_s = ax_s.imshow(address_grid.T, origin="lower", vmin=0.0, vmax=shift_vmax, cmap="magma")
        ax_s.set_title("Addressing shift magnitude")
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")
        ax_s.set_xticks(range(address_grid.shape[0]))
        ax_s.set_yticks(range(address_grid.shape[1]))
        for ix in range(address_grid.shape[0]):
            for iy in range(address_grid.shape[1]):
                val = address_grid[ix, iy]
                if np.isfinite(val):
                    ax_s.text(ix, iy, f"{val:.2f}", ha="center", va="center", color="white")
        fig_s.colorbar(im_s, ax=ax_s, label=r"$|\Delta_{\rm addr}|/2\pi$ (MHz)")
        fig_s.tight_layout()

        if savefig:
            prefix = Path("sweep_protocol" if savefig is True else savefig)
            fig_p.savefig(prefix.with_name(f"{prefix.name}_pulse.png"))
            fig_s.savefig(prefix.with_name(f"{prefix.name}_address.png"))
        if show:
            plt.show()
        return fig_p, fig_s

    @staticmethod
    def _profile_to_grid(
        profile: np.ndarray,
        system: Any | None,
        grid_shape: tuple[int, int] | None,
    ) -> np.ndarray:
        if system is not None and getattr(system, "geometry", None) is not None:
            coords = np.asarray(system.geometry.coords, dtype=float)
            x_vals = np.unique(coords[:, 0])
            y_vals = np.unique(coords[:, 1])
            if len(x_vals) * len(y_vals) == len(profile):
                grid = np.full((len(x_vals), len(y_vals)), np.nan, dtype=float)
                x_index = {x: i for i, x in enumerate(x_vals)}
                y_index = {y: i for i, y in enumerate(y_vals)}
                for site, (x, y) in enumerate(coords):
                    grid[x_index[x], y_index[y]] = profile[site]
                return grid

        if grid_shape is not None:
            if grid_shape[0] * grid_shape[1] != len(profile):
                raise ValueError(f"grid_shape {grid_shape} does not match n_sites={len(profile)}.")
            return profile.reshape(grid_shape)

        side = int(round(np.sqrt(len(profile))))
        if side * side == len(profile):
            return profile.reshape((side, side))
        return profile.reshape((len(profile), 1))

    @staticmethod
    def _n_sites(system) -> int:
        basis = getattr(system, "basis", None)
        n_sites = getattr(basis, "n_sites", None)
        if n_sites is None:
            n_sites = getattr(system, "N", None)
        if n_sites is None and hasattr(system, "meta"):
            n_sites = system.meta("n_sites", None)
        if n_sites is None:
            raise TypeError("SweepProtocol needs a system-like object with basis.n_sites, N, or meta('n_sites').")
        return int(n_sites)

    def phase_420(self, t: float, params: dict) -> complex:
        """Return exp(-i int_0^t Delta(t') dt') for legacy phase consumers."""
        return np.exp(-1j * self._detuning_phase(t))

    def _clamp_time(self, t: float) -> float:
        return float(np.clip(float(t), 0.0, self.t_gate))

    def _ensure_phase_table(self) -> None:
        key = (self.t_gate, id(self.delta_fn))
        if self._phase_table is not None and self._phase_table[:2] == key:
            return

        from scipy.integrate import cumulative_trapezoid

        n_pts = max(2000, 10 * self.n_steps + 1)
        ts = np.linspace(0.0, self.t_gate, n_pts)
        deltas = np.array([self._phase_delta_at(t) for t in ts], dtype=float)
        phases = np.zeros(n_pts, dtype=float)
        phases[1:] = cumulative_trapezoid(deltas, ts)
        self._phase_table = (key[0], key[1], ts, phases)

    def _detuning_phase(self, t: float) -> float:
        self._ensure_phase_table()
        _, _, ts, phases = self._phase_table
        return float(np.interp(self._clamp_time(t), ts, phases))
