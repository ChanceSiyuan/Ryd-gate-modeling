"""Abstract base class for pulse protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Protocol(ABC):
    """Abstract base class for pulse protocols.

    A protocol defines how pulse parameters map to time-dependent
    drive coefficients on named channels.

    Subclasses must implement:
    - ``n_params``, ``validate_params``, ``unpack_params``, ``get_drive_coefficients``

    CZ gate protocols additionally override:
    - ``theta_index``, ``t_gate_index``, ``get_optimization_bounds``
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters in the parameter vector x."""
        ...

    @abstractmethod
    def validate_params(self, x: list[float]) -> None:
        """Raise ValueError if x has wrong length."""
        ...

    @abstractmethod
    def unpack_params(self, x: list[float], system) -> dict:
        """Unpack x into physical quantities for the solver.

        Returns
        -------
        dict
            Must contain at minimum ``'t_gate'``.
        """
        ...

    def phase_420(self, t: float, params: dict) -> complex:
        """Compute exp(-i * phi(t)) for 420nm laser phase modulation.

        The default implementation derives the value from the
        ``drive_420`` coefficient when a protocol exposes that channel.
        """
        coeffs = self.get_drive_coefficients(t, params)
        return coeffs.get("drive_420", 1.0 + 0j)

    # -- New generalized interface -----------------------------------------

    @property
    def required_channels(self) -> frozenset[str]:
        """Channel names this protocol drives.

        Used by the compiler to validate system-protocol compatibility.
        Default assumes the two-photon 420nm drive channels.
        """
        return frozenset({"drive_420", "drive_420_dag"})

    def drive_channels(self, system) -> frozenset[str]:
        """Channel names wired into the Hamiltonian for *system*.

        Defaults to :attr:`required_channels`.  Protocols with site-dependent
        drives override this to include per-site channel names.
        """
        return frozenset(self.required_channels)

    @abstractmethod
    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return {channel_name: coefficient(t)} for each drive term."""
        ...

    # -- Optional hooks (CZ gate protocols override these) ----------------

    @property
    def theta_index(self) -> int | None:
        """Index of theta (single-qubit Z rotation) in x, or None."""
        return None

    @property
    def t_gate_index(self) -> int | None:
        """Index of scaled gate time in x, or None."""
        return None

    def get_optimization_bounds(self) -> tuple | None:
        """Return bounds for optimisation, or None if not applicable."""
        return None

    # -- Visualization -----------------------------------------------------

    def pulse_traces(self, t: float, params: dict) -> dict[str, float] | None:
        """Physically-labeled, same-unit real traces at time *t* (or ``None``).

        Override to expose pulse quantities (e.g. ``{"Omega": ..., "Delta": ...}``)
        so :meth:`plot` shows a clean physical picture instead of the raw drive
        channels.  Returning ``None`` (the default) makes :meth:`plot` fall back to
        sampling :meth:`get_drive_coefficients`.
        """
        return None

    def plot(
        self,
        system=None,
        *,
        params: dict | None = None,
        x=None,
        n_points: int = 400,
        channels=None,
        skip_dag: bool = True,
        group_sites: bool = True,
        stacked: bool = False,
        unit_scale: float = 1.0,
        unit_label: str = "natural units",
        time_scale: float = 1.0,
        time_label: str = "time",
        ax=None,
        title: str | None = None,
        savefig=False,
        show: bool = False,
    ):
        """Plot the pulse schedule over ``[0, t_gate]``.

        Generic visualization available to *every* protocol.  When the protocol
        implements :meth:`pulse_traces`, those physically-labeled real traces are
        drawn; otherwise :meth:`get_drive_coefficients` is sampled and each drive
        channel is plotted (real channels directly; complex channels split into
        real/imag parts; ``*_dag`` conjugate channels skipped; per-site families
        ``name_<i>`` bundled into one faint group).  Subclasses with a richer view
        (e.g. :class:`~ryd_gate.protocols.sweep.SweepProtocol`) may override.

        Parameters
        ----------
        system : object, optional
            System-like object to unpack params from when ``params`` is omitted
            (uses ``x`` if given, else an empty parameter vector).
        params : dict, optional
            Pre-unpacked params (must contain ``"t_gate"``); takes precedence.
        x : sequence, optional
            Parameter vector to unpack against ``system`` (CZ-gate protocols).
        n_points : int
            Number of time samples.
        channels : sequence of str, optional
            Restrict the channel fallback to these names.
        skip_dag : bool
            Skip ``*_dag`` (hermitian-conjugate) channels in the fallback.
        group_sites : bool
            Bundle ``name_<i>`` per-site families into one faint group.
        stacked : bool
            When using :meth:`pulse_traces`, draw each trace in its own vertically
            stacked subplot (shared time axis) instead of one shared axis — so
            differently-scaled traces (e.g. amplitudes vs phase chirps) each
            autoscale.  Returns ``(fig, [axes])`` in this mode.
        unit_scale, unit_label : float, str
            Multiply values by ``unit_scale`` and label the y-axis with
            ``unit_label`` (e.g. ``1/(2*np.pi*1e6)`` and ``"MHz"``).
        time_scale, time_label : float, str
            Multiply the time axis by ``time_scale`` and label it with
            ``time_label`` (e.g. ``1e6`` and ``"time (us)"``).
        ax : matplotlib Axes, optional
            Draw onto an existing axis instead of creating a figure.
        title : str, optional
            Plot title (defaults to the protocol class name).
        savefig : bool or path-like
            If truthy, save the figure (``True`` -> ``<classname>_schedule.png``).
        show : bool
            Call ``plt.show()`` after drawing.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes.
        """
        import re
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        if params is None:
            if system is None:
                raise ValueError("plot() requires `params`, or a `system` to unpack params from.")
            params = self.unpack_params(list(x) if x is not None else [], system)
        if "t_gate" not in params:
            raise ValueError("params must contain 't_gate' to plot the schedule.")
        t_gate = float(params["t_gate"])
        if t_gate <= 0 or int(n_points) < 2:
            raise ValueError("plot() needs a positive t_gate and n_points >= 2.")

        ts = np.linspace(0.0, t_gate, int(n_points))

        # Prefer physically-labeled traces when the protocol provides them.
        use_traces = self.pulse_traces(0.0, params) is not None
        series: dict[str, list[complex]] = {}
        for t in ts:
            sample = self.pulse_traces(float(t), params) if use_traces else self.get_drive_coefficients(float(t), params)
            for name, val in sample.items():
                series.setdefault(name, []).append(complex(val))
        arrays = {name: np.asarray(vals, dtype=complex) for name, vals in series.items()}

        if channels is not None and not use_traces:
            names = [c for c in channels if c in arrays]
        elif use_traces:
            names = list(arrays)
        else:
            names = [c for c in arrays if not (skip_dag and c.endswith("_dag"))]

        site_pat = re.compile(r"^(.*)_(\d+)$")
        families: dict[str, list[str]] = {}
        singles: list[str] = []
        for name in names:
            m = site_pat.match(name)
            if group_sites and not use_traces and m:
                families.setdefault(m.group(1), []).append(name)
            else:
                singles.append(name)
        # A real per-site family has >=2 members (e.g. global_n_0..N-1); a lone
        # trailing-number match like "drive_420" is a single channel, not a bundle.
        for prefix in list(families):
            if len(families[prefix]) < 2:
                singles.extend(families.pop(prefix))

        tgrid = ts * time_scale

        def _draw_trace(target_ax, name: str, label: str) -> bool:
            """Draw one trace on *target_ax* (Re/Im split when complex, else real).
            Returns True if an imaginary part was drawn (caller may add a legend)."""
            y = arrays[name] * unit_scale
            scale = max(1.0, float(np.max(np.abs(y))))
            if float(np.max(np.abs(y.imag))) > 1e-9 * scale:
                target_ax.plot(tgrid, y.real, lw=2, label=f"Re {label}")
                target_ax.plot(tgrid, y.imag, lw=2, ls="--", label=f"Im {label}")
                return True
            target_ax.plot(tgrid, y.real, lw=2, label=label)
            return False

        # Stacked layout: one subplot per trace, shared time axis (top -> bottom).
        # Use it when traces live on incompatible scales (e.g. amplitudes in MHz vs
        # phase chirps), so each panel autoscales independently.
        if stacked and use_traces and ax is None:
            n = max(len(names), 1)
            fig, axs = plt.subplots(
                n, 1, sharex=True, figsize=(10, max(1.8 * n, 3.0)), squeeze=False
            )
            axs = list(axs[:, 0])
            for axi, name in zip(axs, names):
                if _draw_trace(axi, name, name):
                    axi.legend(fontsize=8, loc="upper right")
                axi.set_ylabel(f"{name}\n[{unit_label}]", fontsize=9)
                axi.axhline(0.0, color="k", ls=":", lw=0.8)
                axi.grid(alpha=0.3)
            axs[-1].set_xlabel(time_label)
            axs[0].set_title(title or f"{type(self).__name__} pulse schedule")
            fig.tight_layout()
            if savefig:
                stem = Path(f"{type(self).__name__.lower()}_schedule" if savefig is True else savefig)
                fig.savefig(stem.with_suffix(".png") if stem.suffix == "" else stem)
            if show:
                plt.show()
            return fig, axs

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure

        for name in singles:
            _draw_trace(ax, name, name)
        for prefix, members in families.items():
            members = sorted(members, key=lambda c: int(site_pat.match(c).group(2)))
            for k, name in enumerate(members):
                ax.plot(
                    tgrid, (arrays[name] * unit_scale).real, color="0.6", lw=0.8, alpha=0.6,
                    label=f"{prefix}_i (per-site)" if k == 0 else None,
                )

        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel(time_label)
        ax.set_ylabel(f"drive [{unit_label}]")
        ax.set_title(title or f"{type(self).__name__} pulse schedule")
        ax.grid(alpha=0.3)
        if names:
            ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()

        if savefig:
            stem = Path(f"{type(self).__name__.lower()}_schedule" if savefig is True else savefig)
            fig.savefig(stem.with_suffix(".png") if stem.suffix == "" else stem)
        if show:
            plt.show()
        return fig, ax
