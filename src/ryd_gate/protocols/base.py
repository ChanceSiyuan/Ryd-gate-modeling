"""Abstract base class for pulse protocols.

A protocol is **fully specified at construction**: it carries no runtime
parameter vector.  Binding it to a system resolves its normalized quantities
(durations, Rabi ratios) against that system's physical scales — the protocol
decides the gate duration, and the bound system exposes it as the read-only
``system.t_gate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Protocol(ABC):
    """Abstract base class for pulse protocols.

    A protocol defines time-dependent drive coefficients on named primitive
    ``E[ket,bra]`` channels.  Subclasses implement:

    - ``_resolve(system)`` — resolve the fully specified protocol against the
      bound system's physical scales into the private context dict consumed by
      the compilers (must contain at minimum ``"t_gate"``);
    - ``get_drive_coefficients(t, ctx)`` — the per-channel complex drive
      coefficients at time ``t`` given that resolved context.
    """

    @abstractmethod
    def _resolve(self, system) -> dict:
        """Resolve physical quantities against *system* (private seam).

        Returns the context dict the compilers pass back into
        :meth:`get_drive_coefficients`.  Must contain at minimum ``"t_gate"``
        (seconds).
        """
        ...

    def resolve_t_gate(self, system) -> float:
        """The real gate duration (s) of this protocol bound to *system*."""
        return float(self._resolve(system)["t_gate"])

    @property
    def required_channels(self) -> frozenset[str]:
        """Primitive ``E[ket,bra]`` channel names this protocol drives.

        Used by the compiler to validate system-protocol compatibility.
        Concrete protocols override this; the default drives nothing.
        """
        return frozenset()

    def drive_channels(self, system) -> frozenset[str]:
        """Channel names wired into the Hamiltonian for *system*.

        Defaults to :attr:`required_channels`.  Protocols with site-dependent
        drives override this to include per-site channel names.
        """
        return frozenset(self.required_channels)

    @abstractmethod
    def get_drive_coefficients(self, t: float, ctx: dict) -> dict[str, complex]:
        """Return {channel_name: coefficient(t)} for each drive term."""
        ...

    # -- Visualization -----------------------------------------------------

    def pulse_traces(self, t: float, system) -> dict[str, float] | None:
        """Physically-labeled, real-valued input-control traces at time *t*.

        Small numeric hook for plotting the input controls.  Subclasses expose
        their traces by overriding the private ctx-based
        :meth:`_pulse_traces_ctx`; returning ``None`` (the default) makes
        :meth:`plot` fall back to sampling :meth:`get_drive_coefficients`.
        """
        return self._pulse_traces_ctx(t, self._resolve(system))

    def _pulse_traces_ctx(self, t: float, ctx: dict) -> dict[str, float] | None:
        """Trace hook against a pre-resolved context (private)."""
        del t, ctx
        return None

    def plot(self, system, n_points: int = 400):
        """Plot the input pulse over ``[0, t_gate]`` (fixed compact layout).

        When the protocol implements :meth:`pulse_traces`, each trace gets its
        own vertically stacked subplot (shared time axis, independent
        autoscaling) and ``(fig, [axes])`` is returned.  Otherwise
        :meth:`get_drive_coefficients` is sampled and every drive channel is
        drawn on one axis (complex channels split into Re/Im), returning
        ``(fig, ax)``.  Result plotting does not belong to protocols.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if int(n_points) < 2:
            raise ValueError("plot() needs n_points >= 2.")
        ctx = self._resolve(system)
        t_gate = float(ctx["t_gate"])
        if t_gate <= 0:
            raise ValueError("plot() needs a positive t_gate.")

        ts = np.linspace(0.0, t_gate, int(n_points))
        use_traces = self._pulse_traces_ctx(0.0, ctx) is not None
        series: dict[str, list[complex]] = {}
        for t in ts:
            sample = (
                self._pulse_traces_ctx(float(t), ctx)
                if use_traces
                else self.get_drive_coefficients(float(t), ctx)
            )
            for name, val in sample.items():
                series.setdefault(name, []).append(complex(val))
        arrays = {name: np.asarray(vals, dtype=complex) for name, vals in series.items()}
        names = list(arrays)

        if use_traces:
            n = max(len(names), 1)
            fig, axs = plt.subplots(
                n, 1, sharex=True, figsize=(10, max(1.8 * n, 3.0)), squeeze=False
            )
            axs = list(axs[:, 0])
            for axi, name in zip(axs, names):
                y = arrays[name]
                axi.plot(ts, y.real, lw=2, label=name)
                axi.set_ylabel(name, fontsize=9)
                axi.axhline(0.0, color="k", ls=":", lw=0.8)
                axi.grid(alpha=0.3)
            axs[-1].set_xlabel("time (s)")
            axs[0].set_title(f"{type(self).__name__} pulse schedule")
            fig.tight_layout()
            return fig, axs

        fig, ax = plt.subplots(figsize=(10, 4))
        for name in names:
            y = arrays[name]
            scale = max(1.0, float(np.max(np.abs(y))))
            if float(np.max(np.abs(y.imag))) > 1e-9 * scale:
                ax.plot(ts, y.real, lw=2, label=f"Re {name}")
                ax.plot(ts, y.imag, lw=2, ls="--", label=f"Im {name}")
            else:
                ax.plot(ts, y.real, lw=2, label=name)
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("drive (rad/s)")
        ax.set_title(f"{type(self).__name__} pulse schedule")
        ax.grid(alpha=0.3)
        if names:
            ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        return fig, ax
