"""Abstract base class for pulse protocols (P05/P06/P17).

A protocol is fully specified at construction. Its only public runtime surface
is the constructor and :meth:`plot`. Binding it to a system and lowering it to
the backends happens through the single private seam
``_resolve(system) -> _ResolvedProtocol``; there are no ``required_channels`` /
``drive_channels`` / ``get_drive_coefficients`` / ``resolve_t_gate`` hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ryd_gate.protocols._resolved import _LaserDrive, _ResolvedProtocol


class Protocol(ABC):
    """Abstract base class for fully specified pulse protocols."""

    #: Preset names this protocol is compatible with (checked at system binding).
    _compatible_presets: frozenset[str] = frozenset()

    @abstractmethod
    def _resolve(self, system) -> _ResolvedProtocol:
        """Resolve against *system* into the private canonical drive IR (P17)."""
        ...

    def plot(self, system, n_points: int = 400):
        """Plot the input drive controls over ``[0, t_gate]`` (P06).

        Returns a single :class:`matplotlib.figure.Figure`; it does not run the
        simulation, call ``show()`` or save anything. Use ``figure.axes`` for
        further customization.
        """
        import matplotlib.pyplot as plt

        if int(n_points) < 2:
            raise ValueError("plot() needs n_points >= 2.")
        resolved = self._resolve(system)
        t_gate = float(resolved.t_gate)
        if t_gate <= 0:
            raise ValueError("plot() needs a positive t_gate.")
        ts = np.linspace(0.0, t_gate, int(n_points))
        drives = resolved.drives or ()

        n = max(len(drives), 1)
        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(10, max(1.8 * n, 3.0)), squeeze=False)
        axs = list(axs[:, 0])
        for ax, drive in zip(axs, drives):
            label = (
                f"laser {drive.group}" if isinstance(drive, _LaserDrive) else drive.channel
            )
            vals = np.array([_representative(drive.coefficient(float(t))) for t in ts], dtype=complex)
            ax.plot(ts, vals.real, lw=2, label=f"Re {label}")
            if np.max(np.abs(vals.imag)) > 1e-9 * max(1.0, np.max(np.abs(vals))):
                ax.plot(ts, vals.imag, lw=2, ls="--", label=f"Im {label}")
            ax.set_ylabel(label, fontsize=9)
            ax.axhline(0.0, color="k", ls=":", lw=0.8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axs[-1].set_xlabel("time (s)")
        axs[0].set_title(f"{type(self).__name__} drive schedule")
        fig.tight_layout()
        return fig


def _representative(value) -> complex:
    """Reduce a scalar or per-site coefficient value to one complex number for plotting."""
    arr = np.asarray(value)
    return complex(arr.reshape(-1).mean()) if arr.ndim else complex(value)
