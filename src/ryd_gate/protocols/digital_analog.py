"""Function-defined digital-analog protocol for the 0-1-r Rydberg lattice.

Continuous schedules for four channels:

- ``drive_R``   — hyperfine→Rydberg Rabi amplitude on |1>↔|r| (per atom, Omega_R)
- ``drive_hf``  — hyperfine Rabi amplitude on |0>↔|1| (Omega_hf)
- ``delta_R``   — Rydberg detuning (Delta_R, sign convention: H contains -Delta_R n^r)
- ``delta_hf``  — hyperfine detuning (Delta_hf)

The schedule lives on the protocol, so the parameter vector ``x`` passed to
``simulate()`` is empty.  Each function accepts physical time ``t`` in seconds
and returns either a scalar uniform value or a length-``N`` site profile.

Typical use::

    protocol = DigitalAnalogProtocol(
        t_gate=1e-6,
        omega_R_fn=lambda t: 2*pi*1e6,
        delta_R_fn=lambda t: 0.0,
    )
    system = RydbergSystem.from_lattice(make_chain(2), "01r", protocol=protocol)
    result = simulate(system, [], psi0)
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np

from ryd_gate.protocols.base import Protocol

SiteProfile = float | Sequence[float]
SiteProfileFunction = Callable[[float], SiteProfile]


def is_scalar_profile(value: SiteProfile) -> bool:
    """True when *value* is a single scalar (not a per-site profile)."""
    if isinstance(value, (int, float, complex)):
        return True
    arr = np.asarray(value)
    return arr.ndim == 0


def as_site_profile(value: SiteProfile, n_sites: int) -> np.ndarray:
    """Broadcast a scalar or length-``n_sites`` profile to shape ``(n_sites,)``."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sites, float(arr))
    if arr.shape != (n_sites,):
        raise ValueError(f"Site profile must be a scalar or length-{n_sites} sequence; got shape {arr.shape}.")
    return arr


# (field, global channel, per-site channel prefix, half_factor, negate)
_CHANNEL_SPECS = (
    ("omega_R", "drive_R", "drive_R", True, False),
    ("omega_hf", "drive_hf", "drive_hf", True, False),
    ("delta_R", "delta_R", "delta_R", False, True),
    ("delta_hf", "delta_hf", "delta_hf", False, True),
)


class DigitalAnalogProtocol(Protocol):
    """Function-defined 0-1-r drive schedule.

    Backends still integrate the Hamiltonian using piecewise-constant time
    slices, but sampling continuous schedules is handled inside the protocol.

    Parameters
    ----------
    t_gate : float
        Total evolution time.
    omega_R_fn, omega_hf_fn, delta_R_fn, delta_hf_fn : callable, optional
        Functions ``t -> value`` returning a scalar or length-``N`` site
        profile.  Missing functions default to zero.  Drive functions return
        full Rabi frequencies, not half-Rabi coefficients.
    n_steps : int
        Number of slices that the sparse backend should use to integrate
        the schedule.
    """

    def __init__(
        self,
        *,
        t_gate: float,
        omega_R_fn: SiteProfileFunction | None = None,
        omega_hf_fn: SiteProfileFunction | None = None,
        delta_R_fn: SiteProfileFunction | None = None,
        delta_hf_fn: SiteProfileFunction | None = None,
        n_steps: int = 200,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        self._function_fields = {
            "omega_R": omega_R_fn,
            "omega_hf": omega_hf_fn,
            "delta_R": delta_R_fn,
            "delta_hf": delta_hf_fn,
        }
        for name, fn in self._function_fields.items():
            if fn is not None and not callable(fn):
                raise TypeError(f"{name}_fn must be callable when provided.")
        self._t_gate = float(t_gate)
        self.n_steps = int(n_steps)

    # -- Protocol interface ------------------------------------------------

    @property
    def n_params(self) -> int:
        # Schedule lives on the protocol; x is empty
        return 0

    def validate_params(self, x) -> None:
        if len(x) != 0:
            raise ValueError(
                f"DigitalAnalogProtocol takes no x parameters (schedule is on the protocol); got {len(x)}."
            )

    def unpack_params(self, x, system) -> dict:
        basis = getattr(system, "basis", None)
        n_sites = getattr(basis, "n_sites", None)
        if n_sites is None:
            n_sites = getattr(system, "N", None)
        if n_sites is None and hasattr(system, "meta"):
            n_sites = system.meta("n_sites", None)
        if n_sites is None:
            raise TypeError(
                "DigitalAnalogProtocol.unpack_params() needs a system-like "
                "object with basis.n_sites, N, or meta('n_sites')."
            )
        return {
            "t_gate": self._t_gate,
            "n_sites": int(n_sites),
        }

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset({"drive_R", "drive_hf", "delta_R", "delta_hf"})

    def drive_channels(self, system) -> frozenset[str]:
        """All drive/detuning channels used by this schedule on this lattice."""
        n_sites = system.basis.n_sites
        channels: set[str] = set()
        for field, global_ch, site_prefix, _, _ in _CHANNEL_SPECS:
            if self._function_field_is_scalar(field, n_sites):
                channels.add(global_ch)
            else:
                channels.update(f"{site_prefix}_{i}" for i in range(n_sites))
        return frozenset(channels)

    def _clamp_time(self, t: float) -> float:
        return float(np.clip(float(t), 0.0, self._t_gate))

    def _function_value(self, field: str, t: float) -> SiteProfile:
        fn = self._function_fields[field]
        return 0.0 if fn is None else fn(self._clamp_time(t))

    def _function_field_is_scalar(self, field: str, n_sites: int) -> bool:
        for t in self._function_probe_times():
            value = self._function_value(field, t)
            if not is_scalar_profile(value):
                as_site_profile(value, n_sites)
                return False
        return True

    def _function_probe_times(self) -> tuple[float, ...]:
        return (0.0, 0.25 * self._t_gate, 0.5 * self._t_gate, 0.75 * self._t_gate, self._t_gate)

    def _coeffs_for_function_field(
        self,
        t: float,
        field: str,
        global_ch: str,
        site_prefix: str,
        half_factor: bool,
        negate: bool,
        n_sites: int,
    ) -> dict[str, complex]:
        value = self._function_value(field, t)
        profile = as_site_profile(value, n_sites)
        sign = -1.0 if negate else 1.0
        scale = 0.5 if half_factor else 1.0

        if is_scalar_profile(value):
            return {global_ch: complex(sign * scale * profile[0])}

        return {f"{site_prefix}_{i}": complex(sign * scale * profile[i]) for i in range(n_sites)}

    def get_drive_coefficients(self, t: float, params: dict) -> dict[str, complex]:
        """Return channel coefficients at time *t*.

        Sign / factor conventions matching the compiler:

        - ``drive_R`` / ``drive_R_i``  -> Omega_R / 2  (+ Hermitian conjugate)
        - ``drive_hf`` / ``drive_hf_i`` -> Omega_hf / 2 (+ Hermitian conjugate)
        - ``delta_R`` / ``delta_R_i``  -> -Delta_R on Rydberg projector
        - ``delta_hf`` / ``delta_hf_i`` -> -Delta_hf on |1> projector
        """
        n_sites = int(params.get("n_sites", 1))
        coeffs: dict[str, complex] = {}
        for spec in _CHANNEL_SPECS:
            coeffs.update(self._coeffs_for_function_field(t, *spec, n_sites))
        return coeffs
