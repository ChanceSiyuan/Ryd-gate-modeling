"""Function-defined digital-analog protocol for the 0-1-r Rydberg lattice.

Five mutually independent control functions define the full local 3x3
Hermitian ``{|0>, |1>, |r>}`` Hamiltonian (with the ``|0>`` energy as gauge
zero), in **direct Hamiltonian matrix-element** convention — the same angular
frequency units as the internal Hamiltonian:

- ``coupling_10(t) = H[1, 0](t)`` on ``E[1,0]``  (``|1><0|``)
- ``coupling_r0(t) = H[r, 0](t)`` on ``E[r,0]``  (``|r><0|``)
- ``coupling_r1(t) = H[r, 1](t)`` on ``E[r,1]``  (``|r><1|``)
- ``energy_1(t)   = H[1, 1](t)`` on ``E[1,1]``  (``|1><1|`` projector)
- ``energy_r(t)   = H[r, r](t)`` on ``E[r,r]``  (``|r><r|`` projector)

The three off-diagonal couplings may be complex; the compiler adds **only**
the Hermitian conjugate (no extra 1/2 — a resonant Rabi drive of full Rabi
frequency ``Omega`` is ``coupling_r1 = Omega/2 * exp(-i*phi)``).  The two
diagonal energies must be real.  A function that is not provided means that
channel does not exist.  Each function accepts physical time ``t`` in seconds
and returns either a uniform scalar or a length-``N`` site profile.

Typical use::

    protocol = DigitalAnalogProtocol(
        t_gate=1e-6,
        coupling_r1=lambda t: 0.5 * 2*pi*1e6,   # Omega/2 matrix element
        energy_r=lambda t: 0.0,
    )
    system = RydbergSystem(
        level_structure=level_structure("01r"),
        register=Register.chain(2),
        protocol=protocol,
    )
    result = simulate(system, psi0)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ryd_gate.protocols.base import Protocol

SiteProfile = complex | float | Sequence[complex] | Sequence[float]
SiteProfileFunction = Callable[[float], SiteProfile]


def is_scalar_profile(value: SiteProfile) -> bool:
    """True when *value* is a single scalar (not a per-site profile)."""
    if isinstance(value, (int, float, complex)):
        return True
    arr = np.asarray(value)
    return arr.ndim == 0


def as_site_profile(value: SiteProfile, n_sites: int) -> np.ndarray:
    """Broadcast a scalar or length-``n_sites`` profile to shape ``(n_sites,)``.

    Complex-valued (coupling) profiles are preserved as complex.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Site profile must be numeric; got dtype {arr.dtype}.")
    arr = arr.astype(complex)
    if arr.ndim == 0:
        return np.full(n_sites, complex(arr))
    if arr.shape != (n_sites,):
        raise ValueError(f"Site profile must be a scalar or length-{n_sites} sequence; got shape {arr.shape}.")
    return arr


# (field name, channel, allows complex values).  The field name is the
# constructor keyword; the channel is the primitive E[ket,bra] name the IR
# consumes; couplings may be complex, energies must be real.
_CHANNEL_SPECS = (
    ("coupling_10", "E[1,0]", True),
    ("coupling_r0", "E[r,0]", True),
    ("coupling_r1", "E[r,1]", True),
    ("energy_1", "E[1,1]", False),
    ("energy_r", "E[r,r]", False),
)


class DigitalAnalogProtocol(Protocol):
    """Function-defined 0-1-r drive schedule in matrix-element convention.

    The control functions are sampled at whatever times the backend's
    integrator requests (the exact backend is adaptive), so they must be
    well-defined at every ``t`` in ``[0, t_gate]``, not just on a step grid.

    Parameters
    ----------
    t_gate : float
        Total evolution time.
    coupling_10, coupling_r0, coupling_r1 : callable, optional
        Functions ``t -> value`` returning the (possibly complex) Hamiltonian
        matrix element ``H[upper, lower](t)`` as a scalar or length-``N`` site
        profile.  The compiler adds only the Hermitian conjugate.
    energy_1, energy_r : callable, optional
        Functions ``t -> value`` returning the real diagonal energies
        ``H[1,1]`` / ``H[r,r]`` (the ``|0>`` energy is the gauge zero).
    """

    def __init__(
        self,
        *,
        t_gate: float,
        coupling_10: SiteProfileFunction | None = None,
        coupling_r0: SiteProfileFunction | None = None,
        coupling_r1: SiteProfileFunction | None = None,
        energy_1: SiteProfileFunction | None = None,
        energy_r: SiteProfileFunction | None = None,
    ) -> None:
        if t_gate <= 0:
            raise ValueError("t_gate must be positive.")
        self._function_fields = {
            "coupling_10": coupling_10,
            "coupling_r0": coupling_r0,
            "coupling_r1": coupling_r1,
            "energy_1": energy_1,
            "energy_r": energy_r,
        }
        for name, fn in self._function_fields.items():
            if fn is not None and not callable(fn):
                raise TypeError(f"{name} must be callable when provided.")
        if all(fn is None for fn in self._function_fields.values()):
            raise ValueError("DigitalAnalogProtocol needs at least one control function.")
        self._t_gate = float(t_gate)

    @property
    def t_gate(self) -> float:
        return self._t_gate

    # -- Protocol interface ------------------------------------------------

    def _resolve(self, system) -> dict:
        basis = getattr(system, "basis", None)
        n_sites = getattr(basis, "n_sites", None)
        if n_sites is None:
            n_sites = getattr(system, "N", None)
        if n_sites is None:
            raise TypeError(
                "DigitalAnalogProtocol needs a system-like object with "
                "basis.n_sites or N."
            )
        n_sites = int(n_sites)
        self._validate_energies_real(n_sites)
        return {"t_gate": self._t_gate, "n_sites": n_sites}

    @property
    def required_channels(self) -> frozenset[str]:
        return frozenset(
            channel
            for field, channel, _ in _CHANNEL_SPECS
            if self._function_fields[field] is not None
        )

    def drive_channels(self, system) -> frozenset[str]:
        """All channels used by this schedule on this lattice.

        Scalar control functions declare the global channel name; per-site
        profiles declare ``E[...]_{i}`` for each site.  Absent functions
        declare nothing (the channel does not exist).
        """
        n_sites = system.basis.n_sites
        channels: set[str] = set()
        for field, channel, _ in _CHANNEL_SPECS:
            if self._function_fields[field] is None:
                continue
            if self._function_field_is_scalar(field, n_sites):
                channels.add(channel)
            else:
                channels.update(f"{channel}_{i}" for i in range(n_sites))
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

    def _validate_energies_real(self, n_sites: int) -> None:
        for field, _, allows_complex in _CHANNEL_SPECS:
            if allows_complex or self._function_fields[field] is None:
                continue
            for t in self._function_probe_times():
                profile = as_site_profile(self._function_value(field, t), n_sites)
                if np.any(np.abs(profile.imag) > 0.0):
                    raise ValueError(
                        f"{field} must be real (diagonal energies); got a complex "
                        f"value at t={t!r}."
                    )

    def get_drive_coefficients(self, t: float, ctx: dict) -> dict[str, complex]:
        """Return channel coefficients at time *t*.

        Direct matrix-element convention: each coupling coefficient is the
        Hamiltonian entry ``H[upper, lower](t)`` itself (the compiler adds only
        the Hermitian conjugate), and each energy coefficient multiplies the
        level projector directly.
        """
        n_sites = int(ctx.get("n_sites", 1))
        coeffs: dict[str, complex] = {}
        for field, channel, _ in _CHANNEL_SPECS:
            if self._function_fields[field] is None:
                continue
            value = self._function_value(field, t)
            profile = as_site_profile(value, n_sites)
            if is_scalar_profile(value):
                coeffs[channel] = complex(profile[0])
            else:
                coeffs.update({f"{channel}_{i}": complex(profile[i]) for i in range(n_sites)})
        return coeffs

    def _pulse_traces_ctx(self, t: float, ctx: dict) -> dict[str, float]:
        """Site-averaged |matrix element| traces of the active control functions."""
        n_sites = int(ctx.get("n_sites", 1))
        labels = {
            "coupling_10": r"$|H_{10}|$", "coupling_r0": r"$|H_{r0}|$",
            "coupling_r1": r"$|H_{r1}|$",
            "energy_1": r"$E_1$", "energy_r": r"$E_r$",
        }

        def mean_at(field: str, t_query: float) -> float:
            profile = as_site_profile(self._function_value(field, t_query), n_sites)
            if field.startswith("coupling"):
                return float(np.mean(np.abs(profile)))
            return float(np.mean(profile.real))

        active = [f for f in labels if self._function_fields[f] is not None]
        return {labels[field]: mean_at(field, t) for field in active}
