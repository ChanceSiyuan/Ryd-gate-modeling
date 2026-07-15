"""PEPS mathematical-validity helpers (PEPS03/PEPS34). No YASTN import.

The one private validity relative scale, the private ``PEPSError``, and the
device-scalar-to-host real/complex conversions. Convergence is never a failure
here (PEPS02); only mathematically-meaningless numbers are (NaN/Inf, a non-real
"real" scalar beyond roundoff slack, a non-positive norm).
"""

from __future__ import annotations

import numpy as np

# PEPS34: allowed imaginary roundoff on a theoretically-real scalar is this
# relative scale times max(1, |z|). Independent of environment_tolerance.
_VALIDITY_RTOL = float(np.sqrt(np.finfo(np.float64).eps))


class PEPSError(RuntimeError):
    """Raised when the YASTN PEPS adapter cannot run or validate a request (private)."""


def _to_host_complex(value) -> complex:
    """Bring a device scalar to a host Python ``complex`` (only scalars, PEPS34)."""
    item = value.item() if hasattr(value, "item") else value
    return complex(item)


def real_scalar(value, what: str) -> float:
    """Return ``float(z.real)`` for a theoretically-real scalar; validity-check it (PEPS34)."""
    z = _to_host_complex(value)
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        raise PEPSError(f"{what} is not finite: {z!r}.")
    if abs(z.imag) > _VALIDITY_RTOL * max(1.0, abs(z)):
        raise PEPSError(
            f"{what} has an imaginary part too large to read as real "
            f"({z.imag:.3e} vs slack {_VALIDITY_RTOL * max(1.0, abs(z)):.3e}); this is a "
            "mathematical-validity failure, not a convergence failure."
        )
    return float(z.real)


def positive_norm(value, what: str) -> float:
    """Validate a theoretically-real, strictly-positive scalar (PEPS norm)."""
    v = real_scalar(value, what)
    if not v > 0.0:
        raise PEPSError(f"{what} must be strictly positive; got {v!r}.")
    return v


def finite_complex(value, what: str) -> complex:
    """Bring a scalar to host ``complex`` and require it finite (amplitude numerator)."""
    z = _to_host_complex(value)
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        raise PEPSError(f"{what} is not finite: {z!r}.")
    return z


def finite_nonneg(value, what: str) -> float:
    """Require a finite, non-negative float (NTU / contraction error summaries)."""
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise PEPSError(f"{what} must be finite and non-negative; got {value!r}.")
    return v
