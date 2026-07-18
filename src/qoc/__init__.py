"""qoc — numerical parameter optimization through caller-owned scalar losses.

The base interface is single-start scalar-loss minimization (ADR-0015)::

    import qoc

    result = qoc.minimize(loss, x0, method="l-bfgs-b", bounds=..., scales=...)

``loss`` receives one candidate as an ordinary named mapping of real scalars
and real arrays and returns exactly one finite real scalar. The caller owns
everything inside that function; ``qoc`` neither sees nor reconstructs physical
systems, protocols, or simulations.

``qoc.grape`` holds the discrete-adjoint GRAPE engine over one bilinear
control model (ADR-0024). It consumes only plain arrays and mappings and never
imports a physics package.
"""

__version__ = "0.1.0"

from . import grape
from ._minimize import minimize
from ._result import OptimizationResult

__all__ = [
    "OptimizationResult",
    "grape",
    "minimize",
]
