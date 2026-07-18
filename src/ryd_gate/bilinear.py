"""Bilinear control model export (ADR-0024).

The narrow public window that returns a protocol-bound system's
already-compiled bilinear form as plain arrays for search-side use:

    h0, controls, states = bilinear_control_model(system, states=[["0", "1"]])

with the semantics ``H(t) = h0 + sum_a u_a(t) * controls[a]`` where every
``u_a`` is real and every operator is Hermitian. An off-diagonal drive channel
``E[ket,bra]`` with complex coefficient ``c(t)`` appears as the two quadrature
channels ``E[ket,bra]:x`` (operator ``A + A^H``, value ``Re c``) and
``E[ket,bra]:y`` (operator ``1j*(A - A^H)``, value ``Im c``); a diagonal
channel appears as one real channel (operator ``A``, value ``Re c``).

The export shares the compiler with ``simulate``, uses the bound protocol only
for channel structure, refuses noise realizations, and does not replace
exact-ODE candidate validation. Basis ordering, angular-frequency units
(rad/s), and the quadrature conventions are pinned by a constant-control
parity test against ``simulate``, not by exported metadata.
"""

from __future__ import annotations

import numpy as np

from .backends.exact.compiler import compile_exact
from .core.states import dense_plus_state, dense_product_state


def bilinear_control_model(system, *, states=()):
    """Return ``(h0, controls, state_vectors)`` for a protocol-bound system.

    ``h0`` is the Hermitian static matrix, ``controls`` maps channel labels to
    Hermitian operators (see the module docstring for the channel semantics),
    and ``state_vectors`` maps each requested entry of ``states`` — a per-site
    label sequence such as ``["0", "1"]`` (keyed by its tuple) or the string
    ``"plus"`` — to its dense state vector in the same basis ordering as the
    matrices.
    """
    if getattr(system, "_realization", None) is not None:
        raise ValueError(
            "bilinear_control_model requires a noiseless system; it does not accept noise realizations."
        )

    hamiltonian, _t_gate = compile_exact(system, hamiltonian_format="dense")
    h0 = np.array(hamiltonian.h_static, dtype=complex)

    controls: dict[str, np.ndarray] = {}
    for channel in hamiltonian.channels:
        label = channel._channel
        if np.asarray(channel.coeff(0.0)).ndim != 0:
            raise ValueError(
                f"drive channel {label!r} has a per-site coefficient; "
                "bilinear_control_model supports only scalar-coefficient channels."
            )
        operator = np.asarray(channel.sum_op, dtype=complex)
        if channel.is_diag:
            controls[label] = np.array(operator)
        else:
            controls[f"{label}:x"] = operator + operator.conj().T
            controls[f"{label}:y"] = 1j * (operator - operator.conj().T)

    basis = system._basis
    state_vectors: dict = {}
    for entry in states:
        if isinstance(entry, str):
            if entry != "plus":
                raise ValueError(
                    f"states entries must be per-site label sequences or 'plus'; got {entry!r}."
                )
            state_vectors["plus"] = dense_plus_state(basis)
            continue
        labels = tuple(entry)
        if len(labels) != basis.n_sites or not all(isinstance(label, str) for label in labels):
            raise ValueError(
                f"states entry {entry!r} must give one label string per site ({basis.n_sites} sites)."
            )
        state_vectors[labels] = dense_product_state(labels, basis)

    return h0, controls, state_vectors
