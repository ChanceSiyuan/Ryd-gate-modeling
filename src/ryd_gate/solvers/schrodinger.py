"""Schrödinger equation solvers.

Provides:
- ``solve_gate``: CZ gate solver parameterized by Protocol (phase modulation).
- ``evolve``: Generic solver for arbitrary time-dependent Hamiltonians.
- ``evolve_ir``: Solver that accepts a compiled HamiltonianIR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import integrate

from ryd_gate.blackman import blackman_pulse

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ryd_gate.compilers.ir import HamiltonianIR
    from ryd_gate.core.atomic_system import AtomicSystem
    from ryd_gate.protocols.base import Protocol


def solve_gate(
    system: AtomicSystem,
    protocol: Protocol,
    x: list[float],
    state_mat: "NDArray[np.complexfloating]",
    t_eval: "NDArray[np.floating] | None" = None,
    ham_const_override: "NDArray[np.complexfloating] | None" = None,
    amplitude_scale: float = 1.0,
) -> "NDArray[np.complexfloating]":
    """Evolve a quantum state under the Schrödinger equation.

    Parameters
    ----------
    system : AtomicSystem
        Atomic system with precomputed Hamiltonians.
    protocol : Protocol
        Pulse protocol providing the phase function.
    x : list of float
        Pulse parameters.
    state_mat : ndarray, shape (49,)
        Initial state vector.
    t_eval : ndarray or None
        Times to store solution. None returns only final state.
    ham_const_override : ndarray or None
        If provided, use instead of system.tq_ham_const (for MC perturbations).
    amplitude_scale : float
        Multiplicative scale factor on the 420nm laser amplitude.
        Models quasi-static Rabi frequency fluctuations (default 1.0).

    Returns
    -------
    ndarray
        Shape (49, len(t_eval)) or (49,) if t_eval is None.
    """
    from ryd_gate.core.atomic_system import check_protocol_compatibility
    check_protocol_compatibility(system, protocol)

    params = protocol.unpack_params(x, system)
    t_gate = params["t_gate"]

    ham_const = ham_const_override if ham_const_override is not None else system.tq_ham_const
    ham_static = ham_const + system.tq_ham_1013 + system.tq_ham_1013_conj

    def rhs(t, y):
        phase = protocol.phase_420(t, params)
        phase_conj = np.conjugate(phase)
        amplitude = blackman_pulse(t, system.t_rise, t_gate) if system.blackmanflag else 1
        amplitude *= amplitude_scale

        H = (
            ham_static
            + amplitude * phase * system.tq_ham_420
            + amplitude * phase_conj * system.tq_ham_420_conj
            + amplitude * amplitude * system.tq_ham_lightshift_zero
        )
        return -1j * H @ y

    t_span = [0, t_gate]
    solve_kwargs = dict(
        method="DOP853",
        rtol=1e-8,
        atol=1e-12,
    )
    if t_eval is not None:
        solve_kwargs["t_eval"] = t_eval

    result = integrate.solve_ivp(
        rhs,
        t_span,
        state_mat,
        **solve_kwargs,
    )

    if t_eval is not None:
        return np.array(result.y)
    return np.array(result.y[:, -1])


def evolve(
    hamiltonian_fn: "Callable[[float], NDArray[np.complexfloating]]",
    t_gate: float,
    initial_state: "NDArray[np.complexfloating]",
    t_eval: "NDArray[np.floating] | None" = None,
) -> "NDArray[np.complexfloating]":
    """Generic Schrödinger solver for arbitrary time-dependent H(t).

    Unlike :func:`solve_gate` (which assumes CZ gate Hamiltonian structure),
    this accepts any callable returning a 49x49 Hamiltonian matrix.

    Parameters
    ----------
    hamiltonian_fn : callable
        Function t -> H(t), returning a 49x49 complex matrix.
    t_gate : float
        Total evolution time in seconds.
    initial_state : ndarray, shape (49,)
        Initial state vector.
    t_eval : ndarray or None
        Times to store solution. None returns only final state.

    Returns
    -------
    ndarray
        Shape (49, len(t_eval)) or (49,) if t_eval is None.
    """
    def rhs(t, y):
        H = hamiltonian_fn(t)
        return -1j * H @ y

    t_span = [0, t_gate]
    solve_kwargs = dict(
        method="DOP853",
        rtol=1e-8,
        atol=1e-12,
    )
    if t_eval is not None:
        solve_kwargs["t_eval"] = t_eval

    result = integrate.solve_ivp(
        rhs,
        t_span,
        initial_state,
        **solve_kwargs,
    )

    if t_eval is not None:
        return np.array(result.y)
    return np.array(result.y[:, -1])


def evolve_ir(
    ir: "HamiltonianIR",
    initial_state: "NDArray[np.complexfloating]",
    t_gate: float,
    t_eval: "NDArray[np.floating] | None" = None,
) -> "NDArray[np.complexfloating]":
    """Evolve using a compiled HamiltonianIR.

    This is the new backend-agnostic entry point. The IR describes the
    Hamiltonian as static terms + time-dependent drive terms, with no
    knowledge of the underlying physics (420nm, 1013nm, etc.).

    Parameters
    ----------
    ir : HamiltonianIR
        Compiled Hamiltonian from a Compiler.
    initial_state : ndarray
        Initial state vector.
    t_gate : float
        Total evolution time.
    t_eval : ndarray or None
        Times to store solution. None returns only final state.

    Returns
    -------
    ndarray
        Shape (dim, len(t_eval)) or (dim,) if t_eval is None.
    """
    # Precompute static Hamiltonian (sum of all constant-coefficient terms)
    H_static = np.zeros((ir.dim, ir.dim), dtype=np.complex128)
    for term in ir.static_terms:
        coeff = term.coefficient(0) if callable(term.coefficient) else term.coefficient
        H_static = H_static + coeff * np.asarray(term.operator)

    def rhs(t, y):
        H = H_static.copy()
        for term in ir.drive_terms:
            coeff = term.coefficient(t) if callable(term.coefficient) else term.coefficient
            H = H + coeff * term.operator
            if term.add_hermitian_conjugate:
                H = H + np.conj(coeff) * term.operator.conj().T
        return -1j * H @ y

    t_span = [0, t_gate]
    solve_kwargs = dict(
        method="DOP853",
        rtol=1e-8,
        atol=1e-12,
    )
    if t_eval is not None:
        solve_kwargs["t_eval"] = t_eval

    result = integrate.solve_ivp(
        rhs,
        t_span,
        initial_state,
        **solve_kwargs,
    )

    if t_eval is not None:
        return np.array(result.y)
    return np.array(result.y[:, -1])
