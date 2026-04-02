"""Drive channel name conventions for protocol-system communication.

Channel names used by get_drive_coefficients() to label time-dependent
Hamiltonian terms. The compiler maps these to system block operators.

Two-atom 7-level CZ gates (TO/AR protocols):
    "drive_420"         -> H_420 coupling operator
    "drive_420_dag"     -> H_420 hermitian conjugate
    "lightshift_zero"   -> dark-state light-shift operator

Two-atom 3-level sweep (analog system):
    "drive_420"         -> H_420 coupling
    "drive_420_dag"     -> H_420 hermitian conjugate
    "lightshift_zero"   -> (zero matrix, included for interface uniformity)

N-atom 2-level lattice sweep:
    "global_X"          -> sum_i sigma^x_i (transverse field)
    "global_n"          -> sum_i n_i (detuning / longitudinal field)
"""

# Channel name constants for type safety
DRIVE_420 = "drive_420"
DRIVE_420_DAG = "drive_420_dag"
LIGHTSHIFT_ZERO = "lightshift_zero"
GLOBAL_X = "global_X"
GLOBAL_N = "global_n"
