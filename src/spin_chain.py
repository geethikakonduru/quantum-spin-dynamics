
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

I = sparse.eye(2, format="csr", dtype=complex)

X = sparse.csr_matrix([
    [0, 1],
    [1, 0]
], dtype=complex)

Z = sparse.csr_matrix([
    [1, 0],
    [0, -1]
], dtype=complex)


def local_operator(operator, site, N):
    """Operator acting on a selected site of an N-spin chain."""

    if N < 1 or not 0 <= site < N:
        raise ValueError("Invalid chain length or site index")

    matrices = [
        operator if i == site else I
        for i in range(N)
    ]

    result = matrices[0]

    for matrix in matrices[1:]:
        result = sparse.kron(result, matrix, format="csr")

    return result


def build_hamiltonian(N, J, h):
    """Build the N-spin Ising Hamiltonian with open boundaries."""

    if N < 2:
        raise ValueError("N must be at least 2")

    dimension = 2**N

    H = sparse.csr_matrix(
        (dimension, dimension),
        dtype=complex
    )

    z_ops = [local_operator(Z, i, N) for i in range(N)]
    x_ops = [local_operator(X, i, N) for i in range(N)]

    # Interaction between neighbouring spins
    for i in range(N - 1):
        H -= J * (z_ops[i] @ z_ops[i + 1])

    # External transverse magnetic field
    for i in range(N):
        H -= h * x_ops[i]

    return H


def ground_state(N, J, h):
    """Calculate the lowest-energy eigenpair."""

    H = build_hamiltonian(N, J, h)

    energies, states = eigsh(
        H, k=1, which="SA", tol=1e-10
    )

    return energies[0].real, states[:, 0]


if __name__ == "__main__":
    J = 1.0
    h = 0.8

    for N in [4, 6, 8]:
        energy, state = ground_state(N, J, h)

        print(
            f"N={N:2d} | "
            f"Dimension={2**N:4d} | "
            f"Ground-state energy={energy:.8f}"
        )
