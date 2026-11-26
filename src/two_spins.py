
import numpy as np

# Single-spin operators
I = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


def build_two_spin_hamiltonian(J, h):
    """Construct a two-spin transverse-field Ising Hamiltonian."""

    interaction = np.kron(Z, Z)

    transverse_field = (
        np.kron(X, I)
        + np.kron(I, X)
    )

    H = -J * interaction - h * transverse_field

    return H


if __name__ == "__main__":
    J = 1.0
    h = 0.8

    H = build_two_spin_hamiltonian(J, h)

    print("Hamiltonian:")
    print(H.real)

    # Hermiticity: H = H dagger
    assert np.allclose(H, H.conj().T)

    # Diagonalise the Hamiltonian
    energies, states = np.linalg.eigh(H)

    print("\nEnergy eigenvalues:")
    print(energies)

    print("\nGround-state energy:")
    print(energies[0])

    print("\nGround-state vector:")
    print(states[:, 0])

    # Analytical ground-state energy for J >= 0
    expected = -np.sqrt(J**2 + 4 * h**2)

    assert np.isclose(energies[0], expected)

    print("\nAnalytical validation passed!")
