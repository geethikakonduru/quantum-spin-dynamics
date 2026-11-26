
import numpy as np

from src.two_spins import build_two_spin_hamiltonian
from src.spin_chain import build_hamiltonian, ground_state


def test_hamiltonian_is_hermitian():
    H = build_hamiltonian(N=6, J=1.0, h=0.8)

    difference = H - H.getH()

    assert np.allclose(difference.toarray(), 0)


def test_hilbert_space_dimension():
    for N in [2, 4, 6, 8]:
        H = build_hamiltonian(N, J=1.0, h=0.8)
        assert H.shape == (2**N, 2**N)


def test_two_spin_analytical_result():
    J = 1.0
    h = 0.8

    H = build_two_spin_hamiltonian(J, h)

    energies = np.linalg.eigvalsh(H)

    expected = -np.sqrt(J**2 + 4 * h**2)

    assert np.isclose(energies[0], expected)


def test_general_hamiltonian_matches_two_spins():
    J = 1.0
    h = 0.8

    H_general = build_hamiltonian(2, J, h).toarray()
    H_two_spin = build_two_spin_hamiltonian(J, h)

    assert np.allclose(H_general, H_two_spin)


def test_ground_state_normalisation():
    _, state = ground_state(N=6, J=1.0, h=0.8)

    assert np.isclose(np.vdot(state, state), 1.0)


def test_ground_state_eigenvalue_equation():
    N = 6
    J = 1.0
    h = 0.8

    H = build_hamiltonian(N, J, h)
    energy, state = ground_state(N, J, h)

    residual = H @ state - energy * state

    assert np.linalg.norm(residual) < 1e-8
