
import numpy as np
from scipy.sparse.linalg import expm_multiply

from src.spin_chain import (
    X,
    local_operator,
    build_hamiltonian,
    ground_state,
)


def simulate_quench(N, J, h_initial, h_final, times):
    """
    Prepare the initial ground state and evolve it
    under the post-quench Hamiltonian.
    """

    # Ground state before the quench
    E0, psi0 = ground_state(N, J, h_initial)

    # Hamiltonian after the quench
    H_final = build_hamiltonian(N, J, h_final)

    # Time evolution: |psi(t)> = exp(-i H_final t)|psi0>
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("Provide at least two time points")

    if not np.isclose(times[0], 0):
        raise ValueError("Times must start at zero")

    dt = np.diff(times)

    if np.any(dt <= 0) or not np.allclose(dt, dt[0]):
        raise ValueError("Times must be equally spaced and increasing")

    states = expm_multiply(
        -1j * H_final,
        psi0,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=0.0
    )

    return psi0, H_final, states


def loschmidt_echo(psi0, states):
    """Calculate the return probability to the initial state."""

    overlaps = states @ psi0.conj()
    return np.abs(overlaps) ** 2

def transverse_magnetisation(states, N):
    """Calculate the transverse magnetisation at each time."""

    magnetisation = np.zeros(len(states), dtype=float)

    for i in range(N):
        Xi = local_operator(X, i, N)

        # Apply Xi to every time-evolved state
        applied = (Xi @ states.T).T

        # <psi(t)|Xi|psi(t)>
        expectation = np.sum(
            states.conj() * applied,
            axis=1
        )

        magnetisation += expectation.real

    return magnetisation / N


if __name__ == "__main__":
    N = 6
    J = 1.0
    h_initial = 0.5
    h_final = 1.5

    times = np.linspace(0, 10, 101)

    psi0, H_final, states = simulate_quench(
        N, J, h_initial, h_final, times
    )

    echo = loschmidt_echo(psi0, states)

    print("Loschmidt echo at t=0:", echo[0])
    print("Minimum Loschmidt echo:", echo.min())

    print("Initial state dimension:", psi0.shape)
    print("Time-evolved states shape:", states.shape)

    magnetisation = transverse_magnetisation(states, N)

    print("Initial transverse magnetisation:", magnetisation[0])
    print("Final transverse magnetisation:", magnetisation[-1])

    norms = np.sum(np.abs(states) ** 2, axis=1)
    print("Maximum normalisation error:",
          np.max(np.abs(norms - 1)))


