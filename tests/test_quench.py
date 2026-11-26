
import numpy as np

from src.quantum_quench import (
    simulate_quench,
    loschmidt_echo,
)

def test_probability_conservation():
    times = np.linspace(0, 5, 51)
    _, _, states = simulate_quench(6, 1.0, 0.5, 1.5, times)

    norms = np.sum(np.abs(states)**2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_post_quench_energy_conservation():
    times = np.linspace(0, 5, 51)
    _, H, states = simulate_quench(6, 1.0, 0.5, 1.5, times)

    H_states = (H @ states.T).T
    energies = np.sum(states.conj() * H_states, axis=1).real

    assert np.allclose(energies, energies[0], atol=1e-8)


def test_no_quench_echo():
    times = np.linspace(0, 5, 51)
    psi0, _, states = simulate_quench(6, 1.0, 0.5, 0.5, times)

    echo = loschmidt_echo(psi0, states)
    assert np.allclose(echo, 1.0, atol=1e-8)

def test_loschmidt_echo_bounds():
    times = np.linspace(0, 10, 101)

    psi0, _, states = simulate_quench(
        6, 1.0, 0.5, 1.5, times
    )

    echo = loschmidt_echo(psi0, states)

    assert np.all(echo >= -1e-10)
    assert np.all(echo <= 1.0 + 1e-10)
    assert np.isclose(echo[0], 1.0)