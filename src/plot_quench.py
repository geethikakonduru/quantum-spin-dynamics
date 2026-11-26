
import numpy as np
import matplotlib.pyplot as plt

from src.quantum_quench import (
    simulate_quench,
    loschmidt_echo,
    transverse_magnetisation,
)

N = 6
J = 1.0

h_initial = 0.5
h_final = 1.5

times = np.linspace(0, 10, 201)

psi0, H_final, states = simulate_quench(
    N, J, h_initial, h_final, times
)

echo = loschmidt_echo(psi0, states)
magnetisation = transverse_magnetisation(states, N)

# Loschmidt echo
plt.figure()
plt.plot(times, echo)
plt.xlabel("Time (ħ/J)")
plt.ylabel("Loschmidt echo")
plt.title("Return probability after a quantum quench")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Transverse magnetisation
plt.figure()
plt.plot(times, magnetisation)
plt.xlabel("Time (ħ/J)")
plt.ylabel("Transverse magnetisation")
plt.title("Magnetisation dynamics after a quantum quench")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
