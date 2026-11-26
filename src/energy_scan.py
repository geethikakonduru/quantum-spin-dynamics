
import numpy as np
import matplotlib.pyplot as plt

from spin_chain import ground_state

N = 6
J = 1.0

fields = np.linspace(0.1, 2.0, 30)

energies = []

for h in fields:
    energy, _ = ground_state(N, J, h)
    energies.append(energy / N)

plt.plot(fields, energies, marker="o", markersize=3)

plt.xlabel("Transverse field h / J")
plt.ylabel("Ground-state energy per spin / J")
plt.title(f"Transverse-field Ising model, N={N}")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
