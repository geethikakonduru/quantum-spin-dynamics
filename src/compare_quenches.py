
import numpy as np
import matplotlib.pyplot as plt

from src.quantum_quench import (
    simulate_quench,
    loschmidt_echo,
    transverse_magnetisation,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# System parameters
N = 6
J = 1.0
h_initial = 0.5

# Different post-quench field strengths
final_fields = [0.8, 1.5, 2.0]

# Time interval in units of hbar / J
times = np.linspace(0, 10, 201)

# Store results
results = {}

for h_final in final_fields:

    psi0, H_final, states = simulate_quench(
        N=N,
        J=J,
        h_initial=h_initial,
        h_final=h_final,
        times=times,
    )

    echo = loschmidt_echo(psi0, states)

    magnetisation = transverse_magnetisation(
        states, N
    )

    data = np.column_stack((times, echo, magnetisation))

    np.savetxt(
        RESULTS_DIR / f"quench_h1_{h_final:.1f}.csv",
        data,
        delimiter=",",
        header="time,loschmidt_echo,transverse_magnetisation",
        comments="",
    )

    results[h_final] = {
        "echo": echo,
        "magnetisation": magnetisation,
    }

    print(
        f"h_final={h_final:.1f} | "
        f"quench={h_final - h_initial:.1f} | "
        f"min echo={echo.min():.6f} | "
        f"final magnetisation={magnetisation[-1]:.6f}"
    )


# Plot 1: Loschmidt echo comparison
plt.figure(figsize=(9, 5))

for h_final in final_fields:
    plt.plot(
        times,
        results[h_final]["echo"],
        label=f"h₁ = {h_final}",
    )

plt.xlabel("Time (ħ/J)")
plt.ylabel("Loschmidt echo")
plt.title("Loschmidt Echo for Different Quench Strengths")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "loschmidt_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Plot 2: Magnetisation comparison
plt.figure(figsize=(9, 5))

for h_final in final_fields:
    plt.plot(
        times,
        results[h_final]["magnetisation"],
        label=f"h₁ = {h_final}",
    )

plt.xlabel("Time (ħ/J)")
plt.ylabel("Transverse magnetisation")
plt.title("Magnetisation Dynamics for Different Quench Strengths")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "magnetisation_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
