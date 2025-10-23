import numpy as np
import matplotlib.pyplot as plt

"""
Flory-Huggins: ΔG^m/RT und erste, zweite, dritte Ableitung
Parameter: XN = 10, chi = 1.5
Erzeugt einen Overlay-Plot mit 4 Kurven über φ2 ∈ (0,1).
"""

# Parameter
XN = 10.0
chi = 0.8661

# Funktionen
def f(phi, XN, chi):
    return (1 - phi) * np.log(1 - phi) + (phi / XN) * np.log(phi) + (1 - phi) * phi * chi

# Erste Ableitung
def f1(phi, XN, chi):
    return -np.log(1 - phi) - 1 + (1.0 / XN) * (np.log(phi) + 1) + chi * (1 - 2 * phi)

# Zweite Ableitung
def f2(phi, XN, chi):
    return 1.0 / (1 - phi) + 1.0 / (XN * phi) - 2 * chi

# Dritte Ableitung
def f3(phi, XN, chi):
    return 1.0 / (1 - phi) ** 2 - 1.0 / (XN * phi ** 2)

# Vierte Ableitung
def f4(phi, XN, chi):
    return 2.0 / (1 - phi) ** 3 + 2.0 / (XN * phi ** 3)


def main():
    # Domain (Singularitäten bei 0 und 1 vermeiden)
    eps = 1e-4
    phi = np.linspace(eps, 1 - eps, 2000)

    # Werte berechnen
    y0 = f(phi, XN, chi)
    y1 = f1(phi, XN, chi)
    y2 = f2(phi, XN, chi)
    y3 = f3(phi, XN, chi)
    y4 = f4(phi, XN, chi)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(phi, y0, label=r"$\Delta G^m/RT$")
    plt.plot(phi, y1, label=r"$\frac{d}{d\varphi_2}(\Delta G^m/RT)$")
    plt.plot(phi, y2, label=r"$\frac{d^2}{d\varphi_2^2}(\Delta G^m/RT)$")
    plt.plot(phi, y3, label=r"$\frac{d^3}{d\varphi_2^3}(\Delta G^m/RT)$")
    plt.plot(phi, y4, label=r"$\frac{d^4}{d\varphi_2^4}(\Delta G^m/RT)$")
    plt.xlabel(r"$\varphi_2$")
    plt.ylabel("Wert")
    plt.xlim(0, 1)
    plt.ylim(-0.3, 0.1)
    plt.title(r"Flory-Huggins: $\Delta G^m/RT$ und Ableitungen (X$_N$=10, $\chi$=1.5)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()