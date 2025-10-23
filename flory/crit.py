import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

# Wertebereich für X_N
X_N = np.linspace(1, 1000, 1000)

# Gleichung: chi_c = ((1 + sqrt(X_N))^2) / (2 * X_N)
chi_c = ((1 + np.sqrt(X_N))**2) / (2 * X_N)

# Plot-Grösse in Zoll (16 cm = 6.3 inch, 7 cm = 2.76 inch)
plt.figure(figsize=(6.3, 2.76))

# Plot
plt.plot(X_N, chi_c, label=r'$\chi_c = \frac{(1+\sqrt{X_N})^2}{2X_N}$')
plt.xlim(10, None)
plt.xticks([10, 200, 400,  600,  800,  1000])
plt.ylim(0.4,1)
plt.yticks(np.arange(0.5, 1.0, 0.1))

# Achsenbeschriftungen
plt.xlabel(r'$X_N$')
plt.ylabel(r'$\chi_c$')

# Gitterlinien
plt.grid(True, linestyle='--', alpha=0.6)

# Anzeige
plt.tight_layout()
plt.savefig('flory/chi_c_plot.pdf', bbox_inches='tight')
plt.close()
