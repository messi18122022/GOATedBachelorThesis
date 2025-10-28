import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{siunitx}",
})

# Pfad zur CSV-Datei
file_path = "/Users/musamoin/Desktop/BA-HS25/experiments/water-phase_acid-base-extraction/wässrig.Probe.Rohdaten.csv"

# CSV einlesen
df = pd.read_csv(file_path, sep=";")

# Annahme: Erste Spalte = Wellenlänge, zweite Spalte = Absorbanz
x_col = df.columns[0]
y_col = df.columns[1]

# Plot
plt.figure(figsize=(16/2.54, 8/2.54))
plt.plot(df[x_col], df[y_col], color='blue', linewidth=1.5)

# Maximum in Bereich 480–490 nm markieren
subset = df[(df[x_col] >= 480) & (df[x_col] <= 490)]
if not subset.empty:
    lam_max = 485
    A_max = df.loc[(df[x_col] - lam_max).abs().idxmin(), y_col]
    plt.scatter(lam_max, A_max, color='red', zorder=5)
    plt.axvline(lam_max, color='red', linestyle='--', linewidth=1)
    plt.text(
        lam_max + 2,
        0.02,
        r"$\lambda_\mathrm{max}$ = 485 nm",
        color='red',
        fontsize=11,
        rotation=0,
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )
plt.xlabel(r"Wellenlänge $\lambda$ / \si{\nano\meter}")
plt.ylabel(r"Absorbanz $A$")
plt.xlim(400, 780)
plt.ylim(0, 1.1)
plt.grid(True)
plt.tight_layout()
plt.savefig("water-phase-uvvis/spectrum.pdf", bbox_inches='tight')
plt.show()
