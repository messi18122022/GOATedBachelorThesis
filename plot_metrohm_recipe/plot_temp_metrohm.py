import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": "\\usepackage{siunitx}\n\\sisetup{detect-all}"
})
import numpy as np

# Zeitpunkte in Stunden
x = [0, 1, 2, 3]

# Temperaturverlauf entsprechend den Intervallen
y = [56, 70, 70, 25]

fig, ax = plt.subplots(figsize=(16/2.54, 6/2.54))

# Temperaturverlauf linear plotten
ax.plot(x, y, marker='o')

# X-Achse mit drei gleich breiten Intervallen und Labels
ax.set_xticks(x)
ax.set_xticklabels(["0 h", "5 min", "22 h", "22 h 10 min"])

# Achsenbeschriftungen
ax.set_xlabel(r"Zeit $t$")
ax.set_ylabel(r"Temperatur $T$ / \si{\celsius}")
ax.set_ylim(20, 80)

ax.set_yticks([25, 56, 70])
ax.set_yticklabels([r"$25$", r"$56$", r"$70$"])
ax.yaxis.grid(True, which="major")

# Gitter anzeigen
ax.grid(True)

plt.subplots_adjust(bottom=0.22)
fig.tight_layout()
fig.savefig("plot_temp_metrohm.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()
