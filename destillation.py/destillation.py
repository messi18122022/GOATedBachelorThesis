import numpy as np
import matplotlib.pyplot as plt

# Anonyme Funktion für die Antoine-Gleichung
# a[0] - A
# a[1] - B
# a[2] - C
def fh_antoine(a, p):
    return a[1] / (a[0] - np.log10(p)) - a[2]


# Ausgabe der Antoine-Gleichung in expliziter Form fuer Wasser und Styrol
print("Antoine-Gleichung (nach T aufgeloest):")
antoine_coeff_h2o = [4.6543, 1435.264, -64.848]   # Antoine-Koeffizienten fuer Wasser in bar und K (gem. Tabelle; gueltig fuer -17.1 bis 100 °C)
antoine_coeff_styrol = [4.0593, 1459.909, -59.551]  # Antoine-Koeffizienten fuer Styrol in bar und K (gem. Tabelle; gueltig fuer 32.5 bis 82.2 °C)
print(f"Wasser: T = {antoine_coeff_h2o[1]} / ({antoine_coeff_h2o[0]} - log10(p_bar)) - ({antoine_coeff_h2o[2]})  [K]")
print(f"Styrol: T = {antoine_coeff_styrol[1]} / ({antoine_coeff_styrol[0]} - log10(p_bar)) - ({antoine_coeff_styrol[2]})  [K]")
print("Hinweis: p_bar ist der Druck in bar, T ist die Temperatur in Kelvin.")


# Definiere einen breiten Druckbereich (z.B. 1e-4 bis 2000 mbar), danach wird ueber die Temperaturmasken eingeschraenkt
druck = np.logspace(-4, 3.3, 20000)  # in mbar (1e-4 bis ca. 2000 mbar)
druck = druck / 1000  # in bar

T_h2o = fh_antoine(antoine_coeff_h2o, druck)      # berechnete Siedepunkte für Wasser
T_styrene = fh_antoine(antoine_coeff_styrol, druck) # berechnete Siedepunkte für Styrol


# Gueltigkeitsbereiche aus der Tabelle (Temperatur in °C)
T_range_h2o = (-17.1, 100.0)
T_range_styrol = (32.5, 82.2)

# Masken fuer die gueltigen Temperaturbereiche
mask_h2o = (T_h2o >= T_range_h2o[0] + 273.15) & (T_h2o <= T_range_h2o[1] + 273.15)
mask_styrol = (T_styrene >= T_range_styrol[0] + 273.15) & (T_styrene <= T_range_styrol[1] + 273.15)

# Gefilterte Arrays fuer den Plot, Druck in bar
T_h2o_plot = T_h2o[mask_h2o]
druck_h2o_plot = druck[mask_h2o]
T_styrene_plot = T_styrene[mask_styrol]
druck_styrol_plot = druck[mask_styrol]

# LaTeX Schrift aktivieren
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

 # Plot fuer Wasser
plt.figure(figsize=(8/2.54, 7/2.54))
plt.plot(T_h2o_plot, druck_h2o_plot, color="blue", linewidth=1)
plt.grid(True)
plt.box(True)
plt.ylabel(r"Druck $p$ / bar")
plt.xlabel(r"Temperatur $T$ / K")
plt.xticks(np.round(np.linspace(min(T_h2o_plot), max(T_h2o_plot), 4)).astype(int))
plt.savefig("destillation.py/wasser.pdf", format="pdf", bbox_inches="tight")
plt.close()

# Plot fuer Styrol
plt.figure(figsize=(8/2.54, 7/2.54))
plt.plot(T_styrene_plot, druck_styrol_plot, color="red", linewidth=1)
plt.grid(True)
plt.box(True)
# plt.ylabel(r"Druck $p$ / bar")  # y-Achsenbeschriftung entfernt
plt.xlabel(r"Temperatur $T$ / K")
plt.xticks(np.round(np.linspace(min(T_styrene_plot), max(T_styrene_plot), 4)).astype(int))
plt.savefig("destillation.py/styrol.pdf", format="pdf", bbox_inches="tight")
plt.close()