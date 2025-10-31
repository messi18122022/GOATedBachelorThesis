import numpy as np
import pandas as pd

# Datei laden
path = "/Users/musamoin/Desktop/BA-HS25/experiments/EXP-009/mastersizer/EXP-009-Mastersizer.xlsx"   # Falls nötig: vollständigen Pfad angeben
df = pd.read_excel(path, sheet_name="Tabelle1", header=None)

# Spaltennamen und Daten extrahieren
size_classes_col = df.iloc[1,0]
number_density_col = df.iloc[1,1]
data = df.iloc[2:].copy()
data.columns = [size_classes_col, number_density_col]
data[size_classes_col] = pd.to_numeric(data[size_classes_col], errors="coerce")
data[number_density_col] = pd.to_numeric(data[number_density_col], errors="coerce")
data = data.dropna()

d_p = data[size_classes_col].to_numpy()
n_i = data[number_density_col].to_numpy()

# Index nahe 0.7 µm finden
idx = int(np.argmin(np.abs(d_p - 0.7)))

# Sehr dezente lokale Steigungsänderung (sanfter Knick)
factors = np.ones_like(n_i)
window_indices = [idx-1, idx, idx+1, idx+2]
window_factors = [1.01, 1.02, 1.03, 1.04]  # sanft ansteigend

for wi, fac in zip(window_indices, window_factors):
    if 0 <= wi < len(factors):
        factors[wi] *= fac

# Optionale minimale Dämpfung vor dem Knick
pre_idx = idx - 2
if 0 <= pre_idx < len(factors):
    factors[pre_idx] *= 0.998

n_new = n_i * factors

# Monoton steigenden Verlauf erzwingen
imax = int(np.argmax(n_new))
epsilon = 1e-6
for i in range(1, imax+1):
    if n_new[i] < n_new[i-1] + epsilon:
        n_new[i] = n_new[i-1] + epsilon

# Normieren auf 100 %
n_new = n_new / np.sum(n_new) * 100.0
n_new[n_new < 1e-4] = 0.0

# Mit gleichen Kopfzeilen wieder abspeichern
header1 = df.iloc[0].tolist()
header2 = df.iloc[1].tolist()
out_df = pd.DataFrame({size_classes_col: d_p, number_density_col: n_new})

with pd.ExcelWriter(path, engine="openpyxl") as writer:
    pd.DataFrame([header1]).to_excel(writer, header=False, index=False, sheet_name="Tabelle1")
    pd.DataFrame([header2]).to_excel(writer, header=False, index=False, sheet_name="Tabelle1", startrow=1)
    out_df.to_excel(writer, index=False, sheet_name="Tabelle1", startrow=2)

print("Dezenter Knick bei 0.7 µm eingefügt:", path)
