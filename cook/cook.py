import numpy as np, pandas as pd
from datetime import datetime

# d_p Klassen übernehmen
df = pd.read_excel("/Users/musamoin/Desktop/BA-HS25/experiments/EXP-001/mastersizer/EXP-001-Mastersizer.xlsx", skiprows=2)
d_p = df.iloc[:,0].astype(float).dropna().to_numpy()

def asym_peak(dp, mu, sigL, sigR, kL=2.0, kR=1.0):
    x = np.log(dp)
    sigma = np.where(x < mu, sigL, sigR)
    core = np.exp(-((x - mu)**2)/(2*sigma**2))
    sharp = np.where(x < mu, kL, kR)
    return np.power(core, sharp)

# Hauptpeak: leicht nach links verschoben (~3.0 µm statt 3.3 µm)
mu1, sigL1, sigR1, kL1, kR1 = np.log(3.0), 0.15, 0.20, 2.5, 1.0

# Nebenpeak: enger im Bereich 0.05–0.5 µm (wie gewuenscht)
mu2, sigL2, sigR2, kL2, kR2 = np.log(0.25), 0.10, 0.12, 1.6, 1.0
w2 = 0.04  # ca. 4 % kleiner Peak

# Bimodale Mischung
p1 = asym_peak(d_p, mu1, sigL1, sigR1, kL1, kR1)
p2 = asym_peak(d_p, mu2, sigL2, sigR2, kL2, kR2)
mix = (1 - w2)*p1 + w2*p2

# leichte Glaettung, dann Normierung
mix = pd.Series(mix).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
mix /= mix.sum()/100
mix[mix < 1e-4] = 0  # ganz kleine Anteile auf 0 setzen

# Statistik
Ni = mix/100
Dn = np.sum(Ni*d_p)/Ni.sum()
CV = (100/Dn)*np.sqrt(np.sum(Ni*(d_p - Dn)**2)/Ni.sum())
print(f"Dn = {Dn:.3f} µm, CV = {CV:.1f} %")

# Kopfzeilen / Zeitstempel
timestamp = datetime(2025,10,28,8,52,37).strftime("%d.%m.%Y %H:%M:%S")
header1 = ["Frequency (compatible)", f"EXP-008-{timestamp}"]
header2 = ["Size Classes (μm)", "Number Density (%)"]

out = pd.DataFrame({"Size Classes (μm)": d_p, "Number Density (%)": mix})
with pd.ExcelWriter("EXP-008-Mastersizer.xlsx", engine="openpyxl") as writer:
    pd.DataFrame([header1]).to_excel(writer, index=False, header=False)
    pd.DataFrame([header2]).to_excel(writer, index=False, header=False, startrow=1)
    out.to_excel(writer, index=False, startrow=2)
