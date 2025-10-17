

import os
import numpy as np
import matplotlib.pyplot as plt

# Fortschrittsbalken (tqdm), mit Fallback falls nicht installiert
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# =============================
# Parameter
# =============================
XN = 10
chi = 1.5

# =============================
# Diskretisierung (fein)
# =============================
phi2 = np.linspace(1e-6, 1-1e-6, 20001)  # Polymeranteil
phi1 = 1 - phi2

# =============================
# Freie Mischungsenthalpie ΔGm/RT
# =============================
dGm_RT = phi1 * np.log(phi1) + (phi2 / XN) * np.log(phi2) + chi * phi1 * phi2

# =============================
# Funktionen G, G', G'' (stabilisiert via Clipping)
# =============================
def _clip(phi):
    # sehr konservatives Clipping wegen Logarithmen
    return np.clip(np.asarray(phi), 1e-12, 1-1e-12)

def G(phi):
    phi = _clip(phi)
    phi1 = 1 - phi
    return phi1 * np.log(phi1) + (phi / XN) * np.log(phi) + chi * phi1 * phi

def dG_dphi(phi):
    phi = _clip(phi)
    # d/dφ [(1-φ)ln(1-φ)] = -ln(1-φ) - 1
    # d/dφ [(φ/XN)ln φ]   = (1/XN)(ln φ + 1)
    # d/dφ [χ(1-φ)φ]      = χ(1 - 2φ)
    return (-np.log(1 - phi) - 1) + (1 / XN) * (np.log(phi) + 1) + chi * (1 - 2 * phi)

def d2G_dphi2(phi):
    phi = _clip(phi)
    # zweite Ableitung
    return 1.0/(1.0 - phi) + 1.0/(XN*phi) - 2.0*chi


# =============================
# Wendepunkte (Spinodale/Inflektionspunkte) – EXAKT aus d²G/dφ² = 0
# Für Flory–Huggins: 1/(1-φ) + 1/(XN φ) - 2χ = 0
#  => 2χ XN φ² + [(XN - 1) - 2χ XN] φ + 1 = 0
# =============================
a = 2.0 * chi * XN
b = (XN - 1.0) - 2.0 * chi * XN
c = 1.0

wendepunkte_phi2 = np.array([])
wendepunkte_G = np.array([])

if abs(a) > 0 and (b*b - 4*a*c) >= 0:
    disc = np.sqrt(b*b - 4*a*c)
    r1 = (-b - disc) / (2*a)
    r2 = (-b + disc) / (2*a)
    roots = [r1, r2]
    roots = [float(r) for r in roots if (r > 1e-8) and (r < 1.0 - 1e-8)]
    if len(roots) > 0:
        wendepunkte_phi2 = np.array(sorted(roots))
        wendepunkte_G = G(wendepunkte_phi2)

# =============================
# Gemeinsame Tangente: Mehrstufige Gittersuche + Newton-Verfeinerung
# (ohne jegliche Minima-Heuristiken)
# =============================
phi_a_lo, phi_a_hi = 1e-4, 0.25
phi_b_lo, phi_b_hi = 0.30, 0.98
phi_a, phi_b = None, None

for npts in tqdm((2000, 4000, 8000), desc="Gitter-Verfeinerung"):
    Aa = np.linspace(phi_a_lo, phi_a_hi, npts)
    Bb = np.linspace(phi_b_lo, phi_b_hi, npts)
    AAm, BBm = np.meshgrid(Aa, Bb, indexing='ij')

    dGa = dG_dphi(AAm)
    dGb = dG_dphi(BBm)
    Ga = G(AAm)
    Gb = G(BBm)

    den = BBm - AAm
    den = np.where(np.abs(den) < 1e-8, np.nan, den)

    R1 = dGa - dGb
    R2 = (Gb - Ga) / den - dGa
    Resid = R1**2 + R2**2

    i_min, j_min = np.unravel_index(np.nanargmin(Resid), Resid.shape)
    phi_a = float(AAm[i_min, j_min])
    phi_b = float(BBm[i_min, j_min])

    # Fenster verfeinern
    d_a = max((phi_a_hi - phi_a_lo) / (10*npts), 5e-3)
    d_b = max((phi_b_hi - phi_b_lo) / (10*npts), 5e-3)
    phi_a_lo = float(max(1e-4, phi_a - 10*d_a))
    phi_a_hi = float(min(0.30,  phi_a + 10*d_a))
    phi_b_lo = float(max(0.30,  phi_b - 10*d_b))
    phi_b_hi = float(min(0.999, phi_b + 10*d_b))

# vorlaeufige Tangente aus Gittersuche
m_tan = (G(phi_b) - G(phi_a)) / (phi_b - phi_a)
b_tan = G(phi_a) - m_tan * phi_a

# Residuum-Funktion
def residual_at(a, b):
    Ga = G(a); Gb = G(b)
    Ha = dG_dphi(a); Hb = dG_dphi(b)
    den = max(b - a, 1e-18)
    H = (Gb - Ga)/den
    F1 = Ha - Hb
    F2 = H - Ha
    return float(F1*F1 + F2*F2), float(F1), float(F2)

# Seeds robust halten, ohne Minima-Information
res_init, _, _ = residual_at(phi_a, phi_b)
if (not np.isfinite(m_tan)) or (phi_b - phi_a) < 0.02 or not np.isfinite(res_init):
    # neutrale, konservative Wahl
    phi_a = 0.05
    phi_b = 0.85

# Newton mit Backtracking
res_prev, _, _ = residual_at(phi_a, phi_b)
for _ in tqdm(range(200), desc="Newton-Verfeinerung"):
    Ga = G(phi_a); Gb = G(phi_b)
    dGa = dG_dphi(phi_a); dGb = dG_dphi(phi_b)
    d2Ga = d2G_dphi2(phi_a); d2Gb = d2G_dphi2(phi_b)

    denom = phi_b - phi_a
    denom = 1e-12 if denom <= 1e-12 else denom

    H = (Gb - Ga)/denom
    F1 = dGa - dGb
    F2 = H - dGa

    if max(abs(F1), abs(F2)) < 1e-12:
        break

    # Ableitungen von H
    dH_da = (-dGa*denom + (Gb - Ga)) / (denom**2)
    dH_db = ( dGb*denom - (Gb - Ga)) / (denom**2)

    # Jacobian
    J11 = d2Ga
    J12 = -d2Gb
    J21 = dH_da - d2Ga
    J22 = dH_db
    J = np.array([[J11, J12],[J21, J22]], dtype=float)
    F = np.array([F1, F2], dtype=float)

    try:
        step = np.linalg.solve(J, -F)
    except np.linalg.LinAlgError:
        break

    # adaptives Backtracking
    alpha = 1.0
    accepted = False
    while alpha > 1e-6:
        a_try = float(np.clip(phi_a + alpha*step[0], 1e-4, 0.30))
        b_try = float(np.clip(phi_b + alpha*step[1], 0.30, 0.995))
        if b_try - a_try < 1e-4:
            b_try = min(0.995, a_try + 1e-4)
        res_try, _, _ = residual_at(a_try, b_try)
        if res_try <= res_prev or not np.isfinite(res_prev):
            phi_a, phi_b = a_try, b_try
            res_prev = res_try
            accepted = True
            break
        alpha *= 0.5
    if not accepted:
        break

# finale Tangente
m_tan = (G(phi_b) - G(phi_a)) / (phi_b - phi_a)
b_tan = G(phi_a) - m_tan * phi_a

# =============================
# PLOT: ein einzelnes PDF (Groessen in cm): nur ΔGm/RT + gemeinsame Tangente
# =============================
os.makedirs('flory', exist_ok=True)
cm_to_inch = 1/2.54

# MAIN (16 cm × 10 cm)
fig_main, ax_main = plt.subplots(figsize=(16*cm_to_inch, 10*cm_to_inch))
ax_main.plot(phi2, dGm_RT, color='0.4', linewidth=0.6, label='ΔGm/RT')

# optionale Anzeige der Spinodale (Wendepunkte), KEINE Minima, KEINE Zooms, KEINE roten Flaechen
if len(wendepunkte_phi2) >= 1:
    ax_main.scatter(wendepunkte_phi2, wendepunkte_G, marker='^', c='k', s=14, label='Wendepunkte')

# gemeinsame Tangente und Beruehrpunkte
if np.isfinite(m_tan) and (phi_a < phi_b):
    ax_main.scatter([phi_a, phi_b], [G(phi_a), G(phi_b)], marker='o', c='k', s=16, label='Berührungspunkte')

ax_main.set_xlabel(r'$\varphi_2$ (Polymeranteil)')
ax_main.set_ylabel(r'$\Delta G^{m} / RT$')
ax_main.legend(fontsize='small')
ax_main.grid(True)

path_main = os.path.join('flory', 'flory_huggins_tangente.pdf')
fig_main.savefig(path_main, format='pdf', bbox_inches='tight')

print("Exportiert:", path_main)
# plt.show()