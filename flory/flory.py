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
# Diskretisierung (sehr fein)
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
# Ableitungen, Wendepunkte (Spinodale-Indikatoren) und Minima
# =============================
d1 = np.gradient(dGm_RT, phi2)
d2 = np.gradient(d1, phi2)

sign_change = np.where(np.diff(np.sign(d2)))[0]
wendepunkte_phi2 = phi2[sign_change]
wendepunkte_G = dGm_RT[sign_change]

# Lokale Minima (robust)
cand = np.where((d1[:-1] < 0) & (d1[1:] > 0))[0] + 1
min_idx = cand[d2[cand] > 0]
if len(min_idx) >= 2:
    min_idx = np.sort(min_idx)
    min_left, min_right = min_idx[0], min_idx[-1]
else:
    global_min = int(np.argmin(dGm_RT))
    left_half = np.arange(0, global_min)
    right_half = np.arange(global_min+1, len(dGm_RT))
    if len(left_half) > 0 and len(right_half) > 0:
        min_left = left_half[np.argmin(dGm_RT[left_half])]
        min_right = right_half[np.argmin(dGm_RT[right_half])]
    else:
        min_left = max(global_min-1, 0)
        min_right = min(global_min+1, len(dGm_RT)-1)

phi_minima = np.array([phi2[min_left], phi2[min_right]])
G_minima = np.array([dGm_RT[min_left], dGm_RT[min_right]])

# =============================
# Gemeinsame Tangente: Mehrstufige Gittersuche + Newton-Verfeinerung
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

# vorläufige Tangente aus Gittersuche
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

# Seeds korrigieren, falls Gittersuche kollabiert hat
res_init, _, _ = residual_at(phi_a, phi_b)
if (not np.isfinite(m_tan)) or (phi_b - phi_a) < 0.02 or not np.isfinite(res_init):
    phi_a = max(0.01, float(phi_minima[0] * 0.8))
    phi_b = min(0.95, float(phi_minima[1] - 0.02))

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
# PLOTS: drei einzelne PDFs (Grössen in cm)
# =============================
os.makedirs('flory', exist_ok=True)
cm_to_inch = 1/2.54

# MAIN (16 cm × 10 cm)
fig_main, ax_main = plt.subplots(figsize=(16*cm_to_inch, 10*cm_to_inch))
ax_main.plot(phi2, dGm_RT, label='ΔGm/RT')
ax_main.scatter(phi_minima[0], G_minima[0], color='orange',
                label=f'Minimum 1 ({phi_minima[0]:.3f}, {G_minima[0]:.3f})')
ax_main.scatter(phi_minima[1], G_minima[1], color='red',
                label=f'Minimum 2 ({phi_minima[1]:.3f}, {G_minima[1]:.3f})')
# Gerade durch die beiden Minima (voller Achsenbereich)
m_min = (G_minima[1] - G_minima[0]) / (phi_minima[1] - phi_minima[0])
b_min = G_minima[0] - m_min * phi_minima[0]
x_full = np.array([0.0, 1.0])
ax_main.plot(x_full, m_min * x_full + b_min, '--', color='red', label='Minima verbunden')
if len(wendepunkte_phi2) >= 1:
    ax_main.scatter(wendepunkte_phi2[0], wendepunkte_G[0], color='skyblue',
                    label=f'Wendepunkt 1 ({wendepunkte_phi2[0]:.3f}, {wendepunkte_G[0]:.3f})')
if len(wendepunkte_phi2) >= 2:
    ax_main.scatter(wendepunkte_phi2[1], wendepunkte_G[1], color='blue',
                    label=f'Wendepunkt 2 ({wendepunkte_phi2[1]:.3f}, {wendepunkte_G[1]:.3f})')
if np.isfinite(m_tan) and (phi_a < phi_b):
    # Tangente ueber den vollen Achsenbereich
    x_full = np.array([0.0, 1.0])
    ax_main.plot(x_full, m_tan * x_full + b_tan, '--', color='green', label='Gemeinsame Tangente')
    ax_main.scatter(phi_a, G(phi_a), color='lightgreen', label=f'Berührpunkt links ({phi_a:.3f}, {G(phi_a):.3f})')
    ax_main.scatter(phi_b, G(phi_b), color='green', label=f'Berührpunkt rechts ({phi_b:.3f}, {G(phi_b):.3f})')
ax_main.set_xlabel(r'$\varphi_2$ (Polymeranteil)')
ax_main.set_ylabel(r'$\Delta G^{m} / RT$')
ax_main.set_title(r'Flory-Huggins-Mischungsenthalpie ($X_N = 10$, $\chi = 1.5$)')
ax_main.legend(fontsize='small')
ax_main.grid(True)
ax_main.set_ylim(-0.2, 0.03)
path_main = os.path.join('flory', 'abb2_12_main.pdf')
fig_main.savefig(path_main, format='pdf', bbox_inches='tight')

# ZOOM LINKS (7.5 cm × 7.5 cm)
fig_z1, ax_z1 = plt.subplots(figsize=(7.5*cm_to_inch, 7.5*cm_to_inch))
ax_z1.plot(phi2, dGm_RT)
ax_z1.plot([phi_minima[0], phi_minima[1]], [G_minima[0], G_minima[1]], '--', color='red')
ax_z1.scatter(phi_minima[0], G_minima[0], color='orange')
ax_z1.scatter(phi_minima[1], G_minima[1], color='red')
if np.isfinite(m_tan) and (phi_a < phi_b):
    phi_tan1 = np.linspace(0.0, 0.06, 400)
    ax_z1.plot(phi_tan1, m_tan*phi_tan1 + b_tan, '--', color='green')
    ax_z1.scatter(phi_a, G(phi_a), color='lightgreen')
ax_z1.set_xlim(0.0, 0.006)
ax_z1.set_ylim(-0.0005, 0.0001)
ax_z1.set_xlabel(r'$\varphi_2$ (Polymeranteil)')
ax_z1.set_ylabel(r'$\Delta G^{m} / RT$')
ax_z1.grid(True)
path_z1 = os.path.join('flory', 'abb2_12_zoom_left.pdf')
fig_z1.savefig(path_z1, format='pdf', bbox_inches='tight')

# ZOOM RECHTS (7.5 cm × 7.5 cm, kein y-Label)
fig_z2, ax_z2 = plt.subplots(figsize=(7.5*cm_to_inch, 7.5*cm_to_inch))
ax_z2.plot(phi2, dGm_RT)
ax_z2.plot([phi_minima[0], phi_minima[1]], [G_minima[0], G_minima[1]], '--', color='red')
ax_z2.scatter(phi_minima[0], G_minima[0], color='orange')
ax_z2.scatter(phi_minima[1], G_minima[1], color='red')
if np.isfinite(m_tan) and (phi_a < phi_b):
    phi_tan2 = np.linspace(0.82, 0.90, 400)
    ax_z2.plot(phi_tan2, m_tan*phi_tan2 + b_tan, '--', color='green')
    ax_z2.scatter(phi_b, G(phi_b), color='green')
ax_z2.set_xlim(0.82, 0.90)
ax_z2.set_ylim(-0.110, -0.102)
ax_z2.set_xlabel(r'$\varphi_2$ (Polymeranteil)')
ax_z2.set_ylabel('')
ax_z2.grid(True)
path_z2 = os.path.join('flory', 'abb2_12_zoom_right.pdf')
fig_z2.savefig(path_z2, format='pdf', bbox_inches='tight')

print("Exportiert:\n  ", path_main, "\n  ", path_z1, "\n  ", path_z2)

plt.show()