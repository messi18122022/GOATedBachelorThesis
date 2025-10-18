import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# LaTeX-Rendering aktivieren
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Fortschrittsbalken (tqdm), mit Fallback falls nicht installiert
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

@njit
def _clip_jit(phi):
    return np.clip(phi, 1e-12, 1-1e-12)

@njit
def G_numba(phi, chi, XN):
    phi = _clip_jit(phi)
    phi1 = 1 - phi
    return phi1 * np.log(phi1) + (phi / XN) * np.log(phi) + chi * phi1 * phi

# =============================
# Parameter
# =============================
XN = 10
CHI_MIN, CHI_MAX, N_CHI = 0.866, 1.5, 200

# =============================
# Diskretisierung (fein)
# =============================
phi2 = np.linspace(1e-6, 1-1e-6, 2001)  # Polymeranteil
phi1 = 1 - phi2

# =============================
# Hilfsfunktionen (abhängig von χ)
# =============================

def _clip(phi):
    # konservatives Clipping wegen Logarithmen
    return np.clip(np.asarray(phi), 1e-12, 1-1e-12)


def make_functions(chi):
    def G(phi):
        phi = _clip(phi)
        phi1_local = 1 - phi
        return phi1_local * np.log(phi1_local) + (phi / XN) * np.log(phi) + chi * phi1_local * phi

    def dG_dphi(phi):
        phi = _clip(phi)
        return (-np.log(1 - phi) - 1) + (1 / XN) * (np.log(phi) + 1) + chi * (1 - 2 * phi)

    def d2G_dphi2(phi):
        phi = _clip(phi)
        return 1.0/(1.0 - phi) + 1.0/(XN*phi) - 2.0*chi

    return G, dG_dphi, d2G_dphi2


def spinodals(chi, G):
    # Exakt aus d²G/dφ² = 0: 2χ XN φ² + [(XN - 1) - 2χ XN] φ + 1 = 0
    a = 2.0 * chi * XN
    b = (XN - 1.0) - 2.0 * chi * XN
    c = 1.0
    phi_w = []
    if abs(a) > 0:
        disc = b*b - 4*a*c
        if disc >= 0:
            r1 = (-b - np.sqrt(disc)) / (2*a)
            r2 = (-b + np.sqrt(disc)) / (2*a)
            for r in (r1, r2):
                if 1e-8 < r < 1-1e-8:
                    phi_w.append(float(r))
    phi_w = np.array(sorted(phi_w)) if len(phi_w) else np.array([])
    G_w = G(phi_w) if phi_w.size else np.array([])
    return phi_w, G_w


def common_tangent(chi, G, dG_dphi, d2G_dphi2):
    # Mehrstufige Gittersuche + Newton, ohne Farben, ohne Tangente zeichnen
    phi_a_lo, phi_a_hi = 1e-4, 0.25
    phi_b_lo, phi_b_hi = 0.30, 0.98
    phi_a, phi_b = None, None

    for npts in (200, 400, 800):
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

    def residual_at(a, b):
        Ga = G(a); Gb = G(b)
        Ha = dG_dphi(a); Hb = dG_dphi(b)
        den = max(b - a, 1e-18)
        H = (Gb - Ga)/den
        F1 = Ha - Hb
        F2 = H - Ha
        return float(F1*F1 + F2*F2)

    # plausibler Seed, falls nötig
    if (phi_b - phi_a) < 0.02 or not np.isfinite((G(phi_b)-G(phi_a))/(phi_b-phi_a)):
        phi_a, phi_b = 0.05, 0.85

    # Newton/Backtracking
    res_prev = residual_at(phi_a, phi_b)
    for _ in range(200):
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
        dH_da = (-dGa*denom + (Gb - Ga)) / (denom**2)
        dH_db = ( dGb*denom - (Gb - Ga)) / (denom**2)
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
        alpha = 1.0
        accepted = False
        while alpha > 1e-6:
            a_try = float(np.clip(phi_a + alpha*step[0], 1e-4, 0.30))
            b_try = float(np.clip(phi_b + alpha*step[1], 0.30, 0.995))
            if b_try - a_try < 1e-4:
                b_try = min(0.995, a_try + 1e-4)
            res_try = residual_at(a_try, b_try)
            if res_try <= res_prev or not np.isfinite(res_prev):
                phi_a, phi_b = a_try, b_try
                res_prev = res_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    return float(phi_a), float(phi_b)


# =============================
# Plot: ein PDF mit allen Kurven; Kurven grau, Punkte schwarz
# =============================
os.makedirs('flory', exist_ok=True)
cm_to_inch = 1/2.54
fig_main, ax_main = plt.subplots(figsize=(16*cm_to_inch, 8*cm_to_inch))

chi_values = np.linspace(CHI_MIN, CHI_MAX, N_CHI)

# Sammelpunkte für neues φ2-gegen-χ Diagramm
spinodal_x, spinodal_y = [], []  # Wendepunkte
contact_x, contact_y = [], []    # Berührungspunkte

for chi in tqdm(chi_values, desc="χ-Sweep"):
    G, dG_dphi, d2G_dphi2 = make_functions(chi)
    dGm_RT = G_numba(phi2, chi, XN)  # JIT-kompilierte Version verwenden

    # Wendepunkte für dieses χ
    w_phi, w_G = spinodals(chi, G)

    # Punkte für das neue Diagramm sammeln (Wendepunkte)
    for w in np.atleast_1d(w_phi):
        spinodal_x.append(float(w))
        spinodal_y.append(float(chi))

    # Berührungspunkte (Gemeinsame Tangente), Linie wird NICHT geplottet
    a_phi, b_phi = common_tangent(chi, G, dG_dphi, d2G_dphi2)

    # Punkte für das neue Diagramm sammeln (Berührungspunkte)
    contact_x.extend([float(a_phi), float(b_phi)])
    contact_y.extend([float(chi), float(chi)])

    # Plot: Kurve
    ax_main.plot(phi2, dGm_RT, color='lightgray', linewidth=0.4)

    # Marker: Wendepunkte (Dreiecke, dimgray)
    if w_phi.size:
        ax_main.scatter(w_phi, w_G, marker='^', c='dimgray', s=10, zorder=5)

    # Marker: Berührungspunkte (Kreise, grau)
    ax_main.scatter([a_phi, b_phi], [G(a_phi), G(b_phi)], marker='o', c='gray', s=10, zorder=6)

ax_main.set_xlabel(r'$\varphi_2$ (Polymeranteil)')
ax_main.set_ylabel(r'$\Delta G^{m} / RT$')
ax_main.grid(True)

# =============================
# Neuer Punkt: Mittelwert der Berührungspunkte bei χ = 0.866
# =============================
target_chi = 0.866
middle_phi = 0.241

# =============================
# Zweiter Plot: φ2 auf x-Achse, χ auf y-Achse; Dreiecke = Wendepunkte, Kreise = Berührungspunkte
# =============================
fig_pts, ax_pts = plt.subplots(figsize=(12*cm_to_inch, 10*cm_to_inch))


#
# Wenn Mittelwertspunkt existiert, direkt in die Listen einfügen (vor dem Sortieren!)
if middle_phi is not None:
    contact_x.append(middle_phi)
    contact_y.append(target_chi)
    spinodal_x.append(middle_phi)
    spinodal_y.append(target_chi)

# Danach sortieren
spinodal_sorted = sorted(zip(spinodal_y, spinodal_x))
contact_sorted = sorted(zip(contact_y, contact_x))

# Mittelwertpunkt explizit als Eintrag mit zwei φ-Werten einfügen
if middle_phi is not None:
    contact_sorted.append((target_chi, middle_phi - 1e-10))  # linker φ2
    contact_sorted.append((target_chi, middle_phi + 1e-10))  # rechter φ2
    contact_sorted = sorted(contact_sorted)

    spinodal_sorted.append((target_chi, middle_phi - 1e-10))
    spinodal_sorted.append((target_chi, middle_phi + 1e-10))
    spinodal_sorted = sorted(spinodal_sorted)

# Entpacken
spinodal_y_sorted, spinodal_x_sorted = zip(*spinodal_sorted)
contact_y_sorted, contact_x_sorted = zip(*contact_sorted)


# Neu: Sortiere nach χ (y), dann trenne in linke/rechte Äste nach φ (x)
from collections import defaultdict

def getrennte_äste(punkte):
    punkte_pro_chi = defaultdict(list)
    for chi_val, phi_val in zip(*punkte):
        punkte_pro_chi[chi_val].append(phi_val)

    chi_liste, phi_links, phi_rechts = [], [], []
    for chi_val in sorted(punkte_pro_chi.keys()):
        phi_vals = sorted(punkte_pro_chi[chi_val])
        if len(phi_vals) == 2:
            phi_l, phi_r = phi_vals
        elif len(phi_vals) == 1:
            # Sonderfall: Mittelwertpunkt → ignoriere ihn für die Linienverbindung
            continue
        else:
            continue
        chi_liste.append(chi_val)
        phi_links.append(phi_l)
        phi_rechts.append(phi_r)
    return phi_links, phi_rechts, chi_liste

spinodal_links, spinodal_rechts, chi_vals = getrennte_äste((spinodal_y_sorted, spinodal_x_sorted))
ax_pts.plot(spinodal_links, chi_vals, linestyle='-', color='dimgray', label='Spinodale')
ax_pts.plot(spinodal_rechts, chi_vals, linestyle='-', color='dimgray')

# Binodale (gestrichelte Linie)
contact_links, contact_rechts, chi_vals = getrennte_äste((contact_y_sorted, contact_x_sorted))
ax_pts.plot(contact_links, chi_vals, linestyle='--', color='gray', label='Binodale')
ax_pts.plot(contact_rechts, chi_vals, linestyle='--', color='gray')

# Schattierung der Regionen: instabil (dunkler), metastabil (heller)
ax_pts.fill_betweenx(chi_vals, spinodal_links, spinodal_rechts, color='dimgray', alpha=0.15, label='instabil (zwischen Spinodalen)')
ax_pts.fill_betweenx(chi_vals, contact_links, contact_rechts, color='gray', alpha=0.08, label='metastabil (zwischen Binodalen)')

if middle_phi is not None:
    ax_pts.scatter([middle_phi], [target_chi], marker='o', c='black', s=10, label=r'$\chi_c \approx 0.866$')

ax_pts.set_xlabel(r'$\varphi_2$')
ax_pts.set_ylabel(r'$\chi$')
ax_pts.set_xlim(0.0, 1.0)
ax_pts.set_xticks(np.linspace(0.0, 1.0, 6))
ax_pts.set_ylim(0.8, 1.5)
ax_pts.invert_yaxis()
ax_pts.grid(True)

ax_pts.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

path_main = os.path.join('flory', 'flory_huggins_sweep.pdf')
fig_main.savefig(path_main, format='pdf', bbox_inches='tight')
print("Exportiert:", path_main)

path_pts = os.path.join('flory', 'flory_huggins_points.pdf')
fig_pts.savefig(path_pts, format='pdf', bbox_inches='tight')
print("Exportiert:", path_pts)

# ax_pts.legend()  # Entfernt, da keine Legende mehr angezeigt werden soll
plt.show()