import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================
# Parameter
# =============================
XN = 10
CHI_INIT = 1.5

# Diskretisierung (etwas feiner, aber nicht uebertrieben fuer Interaktivitaet)
PHI2 = np.linspace(1e-6, 1 - 1e-6, 4001)  # Polymeranteil
PHI1 = 1 - PHI2

# =============================
# Mathematische Hilfsfunktionen (mit Chi als Argument)
# =============================

def _clip(phi):
    # sehr konservatives Clipping wegen Logarithmen
    return np.clip(np.asarray(phi), 1e-12, 1 - 1e-12)


def G(phi, chi):
    phi = _clip(phi)
    phi1 = 1 - phi
    return phi1 * np.log(phi1) + (phi / XN) * np.log(phi) + chi * phi1 * phi


def dG_dphi(phi, chi):
    phi = _clip(phi)
    return (-np.log(1 - phi) - 1) + (1 / XN) * (np.log(phi) + 1) + chi * (1 - 2 * phi)


def d2G_dphi2(phi, chi):
    phi = _clip(phi)
    return 1.0 / (1.0 - phi) + 1.0 / (XN * phi) - 2.0 * chi


# =============================
# Kernroutine: berechnet alle Groessen fuer gegebenes chi
# =============================

def compute_state(chi):
    phi2 = PHI2
    dGm_RT = G(phi2, chi)

    # Ableitungen numerisch (fuer robuste Minima/Wendepunkte)
    d1 = np.gradient(dGm_RT, phi2)
    d2 = np.gradient(d1, phi2)

    # Wendepunkte (Vorzeichenwechsel in 2. Ableitung)
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
        right_half = np.arange(global_min + 1, len(dGm_RT))
        if len(left_half) > 0 and len(right_half) > 0:
            min_left = left_half[np.argmin(dGm_RT[left_half])]
            min_right = right_half[np.argmin(dGm_RT[right_half])]
        else:
            min_left = max(global_min - 1, 0)
            min_right = min(global_min + 1, len(dGm_RT) - 1)

    phi_minima = np.array([phi2[min_left], phi2[min_right]])
    G_minima = np.array([dGm_RT[min_left], dGm_RT[min_right]])

    # Gemeinsame Tangente: Mehrstufige Gittersuche + Newton – leicht abgespeckt fuer Interaktivitaet
    def residual_at(a, b):
        Ga = G(a, chi)
        Gb = G(b, chi)
        Ha = dG_dphi(a, chi)
        Hb = dG_dphi(b, chi)
        den = max(b - a, 1e-18)
        H = (Gb - Ga) / den
        F1 = Ha - Hb
        F2 = H - Ha
        return float(F1 * F1 + F2 * F2), float(F1), float(F2)

    # Startfenster
    phi_a_lo, phi_a_hi = 1e-4, 0.25
    phi_b_lo, phi_b_hi = 0.30, 0.98
    phi_a, phi_b = None, None

    for npts in (800, 1600, 3200):
        Aa = np.linspace(phi_a_lo, phi_a_hi, npts)
        Bb = np.linspace(phi_b_lo, phi_b_hi, npts)
        AAm, BBm = np.meshgrid(Aa, Bb, indexing='ij')
        dGa = dG_dphi(AAm, chi)
        dGb = dG_dphi(BBm, chi)
        Ga = G(AAm, chi)
        Gb = G(BBm, chi)
        den = BBm - AAm
        den = np.where(np.abs(den) < 1e-8, np.nan, den)
        R1 = dGa - dGb
        R2 = (Gb - Ga) / den - dGa
        Resid = R1 ** 2 + R2 ** 2
        i_min, j_min = np.unravel_index(np.nanargmin(Resid), Resid.shape)
        phi_a = float(AAm[i_min, j_min])
        phi_b = float(BBm[i_min, j_min])
        # Fenster verfeinern
        d_a = max((phi_a_hi - phi_a_lo) / (10 * npts), 5e-3)
        d_b = max((phi_b_hi - phi_b_lo) / (10 * npts), 5e-3)
        phi_a_lo = float(max(1e-4, phi_a - 10 * d_a))
        phi_a_hi = float(min(0.30, phi_a + 10 * d_a))
        phi_b_lo = float(max(0.30, phi_b - 10 * d_b))
        phi_b_hi = float(min(0.999, phi_b + 10 * d_b))

    res_init, _, _ = residual_at(phi_a, phi_b)
    if (phi_b - phi_a) < 0.02 or not np.isfinite(res_init):
        phi_a = max(0.01, float(phi_minima[0] * 0.8))
        phi_b = min(0.95, float(phi_minima[1] - 0.02))

    # Newton mit Backtracking (Limit fuer Interaktivitaet)
    res_prev, _, _ = residual_at(phi_a, phi_b)
    for _ in range(80):
        Ga = G(phi_a, chi)
        Gb = G(phi_b, chi)
        dGa = dG_dphi(phi_a, chi)
        dGb = dG_dphi(phi_b, chi)
        d2Ga = d2G_dphi2(phi_a, chi)
        d2Gb = d2G_dphi2(phi_b, chi)
        denom = phi_b - phi_a
        denom = 1e-12 if denom <= 1e-12 else denom
        H = (Gb - Ga) / denom
        F1 = dGa - dGb
        F2 = H - dGa
        if max(abs(F1), abs(F2)) < 1e-10:
            break
        dH_da = (-dGa * denom + (Gb - Ga)) / (denom ** 2)
        dH_db = (dGb * denom - (Gb - Ga)) / (denom ** 2)
        J11 = d2Ga
        J12 = -d2Gb
        J21 = dH_da - d2Ga
        J22 = dH_db
        J = np.array([[J11, J12], [J21, J22]], dtype=float)
        F = np.array([F1, F2], dtype=float)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        alpha = 1.0
        accepted = False
        while alpha > 1e-6:
            a_try = float(np.clip(phi_a + alpha * step[0], 1e-4, 0.30))
            b_try = float(np.clip(phi_b + alpha * step[1], 0.30, 0.995))
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
    m_tan = (G(phi_b, chi) - G(phi_a, chi)) / (phi_b - phi_a)
    b_tan = G(phi_a, chi) - m_tan * phi_a

    return {
        "phi2": phi2,
        "dGm_RT": dGm_RT,
        "wendepunkte_phi2": wendepunkte_phi2,
        "wendepunkte_G": wendepunkte_G,
        "phi_minima": phi_minima,
        "G_minima": G_minima,
        "phi_a": phi_a,
        "phi_b": phi_b,
        "m_tan": m_tan,
        "b_tan": b_tan,
    }


# =============================
# Plot + GUI (Tkinter)
# =============================

class FloryApp:
    def __init__(self, master):
        self.master = master
        master.title("Flory-Huggins: ΔGm/RT mit Schieberegler für χ")

        # Figure/Axes
        self.fig, self.ax = plt.subplots(figsize=(7, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Steuerleiste: χ-Label, Slider, Eingabefeld
        ctrl_frame = ttk.Frame(master)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(ctrl_frame, text="χ").pack(side=tk.LEFT, padx=6)

        self.chi_var = tk.DoubleVar(value=CHI_INIT)
        self.scale = ttk.Scale(ctrl_frame, from_=0.1, to=3.0, variable=self.chi_var, command=self._on_slider)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6, pady=6)

        self.entry_var = tk.StringVar(value=f"{CHI_INIT:.3f}")
        self.entry = ttk.Entry(ctrl_frame, width=8, textvariable=self.entry_var)
        self.entry.pack(side=tk.LEFT, padx=6)
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

        # Initial zeichnen
        self._draw(CHI_INIT)

    def _on_slider(self, _value):
        chi = float(self.chi_var.get())
        # Entry synchronisieren
        self.entry_var.set(f"{chi:.3f}")
        # sofort neu zeichnen
        self._draw(chi)

    def _on_entry(self, event=None):
        txt = self.entry_var.get().strip().replace(",", ".")
        try:
            chi = float(txt)
        except ValueError:
            # Ungueltig -> auf aktuellen Sliderwert zuruecksetzen
            self.entry_var.set(f"{float(self.chi_var.get()):.3f}")
            return
        # Begrenzen auf Sliderbereich
        chi = max(0.1, min(3.0, chi))
        # Slider aktualisieren (loest _on_slider aus, welches zeichnet)
        self.chi_var.set(chi)
        self._on_slider(None)

    def _draw(self, chi):
        self.ax.clear()
        state = compute_state(chi)
        phi2 = state["phi2"]
        dGm_RT = state["dGm_RT"]
        self.ax.plot(phi2, dGm_RT, label='ΔGm/RT', linewidth=1.0)

        # Gerade durch die beiden Minima
        phi_min = state["phi_minima"]
        G_min = state["G_minima"]
        if np.all(np.isfinite(phi_min)) and np.all(np.isfinite(G_min)) and (phi_min[1] > phi_min[0]):
            m_min = (G_min[1] - G_min[0]) / (phi_min[1] - phi_min[0])
            b_min = G_min[0] - m_min * phi_min[0]
            x_full = np.array([0.0, 1.0])
            self.ax.plot(x_full, m_min * x_full + b_min, '--', linewidth=0.8, color='red', label='Minima verbunden')
            y_line = m_min * phi2 + b_min
            self.ax.fill_between(phi2, dGm_RT, y_line, where=(dGm_RT < y_line), alpha=0.15, color='red')
            self.ax.scatter(phi_min, G_min, s=12, color='red', label='Minima')

        # Wendepunkte
        wp_phi = state["wendepunkte_phi2"]
        wp_G = state["wendepunkte_G"]
        if len(wp_phi) >= 1:
            self.ax.scatter(wp_phi, wp_G, s=12, color='blue', label='Wendepunkte')
        if len(wp_phi) >= 2:
            m_wend = (wp_G[1] - wp_G[0]) / (wp_phi[1] - wp_phi[0])
            b_wend = wp_G[0] - m_wend * wp_phi[0]
            x_full = np.array([0.0, 1.0])
            self.ax.plot(x_full, m_wend * x_full + b_wend, '--', linewidth=0.8, color='blue')

        # Gemeinsame Tangente
        phi_a = state["phi_a"]
        phi_b = state["phi_b"]
        m_tan = state["m_tan"]
        b_tan = state["b_tan"]
        if np.isfinite(m_tan) and (phi_a < phi_b):
            x_full = np.array([0.0, 1.0])
            self.ax.plot(x_full, m_tan * x_full + b_tan, '--', linewidth=0.8, color='green', label='Gemeinsame Tangente')
            self.ax.scatter([phi_a, phi_b], [G(phi_a, chi), G(phi_b, chi)], s=12, color='green', label='Berührpunkte')

        self.ax.set_xlabel(r"$\varphi_2$ (Polymeranteil)")
        self.ax.set_ylabel(r"$\Delta G^{m} / RT$")
        self.ax.grid(True)
        self.ax.legend(fontsize='small')
        # keine festen y-Limits -> automatische Skalierung fuer verschiedene χ
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    app = FloryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()