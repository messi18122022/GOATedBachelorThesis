#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate Fig. 2.12 style plots (Flory–Huggins free energy for polymer solutions)
inkl. Markierungen fuer lokale Minima und Wendepunkte (Inflection Points).

Generiert pro XN zwei PDFs (XN ∈ {10, 100, 1000}):
  1) Abb2_12_recreation_full_XN{XN}.pdf   : voller Bereich 0<phi_2<1
  2) Abb2_12_recreation_zoom_XN{XN}.pdf   : Zoom x:[ZOOM_X_MIN, ZOOM_X_MAX], y:[ZOOM_Y_MIN, ZOOM_Y_MAX]

Modell:
  ΔG_m/(RT) = φ1 ln φ1 + (φ2/X_N) ln φ2 + χ φ1 φ2, mit φ1 = 1 - φ2
Parameter:
  X_N = 10 (wie in Abb. 2.12)
  chi in {0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}
"""
from __future__ import annotations
import os
from typing import Iterable, Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ---- Matplotlib LaTeX/Fonts ----

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 11,
    'axes.formatter.use_mathtext': True,
})

# cm → inch Helper
_def_cm = lambda x: x/2.54

# ------------------------------- Konfiguration -------------------------------
# Passe nur diese Werte an, um das Zoom-Fenster einzustellen.
ZOOM_X_MIN: float = 0.0
ZOOM_X_MAX: float = 0.06
ZOOM_Y_MIN: float = -0.02
ZOOM_Y_MAX: float = 0.002

# Numerischer Puffer, damit Kurven nicht abgeschnitten werden
PAD_REL: float = 0.15   # relativer Puffer in % der Fensterbreite
PAD_ABS: float = 1e-3   # absoluter Minimalpuffer

# χ-Bereich (viele Werte zwischen 1.00 und 1.50)
CHI_MIN: float = -0.5
CHI_MAX: float = 1.50
CHI_N: int = 20

# Ausgabeverzeichnis
OUTPUT_DIR: str = 'dGm'

# ------------------------- Thermodynamik-Funktionen -------------------------
def delta_g_reduced(phi2: np.ndarray, chi: float, XN: float = 10.0) -> np.ndarray:
    """Reduzierte freie Enthalpie ΔG_m/(RT) fuer gegebene phi2 und chi, XN."""
    phi1 = 1.0 - phi2
    return phi1*np.log(phi1) + (phi2/XN)*np.log(phi2) + chi*phi1*phi2

def gprime(phi2: np.ndarray, chi: float, XN: float = 10.0) -> np.ndarray:
    """Erste Ableitung d/dφ2 von ΔG_m/(RT)."""
    # g' = -ln(1-φ) + (1/XN) ln φ + (1/XN - 1) + χ(1 - 2φ)
    a = 1.0 / XN
    return -np.log(1.0 - phi2) + a*np.log(phi2) + (a - 1.0) + chi*(1.0 - 2.0*phi2)

def gsecond(phi2: np.ndarray, chi: float, XN: float = 10.0) -> np.ndarray:
    """Zweite Ableitung d²/dφ2² von ΔG_m/(RT)."""
    # g'' = 1/(1-φ) + 1/(XN*φ) - 2χ
    return 1.0/(1.0 - phi2) + 1.0/(XN*phi2) - 2.0*chi


# ------------------------------ Root-Finding --------------------------------
def bisection(f: Callable[[float], float], a: float, b: float, maxiter: int = 100, tol: float = 1e-12) -> float:
    """Einfache Bisektion fuer stetige f mit Vorzeichenwechsel auf [a,b]."""
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("Function not finite at interval endpoints")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa*fb > 0:
        raise ValueError("No sign change on interval")
    left, right = a, b
    for _ in range(maxiter):
        mid = 0.5*(left + right)
        fm = f(mid)
        if not np.isfinite(fm):
            # nudge mid slightly
            mid = np.nextafter(mid, right)
            fm = f(mid)
        if abs(fm) < tol or (right - left) < tol:
            return mid
        if fa*fm <= 0:
            right = mid
            fb = fm
        else:
            left = mid
            fa = fm
    return 0.5*(left + right)

def find_roots_on_interval(func: Callable[[np.ndarray], np.ndarray], low: float, high: float, num: int = 5000) -> List[Tuple[float, float]]:
    """Finde Nullstellen von func auf (low, high) via Gitter+Bisektion.
    Gibt Liste von (x_root, f''-dummy) zur Kompatibilitaet zurueck (zweiter Wert wird nicht benutzt).
    """
    xs = np.linspace(low, high, num)
    ys = func(xs)
    roots: List[Tuple[float, float]] = []
    for i in range(len(xs)-1):
        y1, y2 = ys[i], ys[i+1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 == 0.0:
            roots.append((xs[i], 0.0))
        elif y1*y2 < 0:
            try:
                r = bisection(lambda t: float(func(np.array([t]))[0]), xs[i], xs[i+1])
                roots.append((r, 0.0))
            except Exception:
                pass
    return roots

# ------------------------------- Plot-Helfer --------------------------------
def scatter_points(ax, xs: List[float], ys: List[float], marker: str):
    if len(xs) == 0:
        return
    ax.scatter(xs, ys, marker=marker, s=10, linewidths=0.5,
               edgecolors='k', facecolors='k', zorder=10)

def ensure_dir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# ------------------------------- Plot-Funktionen -----------------------------
def make_full_plot(chis: Iterable[float], XN: float = 10.0, outpath: str = "Abb2_12_recreation_full.pdf") -> None:
    """Plot ΔG_m/(RT) vs φ2 auf vollem Bereich, inkl. Minima und Wendepunkte."""
    eps = 1e-8
    phi2 = np.linspace(eps, 1-eps, 4000)
    fig, ax = plt.subplots(figsize=(_def_cm(7.5), _def_cm(7.5)))
    # Farben fuer die aeussersten Kurven (chi=-0.5 und chi=1.5)
    _cycle_cols = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])
    _col_chi_min = _cycle_cols[0]
    _col_chi_max = _cycle_cols[1]

    mins_x_total, mins_y_total = [], []
    xs_inf_total, ys_inf_total = [], []

    for chi in chis:
        g = delta_g_reduced(phi2, chi, XN=XN)
        # Farbzuordnung: aeusserste Kurven farbig, rest grau
        if abs(chi - (-0.5)) < 1e-12:
            line_color = _col_chi_min
        elif abs(chi - 1.5) < 1e-12:
            line_color = _col_chi_max
        else:
            line_color = 'gray'
        ax.plot(phi2, g, linewidth=0.4, color=line_color)

        # Extrema: g'(φ)=0
        f1 = lambda x: gprime(np.array([x]), chi, XN)[0]
        roots_ext = find_roots_on_interval(lambda arr: gprime(arr, chi, XN), eps, 1-eps)
        xs_ext = [r[0] for r in roots_ext]
        ys_ext = [delta_g_reduced(np.array([x]), chi, XN)[0] for x in xs_ext]
        # Klassifikation via g''
        for x, y in zip(xs_ext, ys_ext):
            sec = gsecond(np.array([x]), chi, XN)[0]
            if sec > 0:
                mins_x_total.append(x)
                mins_y_total.append(y)
            # sec==0 selten -> ignorieren

        # Wendepunkte: g''(φ)=0
        roots_inf = find_roots_on_interval(lambda arr: gsecond(arr, chi, XN), eps, 1-eps)
        xs_inf = [r[0] for r in roots_inf]
        ys_inf = [delta_g_reduced(np.array([x]), chi, XN)[0] for x in xs_inf]

        xs_inf_total.extend(xs_inf)
        ys_inf_total.extend(ys_inf)

    scatter_points(ax, mins_x_total, mins_y_total, marker='o')
    scatter_points(ax, xs_inf_total, ys_inf_total, marker='^')

    # In-Plot Labels fuer die Grenzkurven chi=-0.5 und chi=1.5
    try:
        # Positionen fuer die Annotationen (links/rechts im Plot)
        x_left = 0.15
        x_right = 0.85
        # Sicherstellen, dass x innerhalb (0,1) liegt
        x_left = max(1e-6, min(1-1e-6, x_left))
        x_right = max(1e-6, min(1-1e-6, x_right))

        for ch, xpos, halign in [(-0.5, x_left, 'left'), (1.5, x_right, 'right')]:
            y = float(delta_g_reduced(np.array([xpos]), ch, XN)[0])
            col = _col_chi_min if ch < 0 else _col_chi_max
            ax.text(xpos, y,
                    rf"$\chi={ch:.1f}$",
                    ha=halign, va='center', fontsize=9, color=col,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1.0))
    except Exception:
        pass

    ax.set_xlabel(r"$\varphi_2$")
    ax.set_ylabel(r"$\Delta G_m / (RT)$")
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0],[0], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='k', label='Minima'),
        Line2D([0],[0], marker='^', linestyle='None', markeredgecolor='k', markerfacecolor='k', label='Wendepunkte'),
    ]
    leg = ax.legend(handles=marker_handles, loc='lower left', fontsize=8)
    leg.set_zorder(1000)
    plt.tight_layout(pad=0.3)
    ensure_dir_for(outpath)
    plt.savefig(outpath)
    plt.close()


def make_zoom_plot(
    chis: Iterable[float],
    XN: float = 10.0,
    outpath: str = "Abb2_12_recreation_zoom.pdf",
    x_limits: Tuple[float, float] | None = None,
    y_limits: Tuple[float, float] | None = None,
) -> None:
    """Plot ΔG_m/(RT) vs φ2 im Zoom, inkl. Minima/Wendepunkte (falls im Fenster).
    Der numerische Bereich wird leicht ueber x_max hinaus berechnet, damit Kurven nicht abgeschnitten wirken.
    """
    eps = 1e-12
    if x_limits is None:
        x_min, x_max = ZOOM_X_MIN, ZOOM_X_MAX
    else:
        x_min, x_max = x_limits
    if y_limits is None:
        y_min, y_max = ZOOM_Y_MIN, ZOOM_Y_MAX
    else:
        y_min, y_max = y_limits

    # Numerischer Bereich mit Puffer, damit nix abgeschnitten wird
    span = max(1e-9, x_max - x_min)
    pad = max(PAD_ABS, PAD_REL * span)
    phi2_min = max(eps, x_min - pad)
    phi2_max = min(1.0 - eps, x_max + pad)
    if phi2_max <= phi2_min:
        phi2_max = min(1.0 - eps, phi2_min + 10*PAD_ABS)
    phi2 = np.linspace(phi2_min, phi2_max, 6000)

    fig, ax = plt.subplots(figsize=(_def_cm(7.5), _def_cm(7.5)))

    mins_x_total, mins_y_total = [], []
    xs_inf_total, ys_inf_total = [], []

    for chi in chis:
        g = delta_g_reduced(phi2, chi, XN=XN)
        ax.plot(phi2, g, linewidth=0.4, color='gray')

        # Extrema und Wendepunkte im ganzen Definitionsbereich bestimmen,
        # Marker nur anzeigen, wenn sie im sichtbaren Fenster liegen.
        roots_ext = find_roots_on_interval(lambda arr: gprime(arr, chi, XN), eps, 1-1e-8)
        for x,_ in roots_ext:
            if x_min <= x <= x_max:
                y = delta_g_reduced(np.array([x]), chi, XN)[0]
                sec = gsecond(np.array([x]), chi, XN)[0]
                if sec > 0:
                    mins_x_total.append(x)
                    mins_y_total.append(y)

        roots_inf = find_roots_on_interval(lambda arr: gsecond(arr, chi, XN), eps, 1-1e-8)
        for x,_ in roots_inf:
            if x_min <= x <= x_max:
                xs_inf_total.append(x)
                ys_inf_total.append(delta_g_reduced(np.array([x]), chi, XN)[0])

    scatter_points(ax, mins_x_total, mins_y_total, marker='o')
    scatter_points(ax, xs_inf_total, ys_inf_total, marker='^')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\varphi_2$")
    # ax.set_ylabel(r"$\Delta G_m / (RT)$")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='k', label='Minima'),
        Line2D([0],[0], marker='^', linestyle='None', markeredgecolor='k', markerfacecolor='k', label='Wendepunkte'),
    ]
    leg = ax.legend(handles=handles, loc='lower left', fontsize=8)
    leg.set_zorder(1000)
    plt.tight_layout(pad=0.3)
    ensure_dir_for(outpath)
    plt.savefig(outpath)
    plt.close()

def compute_markers_for_XN(chis: Iterable[float], XN: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Berechne Marker fuer (φ2, χ): Minima (g'=0, g''>0) und Wendepunkte (g''=0) fuer ein XN.
    Rueckgabe: (mins_phi, mins_chi, infl_phi, infl_chi).
    """
    eps = 1e-8
    mins_phi, mins_chi = [], []
    infl_phi, infl_chi = [], []

    for chi in chis:
        # Extrema (g' = 0)
        roots_ext = find_roots_on_interval(lambda arr: gprime(arr, chi, XN), eps, 1 - eps)
        for r,_ in roots_ext:
            sec = gsecond(np.array([r]), chi, XN)[0]
            if sec > 0:  # Minimum
                mins_phi.append(r)
                mins_chi.append(chi)
        # Wendepunkte (g'' = 0)
        roots_inf = find_roots_on_interval(lambda arr: gsecond(arr, chi, XN), eps, 1 - eps)
        for r,_ in roots_inf:
            infl_phi.append(r)
            infl_chi.append(chi)

    return (np.array(mins_phi, dtype=float),
            np.array(mins_chi, dtype=float),
            np.array(infl_phi, dtype=float),
            np.array(infl_chi, dtype=float))


def make_marker_summary_plot(chis: Iterable[float], XNs: Iterable[float], outpath: str) -> None:
    """Gemeinsamer Plot: χ (y) gegen φ₂ (x) fuer alle Minima (Punkte) und Wendepunkte (Dreiecke),
    farblich nach X_N.
    """
    fig, ax = plt.subplots(figsize=(_def_cm(16.0), _def_cm(8.0)))

    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])

    # Helfer: segmentiertes Linienplotten basierend auf maximalem Lueckenabstand
    def _plot_segmented(xvals: np.ndarray, yvals: np.ndarray, *, gap: float, linestyle: str, color: str, label: str):
        if xvals.size == 0:
            return
        # nach x sortieren
        order = np.argsort(xvals)
        x = np.asarray(xvals)[order]
        y = np.asarray(yvals)[order]
        start = 0
        used_label = False
        for i in range(1, x.size):
            if (x[i] - x[i-1]) > gap:
                if i - start >= 2:
                    ax.plot(x[start:i], y[start:i], linestyle=linestyle, linewidth=1.0, color=color,
                            label=label if not used_label else None)
                    used_label = True
                start = i
        # Tail-Segment
        if x.size - start >= 2:
            ax.plot(x[start:], y[start:], linestyle=linestyle, linewidth=1.0, color=color,
                    label=label if not used_label else None)

    for idx, XN in enumerate(XNs):
        color = cycle_colors[idx % len(cycle_colors)]
        mins_phi, mins_chi, infl_phi, infl_chi = compute_markers_for_XN(chis, XN)
        # Binodale (Minima): gestrichelt, Luecken > 0.5 NICHT verbinden
        _plot_segmented(np.array(mins_phi), np.array(mins_chi), gap=0.5, linestyle='--', color=color,
                        label=fr"Binodale, $X_n={int(XN)}$")
        # Spinodale (Wendepunkte): durchgezogen, Luecken > 0.5 NICHT verbinden
        _plot_segmented(np.array(infl_phi), np.array(infl_chi), gap=0.5, linestyle='-', color=color,
                        label=fr"Spinodale, $X_n={int(XN)}$")

    ax.set_xlabel(r"$\varphi_2$")
    ax.set_ylabel(r"$\chi$")
    ax.invert_yaxis()
    leg = ax.legend(loc='upper left', fontsize=8, ncol=1)
    leg.set_zorder(1000)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color='0.8')
    plt.tight_layout(pad=0.3)
    ensure_dir_for(outpath)
    plt.savefig(outpath)
    plt.close()

# ----------------------------------- Main -----------------------------------
def main():
    # Chi-Liste aus Konfiguration
    chis = [float(f"{c:.2f}") for c in np.linspace(CHI_MIN, CHI_MAX, CHI_N)]
    outdir = OUTPUT_DIR

    # Plot-Set 1: XN = 10
    XN = 10.0
    make_full_plot(chis, XN=XN, outpath=f"{outdir}/Abb2_12_recreation_full_XN{int(XN)}.pdf")
    make_zoom_plot(
        chis,
        XN=XN,
        outpath=f"{outdir}/Abb2_12_recreation_zoom_XN{int(XN)}.pdf",
        x_limits=(ZOOM_X_MIN, ZOOM_X_MAX),
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX),
    )

    # Plot-Set 2: XN = 100
    XN = 100.0
    make_full_plot(chis, XN=XN, outpath=f"{outdir}/Abb2_12_recreation_full_XN{int(XN)}.pdf")
    make_zoom_plot(
        chis,
        XN=XN,
        outpath=f"{outdir}/Abb2_12_recreation_zoom_XN{int(XN)}.pdf",
        x_limits=(ZOOM_X_MIN, ZOOM_X_MAX),
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX),
    )

    # Plot-Set 3: XN = 1000
    XN = 1000.0
    make_full_plot(chis, XN=XN, outpath=f"{outdir}/Abb2_12_recreation_full_XN{int(XN)}.pdf")
    make_zoom_plot(
        chis,
        XN=XN,
        outpath=f"{outdir}/Abb2_12_recreation_zoom_XN{int(XN)}.pdf",
        x_limits=(ZOOM_X_MIN, ZOOM_X_MAX),
        y_limits=(ZOOM_Y_MIN, ZOOM_Y_MAX),
    )

    # Gemeinsamer Marker-Plot fuer alle XN
    make_marker_summary_plot(
        chis=chis,
        XNs=[10.0, 100.0, 1000.0],
        outpath=f"{outdir}/Markers_Minima_Wendepunkte_all_XN.pdf"
    )

    print("Saved all plot sets for XN = 10, 100, 1000.")

if __name__ == "__main__":
    main()