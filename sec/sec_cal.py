"""
SEC (GPC) Kalibration
Musa: Trage einfach die Elutionszeiten (oder -volumina) fuer jeden Standard unten in
`ELUTION` ein – Reihenfolge ist jeweils Peak1 -> Peak4 wie auf dem Kit-Blatt.

Modell: log10(M) = a - b * Ve

Wenn du Zeiten (min) statt Volumina (mL) eintraegst, setze `flow_rate_mL_min`
entsprechend, dann wird Ve = t * flow_rate umgerechnet.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------
# Mp [Da] pro Peak aus dem ReadyCal-Kit (Polystyrol)
# Quelle: Foto im Laborordner – verwendet werden die Mp-Werte
# -----------------------------
MP: Dict[str, List[float]] = {
    "blue":  [7520000, 579000, 67000, 8400],
    "green": [2930000, 277000, 34800, 3420],
    "red":   [1210000, 127000, 17800, 1620],
}

# Optionale Mw und Mn pro Peak (falls vorhanden). Wenn nicht bekannt, bitte Werte eintragen.
MW: Dict[str, List[float]] = {
    "blue":  [6590000, 564000, 64000, 8100],
    "green": [2740000, 271000, 34000, 3470],
    "red":   [1170000, 120000, 17500, 1560],
}
MN: Dict[str, List[float]] = {
    "blue":  [5330000, 545000, 60000, 7800],
    "green": [2390000, 265000, 32700, 3280],
    "red":   [1070000, 115000, 17000, 1500],
}


def _mass_dict(kind: str) -> Dict[str, List[float]]:
    kind_lower = kind.lower()
    if kind_lower in ("mp", "m_p"):
        return MP
    if kind_lower in ("mw", "m_w"):
        return MW
    if kind_lower in ("mn", "m_n"):
        return MN
    raise ValueError("kind muss 'Mp', 'Mw' oder 'Mn' sein")


# -----------------------------
# HIER deine Elutionszeiten/-volumina eintragen (vier Peaks je Standard)
# Beispiel (in Minuten):
# "red":   [15.23, 17.80, 20.45, 23.10]
# "green": [14.10, 16.60, 19.35, 22.05]
# "blue":  [13.20, 15.55, 18.10, 21.00]
# -----------------------------
ELUTION: Dict[str, List[float]] = {
    "red":   [15.775, 19.862, 23.295, 25.629],
    "green": [14.221, 16.941, 21.207, 24.207],
    "blue":  [14.919, 18.312, 22.439, 25.206],
}


def _flatten_xy(elution: Dict[str, List[float]], flow_rate_mL_min: float, kind: str = "Mp") -> Tuple[np.ndarray, np.ndarray]:
    """Erzeugt (Ve, log10(M)) Vektoren aus Eingaben.

    - Wenn `flow_rate_mL_min` = 1.0 sind die Werte bereits Volumina.
    - Ansonsten werden Zeiten (min) in Volumina (mL) umgerechnet: Ve = t * F.
    """
    ve_list: List[float] = []
    logM_list: List[float] = []

    masses = _mass_dict(kind)

    for color, times_or_vols in elution.items():
        if color not in masses:
            raise ValueError(f"Unbekannte Farbgruppe: {color}")
        mps = masses[color] if masses[color] else None
        if mps is None or len(mps) != 4:
            raise ValueError(f"{color}: fuer {kind} fehlen die 4 M-Werte")
        if len(times_or_vols) != 4:
            raise ValueError(f"{color}: erwarte 4 Peaks, bekommen {len(times_or_vols)}")
        # Umrechnung Zeit -> Volumen
        ve_vals = [v * flow_rate_mL_min for v in times_or_vols]
        ve_list.extend(ve_vals)
        logM_list.extend([np.log10(m) for m in mps])

    Ve = np.asarray(ve_list, dtype=float)
    logM = np.asarray(logM_list, dtype=float)
    return Ve, logM


def calibrate(elution: Dict[str, List[float]], flow_rate_mL_min: float = 1.0, kind: str = "Mp") -> Tuple[float, float, float]:
    """Fittet log10(M) = a - b * Ve und gibt (a, b, r) zurueck.
    Wir verwenden lineare Regression mit np.polyfit.
    """
    Ve, logM = _flatten_xy(elution, flow_rate_mL_min, kind)

    # Fit: logM = m * Ve + c  -> m sollte negativ sein; setze a = c und b = -m
    m, c = np.polyfit(Ve, logM, 1)
    a = float(c)
    b = float(-m)

    # Korrelation (r, nicht r²)
    pred = a - b * Ve
    r = np.corrcoef(logM, pred)[0, 1]
    return a, b, r


def plot_calibration(elution: Dict[str, List[float]], flow_rate_mL_min: float = 1.0, outpath: str | Path = "sec_calibration.png", kind: str = "Mp") -> Tuple[float, float, float, Path]:
    """Erstellt Scatter + Fitlinie und speichert die Grafik.
    Rueckgabe: (a, b, r, Pfad)
    """
    Ve, logM = _flatten_xy(elution, flow_rate_mL_min, kind)
    a, b, r = calibrate(elution, flow_rate_mL_min, kind)

    # Fitlinie
    ve_grid = np.linspace(Ve.min() - 0.5, Ve.max() + 0.5, 200)
    logM_fit = a - b * ve_grid

    # Plotten
    plt.figure()
    # Punkte farbcodiert nach Standard
    start = 0
    for color in ("red", "green", "blue"):
        if color in elution and len(elution[color]) == 4:
            xs = np.asarray(elution[color], dtype=float) * flow_rate_mL_min
            mass_table = _mass_dict(kind)
            ys = np.log10(np.asarray(mass_table[color], dtype=float))
            color_map = {"red": "red", "green": "green", "blue": "blue"}
            plt.scatter(xs, ys, label=f"{color} ({kind})", c=color_map[color])
            start += 4
    plt.plot(ve_grid, logM_fit, label=f"Fit {kind}: log10(M) = a - b*Ve\n(a={a:.4f}, b={b:.4f}, r={r:.3f})")
    plt.xlabel("Elutionsvolumen Ve [mL]")
    plt.ylabel(f"log10({kind})")
    plt.legend()
    plt.tight_layout()

    outpath = Path(outpath)
    plt.savefig(outpath, dpi=300)
    plt.close()
    return a, b, r, outpath


def plot_all_calibrations(elution: Dict[str, List[float]], flow_rate_mL_min: float = 1.0, outpath: str | Path = "sec_calibration_all.png") -> Path:
    plt.figure()
    colors = {"Mp": None, "Mw": None, "Mn": None}
    results = {}
    for kind in ("Mp", "Mw", "Mn"):
        try:
            Ve, logM = _flatten_xy(elution, flow_rate_mL_min, kind)
            a, b, r = calibrate(elution, flow_rate_mL_min, kind)
            ve_grid = np.linspace(Ve.min() - 0.5, Ve.max() + 0.5, 200)
            logM_fit = a - b * ve_grid
            plt.plot(ve_grid, logM_fit, label=f"{kind}: a={a:.3f}, b={b:.3f}, r={r:.3f}")
            # Punkte einmal pro kind zeichnen
            for color in ("red", "green", "blue"):
                xs = np.asarray(elution[color], dtype=float) * flow_rate_mL_min
                ys = np.log10(np.asarray(_mass_dict(kind)[color], dtype=float))
                color_map = {"red": "red", "green": "green", "blue": "blue"}
                plt.scatter(xs, ys, c=[color_map[color]], s=15)
            results[kind] = (a, b, r)
        except Exception:
            # Falls Daten fuer kind fehlen, einfach ueberspringen
            continue
    plt.xlabel("Elutionsvolumen Ve [mL]")
    plt.ylabel("log10(M)")
    plt.legend()
    plt.tight_layout()
    outpath = Path(outpath)
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath


if __name__ == "__main__":
    # Beispiel: setze hier deine Zeiten/Volumina ein und fuehre das Skript aus.
    if any(len(v) != 4 for v in ELUTION.values()):
        print("Bitte fuelle ELUTION fuer red, green, blue mit jeweils 4 Werten aus.")
    else:
        for kind in ("Mp", "Mw", "Mn"):
            try:
                out = f"sec_calibration_{kind}.png"
                a, b, r, path = plot_calibration(ELUTION, flow_rate_mL_min=0.8, outpath=out, kind=kind)
                print(f"Kalibration {kind}: a={a:.6f}, b={b:.6f}, r={r:.4f}. Grafik: {path}")
            except Exception as e:
                print(f"{kind}: Plot nicht erstellt: {e}")
        # Zusaetzlich ein Kombi-Plot, falls alle Daten vorhanden sind
        try:
            path_all = plot_all_calibrations(ELUTION, flow_rate_mL_min=0.8, outpath="sec_calibration_all.png")
            print(f"Zusatzgrafik (alle Moeglichkeiten): {path_all}")
        except Exception as e:
            print(f"Hinweis: Kombi-Plot nicht erstellt: {e}")
