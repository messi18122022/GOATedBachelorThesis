#!/usr/bin/env python3
"""
Liest alle CSV-Dateien in einem angegebenen Ordner ein, erzeugt fuer jede Datei
 einen Plot (Elutionsvolumen auf x, Signal auf y) und speichert den Plot als PDF
 im selben Ordner ab.

Annahmen / Hinweise:
- Wenn eine Spalte mit Elutionsvolumen vorhanden ist (z.B. "Elutionsvolumen",
  "ElutionVolume", "Volume", "Volumen"), wird diese direkt verwendet.
- Dieses Skript erwartet das Format: '#Point,X(Minutes),Y(Response Units)' wie in deinem Beispiel.
- CSV-Trennzeichen und Dezimaltrennzeichen werden automatisch erkannt, soweit
  moeglich.
- Peaks mit y >= 2.0 werden am Apex mit ihrem Elutionsvolumen (mL) beschriftet.
- Kalibration: log10(M) = a + b * V_E + c * V_E^2 + d * V_E^3 (Polynom 3. Grades) aus allen Blue/Green/Red-Dateien.
- Zuordnung: kleinere V_E gehoert zum groesseren M der jeweiligen Liste (siehe MP_DICT).

Aufrufbeispiele:
    python sec.py --path /pfad/zu/csvs --fluss 0.8

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
# Matplotlib: LaTeX + siunitx aktivieren
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 9,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{siunitx}'
})
import numpy as np

# ------------------------------
# Konfiguration / Defaults
# ------------------------------
DEFAULT_FLUSS = 0.8  # mL/min
PEAK_MIN_Y = 5.0  # nur Peaks >= 2 werden annotiert

PEAK_PROMINENCE = 0.5  # minimale Prominenz ueber den lokalen Flanken-Minima
PEAK_MIN_SEP = 5       # minimaler Indexabstand zwischen Peaks (nach Detektion)
SMOOTH_WINDOW = 9      # ungerade; rollender Mittelwert fuer Detektion (nur fuer Detektion, nicht Plot)

# Kalibrations-Standards: Zuordnung Farbe -> Liste von M (absteigend)
MP_DICT: dict[str, list[float]] = {
    "blue":  [7520000, 579000, 67000, 8400],
    "green": [2930000, 277000, 34800, 3420],
    "red":   [1210000, 127000, 17800, 1620],
}

def read_csv_robust(path: Path) -> pd.DataFrame:
    """Liest CSV im fixen Agilent-DAD-Format:
    Zeile 1: Kommentarzeile beginnend mit '#"DAD1 ...'
    Zeile 2: '#Point,X(Minutes),Y(Response Units)'
    Danach: Datenzeilen: idx, zeit[min], signal
    """
    df = pd.read_csv(
        path,
        comment="#",           # ignoriere alle Kommentar-/Headerzeilen
        header=None,            # keine Kopfzeile einlesen (weil mit '#' beginnt)
        names=["Point", "Time_min", "Signal"],
        sep=",",
        engine="python",
        encoding="latin-1",
        skip_blank_lines=True,
    )
    # Sicherstellen, dass 3 Spalten vorhanden sind
    if df.shape[1] != 3:
        raise RuntimeError(f"Unerwartetes Spaltenformat in {path}. Erwartet sind 3 Spalten: Point, Time_min, Signal.")

    # Konvertiere zu numerisch
    df["Point"] = pd.to_numeric(df["Point"], errors="coerce")
    df["Time_min"] = pd.to_numeric(df["Time_min"], errors="coerce")
    df["Signal"] = pd.to_numeric(df["Signal"], errors="coerce")

    # Ungueltige Zeilen entfernen
    df = df.dropna(subset=["Time_min", "Signal"]).reset_index(drop=True)
    return df


def normalize_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """Wandelt Dezimalkommas in Punkten um, falls noetig (spaltenweise)."""
    for col in df.columns:
        if df[col].dtype == object:
            # Versuche, Strings mit Komma-Dezimalen in Floats zu wandeln
            try:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("\u00A0", " ")  # geschuetztes Leerzeichen
                    .str.strip()
                )
                # Wenn viele Kommas vorkommen, probiere decimal=','
                if df[col].str.contains(r",\d").mean() > 0.3:
                    df[col] = df[col].str.replace(".", "", regex=False)  # Tausenderpunkte raus
                    df[col] = df[col].str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                # Wenn es nicht klappt, lassen wir die Spalte wie sie ist
                pass
    return df


def find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def get_axes_series(
    df: pd.DataFrame,
    fluss_ml_min: float,
) -> Tuple[pd.Series, pd.Series, str, str]:
    """Nutzt das feste Format: Time_min (min) und Signal.
    Elutionsvolumen [mL] = Time_min [min] * Flussrate [mL/min].
    """
    t = df["Time_min"]
    y = df["Signal"]
    x = t * fluss_ml_min

    # Nach x sortieren (monotoner Plot)
    order = x.argsort()
    x = x.iloc[order]
    y = y.iloc[order]

    x_label = rf"Elutionsvolumen $V_E$ [\si{{\milli\litre}}] (Zeit $\times$ {fluss_ml_min} \si{{\milli\litre\per\minute}})"
    y_label = "Signal"
    return x, y, x_label, y_label



def _find_peak_indices(y: pd.Series,
                       min_y: float = PEAK_MIN_Y,
                       window: int | None = None,
                       prominence: float = PEAK_PROMINENCE,
                       min_sep: int = PEAK_MIN_SEP,
                       smooth_window: int = SMOOTH_WINDOW) -> list:
    """Robuste Peak-Detektion ohne SciPy.
    Schritte:
    1) optionale Glaettung (rollender Mittelwert) fuer die *Detektion*.
    2) Kandidaten: Vorzeichenwechsel der 1. Ableitung (+ -> -); Plateaus werden zentriert.
    3) Prominenz: Differenz zur minimalen Talhoehe links/rechts (bis zur naechsten Steigung).
    4) Mindesthoehe und Mindestabstand filtern.
    """
    y = pd.Series(y).astype(float).reset_index(drop=True)
    n = len(y)
    if n < 3:
        return []

    # 1) Glaettung fuer Detektion (nicht fuer Plot benutzt)
    sw = max(3, smooth_window)
    if sw % 2 == 0:
        sw += 1
    ys = y.rolling(sw, center=True, min_periods=1).mean()

    # 2) Ableitung und Kandidaten (inkl. Plateaubehandlung)
    dy = ys.diff()
    candidates = []
    i = 1
    while i < n - 1:
        # lokales Maximum strikter Fall
        if dy.iloc[i] > 0 and dy.iloc[i + 1] <= 0:
            candidates.append(i)
            i += 1
            continue
        # Plateau: steigende Flanke, dann mehrere ~0, dann fallende Flanke
        if dy.iloc[i] > 0 and abs(dy.iloc[i + 1]) < 1e-12:
            j = i + 1
            while j < n - 1 and abs(dy.iloc[j]) < 1e-12:
                j += 1
            if j < n - 1 and dy.iloc[j] < 0:
                # maximum liegt in der Mitte des Plateaus [i..j]
                plateau_start = i
                plateau_end = j
                peak_idx = (plateau_start + plateau_end) // 2
                candidates.append(peak_idx)
                i = j + 1
                continue
        i += 1

    if not candidates:
        return []

    # 3) Prominenzberechnung je Kandidat an ys (geglaettete Kurve fuer stabile Minima)
    def prominence_at(k: int) -> float:
        pk = ys.iloc[k]
        # links bis Minimum laufen, solange es abfaellt; stoppe wenn wieder steigt
        left = k - 1
        last = pk
        while left > 0 and ys.iloc[left] <= last:
            last = ys.iloc[left]
            left -= 1
        left_min = ys.iloc[left + 1]
        # rechts
        right = k + 1
        last = pk
        while right < n - 1 and ys.iloc[right] <= last:
            last = ys.iloc[right]
            right += 1
        right_min = ys.iloc[right - 1]
        return float(pk - max(left_min, right_min))

    peaks = []
    for k in candidates:
        if not np.isfinite(y.iloc[k]) or y.iloc[k] < min_y:
            continue
        prom = prominence_at(k)
        if prom >= prominence:
            peaks.append((k, prom, y.iloc[k]))

    if not peaks:
        return []

    # 4) Mindestabstand: priorisiere erst Hoehe, dann Prominenz
    peaks.sort(key=lambda t: (t[2], t[1]), reverse=True)
    selected = []
    for idx, prom, height in peaks:
        if all(abs(idx - s) >= (window if window else min_sep) and abs(idx - s) >= min_sep for s in selected):
            selected.append(idx)
    selected.sort()
    return selected


# Helper zum Extrahieren der Peak-Volumina (nur Apex, y>=Schwelle)
def _extract_peak_volumes(x: pd.Series, y: pd.Series) -> list[float]:
    """Gibt die Elutionsvolumina (mL) der detektierten Apex-Peaks zurueck (y >= PEAK_MIN_Y)."""
    md = max(5, len(y) // 200)
    idx = _find_peak_indices(y, min_y=PEAK_MIN_Y, window=md, prominence=PEAK_PROMINENCE, min_sep=PEAK_MIN_SEP, smooth_window=SMOOTH_WINDOW)
    if not idx:
        return []
    xv = x.iloc[idx].astype(float).tolist()
    xv.sort()  # kleine Ve zuerst (groesseres MP)
    return xv


def plot_and_save(x: pd.Series, y: pd.Series, x_label: str, y_label: str, src: Path) -> Path:
    """Erstellt den Plot und speichert als PDF im selben Ordner wie die CSV."""
    # Sicherheitshalber: Werte als Series sicherstellen
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    # Peak-Detektion (Apex) und Annotation
    # Heuristik fuer Fensterbreite basierend auf Datenlaenge
    md = max(5, len(y) // 200)
    peak_idx = _find_peak_indices(y, min_y=PEAK_MIN_Y, window=md, prominence=PEAK_PROMINENCE, min_sep=PEAK_MIN_SEP, smooth_window=SMOOTH_WINDOW)

    out_path = src.with_suffix(".pdf")
    plt.figure(figsize=(16/2.54, 6.5/2.54))
    plt.plot(x, y)
    # Peaks markieren und Elutionsvolumen anschreiben
    if peak_idx:
        xp = x.iloc[peak_idx]
        yp = y.iloc[peak_idx]
        plt.plot(xp, yp, 'o')
        for xv, yv in zip(xp, yp):
            txt = rf"\SI{{{xv:.2f}}}{{\milli\litre}}"
            plt.annotate(
                txt,
                xy=(xv, yv),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6)
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(top=40)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def process_file(csv_path: Path, fluss: float) -> Optional[Path]:
    try:
        df = read_csv_robust(csv_path)
        df = normalize_decimals(df)
        x, y, x_label, y_label = get_axes_series(df, fluss)
        out_pdf = plot_and_save(x, y, x_label, y_label, csv_path)
        return out_pdf
    except Exception as e:
        print(f"Fehler bei {csv_path.name}: {e}", file=sys.stderr)
        return None


# Kalibrationspunkte sammeln und Fit durchfuehren
def perform_calibration(base: Path, fluss: float) -> Optional[Path]:
    """Sammelt (Ve, log10(M)) Punkte aus allen Blue/Green/Red-Dateien und fit:
    log10(M) = a + b * Ve. Gibt Pfad zur erzeugten JSON-Datei zurueck (ohne Plot).
    Zuordnung: kleinere Ve <-> groessere MP (je Farbe). Nur die ersten 4 Peaks je Farbe
    werden beruecksichtigt und in der Reihenfolge (Ve aufsteigend) den MP-Werten
    (absteigend) gemappt.
    """
    ve_all: list[float] = []
    logM_all: list[float] = []
    color_points: dict[str, list[tuple[float, float]]] = {k: [] for k in MP_DICT}

    files = sorted([p for p in base.iterdir() if p.suffix.lower() == ".csv"])
    if not files:
        return None
    
    for csv_path in files:
        name = csv_path.stem.lower()
        key = None
        for k in MP_DICT.keys():
            if k in name:
                key = k
                break
        if key is None:
            continue  # Datei gehoert nicht zu blue/green/red Standards
        try:
            df = read_csv_robust(csv_path)
            df = normalize_decimals(df)
            x, y, _, _ = get_axes_series(df, fluss)
            ve_peaks = _extract_peak_volumes(x, y)
            if not ve_peaks:
                continue
            ve_peaks = ve_peaks[:len(MP_DICT[key])]  # max 4 Peaks
            # Map: kleine Ve -> grosse MP
            mp_desc = list(MP_DICT[key])  # bereits absteigend
            mp_desc = mp_desc[:len(ve_peaks)]
            # Falls mehr Ve als MP, kuerzen; falls weniger, nur vorhandene verwenden
            for ve, mp in zip(ve_peaks, mp_desc):
                ve_all.append(ve)
                logM_all.append(np.log10(mp))
                color_points[key].append((ve, np.log10(mp)))
        except Exception as e:
            print(f"Kalib-Warnung bei {csv_path.name}: {e}", file=sys.stderr)
            continue

    if len(ve_all) < 2:
        print("Zu wenige Punkte fuer Kalibration.", file=sys.stderr)
        return None

    # Kubischer Fit: log10(M) = a + b*Ve + c*Ve^2 + d*Ve^3
    ve_arr = np.asarray(ve_all, dtype=float)
    logM_arr = np.asarray(logM_all, dtype=float)
    coeffs = np.polyfit(ve_arr, logM_arr, 3)  # [d, c, b, a]
    d, c, b, a = coeffs
    # R^2 auf Basis des kubischen Modells
    pred = np.polyval(coeffs, ve_arr)
    ss_res = float(np.sum((logM_arr - pred)**2))
    ss_tot = float(np.sum((logM_arr - np.mean(logM_arr))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    # Ve-Min/Max berechnen fuer JSON
    ve_min = float(np.min(ve_arr))
    ve_max = float(np.max(ve_arr))

    # Plot fuer Kalibration: farbcodierte Punkte (Marker 'x') und Fit-Kurve
    cal_pdf = base / "calibration_points.pdf"
    try:
        plt.figure(figsize=(16/2.54, 10/2.54))
        # Punkte farblich/marker nach Satz
        color_map = {"green": "green", "red": "red", "blue": "blue"}
        for k, pts in color_points.items():
            if not pts:
                continue
            v = [p[0] for p in pts]
            m = [p[1] for p in pts]
            plt.scatter(v, m, marker='x', color=color_map.get(k, None), label=k)
        # Fit-Kurve (kubisch)
        xfit = np.linspace(float(np.min(ve_arr)), float(np.max(ve_arr)), 400)
        yfit = np.polyval(coeffs, xfit)
        plt.plot(xfit, yfit, label="Fit")
        plt.xlabel(r"Elutionsvolumen $V_E$ [\si{\milli\litre}]")
        plt.ylabel(r"$\log_{10}(M)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cal_pdf, format="pdf")
        plt.close()
    except Exception as e:
        print(f"Konnte {cal_pdf.name} nicht schreiben: {e}", file=sys.stderr)

    # Koeffizienten als JSON speichern
    try:
        import json
        coeffs_json = {
            "model": "log10(M) = a + b*Ve + c*Ve^2 + d*Ve^3",
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "r2": float(r2),
            "n": int(len(ve_all)),
            "Ve_min_mL": ve_min,
            "Ve_max_mL": ve_max,
        }
        with open(base / "calibration_coeffs.json", "w") as f:
            json.dump(coeffs_json, f, indent=2)
        print(f"✔ Gespeichert: {base / 'calibration_coeffs.json'}")
    except Exception as e:
        print(f"Konnte calibration_coeffs.json nicht schreiben: {e}", file=sys.stderr)

    # Punkte als CSV schreiben
    pts_csv = base / "calibration_points.csv"
    try:
        import csv
        with open(pts_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Ve_mL", "log10_M", "set"])
            for k, pts in color_points.items():
                for ve, lm in pts:
                    w.writerow([f"{ve:.6f}", f"{lm:.6f}", k])
    except Exception as e:
        print(f"Konnte calibration_points.csv nicht schreiben: {e}", file=sys.stderr)

    print(f"Kalibration: a={a:.6f}, b={b:.6f}, R^2={r2:.4f}")
    return base / "calibration_coeffs.json"


def main():
    # --- Hier Pfad und Flussrate angeben ---
    pfad = Path("/Users/musamoin/Desktop/BA-HS25/sec/both_columns/cal/")  # <-- hier Pfad angeben
    fluss = 0.8  # mL/min

    base = pfad.expanduser().resolve()
    if not base.exists() or not base.is_dir():
        print(f"Pfad nicht gefunden oder kein Ordner: {base}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted([p for p in base.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        print(f"Keine CSV-Dateien in: {base}")
        sys.exit(0)

    print(f"Gefundene Dateien: {len(csv_files)}")
    ok, fail = 0, 0
    for csv_path in csv_files:
        out = process_file(csv_path, fluss)
        if out is not None:
            ok += 1
            print(f"✔ Gespeichert: {out}")
        else:
            fail += 1

    # Kalibration ueber alle Dateien im Ordner
    perform_calibration(base, fluss)

    print(f"Fertig. Erfolgreich: {ok}, Fehlgeschlagen: {fail}")


if __name__ == "__main__":
    main()