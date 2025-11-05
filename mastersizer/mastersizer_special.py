#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liest genau EINE Mastersizer-Excel-Datei (Pfad im Script angeben oder via CLI),
berechnet Dn (zahlenmittlerer Durchmesser) und CV und erstellt einen Plot
(x: Size classes, y: Number distr. (%)).

Der Plot enthaelt KEINE Legende. Optional koennen x- und y-Achsenlimits gesetzt
werden. Die Ausgabe ist ein gleichnamiges PDF im selben Ordner.

Hinweis zu Dn:
- Dn ist der zahlenmittlere Durchmesser: Dn = Summe(n_i * d_{p,i}) / Summe(n_i).
- In unserem Code ist Dn identisch mit dem gewichteten arithmetischen Mittel \(\mu\)
  (Gewichte = Number-\%); vgl. Kap. 2.8.2.

CV-Definition:
- CV (%) = 100 * sigma / mu, mit mu als gewichtetem arithmetischem Mittel der
  Partikelgroessen und sigma als zugehoeriger Standardabweichung (ebenfalls gewichtet).

Getestet mit Dateien im Stil von EXP-001-Mastersizer.xlsx.
"""

from __future__ import annotations
import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================
# Konfiguration
# =============================
# Pfad zur EINEN zu verarbeitenden Excel-Datei (kann via CLI ueberschrieben werden)
XL_PATH = "/Users/musamoin/Desktop/BA-HS25/experiments/EXP-008/mastersizer/EXP-008-Mastersizer.xlsx"  # <— anpassen!

# Achsenlimits: None = automatisch. Fuer x (log-Skala) muessen beide Grenzen > 0 sein.
X_LIMITS: Optional[Tuple[float, float]] = (0.1, 1.0) # z.B. (0.05, 500.0)
Y_LIMITS: Optional[Tuple[float, float]] = (-0.1,1)  # z.B. (0.0, 8.0)

FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = 6.5

# LaTeX/siunitx aktivieren und Transparenz fuer Saves
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{siunitx}'
})
mpl.rcParams['savefig.transparent'] = True


def cm_to_in(x: float) -> float:
    return x / 2.54


# =============================
# Einlesen & Kennwerte
# =============================

def _try_parse_by_header_markers(xl_path: str) -> Optional[pd.DataFrame]:
    """Versuche, die Daten zu finden, indem wir die Kopfzeile explizit suchen.
    Erwarte eine Zeile mit (ungefaehr) "Size Classes" und "Number Density".
    Liefert DataFrame mit Spalten ["Size", "Number"], oder None.
    """
    try:
        raw = pd.read_excel(xl_path, header=None, dtype=object)
    except Exception:
        return None

    header_row = None
    for i in range(min(len(raw), 30)):
        c0 = str(raw.iloc[i, 0]).lower() if raw.shape[1] > 0 else ""
        c1 = str(raw.iloc[i, 1]).lower() if raw.shape[1] > 1 else ""
        if ("size" in c0 and "class" in c0) and ("number" in c1 and ("density" in c1 or "%" in c1)):
            header_row = i
            break

    if header_row is None:
        return None

    data = raw.iloc[header_row + 1 :, :2].copy()
    data.columns = ["Size", "Number"]

    data["Size"] = pd.to_numeric(data["Size"], errors="coerce")
    data["Number"] = pd.to_numeric(data["Number"], errors="coerce")
    data = data.dropna()
    data = data[(data["Number"] >= 0)]
    if len(data) == 0:
        return None
    return data.reset_index(drop=True)


def _try_parse_by_fixed_skip(xl_path: str) -> Optional[pd.DataFrame]:
    """Fallback fuer bekannte Struktur: 2 Kopfzeilen, dann zwei Spalten (Size, Number)."""
    try:
        df = pd.read_excel(xl_path, skiprows=2, names=["Size", "Number"])  # beobachtetes Format
        df["Size"] = pd.to_numeric(df["Size"], errors="coerce")
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
        df = df.dropna()
        df = df[(df["Number"] >= 0)]
        if len(df) == 0:
            return None
        return df.reset_index(drop=True)
    except Exception:
        return None


def load_mastersizer_table(xl_path: str) -> pd.DataFrame:
    """Lade die Tabelle mit Spalten [Size, Number] als floats (Size in µm, Number in %)."""
    df = _try_parse_by_header_markers(xl_path)
    if df is None:
        df = _try_parse_by_fixed_skip(xl_path)
    if df is None or any(col not in df.columns for col in ["Size", "Number"]):
        raise ValueError(f"Kann Daten nicht einlesen: {xl_path}")

    df = df.sort_values("Size").reset_index(drop=True)
    if df["Number"].max() <= 1.0:
        df["Number"] = df["Number"] * 100.0
    return df


def weighted_mean_and_std(size: pd.Series, number: pd.Series) -> Tuple[float, float]:
    w = number.astype(float)
    x = size.astype(float)
    W = w.sum()
    if W <= 0:
        raise ValueError("Summe der Gewichte ist 0.")
    mu = float((w * x).sum() / W)
    var = float((w * (x - mu) ** 2).sum() / W)
    sigma = np.sqrt(var)
    return mu, sigma


def compute_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    mu, sigma = weighted_mean_and_std(df["Size"], df["Number"])
    Dn = mu
    cv = 100.0 * sigma / Dn if Dn != 0 else float("nan")
    return Dn, cv


# =============================
# Plot
# =============================

def make_plot(df: pd.DataFrame, Dn: float, cv: float, out_pdf: str) -> None:
    fig = plt.figure(figsize=(cm_to_in(FIG_WIDTH_CM), cm_to_in(FIG_HEIGHT_CM)))
    ax = plt.gca()
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    plt.plot(df["Size"], df["Number"], color="black")
    plt.grid(True, which="both", linestyle="-", linewidth=0.2, color="black", alpha=0.5)
    plt.xscale("log")

    # Achsenlimits anwenden (optional) – X zuerst, damit Tick-Formatierung drauf reagieren kann
    if X_LIMITS is not None:
        if X_LIMITS[0] <= 0 or X_LIMITS[1] <= 0:
            print("Warnung: X-Limits fuer log-Skala muessen > 0 sein. Ignoriere X_LIMITS.")
        else:
            ax.set_xlim(X_LIMITS)
    if Y_LIMITS is not None:
        ax.set_ylim(Y_LIMITS)

    # X-Achsenbeschriftung: immer Dezimalzahlen (keine Zehnerpotenzen)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    import math
    xmin, xmax = ax.get_xlim()

    # Falls der Bereich innerhalb EINER Dekade liegt (z. B. 0.1–1.0 oder 1–10),
    # setzen wir lineare Ticks mit Schrittweite der jeweiligen Dekade (0.1 bzw. 1 usw.).
    decade = math.floor(np.log10(xmin))
    def _plain(x, pos):
        # Format ohne wissenschaftliche Notation und ohne trailing zeros
        s = f"{x:.10f}".rstrip('0').rstrip('.')
        return s

    if xmin >= 10**decade and xmax <= 10**(decade + 1):
        step = 10**decade
        start = np.ceil(xmin / step) * step
        stop = np.floor(xmax / step) * step
        if stop < start:
            stop = start
        ticks = np.arange(start, stop + 0.5 * step, step)
        ax.xaxis.set_major_locator(FixedLocator(ticks.tolist()))
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        # Fallback: nur Dezimalwerte an den Dekaden anzeigen
        decade_ticks = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        ax.xaxis.set_major_locator(FixedLocator(decade_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # plt.xlabel(r'Partikelgrösse $d_p$ / \si{\micro\meter}')
    plt.ylabel(r'Partikelanteil $n$ / \si{\percent}')

    # Hilfslinie bei Dn (nicht X-Autoscale auslösen)
    try:
        ymin, ymax = ax.get_ylim()
        ax.vlines(Dn, ymin, ymax, linestyles="dashed", linewidth=1, color="black")
    except Exception:
        pass

    # Nach hinzugefügten Artists können sich Limits wieder automatisch erweitern -> Limits erzwingen
    if X_LIMITS is not None and X_LIMITS[0] > 0 and X_LIMITS[1] > 0:
        ax.set_xlim(X_LIMITS)
    if Y_LIMITS is not None:
        ax.set_ylim(Y_LIMITS)

    plt.tight_layout()
    plt.savefig(out_pdf, transparent=True, bbox_inches='tight')
    plt.close()


def process_file(xl_path: str) -> str:
    df = load_mastersizer_table(xl_path)
    Dn, cv = compute_metrics(df)

    base, _ = os.path.splitext(xl_path)
    out_pdf = base + "_special.pdf"

    make_plot(df, Dn, cv, out_pdf)

    print(f"OK: {xl_path}")
    print(f"  Dn = {Dn:.6g} µm,  CV = {cv:.3f} %")
    print(f"  -> gespeichert: {out_pdf}")
    return out_pdf


# =============================
# Main
# =============================

def main(one_xl_path: Optional[str]) -> int:
    path = one_xl_path or XL_PATH
    if not path or not os.path.isfile(path):
        print("Fehler: Gueltigen Dateipfad angeben (XL_PATH im Script oder als CLI-Argument).")
        return 1
    try:
        process_file(path)
        return 0
    except Exception as e:
        print(f"FEHLER: {e}")
        return 2


if __name__ == "__main__":
    # Erlaube optionales Ueberschreiben via CLI: python mastersizer_special.py /pfad/zur/datei.xlsx
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(arg_path))
