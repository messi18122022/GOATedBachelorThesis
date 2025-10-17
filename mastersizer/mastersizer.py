#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Durchsuche rekursiv den Datenpfad nach Excel-Files mit "Mastersizer" im Dateinamen,
lese die Daten (Size Classes / Number Density), berechne Dn (zahlenmittlerer Durchmesser) und CV,
und erstelle fuer jedes File einen Plot (x: Size classes, y: Number distr. (%)).
Der Plot enthaelt die Werte fuer Dn und CV und wird als gleichnamiges PDF im
selben Ordner gespeichert.

Hinweis zu Dn:
- Dn ist der zahlenmittlere Durchmesser: Dn = Summe(n_i * d_{p,i}) / Summe(n_i).
- In unserem Code ist Dn identisch mit dem gewichteten arithmetischen Mittel \(\mu\) (Gewichte = Number-\%); vgl. Kap. 2.8.2.

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

# LaTeX/siunitx aktivieren und Transparenz fuer Saves
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{siunitx}'
})
mpl.rcParams['savefig.transparent'] = True

# =============================
# Konfiguration
# =============================
BASE_DIR = "/Users/musamoin/Desktop/BA_HS25/experiments"
FILE_KEYWORD = "Mastersizer"
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}

FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = 6.5

def cm_to_in(x: float) -> float:
    return x / 2.54

# =============================
# Hilfsfunktionen
# =============================

def find_excel_files(base_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(base_dir):
        for fn in files:
            name_lower = fn.lower()
            if FILE_KEYWORD.lower() in name_lower:
                ext = os.path.splitext(fn)[1].lower()
                if ext in EXCEL_EXTS:
                    paths.append(os.path.join(root, fn))
    return sorted(paths)


def _try_parse_by_header_markers(xl_path: str) -> Optional[pd.DataFrame]:
    """Versuche, die Daten zu finden, indem wir die Kopfzeile explizit suchen.
    Erwarte eine Zeile mit (ungefaehr) "Size Classes" und "Number Density".
    Liefert DataFrame mit Spalten ["Size", "Number"], oder None, falls nicht gefunden."""
    try:
        raw = pd.read_excel(xl_path, header=None, dtype=object)
    except Exception:
        return None

    header_row = None
    for i in range(min(len(raw), 30)):  # nur die ersten paar Zeilen durchsuchen
        c0 = str(raw.iloc[i, 0]).lower() if raw.shape[1] > 0 else ""
        c1 = str(raw.iloc[i, 1]).lower() if raw.shape[1] > 1 else ""
        if ("size" in c0 and "class" in c0) and ("number" in c1 and ("density" in c1 or "%" in c1)):
            header_row = i
            break

    if header_row is None:
        return None

    data = raw.iloc[header_row + 1 :, :2].copy()
    data.columns = ["Size", "Number"]

    # numerisch bereinigen
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

    # Nach Groesse sortieren und 0-Werte am Rand entfernen
    df = df.sort_values("Size").reset_index(drop=True)
    # Falls Number in [0..1] statt in %, dann auf % skalieren
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


def median_from_discrete_classes(size: pd.Series, number: pd.Series) -> float:
    """Berechne d50 (Median) ueber kumulative Summe und lineare Interpolation
    zwischen benachbarten Klassenmitten.
    """
    w = number.astype(float)
    x = size.astype(float)
    order = np.argsort(x.values)
    x = x.values[order]
    w = w.values[order]

    W = w.sum()
    if W <= 0:
        raise ValueError("Summe der Gewichte ist 0.")

    F = np.cumsum(w) / W
    # erstes Index, bei dem F >= 0.5
    idx = np.searchsorted(F, 0.5)

    if idx == 0:
        return float(x[0])
    if idx >= len(x):
        return float(x[-1])

    x0, x1 = x[idx - 1], x[idx]
    F0, F1 = F[idx - 1], F[idx]
    if F1 == F0:
        return float((x0 + x1) / 2.0)
    # lineare Interpolation
    return float(x0 + (0.5 - F0) / (F1 - F0) * (x1 - x0))


def compute_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """Gibt (Dn, CV_in_percent) zurueck (vgl. Kap. 2.8.2)."""
    mu, sigma = weighted_mean_and_std(df["Size"], df["Number"])
    Dn = mu
    cv = 100.0 * sigma / Dn if Dn != 0 else float("nan")
    return Dn, cv


def make_plot(df: pd.DataFrame, Dn: float, cv: float, title: str, out_pdf: str) -> None:
    fig = plt.figure(figsize=(cm_to_in(FIG_WIDTH_CM), cm_to_in(FIG_HEIGHT_CM)))
    ax = plt.gca()
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    plt.plot(df["Size"], df["Number"], color="black")
    plt.grid(True, which="both", linestyle="-", linewidth=0.2, color="black", alpha=0.5)
    plt.xscale("log")

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    # explizite Labels, damit 0.01 nicht zu 0.0 gerundet wird
    ticks = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    labels = ["0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    plt.xlabel(r'Partikelgrösse $d_p$ / \si{\micro\meter}')
    plt.ylabel(r'Partikelanteil $n$ / \si{\percent}')

    # Textbox mit Dn und CV (oben rechts)
    text = (
        rf"$D_{{n}} = {Dn:.2f}\,\si{{\micro\meter}}$"
        "\n"
        rf"$CV = {cv:.2f}\,\si{{\percent}}$"
    )
    plt.gca().text(0.98, 0.95, text, transform=plt.gca().transAxes,
                   ha="right", va="top", bbox=dict(boxstyle="round", fc="white"))

    # Hilfslinie bei Dn
    try:
        ymin, ymax = plt.ylim()
        plt.vlines(Dn, ymin, ymax, linestyles="dashed", linewidth=1, color="black")
    except Exception:
        pass

    plt.tight_layout()
    # als PDF speichern
    plt.savefig(out_pdf, transparent=True, bbox_inches='tight')
    plt.close()


def process_file(xl_path: str) -> str:
    df = load_mastersizer_table(xl_path)
    Dn, cv = compute_metrics(df)

    base, _ = os.path.splitext(xl_path)
    out_pdf = base + ".pdf"

    # Plot-Titel: Dateiname ohne Pfad
    title = os.path.basename(base)
    make_plot(df, Dn, cv, title, out_pdf)

    print(f"OK: {xl_path}")
    print(f"  Dn = {Dn:.6g} µm,  CV = {cv:.3f} %")
    print(f"  -> gespeichert: {out_pdf}")
    return out_pdf


# =============================
# Main
# =============================

def main(base_dir: str) -> int:
    files = find_excel_files(base_dir)
    if not files:
        print(f"Keine Dateien gefunden in: {base_dir}")
        return 1

    errors = 0
    for p in files:
        try:
            process_file(p)
        except Exception as e:
            errors += 1
            print(f"FEHLER bei {p}: {e}")

    if errors:
        print(f"Fertig mit {len(files)} Dateien, {errors} Fehler.")
    else:
        print(f"Fertig mit {len(files)} Dateien, keine Fehler.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    base = BASE_DIR
    if len(sys.argv) > 1:
        base = sys.argv[1]
    sys.exit(main(base))