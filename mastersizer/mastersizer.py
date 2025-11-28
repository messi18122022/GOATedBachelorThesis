#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lese alle Excel-Files im festen Pfad EXP-017/mastersizer ein und
erstelle zwei Plots:

1) Overlay-Plot aller Kurven (alle Dateien). Legende: Dateiname ohne Endung.
2) Einzel-Plot fuer die Datei "EXP-017 in EtOH.xlsx" mit berechneten Dn, DV
   und PDI_size (= DV / Dn). Die Werte werden per Hilfslinien eingezeichnet
   und in der Legende angegeben.

Definitionen:
- Dn = D[1,0] (Moment-Ratio-Definition-System)
- DV = D[4,3]
- PDI_size = DV / Dn
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
BASE_DIR = "/Users/musamoin/Desktop/BA-HS25/experiments/EXP-017/mastersizer"
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}

FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = 6.5

def cm_to_in(x: float) -> float:
    return x / 2.54

# =============================
# Hilfsfunktionen
# =============================

def find_excel_files(base_dir: str) -> list[str]:
    return sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(base_dir)
        for fn in files
        if os.path.splitext(fn)[1].lower() in EXCEL_EXTS
    )


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


def moment_ratio(size: pd.Series, number: pd.Series, p: int, q: int) -> float:
    """
    Berechnet D[p,q] gemäss Moment-Ratio-Definition:
      - p > q: D[p,q] = [(Σ n_i * X_i^p) / (Σ n_i * X_i^q)]^(1/(p-q))
      - p = q: D[p,p] = exp( (Σ n_i * X_i^p * ln X_i) / (Σ n_i * X_i^p) )
    """
    w = number.astype(float)
    x = size.astype(float)
    Wp = float((w * (x ** p)).sum())
    if p > q:
        Wq = float((w * (x ** q)).sum())
        if Wq <= 0:
            return float("nan")
        ratio = Wp / Wq
        # Schutz gegen numerische Randfälle
        if ratio <= 0:
            return float("nan")
        return float(ratio ** (1.0 / (p - q)))
    else:  # p == q
        # Verwende nur positive x für ln
        mask = (w > 0) & (x > 0)
        w_eff = w[mask]
        x_eff = x[mask]
        if len(x_eff) == 0:
            return float("nan")
        Wp_eff = float((w_eff * (x_eff ** p)).sum())
        if Wp_eff <= 0:
            return float("nan")
        val = float((w_eff * (x_eff ** p) * np.log(x_eff)).sum() / Wp_eff)
        return float(np.exp(val))


def _init_figure_and_axes() -> tuple[plt.Figure, mpl.axes.Axes]:
    """Erzeuge Standard-Figur mit log-x-Achse und einheitlichem Layout."""
    fig = plt.figure(figsize=(cm_to_in(FIG_WIDTH_CM), cm_to_in(FIG_HEIGHT_CM)))
    ax = plt.gca()
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    ax.grid(True, which="both", linestyle="-", linewidth=0.2, color="black", alpha=0.5)
    ax.set_xscale("log")

    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)

    ticks = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    labels = ["0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.set_xlabel(r'Partikelgroesse $d_p$ / \si{\micro\meter}')
    ax.set_ylabel(r'Partikelanteil $n$ / \si{\percent}')
    return fig, ax


def compute_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Gibt (Dn, DV, PDI_size) zurueck (vgl. Kap. 2.8.2)."""
    Dn = moment_ratio(df["Size"], df["Number"], 1, 0)
    DV = moment_ratio(df["Size"], df["Number"], 4, 3)
    pdi = DV / Dn if (Dn and not np.isnan(Dn) and Dn != 0) else float("nan")
    return Dn, DV, pdi


def make_single_plot(df: pd.DataFrame, Dn: float, DV: float, pdi: float, title: str, out_pdf: str) -> None:
    fig, ax = _init_figure_and_axes()
    ax.plot(df["Size"], df["Number"], color="black")

    # Hilfslinien und Legende fuer Dn, DV, PDI_size
    try:
        ymin, ymax = ax.get_ylim()
        ax.vlines(Dn, ymin, ymax, linestyles="dashed", linewidth=1, color="tab:blue",
                  label=rf"$D_{{n}} = {Dn:.2f}\,\si{{\micro\meter}}$")
        ax.vlines(DV, ymin, ymax, linestyles="dashed", linewidth=1, color="tab:red",
                  label=rf"$D_{{V}} = {DV:.2f}\,\si{{\micro\meter}}$")
        ax.plot([], [], " ", label=rf"$\mathrm{{PDI_{{size}}}} = \frac{{D_V}}{{D_n}} = {pdi:.3f}$")
        ax.legend(loc="best")
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_pdf, transparent=True, bbox_inches="tight")
    plt.close(fig)


def make_overlay_plot(curves: dict[str, pd.DataFrame], title: str, out_pdf: str) -> None:
    fig, ax = _init_figure_and_axes()

    for label, df in curves.items():
        ax.plot(df["Size"], df["Number"], label=label, linewidth=0.8)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf, transparent=True, bbox_inches="tight")
    plt.close(fig)


def process_all(files: list[str]) -> int:
    """Erzeuge Overlay-Plot (alle Dateien) und Einzel-Plot fuer EtOH."""
    if not files:
        print("Keine Dateien uebergeben.")
        return 1

    etoh_name = "EXP-017 in EtOH.xlsx"
    etoh_path: Optional[str] = None
    overlay_curves: dict[str, pd.DataFrame] = {}

    for p in files:
        df = load_mastersizer_table(p)
        base = os.path.basename(p)
        overlay_curves[os.path.splitext(base)[0]] = df
        if base == etoh_name:
            etoh_path = p
            etoh_df = df

    # Overlay-Plot
    if overlay_curves:
        overlay_pdf = os.path.join(BASE_DIR, "EXP-017-overlay.pdf")
        make_overlay_plot(overlay_curves, "", overlay_pdf)
        print(f"Overlay-Plot gespeichert: {overlay_pdf}")
    else:
        print("Keine Dateien fuer den Overlay-Plot gefunden.")

    # Einzel-Plot fuer EtOH
    if etoh_path is not None:
        Dn, DV, pdi = compute_metrics(etoh_df)
        base, _ = os.path.splitext(etoh_path)
        single_pdf = base + ".pdf"
        make_single_plot(etoh_df, Dn, DV, pdi, "", single_pdf)
        print(f"EtOH-Plot gespeichert: {single_pdf}")
        print(f"  Dn = {Dn:.6g} µm,  DV = {DV:.6g} µm,  PDI_size = {pdi:.6g}")
    else:
        print(f"ACHTUNG: Datei '{etoh_name}' wurde nicht gefunden.")
    return 0


# =============================
# Main
# =============================

def main(base_dir: str) -> int:
    files = find_excel_files(base_dir)
    if not files:
        print(f"Keine Dateien gefunden in: {base_dir}")
        return 1

    try:
        return_code = process_all(files)
    except Exception as e:
        print(f"FEHLER beim Verarbeiten der Dateien: {e}")
        return_code = 2

    return return_code


if __name__ == "__main__":
    base = BASE_DIR
    if len(sys.argv) > 1:
        base = sys.argv[1]
    sys.exit(main(base))