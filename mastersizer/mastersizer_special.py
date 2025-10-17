#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lese ein einzelnes Mastersizer-Excel (Size Classes / Number Density), berechne d50 (Median) und CV,
und erstelle einen Plot (x: Size classes, y: Number distr. (%)). Der Plot enthaelt d50 und CV
und wird als gleichnamiges PDF im selben Ordner gespeichert.

Aufruf:
    python script.py /pfad/zu/DEINEM_Mastersizer.xlsx
oder (ohne Argument) Konstante INPUT_XLS unten anpassen.

Hinweis zu d50:
- Median ueber kumulierte Verteilung mit linearer Interpolation zwischen Klassenmitten.

CV-Definition:
- CV (%) = 100 * sigma / mu, mit mu als gewichtetem arithmetischem Mittel und sigma als
  entsprechender Standardabweichung (beides gewichtet).
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
# Gib hier optional ein Default-File an, falls kein CLI-Argument uebergeben wird:
INPUT_XLS = "/Users/musamoin/Desktop/BA_HS25/experiments/EXP-004/mastersizer/EXP-004-Mastersizer.xlsx"

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}
FIG_WIDTH_CM = 7.0
FIG_HEIGHT_CM = 6.5

DEFAULT_XLIM = (0.1,10)  # z.B. (0.01, 1000)
DEFAULT_YLIM = (-0.1,0.2)  # z.B. (0, 20)

def cm_to_in(x: float) -> float:
    return x / 2.54

# =============================
# Parsen & Berechnen
# =============================

def _try_parse_by_header_markers(xl_path: str) -> Optional[pd.DataFrame]:
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
    try:
        df = pd.read_excel(xl_path, skiprows=2, names=["Size", "Number"])
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

def median_from_discrete_classes(size: pd.Series, number: pd.Series) -> float:
    w = number.astype(float)
    x = size.astype(float)
    order = np.argsort(x.values)
    x = x.values[order]
    w = w.values[order]

    W = w.sum()
    if W <= 0:
        raise ValueError("Summe der Gewichte ist 0.")

    F = np.cumsum(w) / W
    idx = np.searchsorted(F, 0.5)

    if idx == 0:
        return float(x[0])
    if idx >= len(x):
        return float(x[-1])

    x0, x1 = x[idx - 1], x[idx]
    F0, F1 = F[idx - 1], F[idx]
    if F1 == F0:
        return float((x0 + x1) / 2.0)
    return float(x0 + (0.5 - F0) / (F1 - F0) * (x1 - x0))

def compute_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    d50 = median_from_discrete_classes(df["Size"], df["Number"])
    mu, sigma = weighted_mean_and_std(df["Size"], df["Number"])
    cv = 100.0 * sigma / mu if mu != 0 else float("nan")
    return d50, mu, cv

# =============================
# Plotten
# =============================

def make_plot(df: pd.DataFrame, d50: float, cv: float, title: str, out_pdf: str) -> None:
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
    ticks = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    labels = ["0.01", "0.1", "1.0", "10.0", "100.0", "1000.0"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    if DEFAULT_XLIM is not None:
        ax.set_xlim(DEFAULT_XLIM)
    if DEFAULT_YLIM is not None:
        ax.set_ylim(DEFAULT_YLIM)

    plt.xlabel(r'Partikelgroesse $d_p$ / \si{\micro\meter}')
    plt.ylabel(r'Partikelanteil $n$ / \si{\percent}')

    text = (
        rf"$d_{{50}} = {d50:.4g}\,\si{{\micro\meter}}$"
        "\n"
        rf"$CV = {cv:.2f}\,\si{{\percent}}$"
    )
    plt.gca().text(0.98, 0.95, text, transform=plt.gca().transAxes,
                   ha="right", va="top", bbox=dict(boxstyle="round", fc="white"))

    try:
        ymin, ymax = plt.ylim()
        plt.vlines(d50, ymin, ymax, linestyles="dashed", linewidth=1, color="black")
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_pdf, transparent=True, bbox_inches='tight')
    plt.close()

def process_file(xl_path: str) -> str:
    df = load_mastersizer_table(xl_path)
    d50, mu, cv = compute_metrics(df)

    base, _ = os.path.splitext(xl_path)
    out_pdf = base + "_special.pdf"
    title = os.path.basename(base)
    make_plot(df, d50, cv, title, out_pdf)

    print(f"OK: {xl_path}")
    print(f"  d50 = {d50:.6g} µm,  CV = {cv:.3f} %")
    print(f"  -> gespeichert: {out_pdf}")
    return out_pdf

# =============================
# Main (einzelnes File)
# =============================

def main(xl_path: str) -> int:
    if not os.path.isfile(xl_path):
        print(f"Pfad ist keine Datei: {xl_path}")
        return 1
    ext = os.path.splitext(xl_path)[1].lower()
    if ext not in EXCEL_EXTS:
        print(f"Unerwartete Dateiendung ({ext}). Erwarte eine Excel-Datei: {EXCEL_EXTS}")
        return 1
    try:
        process_file(xl_path)
        print("Fertig, keine Fehler.")
        return 0
    except Exception as e:
        print(f"FEHLER: {e}")
        return 2

if __name__ == "__main__":
    xls = INPUT_XLS
    if len(sys.argv) > 1:
        xls = sys.argv[1]
    sys.exit(main(xls))