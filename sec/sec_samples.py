#!/usr/bin/env python3
"""
Liest eine Kalibrations-JSON (a,b,c,d) und wandelt fuer alle Sample-CSV-Dateien
(Agilent DAD Export: '#Point,X(Minutes),Y(Response Units)') die Zeit -> V_E um
und dann V_E -> Molmasse M via: log10(M) = a + b*V_E + c*V_E^2 + d*V_E^3.

Hinweis: Es werden nur Daten im gueltigen V_E-Bereich der Kalibration geplottet (ve_min..ve_max) um Extrapolation zu vermeiden.

Erzeugt je Sample einen Plot **Signal vs. Molmasse** (x: M [Da], log-Skala) und
speichert als PDF im selben Ordner wie die CSV.

Aufrufbeispiel:
    python sec_samples.py --coeff /pfad/zur/calibration_coeffs.json \
                          --path  /pfad/zu/samples --fluss 0.8
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Defaults
# ------------------------------
DEFAULT_FLUSS = 0.8  # mL/min

# ------------------------------
# CSV Einlesen (fixes Format)
# ------------------------------

def read_csv_fixed(path: Path) -> pd.DataFrame:
    """Liest CSV im Agilent-DAD-Format (siehe sec.py)."""
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=["Point", "Time_min", "Signal"],
        sep=",",
        engine="python",
        encoding="latin-1",
        skip_blank_lines=True,
    )
    if df.shape[1] != 3:
        raise RuntimeError(
            f"Unerwartetes Spaltenformat in {path}. Erwartet sind 3 Spalten: Point, Time_min, Signal."
        )
    df["Point"] = pd.to_numeric(df["Point"], errors="coerce")
    df["Time_min"] = pd.to_numeric(df["Time_min"], errors="coerce")
    df["Signal"] = pd.to_numeric(df["Signal"], errors="coerce")
    df = df.dropna(subset=["Time_min", "Signal"]).reset_index(drop=True)
    return df


def time_to_volume_and_signal(df: pd.DataFrame, fluss_ml_min: float) -> Tuple[pd.Series, pd.Series]:
    t = df["Time_min"].astype(float)
    y = df["Signal"].astype(float)
    x_vol = t * fluss_ml_min  # V_E [mL]
    # Sortieren nach V_E (links kleiner)
    order = x_vol.argsort()
    return x_vol.iloc[order], y.iloc[order]


# ------------------------------
# Kalibration anwenden
# ------------------------------

def load_coeffs(json_path: Path) -> Tuple[float, float, float, float, float, float]:
    with open(json_path, "r") as f:
        d = json.load(f)
    try:
        a = float(d["a"])  # konstante
        b = float(d["b"])  # linear
        c = float(d["c"])  # quadratisch
        d3 = float(d["d"]) # kubisch
        ve_min = float(d.get("ve_min", float("nan")))
        ve_max = float(d.get("ve_max", float("nan")))
    except KeyError as e:
        raise KeyError(f"Schluessel {e} fehlt in {json_path}. Erwartet: a,b,c,d")
    return a, b, c, d3, ve_min, ve_max


def ve_to_mass(ve_ml: pd.Series, a: float, b: float, c: float, d3: float) -> pd.Series:
    ve = pd.Series(ve_ml).astype(float)
    logM = a + b*ve + c*(ve**2) + d3*(ve**3)
    M = np.power(10.0, logM)
    return pd.Series(M)


def mask_valid_range(ve: pd.Series, y: pd.Series, ve_min: float, ve_max: float) -> Tuple[pd.Series, pd.Series]:
    if not np.isfinite(ve_min) or not np.isfinite(ve_max):
        return ve, y
    m = (ve >= ve_min) & (ve <= ve_max)
    return ve[m], y[m]


# ------------------------------
# Plotten und Speichern
# ------------------------------

def plot_signal_vs_mass(M: pd.Series, y: pd.Series, src: Path) -> Path:
    out_path = src.with_suffix(".mass.pdf")
    # Sicherheit: Series + float
    M = pd.Series(M).astype(float)
    y = pd.Series(y).astype(float)
    # Nach Molmasse sortieren fuer sauberen Plot (aufsteigend)
    order = M.argsort()
    M = M.iloc[order]
    y = y.iloc[order]

    plt.figure()
    # Logarithmische x-Achse: Signal vs Molmasse
    plt.semilogx(M, y)
    ax = plt.gca()
    ax.invert_xaxis()  # SEC: groessere M links, kleinere rechts
    plt.xlabel("Molmasse M [Da] (log)")
    plt.ylabel("Signal")
    plt.title(src.stem)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


# Plot: Signal vs. Elutionsvolumen

def plot_signal_vs_volume(ve_ml: pd.Series, y: pd.Series, src: Path) -> Path:
    out_path = src.with_suffix(".ve.pdf")
    ve_ml = pd.Series(ve_ml).astype(float)
    y = pd.Series(y).astype(float)
    order = ve_ml.argsort()
    ve_ml = ve_ml.iloc[order]
    y = y.iloc[order]

    plt.figure()
    plt.plot(ve_ml, y)
    plt.xlabel("Elutionsvolumen V_E [mL]")
    plt.ylabel("Signal")
    plt.title(src.stem)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


# ------------------------------
# Main
# ------------------------------

def main():
    # --- Hier Pfade angeben ---
    coeff_path = Path("/Users/musamoin/Desktop/BA-HS25/sec/both_columns/cal/calibration_coeffs.json")
    base = Path("/Users/musamoin/Desktop/BA-HS25/sec/both_columns/samples")
    fluss = 0.8  # mL/min

    if not coeff_path.exists():
        print(f"Kalibrationsdatei nicht gefunden: {coeff_path}", file=sys.stderr)
        sys.exit(1)
    if not base.exists() or not base.is_dir():
        print(f"Pfad nicht gefunden oder kein Ordner: {base}", file=sys.stderr)
        sys.exit(1)

    a, b, c, d3, ve_min, ve_max = load_coeffs(coeff_path)

    csv_files = sorted([p for p in base.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        print(f"Keine CSV-Dateien in: {base}")
        sys.exit(0)

    print(f"Gefundene Sample-Dateien: {len(csv_files)}")
    ok, fail = 0, 0
    for csv_path in csv_files:
        try:
            df = read_csv_fixed(csv_path)
            ve, y = time_to_volume_and_signal(df, fluss)
            ve, y = mask_valid_range(ve, y, ve_min, ve_max)
            if len(ve) == 0:
                raise RuntimeError("Keine Punkte im gueltigen Kalibrationsbereich (V_E)")
            M = ve_to_mass(ve, a, b, c, d3)
            out_pdf = plot_signal_vs_mass(M, y, csv_path)
            print(f"✔ Gespeichert: {out_pdf}")
            ok += 1
        except Exception as e:
            print(f"Fehler bei {csv_path.name}: {e}", file=sys.stderr)
            fail += 1
    print(f"Fertig. Erfolgreich: {ok}, Fehlgeschlagen: {fail}")


if __name__ == "__main__":
    main()
