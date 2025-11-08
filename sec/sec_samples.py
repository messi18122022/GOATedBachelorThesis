#!/usr/bin/env python3
"""
Markiert SEC-CSV-Dateien anhand der Kalibrationsgrenzen (Ve_min, Ve_max) als
"gruen" bzw. "rot" im Plot dargestellt. Es werden keine CSVs geschrieben.

Einstellungen (im Skript oben konfigurierbar):
- BASE_PATH : Ordner mit CSV-Dateien (Agilent-DAD-Format: '#...'-Kommentarzeilen
              gefolgt von Datenzeilen ohne '#': Point, Time_min, Signal)
- FLUSS_ML_MIN : Flussrate in mL/min (für Umrechnung t[min] -> Ve[mL])
- CAL_JSON  : Pfad zu calibration_coeffs.json (enthält Ve_min_mL und Ve_max_mL)
- INPLACE   : True = im selben Ordner schreiben (mit Suffix)

Hinweis: Es werden **keine neuen CSVs** erzeugt. Stattdessen wird der Plot
als PDF im selben Ordner wie die jeweilige CSV gespeichert. Die Linie des
Chromatogramms ist **grün** innerhalb [Ve_min, Ve_max] und **rot** außerhalb.
"""
from __future__ import annotations

from pathlib import Path
import json
import csv
import sys
from typing import Tuple
import re

import pandas as pd
import numpy as np

# --------------------------------------------------
# Matplotlib: LaTeX + siunitx aktivieren, Gitterlinien hell und dünn
import matplotlib.pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 9,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{siunitx}',
    'grid.color': '#cccccc',   # hellgrau
    'grid.alpha': 0.6,         # etwas transparenter
    'grid.linewidth': 0.4      # dünnere Linien
})

# LaTeX-Fallback robust einbauen
import shutil
try:
    if not shutil.which("latex"):
        raise RuntimeError("LaTeX nicht verfügbar")
except Exception:
    # Fallback ohne LaTeX, damit die Plots nicht leer/fehlschlagen
    plt.rcParams.update({
        'text.usetex': False,
        'font.size': 9,
        'font.family': 'serif',
    })

# Peak-Detektionsparameter analog zum anderen Skript
PEAK_MIN_Y = 2.0
PEAK_PROMINENCE = 0.5
PEAK_MIN_SEP = 5
SMOOTH_WINDOW = 9

# ==============================
# KONFIGURATION (hier anpassen)
# ==============================
BASE_PATH: Path = Path("/Users/musamoin/Desktop/BA-HS25/sec/both_columns/samples/")
FLUSS_ML_MIN: float = 0.8  # mL/min
CAL_JSON: Path = Path("/Users/musamoin/Desktop/BA-HS25/sec/both_columns/cal/calibration_coeffs.json")
INPLACE: bool = True

# ------------------------------
# CSV-Einlesen robust (Agilent-DAD: 2 Kommentarzeilen, dann Daten)
# ------------------------------

def _sniff_delimiter(text_sample: str, default: str = ",") -> str:
    try:
        dialect = csv.Sniffer().sniff(text_sample, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        return default


def read_csv_agilent(path: Path) -> pd.DataFrame:
    """Liest CSV im Agilent-DAD-Layout: Datenzeilen mit drei Spalten
    (Point, Time_min, Signal). Kommentarzeilen beginnen mit '#'.
    """
    sample = path.read_text(encoding="latin-1", errors="ignore")[:4096]
    sep = _sniff_delimiter(sample)
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=["Point", "Time_min", "Signal"],
        sep=sep,
        engine="python",
        encoding="latin-1",
        skip_blank_lines=True,
    )
    if df.shape[1] != 3:
        raise RuntimeError(
            f"Unerwartetes Spaltenformat in {path.name}. Erwartet sind 3 Spalten: Point, Time_min, Signal."
        )

    # numerisch machen; Dezimaltrennzeichen-Korrektur falls noetig
    def _to_float_col(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = (
                s.astype(str)
                .str.replace("\u00A0", " ", regex=False)
                .str.strip()
            )
            # Wenn viele Kommas vorkommen, ersetze Komma-Dezimalen
            try:
                if (s.str.contains(r",\d").mean() or 0) > 0.3:
                    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            except Exception:
                pass
        return pd.to_numeric(s, errors="coerce")

    df["Point"] = _to_float_col(df["Point"]).astype("Int64")
    df["Time_min"] = _to_float_col(df["Time_min"])  # min
    df["Signal"] = _to_float_col(df["Signal"])     # RU
    df = df.dropna(subset=["Time_min", "Signal"]).reset_index(drop=True)
    return df


# ------------------------------
# Kernlogik: Ve berechnen
# ------------------------------

def compute_Ve(df: pd.DataFrame, fluss_ml_min: float) -> pd.Series:
    """Elutionsvolumen Ve [mL] = Time_min [min] * Flussrate [mL/min]."""
    return pd.Series(df["Time_min"], copy=False).astype(float) * float(fluss_ml_min)


# ------------------------------
# Peak-Detektion (wie im anderen Skript)
# ------------------------------

def _find_peak_indices(y: pd.Series,
                       min_y: float = PEAK_MIN_Y,
                       window: int | None = None,
                       prominence: float = PEAK_PROMINENCE,
                       min_sep: int = PEAK_MIN_SEP,
                       smooth_window: int = SMOOTH_WINDOW) -> list:
    y = pd.Series(y).astype(float).reset_index(drop=True)
    n = len(y)
    if n < 3:
        return []

    sw = max(3, smooth_window)
    if sw % 2 == 0:
        sw += 1
    ys = y.rolling(sw, center=True, min_periods=1).mean()

    dy = ys.diff()
    candidates = []
    i = 1
    while i < n - 1:
        if dy.iloc[i] > 0 and dy.iloc[i + 1] <= 0:
            candidates.append(i)
            i += 1
            continue
        if dy.iloc[i] > 0 and abs(dy.iloc[i + 1]) < 1e-12:
            j = i + 1
            while j < n - 1 and abs(dy.iloc[j]) < 1e-12:
                j += 1
            if j < n - 1 and dy.iloc[j] < 0:
                peak_idx = (i + j) // 2
                candidates.append(peak_idx)
                i = j + 1
                continue
        i += 1

    if not candidates:
        return []

    def prominence_at(k: int) -> float:
        pk = ys.iloc[k]
        left = k - 1
        last = pk
        while left > 0 and ys.iloc[left] <= last:
            last = ys.iloc[left]
            left -= 1
        left_min = ys.iloc[left + 1]
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

    peaks.sort(key=lambda t: (t[2], t[1]), reverse=True)
    selected = []
    for idx, prom, height in peaks:
        if all(abs(idx - s) >= (window if window else min_sep) and abs(idx - s) >= min_sep for s in selected):
            selected.append(idx)
    selected.sort()
    return selected


def _extract_peak_volumes(x: pd.Series, y: pd.Series) -> list[float]:
    md = max(5, len(y) // 200)
    idx = _find_peak_indices(y, min_y=PEAK_MIN_Y, window=md,
                             prominence=PEAK_PROMINENCE, min_sep=PEAK_MIN_SEP,
                             smooth_window=SMOOTH_WINDOW)
    if not idx:
        return []
    xv = x.iloc[idx].astype(float).tolist()
    xv.sort()
    return xv



# ------------------------------
# Plot analog zum anderen Skript, aber mit zweifarbiger Linie
# ------------------------------

def plot_and_save(x: pd.Series, y: pd.Series, src: Path) -> Path:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    # Farbsegmentierung: grün innerhalb [Ve_min, Ve_max], rot außerhalb.
    # Wir lesen die Grenzen aus der JSON erneut, damit plot_and_save alleine funktioniert.
    try:
        ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
    except Exception:
        # Fallback: alles rot, falls Grenzen nicht lesbar
        ve_min, ve_max = -np.inf, -np.inf

    inside = (x >= ve_min) & (x <= ve_max)
    y_inside = y.where(inside, np.nan)
    y_outside = y.where(~inside, np.nan)

    # Sichtbarkeits-Flags für Fallback
    has_inside = np.isfinite(y_inside).any()
    has_outside = np.isfinite(y_outside).any()

    out_path = src.with_suffix(".pdf")
    plt.figure(figsize=(16/2.54, 6.5/2.54))
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.gcf().set_facecolor('white')
    # Basis: komplette Kurve in hellgrau, damit immer etwas sichtbar ist
    plt.plot(x, y, linewidth=0.8, alpha=0.7)
    # Overlay: Rot außerhalb, Grün innerhalb
    if has_outside:
        plt.plot(x, y_outside, color='red', linewidth=1.2)
    if has_inside:
        plt.plot(x, y_inside, color='green', linewidth=1.2)

    plt.grid(True)

    plt.xlabel(r"Elutionsvolumen $V_E$ / \si{\milli\liter}")
    plt.ylabel(r"Absorbanz $A$ / $10^{-3}$")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def load_calibration_bounds(cal_path: Path) -> Tuple[float, float]:
    if not cal_path.exists():
        raise FileNotFoundError(f"Kalibrationsdatei nicht gefunden: {cal_path}")
    with cal_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        ve_min = float(data["Ve_min_mL"])
        ve_max = float(data["Ve_max_mL"])
    except KeyError as e:
        raise KeyError("Ve_min_mL/Ve_max_mL fehlen in calibration_coeffs.json") from e
    if not np.isfinite(ve_min) or not np.isfinite(ve_max):
        raise ValueError("Ungueltige Ve_min/Ve_max in calibration_coeffs.json")
    if ve_min > ve_max:
        ve_min, ve_max = ve_max, ve_min
    return ve_min, ve_max


def load_calibration_coeffs(cal_path: Path) -> tuple[float, float, float, float]:
    """Liest a,b,c,d aus calibration_coeffs.json."""
    with cal_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        a = float(data["a"])
        b = float(data["b"])
        c = float(data["c"])
        d = float(data["d"])
    except KeyError as e:
        raise KeyError("a/b/c/d fehlen in calibration_coeffs.json") from e
    return a, b, c, d


def ve_to_M(ve: pd.Series, a: float, b: float, c: float, d: float) -> pd.Series:
    """Berechnet M aus Ve gemäß log10(M) = a + b*Ve + c*Ve^2 + d*Ve^3."""
    ve = pd.Series(ve, copy=False).astype(float)
    logM = a + b*ve + c*(ve**2) + d*(ve**3)
    M = np.power(10.0, logM)
    return pd.Series(M, index=ve.index)


# --------------------------------------------------
# M_n und M_w Berechnung aus Absorbanz (im grünen Bereich)
def compute_Mn_Mw_from_absorbance(ve: pd.Series, absorbance: pd.Series) -> tuple[float | None, float | None, int]:
    """Berechnet M_n und M_w aus Absorbanz (c_i ∝ A) vs. Molmasse im grünen Bereich.
    Formeln:
      M_n = (∑ c_i) / (∑ c_i / M_i)
      M_w = (∑ c_i M_i) / (∑ c_i)
    Rückgabe: (Mn, Mw, n_inside).
    """
    ve = pd.Series(ve).astype(float)
    A = pd.Series(absorbance).astype(float)

    # Grenzen & Koeffizienten laden
    ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
    a, b, c, d = load_calibration_coeffs(CAL_JSON.expanduser().resolve())

    inside = (ve >= ve_min) & (ve <= ve_max)
    if not inside.any():
        return None, None, 0

    M = ve_to_M(ve, a, b, c, d)

    # Filter: nur gültige, nichtnegative Werte im grünen Fenster
    mask = inside & np.isfinite(M) & (M > 0) & np.isfinite(A) & (A >= 0)
    if not mask.any():
        return None, None, 0

    Mi = M[mask].to_numpy(dtype=float)
    ci = A[mask].to_numpy(dtype=float)

    sum_c = float(np.sum(ci))
    sum_c_over_M = float(np.sum(ci / Mi)) if np.all(Mi > 0) else np.nan
    sum_cM = float(np.sum(ci * Mi))

    Mn = (sum_c / sum_c_over_M) if (sum_c_over_M and np.isfinite(sum_c_over_M)) else None
    Mw = (sum_cM / sum_c) if (sum_c and np.isfinite(sum_c)) else None
    return Mn, Mw, int(mask.sum())

# --------------------------------------------------
# Hilfsfunktion: schöne LaTeX-Schreibweise für Zahlen (a · 10^{b})

def _format_sci_tex(value: float, sigfig: int = 3, with_unit: bool = True) -> str:
    if not np.isfinite(value) or value <= 0:
        return "--"
    # Interpret `sigfig` here as number of decimal places for plain decimal output
    fmt = f"{{:.{sigfig}f}}"
    s = fmt.format(float(value)).rstrip('0').rstrip('.')
    if with_unit:
        if plt.rcParams.get('text.usetex', False):
            return rf"{s}\,\si{{\kilo\gram\per\mol}}"
        else:
            return f"{s} kg/mol"
    return s


def plot_mass_and_save(ve: pd.Series, y: pd.Series, src: Path) -> Path:
    """Plot: Signal (y) gegen Molmasse M (x). Es werden nur grüne Punkte (Ve_min..Ve_max) geplottet."""
    ve = pd.Series(ve).astype(float)
    y = pd.Series(y).astype(float)

    # Grenzen und Koeffizienten laden
    ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
    a, b, c, d = load_calibration_coeffs(CAL_JSON.expanduser().resolve())

    inside = (ve >= ve_min) & (ve <= ve_max)
    M = ve_to_M(ve, a, b, c, d)
    M_kg = M / 1000.0

    # Nur grüne Punkte verwenden
    x_inside = M_kg.where(inside, np.nan)
    y_inside = y.where(inside, np.nan)
    has_inside = np.isfinite(x_inside).any() and np.isfinite(y_inside).any()

    out_path = src.with_name(src.stem + "_M.pdf")
    plt.figure(figsize=(16/2.54, 6.5/2.54))
    ax = plt.gca()
    if has_inside:
        ax.plot(x_inside, y_inside, color='black')
    else:
        # Nichts im grünen Bereich – Hinweis einblenden
        ax.plot([], [])
        ax.text(0.5, 0.5, 'Kein Datenpunkt im Kalibrationsfenster', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)

    # Mn, Mw, PDI berechnen (nur grüne Punkte) und als Legende oben rechts anzeigen
    Mn, Mw, n_used = compute_Mn_Mw_from_absorbance(ve, y)
    legend_items = []
    legend_labels = []
    if Mn is not None and Mw is not None and Mn != 0:
        pdi = Mw / Mn
        # vertikale Linien einzeichnen
        mn_line = ax.axvline(Mn/1000.0, linestyle='--', linewidth=1.2, color='C0')
        mw_line = ax.axvline(Mw/1000.0, linestyle='--', linewidth=1.2, color='C3')
        legend_items.extend([mn_line, mw_line])
        mn_str = _format_sci_tex(Mn/1000.0, sigfig=2, with_unit=True)
        mw_str = _format_sci_tex(Mw/1000.0, sigfig=2, with_unit=True)
        legend_labels.extend([rf"$M_n = {mn_str}$", rf"$M_w = {mw_str}$"])
        legend_items.append(plt.Line2D([0],[0], color='none'))  # Platzhalter für PDI-Zeile
        legend_labels.append(rf"PDI $= {pdi:.3f}$")
        ax.legend(legend_items, legend_labels, loc='upper right', framealpha=0.8, frameon=True)
    else:
        ax.text(0.98, 0.98, 'Keine Werte (grüner Bereich leer)', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

    ax.grid(True)
    ax.set_xlabel(r"Molmasse $M$ / \si{\kilo\gram\per\mol}")
    ax.set_ylabel(r"Absorbanz $A$ / $10^{-3}$")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


# --------------------------------------------------
# Zusätzlicher Zoom-Plot für Molmasse (x-Achse 0..500000)
def plot_mass_and_save_zoom(ve: pd.Series, y: pd.Series, src: Path) -> Path:
    ve = pd.Series(ve).astype(float)
    y = pd.Series(y).astype(float)
    ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
    a, b, c, d = load_calibration_coeffs(CAL_JSON.expanduser().resolve())
    inside = (ve >= ve_min) & (ve <= ve_max)
    M = ve_to_M(ve, a, b, c, d)
    M_kg = M / 1000.0
    x_inside = M_kg.where(inside, np.nan)
    y_inside = y.where(inside, np.nan)

    out_path = src.with_name(src.stem + "_M_zoom.pdf")
    plt.figure(figsize=(16/2.54, 6.5/2.54))
    ax = plt.gca()
    has_inside = np.isfinite(x_inside).any() and np.isfinite(y_inside).any()
    if has_inside:
        ax.plot(x_inside, y_inside, color='black')
    else:
        ax.plot([], [])
        ax.text(0.5, 0.5, 'Kein Datenpunkt im Kalibrationsfenster', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)

    # Legende oben rechts + Linien
    Mn, Mw, n_used = compute_Mn_Mw_from_absorbance(ve, y)
    legend_items = []
    legend_labels = []
    if Mn is not None and Mw is not None and Mn != 0:
        pdi = Mw / Mn
        mn_line = ax.axvline(Mn/1000.0, linestyle='--', linewidth=1.2, color='C0')
        mw_line = ax.axvline(Mw/1000.0, linestyle='--', linewidth=1.2, color='C3')
        legend_items.extend([mn_line, mw_line])
        mn_str = _format_sci_tex(Mn/1000.0, sigfig=2, with_unit=True)
        mw_str = _format_sci_tex(Mw/1000.0, sigfig=2, with_unit=True)
        legend_labels.extend([rf"$M_n = {mn_str}$", rf"$M_w = {mw_str}$"])
        legend_items.append(plt.Line2D([0],[0], color='none'))
        legend_labels.append(rf"PDI $= {pdi:.3f}$")
        ax.legend(legend_items, legend_labels, loc='upper right', framealpha=0.8, frameon=True)
    else:
        ax.text(0.98, 0.98, 'Keine Werte (grüner Bereich leer)', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))

    ax.grid(True)
    ax.set_xlim(0, 300)
    ax.set_xlabel(r"Molmasse $M$ /  \si{\kilo\gram\per\mol}")
    ax.set_ylabel(r"Absorbanz $A$ / $10^{-3}$")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path

# --------------------------------------------------
# Overlay: Absorbanz vs. Molmasse (nur Zoom-Bereich, nur grüne Punkte)

def plot_mass_overlay_zoom(csv_files: list[Path]) -> Path:
    out_path = (BASE_PATH / "overlay_M_zoom.pdf").expanduser().resolve()
    try:
        ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
        a, b, c, d = load_calibration_coeffs(CAL_JSON.expanduser().resolve())
    except Exception as e:
        # Falls Kalibration fehlt, abbrechen
        raise RuntimeError(f"Kalibration konnte nicht geladen werden: {e}")

    plt.figure(figsize=(16/2.54, 6.5/2.54))
    ax = plt.gca()
    any_plotted = False

    for src in csv_files:
        try:
            df = read_csv_agilent(src)
            ve = compute_Ve(df, float(FLUSS_ML_MIN))
            y = pd.Series(df["Signal"]).astype(float)
            inside = (ve >= ve_min) & (ve <= ve_max)
            if not inside.any():
                continue
            M = ve_to_M(ve, a, b, c, d)
            M_kg = M / 1000.0
            x_inside = pd.Series(M_kg).where(inside, np.nan)
            y_inside = y.where(inside, np.nan)
            if np.isfinite(x_inside).any() and np.isfinite(y_inside).any():
                # Legendenlabel: nur EXP-Id im \texttt{}-Stil
                stem = src.stem
                m = re.search(r"EXP[-_]?(\d+)", stem, flags=re.IGNORECASE)
                if m:
                    exp_id = f"EXP-{m.group(1).zfill(3)}"
                else:
                    exp_id = stem
                label_txt = rf"\texttt{{{exp_id}}}" if plt.rcParams.get('text.usetex', False) else exp_id
                ax.plot(x_inside, y_inside, linewidth=1.0, label=label_txt)
                any_plotted = True
        except Exception:
            # einzelne Datei ueberspringen
            continue

    if not any_plotted:
        ax.plot([], [])
        ax.text(0.5, 0.5, 'Keine gueltigen Daten im Kalibrationsfenster', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)

    ax.grid(True)
    ax.set_xlim(0, 300)
    ax.set_xlabel(r"Molmasse $M$ /  \si{\kilo\gram\per\mol}")
    ax.set_ylabel(r"Absorbanz $A$ / $10^{-3}$")
    if any_plotted:
        ax.legend(loc='upper right', framealpha=0.85, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def main() -> None:
    base: Path = BASE_PATH.expanduser().resolve()
    fluss: float = float(FLUSS_ML_MIN)
    if not base.exists() or not base.is_dir():
        print(f"Pfad nicht gefunden oder kein Ordner: {base}", file=sys.stderr)
        sys.exit(1)

    try:
        ve_min, ve_max = load_calibration_bounds(CAL_JSON.expanduser().resolve())
    except Exception as e:
        print(f"Kalibration konnte nicht gelesen werden: {e}", file=sys.stderr)
        sys.exit(1)

    # no output directory creation needed, PDFs saved next to CSVs

    csv_files = sorted([p for p in base.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        print(f"Keine CSV-Dateien in: {base}")
        sys.exit(0)

    ok = 0
    fail = 0
    for src in csv_files:
        try:
            df = read_csv_agilent(src)
            ve = compute_Ve(df, fluss)

            # Debug-Infos zur Sichtbarkeit
            inside_mask = (ve >= ve_min) & (ve <= ve_max)
            n_total = int(len(ve))
            n_inside = int(np.count_nonzero(inside_mask))
            n_outside = n_total - n_inside
            y_non_nan = int(np.count_nonzero(np.isfinite(df["Signal"].to_numpy(dtype=float))))
            print(f"   Punkte: {n_total}, inside: {n_inside}, outside: {n_outside}, y-nonNaN: {y_non_nan}")

            # M_n und M_w aus Absorbanz vs. M (nur grüner Bereich)
            Mn, Mw, n_used = compute_Mn_Mw_from_absorbance(ve, df["Signal"])  # Absorbanz ~ Konzentration
            if Mn is not None and Mw is not None:
                pdi = Mw / Mn if Mn else float('nan')
                print(f"   M_n = {Mn/1000.0:.3e} kg/mol, M_w = {Mw/1000.0:.3e} kg/mol, PDI = {pdi:.3f}, Punkte(gruen) = {n_used}")
            else:
                print(f"   M_n/M_w nicht berechenbar (keine gueltigen Punkte im Kalibrationsfenster)")

            # Plot analog zum anderen Skript als PDF (gleicher Pfad wie Original-CSV)
            pdf_path = plot_and_save(ve, df["Signal"], src)
            pdf_mass_path = plot_mass_and_save(ve, df["Signal"], src)
            pdf_mass_zoom_path = plot_mass_and_save_zoom(ve, df["Signal"], src)
            ok += 1
            print(f"✔ Plot: {pdf_path}")
            print(f"✔ Plot (M): {pdf_mass_path}")
            print(f"✔ Plot (M Zoom): {pdf_mass_zoom_path}")
        except Exception as e:
            fail += 1
            print(f"Fehler bei {src.name}: {e}", file=sys.stderr)

    # Overlay-Plot fuer alle Proben (Absorbanz vs. Molmasse, Zoom)
    try:
        overlay_pdf = plot_mass_overlay_zoom(csv_files)
        print(f"✔ Overlay (M Zoom): {overlay_pdf}")
    except Exception as e:
        print(f"Overlay-Plot fehlgeschlagen: {e}", file=sys.stderr)

    print(f"Fertig. Erfolgreich: {ok}, Fehlgeschlagen: {fail}")


if __name__ == "__main__":
    main()