import re
from pathlib import Path

from typing import Dict, List, Tuple
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, ListedColormap
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"


# Pfad zum Ordner mit den exportierten Spektren
DATA_DIR = Path(
    "/Users/musamoin/Desktop/BA-HS25/sec/both_columns/samples/spectra/"
)


def parse_header(header_line: str) -> Tuple[float, str]:
    """Parst die erste Kommentarzeile.

    Erwartetes Format (Beispiel):
    #"UV (15.001 min) 251030_Nr1_EXP-001.d "

    Rueckgabe:
        (retentionszeit_min, basisdateiname)
    """
    # Kommentarzeichen und Anfuehrungszeichen entfernen
    cleaned = header_line.lstrip("#").strip().strip("\"")

    # Zuerst versuchen: UV-Header mit Retentionszeit, z.B.
    # "UV (15.001 min) 251030_Nr1_EXP-001.d"
    match_uv = re.search(r"UV\s*\(([^)]+)\s*min\)\s*(.+)$", cleaned)
    if match_uv:
        time_str = match_uv.group(1).strip()
        name_str = match_uv.group(2).strip()

        # Falls es sich um Blue/Green/Red-Spektren mit Zeitintervall handelt,
        # Retentionszeit komplett ignorieren und einfach 0.0 setzen.
        if any(kw in name_str for kw in ("Blue", "Green", "Red")):
            base_name = Path(name_str).stem
            rt_min = 0.0
            return rt_min, base_name

        # Standardfall: echte Retentionszeit als float (Punkt-Notation erwartet)
        try:
            rt_min = float(time_str.replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"Kann Retentionszeit nicht in float umwandeln: {time_str!r}") from exc

        base_name = Path(name_str).stem
        return rt_min, base_name

    # Zweiter Fall: DAD-Header ohne explizite Retentionszeit, z.B.
    # "DAD1 - B:Sig=254,4  Ref=360,100 251030_Nr1_EXP-001.d"
    match_dad = re.search(r"DAD\d+\s*-\s*B:Sig=[^ ]+\s+Ref=[^ ]+\s+(.+)$", cleaned)
    if match_dad:
        name_str = match_dad.group(1).strip()
        base_name = Path(name_str).stem
        # Keine Zeit im Header -> auf 0.0 setzen, damit Spektren trotzdem geplottet werden
        rt_min = 0.0
        return rt_min, base_name

    # Wenn weder UV- noch DAD-Format erkannt wird:
    raise ValueError(f"Header-Zeile unklar, kann nicht geparst werden: {header_line!r}")


def read_spectrum_file(path: Path) -> Tuple[float, str, List[float], List[float]]:
    """Liest ein einzelnes Spektrum aus einer TXT-Datei.

    Rueckgabe:
        (retentionszeit_min, basisdateiname, wellenlaengen_nm, signal_mau)
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        try:
            header = next(f).strip()
        except StopIteration:
            raise ValueError(f"Datei {path} ist leer.")

        rt_min, base_name = parse_header(header)

        # Naechste Zeile ist der Spaltenkopf, die ueberspringen wir
        try:
            next(f)
        except StopIteration:
            raise ValueError(f"Datei {path} enthaelt keine Datenzeilen.")

        wavelengths: List[float] = []
        intensities: List[float] = []

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 3:
                # Zeile hat nicht die erwartete Struktur, einfach ueberspringen
                continue
            try:
                wl = float(parts[1].replace(",", "."))
                intensity = float(parts[2].replace(",", "."))
            except ValueError:
                # Falls mal Mist in einer Zeile steht, ueberspringen
                continue

            wavelengths.append(wl)
            intensities.append(intensity)

    return rt_min, base_name, wavelengths, intensities


def collect_spectra_by_chromatogram() -> Dict[str, List[Tuple[float, List[float], List[float]]]]:
    """Liest alle TXT-Dateien ein und gruppiert Spektren nach Chromatogramm.

    Rueckgabe:
        dict[chrom_name] = Liste von (retentionszeit, wellenlaengen, intensitaeten)
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Datenordner existiert nicht: {DATA_DIR}")

    spectra_by_chrom: Dict[str, List[Tuple[float, List[float], List[float]]]] = {}

    txt_files = sorted(DATA_DIR.glob("*.txt")) + sorted(DATA_DIR.glob("*.TXT"))

    if not txt_files:
        print(f"Keine .txt oder .TXT Dateien im Datenordner gefunden: {DATA_DIR}")
        return spectra_by_chrom

    for txt_path in txt_files:
        try:
            rt_min, base_name, wl, inten = read_spectrum_file(txt_path)
        except Exception as exc:  # bewusst breit, damit ein kaputtes File nicht alles killt
            print(f"Warnung: Konnte {txt_path.name} nicht einlesen: {exc}")
            continue

        spectra_by_chrom.setdefault(base_name, []).append((rt_min, wl, inten))

    # Spektren pro Chromatogramm nach Retentionszeit sortieren
    for base_name, spectra_list in spectra_by_chrom.items():
        spectra_list.sort(key=lambda x: x[0])

    return spectra_by_chrom


def make_overlay_plots(spectra_by_chrom: Dict[str, List[Tuple[float, List[float], List[float]]]]) -> None:
    """Erstellt fuer jedes Chromatogramm einen Overlayplot aller Spektren.

    Farbskala: je weiter die Retentionszeit fortgeschritten ist, desto dunkler
    (ueber eine kontinuierliche Colormap abgebildet).
    """
    plot_index = 0

    if not spectra_by_chrom:
        print("Keine Spektren gefunden – pruef den Pfad im Skript.")
        return

    # Ausgabeordner fuer Plots (direkt im Datenordner)
    out_dir = DATA_DIR / "overlay_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Blues-Cmap so zuschneiden, dass die ganz hellen (fast weissen) Toene wegfallen
    base_cmap = get_cmap("Blues")
    colors = base_cmap(np.linspace(0.3, 1.0, 256))  # unterste 20 % abschneiden
    cmap = ListedColormap(colors)

    for chrom_name, spectra_list in spectra_by_chrom.items():
        # Farbnamen-Spektren (Blue/Green/Red) hier überspringen, werden separat geplottet
        if any(kw in chrom_name for kw in ("Blue", "Green", "Red")):
            continue

        # VE-Bereich fix setzen: 11.154–20.377 mL und Farbverlauf
        vmin, vmax = 11.154, 20.377
        norm = Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(16 / 2.54, 7 / 2.54))  # 16 cm x 10 cm

        for rt, wl, inten in spectra_list:
            ve = rt * 0.8  # Elutionsvolumen in mL
            color = cmap(norm(ve))
            ax.plot(wl, inten, color=color, linewidth=0.8)

        ax.set_ylabel("Absorbanz $A$ / $10^{-3}$")
        ax.set_xlabel("Wellenlänge $\\lambda$ / nm")

        ax.set_xlim(220, 280)
        ax.set_ylim(-1,10)
        ax.axvline(260, color="red", linestyle="--", linewidth=0.8)
        # Y-Min + kleiner Offset (5 % der Spannweite) und halbtransparente Box
        ylim = ax.get_ylim()
        offset = 0.05 * (ylim[1] - ylim[0])
        ax.text(
            261,
            ylim[0] + offset,
            r"$\lambda_{\max} = 260\,\mathrm{nm}$",
            color="red",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
        )
        ax.grid(True)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$V_E$ / mL")

        out_path = out_dir / f"{chrom_name}_overlay.pdf"
        fig.tight_layout()
        fig.savefig(out_path)

        print(f"Plot gespeichert: {out_path}")

        # Interaktive Anzeige
        plt.show()
        plot_index += 1
        plt.close(fig)


def make_rgb_overlays() -> None:
    """Erstellt Overlays aus Blue/Green/Red-Spektren (falls vorhanden).

    - Je Stichprobe (z.B. 251030_Nr1) werden die Dateien *_Blue-01, *_Green-01, *_Red-01
      zusammen geplottet.
    - Linienfarben entsprechen dem Farbnamen.
    - In der Legende steht die umgerechnete Elutionsvolumen-Range aus dem Header.
    """
    plot_index = 0

    txt_files = sorted(DATA_DIR.glob("*.txt")) + sorted(DATA_DIR.glob("*.TXT"))

    # Mapping: prefix (z.B. 251030_Nr1) -> {"Blue": Path, "Green": Path, "Red": Path}
    groups: Dict[str, Dict[str, Path]] = {}
    for path in txt_files:
        stem = path.stem
        for color in ("Blue", "Green", "Red"):
            token = f"_{color}-"
            if token in stem:
                prefix = stem.split(token)[0]
                groups.setdefault(prefix, {})[color] = path

    if not groups:
        return

    out_dir = DATA_DIR / "overlay_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    color_map = {"Blue": "blue", "Green": "green", "Red": "red"}

    for prefix, files in groups.items():
        # Mindestens ein Spektrum vorhanden, wir plotten alle, die da sind
        fig, ax = plt.subplots(figsize=(16 / 2.54, 7 / 2.54))
        legend_entries: List[str] = []

        for color_name, path in sorted(files.items()):
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                try:
                    header = next(f).strip()
                except StopIteration:
                    continue

                cleaned = header.lstrip("#").strip().strip("\"")
                # Zeitintervall aus dem Header holen, z.B. "UV (13.675-29.061 min) ..."
                m = re.search(r"UV\s*\(([^-]+)-([^)]+)\s*min\)\s*(.+)$", cleaned)
                t_start = t_end = 0.0
                if m:
                    t_start_str = m.group(1).strip()
                    t_end_str = m.group(2).strip()
                    try:
                        t_start = float(t_start_str.replace(",", "."))
                        t_end = float(t_end_str.replace(",", "."))
                    except ValueError:
                        t_start = t_end = 0.0

                # Spaltenkopf überspringen
                try:
                    next(f)
                except StopIteration:
                    continue

                wls: List[float] = []
                ints: List[float] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(";")
                    if len(parts) < 3:
                        continue
                    try:
                        wl = float(parts[1].replace(",", "."))
                        inten = float(parts[2].replace(",", "."))
                    except ValueError:
                        continue
                    wls.append(wl)
                    ints.append(inten)

            if not wls:
                continue

            ve_min = t_start * 0.8
            ve_max = t_end * 0.8

            ax.plot(wls, ints, linewidth=0.8, color=color_map.get(color_name, None), label=None)

            german = {"Blue": "Blau", "Green": "Grün", "Red": "Rot"}
            legend_entries.append(
                rf"{german.get(color_name, color_name)}: $V_E = {ve_min:.3f}$ bis ${ve_max:.3f}\,\mathrm{{mL}}$"
            )

        if not legend_entries:
            plt.close(fig)
            continue

        ax.set_ylabel("Absorbanz $A$ / $10^{-3}$")
        ax.set_xlabel("Wellenlänge $\\lambda$ / nm")

        ax.set_xlim(220, 280)
        ax.set_ylim(-1,10)
        ax.axvline(260, color="red", linestyle="--", linewidth=0.8)
        ylim = ax.get_ylim()
        offset = 0.05 * (ylim[1] - ylim[0])
        ax.text(
            261,
            ylim[0] + offset,
            r"$\lambda_{\max} = 260\,\mathrm{nm}$",
            color="red",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"),
        )
        ax.grid(True)

        ax.legend(legend_entries, loc="best")

        out_path = out_dir / f"{prefix}_RGB_overlay.pdf"
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"Plot gespeichert: {out_path}")

        plt.show()
        plot_index += 1
        plt.close(fig)


def main() -> None:
    spectra_by_chrom = collect_spectra_by_chromatogram()
    make_overlay_plots(spectra_by_chrom)
    make_rgb_overlays()


if __name__ == "__main__":
    main()