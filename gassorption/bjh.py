#!/usr/bin/env python3
"""BJH-Auswertung fuer EXP-029

Liest das von NovaWin exportierte TXT-File
"EXP-029 (BJH Pore Size Distribution  Adsorption ).txt" ein
und erzeugt zwei Plots (Oberflaeche und Volumen) mit
logarithmischer x-Achse. Die Plots werden als PDF in
"/Users/musamoin/Desktop/BA-HS25/gassorption/" gespeichert.
"""

from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Pfade und Dateinamen
# ---------------------------------------------------------
BASE_DIR = Path("/Users/musamoin/Desktop/BA-HS25/gassorption/")
TXT_FILENAME = "EXP-029 (BJH Pore Size Distribution  Adsorption ).txt"
INPUT_FILE = BASE_DIR / TXT_FILENAME


def load_bjh_file(path: Path):
    """Liest das Quantachrome-BJH-TXT und gibt ein Dict mit Listen zurueck.

    Erwartete Tabellenstruktur (7 Spalten):
        Diameter (nm)
        Pore Volume (cc/g)
        Pore Surface Area (m^2/g)
        dV(d) (cc/nm/g)
        dS(d) (m^2/nm/g)
        dV(log d) (cc/g)
        dS(log d) (m^2/g)
    """

    lines = path.read_text(encoding="latin-1").splitlines()

    diameter = []
    pore_volume = []
    pore_surface = []
    dV_logd = []
    dS_logd = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Nur Zeilen mit genau 7 numerischen Spalten verwenden
        # (das sind die eigentlichen BJH-Datenzeilen)
        if s[0].isdigit() or (s[0] == "-" and len(s) > 1 and s[1].isdigit()):
            parts = s.split()
            if len(parts) == 7:
                d, pv, ps, dv_d, ds_d, dv_logd_val, ds_logd_val = map(float, parts)
                diameter.append(d)
                pore_volume.append(pv)
                pore_surface.append(ps)
                dV_logd.append(dv_logd_val)
                dS_logd.append(ds_logd_val)

    if not diameter:
        raise ValueError(
            "Keine Datensaetze gefunden – stimmt der Dateiname und das Format des TXT-Files?"
        )

    return {
        "diameter_nm": diameter,
        "pore_volume_cc_g": pore_volume,
        "pore_surface_area_m2_g": pore_surface,
        "dV_logd_cc_g": dV_logd,
        "dS_logd_m2_g": dS_logd,
    }


def plot_volume(data: dict, outpath: Path) -> None:
    """Plot kumulatives Porenvolumen und dV/d(log d) vs. Porendurchmesser."""

    x = data["diameter_nm"]
    y_cum = data["pore_volume_cc_g"]
    y_diff = data["dV_logd_cc_g"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # kumulatives Volumen (rot)
    ax1.plot(x, y_cum, "o-", label="kumulatives Porenvolumen", color="red")
    ax1.set_xscale("log")
    ax1.set_xticks([3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40])
    ax1.set_xlim(0,40)
    ax1.set_xticklabels(["3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40"])
    ax1.set_xlabel("Porendurchmesser / nm")
    ax1.set_ylabel("Kumulatives Porenvolumen / (cc g$^{-1}$)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # dV/d(log d) (blau), zweite y-Achse
    ax2 = ax1.twinx()
    ax2.plot(x, y_diff, "s-", label="dV/d(log d)", color="blue")
    ax2.set_ylabel(r"$dV/d(\log d)$ / (cc g$^{-1}$)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_surface(data: dict, outpath: Path) -> None:
    """Plot kumulative Oberflaeche und dS/d(log d) vs. Porendurchmesser."""

    x = data["diameter_nm"]
    y_cum = data["pore_surface_area_m2_g"]
    y_diff = data["dS_logd_m2_g"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # kumulative Oberflaeche (rot)
    ax1.plot(x, y_cum, "o-", label="kumulative Oberflaeche", color="red")
    ax1.set_xscale("log")
    ax1.set_xticks([3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40])
    ax1.set_xticklabels(["3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40"])
    ax1.set_xlabel("Porendurchmesser / nm")
    ax1.set_ylabel("Kumulative Oberflaeche / (m$^2$ g$^{-1}$)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # dS/d(log d) (blau), zweite y-Achse
    ax2 = ax1.twinx()
    ax2.plot(x, y_diff, "s-", label="dS/d(log d)", color="blue")
    ax2.set_ylabel(r"$dS/d(\log d)$ / (m$^2$ g$^{-1}$)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    # Sicherstellen, dass der Ausgabeordner existiert
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_bjh_file(INPUT_FILE)

    vol_pdf = BASE_DIR / "EXP-029_BJH_Porenvolumen.pdf"
    sa_pdf = BASE_DIR / "EXP-029_BJH_Oberflaeche.pdf"

    plot_volume(data, vol_pdf)
    plot_surface(data, sa_pdf)

    print("Plots gespeichert:")
    print(f"  {vol_pdf}")
    print(f"  {sa_pdf}")


if __name__ == "__main__":
    main()
