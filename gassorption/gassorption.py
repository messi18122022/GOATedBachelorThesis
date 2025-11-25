import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx}",
    "font.family": "serif",
})


# Pfad zu deinem Isothermen-CSV
ISO_CSV_PATH = Path("/Users/musamoin/Desktop/BA-HS25/gassorption/EXP-029 (Isotherm).csv")


def plot_isotherm(
    csv_path=None,
    show=True,
    save_path=None,
):
    """
    Liesst die Isothermendaten aus einer CSV-Datei und erstellt einen Plot.

    Erwartet im CSV mindestens die Spalten:
    - 'Relative Pressure'
    - 'Volume @ STP'
    """
    if csv_path is None:
        csv_path = ISO_CSV_PATH

    csv_path = Path(csv_path)

    # CSV einlesen
    df = pd.read_csv(csv_path)
    df["mol_per_g"] = df["Volume @ STP"] / 22414

    x_vals = df["Relative Pressure"]
    n_vals = df["mol_per_g"]
    df["n_transformed"] = x_vals / (n_vals * (1 - x_vals))

    # Sicherstellen, dass die benoetigten Spalten vorhanden sind
    required_cols = ["Relative Pressure", "Volume @ STP"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(
                f"Spalte {col!r} nicht im CSV gefunden. Verfuegbare Spalten: {list(df.columns)}"
            )

    x = df["Relative Pressure"]
    y = df["mol_per_g"]

    # Steigende und sinkende Bereiche automatisch bestimmen
    # Maximum im Volumen finden
    idx_max = df["Volume @ STP"].idxmax()

    # Steigende Phase: alles bis zum Maximum (inkl. Maximum)
    ads_df = df.loc[:idx_max]

    # Sinkende Phase: alles nach dem Maximum
    des_df = df.loc[idx_max + 1:]

    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # Steigende Phase grün
    ax.plot(
        ads_df["Relative Pressure"],
        ads_df["mol_per_g"],
        marker="o",
        markersize=4,
        linestyle="-",
        color="green",
        label="Adsorption"
    )

    # Sinkende Phase rot
    ax.plot(
        des_df["Relative Pressure"],
        des_df["mol_per_g"],
        marker="o",
        markersize=4,
        linestyle="-",
        color="red",
        label="Desorption"
    )

    # Höchster Punkt schwarz markieren und mit Desorption verbinden
    max_x = df.loc[idx_max, "Relative Pressure"]
    max_y = df.loc[idx_max, "mol_per_g"]

    # Schwarzer Marker auf dem Maximum
    ax.plot(max_x, max_y, marker="o", markersize=4, color="black")

    # Linie vom Maximum zum ersten Desorptionspunkt
    if not des_df.empty:
        first_des_x = des_df.iloc[0]["Relative Pressure"]
        first_des_y = des_df.iloc[0]["mol_per_g"]
        ax.plot([max_x, first_des_x], [max_y, first_des_y], color="red", linestyle="-")

    ax.set_xlabel(r"$p/p_0$")
    ax.set_ylabel(r"$n_\mathrm{ads}$ / $\left( \si{\mole\per\gram} \right)$")
    ax.grid(True)
    ax.legend()

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    fig.tight_layout()

    # Immer als PDF im gleichen Pfad speichern
    pdf_path = csv_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300)
    
    # Falls zusätzlich ein expliziter Speicherpfad gewünscht ist
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300)

    # Optional anzeigen
    if show:
        plt.show()

    return fig, ax


def plot_transformed(
    csv_path=None,
    show=True,
    save_path=None,
):
    """
    Plottet die transformierte Groesse y = x / (n_ads * (1 - x))
    gegen den relativen Druck x.
    """

    if csv_path is None:
        csv_path = ISO_CSV_PATH

    csv_path = Path(csv_path)

    # CSV einlesen
    df = pd.read_csv(csv_path)
    df["mol_per_g"] = df["Volume @ STP"] / 22414

    x_vals = df["Relative Pressure"]
    n_vals = df["mol_per_g"]
    df["n_transformed"] = x_vals / (n_vals * (1 - x_vals))

    # Nur Adsorptionsdaten im Bereich 0.05–0.30 verwenden
    idx_max = df["Volume @ STP"].idxmax()
    ads_all = df.loc[:idx_max]
    ads_df = ads_all[(ads_all["Relative Pressure"] >= 0.05) & (ads_all["Relative Pressure"] <= 0.30)]

    x = ads_df["Relative Pressure"].to_numpy()
    y = ads_df["n_transformed"].to_numpy()

    # Regression mit allen Punkten im Bereich 0.05–0.30
    m, b = np.polyfit(x, y, 1)
    C = 1 + m / b

    print(f"Koeffizienten (alle Punkte 0.05–0.30): a = {b}, b = {m}, C = {C}")

    nm = 1 / (b + m)
    nm_str = f"{nm:.2e}"
    nm_mantissa, nm_exp = nm_str.split("e")
    nm_exp = int(nm_exp)

    # spezifische Oberfläche berechnen
    a_s = nm * 97556
    a_s_str = f"{a_s:.2e}"
    a_s_mantissa, a_s_exp = a_s_str.split("e")
    a_s_exp = int(a_s_exp)

    # Korrelationskoeffizient r
    r = np.corrcoef(x, y)[0, 1]

    # Residuen berechnen
    y_pred = m * x + b
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))
    # Nur Punkte zeichnen (keine Verbindung)
    ax.plot(
        ads_df["Relative Pressure"],
        ads_df["n_transformed"],
        marker="o",
        markersize=4,
        linestyle="None",
        color="black",
        label="Daten",
    )

    # Regressionsgerade
    x_fit = np.linspace(ads_df["Relative Pressure"].min(), ads_df["Relative Pressure"].max(), 100)
    y_fit = m * x_fit + b
    ax.plot(
        x_fit,
        y_fit,
        linestyle="--",
        color="black",
        label=(
            rf"$r = {r:.4f}$"
            "\n" rf"$a = {b:.4f}$"
            "\n" rf"$b = {m:.4f}$"
            "\n" rf"$n_m = {nm_mantissa}\cdot 10^{{{nm_exp}}}$"
            "\n" rf"$C = {C:.4f}$"
            "\n" rf"$a_\mathrm{{s}}(\mathrm{{BET}}) = {a_s_mantissa}\cdot 10^{{{a_s_exp}}}\ \si{{\meter\tothe{{2}}\per\gram}}$"
        ),
    )

    ax.set_xlabel(r"relativer Druck $p/p_0$")
    ax.set_ylabel(r"$\frac{x}{n_\mathrm{ads}(1-x)}$")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    # PDF speichern
    pdf_path = csv_path.with_name(csv_path.stem + "_transformed.pdf")
    fig.savefig(pdf_path, dpi=300)

    # optionaler Speicherpfad
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300)

    # Residuen-Plot
    fig_res, ax_res = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax_res.axhline(0, linestyle="--", color="black")
    ax_res.plot(
        x,
        residuals,
        marker="o",
        markersize=4,
        linestyle="None",
        color="black",
    )
    ax_res.set_xlabel(r"relativer Druck $p/p_0$")
    ax_res.set_ylabel(r"Residuen")
    ax_res.grid(True)
    fig_res.tight_layout()

    # Residuen-PDF speichern
    pdf_path_res = csv_path.with_name(csv_path.stem + "_transformed_residuals.pdf")
    fig_res.savefig(pdf_path_res, dpi=300)

    if show:
        plt.show()

    return fig, ax


# Neue Funktion: plot_transformed_positive_C
def plot_transformed_positive_C(
    csv_path=None,
    show=True,
    save_path=None,
):
    """
    BET-Plot wie plot_transformed, aber der Bereich innerhalb 0.05–0.30 wird
    so eingeschraenkt, dass C > 0 ist (oberes Limit wird schrittweise reduziert).
    """

    if csv_path is None:
        csv_path = ISO_CSV_PATH

    csv_path = Path(csv_path)

    # CSV einlesen
    df = pd.read_csv(csv_path)
    df["mol_per_g"] = df["Volume @ STP"] / 22414

    x_vals = df["Relative Pressure"]
    n_vals = df["mol_per_g"]
    df["n_transformed"] = x_vals / (n_vals * (1 - x_vals))

    # Adsorptionszweig bestimmen
    idx_max = df["Volume @ STP"].idxmax()
    ads_all = df.loc[:idx_max]

    # Bereich 0.05–0.30, oberes Limit so anpassen, dass C > 0 wird
    upper = 0.30
    best_ads_df = None
    best_m = None
    best_b = None
    best_C = None
    best_r = None

    while upper >= 0.05:
        candidate = ads_all[(ads_all["Relative Pressure"] >= 0.05) & (ads_all["Relative Pressure"] <= upper)]
        x = candidate["Relative Pressure"].to_numpy()
        y = candidate["n_transformed"].to_numpy()

        if len(x) < 2:
            break

        m, b = np.polyfit(x, y, 1)
        C = 1 + m / b
        r = np.corrcoef(x, y)[0, 1]

        if C > 0:
            best_ads_df = candidate
            best_m = m
            best_b = b
            best_C = C
            best_r = r
            break

        upper -= 0.005

    if best_ads_df is None:
        # Fallback: letzter Kandidat, auch wenn C <= 0
        best_ads_df = candidate
        best_m = m
        best_b = b
        best_C = C
        best_r = r

    ads_df = best_ads_df
    x = ads_df["Relative Pressure"].to_numpy()
    y = ads_df["n_transformed"].to_numpy()
    m = best_m
    b = best_b
    C = best_C
    r = best_r

    print(
        f"Koeffizienten (angepasster Bereich C>0): "
        f"a = {b}, b = {m}, C = {C}, "
        f"p/p0 von {x.min()} bis {x.max()}"
    )

    nm = 1 / (b + m)
    nm_str = f"{nm:.2e}"
    nm_mantissa, nm_exp = nm_str.split("e")
    nm_exp = int(nm_exp)

    # spezifische Oberfläche berechnen
    a_s = nm * 97556
    a_s_str = f"{a_s:.2e}"
    a_s_mantissa, a_s_exp = a_s_str.split("e")
    a_s_exp = int(a_s_exp)

    # Residuen berechnen
    y_pred = m * x + b
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(16/2.54, 8/2.54))

    # Nur Punkte zeichnen (keine Verbindung)
    ax.plot(
        ads_df["Relative Pressure"],
        ads_df["n_transformed"],
        marker="o",
        markersize=4,
        linestyle="None",
        color="black",
        label="Daten",
    )

    # Regressionsgerade
    x_fit = np.linspace(ads_df["Relative Pressure"].min(), ads_df["Relative Pressure"].max(), 100)
    y_fit = m * x_fit + b
    ax.plot(
        x_fit,
        y_fit,
        linestyle="--",
        color="black",
        label=(
            rf"$r = {r:.4f}$"
            "\n" rf"$a = {b:.4f}$"
            "\n" rf"$b = {m:.4f}$"
            "\n" rf"$n_m = {nm_mantissa}\cdot 10^{{{nm_exp}}}$"
            "\n" rf"$C = {C:.4f}$"
            "\n" rf"$a_\mathrm{{s}}(\mathrm{{BET}}) = {a_s_mantissa}\cdot 10^{{{a_s_exp}}}\ \si{{\meter\tothe{{2}}\per\gram}}$"
        ),
    )

    ax.set_xlabel(r"relativer Druck $p/p_0$")
    ax.set_ylabel(r"$\frac{x}{n_\mathrm{ads}(1-x)}$")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    # PDF speichern
    pdf_path = csv_path.with_name(csv_path.stem + "_transformed_posC.pdf")
    fig.savefig(pdf_path, dpi=300)

    # optionaler Speicherpfad
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300)

    # Residuen-Plot
    fig_res, ax_res = plt.subplots(figsize=(16/2.54, 8/2.54))
    ax_res.axhline(0, linestyle="--", color="black")
    ax_res.plot(
        x,
        residuals,
        marker="o",
        markersize=4,
        linestyle="None",
        color="black",
    )
    ax_res.set_xlabel(r"relativer Druck $p/p_0$")
    ax_res.set_ylabel(r"Residuen")
    ax_res.grid(True)
    fig_res.tight_layout()

    # Residuen-PDF speichern
    pdf_path_res = csv_path.with_name(csv_path.stem + "_transformed_posC_residuals.pdf")
    fig_res.savefig(pdf_path_res, dpi=300)

    if show:
        plt.show()

    return fig, ax


def analyze_bet_point_counts(
    csv_path=None,
):
    """
    Analysiert den BET-Plot fuer verschiedene Anzahlen von Punkten
    im Adsorptionsbereich 0.05 <= p/p0 <= 0.30.

    Es werden nacheinander Fits mit N Punkten gemacht (N = 12, 11, ..., 3),
    und jeweils a, b, r, C, n_m und a_s(BET) ausgegeben.
    """

    if csv_path is None:
        csv_path = ISO_CSV_PATH

    csv_path = Path(csv_path)

    # CSV einlesen
    df = pd.read_csv(csv_path)
    df["mol_per_g"] = df["Volume @ STP"] / 22414

    x_vals = df["Relative Pressure"]
    n_vals = df["mol_per_g"]
    df["n_transformed"] = x_vals / (n_vals * (1 - x_vals))

    # Adsorptionszweig bestimmen
    idx_max = df["Volume @ STP"].idxmax()
    ads_all = df.loc[:idx_max]

    # Bereich 0.05–0.30
    ads_range = ads_all[(ads_all["Relative Pressure"] >= 0.05) & (ads_all["Relative Pressure"] <= 0.30)]

    total_points = len(ads_range)
    print(f"BET-Analyse im Bereich 0.05 ≤ p/p0 ≤ 0.30 mit insgesamt {total_points} Punkten")

    # Von N = total_points bis 3 heruntergehen
    for N in range(total_points, 2, -1):
        subset = ads_range.head(N)

        x = subset["Relative Pressure"].to_numpy()
        y = subset["n_transformed"].to_numpy()

        # Lineare Regression
        m, b = np.polyfit(x, y, 1)
        a = b
        b_slope = m

        # Korrelationskoeffizient r
        r = np.corrcoef(x, y)[0, 1]

        # BET-Parameter
        C = 1 + b_slope / a
        nm = 1 / (a + b_slope)
        a_s = nm * 97556

        # Plot für dieses N erstellen
        figN, axN = plt.subplots(figsize=(16/2.54, 8/2.54))

        # nur Punkte
        axN.plot(
            x,
            y,
            marker="o",
            markersize=4,
            linestyle="None",
            color="black",
            label="Daten",
        )

        # Regressionsgerade
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + b
        axN.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color="black",
            label=(
                rf"$r = {r:.4f}$"
                "\n" rf"$a = {a:.4f}$"
                "\n" rf"$b = {b_slope:.4f}$"
                "\n" rf"$n_m = {nm:.2e}$"
                "\n" rf"$C = {C:.4f}$"
                "\n" rf"$a_\mathrm{{s}} = {a_s:.2e}\ \si{{\meter\tothe{{2}}\per\gram}}$"
            ),
        )

        axN.set_xlabel(r"relativer Druck $p/p_0$")
        axN.set_ylabel(r"$\frac{x}{n_\mathrm{ads}(1-x)}$")
        axN.grid(True)
        axN.legend()
        figN.tight_layout()

        # speichern
        pdf_pathN = csv_path.with_name(csv_path.stem + f"_BET_N{N}.pdf")
        figN.savefig(pdf_pathN, dpi=300)

        # Residuen berechnen
        y_pred = m * x + b
        residuals = y - y_pred

        # Residuen-Plot
        fig_resN, ax_resN = plt.subplots(figsize=(16/2.54, 8/2.54))
        ax_resN.axhline(0, linestyle="--", color="black")
        ax_resN.plot(
            x,
            residuals,
            marker="o",
            markersize=4,
            linestyle="None",
            color="black",
        )
        ax_resN.set_xlabel(r"relativer Druck $p/p_0$")
        ax_resN.set_ylabel(r"Residuen")
        ax_resN.grid(True)
        fig_resN.tight_layout()

        # speichern
        pdf_path_resN = csv_path.with_name(csv_path.stem + f"_BET_N{N}_residuals.pdf")
        fig_resN.savefig(pdf_path_resN, dpi=300)

        print(
            f"N = {N:2d}, p/p0 von {x.min():.5f} bis {x.max():.5f} | "
            f"a = {a:.4f}, b = {b_slope:.4f}, r = {r:.6f}, "
            f"C = {C:.4f}, n_m = {nm:.4e} mol/g, a_s(BET) = {a_s:.4e} m^2/g"
        )


if __name__ == "__main__":
    # Wenn du das Skript direkt startest, wird automatisch die Isotherme geplottet.
    plot_isotherm()
    plot_transformed()
    plot_transformed_positive_C()
    analyze_bet_point_counts()