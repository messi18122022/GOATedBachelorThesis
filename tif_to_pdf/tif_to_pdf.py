

#!/usr/bin/env python3
"""
tif_to_pdf.py

Konvertiert rekursiv alle .tif/.tiff Dateien unter einem Wurzelpfad zu PDFs
und legt die PDFs direkt neben den Ausgangsdateien ab (gleicher Dateiname, Endung .pdf).

Voraussetzung: Pillow
    pip install Pillow
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageSequence


def find_tiffs(root: Path) -> List[Path]:
    """Finde alle .tif/.tiff Dateien (case-insensitive) rekursiv."""
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    # Duplikate entfernen, Sortierung fuer stabile Ausgabe
    uniq = sorted(set(files))
    return uniq


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Konvertiere Bild in ein fuer PDF-gueltiges Farbraum-Format (RGB oder L)."""
    if img.mode in ("RGB", "L"):
        return img
    # CMYK, P, I;16 etc. -> RGB
    return img.convert("RGB")


def tiff_to_pdf(tif_path: Path, pdf_path: Path) -> Tuple[bool, str]:
    """
    Konvertiere eine TIFF-Datei (ein- oder mehrseitig) zu PDF.
    Gibt (ok, msg) zurueck.
    """
    try:
        with Image.open(tif_path) as im:
            # Sammle Frames
            frames = [ensure_rgb(frame.copy()) for frame in ImageSequence.Iterator(im)]
            if not frames:
                return False, "Keine Frames erkannt"
            first = frames[0]
            rest = frames[1:]

            # Zielordner anlegen
            pdf_path.parent.mkdir(parents=True, exist_ok=True)

            if rest:
                # Mehrseitiges PDF
                first.save(pdf_path, "PDF", save_all=True, append_images=rest)
            else:
                # Einseitig
                first.save(pdf_path, "PDF")
        return True, "OK"
    except Exception as e:
        return False, f"Fehler: {e}"


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Konvertiere alle TIFFs zu PDFs (rekursiv).")
    parser.add_argument(
        "root",
        nargs="?",
        default="/Users/musamoin/Desktop/BA_HS25/experiments",
        help="Wurzelordner (Standard: /Users/musamoin/Desktop/BA_HS25/REM)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Existierende PDFs ueberschreiben (Standard: ueberspringen).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur anzeigen, was getan wuerde; keine Dateien schreiben.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"[FEHLER] Ordner nicht gefunden: {root}", file=sys.stderr)
        return 2

    tiffs = find_tiffs(root)
    if not tiffs:
        print("[INFO] Keine TIFF-Dateien gefunden.")
        return 0

    total = len(tiffs)
    ok_count = 0
    skip_count = 0
    err_count = 0

    print(f"[INFO] Gefundene TIFFs: {total}")
    for idx, tif_path in enumerate(tiffs, start=1):
        pdf_path = tif_path.with_suffix(".pdf")
        if pdf_path.exists() and not args.force:
            print(f"[{idx}/{total}] ueberspringe (PDF existiert bereits): {pdf_path}")
            skip_count += 1
            continue

        if args.dry_run:
            action = "wuerde ueberschreiben" if pdf_path.exists() else "wuerde erzeugen"
            print(f"[{idx}/{total}] {action}: {pdf_path}")
            continue

        ok, msg = tiff_to_pdf(tif_path, pdf_path)
        if ok:
            ok_count += 1
            print(f"[{idx}/{total}] erstellt: {pdf_path}")
        else:
            err_count += 1
            print(f"[{idx}/{total}] FEHLER bei {tif_path}: {msg}", file=sys.stderr)

    print("\n[ERGEBNIS] erstellt:", ok_count, "uebersprungen:", skip_count, "fehler:", err_count)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))