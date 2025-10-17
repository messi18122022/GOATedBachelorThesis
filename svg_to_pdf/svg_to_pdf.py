import re
from typing import Optional, Tuple

def _sanitize_svg_colors(svg_text: str) -> Tuple[str, int]:
    """Ersetzt ungültige Hex-Farben in fill/stroke/stop-color/color Attributen (auch in style) durch #000000.
    Lässt url(#id) & Co. unangetastet. Gibt (bereinigter_text, anzahl_ersetzungen) zurück."""
    replacements = 0

    # 1) Direkte Attribute, z.B. fill="#ig1234"
    attr_pattern = re.compile(r'\b(fill|stroke|stop-color|color)\s*=\s*(["\'])#([^"\']+?)\2', re.IGNORECASE)

    def _attr_sub(m: re.Match) -> str:
        nonlocal replacements
        name, quote, token = m.group(1), m.group(2), m.group(3)
        if _is_valid_hex_color(token):
            return m.group(0)
        replacements += 1
        return f"{name}={quote}#000000{quote}"

    svg_text = attr_pattern.sub(_attr_sub, svg_text)

    # 2) style-Attribute: style="fill:#ig12; stroke:#xyz; clip-path:url(#id)"
    style_pattern = re.compile(r'style\s*=\s*(\"|\')([^\"\']*)(\1)', re.IGNORECASE)

    def _fix_style(style_str: str) -> str:
        def _make_decl(prop: str):
            def _fix_decl(m: re.Match) -> str:
                nonlocal replacements
                token = m.group(1)
                if _is_valid_hex_color(token):
                    return m.group(0)
                replacements += 1
                return f"{prop}:#000000"
            return _fix_decl

        for prop in ("fill", "stroke", "stop-color", "color"):
            pattern = re.compile(rf'(?i)(?<!-)\b{prop}\s*:\s*#([A-Za-z0-9_-]{{1,64}})')
            style_str = pattern.sub(_make_decl(prop), style_str)
        return style_str

    def _style_sub(m: re.Match) -> str:
        quote, body = m.group(1), m.group(2)
        fixed = _fix_style(body)
        return f"style={quote}{fixed}{quote}"

    svg_text = style_pattern.sub(_style_sub, svg_text)

    return svg_text, replacements


def choose_svg_file() -> Optional[Path]:
    # function body remains unchanged
    pass

def convert_svg_to_pdf(svg_path: Path) -> Path:
    """Konvertiert die angegebene SVG-Datei in ein PDF am gleichen Ort mit gleichem Namen."""
    pdf_path = svg_path.with_suffix(".pdf")

    if pdf_path.exists():
        root = tk.Tk(); root.withdraw()
        overwrite = messagebox.askyesno(
            "Datei existiert",
            f"Die Datei\n{pdf_path}\nexistiert bereits. Überschreiben?"
        )
        root.destroy()
        if not overwrite:
            return pdf_path

    # 1. Versuch (ohne Modifikation)
    try:
        cairosvg.svg2pdf(url=svg_path.as_uri(), write_to=str(pdf_path))
        return pdf_path
    except Exception as e1:
        # 2. Versuch mit Bereinigung, wenn es nach fehlerhaften Hex-Farben aussieht
        msg = str(e1)
        looks_like_hex_issue = ("base 16" in msg) or ("hex" in msg.lower())
        if not looks_like_hex_issue:
            raise

        try:
            original = svg_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raise

        fixed, n = _sanitize_svg_colors(original)
        if n == 0:
            raise

        try:
            cairosvg.svg2pdf(bytestring=fixed.encode("utf-8"), write_to=str(pdf_path))
        except Exception:
            raise e1

        root = tk.Tk(); root.withdraw()
        messagebox.showinfo(
            "Hinweis",
            f"Einige ungültige Hex-Farben wurden automatisch korrigiert (ersetzt durch #000000).\nAnzahl Ersetzungen: {n}"
        )
        root.destroy()
        return pdf_path


def main() -> None:
    svg_path = choose_svg_file()
    if svg_path is None:
        return

    try:
        out_pdf = convert_svg_to_pdf(svg_path)
    except Exception as e:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Fehler bei der Konvertierung", f"Konvertierung ist fehlgeschlagen:\n{e}")
        root.destroy()
        print(f"Fehler bei der Konvertierung: {e}", file=sys.stderr)
        sys.exit(2)

    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("Fertig", f"PDF gespeichert:\n{out_pdf}")
    root.destroy()
    print(f"PDF gespeichert: {out_pdf}")


def choose_svg_file() -> Optional[Path]:
    """Öffnet einen Dateidialog und gibt den gewählten SVG-Pfad zurück (oder None)."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename(
        title="SVG Datei wählen",
        filetypes=[("SVG Dateien", "*.svg"), ("Alle Dateien", "*.*")]
    )

    root.destroy()
    if not file_path:
        return None
    p = Path(file_path)
    if p.suffix.lower() != ".svg":
        messagebox.showwarning("Ungültige Datei", "Bitte eine *.svg Datei auswählen.")
        return None
    return p


if __name__ == "__main__":
    main()