#!/usr/bin/env python3
"""Utility to extract delivery note and invoice data from PDFs.

The script can optionally perform OCR on hand written annotations and stores
all extracted information in either a SQLite database or an exported CSV file.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text

try:  # Optional dependency for OCR support.
    import pytesseract

    OCR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency only.
    OCR_AVAILABLE = False

try:  # Optional improvement for OCR preprocessing.
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency only.
    CV_AVAILABLE = False


CONFIG_PATH = Path.home() / ".lieferschein_cfg.json"

# Regular expressions that are shared across the module.
DATE_PATTERN = re.compile(r"(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})")
UNIT_PATTERN = re.compile(r"(\d+[.,]?\d*)\s*(mm|cm|m)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

LOG = logging.getLogger("lauf_local")


def load_directory_config(config_path: Path = CONFIG_PATH) -> Optional[Tuple[Path, Path, Path]]:
    """Return cached directories if they are valid."""

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        LOG.warning("Konfigurationsdatei %s konnte nicht gelesen werden: %s", config_path, exc)
        return None

    try:
        delivery = Path(raw["lieferscheine"]).expanduser()
        invoices = Path(raw["rechnungen"]).expanduser()
        output = Path(raw["out"]).expanduser()
    except KeyError as missing_key:
        LOG.warning("Konfiguration unvollständig, Schlüssel fehlt: %s", missing_key)
        return None

    paths = (delivery, invoices, output)
    missing_dirs = [path for path in paths if not path.is_dir()]
    if missing_dirs:
        LOG.warning("Konfigurationsordner fehlen: %s", ", ".join(str(path) for path in missing_dirs))
        return None

    return paths



def save_directory_config(delivery: Path, invoices: Path, output: Path, config_path: Path = CONFIG_PATH) -> None:
    """Persist directory configuration for reuse."""

    payload = {
        "lieferscheine": str(delivery),
        "rechnungen": str(invoices),
        "out": str(output),
    }
    try:
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOG.warning("Konfiguration konnte nicht gespeichert werden: %s", exc)


def configure_logging(verbose: bool = False) -> None:
    """Initialise a simple logging configuration."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def to_mm(value: str, unit: str) -> float:
    """Convert a numeric string value to millimetres."""

    value = value.replace(",", ".")
    numeric = float(value)
    unit = unit.lower()
    if unit == "mm":
        return numeric
    if unit == "cm":
        return numeric * 10.0
    if unit == "m":
        return numeric * 1000.0
    return numeric


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_date(text: Optional[str]) -> Optional[str]:
    """Extract the first occurrence of a date from *text* in ISO format."""

    if not text:
        return None
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1)
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def ensure_db(db_path: Path) -> sqlite3.Connection:
    """Ensure the SQLite schema exists and return a connection."""

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS positionen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datum TEXT,
            projektname TEXT,
            dateiname_gedruckt TEXT,
            anzahl INTEGER,
            laenge REAL,
            breite REAL,
            rechnungsnummer TEXT,
            einzelpreis REAL,
            handschrift_notizen TEXT,
            quelle_pdf_path TEXT,
            quelle_pdf_sha256 TEXT,
            lieferscheinnummer TEXT,
            rechnungsdatum TEXT,
            waehrung TEXT DEFAULT 'EUR'
        )
        """
    )
    connection.commit()
    return connection


def insert_position(connection: sqlite3.Connection, row: Dict[str, object]) -> None:
    """Insert a single row into the *positionen* table."""

    columns = ",".join(row.keys())
    placeholders = ",".join(["?"] * len(row))
    connection.execute(
        f"INSERT INTO positionen ({columns}) VALUES ({placeholders})",
        list(row.values()),
    )


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------


def pdf_page_to_image(reader: PdfReader, index: int) -> Optional[Image.Image]:
    """Try to obtain a PIL image for *index* from *reader*.

    The implementation accesses the XObject images. This is a pragmatic
    approach that works for scanned PDFs that embed a raster image per page.
    """

    try:
        page = reader.pages[index]
        resources = page.get("/Resources")
        if not resources or "/XObject" not in resources:
            return None
        xobjects = resources["/XObject"]
        for name in xobjects:
            obj = xobjects[name]
            if obj.get("/Subtype") != "/Image":
                continue
            size = (obj.get("/Width"), obj.get("/Height"))
            data = obj.get_data()
            mode = "RGB"
            if obj.get("/ColorSpace") == "/DeviceGray":
                mode = "L"
            return Image.frombytes(mode, size, data)
    except Exception:
        LOG.debug("Unable to render page %s for OCR", index, exc_info=True)
    return None


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Apply a light-weight pre-processing pipeline for OCR."""

    grayscale = image.convert("L")
    if not CV_AVAILABLE:
        return grayscale

    # Adaptive thresholding helps to isolate hand-written text.
    array = np.array(grayscale)
    array = cv2.adaptiveThreshold(
        array,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return Image.fromarray(array)


def ocr_image(image: Image.Image) -> str:
    """Perform OCR on *image* and return the extracted text."""

    if not OCR_AVAILABLE:
        return ""
    prepared = preprocess_for_ocr(image)
    return pytesseract.image_to_string(prepared, lang="deu+eng")


# ---------------------------------------------------------------------------
# Parsing heuristics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Position:
    datum: Optional[str]
    projektname: Optional[str]
    dateiname_gedruckt: str
    anzahl: int
    laenge: float
    breite: float
    lieferscheinnummer: Optional[str]
    source_page_indices: Sequence[int]


def extract_positions_from_text(
    pages: Sequence[str],
) -> List[Position]:
    """Extract position information from the textual representation of a PDF."""

    positions: List[Position] = []
    header_lines: List[str] = []
    for page in pages:
        if not page:
            continue
        header_lines.extend(page.splitlines()[:40])
    header_text = "\n".join(header_lines)
    datum = parse_date(header_text)
    projektname = None
    lieferscheinnummer = None

    project_match = re.search(r"(Projekt|Kunde|Auftrag)\s*[:\-]\s*(.+)", header_text, re.IGNORECASE)
    if project_match:
        projektname = project_match.group(2).strip()

    ls_match = re.search(r"(?:LS|Lieferschein)[\s:\-]*([A-Z0-9\-/]+)", header_text, re.IGNORECASE)
    if ls_match:
        lieferscheinnummer = ls_match.group(1).strip()

    for page_index, page_text in enumerate(pages):
        if not page_text:
            continue
        for line in (line.strip() for line in page_text.splitlines() if line.strip()):
            file_match = re.search(r"(Datei|File)\s*[:\-]\s*([^|]+)", line, re.IGNORECASE)
            qty_match = re.search(r"(Anzahl|Qty|Stk)\s*[:\-]?\s*(\d+)", line, re.IGNORECASE)
            dims = re.findall(UNIT_PATTERN, line)
            if not file_match or not qty_match or len(dims) < 2:
                continue

            file_name = file_match.group(2).strip()
            quantity = int(qty_match.group(2))
            (length_value, length_unit), (width_value, width_unit) = dims[0], dims[1]
            length_mm = to_mm(length_value, length_unit)
            width_mm = to_mm(width_value, width_unit)

            positions.append(
                Position(
                    datum=datum,
                    projektname=projektname,
                    dateiname_gedruckt=file_name,
                    anzahl=quantity,
                    laenge=length_mm,
                    breite=width_mm,
                    lieferscheinnummer=lieferscheinnummer,
                    source_page_indices=(page_index,),
                )
            )

    return positions


@dataclasses.dataclass
class InvoiceInfo:
    number: Optional[str]
    date: Optional[str]
    price_lines: List[Tuple[str, float]]
    project_candidates: List[str]
    referenced_lieferscheine: List[str]


def parse_invoice(text: str) -> InvoiceInfo:
    """Parse invoice information from textual PDF content."""

    number = None
    number_match = re.search(r"(?:RE|Rechnung)[\s:\-]*([A-Z0-9\-/]+)", text, re.IGNORECASE)
    if number_match:
        number = number_match.group(1).strip()

    invoice_date = parse_date(text)

    price_lines: List[Tuple[str, float]] = []
    for line in text.splitlines():
        if "€" not in line and "EUR" not in line.upper():
            continue
        price_match = re.search(r"[à@]\s*([0-9]+[.,][0-9]{2})", line)
        file_match = re.search(r"([\w\-]+\.pdf)", line, re.IGNORECASE)
        if price_match and file_match:
            price = float(price_match.group(1).replace(",", "."))
            price_lines.append((file_match.group(1), price))

    project_candidates: List[str] = []
    for match in re.finditer(r"(Projekt|Kunde|Auftrag)\s*[:\-]\s*(.+)", text, re.IGNORECASE):
        candidate = match.group(2).strip()
        if candidate and candidate not in project_candidates:
            project_candidates.append(candidate)

    referenced_ls = list(
        {
            m.group(1).strip()
            for m in re.finditer(r"(?:LS|Lieferschein)[\s:\-]*([A-Z0-9\-/]+)", text, re.IGNORECASE)
        }
    )

    return InvoiceInfo(
        number=number,
        date=invoice_date,
        price_lines=price_lines,
        project_candidates=project_candidates,
        referenced_lieferscheine=referenced_ls,
    )


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def pdf_text_per_page(path: Path) -> List[str]:
    """Return extracted text for every page using PyPDF2 as a fallback."""

    try:
        reader = PdfReader(str(path))
    except Exception:
        LOG.warning("Unable to read PDF %s", path, exc_info=True)
        return []

    texts: List[str] = []
    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            LOG.debug("Failed to extract text from page %s of %s", page_index, path, exc_info=True)
            text = ""
        texts.append(text)
    return texts



def extract_ocr_texts(reader: PdfReader) -> List[str]:
    """Return OCR-derived text per page for *reader*."""

    if not OCR_AVAILABLE:
        return []

    texts: List[str] = []
    for page_index in range(len(reader.pages)):
        image = pdf_page_to_image(reader, page_index)
        if not image:
            texts.append("")
            continue
        text = ocr_image(image).strip()
        if text:
            LOG.debug("OCR Text (Seite %s): %s", page_index + 1, text)
        texts.append(text)
    return texts


def assign_notes_to_positions(
    positions: Sequence[Position],
    page_texts: Sequence[str],
    ocr_texts: Sequence[str],
    pdf_path: Path,
) -> Dict[str, Optional[str]]:
    """Return a mapping from file names to detected hand-written notes."""

    notes: Dict[str, Optional[str]] = {pos.dateiname_gedruckt: None for pos in positions}
    unassigned: List[str] = []

    for pos in positions:
        note = None
        for page_index in pos.source_page_indices:
            if page_index >= len(ocr_texts):
                continue
            text = ocr_texts[page_index]
            if text:
                note = text
                break
        if note is None and ocr_texts and len(set(filter(None, ocr_texts))) == 1:
            note = next(filter(None, ocr_texts), None)
        notes[pos.dateiname_gedruckt] = note

    if ocr_texts:
        for index, text in enumerate(ocr_texts):
            if not text:
                continue
            page_text = page_texts[index] if index < len(page_texts) else ""
            referenced = [pos for pos in positions if pos.dateiname_gedruckt in page_text]
            if not referenced:
                unassigned.append(f"Seite {index + 1}: {text}")

    if unassigned:
        LOG.warning("OCR-Notizen in %s konnten nicht eindeutig zugeordnet werden: %s", pdf_path, "; ".join(unassigned))

    return notes



def import_lieferscheine(
    source: Path,
    connection: sqlite3.Connection,
    enable_ocr: bool,
) -> None:
    """Import all delivery notes from *source* into the database."""

    if not source.exists():
        LOG.warning("Lieferschein-Ordner existiert nicht: %s", source)
        return

    if enable_ocr and not OCR_AVAILABLE:
        LOG.warning("OCR wurde angefordert, aber pytesseract ist nicht verfuegbar.")
        enable_ocr = False

    for pdf_path in sorted(source.glob("*.pdf")):
        checksum = sha256_file(pdf_path)
        cursor = connection.execute(
            "SELECT 1 FROM positionen WHERE quelle_pdf_sha256 = ? LIMIT 1",
            (checksum,),
        )
        if cursor.fetchone():
            LOG.info("Ueberspringe bereits importierten Lieferschein: %s", pdf_path.name)
            continue

        LOG.info("Verarbeite Lieferschein: %s", pdf_path.name)

        reader: Optional[PdfReader] = None
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            LOG.warning("PyPDF2 konnte %s nicht oeffnen", pdf_path, exc_info=True)

        ocr_texts: List[str] = []
        page_texts: List[str] = []

        if enable_ocr and reader is not None:
            ocr_texts = extract_ocr_texts(reader)
            if any(text.strip() for text in ocr_texts):
                page_texts = ocr_texts

        if not page_texts:
            page_texts = pdf_text_per_page(pdf_path)

        if not page_texts:
            try:
                fallback_text = extract_text(str(pdf_path)) or ""
            except Exception:
                LOG.warning("Konnte Text aus %s nicht extrahieren", pdf_path, exc_info=True)
                fallback_text = ""
            if fallback_text:
                page_texts = [fallback_text]

        if not page_texts:
            LOG.warning("Keine Textinformation in Lieferschein %s gefunden", pdf_path)
            continue

        used_ocr_pages = page_texts is ocr_texts

        positions = extract_positions_from_text(page_texts)

        if not positions and enable_ocr and ocr_texts and not used_ocr_pages:
            positions = extract_positions_from_text(ocr_texts)

        if not positions:
            LOG.warning("Keine Positionen in Lieferschein %s gefunden", pdf_path)
            continue

        notes_map = assign_notes_to_positions(positions, page_texts, ocr_texts, pdf_path)

        for pos in positions:
            row = {
                "datum": pos.datum,
                "projektname": pos.projektname,
                "dateiname_gedruckt": pos.dateiname_gedruckt,
                "anzahl": pos.anzahl,
                "laenge": pos.laenge,
                "breite": pos.breite,
                "rechnungsnummer": None,
                "einzelpreis": None,
                "handschrift_notizen": notes_map.get(pos.dateiname_gedruckt),
                "quelle_pdf_path": str(pdf_path.resolve()),
                "quelle_pdf_sha256": checksum,
                "lieferscheinnummer": pos.lieferscheinnummer,
                "rechnungsdatum": None,
                "waehrung": "EUR",
            }
            insert_position(connection, row)

        connection.commit()


def assign_invoice_to_positions(
    connection: sqlite3.Connection,
    info: InvoiceInfo,
    pdf_path: Path,
) -> None:
    """Update existing positions with invoice metadata using heuristics."""

    if not info.number and not info.date and not info.price_lines:
        LOG.debug("Rechnung %s enthält keine verwertbaren Informationen", pdf_path)
        return

    LOG.info("Verarbeite Rechnung: %s", pdf_path.name)

    if info.referenced_lieferscheine:
        for ls_number in info.referenced_lieferscheine:
            LOG.debug("Verknüpfe per Lieferscheinnummer %s", ls_number)
            connection.execute(
                """
                UPDATE positionen
                SET rechnungsnummer = COALESCE(?, rechnungsnummer),
                    rechnungsdatum = COALESCE(?, rechnungsdatum)
                WHERE lieferscheinnummer = ?
            """,
                (info.number, info.date, ls_number),
            )

    for file_name, price in info.price_lines:
        LOG.debug("Verknüpfe per Dateiname %s", file_name)
        escaped = file_name.replace("%", "\\%").replace("_", "\\_")
        connection.execute(
            """
            UPDATE positionen
            SET rechnungsnummer = COALESCE(?, rechnungsnummer),
                rechnungsdatum = COALESCE(?, rechnungsdatum),
                einzelpreis = ?
            WHERE dateiname_gedruckt LIKE ? ESCAPE '\\'
        """,
            (info.number, info.date, price, escaped),
        )

    if info.project_candidates:
        LOG.debug("Verknüpfe per Projektnamen")
        for project in info.project_candidates:
            escaped_project = project.replace("%", "\\%").replace("_", "\\_")
            connection.execute(
                """
                UPDATE positionen
                SET rechnungsnummer = COALESCE(?, rechnungsnummer),
                    rechnungsdatum = COALESCE(?, rechnungsdatum)
                WHERE projektname LIKE ? ESCAPE '\\'
                  AND (rechnungsnummer IS NULL OR rechnungsnummer = '')
            """,
                (info.number, info.date, f"%{escaped_project}%"),
            )

    if info.date:
        date_obj = datetime.fromisoformat(info.date)
        start = (date_obj - timedelta(days=30)).date().isoformat()
        end = (date_obj + timedelta(days=30)).date().isoformat()
        LOG.debug("Verknüpfe per Datumskorridor %s – %s", start, end)
        connection.execute(
            """
            UPDATE positionen
            SET rechnungsnummer = COALESCE(?, rechnungsnummer),
                rechnungsdatum = COALESCE(?, rechnungsdatum)
            WHERE datum BETWEEN ? AND ?
              AND (rechnungsnummer IS NULL OR rechnungsnummer = '')
        """,
            (info.number, info.date, start, end),
        )

    connection.commit()



def import_rechnungen(
    source: Path,
    connection: sqlite3.Connection,
    enable_ocr: bool,
) -> None:
    """Import invoice metadata and attach it to existing positions."""

    if not source.exists():
        LOG.warning("Rechnungs-Ordner existiert nicht: %s", source)
        return

    if enable_ocr and not OCR_AVAILABLE:
        LOG.warning("OCR wurde angefordert, aber pytesseract ist nicht verfuegbar.")
        enable_ocr = False

    for pdf_path in sorted(source.glob("*.pdf")):
        text_content = ""

        if enable_ocr:
            reader: Optional[PdfReader] = None
            try:
                reader = PdfReader(str(pdf_path))
            except Exception:
                LOG.warning("PyPDF2 konnte %s nicht oeffnen", pdf_path, exc_info=True)
            if reader is not None:
                ocr_texts = extract_ocr_texts(reader)
                text_content = "\n".join(t for t in ocr_texts if t).strip()

        if not text_content:
            try:
                text_content = extract_text(str(pdf_path)) or ""
            except Exception:
                LOG.warning("Konnte Rechnung %s nicht lesen", pdf_path, exc_info=True)
                continue

        if not text_content:
            LOG.warning("Rechnung %s enthaelt keinen verwertbaren Text", pdf_path)
            continue

        info = parse_invoice(text_content)
        assign_invoice_to_positions(connection, info, pdf_path)


def export_to_csv(connection: sqlite3.Connection, output_file: Path) -> None:
    """Export the *positionen* table to CSV."""

    LOG.info("Exportiere CSV nach %s", output_file)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        header = [
            "datum",
            "projektname",
            "dateiname_gedruckt",
            "anzahl",
            "laenge",
            "breite",
            "rechnungsnummer",
            "einzelpreis",
            "handschrift_notizen",
            "quelle_pdf_path",
            "quelle_pdf_sha256",
            "lieferscheinnummer",
            "rechnungsdatum",
            "waehrung",
        ]
        writer.writerow(header)
        for row in connection.execute(
            "SELECT datum, projektname, dateiname_gedruckt, anzahl, laenge, breite,"
            " rechnungsnummer, einzelpreis, handschrift_notizen, quelle_pdf_path,"
            " quelle_pdf_sha256, lieferscheinnummer, rechnungsdatum, waehrung"
            " FROM positionen ORDER BY id"
        ):
            writer.writerow(row)


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------


def pick_directories_via_gui() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Let the user pick directories for delivery notes, invoices and output."""

    cached = load_directory_config()
    if cached:
        LOG.info("Verwende gespeicherte Verzeichnisse aus %s", CONFIG_PATH)
        return cached

    import tkinter as tk
    from tkinter import filedialog, messagebox

    previous: Dict[str, str] = {}
    if CONFIG_PATH.exists():
        try:
            previous = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            LOG.debug("Bestehende Konfiguration konnte nicht gelesen werden.", exc_info=True)

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Verzeichnisse wählen", "Wähle den Ordner mit LIEFERSCHEINEN.")
    delivery = filedialog.askdirectory(initialdir=previous.get("lieferscheine"))
    if not delivery:
        root.destroy()
        return None, None, None

    messagebox.showinfo("Verzeichnisse wählen", "Wähle den Ordner mit RECHNUNGEN.")
    invoices = filedialog.askdirectory(initialdir=previous.get("rechnungen"))
    if not invoices:
        root.destroy()
        return None, None, None

    messagebox.showinfo("Verzeichnisse wählen", "Wähle den AUSGABE-Ordner.")
    output = filedialog.askdirectory(initialdir=previous.get("out"))
    if not output:
        root.destroy()
        return None, None, None

    root.destroy()

    delivery_path = Path(delivery)
    invoices_path = Path(invoices)
    output_path = Path(output)
    save_directory_config(delivery_path, invoices_path, output_path)

    return delivery_path, invoices_path, output_path



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lieferscheine/Rechnungen verarbeiten (inkl. Handschrift-OCR)."
    )
    parser.add_argument("--lieferscheine", type=Path, help="Ordner mit Lieferschein-PDFs")
    parser.add_argument("--rechnungen", type=Path, help="Ordner mit Rechnungs-PDFs")
    parser.add_argument("--outdir", type=Path, default=Path("./out"), help="Ausgabe-Ordner")
    parser.add_argument("--db", choices=["sqlite", "csv"], default="sqlite")
    parser.add_argument("--ocr", action="store_true", help="OCR fuer gescannte PDFs aktivieren")
    parser.add_argument("--gui", action="store_true", help="Verzeichnisse per Dialog wählen")
    parser.add_argument("--verbose", action="store_true", help="Ausführliche Log-Ausgabe aktivieren")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.gui:
        lieferscheine, rechnungen, outdir = pick_directories_via_gui()
        if not (lieferscheine and rechnungen and outdir):
            LOG.error("GUI-Auswahl abgebrochen.")
            return 1
        args.lieferscheine = lieferscheine
        args.rechnungen = rechnungen
        args.outdir = outdir

    if not args.lieferscheine or not args.rechnungen:
        LOG.error("Bitte Ordner für --lieferscheine und --rechnungen angeben (oder --gui nutzen).")
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)

    db_path = args.outdir / "positionen.sqlite3"
    connection = ensure_db(db_path)

    import_lieferscheine(args.lieferscheine, connection, args.ocr)
    import_rechnungen(args.rechnungen, connection, args.ocr)

    if args.db == "csv":
        export_to_csv(connection, args.outdir / "positionen.csv")

    LOG.info("Fertig. Datenbank: %s", db_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
