"""Stamp measure numbers onto the original scanned PDF.

Puts the number of the first measure of each staff above its clef, the usual
place in printed parts, or the nearest spot with no ink so nothing on the
page (rehearsal letters, pencil markings) is covered. The scan itself is
untouched; numbers are added as a new layer of text. With show_flags, staves
the tool is unsure about also get an orange "?" so a reviewer can find them.
"""

from pathlib import Path

import numpy as np
import pymupdf

DPI = 300  # coordinates in measure_count.json are 300-dpi pixels
PT = 72 / DPI  # pixels -> PDF points
FONT_SIZE = 9
MAX_INK = 0.01  # share of dark pixels allowed under a number
PAD = 3  # points of clear space required around a number


def ink_share(gray, rect):
    """Fraction of dark pixels in a PDF-point rectangle of a 300-dpi page image."""
    x0, y0, x1, y1 = (int(v / PT) for v in rect)
    area = gray[max(0, y0):y1, max(0, x0):x1]
    return (area < 128).mean() if area.size else 1.0


def place(gray, st, width):
    """Baseline position for a label: above the clef, higher, or in the margin."""
    s, left, top = st["space"] * PT, st["left"] * PT, st["top"] * PT
    candidates = [(left, top - k * s) for k in (1.8, 2.6, 3.4)]
    candidates.append((left - width - 0.8 * s, top + 1.0 * s))  # left margin
    for x, y in candidates:
        box = (x - PAD, y - FONT_SIZE * 0.8 - PAD, x + width + PAD, y + PAD)
        if ink_share(gray, box) < MAX_INK:
            return x, y
    return candidates[-1]


def stamp(pdf_path, data, out_path, show_flags=False):
    """Write a copy of pdf_path with each staff's first measure number stamped on."""
    flagged = {(b["page"], b["staff"]) for b in data["bars"] if b.get("flags")}
    doc = pymupdf.open(pdf_path)

    images = {}
    for st in data["staves"]:
        page = doc[st["page"] - 1]
        if st["page"] not in images:
            pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
            images[st["page"]] = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width)
        number = st["first_measure"]
        if number == 1 and not show_flags:
            continue  # parts don't number the first bar of a movement
        label = str(number)
        width = pymupdf.get_text_length(label, fontname="hebo", fontsize=FONT_SIZE)
        x, y = place(images[st["page"]], st, width)
        page.insert_text((x, y), label, fontname="hebo", fontsize=FONT_SIZE)
        if show_flags and (st["page"], st["staff"]) in flagged:
            page.insert_text((x + width + 3, y), "?", fontname="hebo",
                             fontsize=FONT_SIZE + 2, color=(0.95, 0.45, 0))

    out_path = Path(out_path)
    doc.save(out_path, garbage=3, deflate=True)
    return out_path
