"""Sharps and naturals in front of notes, read from the scan.

Audiveris misses many accidentals in older engravings, and sometimes reads
a sharp as a notehead. This looks just left of every notehead for the two
vertical strokes a sharp or natural is made of, measured in staff spaces
from the notehead's centre:

  sharp    two strokes 0.4-1 apart, from about -1.2 to +1.25
  natural  two strokes, the left one higher, about 3 spaces tall

Flats are not attempted: a stem with a note's tail looks too much like one.
Checked against a player's labels of 60 notes (evaluation/drafts): 8 right,
none wrong, 2 missed (both of which Audiveris found).
"""

import re
import zipfile

import numpy as np
import pymupdf

from .cues import page_image

SHARP, NATURAL = 1, 0


def strokes(ink, x0, x1, y0, y1, s):
    """Vertical strokes in a window: [(x, top, bottom)], from the longest
    ink run in each column. Runs reaching the window's top or bottom (stems,
    barlines) are skipped."""
    found = []
    for x in range(max(0, x0), x1):
        col = np.append(ink[y0:y1, x], 0)
        best, start = (0, 0, 0), None
        for y, v in enumerate(col):
            if v and start is None:
                start = y
            elif not v and start is not None:
                if y - start > best[0]:
                    best = (y - start, start, y)
                start = None
        length, top, bottom = best
        if length >= 1.6 * s and top > 1 and bottom < len(col) - 2:
            found.append((x, top + y0, bottom + y0))
    groups = []
    for x, t, b in found:
        if groups and x - groups[-1][-1][0] <= 2:
            groups[-1].append((x, t, b))
        else:
            groups.append([(x, t, b)])
    return [(sum(g[0] for g in grp) / len(grp), min(g[1] for g in grp), max(g[2] for g in grp))
            for grp in groups if len(grp) <= 0.35 * s]


def classify(ink, left, middle, s):
    """(SHARP or NATURAL, x span) for the notehead whose left edge is `left`
    and centre height `middle`, or None."""
    found = [(x, (t - middle) / s, (b - middle) / s) for x, t, b in
             strokes(ink, int(left - 1.75 * s), int(left - 0.02 * s),
                     int(middle - 2.4 * s), int(middle + 2.4 * s), s)]
    if len(found) < 2:
        return None
    a, b = found[-2], found[-1]
    gap = (b[0] - a[0]) / s
    if not (0.38 <= gap <= 1.0 and min(a[1], b[1]) < -0.8 and max(a[2], b[2]) > 0.8
            and min(a[2] - a[1], b[2] - b[1]) >= 1.9):
        return None
    height = max(a[2], b[2]) - min(a[1], b[1])
    kind = NATURAL if height >= 2.75 and a[1] < b[1] - 0.3 else SHARP
    return kind, (a[0], b[0])


def find(omr, pdf, staves):
    """{(page, staff): [(x centre, pitch position, alter, (x0, x1))]} for every
    notehead with a sharp or natural in front of it, (x0, x1) being the
    accidental's strokes."""
    out = {}
    doc, cache = pymupdf.open(pdf), {}
    space = {(s["page"], s["staff"]): s["space"] for s in staves}
    with zipfile.ZipFile(omr) as z:
        for page in sorted({s["page"] for s in staves}):
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            ink = None
            for k, system in enumerate(xml.split("<system ")[1:], start=1):
                if (page, k) not in space:
                    continue
                s = space[(page, k)]
                for attrs, x, y, w, h in re.findall(
                        r'<head ([^>]*)>\s*<bounds x="([\d.]+)" y="([\d.]+)" w="([\d.]+)" h="([\d.]+)"', system):
                    a = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
                    if "pitch" not in a or float(a.get("grade", 1)) < 0.4:
                        continue
                    if ink is None:
                        ink = (page_image(doc, page, cache) < 140).astype(np.uint8)
                    found = classify(ink, float(x), float(y) + float(h) / 2, s)
                    if found:
                        out.setdefault((page, k), []).append(
                            (float(x) + float(w) / 2, int(float(a["pitch"])), found[0], found[1]))
    return out
