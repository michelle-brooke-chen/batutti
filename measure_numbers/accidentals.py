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

import cv2
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


# where each sharp of a key signature sits, in order (F C G D A E B), in
# staff steps above the middle line, per clef
SHARP_STEPS = {("G", 2): (4, 1, 5, 2, -1, 3, 0), ("F", 4): (2, -1, 3, 0, -3, 1, -2),
               ("C", 3): (3, 0, 4, 1, -2, 2, -1), ("C", 4): (-2, 2, -1, 3, 0, 4, 1)}


def _lines_out(ink, s):
    """Staff lines removed, gaps in a blotchy stroke closed."""
    out = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(3, int(0.5 * s)), 1), np.uint8))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((max(3, int(0.4 * s)), 1), np.uint8))


def _key_area(system, staff):
    """(x where the key signature starts, y0, y1) for one line."""
    s = staff["space"]
    # a clef read with no confidence is often the key's sharps
    clef = re.search(r'<clef [^>]*ctx-grade="([\d.]+)"[^>]*>\s*<bounds x="([\d.]+)" y="[\d.]+" w="([\d.]+)"', system)
    start = (float(clef.group(2)) + float(clef.group(3)) if clef and float(clef.group(1)) >= 0.5
             else staff["left"] + 2 * s)
    return start, int(staff["top"] - 3 * s), int(staff["top"] + 7 * s)


def key_sharps(omr, pdf, staves, clefs):
    """{(page, staff): number of sharps} in each line's key signature, read on
    the scan. `clefs` gives each staff's clef as printed, as (sign, line).

    A key's sharps always come in the same order at the same heights: the
    first is looked for at F's height just after the clef, each next one at
    its own height a little to the right of the one before, stopping at the
    first missing. A sharp is recognised by matching a sharp printed in the
    same part (a lone blob about a space wide and two and a half tall, wide
    at the top too, where a flat or natural is one thin stroke), so sharps
    printed touching are still told apart. Audiveris sometimes misses a
    sharp, or the whole key when it misreads the clef."""
    doc, cache = pymupdf.open(pdf), {}
    with zipfile.ZipFile(omr) as z:
        sheets = {page: z.read(f"sheet#{page}/sheet#{page}.xml").decode().split("<system ")
                  for page in sorted({s["page"] for s in staves})}
    lines = []
    for staff, clef in zip(staves, clefs):
        if tuple(clef[:2]) in SHARP_STEPS:
            start, y0, y1 = _key_area(sheets[staff["page"]][staff["staff"]], staff)
            ink = (page_image(doc, staff["page"], cache) < 140).astype(np.uint8)
            x0 = int(start - 0.3 * staff["space"])
            crop = _lines_out(ink[y0:y1, x0:int(start + 9 * staff["space"])], staff["space"])
            lines.append((staff, SHARP_STEPS[tuple(clef[:2])], x0, y0, crop))
    # a template: a lone sharp where a key's first sharp goes, scaled to a
    # 20-pixel space
    template = None
    for staff, steps, x0, y0, crop in lines:
        s = staff["space"]
        f_sharp = staff["top"] + 2 * s - steps[0] * s / 2 - y0
        count, labels, stats, centres = cv2.connectedComponentsWithStats(crop, 8)
        for k in range(1, count):
            x, y, w, h = stats[k][:4]
            if not (0.6 * s <= w <= 1.4 * s and 2 * s <= h <= 3.4 * s
                    and abs(centres[k][1] - f_sharp) < 0.5 * s and x < 2.6 * s):
                continue
            upper = labels[y + int(0.1 * h):y + int(0.3 * h), x:x + w] == k
            lower = labels[y + int(0.7 * h):y + int(0.9 * h), x:x + w] == k
            if upper.any(axis=0).sum() >= 0.5 * s and lower.any(axis=0).sum() >= 0.5 * s:
                patch = (labels[y:y + h, x:x + w] == k).astype(np.float32)
                template = cv2.resize(patch, (round(w * 20 / s), round(h * 20 / s)))
                break
        if template is not None:
            break
    out = {}
    if template is None:
        return out
    for staff, steps, x0, y0, crop in lines:
        s = staff["space"]
        t = cv2.resize(template, (max(3, round(template.shape[1] * s / 20)), max(3, round(template.shape[0] * s / 20))))
        th, tw = t.shape
        score = cv2.matchTemplate(crop.astype(np.float32), t, cv2.TM_CCOEFF_NORMED)
        middle = staff["top"] + 2 * s - y0
        n, last = 0, 0.0
        for step in steps:
            centre = middle - step * s / 2
            r0, r1 = int(centre - th / 2 - 0.3 * s), int(centre - th / 2 + 0.3 * s) + 1
            c0 = int(last + (0.5 * s if n else 0))
            c1 = int(last + 2.2 * s) + 1
            if r0 < 0 or r1 > score.shape[0] or c0 >= score.shape[1]:
                break
            window = score[r0:r1, c0:min(c1, score.shape[1])]
            if window.size == 0 or window.max() < SHARP_MATCH:
                break
            n, last = n + 1, c0 + np.unravel_index(window.argmax(), window.shape)[1] + 0.0
        out[(staff["page"], staff["staff"])] = n
    return out


SHARP_MATCH = 0.6
