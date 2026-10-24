"""Staff and barline detection on scanned parts, with plain image processing.

Used as a second opinion next to Audiveris: it finds barlines Audiveris
missed and confirms the ones it reports.
"""

import cv2
import numpy as np


def binarize(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def find_staves(ink):
    """Group long horizontal lines into 5-line staves."""
    w = ink.shape[1]
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, w // 10), np.uint8))
    rows = np.where(lines.sum(axis=1) / 255 > w * 0.3)[0]
    # merge adjacent rows into single line centres
    centres, run = [], [rows[0]] if len(rows) else []
    for r in rows[1:]:
        if r - run[-1] <= 2:
            run.append(r)
        else:
            centres.append(sum(run) / len(run))
            run = [r]
    if run:
        centres.append(sum(run) / len(run))

    staves, i = [], 0
    while i + 4 < len(centres):
        five = centres[i:i + 5]
        gaps = np.diff(five)
        if gaps.max() - gaps.min() <= max(3, 0.25 * gaps.mean()):
            cols = np.where(lines[[int(round(y)) for y in five]].max(axis=0) > 0)[0]
            staves.append({
                "top": five[0], "bottom": five[4], "space": float(gaps.mean()),
                "left": int(cols.min()), "right": int(cols.max()),
            })
            i += 5
        else:
            i += 1
    return staves, lines


END_WINDOW = 0.4  # staff spaces either side of a stroke checked for attachments
END_INK = 0.15  # max ink fraction in that window for a barline


def ends_are_clear(bare, x, w, staff):
    """True if nothing is attached at the top and bottom of a vertical stroke.

    `bare` is the ink with staff lines removed. A stem has a notehead or beam
    at one end; a barline has only the staff lines it crosses.
    """
    top, bottom, s = staff["top"], staff["bottom"], staff["space"]
    x0, x1 = int(x - END_WINDOW * s), int(x + w + END_WINDOW * s)
    for y0, y1 in ((top - 0.8 * s, top + 0.5 * s), (bottom - 0.5 * s, bottom + 0.8 * s)):
        window = bare[int(y0):int(y1), x0:x1].copy()
        window[:, max(0, int(x - 1) - x0):int(x + w + 1) - x0] = 0
        if (window > 0).mean() > END_INK:
            return False
    return True


def find_barlines(ink, bare, staff, check_ends=True):
    """Vertical strokes spanning exactly the staff height.

    Stems usually start inside the staff or run past it (to a beam or a
    notehead outside), so requiring both ends to sit on the outer staff lines
    filters most of them out. With check_ends=False, strokes with something
    touching an end (e.g. a slur crossing a barline) are kept too.
    """
    top, bottom, s = staff["top"], staff["bottom"], staff["space"]
    y0, y1 = int(top - 3 * s), int(bottom + 3 * s)
    right = min(ink.shape[1], int(staff["right"] + 2 * s))
    band = ink[max(0, y0):y1, staff["left"]:right]
    height = bottom - top
    vert = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((int(height * 0.9), 1), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(vert)

    found = []
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        y_top, y_bot = y + max(0, y0), y + h + max(0, y0)
        covers = y_top <= top + 0.3 * s and y_bot >= bottom - 0.3 * s
        overshoot = top - y_top > 0.6 * s or y_bot - bottom > 0.6 * s
        x_abs = x + staff["left"]
        ends_ok = not check_ends or ends_are_clear(bare, x_abs, w, staff)
        if covers and not overshoot and w <= 1.2 * s and ends_ok:
            found.append({"x": int(x + staff["left"]), "w": int(w)})

    # merge double / thick / repeat barlines into one
    found.sort(key=lambda b: b["x"])
    merged = []
    for b in found:
        if merged and b["x"] - (merged[-1]["x"] + merged[-1]["w"]) < 1.2 * s:
            last = merged[-1]
            last["w"] = b["x"] + b["w"] - last["x"]
            last["parts"] += 1
        else:
            merged.append({**b, "parts": 1})
    # a barline right at the start of the staff opens the system, not a measure
    return [b for b in merged if b["x"] - staff["left"] > 2 * s]
