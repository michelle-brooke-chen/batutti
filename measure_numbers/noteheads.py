"""Chord noteheads Audiveris merged, re-read from the scan.

Audiveris sometimes reads two black noteheads a third apart (a C sharp and
an E) as one head between them (a D). This looks again at every chord
column it read: a template made from the part's own clearly read black
noteheads (staff lines taken out) slides along the column, and each clear
peak is a notehead at that place on the staff. A head read is split in two
only when the scan shows clear heads a step above and below it, none where
it was read, and nothing else changed in the column; the new heads must be
joined to the chord's ink, shaped like a head, and not where Audiveris read
a beam, clef, sign or the free end of the stem. Lone extra heads are not
added: on Karelia they were fingerings, beam ends and slur ends.
"""

import re
import zipfile

import cv2
import numpy as np
import pymupdf

from .cues import page_image

MATCH = 0.6


def _lines_out(ink, s):
    """Staff and ledger lines removed, nothing else: the white between
    stacked beams has to stay."""
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(3, int(0.5 * s)), 1), np.uint8))
STATS = None


def _heads(system):
    """[(grade, position, x, y, w, h)] of the black noteheads in one system."""
    out = []
    for attrs, x, y, w, h in re.findall(
            r'<head ([^>]*)>\s*<bounds x="([\d.]+)" y="([\d.]+)" w="([\d.]+)" h="([\d.]+)"', system):
        a = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
        if a.get("shape") == "NOTEHEAD_BLACK" and "pitch" in a:
            out.append((float(a["grade"]), int(float(a["pitch"])), float(x), float(y), float(w), float(h)))
    return out


def _taken(system):
    """What Audiveris read as something other than a notehead: beams as
    (x0, x1, y at x0, y at x1, half height), other signs as boxes."""
    beams = [(float(x1), float(x2), float(y1), float(y2), float(h) / 2) for h, x1, y1, x2, y2 in re.findall(
        r'<beam(?:-hook)? height="([\d.]+)"[^>]*>.*?<p1 x="([\d.]+)" y="([\d.]+)"/>\s*<p2 x="([\d.]+)" y="([\d.]+)"/>',
        system, re.S)]
    stems = [tuple(map(float, b)) for b in re.findall(
        r'<stem\b[^>]*>\s*<bounds x="([\d.]+)" y="([\d.]+)" w="([\d.]+)" h="([\d.]+)"', system)]
    boxes = [tuple(map(float, b)) for b in re.findall(
        r'<(?:clef|key|key-alter|time-[\w-]+|flag|alter|dynamics|fingering|text|word|sentence)\b[^>]*>\s*'
        r'<bounds x="([\d.]+)" y="([\d.]+)" w="([\d.]+)" h="([\d.]+)"', system)]
    return beams, stems, boxes


def _at_tip(x, y, ys, s, stems):
    """Whether a head at (x, y) would sit at the free end of the stem of the
    heads at heights ys: a chord's heads are all at one end of its stem,
    which runs on two spaces or more past them, to a beam or flag."""
    for sx, sy, sw, sh in stems:
        if sx - 1.2 * s <= x <= sx + sw + 1.2 * s and sy - 0.5 * s <= min(ys) and max(ys) <= sy + sh + 0.5 * s:
            heads_low = abs(max(ys) - (sy + sh)) < abs(min(ys) - sy)
            tip = sy if heads_low else sy + sh
            if abs(y - tip) < 2 * s:
                return True
    return False


def _on_sign(x, y, s, taken):
    beams, _, boxes = taken
    for x0, x1, y0, y1, half in beams:
        if x0 - 0.3 * s <= x <= x1 + 0.3 * s:
            at = y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y0
            if abs(y - at) <= half + 0.3 * s:
                return True
    return any(bx - 0.1 * s <= x <= bx + bw + 0.1 * s and by - 0.1 * s <= y <= by + bh + 0.1 * s
               for bx, by, bw, bh in boxes)


def reread(omr, pdf, staves):
    """{(page, staff): [(x centre, positions read, positions found)]} for every
    chord column where the scan shows other noteheads than Audiveris read.
    Positions count staff steps down from the middle line."""
    doc, cache = pymupdf.open(pdf), {}
    lines = []
    with zipfile.ZipFile(omr) as z:
        for page in sorted({s["page"] for s in staves}):
            systems = z.read(f"sheet#{page}/sheet#{page}.xml").decode().split("<system ")
            ink = None
            for staff in (s for s in staves if s["page"] == page):
                heads = _heads(systems[staff["staff"]])
                if not heads:
                    continue
                taken = _taken(systems[staff["staff"]])
                if ink is None:
                    ink = (page_image(doc, page, cache) < 140).astype(np.uint8)
                s = staff["space"]
                y0 = int(staff["top"] - 5 * s)
                lines.append((staff, heads, taken, y0, _lines_out(ink[y0:int(staff["top"] + 9 * s)], s).astype(np.float32)))
    # the template: the part's clearly read heads, averaged, at a 20-pixel space
    patches = []
    for staff, heads, taken, y0, clean in lines:
        s = staff["space"]
        for grade, _, x, y, w, h in heads:
            if grade >= 0.7 and len(patches) < 200:
                patch = clean[int(y) - y0:int(y + h) - y0, int(x):int(x + w)]
                if patch.size:
                    patches.append(cv2.resize(patch, (round(1.2 * 20), round(20))))
    if len(patches) < 10:
        return {}
    template = np.mean(patches, axis=0).astype(np.float32)
    out = {}
    for staff, heads, taken, y0, clean in lines:
        s = staff["space"]
        tw, th = max(5, round(1.2 * s)), max(5, round(s))
        t = cv2.resize(template, (tw, th))
        score = cv2.matchTemplate(clean, t, cv2.TM_CCOEFF_NORMED)
        # a chord's heads are joined by its stem; fingerings and signs are not
        _, blobs = cv2.connectedComponents((clean > 0).astype(np.uint8), connectivity=8)
        # the middle line's height along the line (scans are often tilted),
        # from where Audiveris put the heads it read
        xs = np.array([x + w / 2 for _, _, x, _, w, _ in heads])
        ys = np.array([y + h / 2 - pos * s / 2 - y0 for _, pos, _, y, _, h in heads])
        tilt, level = np.polyfit(xs, ys, 1) if len(heads) >= 4 and np.ptp(xs) > 10 * s else (0.0, float(np.median(ys)))
        # chord columns: heads read at the same x
        columns = []
        for head in sorted(heads, key=lambda h: h[2]):
            if columns and abs(head[2] - columns[-1][0][2]) < 0.4 * s:
                columns[-1].append(head)
            else:
                columns.append([head])
        for column in columns:
            x = int(np.mean([h[2] for h in column]))
            read = sorted({h[1] for h in column})
            middle = level + tilt * (x + column[0][4] / 2)
            # look from a fourth beyond the outer heads read
            lo = middle + (read[0] - 5) * s / 2 - th / 2
            hi = middle + (read[-1] + 5) * s / 2 - th / 2
            strip = score[max(0, int(lo)):min(score.shape[0], int(hi) + 1),
                          max(0, x - 3):min(score.shape[1], x + 4)]
            if strip.size == 0:
                continue
            col = strip.max(axis=1)
            near = max(4, int(0.6 * s))  # one peak per head: heads in a column are a space apart or more
            peaks = [((max(0, int(lo)) + r + th / 2 - middle) / (s / 2), float(col[r])) for r in range(len(col))
                     if col[r] >= 0.5 and col[r] == col[max(0, r - near):r + near + 1].max()
                     and near <= r < len(col) - near]  # not cut off by the strip's ends
            # measured from the heads read here: the page bends, and a half
            # step is only a few pixels
            shifts = [min((pos - p for pos, _ in peaks), key=abs, default=9) for p in read]
            shifts = [d for d in shifts if abs(d) <= 0.75]
            shift = float(np.median(shifts)) if shifts else 0.0
            if STATS is not None:
                STATS.extend((round(pos - shift) in read and abs(pos - shift - round(pos - shift)) < 0.35, v,
                              min([pos - shift - q for q in read], key=abs)) for pos, v in peaks)
            found, sure = [], True
            for pos, v in peaks:
                pos -= shift
                if v < MATCH:
                    continue  # stems, beams and dots match weakly
                if abs(pos - round(pos)) > 0.35:
                    sure = False  # a head off the staff's steps: leave the column
                    break
                found.append(round(pos))
            found = sorted(set(found))
            if (not sure or not found or found == read
                    or any(b - a < 2 for a, b in zip(found, found[1:]))):
                continue

            def blob(pos):
                y = int(middle + (pos + shift) * s / 2)
                patch = blobs[max(0, y - 2):y + 3, x + tw // 2 - 2:x + tw // 2 + 3]
                return np.bincount(patch.ravel()).argmax() if patch.size else 0

            def solid(pos):
                """Shaped like a black head: filled for a space's height (a
                beam is half that), open just above or below, and no wider
                than a head with its stem (a beam carries on)."""
                y = middle + (pos + shift) * s / 2
                c0, c1 = x + int(0.25 * tw), x + int(0.75 * tw)
                inner = clean[int(y - 0.35 * s):int(y + 0.35 * s) + 1, c0:c1]
                above = clean[int(y - 0.95 * s):int(y - 0.7 * s) + 1, c0:c1]
                below = clean[int(y + 0.7 * s):int(y + 0.95 * s) + 1, c0:c1]
                centre, widest = x + tw // 2, 0
                for row in clean[int(y - 0.3 * s):int(y + 0.3 * s) + 1]:
                    if not row[centre]:
                        continue
                    a = b = centre
                    while a > 0 and row[a - 1]:
                        a -= 1
                    while b < len(row) - 1 and row[b + 1]:
                        b += 1
                    widest = max(widest, b - a + 1)
                return (inner.size and inner.mean() >= 0.85 and widest <= 1.7 * s
                        and min(above.mean() if above.size else 1, below.mean() if below.size else 1) <= 0.3)

            # new heads must be joined to a head read (by the stem) and look like one
            anchors = {blob(p) for p in read} - {0}
            if any(blob(p) not in anchors or not solid(p)
                   or _on_sign(x + tw / 2, y0 + middle + (p + shift) * s / 2, s, taken)
                   or _at_tip(x + tw / 2, y0 + middle + (p + shift) * s / 2,
                              [y0 + middle + (q + shift) * s / 2 for q in read], s, taken[1])
                   for p in found if p not in read):
                continue
            # only a head read between two heads a third apart, with nothing
            # where it was read, is split into them: a head Audiveris read
            # is never dropped, and a lone extra head is too often a
            # fingering, a beam's end or a slur's
            split = [p for p in read if p not in found and p - 1 in found and p + 1 in found]
            if not split or set(found) != (set(read) - set(split)) | {q for p in split for q in (p - 1, p + 1)}:
                continue
            out.setdefault((staff["page"], staff["staff"]), []).append(
                (x + float(np.mean([h[4] for h in column])) / 2, read, found))
    return out


def apply(measures, staves, found):
    """Put the noteheads found on the scan into the notes read, before
    pitches are respelled: a note whose heads are the column's heads read
    takes the heads found (new heads aren't tied)."""
    for measure in measures:
        staff = staves[measure["system"]]
        here = found.get((staff["page"], staff["staff"]))
        if not here:
            continue
        read_middle = measure.get("read_middle")
        if read_middle is None:
            from .playback import middle_line
            read_middle = middle_line(measure["clef"])
        s = staff["space"]
        for e in measure["events"]:
            if not e["spelled"] or "px" not in e:
                continue
            positions = [read_middle - (7 * o + "CDEFGAB".index(st)) for st, o, _ in e["spelled"]]
            match = [f for x, read, f in here
                     if abs(x - e["px"] - 0.6 * s) < 1.2 * s and read == sorted(set(positions))]
            if len(match) != 1:
                continue
            ties = dict(zip(positions, e.get("tied") or [False] * len(positions)))
            accs = {p: a for p, (_, _, a) in zip(positions, e["spelled"])}
            e["spelled"], e["tied"] = [], []
            for pos in match[0]:
                d = read_middle - pos
                e["spelled"].append(("CDEFGAB"[d % 7], d // 7, accs.get(pos)))
                e["tied"].append(ties.get(pos, False))
            e["midi"] = [0] * len(e["spelled"])  # set by respell


def _staff_lines(ink, x, staff):
    """Heights of the five staff lines right beside x, from the scan, or
    None where they can't be told apart (a beam lying along them)."""
    s = staff["space"]
    y0 = int(staff["top"] - 1.5 * s)
    strip = ink[y0:int(staff["top"] + 5.5 * s), max(0, int(x - 3 * s)):int(x + 3 * s)]
    rows = np.flatnonzero(strip.mean(axis=1) > 0.6)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []
    lines = [y0 + g.mean() for g in groups if len(g)]
    if len(lines) != 5:
        return None
    gaps = np.diff(lines)
    return lines if gaps.max() < 1.2 * gaps.min() else None


def reheight(omr, pdf, staves):
    """Black noteheads Audiveris put a step off, measured again on the scan.
    Returns {(page, staff): [(x centre, position read, position measured)]}.

    Each head's height is found by matching a notehead template close to
    where Audiveris put it, and measured against the five staff lines right
    beside it (scans bend, so the staff isn't one straight line). Only a
    head Audiveris was unsure of (grade under 0.5) is moved, by one step,
    when it sits squarely on the next line or space: its confident heads
    measure where it put them."""
    doc, cache = pymupdf.open(pdf), {}
    out = {}
    with zipfile.ZipFile(omr) as z:
        sheets = {page: z.read(f"sheet#{page}/sheet#{page}.xml").decode().split("<system ")
                  for page in sorted({s["page"] for s in staves})}
    template = None
    for staff in staves:
        heads = _heads(sheets[staff["page"]][staff["staff"]])
        if not heads:
            continue
        s = staff["space"]
        ink = (page_image(doc, staff["page"], cache) < 140).astype(np.uint8)
        clean = _lines_out(ink, s).astype(np.float32)
        if template is None:
            patches = [cv2.resize(clean[int(y):int(y + h), int(x):int(x + w)], (24, 20))
                       for st in staves for g, _, x, y, w, h in _heads(sheets[st["page"]][st["staff"]])[:60]
                       if g >= 0.7 and st["page"] == staff["page"]]
            if len(patches) < 10:
                continue
            template = np.mean(patches, axis=0).astype(np.float32)
        tw, th = max(5, round(1.2 * s)), max(5, round(s))
        t = cv2.resize(template, (tw, th))
        for grade, pos, x, y, w, h in heads:
            lines = _staff_lines(ink, x + w / 2, staff)
            if lines is None:
                continue
            y0, y1 = int(y + h / 2 - 1.2 * s), int(y + h / 2 + 1.2 * s)
            window = clean[y0:y1 + th, int(x) - 3:int(x) + tw + 3]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            score = cv2.matchTemplate(window, t, cv2.TM_CCOEFF_NORMED)
            r, c = np.unravel_index(score.argmax(), score.shape)
            if score[r, c] < MATCH:
                continue
            step = (lines[4] - lines[0]) / 8
            measured = (y0 + r + th / 2 - lines[2]) / step
            if (grade < 0.5 and abs(measured - round(measured)) <= 0.3
                    and abs(round(measured) - pos) == 1):
                out.setdefault((staff["page"], staff["staff"]), []).append((x + w / 2, pos, round(measured)))
    return out


def move(measures, staves, moved):
    """Put heads re-measured by reheight on their line or space, before
    pitches are respelled."""
    from .playback import middle_line
    for measure in measures:
        staff = staves[measure["system"]]
        here = moved.get((staff["page"], staff["staff"]))
        if not here:
            continue
        read_middle = measure.get("read_middle") or middle_line(measure["clef"])
        s = staff["space"]
        for e in measure["events"]:
            if not e["spelled"] or "px" not in e:
                continue
            spelled = []
            for step, octave, acc in e["spelled"]:
                pos = read_middle - (7 * octave + "CDEFGAB".index(step))
                new = [b for x, a, b in here if a == pos and abs(x - e["px"] - 0.6 * s) < 1.2 * s]
                if len(new) == 1:
                    d = read_middle - new[0]
                    step, octave = "CDEFGAB"[d % 7], d // 7
                spelled.append((step, octave, acc))
            e["spelled"] = spelled


def time_signature_heads(omr, staves, opening):
    """{(page, staff): [x centre]} of hollow heads Audiveris was unsure of
    where a time signature is printed: just after the clef and key on a
    line that opens a movement (`opening`, a set of (page, staff)), before
    any note it read with confidence. Its digits get read as a half note."""
    out = {}
    with zipfile.ZipFile(omr) as z:
        sheets = {page: z.read(f"sheet#{page}/sheet#{page}.xml").decode().split("<system ")
                  for page in sorted({s["page"] for s in staves})}
    for staff in staves:
        if (staff["page"], staff["staff"]) not in opening:
            continue
        system, s = sheets[staff["page"]][staff["staff"]], staff["space"]
        edges = [float(x) + float(w) for x, w in re.findall(
            r'<(?:clef|key)\b[^>]*>\s*<bounds x="([\d.]+)" y="[\d.]+" w="([\d.]+)"', system)]
        if not edges:
            continue
        start = max(edges)
        confident = [float(x) for attrs, x in re.findall(r'<head ([^>]*)>\s*<bounds x="([\d.]+)"', system)
                     if float(re.search(r'\bgrade="([\d.]+)"', attrs).group(1)) >= 0.5]
        first = min(confident, default=float("inf"))
        for attrs, x, w in re.findall(r'<head ([^>]*)>\s*<bounds x="([\d.]+)" y="[\d.]+" w="([\d.]+)"', system):
            grade = float(re.search(r'\bgrade="([\d.]+)"', attrs).group(1))
            if ('NOTEHEAD_VOID' in attrs and grade < 0.5
                    and start <= float(x) <= start + 3 * s and float(x) < first):
                out.setdefault((staff["page"], staff["staff"]), []).append(float(x) + float(w) / 2)
    return out
