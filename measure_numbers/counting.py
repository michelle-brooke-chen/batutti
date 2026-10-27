"""Count the measures of a scanned part from Audiveris output and the page image.

Audiveris supplies staves, barlines and noteheads. On top of that this module
reads rests (and the numbers above multi-bar rests), finds barlines Audiveris
missed or invented, applies 1st/2nd-ending numbering, splits the part into
movements, and flags every bar it is unsure about for the player to check.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rapidocr import RapidOCR

from . import barlines as detect_barlines
from . import digits

DPI = 300  # Audiveris sheet coordinates are at 300 dpi for these scans
MOVEMENT_TITLE = re.compile(r"^\s*(I{1,3}|IV|VI{0,3}|IX|X)\s*[.,]\s*[A-Z][A-Za-z]{2,}")
# a heading that is only a part word and an ordinal ("SECONDE PARTIE", "Part Two")
PART_TITLE = re.compile(r"^\s*(?:([\w.]+)\s+)?(PARTIE|PART|TEIL)(?:\s+([\w.]+))?\s*\.?\s*$", re.I)
ORDINALS = [  # OCR can drop an accent or a letter ("PREMÈRE")
    (1, r"prem\w*re|first|one|erster?|i|1"),
    (2, r"seconde?|deuxi\w*me|two|zweiter?|ii|2"),
    (3, r"troisi\w*me|third|three|dritter?|iii|3"),
]
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
OCR_TRUST = 0.9  # below this, a rest count is flagged for the player to confirm
SURE_NUMBER = 0.9  # an unambiguous number this confident outranks unclear rest shapes
MIN_SOLID = 0.8  # share of its bounding box a rest symbol fills (in the ink)
MAX_REST = 199  # larger "numbers" are misread text such as tempo markings
WIDE = 1.8  # a bar this many times the typical width is flagged for checking
SUSPECT = 1.4  # bars wider than this get a second barline search
HEAD_CLEARANCE = 0.8  # staff spaces between a notehead centre and a real barline
MIN_HEAD_GRADE = 0.3  # Audiveris confidence below which a notehead may be a slur end

_engine = RapidOCR()
_ocr_cache = {}  # image hash -> (texts, scores, boxes); saved per part between runs


class _Result:
    def __init__(self, txts, scores, boxes):
        self.txts, self.scores, self.boxes = txts, scores, boxes


def ocr(image):
    """RapidOCR with a cache: re-checking a part re-reads the same crops."""
    key = hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()
    if key not in _ocr_cache:
        r = _engine(image)
        boxes = None if r.boxes is None else [[list(map(float, p)) for p in b] for b in r.boxes]
        _ocr_cache[key] = (list(r.txts or ()), [float(x) for x in r.scores or ()], boxes)
    return _Result(*_ocr_cache[key])


# ---------- reading Audiveris output ----------

def bounds(el):
    b = el.find("bounds")
    return tuple(float(b.get(k)) for k in ("x", "y", "w", "h")) if b is not None else None


def read_systems(xml):
    """Barline x positions, notehead centres and symbol boxes per system."""
    root = ET.fromstring(xml)
    systems = []
    for sysm in root.iter("system"):
        bars = sorted(
            bounds(b) for b in sysm.iter("barline") if b.get("shape") != "DUMMY_BARLINE"
        )
        staff = next(sysm.iter("staff"))
        ys = [np.mean([float(pt.get("y")) for pt in line.iter("point")])
              for line in staff.find("lines")]
        geometry = {
            "top": min(ys), "bottom": max(ys), "space": (max(ys) - min(ys)) / 4,
            "left": float(staff.get("left")), "right": float(staff.get("right")),
        }
        # double and repeat barlines arrive as several glyphs, and Audiveris
        # sometimes reports one barline twice; no real bar is under 2.5 spaces wide
        groups = []
        for x, y, w, h in bars:
            if groups and x - groups[-1]["x1"] < 2.5 * geometry["space"]:
                groups[-1]["x1"] = x + w
            else:
                groups.append({"x0": x, "x1": x + w, "y": y + h / 2})
        heads = [b for b in (bounds(h) for h in sysm.iter("head")) if b]
        # faint noteheads are often slur ends touching a barline; they must not
        # make that barline look like a stem
        solid_heads = [bounds(h) for h in sysm.iter("head")
                       if bounds(h) and float(h.get("grade")) >= MIN_HEAD_GRADE]
        stems = [b for b in (bounds(t) for t in sysm.iter("stem")) if b]
        # symbols that must not be mistaken for rests when we look at the image
        masks = [
            bounds(el) for el in sysm.find(".//inters")
            if el.tag in ("clef", "key-alter", "time-number", "head", "stem", "beam",
                          "slur", "dynamics", "fermata", "fermata-arc", "fermata-dot")
            and bounds(el)
        ]
        preamble = [
            bounds(el) for el in sysm.find(".//inters")
            if el.tag in ("clef", "key-alter", "time-number") and bounds(el)
        ]
        # fermatas, for telling where the conductor restarts the music
        fermatas = [bounds(el) for el in sysm.find(".//inters")
                    if el.tag in ("fermata", "fermata-arc") and bounds(el)]
        systems.append({"staff": geometry, "bars": groups, "heads": heads, "stems": stems,
                        "fermatas": fermatas,
                        "solid_heads": solid_heads,
                        "masks": masks, "preamble": preamble})
    return systems


def bar_segments(sysm):
    """(left, right) of each bar on a staff, from the staff start and barlines."""
    st = sysm["staff"]
    edges = [st["left"]] + [b["x0"] for b in sysm["bars"]]
    if len(edges) > 1 and edges[1] - edges[0] < 3 * st["space"]:
        edges.pop(1)  # barline opening the system
    # notes after the last barline mean its closing barline was missed
    # (a courtesy key signature there has no noteheads, so it is ignored)
    last = sysm["bars"][-1]["x1"] if sysm["bars"] else st["left"]
    if any(hx > last + st["space"] for hx, hy, hw, hh in sysm["heads"]):
        edges.append(st["right"])
    return list(zip(edges, edges[1:]))


def merge_split_bars(segments, sysm, gray, bare, ink):
    """Rejoin a bar split at the stem of a note Audiveris didn't recognise.

    Two neighbouring "bars" with no notes or rests that are each narrow, but
    together about as wide as the staff's typical bar, are one bar (tied half notes,
    say, whose stems were taken for a barline). Rests whose symbol wasn't
    recognised still show a thick stroke or a number above, so they stay.
    """
    st, s = sysm["staff"], sysm["staff"]["space"]

    def empty(x0, x1):
        return (not any(x0 < hx + hw / 2 < x1 for hx, hy, hw, hh in sysm["heads"])
                and not classify_rest(bare, st, x0, x1, sysm["masks"], sysm["preamble"], ink)
                and rest_stroke(ink, st, x0, x1) < 1.5
                and read_number(gray, st, x0, x1)[0] is None)

    normal = median(x1 - x0 for x0, x1 in segments[1:])  # the staff's typical bar
    out, i = [], 0
    while i < len(segments):
        if 0 < i < len(segments) - 1:
            (a0, a1), (b0, b1) = segments[i], segments[i + 1]
            if (normal and a1 - a0 < 0.7 * normal and b1 - b0 < 0.7 * normal
                    and 0.6 * normal <= b1 - a0 <= 1.4 * normal
                    and empty(a0, a1) and empty(b0, b1)):
                out.append((a0, b1))
                i += 2
                continue
        out.append(segments[i])
        i += 1
    return out


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def missed_note(barlines):
    """Flag text for barlines added inside a bar; the count already includes them."""
    return (f"found {len(barlines)} barline(s) Audiveris missed and split the bar"
            " there (already included in the count) - please check")


def piece(x0, x1, count, plays, rest):
    """One stretch of a bar on the timeline: where it is, its measures, and
    whether the part plays there (`rest` of its measures are rests)."""
    return {"x0": x0, "x1": x1, "count": count, "plays": plays, "rest": rest}


def add_pieces(timeline, place, pieces):
    """Add a bar to the timeline, one entry per piece (a bar holding a missed
    barline has several), numbering the pieces in order. Returns the entries."""
    added, measure = [], place["measure"]
    for k, p in enumerate(pieces):
        entry = {**place, **p, "measure": measure}
        if len(pieces) > 1:
            entry["piece"] = k + 1
        measure += p["count"]
        timeline.append(entry)
        added.append(entry)
    return added


def key_change(heads, start, space):
    """Whether a bar's "noteheads" are one stacked column just after `start` (its
    barline, or the end of its clef and key), as key-change accidentals or
    time-signature digits look when read as notes."""
    xs = [hx + hw / 2 for hx, hy, hw, hh in heads]
    return bool(xs) and max(xs) - min(xs) < space and max(xs) < start + 4 * space


def rest_stroke(ink, staff, x0, x1):
    """Longest thick horizontal stroke around the middle line, in staff spaces.

    A multi-bar rest leaves one even when its shape isn't recognised; notes and
    cue notes don't.
    """
    s, mid = staff["space"], staff["top"] + 2 * staff["space"]
    band = ink[int(mid - 0.6 * s):int(mid + 0.6 * s), int(x0 + 0.5 * s):int(x1 - 0.5 * s)] > 0
    best = run = 0
    for thick in band.sum(axis=0) >= 0.35 * s:
        run = run + 1 if thick else 0
        best = max(best, run)
    return best / s


def could_be_stem(x, staff, heads):
    """Could a staff-high stroke at x be the stem of a nearby notehead?

    An up-stem leaves the right side of a notehead at its bottom end; a
    down-stem leaves the left side of a notehead at its top end. A stroke
    beside a notehead in any other arrangement is a barline. Stems are about
    3.5 spaces long, so a head outside the staff can't own a stroke spanning
    all of it (such "heads" are often slurs or trill lines).
    """
    top, bottom, s = staff["top"], staff["bottom"], staff["space"]
    for hx, hy, hw, hh in heads:
        centre_x, centre_y = hx + hw / 2, hy + hh / 2
        if abs(x - centre_x) >= HEAD_CLEARANCE * s:
            continue
        if centre_y < top - 0.3 * s or centre_y > bottom + 0.3 * s:
            continue
        if x > centre_x and centre_y > bottom - 1.2 * s:
            return True
        if x < centre_x and centre_y < top + 1.2 * s:
            return True
    return False


# ---------- looking at the image ----------

def preamble_end(preamble, x0, x1, space):
    """Where the clef / key / time signature opening a bar ends (x0 if none).

    Only symbols running on from the bar's start count; a clef change later in
    the bar is not part of it.
    """
    end = x0
    for px, py, pw, ph in sorted(preamble):
        if x0 <= px < x1 and px <= end + 1.5 * space:
            end = max(end, px + pw)
    return end


def classify_rest(bare, staff, x0, x1, masks, preamble=(), ink=None):
    """Find rest symbols in a bar as (kind, left, right), from shape and position.

    Values are in staff spaces. Old engravings draw a multi-bar rest as a
    thick block about two spaces tall, a 4-bar rest as a narrow block filling
    two spaces, a 2-bar rest as a block filling one space, and a whole rest as a half-space block hanging from the 4th line
    (up to 2.8 spaces wide in some editions).
    """
    top, s = staff["top"], staff["space"]
    # skip the barlines at either edge and any clef / key / time signature at the
    # start (a clef change at the end of the bar stays: rests come before it)
    x0 = max(x0 + 0.6 * s, preamble_end(preamble, x0, x1, s) + 0.5 * s)
    x1 = x1 - 0.6 * s
    if x1 - x0 < s:
        return []
    y0, y1 = int(top - 0.3 * s), int(staff["bottom"] + 0.3 * s)
    region = bare[y0:y1, int(x0):int(x1)].copy()
    for mx, my, mw, mh in masks:
        rx0, rx1 = int(mx - x0 - 4), int(mx + mw - x0 + 4)
        ry0, ry1 = int(my - y0 - 4), int(my + mh - y0 + 4)
        if rx1 > 0 and rx0 < region.shape[1] and ry1 > 0 and ry0 < region.shape[0]:
            region[max(0, ry0):ry1, max(0, rx0):rx1] = 0

    # strip scraps thinner than a third of a space (leftover bits of staff line
    # can fuse with a rest and distort its size); every rest symbol is thicker
    thin = max(2, int(0.3 * s))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((thin, thin), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(region)
    pieces = [
        [x, y, x + w, y + h, area] for x, y, w, h, area in stats[1:]
        if area >= 0.05 * s * s and h >= 0.25 * s  # drop staff-line remnants
    ]
    # removing staff lines cuts one symbol into stacked pieces: re-join pieces
    # that overlap horizontally and sit directly above one another
    boxes = []
    for piece in sorted(pieces, key=lambda b: (b[0], b[1])):
        for box in boxes:
            overlap = min(box[2], piece[2]) - max(box[0], piece[0])
            narrow = min(box[2] - box[0], piece[2] - piece[0])
            gap = max(box[1], piece[1]) - min(box[3], piece[3])
            if overlap >= 0.5 * narrow and gap <= 0.4 * s:
                box[:] = [min(box[0], piece[0]), min(box[1], piece[1]),
                          max(box[2], piece[2]), max(box[3], piece[3]), box[4] + piece[4]]
                break
        else:
            boxes.append(piece)

    found = []
    for x, y, xr, yb, area in sorted(boxes):
        span = (x0 + x, x0 + xr)
        w, h = (xr - x) / s, (yb - y) / s
        rel_top, rel_bottom = (y + y0 - top) / s, (yb + y0 - top) / s
        if area < 0.15 * s * s:
            continue
        # rests are solid blocks; tremolo slashes and hollow noteheads (which
        # Audiveris can miss) leave white gaps. Measured in the original ink,
        # where staff lines crossing a rest are still black.
        source = ink if ink is not None else bare
        block = source[y0 + y:y0 + yb, int(x0) + x:int(x0) + xr]
        if block.size and (block > 0).mean() < MIN_SOLID:
            continue
        # a whole rest hangs from the 4th line and can be drawn wide (2.5 spaces in
        # some editions); a multi-bar rest sits lower, centred on the middle line
        # a 2-bar block is narrow (under 1 space); a whole rest is wider or flatter
        if 0.5 <= w <= 1.0 and 0.6 <= h <= 1.3 and 0.8 <= rel_top <= 1.3:
            found.append(("two-bar", *span))
        elif 0.5 <= w <= 2.8 and 0.3 <= h <= 0.8 and 0.7 <= rel_top <= 1.3:
            # (a scrap of staff line fused to it can make it look a little taller)
            found.append(("whole", *span))
        elif 0.5 <= w <= 1.6 and 1.6 <= h <= 2.4 and 0.8 <= rel_top <= 1.5:
            # a block filling two spaces (removing staff lines can trim its top)
            found.append(("four-bar", *span))
        elif w >= 2.5 and 0.3 <= h <= 2.4 and 1.2 <= (rel_top + rel_bottom) / 2 <= 2.8:
            found.append(("multi", *span))
        elif 0.5 <= w <= 1.6 and 0.75 <= h <= 1.3 and 0.8 <= rel_top <= 1.3:
            found.append(("two-bar", *span))

    # one rest can be found as two touching pieces
    merged = []
    for kind, l, r in found:
        if merged and merged[-1][0] == kind and l - merged[-1][2] < 0.3 * s:
            merged[-1] = (kind, merged[-1][1], max(r, merged[-1][2]))
        else:
            merged.append((kind, l, r))
    found = merged

    # a long multi-bar rest can be erased along with the staff lines (it is a
    # long horizontal run too); look for it in the original ink, keeping only
    # strokes thicker than a staff line
    if ink is not None and not any(k == "multi" for k, _, _ in found):
        band = ink[y0:y1, int(x0):int(x1)].copy()
        for mx, my, mw, mh in masks:
            band[max(0, int(my - y0 - 4)):max(0, int(my + mh - y0 + 4)),
                 max(0, int(mx - x0 - 4)):max(0, int(mx + mw - x0 + 4))] = 0
        kernel = np.ones((max(2, int(0.4 * s)), int(2.5 * s)), np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel))
        for x, y, w, h, _ in stats[1:]:
            centre = (y + h / 2 + y0 - top) / s
            if w >= 2.5 * s and h <= 1.3 * s and 1.4 <= centre <= 2.6:
                found.append(("multi", x0 + x, x0 + x + w))
                break
    return found


def read_number(gray, staff, x0, x1, multi=False, spans=(), below=False, deep=False):
    """Digits printed above a bar, read from several preparations of the crop.

    Old engravings make 3 and 8 look alike, so each preparation is read and
    the most confident reading wins (misreads score well below true ones).
    `spans` are narrower x ranges to read as well, such as just above a rest
    symbol, clear of rehearsal letters. Over a multi-bar rest (`multi`), 1 is
    never the count, so it is dropped. With `below`, digits under the staff are
    read instead (running counts under cue bars); `deep` reaches lower, for a
    rest's count printed under the staff because a rehearsal number took its
    place above.
    Returns the reading, its confidence, and every distinct reading.
    """
    top, s = staff["top"], staff["space"]
    if below:
        rows = (int(top + 4.3 * s), int(top + (8.2 if deep else 7.5) * s))
    else:
        rows = (int(top - 3.5 * s), int(top - 0.2 * s))
    variants = []
    for a, b in [(x0, x1), *spans]:
        crop = gray[rows[0]:rows[1], int(a):int(b)]
        if crop.size == 0:
            continue
        x2 = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        variants += [
            crop,
            x2,
            cv2.dilate(x2, np.ones((2, 2), np.uint8)),  # thins the black strokes
            cv2.threshold(x2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        ]
    readings = digits_in(variants, multi)
    if not readings and multi:
        # faint, blurred print: enlarge further and thicken the strokes
        faint = [cv2.erode(cv2.resize(gray[rows[0]:rows[1], int(a):int(b)], None, fx=3, fy=3,
                                      interpolation=cv2.INTER_CUBIC), np.ones((3, 3), np.uint8))
                 for a, b in [(x0, x1), *spans] if b > a]
        readings = digits_in(faint, multi)
    if not readings:
        return None, 0.0, []
    number, score = max(readings, key=lambda r: r[1])
    return number, score, sorted({n for n, _ in readings})


def digits_in(images, multi=False):
    """(number, confidence) for every number the OCR reads in the images."""
    readings = []
    for img in images:
        img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        result = ocr(img)
        for text, score in zip(result.txts or (), result.scores or ()):
            # a number can run into a tempo word ("Meno.23"), so take each
            # group of up to three digits
            for group in re.findall(r"\d+", text):
                if len(group) <= 3 and int(group) <= MAX_REST:
                    readings.append((int(group), float(score)))
    if multi:
        readings = [(n, sc) for n, sc in readings if n > 1]
    return readings


def staff_text(gray, staves):
    """Read the text between staves once: movement titles {staff: (number, text)}
    and the numbers just over each staff [(staff, x centre, number)].

    A title is a Roman numeral and a full stop followed by a capitalised word
    ("II. Ballade"); a row of bowing marks can read as "V.vO", so the word matters.
    A movement the part doesn't play is marked "tacet" ("II. Ballade tacet."),
    often at the foot of a page; it starts no staff, so it is skipped.
    A larger work's halves are headed by a part word ("SECONDE PARTIE"); the
    ordinal may be read as its own line, and an unread one is left as None
    (the next movement). The numbers within 4.5 spaces over a staff include
    its rehearsal numbers and rest counts.
    """
    titles, marks = {}, []
    edges = [0] + [int(st["bottom"] + 2 * st["space"]) for st in staves]
    for i, st in enumerate(staves):
        s = st["space"]
        y0, y1 = min(edges[i], int(st["top"] - 4.5 * s)), int(st["top"] - 0.2 * s)
        if y1 - y0 < s:
            continue
        result = ocr(gray[max(0, y0):y1])
        lines = []
        for text, box in zip(result.txts or (), result.boxes or ()):
            ys = [max(0, y0) + p[1] for p in box]
            if min(ys) < st["top"] - 1.5 * s:  # titles sit clear of the staff
                lines.append(text.strip())
            if max(ys) > st["top"] - 4.5 * s and (
                    match := re.fullmatch(r"\W*(\d{1,3})\W*", text)):
                marks.append((i, sum(p[0] for p in box) / len(box), int(match.group(1))))
        if int(st["top"] - 1.5 * s) - edges[i] < 3 * s:
            continue  # no room for a title
        for k, text in enumerate(lines):
            if (match := MOVEMENT_TITLE.match(text)) and "tacet" not in text.lower():
                titles[i] = (ROMAN[match.group(1)], text)
            elif (match := PART_TITLE.match(text)) and match.group(2)[0].isupper():
                words = [w for w in match.group(1, 3) if w]
                if not words and k and ordinal(lines[k - 1]):
                    words, text = [lines[k - 1]], f"{lines[k - 1]} {text}"
                numbers = [ordinal(w) for w in words]
                if None not in numbers:  # any other word must be an ordinal
                    titles[i] = (numbers[0] if numbers else None, text)
    return titles, marks


def ordinal(word):
    """The number an ordinal word stands for ("Seconde" -> 2), or None."""
    word = word.strip(" .").lower()
    return next((n for n, pattern in ORDINALS if re.fullmatch(pattern, word)), None)


def title_word(title):
    """A movement title's name without its numeral: "II. Ballade." -> "ballade"."""
    match = MOVEMENT_TITLE.match(title or "")
    return (title[match.end(1):].strip(" .,").split() or [""])[0].lower() if match else ""


def find_endings(ink, gray, staff, starts, first_barline):
    """1st/2nd-ending brackets over a staff: [{x0, x1, number}], left to right.

    A bracket is a long thin line 1.5-4.5 spaces above the staff (beams are
    thicker; hairpins sit lower). A labelled bracket has a hook down at its
    left end with "1." or "2." inside, and begins where a bar begins (`starts`,
    or anywhere in the first bar after the clef). Two brackets side by side
    are read as the 1st and 2nd endings, which also covers editions that leave
    the 1st ending unlabelled.
    """
    s, top = staff["space"], staff["top"]
    y0, y1 = int(top - 4.5 * s), int(top - 1.5 * s)
    band = ink[y0:y1, int(staff["left"]):int(staff["right"])]
    lines = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((1, int(3 * s)), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines)
    pieces = sorted((int(x + staff["left"]), int(y + y0), int(w))
                    for x, y, w, h, _ in stats[1:] if h <= 0.35 * s)
    # a scan gap can break one bracket line into pieces: rejoin tiny gaps only
    # (a 1st and 2nd ending can sit as little as half a space apart)
    brackets = []
    for x, y, w in pieces:
        if brackets and abs(y - brackets[-1][1]) <= 0.3 * s and x - brackets[-1][2] <= 0.3 * s:
            brackets[-1][2] = max(brackets[-1][2], x + w)
        else:
            brackets.append([x, y, x + w])

    def hook(x, y):
        area = ink[y:int(y + 1.2 * s), max(0, x - 3):int(x + 0.4 * s)]
        return area.size and (area.max(axis=1) > 0).mean() >= 0.5

    def stems_to_staff(left, line_y, right):
        """Strokes running unbroken from the line down to the staff.

        A beam's stems reach down to noteheads on the staff; a bracket's hooks
        stop well short, and bowing marks under it are separate symbols.
        """
        drop = ink[int(line_y + 0.4 * s):int(top - 0.1 * s), left:right]
        if drop.size == 0:
            return 0
        full = drop.min(axis=0) > 0
        return int(np.count_nonzero(np.diff(full.astype(np.int8)) == 1) + full[:1].sum())

    endings = []
    for left, line_y, right in brackets:
        if right - left < 6 * s or stems_to_staff(left, line_y, right) >= 2:
            continue  # too short, or a beam with stems running down to its notes
        at_bar = left < first_barline or any(abs(left - x0) < 1.5 * s for x0 in starts)
        number = ending_number(gray, left, line_y, s) if at_bar and hook(left, line_y) else None
        endings.append({"x0": left, "x1": right, "number": number})
    # two brackets side by side are the 1st and 2nd endings once either label
    # reads correctly (a bold "2" against a thick repeat barline is easily
    # misread, and some editions leave the 1st unlabelled); two unlabelled
    # lines side by side are something else, such as beams
    for a, b in zip(endings, endings[1:]):
        if b["x0"] - a["x1"] < 2 * s and (a["number"] == 1 or b["number"] == 2):
            a["number"], b["number"] = 1, 2
    return [e for e in endings if e["number"] is not None]


def ending_number(gray, left, line_y, s):
    """Read the "1." or "2." just inside an ending bracket's hook.

    Pencil marks are grey and printed ink is black, so only black is kept.
    OCR reads a lone 2 well but often misses a lone 1, so if it finds nothing
    the digit's shape decides: a printed 1 has a vertical stem running nearly
    its full height; no column of a 2 does.
    """
    # start right of the hook, which OCR would otherwise read as a "1"
    crop = gray[int(line_y + 0.25 * s):int(line_y + 2.1 * s), int(left + 0.35 * s):int(left + 2.4 * s)]
    black = np.where(crop < 90, 0, 255).astype(np.uint8)
    big = cv2.resize(black, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    result = ocr(cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255))
    for text in result.txts or ():
        digit = re.sub(r"\D", "", text)[:1]
        if digit in ("1", "2"):
            return int(digit)
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - black)
    blobs = [st for st in stats[1:]  # digit-sized shapes only
             if 0.6 * s <= st[3] <= 1.5 * s and st[2] <= 1.2 * s]
    if not blobs:
        return None
    x, y, w, h, _ = max(blobs, key=lambda b: b[4])
    glyph = (255 - black)[y:y + h, x:x + w] > 0
    stem = glyph.mean(axis=0).max()  # best column's share of the glyph height
    return 1 if stem > 0.8 else 2


# ---------- putting it together ----------

VALUE = {"four-bar": 4, "two-bar": 2, "whole": 1}


def decide(kinds, number, score, readings):
    """How many measures a rest-only bar is worth, and what to flag."""
    flags = []
    if len(readings) > 1:
        flags.append(f"number above read as {' or '.join(map(str, readings))} - please confirm")
    if "multi" in kinds:
        if number is None:
            return 1, ["multi-bar rest but no number read - enter the count"]
        if score < OCR_TRUST and len(readings) == 1:
            flags.append(f"count read as {number} (confidence {score:.2f}) - please confirm")
        return number, flags
    sure = number is not None and len(readings) == 1 and score >= SURE_NUMBER
    if not kinds:
        if sure and number > 1:
            return number, flags + [f"no rest symbol recognised, but the number above reads"
                                    f" {number} - counted as {number}; please check"]
        return 1, flags + ["bar with no notes and no recognised rest - please check"]
    count = sum(VALUE[k] for k in kinds)
    # an old 3 is a 2-bar and a 1-bar block, which can read as two 2-bar blocks
    two_part = len(kinds) > 1 and number == count - 1 and score >= SURE_NUMBER
    if (sure or two_part) and number < count:
        # rest shapes were over-read; a larger number is usually a running count
        # over cue bars, so the shapes stand then
        return number, flags + [f"rest symbols add up to {count} but the number above"
                                f" reads {number} - counted as {number}; please check"]
    if number is not None and number != count and len(readings) == 1:
        flags.append(f"rest symbols add up to {count} but the number above reads {number}")
    return count, flags


def rising_chain(marks):
    """The longest run of marks whose numbers rise in reading order."""
    best = [[m] for m in marks]
    for i, m in enumerate(marks):
        for j in range(i):
            if marks[j][-1] < m[-1] and len(best[j]) + 1 > len(best[i]):
                best[i] = best[j] + [m]
    return max(best, key=len) if best else []


def rehearsal_over_rests(rest_numbers, marks, movements, staves, timeline, doc):
    """Multi-bar rests whose "count" is the rehearsal number printed over the
    start of the bar. Rehearsal numbers rise through a movement (60, 61, 62),
    so a number is taken as one when it is in that rising chain (mostly steps
    of one, at most two a staff on average, unlike rest counts and running bar
    counts), next to its neighbours, and sits at the bar's start rather than
    over the rest. The count is then the one other number over the bar, or the
    one under the staff, where such editions print it. If there is none, the
    rest is probably as long as the mark's number, so it is kept and the player
    asked. Returns the rests changed."""
    geometry = {(st["page"], st["staff"]): st for st in staves}
    chain = {}
    for movement in {m[0] for m in marks}:
        found = sorted((m for m in marks if m[0] == movement), key=lambda m: m[1:4])
        run = rising_chain(found)
        steps = sum(b[-1] - a[-1] == 1 for a, b in zip(run, run[1:]))
        if steps < max(5, len(run) / 2) or len(run) > 2 * len({m[1:3] for m in run}):
            # rest counts rising by chance, or running bar counts (several a
            # staff), not rehearsal numbers
            continue
        for k, m in enumerate(run):
            near = [o[-1] for o in run[max(0, k - 1):k + 2] if o is not m]
            if near and all(abs(n - m[-1]) <= 3 for n in near):
                chain[m[1:3] + (m[-1],)] = m[3]  # (page, staff, number) -> x
        # a misread mark can take its neighbour's number (95 read as 96), so the
        # chain keeps one of the two; the other is a mark too
        for m in found:
            k = sum(o[1:4] <= m[1:4] for o in run)  # marks of the run up to m
            if any(o[-1] == m[-1] and o[1:3] + (o[-1],) in chain
                   for o in run[max(0, k - 1):k + 1]):
                chain.setdefault(m[1:3] + (m[-1],), m[3])
    pages, changed = {}, []
    for rest, bar, place, _ in rest_numbers:
        number = rest.get("number")
        x = chain.get((place["page"], place["staff"], number))
        if rest is not bar or rest["symbols"] != ["multi"] or x is None:
            continue
        st = geometry[(place["page"], place["staff"])]
        # a staff's first bar starts after the clef, under a mark printed over it
        start = st["left"] if place["bar"] == 1 else place["x0"] - 2 * st["space"]
        if not start <= x <= (place["x0"] + place["x1"]) / 2:
            continue  # centred over the rest: a count
        here = [m for m in marks if m[1:3] == (place["page"], place["staff"])
                and start <= m[3] <= place["x1"]]
        same = [m[3] for m in here if m[-1] == number]
        if max(same) - min(same) > 1.5 * st["space"]:
            continue  # printed twice: the mark, and the same count over the rest
        beside = {m[-1] for m in here if m[-1] not in (number, 1)
                  and place["x0"] <= m[3] <= place["x1"]
                  and m[1:3] + (m[-1],) not in chain}  # the next bar's mark isn't a count
        count = beside.pop() if len(beside) == 1 else None
        where = "beside it"
        if count is None:
            if place["page"] not in pages:
                pix = doc[place["page"] - 1].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
                pages[place["page"]] = np.frombuffer(pix.samples, np.uint8).reshape(
                    pix.height, pix.width)
            count = read_number(pages[place["page"]], st, place["x0"], place["x1"],
                                multi=True, below=True, deep=True)[0]
            where = "printed under the staff"
        if count is None or count == number:
            # no other count printed: the rest is likely as long as the mark's number
            bar["flags"].append(f"{number} over this rest is also rehearsal number {number}"
                                f" - please confirm the rest is {number} bars")
            continue
        bar["flags"].append(f"{number} over this rest is rehearsal number {number};"
                            f" counted the {count} {where} - please confirm")
        rest["number"] = count
        shift(bar, place, count - bar["count"], movements, staves, timeline)
        changed.append(rest)
    return changed


def reread_by_shape(rest_numbers, movements, staves, timeline):
    """Correct multi-bar rest counts the part's own digit shapes read differently."""
    pairs = [(rest, shapes) for rest, _, _, shapes in rest_numbers]
    # record the shapes' own reading on each whole-bar rest, as a second opinion
    # the review agent must agree with before changing a count
    for i, seen in digits.confident(pairs).items():
        rest, bar, _, _ = rest_numbers[i]
        if rest is bar:
            bar["shape"] = seen
    changes = digits.recheck(pairs)
    for i, (seen, margin) in changes.items():
        rest, bar, place, _ = rest_numbers[i]
        if rest.get("confirm"):
            # the shapes settle an ambiguous number over an unrecognised rest
            delta = seen - bar["count"]
            bar["flags"].append(f"number confirmed as {seen} from this part's digit shapes"
                                f" - counted as {seen}; please confirm")
            place["rest"] = seen
        elif rest["symbols"] == ["multi"]:
            # a whole bar's count changes to the new number; one of two rests in a
            # bar changes by the difference
            delta = seen - (bar["count"] if rest is bar else rest["number"])
            said = (f"text reader said {rest['number']}" if rest["number"] is not None
                    else "text reader found none")
            bar["flags"] = [f for f in bar["flags"]
                            if not f.startswith("multi-bar rest but no number")]
            bar["flags"].append(f"number read as {seen} from this part's digit shapes"
                                f" ({said}) - please confirm")
            rest["number"] = seen
        else:
            continue
        shift(bar, place, delta, movements, staves, timeline)


def shift(bar, place, delta, movements, staves, timeline):
    """Change a bar's count by delta; everything after it in the movement moves."""
    bar["count"] += delta
    place["count"] += delta
    if not place["plays"] and place["rest"]:
        place["rest"] = place["count"]
    after = False
    for t in timeline:
        if t is place:
            after = True
        elif after and t["movement"] == place["movement"]:
            t["measure"] += delta
    for st in staves:
        if st["movement"] != place["movement"]:
            continue
        if (st["page"], st["staff"]) == (place["page"], place["staff"]):
            st["measures"] += delta
            st["last_measure"] += delta
        elif (st["page"], st["staff"]) > (place["page"], place["staff"]):
            st["first_measure"] += delta
            st["last_measure"] += delta
    next(m for m in movements if m["title"] == place["movement"])["measures"] += delta


def count_part(omr_path, pdf_path, out_dir, progress=None):
    """Count measures; write and return measure_count.json's contents.

    `progress(page, pages)` is called as each page is finished.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "ocr_cache.json"
    if cache_path.exists():
        _ocr_cache.update({k: tuple(v) for k, v in json.loads(cache_path.read_text()).items()})
    doc = pymupdf.open(pdf_path)
    movements, report, staves, notes = [], [], [], []
    timeline = []  # every bar in reading order: its numbers and whether the part plays
    # numbered rests for the digit-shape pass: (the rest's reading, its report
    # entry, its timeline entry, digit shapes); a bar holding two rests has two
    rest_numbers = []
    marks = []  # numbers over each staff, for finding rehearsal numbers
    unread_pages = []  # pages Audiveris couldn't read; their bars aren't counted
    with zipfile.ZipFile(omr_path) as z:
        read = set(z.namelist())
        for page in range(1, doc.page_count + 1):
            if f"sheet#{page}/sheet#{page}.xml" not in read:
                unread_pages.append(page)
                continue
            pix = doc[page - 1].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            ink = detect_barlines.binarize(gray)
            _, lines = detect_barlines.find_staves(ink)
            bare = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 1), np.uint8)))
            systems = read_systems(z.read(f"sheet#{page}/sheet#{page}.xml").decode())
            titles, page_marks = staff_text(gray, [sy["staff"] for sy in systems])
            # typical bar width on the page (first bars hold the clef, so skip them);
            # a staff's own median fails when half its bars are merged
            page_typical = median(
                x1 - x0 for sy in systems for x0, x1 in bar_segments(sy)[1:])

            for i, sysm in enumerate(systems):
                st = sysm["staff"]
                # a new movement needs a later numeral (after II, III or IV, since
                # parts skip the movements they don't play)
                number, title = titles.get(i, (None, None))
                if title and number is None:  # a part heading with an unread ordinal
                    number = movements[-1]["number"] + 1 if movements else 1
                current = title_word(movements[-1]["title"]) if movements else ""
                if number and current and number <= movements[-1]["number"] and (
                        title_word(title) != current):
                    # a misread numeral ("I. Ballade" after the Intermezzo)
                    number = movements[-1]["number"] + 1
                    numeral = next(k for k, v in ROMAN.items() if v == number)
                    title = MOVEMENT_TITLE.sub(lambda m: m.group(0).replace(
                        m.group(1), numeral, 1), title, count=1)
                if not movements or (number or 0) > movements[-1]["number"]:
                    movements.append({"title": title or "(start)", "number": number or 1,
                                      "measures": 0, "start": f"p{page} staff {i + 1}"})
                marks += [(movements[-1]["title"], page, i + 1, x, n)
                          for k, x, n in page_marks if k == i]
                segments = merge_split_bars(bar_segments(sysm), sysm, gray, bare, ink)
                typical = page_typical or (segments[-1][1] - segments[0][0])
                staff_typical = median(x1 - x0 for x0, x1 in segments[1:]) or typical
                # second opinion from our own barline detector
                geometry = {k: int(v) for k, v in st.items()}
                # A stroke clear of noteheads is a barline Audiveris missed. One right
                # beside a notehead may be a stem instead, so it is only flagged.
                # (Audiveris's own stems can't be used to decide: the barlines it
                # misses are typically ones it took for stems.)
                second = [
                    (c["x"], not could_be_stem(c["x"], st, sysm["solid_heads"]))
                    for c in detect_barlines.find_barlines(ink, bare, geometry)
                ]
                # Audiveris sometimes reports a note stem as a barline, leaving a sliver
                # of a bar. A barline never runs past the staff; a stem running on to its
                # beam does. So a barline that leaves a bar under half the usual width
                # is kept only if our own detector also sees a staff-high barline there.
                confirmed = [c["x"] for c in
                             detect_barlines.find_barlines(ink, bare, geometry, check_ends=False)]
                for k in range(len(segments) - 1, 0, -1):
                    edge = segments[k][0]
                    narrow = min(segments[k - 1][1] - segments[k - 1][0],
                                 segments[k][1] - segments[k][0]) < 0.5 * staff_typical
                    if narrow and not any(abs(edge - x) < 0.6 * st["space"] for x in confirmed):
                        segments[k - 1:k + 1] = [(segments[k - 1][0], segments[k][1])]
                        notes.append({"page": page, "staff": i + 1, "bar": k,
                                      "movement": movements[-1]["title"],
                                      "note": "ignored a reported barline that runs past the"
                                              " staff (likely a note stem)"})
                # inside very wide bars, also accept strokes with a slur touching an end
                loose = [
                    c["x"] for c in detect_barlines.find_barlines(ink, bare, geometry, check_ends=False)
                    if not could_be_stem(c["x"], st, sysm["solid_heads"])
                ]
                # a closing barline Audiveris missed: the last bar may hold only rests,
                # so the notes-after-the-last-barline rule can't see it
                closing = [x for x, _ in second if x > st["right"] - 1.5 * st["space"]]
                if segments and closing and segments[-1][1] < st["right"] - 3 * st["space"]:
                    # end the bar where the closing (possibly double) barline starts
                    segments.append((segments[-1][1], min(closing)))

                clef_end = preamble_end(sysm["preamble"], st["left"], st["right"], st["space"])

                def second_opinion(x0, x1, first_bar=False):
                    """Barlines Audiveris missed inside a bar: (split points, flags).

                    In the first bar the clef and key make the bar wide, so width
                    is measured from where they end.
                    """
                    pad = 2.5 * st["space"]  # no real bar is narrower than this
                    inside = [(x, clear) for x, clear in second if x0 + pad < x < x1 - pad]
                    if first_bar:
                        past = [(x, clear) for x, clear in inside if x > clef_end + pad]
                        sure = [x for x, clear in past if clear]
                        if sure:
                            return sure, [missed_note(sure)]
                        unsure = [x for x, clear in past if not clear]
                        ratio = (x1 - clef_end) / staff_typical
                        if ratio > WIDE and not past:
                            unsure = [x for x in loose if clef_end + pad < x < x1 - pad]
                        if ratio > WIDE and unsure:
                            return unsure, [f"bar is {ratio:.1f}x the usual width after the clef"
                                            " and has a possible barline inside; counted as"
                                            " split - please check"]
                        return [], []
                    sure = [x for x, clear in inside if clear]
                    unsure = [x for x, clear in inside if not clear]
                    # a key or clef change inside the bar makes it wide too
                    ratio = (x1 - preamble_end(sysm["preamble"], x0, x1, st["space"])) / staff_typical
                    if ratio > WIDE and not inside:
                        unsure = [x for x in loose if x0 + pad < x < x1 - pad]
                    if x1 - x0 < SUSPECT * typical:
                        # ordinary-width bar: act only on confident barlines
                        return (sure, [missed_note(sure)]) if sure else ([], [])
                    if sure:
                        return sure, [missed_note(sure)]
                    if ratio > WIDE and unsure:
                        # two weak signals agree: a very wide bar with a possible
                        # barline inside it
                        return unsure, [f"bar is {ratio:.1f}x the usual width and has a possible"
                                        " barline inside; counted as split - please check"]
                    if ratio > WIDE:
                        # width alone over-counted on an unseen part (Violin II), so
                        # it only raises a flag
                        return [], [f"bar is {ratio:.1f}x the usual width"
                                    " - not counted; please check for a missing barline"]
                    if unsure:
                        return [], ["possible barline beside a note (could be a stem)"
                                    " - not counted; please check"]
                    return [], []

                def piece_count(xa, xb):
                    """Measures in one piece of a split bar: 1 with notes, else its rests."""
                    if any(xa < hx + hw / 2 < xb for hx, hy, hw, hh in sysm["heads"]):
                        return 1, ["notes"]
                    kinds = [k for k, _, _ in
                             classify_rest(bare, st, xa, xb, sysm["masks"], sysm["preamble"], ink)]
                    number, score, readings = read_number(gray, st, xa, xb)
                    n, flags = decide(kinds, number, score, readings)
                    return n, flags + (["rest"] if kinds else [])

                def split_count(x0, x1, splits):
                    """Total measures, flags and pieces after splitting a bar at missed
                    barlines; each piece is a timeline entry's (x0, x1, count, plays, rest)."""
                    edges = [x0] + sorted(splits) + [x1]
                    total, flags, pieces = 0, [], []
                    for xa, xb in zip(edges, edges[1:]):
                        n, f = piece_count(xa, xb)
                        plays = "notes" in f
                        pieces.append(piece(xa, xb, n, plays, 0 if plays else n * ("rest" in f)))
                        total += n
                        flags += [x for x in f if x not in ("notes", "rest")]
                    return total, flags, pieces

                before = movements[-1]["measures"]  # last number used before this staff
                first = None  # number of the staff's first bar, after any ending reset
                # where bars begin: barlines, plus the end of the clef/key/time area
                starts = [b["x1"] for b in sysm["bars"]] + [
                    max([st["left"]] + [px + pw for px, py, pw, ph in sysm["preamble"]])]
                first_barline = sysm["bars"][0]["x0"] if sysm["bars"] else st["right"]
                endings = find_endings(ink, gray, st, starts, first_barline)
                for b, (x0, x1) in enumerate(segments, start=1):
                    # 1st/2nd endings: the 2nd ending reuses the 1st ending's numbers,
                    # so every part (with or without endings) numbers alike
                    for e in (e for e in endings if x0 - st["space"] <= e["x0"] < x1 - st["space"]):
                        mov = movements[-1]
                        where = {"page": page, "staff": i + 1, "bar": b, "movement": mov["title"]}
                        if e["number"] == 1:
                            mov["ending1"] = mov["measures"] + 1
                            notes.append({**where, "note": f"1st ending starts at m.{mov['ending1']}"})
                        elif e["number"] == 2 and "ending1" in mov:
                            mov["measures"] = mov.pop("ending1") - 1
                            notes.append({**where, "note": "2nd ending reuses the 1st ending's"
                                          f" numbers from m.{mov['measures'] + 1}"})
                        else:
                            report.append({**where, "symbols": ["ending"], "number": e["number"],
                                           "count": 0, "flags": [
                                               "ending bracket without a matching 1st ending"
                                               " or a readable number - please check numbering"]})
                    if first is None:
                        first = movements[-1]["measures"] + 1
                    # a rehearsal letter sits on the barline above the staff, and
                    # Audiveris can take it for a notehead
                    found = classify_rest(bare, st, x0, x1, sysm["masks"], sysm["preamble"], ink)
                    # a rehearsal letter above the staff can be taken for a notehead:
                    # one on the bar's opening barline, or one standing between two
                    # rests (on a double barline Audiveris missed), is not a note
                    heads_here = [
                        (hx, hy, hw, hh) for hx, hy, hw, hh in sysm["heads"]
                        if x0 < hx + hw / 2 < x1 and not (
                            hy + hh / 2 < st["top"] - 0.5 * st["space"]
                            and hx + hw / 2 < x0 + 1.5 * st["space"])
                        and not (st["top"] - 1.5 * st["space"] < hy + hh / 2 < st["top"]
                                 and any(r < hx for _, _, r in found)
                                 and any(l > hx + hw for _, l, _ in found))
                        # nor is a time signature read twice, as digits and as notes
                        and not any(px - 0.3 * st["space"] < hx + hw / 2 < px + pw + 0.3 * st["space"]
                                    for px, py, pw, ph in sysm["preamble"])]
                    # a letter between two rests (one multi-bar) marks a double
                    # barline Audiveris missed: the bar is two rests, each under its
                    # own number
                    sp = st["space"]
                    letter = None
                    if any(k == "multi" for k, _, _ in found):
                        for hx, hy, hw, hh in sysm["heads"]:
                            if not (x0 < hx < x1 and st["top"] - 1.5 * sp < hy + hh / 2 < st["top"]):
                                continue
                            # a rest on each side: recognised, or a rest's thick stroke
                            if not ((any(r < hx for _, _, r in found)
                                     and any(l > hx + hw for _, l, _ in found))
                                    or (rest_stroke(ink, st, x0, hx - 0.5 * sp) >= 3
                                        and rest_stroke(ink, st, hx + 1.5 * sp, x1) >= 3)):
                                continue
                            start = preamble_end(sysm["preamble"], x0, x1, sp)
                            left = read_number(gray, st, start, hx, multi=True)[0]
                            right = read_number(gray, st, hx, x1, multi=True)[0]
                            if left and right:
                                letter = hx
                                break
                    if letter is not None:
                        heads_here = [h for h in heads_here if abs(h[0] - letter) > sp]
                    # a rest's printed number can itself be read as a notehead high
                    # above the staff: with a rest in the bar and a number read
                    # there, such "heads" are the number
                    if (found and heads_here
                            and all(hy + hh / 2 < st["top"] - sp for hx, hy, hw, hh in heads_here)
                            and read_number(gray, st, x0, x1)[1] >= OCR_TRUST):
                        heads_here = []
                    has_notes = bool(heads_here)
                    kinds = [k for k, _, _ in found]
                    entry = {"page": page, "staff": i + 1, "bar": b,
                             "movement": movements[-1]["title"]}
                    place = {**entry, "x0": x0, "x1": x1,
                             "measure": movements[-1]["measures"] + 1,
                             "fermata": any(x0 < fx + fw / 2 < x1
                                            for fx, fy, fw, fh in sysm["fermatas"])}
                    if has_notes:
                        count, flags = 1, []
                        # a multi-bar rest has no notes under it (a beam always does)
                        # and a number above it
                        hidden = [
                            (k, l, r) for k, l, r in found if k != "whole" and not any(
                                l - st["space"] < hx + hw / 2 < r + st["space"]
                                for hx, hy, hw, hh in heads_here)
                        ]
                        multi = [(l, r) for k, l, r in hidden if k == "multi"]
                        number, score, readings = (
                            read_number(gray, st, *multi[0], multi=True) if multi
                            else (None, 0.0, []))
                        rest_count, pieces = 0, []
                        if multi and number is not None:
                            rest_count, flags = decide(["multi"], number, score, readings)
                        elif hidden and not multi:
                            # 2- and 4-bar rest shapes also turn up among notes (half
                            # notes, beams), so they need the printed number to agree
                            value = sum(VALUE[k] for k, _, _ in hidden)
                            seen, _, _ = read_number(
                                gray, st, hidden[0][1] - 1.5 * st["space"],
                                hidden[-1][2] + 1.5 * st["space"])
                            rest_count, flags = (value, []) if seen == value else (0, [])
                        # notes after the rest with no barline between are cues in its
                        # last bar; otherwise a barline was missed and they are a bar
                        # of their own
                        rest_first = bool(rest_count) and all(
                            hx + hw / 2 > hidden[-1][2] for hx, hy, hw, hh in heads_here)
                        first_head = min(hx + hw / 2 for hx, hy, hw, hh in heads_here)
                        cues = rest_first and not any(
                            hidden[-1][2] < x < first_head
                            for x in second_opinion(x0, x1)[0])
                        under = None
                        if cues:
                            # running count over or under the cue notes
                            under = read_number(gray, st, first_head - st["space"], x1,
                                                below=True)[0]
                            if under != rest_count + 1:
                                under = read_number(gray, st, first_head - st["space"], x1)[0]
                        rest_end = hidden[-1][2] if hidden else x0
                        if cues and under == rest_count + 1:
                            # the cue bar carries the rest's running count ("19" after
                            # an 18-bar rest): a bar of its own
                            count = rest_count + 1
                            cut = (rest_end + first_head - 0.6 * st["space"]) / 2
                            pieces = [piece(x0, cut, rest_count, False, rest_count),
                                      piece(cut, x1, 1, False, 1)]
                            flags.append(f"cue notes counted {under} after a {rest_count}-bar"
                                         " rest - a barline is probably missing; please check")
                        elif cues:
                            count = rest_count
                            flags.append(f"cue notes after a {rest_count}-bar rest in one bar"
                                         " - please check")
                        elif rest_count and key_change(
                                heads_here, preamble_end(sysm["preamble"], x0, x1, st["space"]),
                                st["space"]):
                            # accidentals of a key change, or the digits of a time
                            # signature, read as a stacked column of noteheads just
                            # after the barline or clef: the bar is only the rest
                            count = rest_count
                            cues = True
                            flags.append(f"key change before a {rest_count}-bar rest - please check")
                        elif rest_count:
                            count += rest_count
                            flags.append(f"notes and a {rest_count}-bar rest share one bar"
                                         " - a barline is probably missing")
                            if rest_first:
                                cut = (rest_end + first_head - 0.6 * st["space"]) / 2
                                pieces = [piece(x0, cut, rest_count, False, rest_count),
                                          piece(cut, x1, 1, True, 0)]
                            else:
                                rest_start = hidden[0][1]
                                last = max([hx + hw for hx, hy, hw, hh in heads_here
                                            if hx + hw / 2 < rest_start] or [x0])
                                cut = (last + rest_start) / 2
                                pieces = [piece(x0, cut, 1, True, 0),
                                          piece(cut, x1, rest_count, False, rest_count)]
                        else:
                            splits, flags = second_opinion(x0, x1, first_bar=b == 1)
                            if splits:
                                count, extra, pieces = split_count(x0, x1, splits)
                                flags = [f + f" - now counted as {count}" if k == 0 else f
                                         for k, f in enumerate(flags + extra)]
                        movements[-1]["measures"] += count
                        # cue notes count as resting; with a missed barline each piece
                        # is a bar (or rest) of its own
                        add_pieces(timeline, place, pieces or [
                            piece(x0, x1, count, not cues, count if cues else rest_count)])
                        if flags:
                            shown = [hidden[0][0]] if rest_count else ["notes"]
                            report.append({**entry, "symbols": shown, "number": None,
                                           "count": count, "flags": flags})
                        continue
                    after_preamble = preamble_end(sysm["preamble"], x0, x1, st["space"])
                    # clef/key before a start-repeat or new section, not a measure; a
                    # wide key signature can stretch it past 8 spaces, but then a
                    # double barline closes it with little room after the key
                    closing = sysm["bars"][0] if sysm["bars"] else None
                    double = closing and closing["x1"] - closing["x0"] > 0.8 * st["space"]
                    if b == 1 and not kinds and (x1 - x0 < 8 * st["space"] or (
                            double and x1 - after_preamble < 3 * st["space"])):
                        continue
                    # the number sits over the rest symbol; reading only there keeps
                    # rehearsal letters, tempo marks and the time signature out
                    span = (min(l for _, l, _ in found) - 1.5 * st["space"],
                            max(r for _, _, r in found) + 1.5 * st["space"]) if found else None
                    spans = [(max(x0, span[0]), min(x1, span[1]))] if found else []
                    number, score, readings = read_number(
                        gray, st, after_preamble, x1, multi="multi" in kinds, spans=spans)
                    if number is None and after_preamble > x0:
                        # the number may sit over the time signature's end
                        number, score, readings = read_number(
                            gray, st, x0, x1, multi="multi" in kinds, spans=spans)
                    count, flags = decide(kinds, number, score, readings)
                    pieces = []
                    multis = sorted((l, r) for k, l, r in found if k == "multi")
                    gaps = [b_l - a_r for (_, a_r), (b_l, _) in zip(multis, multis[1:])]
                    sides = []  # (x0, x1, reading) of each rest when a bar holds two
                    if letter is not None:
                        sides = [(a, b, read_number(gray, st, a, b, multi=True))
                                 for a, b in ((after_preamble, letter), (letter, x1))]
                        each = [r[0] for _, _, r in sides]
                        if all(each):
                            count = sum(each)
                            pieces = [piece(x0 if k == 0 else a, b, n, False, n)
                                      for k, (a, b, n) in enumerate(
                                          (a, b, r[0]) for a, b, r in sides)]
                            flags = [f"{each[0]} and {each[1]}-bar rests share one bar at a"
                                     " rehearsal letter - a barline is probably missing; please check"]
                    elif len(multis) > 1 and min(gaps) > 2 * st["space"]:
                        # a barline between two multi-bar rests was missed; read each
                        # number only on its own side of the gap
                        edges = [after_preamble] + [(a_r + b_l) / 2 for (_, a_r), (b_l, _)
                                                    in zip(multis, multis[1:])] + [x1]
                        sides = [(a, b, read_number(gray, st, a, b, multi=True))
                                 for a, b in zip(edges, edges[1:])]
                        each = [r[0] for _, _, r in sides]
                        if all(each):
                            count = sum(each) + sum(VALUE[k] for k in kinds if k != "multi")
                            pieces = [piece(x0 if k == 0 else a, b, n, False, n)
                                      for k, (a, b, n) in enumerate(
                                          (a, b, r[0]) for a, b, r in sides)]
                            pieces[-1]["count"] += count - sum(each)  # whole rests beside them
                            pieces[-1]["rest"] = pieces[-1]["count"]
                            flags = [f"{' and '.join(map(str, each))}-bar rests share one bar"
                                     " - a barline is probably missing; please check"]
                    under = read_number(gray, st, x0, x1, below=True)[0] if (
                        kinds and "multi" not in kinds and count > 1) else None
                    if under is not None and under >= 5:
                        # a running count printed under a cue bar: one bar
                        count = 1
                        flags = [f"cue bar counted {under} below the staff - counted as 1;"
                                 " please check"]
                    elif (kinds and "multi" not in kinds and number and number > count
                            and len(readings) == 1):
                        # "5" over a cue bar read as a rest, between bars numbered
                        # "4" and "6": a running count of rest bars, so one bar
                        near = [read_number(gray, st, *segments[k])[0]
                                for k in (b - 2, b) if 0 <= k < len(segments)]
                        if number - 1 in near or number + 1 in near:
                            count = 1
                            flags = [f"the number above ({number}) continues a running count"
                                     " - counted as 1; please check"]
                    if not kinds and count > 1 and rest_stroke(ink, st, x0, x1) < 3:
                        # without a rest's thick stroke the number is a running count
                        # over cue bars, or a bar number some editions print
                        count = 1
                        flags = ["bar with no notes and no recognised rest; the number"
                                 f" above ({number}) looks like a count, not a rest - please check"]
                    if "multi" not in kinds:
                        splits, extra = second_opinion(x0, x1)
                        if splits:
                            count, more, pieces = split_count(x0, x1, splits)
                            extra = [extra[0] + f" - now counted as {count}"] + extra[1:] + more
                        flags += extra
                    if pieces and sum(p["count"] for p in pieces) != count:
                        pieces = []  # a later rule changed the count; keep the bar whole
                    movements[-1]["measures"] += count
                    # a bar with nothing recognised under a running count ("21" over
                    # m.149) holds another part's cue notes: the part rests there.
                    # A number equal to the bar's own is a printed bar number.
                    running = (not kinds and count == 1 and number is not None
                               and number != place["measure"])
                    added = add_pieces(timeline, place, pieces or [
                        piece(x0, x1, count, False, count if kinds or running else 0)])
                    report.append({**entry, "symbols": kinds, "number": number,
                                   "ocr": round(score, 2), "readings": readings,
                                   "count": count, "flags": flags})
                    if sides and all(r[0] for _, _, r in sides):
                        for k, (a, b, (n, sc, rd)) in enumerate(sides):
                            side = {"symbols": ["multi"], "number": n, "ocr": sc, "readings": rd}
                            shapes = digits.digit_shapes(gray, st, a + 1.5 * st["space"],
                                                         b - st["space"])
                            # each rest's own piece of the bar takes any correction
                            target = added[k] if len(added) == len(sides) else added[-1]
                            rest_numbers.append((side, report[-1], target, shapes))
                    elif kinds and (number is not None or "multi" in kinds):
                        shapes = digits.digit_shapes(gray, st, x0 + 1.5 * st["space"],
                                                     x1 - st["space"])
                        rest_numbers.append((report[-1], report[-1],
                                             max(added, key=lambda e: e["rest"]), shapes))
                    elif number and number > 1 and count == 1 and len(readings) > 1 and (
                            rest_stroke(ink, st, x0, x1) >= 0.6):
                        # an old-style rest whose blocks weren't recognised, under a
                        # number read two ways ("3 or 8"): the digit shapes decide
                        shapes = digits.digit_shapes(gray, st, x0 + 1.5 * st["space"],
                                                     x1 - st["space"])
                        unread = {"symbols": [], "number": number, "ocr": score,
                                  "readings": readings, "confirm": True}
                        rest_numbers.append((unread, report[-1], added[0], shapes))
                staves.append({
                    "page": page, "staff": i + 1, "movement": movements[-1]["title"],
                    # printed at the start of the staff; a 2nd ending opening the staff
                    # reuses an earlier number, so this can be below before + 1
                    "first_measure": first or before + 1,
                    "last_measure": movements[-1]["measures"],
                    "measures": movements[-1]["measures"] - before,  # new numbers added
                    "reset": before + 1 - (first or before + 1),
                    "left": st["left"], "top": st["top"], "space": st["space"],
                })
            if progress:
                progress(page, doc.page_count)

    changed = rehearsal_over_rests(rest_numbers, marks, movements, staves, timeline, doc)
    reread_by_shape([r for r in rest_numbers if not any(r[0] is c for c in changed)],
                    movements, staves, timeline)
    cache_path.write_text(json.dumps(_ocr_cache))
    data = {"movements": movements, "staves": staves, "bars": report, "notes": notes,
            "timeline": timeline, "unread_pages": unread_pages}
    (out_dir / "measure_count.json").write_text(json.dumps(data, indent=1))
    return data
