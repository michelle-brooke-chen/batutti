"""Re-read multi-bar rest numbers by the shapes of this part's own digits.

The general-purpose OCR confuses this engraving's 2 and 7, 3 and 8, and 13 and
18, sometimes with high confidence. But a part prints the same digit shapes
throughout, and most of its numbers are read with certainty. Those numbers
teach us what each digit looks like; a doubtful number is then read digit by
digit by matching shapes. The shape reading replaces the OCR only when every
digit is a close, clear match, and it either has as many digits as the OCR's
reading, is one of the OCR's other readings, or fills in a number it missed.
"""

import cv2
import numpy as np

SIZE = (20, 28)  # digits are compared at this size
SURE_OCR = 0.99  # OCR readings this confident (and unambiguous) teach the shapes
MARGIN = 0.35  # how much better the best digit must match than the next
WEAK_MARGIN = 0.25  # the same, when the OCR itself was unsure of its reading
WEAK_OCR = 0.9
MATCH = 0.8  # how well each digit must match its example (a digit the part
             # never printed confidently has no example, and matches poorly)


def digit_shapes(gray, staff, x0, x1):
    """Digit-sized blobs above the staff between x0 and x1, left to right."""
    top, s = staff["top"], staff["space"]
    crop = gray[int(top - 3.5 * s):int(top - 0.2 * s), int(x0):int(x1)]
    if crop.size == 0:
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats((crop < 128).astype(np.uint8), 8)
    blobs = []
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        if 1.0 * s <= h <= 2.2 * s and 0.12 * s <= w <= 1.6 * s:
            blobs.append((x, y + h, (labels[y:y + h, x:x + w] == i).astype(np.float32)))
    if not blobs:
        return []
    # a number's digits share a baseline
    base = max(b for _, b, _ in blobs)
    blobs = sorted(blob for blob in blobs if abs(blob[1] - base) < 0.4 * s)
    return [cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA) for _, _, img in blobs]


def similarity(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def read_shapes(shapes, library):
    """The number the shapes spell, the weakest digit's match, and its margin."""
    text, weakest, closest = "", 1.0, 1.0
    for shape in shapes:
        best = {}
        for digit, example in library:
            best[digit] = max(best.get(digit, -1.0), similarity(shape, example))
        ranked = sorted(best.values(), reverse=True)
        if not ranked:
            return None, 0.0, 0.0
        text += max(best, key=best.get)
        closest = min(closest, ranked[0])
        weakest = min(weakest, ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0))
    return (int(text) if text else None), closest, weakest


def library_from(rests):
    """Digit examples from the numbers the OCR read with certainty."""
    library = []
    for bar, shapes in rests:
        readings = bar.get("readings") or []
        if (len(readings) == 1 and (bar.get("ocr") or 0) >= SURE_OCR
                and len(str(readings[0])) == len(shapes)):
            library += zip(str(readings[0]), shapes)
    return library


def confident(rests):
    """{index: number} for every rest whose digit shapes clearly spell a number;
    a second, independent reading to check the OCR against."""
    library = library_from(rests)
    found = {}
    for i, (bar, shapes) in enumerate(rests):
        if shapes:
            seen, match, margin = read_shapes(shapes, library)
            if seen is not None and seen > 1 and match >= MATCH and margin >= MARGIN:
                found[i] = seen
    return found


def recheck(rests):
    """Shape readings for doubtful rest numbers.

    `rests` is a list of (bar, shapes) for every rest bar with a number read
    above it, where bar has "number", "ocr" and "readings". Returns
    {index: (new number, margin)} for bars whose shapes clearly spell a
    different number with as many digits.
    """
    library = library_from(rests)
    changes = {}
    for i, (bar, shapes) in enumerate(rests):
        number, readings = bar.get("number"), bar.get("readings") or []
        if not shapes:
            continue
        seen, match, margin = read_shapes(shapes, library)
        if bar.get("confirm"):
            # asked only to settle an ambiguous number: agree with one reading
            if seen in readings and seen > 1 and match >= MATCH and margin >= MARGIN:
                changes[i] = (seen, margin)
            continue
        if seen is None or seen <= 1 or seen == number or match < MATCH:
            continue
        if number is not None and len(shapes) == len(str(number)):
            needed = MARGIN if (bar.get("ocr") or 0) >= WEAK_OCR else WEAK_MARGIN
        elif number is None or (seen in readings and len(shapes) > len(str(number))):
            # a number the OCR missed, or one of its own other readings that keeps
            # a digit its choice dropped ("4 or 22")
            needed = MARGIN
        else:
            continue
        if margin >= needed:
            changes[i] = (seen, margin)
    return changes
