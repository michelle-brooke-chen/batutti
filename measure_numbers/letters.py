"""Rehearsal letters: read them in each part and compare the parts.

Every part of a work prints letter A, B, C... at the same measure. So once
parts are numbered, the letters cross-check them: a part whose letter falls
somewhere else is numbered wrong before that point, or misprinted.
"""

import re
from collections import Counter

import cv2
import numpy as np
import pymupdf

from . import counting

LETTERS = "ABCDEFGHIK"  # rehearsal letters (J is skipped by convention; L is rare)
MIN_SCORE = 0.8  # OCR confidence for a single capital letter
MIN_AGREE = 3  # parts that must agree before a letter's measure is trusted


def read_letters(data, pdf_path):
    """Every letter-like mark over the start of a bar: [(movement number, letter,
    measure, page, staff)]. Uses (and fills) the counter's OCR cache."""
    doc = pymupdf.open(pdf_path)
    numbers = {m["title"]: m.get("number", i + 1) for i, m in enumerate(data["movements"])}
    staves = {(s["page"], s["staff"]): s for s in data["staves"]}
    pages, found = {}, []
    for bar in data["timeline"]:
        if bar.get("piece", 1) > 1:
            continue  # a letter sits on a printed barline, at a bar's first piece
        st = staves[(bar["page"], bar["staff"])]
        if bar["page"] not in pages:
            pix = doc[bar["page"] - 1].get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
            pages[bar["page"]] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        gray, s = pages[bar["page"]], st["space"]
        crop = gray[max(0, int(st["top"] - 5 * s)):int(st["top"] - 0.5 * s),
                    max(0, int(bar["x0"] - 2 * s)):int(bar["x0"] + 2.5 * s)]
        if crop.size == 0:
            continue
        crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        result = counting.ocr(crop)
        for text, score in zip(result.txts or (), result.scores or ()):
            text = re.sub(r"[^A-Za-z0-9]", "", text)
            if len(text) == 1 and text in LETTERS and score >= MIN_SCORE:
                found.append((numbers[bar["movement"]], text, bar["measure"],
                              bar["page"], bar["staff"]))
    return sorted(set(found))


def consensus(found_by_part):
    """{(movement number, letter): measure} that at least MIN_AGREE parts agree on,
    keeping only letters that rise with the measure numbers."""
    votes = Counter((mv, letter, measure) for found in found_by_part.values()
                    for mv, letter, measure in {(f[0], f[1], f[2]) for f in found})
    best = {}
    for (mv, letter, measure), n in votes.items():
        if n >= MIN_AGREE and n > best.get((mv, letter), (0, 0))[1]:
            best[(mv, letter)] = (measure, n)
    agreed = {k: v[0] for k, v in best.items()}
    # drop any letter out of order with its neighbours (a cue number read as a letter)
    keep = {}
    for mv in {k[0] for k in agreed}:
        row = sorted((LETTERS.index(l), m, l) for (k, l), m in agreed.items() if k == mv)
        for i, (_, m, l) in enumerate(row):
            if all(m2 < m for _, m2, _ in row[:i]) and all(m2 > m for _, m2, _ in row[i + 1:]):
                keep[(mv, l)] = m
    return keep


def disagreements(found, agreed):
    """Letters this part prints somewhere other than the agreed measure (and not
    at the agreed measure too): [{movement, letter, here, expected, page, staff}]."""
    at = {(mv, l, m) for mv, l, m, _, _ in found}
    out = []
    for (mv, letter), measure in sorted(agreed.items()):
        if (mv, letter, measure) in at:
            continue
        elsewhere = [f for f in found if f[0] == mv and f[1] == letter]
        if elsewhere:
            mv_, l_, here, page, staff = min(elsewhere, key=lambda f: abs(f[2] - measure))
            out.append({"movement": mv, "letter": letter, "here": here, "expected": measure,
                        "page": page, "staff": staff})
    return out
