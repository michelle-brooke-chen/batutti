"""Play a part back from its scan: notes from Audiveris, bars from batutti.

Audiveris reads pitches fairly well but rhythm poorly, and it misreads
clefs and key signatures now and then. batutti already knows every bar
exactly, so each bar is rebuilt on its piece of the timeline: long rests
last their real length, printed cue notes stay silent, clefs and keys are
checked against Audiveris's confidence in them, and marks read as notes are
dropped. A bar's rhythm is then taken, in order of trust, from:

  1. adds up      the bar fills its time signature with a normally written
                  rhythm, and Audiveris played every notehead it saw
  2. same as      a clean bar elsewhere in the movement with these noteheads
  3. same rhythm  a clean bar whose noteheads are spaced the same way
  4. even         running notes, one per notehead (filling one it missed)
  5. fixed        the smallest change that gives a normally written rhythm
                  that fills the bar: a note or beamed group doubled,
                  halved or made triplets, a dot added or dropped, two such
                  changes, or a rest Audiveris missed
  6. spaced       notes laid out by their spacing on the page (unsure)

A bar with no notes read and no rest symbol (tremolos, hollow notes)
repeats the bar before. Timpani follow the tuning printed above the staff.

Checked against a reference score by evaluation/playback_check.py.

    python -m measure_numbers.playback RESULT_DIR OUT.wav [MOVEMENT FIRST LAST]
"""

import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .omr import audiveris_path

STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACCIDENTALS = {"sharp": 1, "flat": -1, "natural": 0, "double-sharp": 2, "flat-flat": -2}


def export_scores(omr):
    """MusicXML for each movement Audiveris found, saved next to the .omr
    (it reuses the saved recognition, so this takes seconds). Returns the
    .mxl paths in order."""
    omr = Path(omr)

    def found():
        return (sorted(omr.parent.glob(f"{omr.stem}.mvt*.mxl"), key=_mvt)
                or sorted(omr.parent.glob(f"{omr.stem}.mxl")))

    if found():
        return found()
    program = audiveris_path()
    if program is None:
        raise RuntimeError("Audiveris is not installed (set AUDIVERIS to its path)")
    with open(omr.parent / "export.log", "w") as log:
        subprocess.run([program, "-batch", "-export", "--", str(omr)], stdout=log, stderr=log)
    return found()


def _mvt(path):
    m = re.search(r"\.mvt(\d+)\.mxl$", str(path))
    return int(m.group(1)) if m else 0


def _score_xml(mxl):
    with zipfile.ZipFile(mxl) as z:
        name = next(n for n in z.namelist() if n.endswith(".xml") and not n.startswith("META"))
        return ET.fromstring(z.read(name))


def read_systems(mxls):
    """Every measure in reading order, as one list: [{"measures": [{"width",
    "time", "clef", "fifths", "events": [event]}]}]. An event is one note,
    chord or rest of one voice: {"voice", "x" (tenths from the measure
    start), "dur" (quarters), "midi" ([] for a rest), "type", "dots",
    "tuplet", "grace", "beam"}.

    Where the cellos or basses divide onto two staves, Audiveris may split a
    movement into two parts, each holding the notes of some lines and empty
    measures for the rest; each measure is taken from the part that has it."""
    measures = []
    for mxl in mxls:
        parts = [read_part(part) for part in _score_xml(mxl).findall("part")]
        for row in zip(*parts):
            measures.append(max(row, key=lambda m: (m["width"] > 0,
                                                    sum(1 for e in m["events"] if e["midi"]))))
    return [{"measures": measures}]


def read_part(part):
    """The measures of one MusicXML part."""
    out = []
    divisions, time, clef, fifths = 1, None, ("G", 2, 0), 0
    for measure in part.iter("measure"):
        attrs = measure.find("attributes")
        if attrs is not None:
            if attrs.findtext("divisions"):
                divisions = int(attrs.findtext("divisions")) or divisions
            c = attrs.find("clef")
            if c is not None and c.findtext("sign"):
                clef = (c.findtext("sign"), int(c.findtext("line") or 0),
                        int(c.findtext("clef-octave-change") or 0))
            if attrs.findtext("key/fifths"):
                fifths = int(attrs.findtext("key/fifths"))
            t = attrs.find("time")
            if t is not None and t.findtext("beats"):
                time = Fraction(4 * int(t.findtext("beats")), int(t.findtext("beat-type")))
        events = []
        for note in measure.iter("note"):
            pitch = note.find("pitch")
            midi, spelled = [], []
            if pitch is not None:
                midi = [12 * (int(pitch.findtext("octave")) + 1) + STEPS[pitch.findtext("step")]
                        + int(float(pitch.findtext("alter") or 0))]
                spelled = [(pitch.findtext("step"), int(pitch.findtext("octave")),
                            ACCIDENTALS.get(note.findtext("accidental")))]
            tied = [any(t.get("type") == "stop" for t in note.findall("tie"))] if spelled else []
            if note.find("chord") is not None and events:
                events[-1]["midi"] += midi
                events[-1]["spelled"] += spelled
                events[-1]["tied"] += tied
                continue
            x = note.get("default-x")
            events.append({
                "voice": note.findtext("voice") or "1",
                "x": float(x) if x is not None else None,
                "dur": Fraction(int(note.findtext("duration") or 0), divisions),
                "midi": midi,
                "spelled": spelled,
                "type": note.findtext("type"),
                "dots": len(note.findall("dot")),
                "tuplet": note.find("time-modification") is not None,
                "grace": note.find("grace") is not None,
                "beam": note.findtext("beam"),
                "tied": tied,  # per pitch: held over from the note before
            })
        key_here = None
        if attrs is not None and attrs.findtext("key/fifths"):
            key_here = int(attrs.findtext("key/fifths"))
        out.append({"width": float(measure.get("width") or 0),
                    "key": key_here, "clef_here": attrs is not None and attrs.find("clef") is not None,
                    "time": time, "clef": clef, "fifths": fifths, "events": events})
    return out


SMALL_HEAD = 0.78  # cue notes print at about 70% size; real notes rarely read below 0.9


def ink_width(ink, x, y):
    """Width of the run of ink through (x, y), allowing a pixel or two of slack."""
    row = ink[y]
    for dx in (0, -1, 1, -2, 2):
        if 0 <= x + dx < len(row) and row[x + dx]:
            x += dx
            break
    else:
        return 0
    a = b = x
    while a > 0 and row[a - 1]:
        a -= 1
    while b < len(row) - 1 and row[b + 1]:
        b += 1
    return b - a + 1


def head_widths(omr, pdf, staves):
    """{(page, staff): [(x, width in staff spaces)]} of every filled notehead,
    measured on the page just above and below its centre (so a staff line
    through it doesn't count). Audiveris's own bounds are template-sized."""
    import numpy as np
    import pymupdf
    from .cues import page_image

    space = {(s["page"], s["staff"]): s["space"] for s in staves}
    doc, cache, out = pymupdf.open(pdf), {}, {}
    with zipfile.ZipFile(omr) as z:
        for page in sorted({s["page"] for s in staves}):
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            ink = None
            for k, system in enumerate(xml.split("<system ")[1:], start=1):
                # two staves braced (a duet or divisi) print smaller and
                # don't carry cues: leave them out
                if (page, k) not in space or len(re.findall(r"<staff [^>]*id=", system)) > 1:
                    continue
                s = space[(page, k)]
                for attrs, x, y, w, h in re.findall(
                        r'<head ([^>]*)>\s*<bounds x="([\d.]+)" y="([\d.]+)" w="([\d.]+)" h="([\d.]+)"', system):
                    if "NOTEHEAD_BLACK" not in attrs:
                        continue
                    if ink is None:
                        ink = (page_image(doc, page, cache) < 140).astype(np.uint8)
                    cx, cy = int(float(x) + float(w) / 2), float(y) + float(h) / 2
                    width = min(ink_width(ink, cx, int(cy + d * s)) for d in (-0.22, 0.22)) / s
                    if width > 0:
                        out.setdefault((page, k), []).append((cx, width))
    return out


def omr_rests(omr, pages):
    """{(page, staff): [x]} of every rest symbol Audiveris found."""
    out = {}
    with zipfile.ZipFile(omr) as z:
        for page in pages:
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            for k, system in enumerate(xml.split("<system ")[1:], start=1):
                for x in re.findall(r'<rest [^>]*>\s*<bounds x="([\d.]+)"', system):
                    out.setdefault((page, k), []).append(float(x))
    return out


def omr_heads(omr, pages):
    """{(page, staff): [(x, pitch position, weak)]} for every notehead
    Audiveris saw. Weak heads are poor matches for their shape: printed
    digits and marks read as notes are nearly always weak, real heads rarely."""
    heads = {}
    with zipfile.ZipFile(omr) as z:
        for page in pages:
            root = ET.fromstring(z.read(f"sheet#{page}/sheet#{page}.xml"))
            for k, system in enumerate(root.iter("system"), start=1):
                for h in system.iter("head"):
                    b = h.find("bounds")
                    if b is not None and h.get("pitch") is not None:
                        x = float(b.get("x")) + float(b.get("w")) / 2
                        weak = float(h.get("grade")) < (0.6 if h.get("shape") == "NOTEHEAD_BLACK" else 0.4)
                        heads.setdefault((page, k), []).append((x, int(float(h.get("pitch"))), weak))
    return heads


# the note each clef names (G4, F3, C4), counted in letter steps from C0
CLEF_NOTE = {"G": 4 * 7 + 4, "F": 3 * 7 + 3, "C": 4 * 7}
SHARPS = "FCGDAEB"


def position_midi(position, clef, fifths):
    """MIDI pitch of a notehead at a staff position (0 = middle line, counting
    down), from the clef and key signature; accidentals in the bar are not seen."""
    step = middle_line(clef) - position
    letter, octave = "CDEFGAB"[step % 7], step // 7
    return 12 * (octave + 1) + STEPS[letter] + position_alter(letter, fifths)


def onsets(xs, space):
    """Group x positions closer than half a staff space into one onset."""
    groups = []
    for x in sorted(xs):
        if groups and x - groups[-1][-1] < space / 2:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def omr_keys(omr, pages):
    """(fifths, grade) of the key signature at the start of each system, or
    None where Audiveris found none, in reading order."""
    out = []
    with zipfile.ZipFile(omr) as z:
        for page in pages:
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            for system in xml.split("<system ")[1:]:
                m = re.search(r'<key fifths="(-?\d+)" grade="([\d.]+)"', system)
                out.append((int(m.group(1)), float(m.group(2))) if m else None)
    return out


def key_room(omr, staves):
    """Per system, in reading order: the space (in staff spaces) between the
    clef and the first note or barline, where a key signature goes; None if
    no clef was read with confidence. About 1.5 with no key signature, 3 or
    more with one."""
    out = []
    with zipfile.ZipFile(omr) as z:
        sheets = {}
        for s in staves:
            if s["page"] not in sheets:
                sheets[s["page"]] = z.read(f"sheet#{s['page']}/sheet#{s['page']}.xml").decode().split("<system ")
            system = sheets[s["page"]][s["staff"]]
            clef = re.search(r'<clef ([^>]*)>\s*<bounds x="([\d.]+)" y="[\d.]+" w="([\d.]+)"', system)
            grade = re.search(r'ctx-grade="([\d.]+)"', clef.group(1)) if clef else None
            if not grade or float(grade.group(1)) < 0.5:
                out.append(None)
                continue
            start = float(clef.group(2)) + float(clef.group(3))
            after = [float(x) for x in re.findall(r'<(?:head|barline) [^>]*>\s*<bounds x="([\d.]+)"', system)
                     if float(x) > start + 0.2 * s["space"]]
            out.append((min(after) - start) / s["space"] if after else None)
    return out


def system_keys(found, movements, room=None):
    """Each system's key signature, correcting Audiveris's misreadings: a
    system with no key read keeps the one before in the same movement (none
    at the start of a movement), unless there is no room for one between the
    clef and the first note (`room`, from key_room): then it has none. A key
    differing from the
    systems on both sides, which agree, is theirs; and a weakly read key one
    accidental short of or over the one before keeps the one before."""
    keys, grades = [], []
    for n, k in enumerate(found):
        if k is None:
            same = n and movements[n] == movements[n - 1]
            bare = room is not None and room[n] is not None and room[n] < 2.5
            keys.append(keys[-1] if same and not bare else 0)
            grades.append(0.0)
        else:
            keys.append(k[0])
            grades.append(k[1])
    for i in range(len(keys)):
        before = keys[i - 1] if i and movements[i - 1] == movements[i] else None
        after = keys[i + 1] if i + 1 < len(keys) and movements[i + 1] == movements[i] else None
        if before is not None and keys[i] != before and after == before:
            keys[i] = before
            continue
        # a weak key one accidental off its neighbour (the one before, or
        # after at the start of a movement) is that neighbour's key
        near = before if before is not None else after
        if (near is not None and grades[i] < 0.6 and abs(keys[i] - near) == 1
                and keys[i] * near > 0 and after != keys[i]):
            keys[i] = near
    return keys


CLEF_KINDS = {"TREBLE": ("G", 2), "BASS": ("F", 4), "ALTO": ("C", 3), "TENOR": ("C", 4),
              "SOPRANO": ("C", 1), "MEZZO_SOPRANO": ("C", 2), "BARITONE": ("F", 3)}


def omr_clefs(omr, pages):
    """[(kind, grade, octave)] of each clef Audiveris found, per system, left
    to right (octave -1 for a clef with an 8 below, +1 with an 8 above)."""
    out = []
    with zipfile.ZipFile(omr) as z:
        for page in pages:
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            for system in xml.split("<system ")[1:]:
                found = []
                for attrs in re.findall(r"<clef ([^>]*)>", system):
                    a = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
                    grade = a.get("ctx-grade", a.get("grade"))
                    if grade is not None:
                        shape = a.get("shape", "")
                        octave = -1 if "8VB" in shape else 1 if "8VA" in shape else 0
                        found.append((a.get("kind"), float(grade), octave))
                out.append(found)
    return out


def correct_clefs(measures, found):
    """Each measure's clef as Audiveris read its notes ("clef") and as
    printed ("true_clef").

    Audiveris spells a line's notes with the clef it recognised there (treble
    where it found none), though its MusicXML sometimes carries on the clef
    of the line before. A weakly read clef keeps the clef
    before (or, at the start, the part's usual clef); a clef change inside a
    line counts only if read strongly, to a clef that opens a line somewhere
    in the part; octave clefs are read as plain."""
    strong = [CLEF_KINDS[k] for row in found for k, g, _ in row if g >= 0.5 and k in CLEF_KINDS]
    usual = Counter(strong).most_common(1)[0][0] if strong else ("G", 2)
    # clefs that open a line somewhere: a part changes only among these
    opening = {CLEF_KINDS[row[0][0]] for row in found if row and row[0][1] >= 0.5 and row[0][0] in CLEF_KINDS}
    current, line_of, read = None, None, None
    for measure in measures:
        row = found[measure["system"]]
        if measure["system"] != line_of:
            line_of = measure["system"]
            kind, grade, octave = row[0] if row else (None, 0.0, 0)
            read = (*CLEF_KINDS[kind], octave) if kind in CLEF_KINDS else ("G", 2, 0)
            if grade >= 0.5 and kind in CLEF_KINDS:
                current = CLEF_KINDS[kind]
            elif current is None:
                current = usual
        elif measure["clef_here"]:
            read = None
            later = [CLEF_KINDS[k] for k, g, _ in row[1:]
                     if g >= 0.6 and k in CLEF_KINDS and CLEF_KINDS[k] in opening]
            if later:
                current = later[0]
        if read is not None:
            measure["clef"] = read
        measure["true_clef"] = current


def spelled_with(measures, heads, staves):
    """Find the clef Audiveris actually spelled each line's notes with: the
    note on the middle line, from notes matched to their noteheads (whose
    place on the staff it records). Its MusicXML clef is sometimes not the
    one it used. Sets measure["read_middle"] where the notes agree."""
    by_line = {}
    for measure in measures:
        staff = staves[measure["system"]]
        here = heads.get((staff["page"], staff["staff"]), ())
        space = staff["space"]
        for e in measure["events"]:
            if not e["spelled"] or "px" not in e:
                continue
            if len(e["spelled"]) != 1:
                continue
            # every notehead nearby votes; only the right clef agrees across notes
            step, octave, _ = e["spelled"][0]
            votes = by_line.setdefault(measure["system"], Counter())
            for pos in {pos for x, pos, weak in here if not weak and abs(x - e["px"]) < 1.6 * space}:
                votes[7 * octave + "CDEFGAB".index(step) + pos] += 1
    for measure in measures:
        votes = by_line.get(measure["system"])
        if votes and not measure["clef_here"]:
            (middle, n), *rest = votes.most_common(2) + [(None, 0)]
            if n >= 3 and n >= 2 * rest[0][1]:
                measure["read_middle"] = middle


def middle_line(clef):
    """The note on the staff's middle line, in letter steps from C0: the
    clef's note sits on its line, two steps per line from the middle."""
    sign, line = clef[0], clef[1]
    shift = clef[2] if len(clef) > 2 else 0
    if sign not in CLEF_NOTE or not 1 <= line <= 5:
        sign, line = "G", 2
    return CLEF_NOTE[sign] + 2 * (3 - line) + 7 * shift


def respell(measures, keys, marks=None, staves=None, heads=None):
    """Recompute every pitch from its place on the staff, the clef and key
    signature in force and the accidentals earlier in the bar. A note tied
    over the barline keeps the pitch it was tied from. `marks` (from
    accidentals.find) adds sharps and naturals Audiveris missed, on the
    notehead nearest each note at its place on the staff."""
    line_of, before = None, {}
    for measure in measures:
        if measure["system"] != line_of:
            key, line_of = keys[measure["system"]], measure["system"]
        elif measure["key"] is not None:
            key = measure["key"]  # a key change inside the line
        measure["fifths"] = key
        read = measure.get("read_middle") or middle_line(measure["clef"])
        true = middle_line(measure["true_clef"])
        measure["clef"] = measure["true_clef"]
        held, sounded = {}, {}
        for e in sorted(measure["events"], key=lambda e: e["x"] or 0):
            midi, spelled = [], []
            ties = e.get("tied") or [False] * len(e["spelled"])
            for (step, octave, acc), tied in zip(e["spelled"], ties):
                position = read - (7 * octave + "CDEFGAB".index(step))
                if acc is None and marks and "px" in e:
                    acc = mark_at(e, position, staves[measure["system"]], marks, heads)
                d = true - position
                step, octave = "CDEFGAB"[d % 7], d // 7
                spelled.append((step, octave, acc))
                if acc is not None:
                    held[(step, octave)] = acc
                alter = held.get((step, octave))
                if alter is None and tied and (step, octave) in before:
                    alter = before[(step, octave)]  # only this note: the bar's others follow the key
                if alter is None:
                    alter = position_alter(step, key)
                sounded[(step, octave)] = alter
                midi.append(12 * (octave + 1) + STEPS[step] + alter)
            e["midi"], e["spelled"] = midi, spelled
        before = sounded


def mark_at(e, position, staff, marks, heads):
    """The detected accidental on the notehead that note e stands on, if any."""
    key, space = (staff["page"], staff["staff"]), staff["space"]
    here = [x for x, pos, weak in heads.get(key, ()) if pos == position and not weak
            and abs(x - e["px"] - 0.6 * space) < 2 * space]
    if not here:
        return None
    x = min(here, key=lambda x: abs(x - e["px"] - 0.6 * space))
    return next((alter for mx, pos, alter, _ in marks.get(key, ()) if pos == position and abs(mx - x) < 1), None)


def position_alter(letter, fifths):
    if fifths > 0:
        return 1 if letter in SHARPS[:fifths] else 0
    return -1 if letter in SHARPS[::-1][:-fifths] else 0


# semitones from written to sounding pitch, by instrument and the key it is
# "in" (German B = B flat, Es = E flat, H = B natural): an instrument in F
# sounds a written C as F
TRANSPOSE = {
    "clarinet": {"C": 0, "Bb": -2, "A": -3, "Eb": 3, "D": 2},
    "horn": {"F": -7, "E": -8, "Eb": -9, "D": -10, "C": -12, "Bb": -14, "B": -13,
             "G": -5, "Ab": -4, "A": -3},
    "trumpet": {"F": 5, "E": 4, "Eb": 3, "D": 2, "C": 0, "Bb": -2, "A": -3},
}
FAMILY = [  # (family, pattern in a part's name or title), first match wins
    ("oboe", r"oboe|hautbois"),
    ("cor anglais", r"ingl|anglais|english horn"),
    ("clarinet", r"clar|klar"),
    ("horn", r"\bcorn[oi]\b|\bhorn|\bcor\b"),
    ("trumpet", r"tromba|trumpet|trompet"),
    ("piccolo", r"picc"),
    ("octave down", r"contrab|kontrab|double bass|(\b|_)cb\b|contrafag|contrebass"),
]
FIXED = {"cor anglais": -7, "piccolo": 12, "octave down": -12}
KEY_IN = re.compile(r"\bin\s+([A-H])(b|♭|es|s)?\b\.?", re.I)


def family_of(text):
    for family, pattern in FAMILY:
        if re.search(pattern, text, re.I):
            return family
    return None


def transpositions(name, words):
    """[((movement, measure), semitones)] in order: where the part starts
    sounding at a new pitch from what is written. From the part's name and
    title, and printed keys ("in Bb", "in A.", "muta in E.", "Cor. ingl.")."""
    early = " ".join(text for mv, measure, text in words if mv == 1 and measure <= 2)
    family = family_of(name) or family_of(early)
    if family is None:
        return []
    out = [((1, 1), FIXED.get(family, 0))]
    for mv, measure, text in sorted(words):
        if family == "oboe":
            # an oboist switching to cor anglais ("muta in Cor. ingl.") and
            # back; the title ("Oboe I & Cor Anglais") is no switch
            if (mv, measure) <= (1, 2):
                continue
            if re.search(r"ingl|anglais", text, re.I) and not re.search(r"oboe", text, re.I):
                out.append(((mv, measure), FIXED["cor anglais"]))
            elif re.search(r"muta in ob", text, re.I):
                out.append(((mv, measure), 0))
            continue
        m = KEY_IN.search(text)
        if not m or family not in TRANSPOSE or family_of(text) not in (None, family):
            continue
        letter, acc = m.group(1).upper(), (m.group(2) or "").lower()
        key = {"H": "B"}.get(letter, letter)
        if acc in ("b", "♭", "es", "s") or (letter == "B" and not acc):
            key += "b"  # German B (and "in B" for a wind part) is B flat
        if key in TRANSPOSE[family]:
            out.append(((mv, measure), TRANSPOSE[family][key]))
    return out


NOTE_NAME = re.compile(r"([A-H])(b|♭|#|is|es|s)?(?![a-z])")


def parse_tuning(text):
    """(drums retuned, {letter: alter} tuned to, whether all drums are new)
    from a timpani tuning ("in Eb. Bb.", "in E.A.", "muta E in F"), or None.
    German names count too (Es, As, B for B flat when H is used)."""
    m = re.search(r"\bmuta\b(.*?)\bin\b\s*(.*)", text, re.I)
    fresh = m is None
    if m:
        old, new = m.group(1), m.group(2)
    else:
        m = re.search(r"\bin\s+([A-H].*)", text)
        if not m:
            return None
        old, new = "", m.group(1)
    tokens = NOTE_NAME.findall(new)
    if not tokens or re.sub(r"[\s.,;:\d]", "", NOTE_NAME.sub("", new)):
        return None  # more than note names: not a tuning
    german = "H" in new or any(a in ("es", "is", "s") for _, a in tokens)
    drums = {}
    for letter, acc in tokens:
        if letter == "H":
            drums["B"] = 0
        elif letter == "B" and german and not acc:
            drums["B"] = -1
        else:
            drums[letter] = 1 if acc in ("#", "is") else -1 if acc in ("b", "♭", "es", "s") else 0
    return [letter for letter, _ in NOTE_NAME.findall(old)], drums, fresh


def omr_stacks(omr, pages):
    """[(left, right)] in pixels of each bar Audiveris found, per system, in
    reading order."""
    out = []
    with zipfile.ZipFile(omr) as z:
        for page in pages:
            xml = z.read(f"sheet#{page}/sheet#{page}.xml").decode()
            for system in xml.split("<system ")[1:]:
                out.append([(float(a), float(b)) for a, b in
                            re.findall(r'<stack [^>]*left="([\d.]+)" right="([\d.]+)"', system)])
    return out


def place_events(data, systems, stacks):
    """Put every event on its timeline piece. Returns {piece index: {"time",
    "clef", "fifths", "voices": {voice: [event]}}}, with each event's page x
    in pixels.

    The export's line breaks aren't reliable, so its measures are matched in
    order to Audiveris's bars by width (skipping a bar the export leaves
    out), and notes are placed from the matched bar's left edge."""
    staves = sorted(data["staves"], key=lambda s: (s["page"], s["staff"]))
    if len(staves) != len(stacks):
        raise RuntimeError(f"{len(stacks)} systems recognised but {len(staves)} staves counted")
    bars = [(n, staff, left, right) for n, (staff, row) in enumerate(zip(staves, stacks))
            for left, right in row]
    pieces = {}
    for i, t in enumerate(data["timeline"]):
        pieces.setdefault((t["page"], t["staff"]), []).append(i)

    def misfit(k, measure):
        _, staff, left, right = bars[k]
        return abs((right - left) * 10 / staff["space"] - measure["width"]) / measure["width"]

    placed, k = {}, 0
    for measure in (m for system in systems for m in system["measures"]):
        measure["system"] = bars[min(k, len(bars) - 1)][0]
        # a measure with no width is the tail of a multi-bar rest, unless it
        # holds notes (Audiveris sometimes misreads a long stretch as a rest)
        sized = measure["width"] > 0
        if k >= len(bars) or not (sized or any(e["midi"] for e in measure["events"])):
            continue
        if sized and misfit(k, measure) > 0.15:
            ahead = [j for j in range(k + 1, min(k + 4, len(bars))) if misfit(j, measure) <= 0.15]
            if ahead:
                k = ahead[0]
        measure["system"], staff, left, _ = bars[k]
        k += 1
        mine = pieces.get((staff["page"], staff["staff"]), [])
        if not mine:
            continue
        scale = staff["space"] / 10
        for e in measure["events"]:
            x = left + (e["x"] if e["x"] is not None else 8) * scale
            e["px"] = x
            inside = [i for i in mine if data["timeline"][i]["x0"] <= x < data["timeline"][i]["x1"]]
            i = inside[0] if inside else min(mine, key=lambda i: abs(data["timeline"][i]["x0"] - x))
            bar = placed.setdefault(i, {"time": measure["time"], "measure": measure, "voices": {}})
            bar["voices"].setdefault(e["voice"], []).append(e)
    for bar in placed.values():
        for evs in bar["voices"].values():
            evs.sort(key=lambda e: e["px"])
    return placed


def _beats(durs, beat):
    """The bar's rhythm cut into beats: a tuple of note lengths per beat, or
    the whole bar when a note crosses a beat."""
    out, group, t = [], [], Fraction(0)
    for d in durs:
        group.append(d)
        t += d
        if t % beat == 0:
            out.append(tuple(group))
            group = []
    return out if not group else [tuple(durs)]


def _groups(evs):
    """(start, end) of each beamed group of notes."""
    out, start = [], None
    for k, e in enumerate(evs):
        if e["beam"] == "begin":
            start = k
        elif e["beam"] == "end" and start is not None:
            out.append((start, k + 1))
            start = None
    return out


def tidy(durs, length):
    """Whether no note shorter than a beat runs across a beat, as printed
    music almost never does (a misread rhythm often does)."""
    beat = Fraction(1) if length % 1 == 0 else Fraction(1, 2)
    t = Fraction(0)
    for d in durs:
        if d < beat and (t // beat) != ((t + d - Fraction(1, 1000)) // beat):
            return False
        t += d
    return True


def regular(durs, length):
    """Whether a bar's rhythm is written the way music normally is: the bar
    splits into beats (or halves), and each span into even notes, triplets,
    a dotted pair, a short-long-short syncopation, or two regular halves.
    Misreadings that still add up (a missed dot and an extra flag, a missed
    triplet mark) usually break this."""
    beat = Fraction(1) if length % 1 == 0 else Fraction(1, 2)

    def span(ds, total):
        if sum(ds) != total:
            return False
        if len(ds) == 1:
            return True
        if len(set(ds)) == 1:
            return len(ds) in (2, 3, 4, 6, 8, 12, 16)
        if ds in ([total * 3 / 4, total / 4], [total / 4, total * 3 / 4],
                  [total / 4, total / 2, total / 4], [total * 2 / 3, total / 3],
                  [total / 3, total * 2 / 3]):
            return True
        if total > beat and total % beat == 0:
            cuts = [k * beat for k in range(1, int(total / beat))]
        else:
            cuts = [total / 2]
        t, i = Fraction(0), 0
        for k, d in enumerate(ds):
            t += d
            if t in cuts and k + 1 < len(ds):
                if span(ds[:k + 1], t) and span(ds[k + 1:], total - t):
                    return True
        return False

    return span(list(durs), length)


class Part:
    """One part's bars, rebuilt for playback."""

    def __init__(self, data, omr, mxls, pdf=None, name=""):
        self.data = data
        pages = sorted({s["page"] for s in data["staves"]})
        self.systems = read_systems(mxls)
        self.placed = place_events(data, self.systems, omr_stacks(omr, pages))
        measures = [m for system in self.systems for m in system["measures"]]
        correct_clefs(measures, omr_clefs(omr, pages))
        self.heads = omr_heads(omr, pages)
        staves = sorted(data["staves"], key=lambda s: (s["page"], s["staff"]))
        spelled_with(measures, self.heads, staves)
        marks = {}
        if pdf:
            from . import accidentals
            marks = accidentals.find(omr, pdf, data["staves"])
            self._drop_read_accidentals(marks, measures, staves)
        movements = [s["movement"] for s in staves]
        keys = system_keys(omr_keys(omr, pages), movements, key_room(omr, staves))
        respell(measures, keys, marks, staves, self.heads)
        for bar in self.placed.values():
            bar["clef"], bar["fifths"] = bar["measure"]["clef"], bar["measure"]["fifths"]
        self.rests = omr_rests(omr, pages)
        self.space = {(s["page"], s["staff"]): s["space"] for s in data["staves"]}
        self._drop_weak_pitches()
        self.meter = self._meters()
        self.cues = self._cue_notes(omr, pdf) if pdf else set()
        self.words = []
        if pdf:
            from .navigator import printed_words
            self.words = printed_words(data, pdf)
            self._tune_timpani()
        self.shifts = transpositions(name, self.words)
        self.bars = {}
        self.tremolo_bars = set()  # held notes played as repeated 16ths

    def _drop_read_accidentals(self, marks, measures, staves):
        """Audiveris sometimes reads a sharp as a notehead: drop noteheads
        (and the notes Audiveris built on them) that sit on an accidental."""
        for (page, k), found in marks.items():
            space = next(s["space"] for s in staves if (s["page"], s["staff"]) == (page, k))
            spans = [(x0 - 0.3 * space, x1 + 0.3 * space, pos) for _, pos, _, (x0, x1) in found]
            heads = self.heads.get((page, k), [])
            fake = [h for h in heads if any(a <= h[0] <= b and abs(h[1] - pos) <= 1 for a, b, pos in spans)]
            if not fake:
                continue
            self.heads[(page, k)] = [h for h in heads if h not in fake]
            for measure in measures:
                if (staves[measure["system"]]["page"], staves[measure["system"]]["staff"]) != (page, k):
                    continue
                kept = []
                for e in measure["events"]:
                    if "px" in e and e["midi"]:
                        nearest = min(heads, key=lambda h: abs(h[0] - e["px"] - 0.6 * space))
                        if nearest in fake:
                            continue
                    kept.append(e)
                measure["events"][:] = kept
        for bar in self.placed.values():
            for v, evs in bar["voices"].items():
                evs[:] = [e for e in evs if any(e is x for x in bar["measure"]["events"])]

    def shift(self, i):
        """Semitones from written to sounding pitch for piece i."""
        t = self.data["timeline"][i]
        numbers = {m["title"]: m.get("number", k + 1) for k, m in enumerate(self.data["movements"])}
        here, out = (numbers[t["movement"]], t["measure"]), 0
        for at, semitones in self.shifts:
            if at > here:
                break
            out = semitones
        return out

    def _tune_timpani(self):
        """Old timpani parts write plain letters and print the tuning above
        the staff ("in Eb. Bb.", "muta E in F"): from each tuning on, a
        written E sounds as the drum tuned to E (here E flat)."""
        words = self.words
        if not any(re.search(r"timp|pauk", text, re.I) for _, _, text in words):
            return
        changes = sorted((mv, measure, tuning) for mv, measure, text in words
                         if (tuning := parse_tuning(text)) is not None)
        if not changes:
            return
        numbers = {m["title"]: m.get("number", k + 1) for k, m in enumerate(self.data["movements"])}
        for i, bar in self.placed.items():
            t = self.data["timeline"][i]
            here = (numbers[t["movement"]], t["measure"])
            drums = {}  # each movement prints its own tuning
            for mv, measure, (old, new, fresh) in changes:
                if (mv, measure) > here:
                    break
                if mv != here[0]:
                    continue
                if fresh:
                    drums = {}
                for letter in old:
                    drums.pop(letter, None)
                drums.update(new)
            for evs in bar["voices"].values():
                for e in evs:
                    e["midi"] = [12 * (octave + 1) + STEPS[step] + (drums[step] if acc is None and step in drums
                                 else acc if acc is not None else position_alter(step, bar["fifths"]))
                                 for step, octave, acc in e["spelled"]]

    def _cue_notes(self, omr, pdf):
        """Bars printed as cue notes for another part (silent in playback).

        Cue notes are small: a bar whose median notehead is under SMALL_HEAD
        of the part's usual one is a cue, if the run of such bars starts
        right after a rest (cues lead from a rest to the player's entrance;
        measured widths are too noisy to trust alone). A bar under a running
        count of the rest (_cue_bars) is a cue only if its noteheads are at
        least somewhat small too: string parts number repeated tremolo bars
        the same way."""
        widths = head_widths(omr, pdf, self.data["staves"])
        every = sorted(w for row in widths.values() for _, w in row)
        if not every:
            return set()
        usual = every[len(every) // 2]
        timeline = self.data["timeline"]
        size = {}
        for i, t in enumerate(timeline):
            here = sorted(w for x, w in widths.get((t["page"], t["staff"]), ()) if t["x0"] < x < t["x1"])
            if here:
                size[i] = here[len(here) // 2] / usual
        cues = set()
        for i in sorted(i for i, r in size.items() if r < SMALL_HEAD):
            if i - 1 in cues or (i and not timeline[i - 1]["plays"]
                                 and timeline[i - 1]["movement"] == timeline[i]["movement"]):
                cues.add(i)
        counted = {i for i in self._cue_bars(pdf) if size.get(i, 1) < 0.9}
        return cues | counted

    def _cue_bars(self, pdf):
        """Bars printed with another part's notes as a cue (under a running
        count of the rest): those are silent. Checked from each rest onward,
        bar by bar, while the cue lasts."""
        from .cues import CueNotes, by_number

        finder = CueNotes({"part": self.data}, {"part": Path(pdf)})
        numbers = {title: n for n, (_, title) in by_number(self.data).items()}
        cues, timeline = set(), self.data["timeline"]
        for i, t in enumerate(timeline):
            if not i or not t["plays"] or timeline[i - 1]["movement"] != t["movement"]:
                continue
            if (not timeline[i - 1]["plays"] or i - 1 in cues) and \
                    finder.is_cue("part", numbers[t["movement"]], t["measure"]):
                cues.add(i)
        return cues

    def _drop_weak_pitches(self):
        """Take out chord notes that rest only on weak noteheads (a printed
        number or mark read as a note), when a strong head holds the chord."""
        for i, bar in self.placed.items():
            t = self.data["timeline"][i]
            heads = self.heads.get((t["page"], t["staff"]), ())
            space = self.space[(t["page"], t["staff"])]
            for evs in bar["voices"].values():
                for e in evs:
                    if len(e["spelled"]) < 2:
                        continue
                    near = [h for h in heads if abs(h[0] - e["px"] - 0.6 * space) < 1.2 * space]
                    keep = []
                    for (step, octave, _), midi in zip(e["spelled"], e["midi"]):
                        pos = middle_line(bar["clef"]) - (7 * octave + "CDEFGAB".index(step))
                        found = [h for h in near if h[1] == pos]
                        keep.append(not found or any(not h[2] for h in found))
                    if any(keep) and not all(keep):
                        e["spelled"] = [x for x, k in zip(e["spelled"], keep) if k]
                        e["midi"] = [x for x, k in zip(e["midi"], keep) if k]
                    e["midi"] = sorted(set(e["midi"]))

    def _meters(self):
        """Bar length of each movement: whichever of its printed time
        signatures (Audiveris misreads some) or the length its bars most
        often add up to fits the most bars."""
        sums, printed = {}, {}
        for i, bar in self.placed.items():
            t = self.data["timeline"][i]
            if bar["time"]:
                printed.setdefault(t["movement"], set()).add(bar["time"])
            if not t["plays"] or t["count"] != 1:
                continue
            for evs in bar["voices"].values():
                total = sum(e["dur"] for e in evs if not e["grace"])
                if total:
                    sums.setdefault(t["movement"], Counter())[total] += 1
        meters = {}
        for mv, c in sums.items():
            options = printed.get(mv, set()) | {c.most_common(1)[0][0]}
            meters[mv] = max(options, key=lambda m: c[m])
        return meters

    def length(self, i):
        t = self.data["timeline"][i]
        return self.meter.get(t["movement"], Fraction(4)) * t["count"]

    def head_onsets(self, i, with_x=False):
        """The pitch positions at each onset Audiveris saw noteheads, by x.
        Weak heads beside a strong one at the same onset are left out."""
        t = self.data["timeline"][i]
        heads = [h for h in self.heads.get((t["page"], t["staff"]), ()) if t["x0"] < h[0] < t["x1"]]
        groups = onsets([h[0] for h in heads], self.space[(t["page"], t["staff"])])
        found = []
        for g in groups:
            here = [h for h in heads if g[0] <= h[0] <= g[-1]]
            strong = [h for h in here if not h[2]]
            if not strong:
                found.append(None)  # only weak heads: a number, mark or beam read as notes
                continue
            found.append(sorted({p for _, p, _ in strong}))
        if with_x:
            return [(sum(g) / len(g), ps) for g, ps in zip(groups, found) if ps is not None]
        return [ps for ps in found if ps is not None]

    def _same_rhythm(self, i, clean):
        """Rebuild a bar from its noteheads, borrowing the rhythm of a clean
        bar with as many onsets spaced the same way. Pitches come from the
        notes Audiveris read at each onset, else from the notehead's place on
        the staff. None if no clean bar is spaced closely enough."""
        t, bar = self.data["timeline"][i], self.placed[i]
        mine = self.head_onsets(i, with_x=True)
        if len(mine) < 2:
            return None

        def shape(onsets_x, piece):
            span = piece["x1"] - onsets_x[0][0]
            return [(x - onsets_x[0][0]) / span for x, _ in onsets_x]

        target, best = shape(mine, t), None
        for j in clean.get(t["movement"], []):
            voices = self.bars[j]["voices"]
            if self.length(j) != self.length(i) or len(voices) != 1:
                continue
            theirs = self.head_onsets(j, with_x=True)
            if len(theirs) != len(mine) or len(voices[0]) != len(mine) or not all(m for _, m in voices[0]):
                continue
            gap = max(abs(a - b) for a, b in zip(target, shape(theirs, self.data["timeline"][j])))
            if gap < 0.05 and (best is None or gap < best[0]):
                best = (gap, j)
        if best is None:
            return None
        durations = [d for d, _ in self.bars[best[1]]["voices"][0]]
        return [self._onset_notes(i, mine, durations)], best[1]

    def _onset_notes(self, i, onsets_x, durations):
        """One note per notehead onset with the given durations. Pitches come
        from the notes Audiveris read there, else from the notehead's place
        on the staff."""
        t, bar = self.data["timeline"][i], self.placed[i]
        events = [e for evs in bar["voices"].values() for e in evs if e["midi"]]
        space = self.space[(t["page"], t["staff"])]
        notes = []
        for (x, positions), d in zip(onsets_x, durations):
            near = [e for e in events if abs(e["px"] + 0.6 * space - x) < 0.8 * space]
            midi = near[0]["midi"] if near else [position_midi(p, bar["clef"], bar["fifths"]) for p in positions]
            notes.append((d, midi))
        return notes

    def _even(self, i):
        """Rebuild a bar of running notes (evenly spaced noteheads) that
        Audiveris played short: one even note per notehead. A gap twice the
        usual spacing is a notehead it missed; running figures repeat, so it
        takes the pitch two notes back. None unless the spacing is even and
        the count divides the bar."""
        mine = self.head_onsets(i, with_x=True)
        played = sum(1 for evs in self.placed[i]["voices"].values() for e in evs if e["midi"])
        if len(mine) < 3 or played >= 16:
            return None
        gaps = [b[0] - a[0] for a, b in zip(mine, mine[1:])]
        step = sorted(gaps)[len(gaps) // 2]
        slots = [round(g / step) for g in gaps]
        if any(k not in (1, 2) or abs(g - k * step) > 0.25 * step for g, k in zip(gaps, slots)):
            return None
        onsets = [mine[0]]
        for (x, positions), k in zip(mine[1:], slots):
            if k == 2:
                onsets.append((x - step, onsets[-2][1] if len(onsets) >= 2 else positions))
            onsets.append((x, positions))
        if len(onsets) not in (4, 6, 8, 12, 16) or played >= len(onsets):
            return None
        length = self.length(i)
        return [self._onset_notes(i, onsets, [length / len(onsets)] * len(onsets))]

    def rebuild(self):
        """Fill self.bars {piece: {"voices": [[(dur, midi)]], "how": ...}}."""
        clean, pending = {}, []
        for i, t in enumerate(self.data["timeline"]):
            bar = self.placed.get(i)
            if not t["plays"] or not bar or i in self.cues:
                self.bars[i] = {"voices": [], "how": "cue" if i in self.cues else "rest"}
                continue
            length = self.length(i)
            voices = [[e for e in evs if not e["grace"]] for evs in bar["voices"].values()]
            voices = [v for v in voices if v]
            # a voice that repeats another note for note is the same notes read twice
            unique = []
            for v in voices:
                if not any([(e["dur"], e["midi"]) for e in v] == [(e["dur"], e["midi"]) for e in u] for u in unique):
                    unique.append(v)
            voices = unique
            tremolo = self._tremolo(i, voices, length)
            if tremolo:
                self.bars[i] = {"voices": [tremolo], "how": "tremolo"}
                continue
            notes = sum(1 for v in voices for e in v if e["midi"])
            seen = len(self.head_onsets(i))
            fits = all(sum(e["dur"] for e in v) == length and tidy([e["dur"] for e in v], length)
                       and regular([e["dur"] for e in v], length) for v in voices)
            if fits and (notes == seen or len(voices) > 1):
                self.bars[i] = {"voices": [[(e["dur"], e["midi"]) for e in v] for v in voices],
                                "how": "adds up"}
                clean.setdefault(t["movement"], []).append(i)
            else:
                pending.append((i, voices, length))

        vocab = self._vocabulary(clean)
        fixed_bars, left = {}, []
        for i, voices, length in pending:
            if self._borrow(i, clean):
                continue
            voices = [[e for e in v if not self._unsupported(i, e)] for v in voices]
            voices = [v for v in voices if v]
            fixed, how = [], "adds up"
            for v in voices:
                voice, h = self._fix(i, v, length, vocab.get(self.data["timeline"][i]["movement"], Counter()))
                fixed.append(voice)
                if h != "adds up":
                    how = h if how in ("adds up", "fixed") else how
            if how == "adds up":
                how = "unsure"  # fills the bar, but not every notehead was played
            self.bars[i] = {"voices": fixed, "how": how}
            if how == "fixed":
                fixed_bars.setdefault(self.data["timeline"][i]["movement"], []).append(i)
            else:
                left.append(i)
        # a second look for what's left, now also against the repaired bars
        for i in left:
            self._borrow(i, fixed_bars)
        self._repeat_unread()
        return self.bars

    def _repeat_unread(self):
        """A bar with no notes read and no rest printed, after a bar of one
        held note or chord, repeats that bar: tremolos and hollow notes are
        often missed, and such passages hold or repeat. A one-bar rest the
        counter found where Audiveris saw no rest symbol counts too: thick
        tremolo slashes look like a rest."""
        timeline = self.data["timeline"]
        for i, t in enumerate(timeline):
            symbol = any(t["x0"] < x < t["x1"] for x in self.rests.get((t["page"], t["staff"]), ()))
            # nothing playable read: no notes, or only ones too long to fit the bar
            playable = [e for evs in self.placed.get(i, {}).get("voices", {}).values()
                        for e in evs if e["midi"] and e["dur"] <= self.length(i)]
            if (i and self.bars[i]["how"] == "rest" and t["rest"] <= 1 and not symbol and t["count"] == 1
                    and i not in self.cues and not playable
                    and timeline[i - 1]["movement"] == t["movement"]
                    and self.length(i - 1) == self.length(i)
                    and len(self.bars[i - 1]["voices"]) == 1
                    and len(self.bars[i - 1]["voices"][0]) == 1
                    and self.bars[i - 1]["voices"][0][0][1]):
                self.bars[i] = {"voices": self.bars[i - 1]["voices"], "how": "repeated"}
                # unread because of tremolo slashes: it and the note it repeats tremble
                self.tremolo_bars.update({i, i - 1})

    def _tremolo(self, i, voices, length):
        """A fingered tremolo (two notes alternating, joined by slashes):
        Audiveris reads it as two voices of one long note each, side by side
        rather than stacked as divisi notes are. Returns the two notes, each
        half the bar, or None."""
        if len(voices) != 2 or any(len(v) != 1 or not v[0]["midi"] for v in voices):
            return None
        a, b = sorted((v[0] for v in voices), key=lambda e: e["px"])
        t = self.data["timeline"][i]
        if b["px"] - a["px"] < 1.5 * self.space[(t["page"], t["staff"])]:
            return None
        return [(length / 2, a["midi"]), (length / 2, b["midi"])]

    def _unsupported(self, i, e):
        """A filled-in note no strong notehead stands under (a mark or dot
        read as a note). Hollow heads (half and whole notes) always read
        weakly, so those are kept."""
        if not e["midi"] or e["type"] in ("half", "whole", "breve"):
            return False
        t = self.data["timeline"][i]
        space = self.space[(t["page"], t["staff"])]
        near = [h for h in self.heads.get((t["page"], t["staff"]), ())
                if abs(h[0] - e["px"] - 0.6 * space) < 1.2 * space and not h[2]]
        return not near

    def _borrow(self, i, sources):
        """Take a bar's reading or rhythm from a matching bar in `sources`."""
        twin = self._twin(i, sources)
        if twin is not None:
            self.bars[i] = {"voices": self.bars[twin]["voices"], "how": "same as", "twin": twin}
            return True
        borrowed = self._same_rhythm(i, sources)
        if borrowed is not None:
            self.bars[i] = {"voices": borrowed[0], "how": "same rhythm as", "twin": borrowed[1]}
            return True
        even = self._even(i)
        if even is not None:
            self.bars[i] = {"voices": even, "how": "even"}
            return True
        return False

    def _vocabulary(self, clean):
        """How often each one-beat rhythm appears in clean bars, per movement."""
        vocab = {}
        for mv, bars in clean.items():
            c = vocab.setdefault(mv, Counter())
            for i in bars:
                beat = Fraction(1) if self.length(i) % 1 == 0 else Fraction(1, 2)
                for v in self.bars[i]["voices"]:
                    c.update(_beats([d for d, _ in v], beat))
        return vocab

    def _twin(self, i, clean):
        """A clean bar in the same movement whose noteheads this bar contains,
        onset for onset. A few extra onsets here are allowed (misread heads)."""
        t = self.data["timeline"][i]
        mine = self.head_onsets(i)
        if len(mine) < 2:
            return None
        for j in sorted(clean.get(t["movement"], []), key=lambda j: abs(j - i)):
            if self.length(j) != self.length(i):
                continue
            theirs, k, extra = self.head_onsets(j), 0, 0
            for onset in mine:
                if k < len(theirs) and set(theirs[k]) <= set(onset):
                    k += 1
                else:
                    extra += 1
            if len(theirs) >= 2 and k == len(theirs) and extra <= max(1, len(theirs) // 4):
                return j
        return None

    def _fix(self, i, evs, length, vocab):
        """The most likely rhythm for a bar that doesn't add up, or adds up
        but isn't regularly written. Returns ([(duration, midi)], how).

        Candidates: one note or one beamed group doubled or halved (a missed
        or extra beam), a group made triplets or un-tripleted (a missed or
        invented triplet mark), one note dotted or undotted, two such note
        changes, or the written lengths kept with each note placed where it
        sits on the page (missed rests). Regular rhythms win, then fewer
        changes, then the rhythms the part uses elsewhere, then the closest
        fit to the spacing on the page."""
        durs = [e["dur"] for e in evs]
        if sum(durs) == length and tidy(durs, length) and regular(durs, length):
            return [(d, e["midi"]) for d, e in zip(durs, evs)], "adds up"
        beat = Fraction(1) if length % 1 == 0 else Fraction(1, 2)
        options = []

        def consider(voice, edits):
            ds = [d for d, _ in voice]
            if sum(ds) != length or any(d <= 0 for d in ds) or not tidy(ds, length):
                return
            known = sum(vocab[b] for b in _beats(ds, beat))
            fit = self._misfit(evs, [d for d, _ in voice], length) if len(voice) == len(evs) else 1.0
            options.append((not regular(ds, length), edits, -known, fit, voice))

        def notes(ds):
            return [(d, e["midi"]) for d, e in zip(ds, evs)]

        singles = [(k, new) for k, d in enumerate(durs)
                   for new in (d * 2, d / 2, d * Fraction(3, 2), d * Fraction(2, 3))]
        for k, new in singles:
            consider(notes(durs[:k] + [new] + durs[k + 1:]), 1)
        for a, b in _groups(evs) + [(0, len(evs))]:
            for f in (2, Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(4, 3), Fraction(3, 4),
                      4, Fraction(1, 4)):
                consider(notes(durs[:a] + [d * f for d in durs[a:b]] + durs[b:]), 1)
        # all notes even (a figure read with too few or too many beams)
        if len(durs) in (2, 3, 4, 6, 8, 12, 16):
            consider(notes([length / len(durs)] * len(durs)), 2)
        # the bar's first beat read right, the rest repeating it
        t, k = Fraction(0), 0
        while k < len(durs) and t < beat:
            t, k = t + durs[k], k + 1
        if t == beat and 0 < k < len(durs) and len(durs) % k == 0 and len(durs) // k == length / beat:
            consider(notes(durs[:k] * (len(durs) // k)), 2)
        if len(durs) <= 16 and not any(o[0] is False and o[1] == 1 for o in options):
            for n, (k1, new1) in enumerate(singles):
                for k2, new2 in singles[n + 1:]:
                    if k2 != k1:
                        trial = list(durs)
                        trial[k1], trial[k2] = new1, new2
                        consider(notes(trial), 2)
        placed = self._placed(i, evs, length)
        if placed:
            # nothing read is changed, only silence added: Audiveris often drops rests
            consider(placed, Fraction(1, 2))
        if options:
            options.sort(key=lambda o: o[:4])
            best = options[0]
            if sum(durs) == length and best[0]:
                # adds up already and nothing regular was found: keep it
                return notes(durs), "unsure"
            return best[4], "fixed"
        if sum(durs) == length:
            return notes(durs), "unsure"
        return notes(self._spaced(evs, length)), "spaced"

    def _placed(self, i, evs, length):
        """Written lengths kept, each note starting where it sits on the page
        (on a 16th grid), with rests filling the gaps. None if notes would
        overlap or run past the bar."""
        if not evs:
            return None
        t = self.data["timeline"][i]
        start, end = evs[0]["px"], t["x1"] - 0.5 * self.space[(t["page"], t["staff"])]
        if end <= start:
            return None
        grid = Fraction(1, 4)
        voice, now = [], Fraction(0)
        for e in evs:
            onset = round(Fraction((e["px"] - start) / (end - start)).limit_denominator(64) * length / grid) * grid
            if onset < now or onset + e["dur"] > length:
                return None
            if onset > now:
                voice.append((onset - now, []))
            voice.append((e["dur"], e["midi"]))
            now = onset + e["dur"]
        if now < length:
            voice.append((length - now, []))
        return voice

    @staticmethod
    def _misfit(evs, durs, length):
        """How far note lengths are from the gaps between notes on the page."""
        xs = [e["px"] for e in evs]
        if len(xs) < 2:
            return 0.0
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        gaps.append(sum(gaps) / len(gaps))
        if sum(gaps) <= 0:
            return 1.0
        return sum(abs(float(d / length) - g / sum(gaps)) for d, g in zip(durs, gaps))

    @staticmethod
    def _spaced(evs, length):
        xs = [e["px"] for e in evs]
        if len(xs) < 2:
            return [length]
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        gaps.append(sum(gaps) / len(gaps))
        if sum(gaps) <= 0:
            return [length / len(evs)] * len(evs)
        grid = Fraction(1, 4) if length >= 1 else Fraction(1, 8)
        durs = [max(grid, round(Fraction(g / sum(gaps)).limit_denominator(64) * length / grid) * grid)
                for g in gaps]
        durs[-1] += length - sum(durs)
        if durs[-1] <= 0:
            total = sum(e["dur"] for e in evs) or 1
            durs = [e["dur"] * length / total for e in evs]
        return durs

    def notes(self, movement=None, first=None, last=None):
        """[(start, duration, sounding midi pitches)] in quarters, plus the
        start of each bar: {piece: start}."""
        out, starts, t0 = [], {}, Fraction(0)
        for i, t in enumerate(self.data["timeline"]):
            if movement and not t["movement"].startswith(movement):
                continue
            if first and t["measure"] + t["count"] - 1 < first:
                continue
            if last and t["measure"] > last:
                continue
            starts[i] = t0
            ties = self._tie_flags(i)
            for n, v in enumerate(self.bars[i]["voices"]):
                s, k = t0, 0
                pitched = [m for _, m in v if m]
                for d, midi in v:
                    if midi:
                        sound = [m + self.shift(i) for m in midi]
                        if i in self.tremolo_bars:
                            out += [(s + j * Fraction(1, 4), Fraction(1, 4), sound) for j in range(int(d * 4))]
                            k += 1
                            s += d
                            continue
                        held = (n == 0 and len(ties) == len(pitched) and ties[k])
                        before = next((j for j in range(len(out) - 1, -1, -1)
                                       if out[j][0] + out[j][1] == s and out[j][2] == sound), None)
                        if held and before is not None:
                            out[before] = (out[before][0], out[before][1] + d, sound)  # a tie: hold on
                        else:
                            out.append((s, d, sound))
                        k += 1
                    s += d
            t0 += self.length(i)
            if t.get("fermata"):
                t0 += 1
        return out, starts

    def _tie_flags(self, i):
        """For the first voice's notes in bar i, in order: tied to the note
        before (so held, not played again)."""
        voices = list(self.placed.get(i, {}).get("voices", {}).values())
        if not voices:
            return []
        return [any(e.get("tied") or [False]) for e in voices[0] if e["midi"] and not e["grace"]]

    def summary(self):
        return Counter(b["how"] for b in self.bars.values())


def render(notes, tempo, out, rate=44100):
    """Write a WAV with a soft string-like tone."""
    import numpy as np
    from scipy.io import wavfile

    beat = 60 / tempo
    end = max(float(s + d) for s, d, _ in notes) * beat + 1
    audio = np.zeros(int(end * rate))
    for s, d, pitches in notes:
        secs = max(float(d) * beat * 0.95, 0.05)
        n = int(secs * rate)
        tt = np.arange(n) / rate
        env = np.minimum(1, np.minimum(tt / min(0.04, secs / 3), (secs - tt) / min(0.08, secs / 3))).clip(0)
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * tt) * np.clip(tt / 0.25, 0, 1)
        for m in pitches:
            phase = 2 * np.pi * 440 * 2 ** ((m - 69) / 12) * np.cumsum(vib) / rate
            wave = sum(a * np.sin(k * phase) for k, a in enumerate((1, .5, .33, .2, .12, .08), 1))
            i = int(float(s) * beat * rate)
            audio[i:i + n] += wave * env / len(pitches)
    audio = audio / max(np.abs(audio).max(), 1e-9) * 0.8
    wavfile.write(out, rate, (audio * 32767).astype(np.int16))


if __name__ == "__main__":
    import json
    import sys

    result, out = Path(sys.argv[1]), sys.argv[2]
    data = json.loads((result / "measure_count.json").read_text())
    part_dir = Path("evaluation/parts") / result.name
    mxls = export_scores(part_dir / "part.omr")
    part = Part(data, part_dir / "part.omr", mxls)
    part.rebuild()
    print(dict(part.summary()))
    mv = sys.argv[3] if len(sys.argv) > 3 else None
    first, last = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (None, None)
    notes, _ = part.notes(mv, first, last)
    render(notes, 112, out)
