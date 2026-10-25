"""How well does playback match the music? Scores every bar batutti rebuilds
from a scan against a reference score (MusicXML of the full score, not in the
repo; put it in evaluation/reference/).

A bar is right when every note starts at the right time, lasts the right
length and has the right pitches as written (a note tied over from the bar
before counts as a note of this bar, as printed). Rhythm and pitch are also
scored on their own, and so is Audiveris's reading before repair.

Transposing parts are compared as printed: the reference labels its
clarinets in B flat throughout, though its notes follow the parts, which
switch to A for the Ballade and Alla marcia. Playback itself sounds every
part at concert pitch.

    python evaluation/playback_check.py [part ...]
"""

import json
import sys
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from measure_numbers.playback import Part, export_scores  # noqa: E402

REFERENCE = HERE / "reference" / "karelia-suite-jean-sibelius.mxl"

# reference staff name for each part folder (unpitched percussion is left out);
# the scanned Oboe I part doubles on cor anglais, so either line counts
NAMES = {
    "karelia_piccolo": "Piccolo", "karelia_flute_1": "Flute 1", "karelia_flute_2": "Flute 2",
    "karelia_oboe_1": ("Oboe 1", "English Horn"), "karelia_oboe_2": "Oboe 2",
    "karelia_clarinet_1": "Clarinet 1 in Bb", "karelia_clarinet_2": "Clarinet 2 in Bb",
    "karelia_bassoon_1": "Bassoon 1", "karelia_bassoon_2": "Bassoon 2",
    "karelia_horn_1": "Horn 1 in F", "karelia_horn_2": "Horn 2 in F",
    "karelia_horn_3": "Horn 3 in F", "karelia_horn_4": "Horn 4 in F",
    "karelia_trumpet_1": "Trumpet 1 in F", "karelia_trumpet_2": "Trumpet 2 in F",
    "karelia_trumpet_3": "Trumpet 3 in F",
    "karelia_trombone_1": "Trombone 1", "karelia_trombone_2": "Trombone 2",
    "karelia_trombone_3": "Trombone 3", "karelia_tuba": "Tuba", "karelia_timpani": "Timpani",
    "karelia_vln1": "Violins 1", "karelia_vln2": "Violins 2", "karelia_vla": "Violas",
    "karelia_vc": "Violoncellos", "karelia_cb": "Contrabasses",
}


_SCORE = []


# bars a part's print numbers differently from the score: (movement, first
# printed bar affected) -> how many bars later the print is. Oboe I's Ballade
# opens with a 23-bar rest where the score has 13 (a confirmed misprint).
OFFSETS = {"karelia_oboe_1": {(2, 24): 10}}


def score_key(folder, movement, measure):
    """The reference bar for a printed bar of a part."""
    for (mv, first), shift in OFFSETS.get(folder, {}).items():
        if mv == movement and measure >= first:
            return movement, measure - shift
    return movement, measure


def reference_bars(name):
    """{(movement number, measure): [(offset, duration, (midi, ...))]} for one
    staff of the reference score. Bars numbered with a suffix (the extra
    bars around repeats) are left out."""
    from music21 import converter

    if not _SCORE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _SCORE.append(converter.parse(REFERENCE))
    part = next(p for p in _SCORE[0].parts if p.partName == name)
    bars, movement = {}, 0
    for m in part.getElementsByClass("Measure"):
        if m.number == 1 and not m.numberSuffix:
            movement += 1
        if m.numberSuffix:
            continue
        notes = []
        for n in m.recurse().notes:
            if n.duration.isGrace:
                continue
            offset = Fraction(n.getOffsetInHierarchy(m)).limit_denominator(48)
            notes.append((offset, Fraction(n.quarterLength).limit_denominator(48),
                          tuple(sorted({p.midi for p in n.pitches}))))
        bars[(movement, m.number)] = sorted(notes)
    return bars


def ours(part, i, raw=False):
    """[(offset, duration, written midi)] of one rebuilt bar, or of
    Audiveris's reading."""
    if raw:
        bar = part.placed.get(i, {"voices": {}})
        voices = [[(e["dur"], e["midi"]) for e in evs if not e["grace"]] for evs in bar["voices"].values()]
    else:
        voices = part.bars[i]["voices"]
    notes = []
    for v in voices:
        t = Fraction(0)
        for d, midi in v:
            if midi:
                notes.append((t, d, tuple(sorted(set(midi)))))
            t += d
    return sorted(notes)


def compare(mine, theirs):
    """(rhythm right, pitches right, both right) for one bar."""
    rhythm = [(o, d) for o, d, _ in mine] == [(o, d) for o, d, _ in theirs]
    pitch = sorted(m for _, _, m in mine) == sorted(m for _, _, m in theirs)
    return rhythm, pitch, rhythm and pitch and mine == theirs


def check(folder):
    data = json.loads((HERE / "results" / folder / "measure_count.json").read_text())
    omr = HERE / "parts" / folder / "part.omr"
    part = Part(data, omr, export_scores(omr), pdf=HERE / "parts" / folder / "part.pdf", name=folder)
    part.rebuild()
    names = NAMES[folder] if isinstance(NAMES[folder], tuple) else (NAMES[folder],)
    lines = [reference_bars(n) for n in names]
    ref = {key: [line[key] for line in lines] for key in lines[0]}
    numbers = {mv["title"]: mv["number"] for mv in data["movements"]}

    scores = {"before": Counter(), "after": Counter()}
    by_how = {}
    for i, t in enumerate(data["timeline"]):
        mv = numbers[t["movement"]]
        for k in range(t["count"]):
            key = score_key(folder, mv, t["measure"] + k)
            if key not in ref:
                continue
            if t["count"] > 1 or part.bars[i]["how"] in ("rest", "cue"):
                right = not any(ref[key])  # a rest bar is right if the music rests
                for s in scores.values():
                    s.update({"bars": 1, "rhythm": right, "pitch": right, "right": right,
                              "playing": any(ref[key]), "playing right": False})
                by_how.setdefault("rest", Counter()).update({"bars": 1, "right": right})
                continue
            playing = any(ref[key])
            for name, raw in (("before", True), ("after", False)):
                mine = ours(part, i, raw)
                r, p, both = max(compare(mine, line) for line in ref[key])
                scores[name].update({"bars": 1, "rhythm": r, "pitch": p, "right": both,
                                     "playing": playing, "playing right": both and playing})
            how = part.bars[i]["how"]
            by_how.setdefault(how, Counter()).update(
                {"bars": 1, "right": max(compare(ours(part, i), line)[2] for line in ref[key])})
    return scores, by_how


def line(name, scores):
    a, b = scores["after"], scores["before"]
    print(f"{name:22} {a['bars']:4} bars  all right {b['right'] / a['bars']:4.0%} -> {a['right'] / a['bars']:4.0%}"
          f"   {a['playing']:4} playing  right {b['playing right'] / max(a['playing'], 1):4.0%}"
          f" -> {a['playing right'] / max(a['playing'], 1):4.0%}")


def main():
    folders = [f"karelia_{a}" if not a.startswith("karelia_") else a for a in sys.argv[1:]] or list(NAMES)
    results, total = {}, {"before": Counter(), "after": Counter()}
    for folder in folders:
        scores, by_how = check(folder)
        results[folder] = {"scores": {k: dict(v) for k, v in scores.items()},
                           "by_how": {k: dict(v) for k, v in by_how.items()}}
        for k in total:
            total[k].update(scores[k])
        line(folder, scores)
    line("all", total)
    out = HERE / "results" / "playback_check.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
