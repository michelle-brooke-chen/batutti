"""How far can beat counting be trusted? Compares Audiveris's bar lengths with
the confirmed bars.

For each bar Audiveris recognises (a "stack") it records the length the time
signature expects and the length its notes and rests add up to. This script
matches those stacks with the counter's bars on the confirmed parts (whose
numbering the answer keys vouch for) and reports:
  - real bars: how often the notes add up exactly, fall short, or run over
  - false barlines (Audiveris split one real bar): did both halves fall short?
  - missed barlines (one Audiveris bar holds several real ones): did it run over?

    python evaluation/beat_check.py
"""

import json
import re
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = re.compile(r'<stack ([^>]*)>')


def stacks(omr):
    """{page: [system: [(left, right, expected, duration, special)]]}"""
    out = {}
    with zipfile.ZipFile(omr) as z:
        for name in z.namelist():
            m = re.fullmatch(r"sheet#(\d+)/sheet#\1\.xml", name)
            if not m:
                continue
            xml = z.read(name).decode()
            systems = []
            for part in xml.split("<system ")[1:]:
                rows = []
                for attrs in STACK.findall(part):
                    a = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', attrs))
                    rows.append((float(a["left"]), float(a["right"]),
                                 Fraction(a.get("expected", "0")), Fraction(a.get("duration", "0")),
                                 a.get("special")))
                systems.append(rows)
            out[int(m.group(1))] = systems
    return out


def judge(expected, duration):
    if expected == 0:
        return "no time signature"
    if duration == expected:
        return "adds up"
    return "short" if duration < expected else "over"


def main():
    real, fake, missed = Counter(), Counter(), Counter()
    for part in sorted((HERE / "parts").iterdir()):
        result = HERE / "results" / part.name / "measure_count.json"
        if not result.exists():
            continue
        data = json.loads(result.read_text())
        by_page = stacks(part / "part.omr")
        for t in data["timeline"]:
            rows = by_page[t["page"]][t["staff"] - 1]
            # Audiveris bars overlapping this bar (by more than a sliver)
            inside = [r for r in rows if min(r[1], t["x1"]) - max(r[0], t["x0"]) > 0.3 * (r[1] - r[0])]
            if t["count"] == 1 and len(inside) == 1 and inside[0][4] is None:
                l, r, exp, dur, _ = inside[0]
                # does the Audiveris bar also cover a neighbour (a missed barline)?
                if (r - l) > 1.6 * (t["x1"] - t["x0"]):
                    missed[judge(exp, dur)] += 1
                else:
                    real[("notes" if t["plays"] else "rests", judge(exp, dur))] += 1
            elif t["count"] == 1 and len(inside) > 1:
                # one real bar split by a false barline: judge the halves
                verdicts = tuple(sorted(judge(r[2], r[3]) for r in inside))
                fake[verdicts] += 1
    print("Real single bars (Audiveris saw the same bar):")
    for kind in ("notes", "rests"):
        n = sum(v for (k, _), v in real.items() if k == kind)
        print(f"  {kind:<6} {n:>5} bars: " + ", ".join(
            f"{j} {real[(kind, j)]} ({real[(kind, j)] / n:.0%})"
            for j in ("adds up", "short", "over", "no time signature") if real[(kind, j)]))
    print(f"\nFalse barlines (one real bar split in two by Audiveris): {sum(fake.values())}")
    for v, n in fake.most_common():
        print(f"  halves {' + '.join(v)}: {n}")
    print(f"\nMissed barlines (one Audiveris bar holding two real ones): {sum(missed.values())}")
    for v, n in missed.most_common():
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
