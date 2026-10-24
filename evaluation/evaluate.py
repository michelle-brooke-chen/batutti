"""Score the measure counter against hand-checked answer keys.

Each folder in evaluation/parts holds a scanned part (part.pdf), its saved
Audiveris result (part.omr) and an answer key confirmed by a player: measures
per movement and, staff by staff, how many numbers each staff adds and the
number printed at its start. Run after every rule change; all parts should
stay exact.

Usage: python evaluation/evaluate.py [part folder names...]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from measure_numbers.counting import count_part  # noqa: E402


def score(part_dir):
    """Count one part and compare it with its answer key."""
    key = json.loads((part_dir / "answer_key.json").read_text())
    data = count_part(part_dir / "part.omr", part_dir / "part.pdf",
                      HERE / "results" / part_dir.name)
    got = {(s["page"], s["staff"]): s for s in data["staves"]}
    wrong, checked = [], 0
    for movement, rows in key["staff_counts"].items():
        for row in rows:
            checked += 1
            st = got.get((row["page"], row["staff"]))
            if st is None or st["measures"] != row["measures"] or (
                    "first_measure" in row and st["first_measure"] != row["first_measure"]):
                wrong.append(f"{movement} p{row['page']} staff {row['staff']}")
    totals = {}
    for m in data["movements"]:
        # titles are matched ignoring spaces: some scans print "I.Intermezzo."
        title = next((k for k in key["movement_totals"]
                      if m["title"].replace(" ", "").startswith(k.replace(" ", ""))), m["title"])
        totals[title] = (m["measures"], key["movement_totals"].get(title))
    flagged = len({(b["page"], b["staff"]) for b in data["bars"] if b.get("flags")})
    return part_dir.name, checked, wrong, totals, flagged


def main(names):
    parts = [p for p in sorted((HERE / "parts").iterdir())
             if p.is_dir() and (not names or p.name in names)]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(score, parts))
    all_exact = True
    for name, checked, wrong, totals, flagged in results:
        moves = "  ".join(f"{t.split('.')[0]}: {got}/{true}" for t, (got, true) in totals.items())
        print(f"{name:<14} staves {checked - len(wrong)}/{checked} correct   {moves}"
              f"   flagged staves: {flagged}")
        for w in wrong:
            print(f"    wrong: {w}")
        all_exact &= not wrong and all(got == true for got, true in totals.values())
    print("ALL EXACT" if all_exact else "SOME PARTS ARE OFF")
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
