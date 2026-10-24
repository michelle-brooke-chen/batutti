"""Draft answer keys from the app's results, for a player to confirm.

    python evaluation/draft_keys.py RESULTS_DIR SCANS_DIR

For each part in RESULTS_DIR (as written by run_parts.py) writes
evaluation/drafts/karelia_<part>/ with the scan, its saved recognition and an
answer_key.json in the same form as the confirmed keys in evaluation/parts.
A draft moves to evaluation/parts once every staff is confirmed.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TITLES = {1: "I. Intermezzo", 2: "II. Ballade", 3: "III. Alla marcia"}
TOTALS = {1: 157, 2: 162, 3: 168}


def draft(work, scan):
    data = json.loads((work / "measure_count.json").read_text())
    titles = {m["title"]: TITLES[m.get("number", 1)] for m in data["movements"]}
    key = {
        "part": f"Sibelius, Karelia Suite Op. 11, {scan.stem} (incoming/scans/{scan.name})",
        "status": "draft: from the app's count, not yet confirmed staff by staff",
        "movement_totals": {titles[m["title"]]: TOTALS[m.get("number", 1)]
                            for m in data["movements"]},
        "staff_counts": {},
    }
    for st in data["staves"]:
        key["staff_counts"].setdefault(titles[st["movement"]], []).append(
            {"page": st["page"], "staff": st["staff"], "measures": st["measures"],
             "first_measure": st["first_measure"]})
    out = HERE / "drafts" / f"karelia_{work.name}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(scan, out / "part.pdf")
    shutil.copy(work / "part.omr", out / "part.omr")
    (out / "answer_key.json").write_text(json.dumps(key, indent=1) + "\n")
    return out


def main(results, scans):
    results, scans = Path(results), Path(scans)
    by_slug = {p.stem.lower().replace(" ", "_"): p for p in scans.glob("*.pdf")}
    for work in sorted(p for p in results.iterdir() if (p / "measure_count.json").exists()):
        print(draft(work, by_slug[work.name]))


if __name__ == "__main__":
    main(*sys.argv[1:])
