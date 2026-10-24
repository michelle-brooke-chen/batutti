"""Cross-check the confirmed parts by their rehearsal letters.

Reads the letters in every part of evaluation/parts (using the counts in
evaluation/results, so run evaluate.py first), finds the measure each letter
falls at in most parts, and lists the parts that print a letter elsewhere.

    python evaluation/check_letters.py
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from measure_numbers import counting, letters  # noqa: E402

NUMERALS = ["", "I", "II", "III", "IV", "V"]


def read(name):
    work = HERE / "results" / name
    cache = work / "ocr_cache.json"
    if cache.exists():
        counting._ocr_cache.update({k: tuple(v) for k, v in json.loads(cache.read_text()).items()})
    data = json.loads((work / "measure_count.json").read_text())
    found = letters.read_letters(data, HERE / "parts" / name / "part.pdf")
    cache.write_text(json.dumps(counting._ocr_cache))
    return name, found


def main():
    names = sorted(p.name for p in (HERE / "parts").iterdir() if p.is_dir())
    with ProcessPoolExecutor(4) as pool:
        found = dict(pool.map(read, names))
    agreed = letters.consensus(found)
    print("letters: " + "  ".join(f"{NUMERALS[mv]} {l}=m.{m}" for (mv, l), m in sorted(agreed.items())))
    report = {}
    for name, f in found.items():
        off = letters.disagreements(f, agreed)
        report[name] = off
        print(f"{name:<30} " + ("ok" if not off else "DISAGREES: " + ", ".join(
            f"{NUMERALS[d['movement']]} {d['letter']} at m.{d['here']} (expected m.{d['expected']},"
            f" p{d['page']} s{d['staff']})" for d in off)))
    (HERE / "letters.json").write_text(json.dumps(
        {"agreed": {f"{mv}{l}": m for (mv, l), m in agreed.items()}, "parts": report}, indent=1))


if __name__ == "__main__":
    main()
