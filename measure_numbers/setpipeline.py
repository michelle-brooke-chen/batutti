"""Number a whole set of parts and check them against each other.

Each part goes through the single-part pipeline (recognise, count, stamp);
then the parts' rehearsal letters and movement totals are compared, and the
set-checker agent explains each disagreement. Results go to work_dir:
set_result.json, one folder per part, and numbered_parts.zip.
"""

import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pymupdf

import shutil

from . import counting, cues, letters, navigator, setcheck
from .keys import portkey_key
from .pipeline import number_part

WORKERS = 4  # parts counted at once (recognition, when needed, is the slow step)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "part"


def number_one(pdf, work, cache_dir):
    """Number one part (no per-part AI review: the set check covers the set) and
    read its rehearsal letters. Runs in a worker process."""
    result = number_part(pdf, work, use_agent=False, cache_dir=cache_dir)
    cache = Path(work) / "ocr_cache.json"
    if cache.exists():
        counting._ocr_cache.update({k: tuple(v) for k, v in json.loads(cache.read_text()).items()})
    found = letters.read_letters(result["data"], Path(work) / "part.pdf")
    words = navigator.printed_words(result["data"], Path(work) / "part.pdf")
    cache.write_text(json.dumps(counting._ocr_cache))
    return result["data"], found, words


def staff_image(pdf, data, page, staff, out):
    """Save the staff the agent pointed to, with the app's measure numbers."""
    st = next((s for s in data["staves"] if (s["page"], s["staff"]) == (page, staff)), None)
    if st is None:
        return None
    sp = st["space"]
    pix = pymupdf.open(pdf)[page - 1].get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    k = 200 / counting.DPI
    for t in (t for t in data["timeline"] if (t["page"], t["staff"]) == (page, staff)):
        m = t["measure"]
        label = str(m) if t["count"] == 1 else f"{m}-{m + t['count'] - 1}"
        x = int(t["x0"] * k) + 4
        cv2.putText(img, label, (x, int((st["top"] + 7.2 * sp) * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (47, 111, 214), 2)
    crop = img[max(0, int((st["top"] - 5 * sp) * k)):int((st["top"] + 8 * sp) * k)]
    cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return out


def add_set_cues(parts, names, work_dir, key, say, cue_cache=None, words=None):
    """Pick and print cues into every part's numbered PDF. Returns (cue list, trace)."""
    pdfs = {s: work_dir / "parts" / s / "part.pdf" for s in parts}
    if cue_cache and Path(cue_cache).exists():
        counting._ocr_cache.update({k: tuple(v) for k, v in
                                    json.loads(Path(cue_cache).read_text()).items()})
    say("cueing", "Finding entrances after long rests")
    needs = cues.cue_needs(parts, names, pdfs, words)
    say("cueing", f"Choosing cues for {len(needs)} entrances")
    picks, trace = cues.pick_cues(needs, key)
    say("cueing", "Checking that each cue is the instrument's own music")
    replaced = cues.check_picks(needs, picks, cues.CueNotes(parts, pdfs))
    trace["checked_and_replaced"] = replaced
    cue_list = []
    for k, s in enumerate(sorted(parts, key=lambda s: names[s]), start=1):
        say("cueing", f"Printing cues: {k} of {len(parts)} parts ({names[s]})")
        mine = [n for n in needs if n["part"] == s and n["id"] in picks]
        if not mine:
            continue
        numbered = work_dir / "parts" / s / "numbered.pdf"
        cued = work_dir / "parts" / s / "numbered_cued.pdf"
        placed, spare = cues.add_cues(
            numbered, cued, parts[s],
            [({"movement": n["movement"], "measure": n["measure"]}, picks[n["id"]][0],
              names[picks[n["id"]][0]]) for n in mine],
            {o: (parts[o], pdfs[o]) for o in parts})
        shutil.move(cued, numbered)
        on_page = {(e["movement"], e["measure"]): bars for e, _, bars in placed}
        cue_list += [{"part": s, "movement": n["movement"], "measure": n["measure"],
                      "rest": n["rest"], "cue": names[picks[n["id"]][0]],
                      "reason": picks[n["id"]][1],
                      "bars": on_page.get((n["movement"], n["measure"])),
                      "where": "on the page" if (n["movement"], n["measure"]) in on_page
                      else "no room"} for n in mine]
    if cue_cache:
        Path(cue_cache).write_text(json.dumps(counting._ocr_cache))
    return cue_list, trace


def number_set(pdfs, work_dir, score_totals=None, progress=None, cache_dir=None,
               use_agent=True, add_cues=False, cue_cache=None):
    """Number every part in `pdfs` ({part name: pdf path}) and check the set.

    `score_totals` lists measures per movement from the score, if known;
    otherwise the parts are compared with each other.
    """
    say = progress or (lambda step, detail="": None)
    work_dir = Path(work_dir)
    names = {slug(name): name for name in pdfs}
    paths = {slug(name): Path(path) for name, path in pdfs.items()}
    parts, found, words = {}, {}, {}
    say("counting", f"Numbering {len(pdfs)} parts")
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(number_one, paths[s], work_dir / "parts" / s, cache_dir): s
                   for s in names}
        for k, future in enumerate(as_completed(futures), start=1):
            s = futures[future]
            parts[s], found[s], words[s] = future.result()
            say("counting", f"{k} of {len(pdfs)} parts numbered ({names[s]})")
    totals_given = {i + 1: n for i, n in enumerate(score_totals or [])}
    findings, agreed, totals = setcheck.find_disagreements(parts, found, totals_given)

    verdicts, trace = [], None
    key = portkey_key()
    if findings and use_agent and key:
        say("reviewing", f"Looking into {len(findings)} disagreement(s) between parts")
        try:
            verdicts, trace = setcheck.run_set_check(parts, findings, agreed, totals, key, names)
        except Exception as exc:  # the comparison stands without the explanation
            verdicts = [{"part": f["part"], "movement": f["movement"], "verdict": "unclear",
                         "page": None, "staff": None, "evidence": "",
                         "note": f"The agent check could not run ({type(exc).__name__}); this part"
                                 f" differs from the others by {f['total_difference']:+d} bars."}
                        for f in findings]
    else:
        verdicts = [{"part": f["part"], "movement": f["movement"], "verdict": "unclear",
                     "page": None, "staff": None, "evidence": "",
                     "note": f"{names[f['part']]} differs from the other parts by"
                             f" {f['total_difference']:+d} bars in movement"
                             f" {setcheck.NUMERALS[f['movement']]}; please check."}
                    for f in findings]
    for k, v in enumerate(verdicts):
        if v.get("page"):
            img = staff_image(work_dir / "parts" / v["part"] / "part.pdf", parts[v["part"]],
                              v["page"], v["staff"], work_dir / f"finding_{k}.png")
            v["image"] = img.name if img else None

    cue_list, cue_trace = [], None
    if add_cues:
        for s in parts:  # the cue checks read numbers these parts' caches already hold
            cache = work_dir / "parts" / s / "ocr_cache.json"
            if cache.exists():
                counting._ocr_cache.update({k: tuple(v) for k, v in
                                            json.loads(cache.read_text()).items()})
        cue_list, cue_trace = add_set_cues(parts, names, work_dir,
                                           key if use_agent else None, say,
                                           cue_cache or work_dir / "cue_ocr_cache.json", words)

    say("stamping", "Packing the numbered parts")
    with zipfile.ZipFile(work_dir / "numbered_parts.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for s, name in names.items():
            numbered = work_dir / "parts" / s / "numbered.pdf"
            if numbered.exists():
                z.write(numbered, f"{name} - numbered{' with cues' if add_cues else ''}.pdf")

    summary = {
        "parts": [{"id": s, "name": names[s],
                   "movements": {setcheck.NUMERALS[mv]: n
                                 for mv, n in setcheck.movement_totals(parts[s]).items()},
                   "flagged_staves": len({(b["page"], b["staff"]) for b in parts[s]["bars"]
                                          if b.get("flags")}),
                   "letters_read": len(found[s])}
                  for s in sorted(names, key=lambda s: names[s])],
        "totals": {setcheck.NUMERALS[mv]: n for mv, n in sorted(totals.items())},
        "totals_from": "score" if score_totals else "most parts",
        "letters": {f"{setcheck.NUMERALS[mv]} {l}": m for (mv, l), m in sorted(agreed.items())},
        "findings": findings, "verdicts": verdicts, "trace": trace,
        "cues": cue_list, "cue_trace": cue_trace,
        "movement_titles": {m.get("number", i + 1): m["title"] for data in parts.values()
                            for i, m in enumerate(data["movements"])},
        "words": {s: w for s, w in words.items()},
    }
    (work_dir / "set_result.json").write_text(json.dumps(summary, indent=1))
    return summary
