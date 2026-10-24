"""Number a scanned part end to end: recognise, count, review, stamp."""

import hashlib
import json
import shutil
from pathlib import Path

from . import counting, review, stamping
from .keys import portkey_key
from .omr import run_audiveris


REUSABLE = ("part.omr", "ocr_cache.json")  # slow results that depend only on the PDF


def number_part(pdf_path, work_dir, reference_totals=None, use_agent=True, progress=None,
                cache_dir=None):
    """Run every step for one part and return a summary of the results.

    `reference_totals` lists measures per movement of the whole work, in order
    (I, II, III...); the agent uses them as cross-checks.
    `progress(step, detail)` reports what is happening, for the web page.
    `cache_dir` keeps recognition results per PDF, so the same file is fast
    the second time.
    """
    say = progress or (lambda step, detail="": None)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf = work_dir / "part.pdf"
    if Path(pdf_path).resolve() != pdf.resolve():
        shutil.copy(pdf_path, pdf)

    saved = None
    if cache_dir:
        saved = Path(cache_dir) / hashlib.sha256(pdf.read_bytes()).hexdigest()[:20]
        for name in REUSABLE:
            if (saved / name).exists() and not (work_dir / name).exists():
                shutil.copy(saved / name, work_dir / name)

    # Audiveris is slow (about 25 s a page), so its result is reused when present
    omr = work_dir / "part.omr"
    if not omr.exists():
        say("recognising", "Reading the music (about 25 seconds a page)")
        run_audiveris(pdf, work_dir)

    say("counting", "Counting measures")
    data = counting.count_part(
        omr, pdf, work_dir, progress=lambda page, pages: say("counting", f"Page {page} of {pages}"))
    if saved:
        saved.mkdir(parents=True, exist_ok=True)
        for name in REUSABLE:
            shutil.copy(work_dir / name, saved / name)

    # totals are listed for every movement of the work; a part may skip some
    totals = reference_totals or []
    reference = {m["title"]: totals[m["number"] - 1] for m in data["movements"]
                 if m.get("number", 0) <= len(totals)}
    (work_dir / "reference.json").write_text(json.dumps(reference))

    questions, trace = [], None
    key = portkey_key()
    if use_agent and key and any(b.get("flags") for b in data["bars"]):
        say("reviewing", "Checking flagged bars against the evidence")
        try:
            data, questions, trace = review.run_review(data, key, reference)
            (work_dir / "agent_trace.json").write_text(json.dumps(trace, indent=1))
        except Exception as exc:  # the numbering stands without the review
            questions = [{"staves": [], "question": "The AI review could not run"
                          f" ({type(exc).__name__}); please check the orange staves yourself."}]

    (work_dir / "result.json").write_text(json.dumps(data, indent=1))
    (work_dir / "questions.json").write_text(json.dumps(questions, indent=1))
    say("stamping", "Writing the numbered PDF")
    stamping.stamp(pdf, data, work_dir / "numbered.pdf")
    return {"data": data, "questions": questions, "trace": trace,
            "agent_used": trace is not None}


def apply_corrections(work_dir, staff_counts):
    """Re-number with the player's corrected staff counts and re-stamp the PDF."""
    work_dir = Path(work_dir)
    data = json.loads((work_dir / "result.json").read_text())
    data = review.renumber(data, staff_counts)
    (work_dir / "result.json").write_text(json.dumps(data, indent=1))
    stamping.stamp(work_dir / "part.pdf", data, work_dir / "numbered.pdf")
    return data
