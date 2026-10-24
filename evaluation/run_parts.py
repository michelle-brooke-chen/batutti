"""Number every PDF in a folder through the full pipeline, as the app would.

    python evaluation/run_parts.py [--no-agent] FOLDER OUT_DIR [TOTALS ...]

Writes each part's results to OUT_DIR/<part name>/ and a summary to
OUT_DIR/summary.json: movement totals, flags, the agent's questions, tokens
and timings.
"""

import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure_numbers.pipeline import number_part  # noqa: E402

WORKERS = 3  # Audiveris is itself multi-threaded


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run(pdf, out_dir, totals, use_agent=True):
    work = Path(out_dir) / slug(pdf.stem)
    times, start = {}, time.time()

    def progress(step, detail=""):
        times.setdefault(step, round(time.time() - start, 1))

    try:
        result = number_part(pdf, work, reference_totals=totals, use_agent=use_agent,
                             progress=progress)
    except Exception as exc:
        return {"part": pdf.stem, "error": f"{type(exc).__name__}: {exc}"}
    data, trace = result["data"], result["trace"] or {}
    return {
        "part": pdf.stem,
        "pages": max(st["page"] for st in data["staves"]) if data["staves"] else 0,
        "movements": [[m["title"], m["measures"]] for m in data["movements"]],
        "checks": [[m["title"].split(".")[0].strip(), m["measures"], m.get("number", 1)]
                   for m in data["movements"]],
        "flagged_staves": len({(b["page"], b["staff"]) for b in data["bars"] if b.get("flags")}),
        "questions": [q["question"] for q in result["questions"]],
        "changes": trace.get("changes", []),
        "model_calls": trace.get("model_calls", 0),
        "tokens": {k: trace.get(k, 0) for k in ("tokens_in", "tokens_cached", "tokens_out",
                                                 "cost_usd")},
        "seconds": {**times, "total": round(time.time() - start, 1)},
    }


def main():
    args = sys.argv[1:]
    use_agent = "--no-agent" not in args
    args = [a for a in args if a != "--no-agent"]
    folder, out_dir = Path(args[0]), Path(args[1])
    totals = [int(t) for t in args[2:]] or None
    pdfs = sorted(folder.glob("*.pdf"))
    with ProcessPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run, pdfs, [out_dir] * len(pdfs), [totals] * len(pdfs),
                                [use_agent] * len(pdfs)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    for r in results:
        if "error" in r:
            print(f"{r['part']:<24} ERROR {r['error']}")
            continue
        got = " ".join(f"{t.split('.')[0].strip()}:{n}" for t, n in r["movements"])
        wrong = [t for t, n, number in r["checks"] if totals and n != totals[number - 1]]
        ok = "CHECK " + ",".join(wrong) if wrong else "exact"
        print(f"{r['part']:<24} {got:<24} {ok:<14} flagged {r['flagged_staves']:>2}"
              f"  questions {len(r['questions'])}  {r['seconds']['total']:>5}s")


if __name__ == "__main__":
    main()
