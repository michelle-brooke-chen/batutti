"""Run the whole-set check on the confirmed Karelia parts (results from evaluate.py).

    python evaluation/set_check.py [--no-agent]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from measure_numbers import setcheck  # noqa: E402
from measure_numbers.keys import portkey_key  # noqa: E402
from check_letters import main as read_all_letters  # noqa: E402


def main(use_agent=True):
    if not (HERE / "letters.json").exists():
        read_all_letters()
    names = sorted(p.name for p in (HERE / "parts").iterdir() if p.is_dir())
    parts = {n: json.loads((HERE / "results" / n / "measure_count.json").read_text())
             for n in names}
    found = {}
    from measure_numbers import counting, letters
    for n in names:
        cache = HERE / "results" / n / "ocr_cache.json"
        counting._ocr_cache.update({k: tuple(v) for k, v in json.loads(cache.read_text()).items()})
        found[n] = letters.read_letters(parts[n], HERE / "parts" / n / "part.pdf")
    findings, agreed, totals = setcheck.find_disagreements(parts, found)
    print(json.dumps(findings, indent=1))
    if use_agent:
        pretty = {n: n.removeprefix("karelia_").replace("_", " ").title() for n in names}
        verdicts, trace = setcheck.run_set_check(parts, findings, agreed, totals,
                                                 portkey_key(), pretty)
        print(json.dumps(verdicts, indent=1))
        print("summary:", trace["summary"])
        print("calls", trace["model_calls"], "tokens", trace["tokens_in"], trace["tokens_out"])
        (HERE / "results" / "set_check_trace.json").write_text(json.dumps(
            {"findings": findings, "verdicts": verdicts, "trace": trace}, indent=1))


if __name__ == "__main__":
    main("--no-agent" not in sys.argv)
