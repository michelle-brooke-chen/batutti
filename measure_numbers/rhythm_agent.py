"""The rhythm agent: proposes the rhythm of bars that code couldn't repair
with confidence (playback.py's "fixed", "spaced" and "unsure" bars).

For each bar it sees what Audiveris read (each note's written value, dots,
beams and the space after it on the page), the clean bars around it and
clean bars elsewhere with the same number of notes. It answers with one note
value per event, or says it is unsure. Code accepts an answer only if it
keeps every event, fills the bar exactly and no short note crosses a beat;
otherwise the bar keeps code's repair.
"""

import json
from fractions import Fraction

from openai import OpenAI

from . import usage
from .playback import tidy

BASE_URL = "https://api.portkey.ai/v1"
MODEL = "gpt-6-luna"
CALL_TIMEOUT = 120
BATCH = 25  # bars per model call
EFFORT = "low"

VALUES = {
    "32nd": Fraction(1, 8), "16th": Fraction(1, 4), "dotted 16th": Fraction(3, 8),
    "8th": Fraction(1, 2), "dotted 8th": Fraction(3, 4), "quarter": Fraction(1),
    "dotted quarter": Fraction(3, 2), "half": Fraction(2), "dotted half": Fraction(3),
    "whole": Fraction(4), "16th triplet": Fraction(1, 6), "8th triplet": Fraction(1, 3),
    "quarter triplet": Fraction(2, 3),
}
NAMES = {v: k for k, v in VALUES.items()}

INSTRUCTIONS = """You repair rhythms that optical music recognition misread in an orchestral part.

Each bar lists its events in order (notes and rests) as the recognition read them: written value,
whether it is beamed (begin/continue/end), and the space after it as a share of the bar's width.
The bar's notes do not fill its time signature, or fewer notes were read than noteheads were seen.

For each bar, give one value per event, in order, so the bar adds up exactly to its time signature.
Use only these values: 32nd, 16th, dotted 16th, 8th, dotted 8th, quarter, dotted quarter, half,
dotted half, whole, 16th triplet, 8th triplet, quarter triplet.

What usually goes wrong: a missed beam (16ths read as 8ths, 32nds as 16ths), a missed or
invented dot, a flag misread. Notes in one beamed group usually fill one beat. Notes with the
same spacing usually have the same value; a longer note leaves more space after it. The bars
before and after, and other clean bars of the movement with the same number of notes, show the
rhythms this music uses; a bar often repeats a neighbour's rhythm.

Keep every event and its order. If no rhythm is convincing, mark the bar unsure rather than
guess. Answer with set_rhythms only."""

TOOLS = [{
    "type": "function", "name": "set_rhythms",
    "description": "The repaired rhythm for each bar, or unsure.",
    "parameters": {
        "type": "object",
        "properties": {"bars": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "bar": {"type": "string"},
                "values": {"type": "array", "items": {"type": "string", "enum": list(VALUES)}},
                "unsure": {"type": "boolean"},
            },
            "required": ["bar", "values", "unsure"], "additionalProperties": False}}},
        "required": ["bars"], "additionalProperties": False}}]

# a pitch name for MIDI numbers, sharps only (the agent needs contour, not spelling)
LETTERS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(midi):
    return "rest" if not midi else "+".join(f"{LETTERS[m % 12]}{m // 12 - 1}" for m in midi)


def value_name(d):
    return NAMES.get(d, f"{d} quarters")


def rhythm_text(voice):
    return ", ".join(f"{pitch_name(m)} {value_name(d)}" for d, m in voice)


def meter(length):
    return f"{length * 2}/8" if length % 1 else f"{length}/4"


def describe(part, i, clean_by_size):
    """What the agent sees for one bar, or None if it isn't a one-voice bar."""
    t = part.data["timeline"][i]
    bar = part.placed[i]
    voices = [[e for e in evs if not e["grace"]] for evs in bar["voices"].values()]
    voices = [v for v in voices if v]
    if len(voices) != 1:
        return None
    evs = voices[0]
    width = t["x1"] - t["x0"]
    events = []
    for k, e in enumerate(evs):
        nxt = evs[k + 1]["px"] if k + 1 < len(evs) else t["x1"]
        events.append({"pitch": pitch_name(e["midi"]),
                       "read_as": value_name(e["dur"]),
                       "beam": e["beam"] or "none",
                       "space_after": round((nxt - e["px"]) / width, 2)})
    timeline = part.data["timeline"]

    def neighbour(j):
        if 0 <= j < len(timeline) and timeline[j]["movement"] == t["movement"]:
            b = part.bars.get(j, {})
            if b.get("how") in ("adds up", "same as", "same rhythm as") and len(b["voices"]) == 1:
                return rhythm_text(b["voices"][0])
        return None

    similar = [rhythm_text(part.bars[j]["voices"][0])
               for j in clean_by_size.get((t["movement"], len(evs)), []) if j != i][:3]
    return {"bar": f"m.{t['measure']} ({t['movement'].split('.')[0]})",
            "time": meter(part.length(i)), "events": events,
            "noteheads_seen": len(part.head_onsets(i)),
            "bar_before": neighbour(i - 1), "bar_after": neighbour(i + 1),
            "clean_bars_with_as_many_events": similar}


def repair(part, key, kinds=("fixed", "spaced", "unsure")):
    """Ask the agent about every bar of `part` repaired as one of `kinds`.
    Accepted answers replace the bar (how "agent"). Returns the usage trace."""
    trace = usage.new_trace(MODEL, bars_asked=0, bars_changed=0, bars_unsure=0, bars_rejected=0)
    clean_by_size = {}
    for j, b in part.bars.items():
        if b["how"] in ("adds up", "same as") and len(b["voices"]) == 1:
            t = part.data["timeline"][j]
            clean_by_size.setdefault((t["movement"], len(b["voices"][0])), []).append(j)

    asked = {}
    for i, b in part.bars.items():
        if b["how"] in kinds:
            d = describe(part, i, clean_by_size)
            if d:
                asked[d["bar"]] = (i, d)
    if not asked:
        return trace
    client = OpenAI(api_key=key, base_url=BASE_URL, default_headers={"x-portkey-api-key": key},
                    timeout=CALL_TIMEOUT, max_retries=1)
    items = list(asked.values())
    for start in range(0, len(items), BATCH):
        batch = [d for _, d in items[start:start + BATCH]]
        trace["bars_asked"] += len(batch)
        try:
            response = client.responses.create(
                model=MODEL, instructions=INSTRUCTIONS, tools=TOOLS,
                tool_choice={"type": "function", "name": "set_rhythms"},
                reasoning={"effort": EFFORT},
                input=[{"role": "user", "content": json.dumps(batch)}])
        except Exception:
            continue  # these bars keep code's repair
        usage.add(trace, response.usage)
        for out in response.output:
            if out.type != "function_call":
                continue
            for answer in json.loads(out.arguments).get("bars", []):
                if answer["bar"] not in asked:
                    continue
                i, d = asked[answer["bar"]]
                if answer.get("unsure"):
                    trace["bars_unsure"] += 1
                    continue
                durs = [VALUES.get(v) for v in answer["values"]]
                length = part.length(i)
                voice = part.placed[i]["voices"]
                evs = [e for e in next(iter(voice.values())) if not e["grace"]]
                if (None in durs or len(durs) != len(evs) or sum(durs) != length
                        or not tidy(durs, length)):
                    trace["bars_rejected"] += 1
                    continue
                part.bars[i] = {"voices": [[(dur, e["midi"]) for dur, e in zip(durs, evs)]],
                                "how": "agent", "was": part.bars[i]["how"]}
                trace["bars_changed"] += 1
    return trace
