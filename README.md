# batutti

*every bar, for everyone*

An AI rehearsal assistant for orchestras. Upload the scanned parts of a piece
and batutti:

- **stamps measure numbers** at the start of every line, on the player's own
  pages, and asks the player only about bars it is unsure of;
- **checks the whole set** against itself: rehearsal letters and movement
  lengths must agree across parts, and an agent explains each disagreement as
  a misprint in the part or a misreading by the app;
- **prints cues** (*rest easy, come in right*): a few small bars of an audible
  line from another part, just before entrances after long rests;
- **finds a place** the conductor names ("two before D in the Ballade", "the
  cor anglais solo") and shows that bar in every part.

Orchestral players spend a long time pencilling measure numbers into parts
that don't have them, and a single wrong number derails rehearsals ("from
m.98" must mean the same bar on every stand). Numbering is done by code and is
exact or asks; the AI is used only where judgment is needed.

## How it works

1. **Audiveris** (open-source optical music recognition) finds staves,
   barlines and notes on each page.
2. **The counter** (`measure_numbers/counting.py`) reads rests and the numbers
   printed above multi-bar rests, finds barlines Audiveris missed or invented,
   applies 1st/2nd-ending numbering, and flags anything uncertain.
3. **The review agent** (`review.py`) works through the flags when a
   movement's total disagrees with the score. It may only choose counts the
   scan supports, and asks the player about anything else.
4. **The set checker** (`letters.py`, `setcheck.py`) compares the parts in
   code, and an agent decides whether each outlier is a misprint, a misreading,
   or unclear.
5. **Cues** (`cues.py`): code finds entrances after 4+ bars of rest, an agent
   picks the most audible melodic line for each, and code checks the pick and
   prints those bars in clear space before the entrance.
6. **The navigator** (`navigator.py`) parses bar numbers and letters itself and
   hands anything else to an agent with search tools.
7. **The stamper** writes the numbers onto the original scan, in clear space so
   nothing on the page is covered.

All agents use gpt-6-luna through Portkey, with low reasoning effort, call
caps, and every tool call checked in code. Token use and cost are logged for
each run (`measure_numbers/usage.py`) and shown in the app.

## Run it

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add PORTKEY_API_KEY (optional, enables the AI)
python app.py
```

Open http://127.0.0.1:8050 and click **Try an example (Karelia Suite, all 29
parts)**. It uses saved recognition, so it works without installing Audiveris,
and takes about two minutes. Without a key, parts are still numbered, checked
and cued (cues use a simple rule instead of the agent). The **One part** tab
has a single-part example too.

To number your own scans, install [Audiveris](https://github.com/Audiveris/audiveris/releases)
(5.11; macOS, Windows and Linux installers include Java). The app finds it in
the usual install location, or set `AUDIVERIS` in `.env` to its path.
Recognition takes about 25 seconds a page.

The app looks for `.env` in this folder first, then in the folder above.

## Accuracy

`evaluation/parts` holds all 29 parts of the *Karelia Suite* (strings, winds,
brass, timpani and percussion) with answer keys checked by a player: 916
staves, each staff's printed starting number, and every movement total. The
Oboe I scan misprints a 23-bar rest where the score has 13; its key follows the
print, and the app flags the 10 extra bars against the score.

```bash
python evaluation/evaluate.py
```

All 29 parts are currently exact. The four non-violin string parts and all 24
wind, brass and percussion parts were first run unseen; their first-run results
and the fixes they led to are described in the report.

## Safety tests

```bash
python evaluation/safety_tests.py SET_JOB_DIR OBOE_1_JOB_DIR [--live]
```

Tries changes the evidence doesn't support, invented findings, bars that don't
exist, and prompt injection (typed, and printed inside a part). Results from
the last run are in `evaluation/results/safety_results.json`.

## Report and slides

`report.html` and `slides.html` are built from `docs/` with every image
embedded:

```bash
python docs/build.py
```
