# Measure Numbers

Stamps measure numbers onto a scanned orchestral part, on the player's own
pages, and asks the player only about the bars it is unsure of.

Orchestral players spend a long time pencilling measure numbers into parts
that don't have them, and a single wrong number derails rehearsals ("from
m.98" must mean the same bar on every stand). This app reads the scan, counts
every measure (including multi-bar rests and 1st/2nd endings), checks itself,
and writes a numbered PDF.

## How it works

1. **Audiveris** (open-source optical music recognition) finds staves,
   barlines and notes on each page.
2. **The counter** (`measure_numbers/counting.py`) reads rests and the numbers
   printed above multi-bar rests, finds barlines Audiveris missed or invented,
   applies 1st/2nd-ending numbering, and flags anything uncertain.
3. **The review agent** (`measure_numbers/review.py`, gpt-6-luna) works
   through the flags using the tools' evidence and the movement totals from
   the score, if given. It may only choose counts the evidence supports, and
   asks the player about anything else.
4. **The stamper** writes each staff's first measure number onto the original
   scan, in clear space so nothing on the page is covered.

The player reviews pages labelled with each staff's measure range, corrects
any staff count, and downloads the numbered PDF.

## Run it

Requires Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add PORTKEY_API_KEY (optional, enables the AI review)
python app.py
```

Open http://127.0.0.1:8050 and click **Try an example**: it numbers the
Violin I part of Sibelius's *Karelia Suite* using its saved recognition, so
it works without installing Audiveris.

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
