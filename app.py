"""Measure Numbers: a web app that stamps measure numbers onto scanned parts.

Run:  python app.py   then open http://127.0.0.1:8050
"""

import base64
import json
import re
import shutil
import threading
import uuid
from pathlib import Path

import pymupdf
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update
from flask import abort, send_from_directory

from measure_numbers.keys import portkey_key
from measure_numbers.omr import audiveris_path
from measure_numbers.pipeline import apply_corrections, number_part
from measure_numbers.preview import render_pages
from measure_numbers import navigator
from measure_numbers.setpipeline import number_set, slug

HERE = Path(__file__).resolve().parent
JOBS = HERE / "jobs"
CACHE = JOBS / "cache"
EXAMPLE = HERE / "evaluation" / "parts" / "karelia_vln1"
EXAMPLE_TOTALS = "157, 162, 168"
EXAMPLE_SET = HERE / "evaluation" / "parts"  # all 29 Karelia parts, with saved recognition
SHORT_NAMES = {"vln1": "Violin I", "vln2": "Violin II", "vla": "Viola", "vc": "Cello", "cb": "Bass"}
MAX_SET_PARTS = 40
MAX_PAGES_MB = 40  # uploads larger than this are refused
MAX_PAGES = 30  # longer parts are refused, to bound recognition time and model cost
MAX_RUNNING = 2  # parts processed at once; more would starve the server
STEPS = [("recognising", "Read the music"), ("counting", "Count measures"),
         ("reviewing", "Check flagged bars"), ("stamping", "Stamp numbers"), ("done", "Done")]

jobs = {}  # job id -> {"step", "detail", "error"}
jobs_lock = threading.Lock()


# ---------- background work ----------

def set_status(job_id, **fields):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(fields)


def parse_totals(text):
    """'157, 162, 168' -> [157, 162, 168]; ignores anything that isn't a number."""
    return [int(n) for n in re.findall(r"\d+", text or "")][:20]


def run_job(job_id, pdf_path, totals, use_agent):
    work = JOBS / job_id
    try:
        def progress(step, detail=""):
            set_status(job_id, step=step, detail=detail)

        result = number_part(pdf_path, work, reference_totals=totals, use_agent=use_agent,
                             progress=progress, cache_dir=CACHE)
        set_status(job_id, step="stamping", detail="Preparing the pages for review")
        render_pages(work / "part.pdf", result["data"], work / "pages")
        set_status(job_id, step="done", detail="")  # only once everything exists
    except Exception as exc:  # shown on the page instead of crashing the app
        set_status(job_id, step="error", error=f"{type(exc).__name__}: {exc}")


def start_job(pdf_bytes=None, example=False, totals_text="", use_agent=True):
    job_id = uuid.uuid4().hex[:12]
    work = JOBS / job_id
    work.mkdir(parents=True)
    pdf = work / "part.pdf"
    if example:
        shutil.copy(EXAMPLE / "part.pdf", pdf)
        shutil.copy(EXAMPLE / "part.omr", work / "part.omr")
        cache = HERE / "evaluation" / "results" / EXAMPLE.name / "ocr_cache.json"
        if cache.exists():
            shutil.copy(cache, work / "ocr_cache.json")
    else:
        pdf.write_bytes(pdf_bytes)
    set_status(job_id, step="starting", detail="", error=None)
    thread = threading.Thread(target=run_job, daemon=True,
                              args=(job_id, pdf, parse_totals(totals_text), use_agent))
    thread.start()
    return job_id


def example_name(folder):
    name = folder.name.removeprefix("karelia_")
    return SHORT_NAMES.get(name, name.replace("_", " ").title().replace(" And ", " and "))


def run_set_job(job_id, pdfs, totals, use_agent, add_cues, cue_cache):
    work = JOBS / job_id
    try:
        def progress(step, detail=""):
            set_status(job_id, step=step, detail=detail)

        number_set(pdfs, work, score_totals=totals, progress=progress, cache_dir=CACHE,
                   use_agent=use_agent, add_cues=add_cues, cue_cache=cue_cache)
        set_status(job_id, step="done", detail="")
    except Exception as exc:
        set_status(job_id, step="error", error=f"{type(exc).__name__}: {exc}")


def start_set_job(files=(), example=False, totals_text="", use_agent=True, add_cues=True):
    """`files` is [(part name, pdf bytes)]."""
    job_id = uuid.uuid4().hex[:12]
    work = JOBS / job_id
    pdfs = {}
    if example:
        for folder in sorted(p for p in EXAMPLE_SET.iterdir() if p.is_dir()):
            name = example_name(folder)
            part = work / "parts" / slug(name)
            part.mkdir(parents=True)
            shutil.copy(folder / "part.pdf", part / "part.pdf")
            shutil.copy(folder / "part.omr", part / "part.omr")
            cache = HERE / "evaluation" / "results" / folder.name / "ocr_cache.json"
            if cache.exists():
                shutil.copy(cache, part / "ocr_cache.json")
            pdfs[name] = part / "part.pdf"
    else:
        for name, pdf_bytes in files:
            part = work / "parts" / slug(name)
            part.mkdir(parents=True, exist_ok=True)
            (part / "part.pdf").write_bytes(pdf_bytes)
            pdfs[name] = part / "part.pdf"
    # the example reuses saved readings for its cue checks
    cue_cache = HERE / "evaluation" / "results" / "cue_ocr_cache.json" if example else None
    set_status(job_id, step="starting", detail="", error=None)
    threading.Thread(target=run_set_job, daemon=True,
                     args=(job_id, pdfs, parse_totals(totals_text), use_agent, add_cues,
                           cue_cache)).start()
    return job_id


# ---------- page ----------

app = Dash(__name__, title="Measure Numbers")
server = app.server


@server.route("/jobs/<job_id>/findings/<name>")
def finding_image(job_id, name):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not re.fullmatch(
            r"(finding|nav)_[a-z0-9_]+\.png", name):
        abort(404)
    return send_from_directory(JOBS / job_id, name)


@server.route("/jobs/<job_id>/pages/<name>")
def page_image(job_id, name):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id) or not re.fullmatch(r"page_\d+\.png", name):
        abort(404)
    return send_from_directory(JOBS / job_id / "pages", name)


CARD = {"background": "white", "border": "1px solid #e3e3e3", "borderRadius": "10px",
        "padding": "20px 24px", "marginBottom": "18px"}
BUTTON = {"padding": "10px 18px", "borderRadius": "8px", "border": "none",
          "background": "#1f4e79", "color": "white", "fontSize": "15px", "cursor": "pointer",
          "marginRight": "10px"}
SECONDARY = {**BUTTON, "background": "white", "color": "#1f4e79", "border": "1px solid #1f4e79"}

agent_note = ("The AI review is available." if portkey_key() else
              "No PORTKEY_API_KEY found, so parts are numbered without the AI review.")
audiveris_note = ("" if audiveris_path() else
                  " Audiveris is not installed here, so only the example can be run.")

VERDICTS = {"misprint": ("Misprint in the part", "#b00020"),
            "reading_error": ("Possible misreading by the app", "#b26a00"),
            "unclear": ("Needs a look", "#555")}


def set_tab():
    return [
        html.Div(style=CARD, children=[
            html.H3("1. All the parts of a piece", style={"marginTop": 0}),
            html.P("Number every part at once. The parts are then checked against each other: "
                   "movement lengths and rehearsal letters must agree, which catches the app's "
                   "own slips and misprints in the parts.", style={"color": "#555"}),
            dcc.Upload(id="set-upload", accept=".pdf", multiple=True,
                       max_size=MAX_PAGES_MB * 1024 * 1024,
                       children=html.Div(id="set-upload-label",
                                         children="Drop the parts' PDFs here, or click to choose"),
                       style={"border": "2px dashed #b8c4d0", "borderRadius": "10px",
                              "padding": "26px", "textAlign": "center", "cursor": "pointer",
                              "background": "#fbfcfd"}),
            html.Label("Measures per movement from the score (optional; otherwise the parts "
                       "are compared with each other)",
                       style={"display": "block", "marginTop": "16px", "fontWeight": 600}),
            dcc.Input(id="set-totals", placeholder="e.g. 157, 162, 168", debounce=True,
                      style={"width": "260px", "padding": "8px", "marginTop": "6px"}),
            dcc.Checklist(id="set-use-agent",
                          options=[{"label": " Let the AI look into disagreements and choose cues",
                                    "value": "yes"}],
                          value=["yes"] if portkey_key() else [], style={"marginTop": "12px"}),
            dcc.Checklist(id="set-add-cues",
                          options=[{"label": " Add cues before entrances after long rests",
                                    "value": "yes"}],
                          value=["yes"], style={"marginTop": "6px"}),
            html.P(agent_note + audiveris_note, style={"color": "#777", "fontSize": "13px"}),
            html.Button("Number and check the set", id="set-start", style=BUTTON),
            html.Button("Try an example (Karelia Suite, all 29 parts)", id="set-example",
                        style=SECONDARY),
        ]),
        html.Div(id="set-progress-card", style={**CARD, "display": "none"}, children=[
            html.H3("2. Working on it", style={"marginTop": 0}),
            html.Div(id="set-progress"),
        ]),
        html.Div(id="set-results-card", style={**CARD, "display": "none"}, children=[
            html.H3("3. What the parts say", style={"marginTop": 0}),
            html.Div(id="set-findings"),
            html.H4("Find a place in the music"),
            html.P("Type what the conductor says, e.g. \u201ctwo before D in the Ballade\u201d, "
                   "\u201cm.98 Alla marcia\u201d or \u201cthe cor anglais solo\u201d. Every part "
                   "shows that bar.", style={"color": "#555"}),
            html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
                dcc.Input(id="nav-query", placeholder="two before D in the Ballade",
                          debounce=False, n_submit=0,
                          style={"flex": "1", "padding": "8px", "fontSize": "15px"}),
                html.Button("Find", id="nav-go", style=BUTTON)]),
            dcc.Dropdown(id="nav-parts", multi=True, placeholder="All parts",
                         style={"marginTop": "8px"}),
            dcc.Loading(html.Div(id="nav-result", style={"marginTop": "12px"})),
            html.H4("Cues"),
            html.Div(id="set-cues"),
            html.H4("Every part"),
            html.Div(id="set-table"),
            html.Div(style={"marginTop": "14px"}, children=[
                html.Button("Download all numbered parts (.zip)", id="set-download-button",
                            style=BUTTON)]),
            dcc.Download(id="set-download"),
        ]),
        html.Div(id="set-error", style={"color": "#b00020"}),
        dcc.Store(id="set-job-id"),
        dcc.Interval(id="set-tick", interval=1000, disabled=True),
    ]


app.layout = html.Div(style={"maxWidth": "980px", "margin": "0 auto", "padding": "24px",
                             "fontFamily": "-apple-system, Helvetica, Arial, sans-serif",
                             "background": "#f6f7f9", "color": "#222"}, children=[
    html.H1("Measure Numbers", style={"marginBottom": "4px"}),
    html.P("Upload scanned orchestral parts. The app numbers their measures on your original "
           "pages, checks itself and the parts against each other, and asks you only about "
           "what it is unsure of.", style={"color": "#555", "marginTop": 0}),
    dcc.Tabs(id="tabs", value="set", style={"marginBottom": "18px"}, children=[
    dcc.Tab(label="Whole set of parts", value="set", children=set_tab()),
    dcc.Tab(label="One part", value="one", children=[
    html.Div(style=CARD, children=[
        html.H3("1. Your part", style={"marginTop": 0}),
        dcc.Upload(id="upload", accept=".pdf", max_size=MAX_PAGES_MB * 1024 * 1024,
                   children=html.Div(id="upload-label", children="Drop a PDF here, or click to choose"),
                   style={"border": "2px dashed #b8c4d0", "borderRadius": "10px", "padding": "26px",
                          "textAlign": "center", "cursor": "pointer", "background": "#fbfcfd"}),
        html.Label("Measures per movement from the score (optional, helps catch mistakes)",
                   style={"display": "block", "marginTop": "16px", "fontWeight": 600}),
        dcc.Input(id="totals", placeholder="e.g. 157, 162, 168", debounce=True,
                  style={"width": "260px", "padding": "8px", "marginTop": "6px"}),
        dcc.Checklist(id="use-agent", options=[{"label": " Let the AI review flagged bars",
                                                "value": "yes"}],
                      value=["yes"] if portkey_key() else [], style={"marginTop": "12px"}),
        html.P(agent_note + audiveris_note, style={"color": "#777", "fontSize": "13px"}),
        html.Button("Number this part", id="start", style=BUTTON),
        html.Button("Try an example (Sibelius, Karelia Suite, Violin I)", id="example",
                    style=SECONDARY),
    ]),

    html.Div(id="progress-card", style={**CARD, "display": "none"}, children=[
        html.H3("2. Working on it", style={"marginTop": 0}),
        html.Div(id="progress"),
    ]),

    html.Div(id="results-card", style={**CARD, "display": "none"}, children=[
        html.H3("3. Check and download", style={"marginTop": 0}),
        html.Div(id="summary"),
        html.Div(id="questions"),
        html.H4("Pages"),
        html.P("Each staff shows the measure numbers on it. Orange staves have something "
               "the app was unsure about.", style={"color": "#555"}),
        html.Div(id="pages", style={"maxHeight": "640px", "overflowY": "auto",
                                    "border": "1px solid #eee", "borderRadius": "8px"}),
        html.H4("Corrections"),
        html.P("If a staff's count is wrong, change it below and apply. Count the bars on the "
               "staff; bars of a 2nd ending don't count (they reuse the 1st ending's numbers).",
               style={"color": "#555"}),
        dash_table.DataTable(
            id="staff-table", editable=False, page_size=15,
            columns=[{"name": "Movement", "id": "movement"},
                     {"name": "Page", "id": "page"}, {"name": "Staff", "id": "staff"},
                     {"name": "Numbers", "id": "numbers"},
                     {"name": "Measures (edit)", "id": "measures", "editable": True,
                      "type": "numeric"},
                     {"name": "Why it was flagged", "id": "why"}],
            style_cell={"fontFamily": "-apple-system, Helvetica, Arial, sans-serif",
                        "fontSize": "14px", "textAlign": "left",
                        "padding": "6px", "whiteSpace": "normal", "height": "auto"},
            style_data_conditional=[{"if": {"filter_query": "{why} != ''"},
                                     "backgroundColor": "#fff4e5"},
                                    {"if": {"column_id": "measures"},
                                     "backgroundColor": "#eef5ff", "fontWeight": 600}],
        ),
        html.Div(style={"marginTop": "14px"}, children=[
            html.Button("Apply corrections", id="apply", style=SECONDARY),
            html.Button("Download numbered PDF", id="download-button", style=BUTTON),
        ]),
        dcc.Download(id="download"),
    ]),
    ]),
    ]),

    html.Div(id="error", style={"color": "#b00020"}),
    dcc.Store(id="job-id"),
    dcc.Interval(id="tick", interval=1000, disabled=True),
])


@app.callback(Output("upload-label", "children"), Input("upload", "filename"))
def show_filename(name):
    return f"Selected: {name}" if name else "Drop a PDF here, or click to choose"


@app.callback(Output("job-id", "data"), Output("tick", "disabled"), Output("error", "children"),
              Input("start", "n_clicks"), Input("example", "n_clicks"),
              State("upload", "contents"), State("upload", "filename"),
              State("totals", "value"), State("use-agent", "value"),
              prevent_initial_call=True)
def start(_start, _example, contents, filename, totals, use_agent):
    from dash import ctx
    agent = bool(use_agent)
    if ctx.triggered_id == "example":
        return start_job(example=True, totals_text=totals or EXAMPLE_TOTALS,
                         use_agent=agent), False, ""
    if not contents:
        return no_update, no_update, "Choose a PDF first."
    header, _, body = contents.partition(",")
    pdf_bytes = base64.b64decode(body)
    if not pdf_bytes.startswith(b"%PDF"):
        return no_update, no_update, f"{filename} is not a PDF."
    try:
        pages = pymupdf.open(stream=pdf_bytes, filetype="pdf").page_count
    except Exception:
        return no_update, no_update, f"{filename} could not be opened as a PDF."
    if pages > MAX_PAGES:
        return no_update, no_update, f"{filename} has {pages} pages; the limit is {MAX_PAGES}."
    if not audiveris_path():
        return no_update, no_update, "Audiveris isn't installed here; try the example instead."
    with jobs_lock:
        running = sum(j.get("step") not in ("done", "error") for j in jobs.values())
    if running >= MAX_RUNNING:
        return no_update, no_update, "The app is busy with other parts; please try again in a few minutes."
    return start_job(pdf_bytes=pdf_bytes, totals_text=totals, use_agent=agent), False, ""


def step_list(step, detail):
    names = [s for s, _ in STEPS]
    current = names.index(step) if step in names else -1
    rows = []
    for k, (name, label) in enumerate(STEPS[:-1]):
        mark = "✓" if k < current or step == "done" else ("…" if k == current else "·")
        text = f"{mark}  {label}" + (f": {detail}" if k == current and detail else "")
        rows.append(html.Div(text, style={"padding": "3px 0",
                                          "fontWeight": 600 if k == current else 400}))
    return rows


@app.callback(Output("progress", "children"), Output("progress-card", "style"),
              Output("results-card", "style"), Output("tick", "disabled", allow_duplicate=True),
              Output("summary", "children"), Output("questions", "children"),
              Output("pages", "children"), Output("staff-table", "data"),
              Input("tick", "n_intervals"), State("job-id", "data"),
              prevent_initial_call=True)
def poll(_n, job_id):
    with jobs_lock:
        status = dict(jobs.get(job_id, {}))
    shown = {**CARD, "display": "block"}
    hidden = {**CARD, "display": "none"}
    if status.get("step") == "error":
        return (html.Div(f"Something went wrong: {status['error']}", style={"color": "#b00020"}),
                shown, hidden, True, no_update, no_update, no_update, no_update)
    if status.get("step") != "done":
        return (step_list(status.get("step"), status.get("detail")), shown, hidden, False,
                no_update, no_update, no_update, no_update)
    return (step_list("done", ""), shown, shown, True, *results_view(job_id))


def results_view(job_id):
    work = JOBS / job_id
    data = json.loads((work / "result.json").read_text())
    questions = json.loads((work / "questions.json").read_text())
    reference = json.loads((work / "reference.json").read_text()) if (
        work / "reference.json").exists() else {}

    # a matching total can hide two errors that cancel out, so flagged staves
    # are still pointed out
    flagged = {}
    for b in data["bars"]:
        if b.get("flags"):
            flagged.setdefault(b["movement"], set()).add((b["page"], b["staff"]))
    rows = []
    for m in data["movements"]:
        ref = reference.get(m["title"])
        unsure = len(flagged.get(m["title"], ()))
        staves = f"{unsure} flagged staff" if unsure == 1 else f"{unsure} flagged staves"
        if ref is None:
            ok = f"{staves} to check" if unsure else ""
        elif ref != m["measures"]:
            ok = f"✗ score says {ref}"
        elif unsure:
            ok = f"total matches the score; check the {staves}"
        else:
            ok = "✓ matches the score"
        rows.append(html.Tr([html.Td(m["title"].rstrip(".")), html.Td(f"{m['measures']} measures"),
                             html.Td(ok)]))
    summary = html.Table(rows, style={"marginBottom": "10px", "borderSpacing": "18px 4px"})

    trace_path = work / "agent_trace.json"
    asked = [html.Li(q["question"]) for q in questions]
    agent = []
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        agent.append(html.P(f"AI review: {trace.get('summary') or 'done.'}",
                            style={"color": "#444"}))
    if asked:
        agent += [html.B("Please check:"), html.Ul(asked)]

    pages = [html.Img(src=f"/jobs/{job_id}/pages/{p.name}?v={uuid.uuid4().hex[:6]}",
                      style={"width": "100%", "display": "block", "borderBottom": "1px solid #eee"})
             for p in sorted((work / "pages").glob("page_*.png"),
                             key=lambda p: int(p.stem.split("_")[1]))]

    why = {}
    for b in data["bars"]:
        if b.get("flags"):
            why.setdefault((b["page"], b["staff"]), []).extend(b["flags"])
    table = [{"movement": s["movement"].rstrip("."), "page": s["page"], "staff": s["staff"],
              "numbers": f"m.{s['first_measure']}-{s['last_measure']}",
              "measures": s["measures"],
              "why": "; ".join(why.get((s["page"], s["staff"]), []))}
             for s in data["staves"]]
    return summary, agent, pages, table


@app.callback(Output("summary", "children", allow_duplicate=True),
              Output("questions", "children", allow_duplicate=True),
              Output("pages", "children", allow_duplicate=True),
              Output("staff-table", "data", allow_duplicate=True),
              Input("apply", "n_clicks"), State("job-id", "data"), State("staff-table", "data"),
              prevent_initial_call=True)
def apply(_n, job_id, table):
    if not job_id:
        return no_update, no_update, no_update, no_update
    counts = {(int(r["page"]), int(r["staff"])): int(r["measures"])
              for r in table if r.get("measures") not in (None, "")}
    data = apply_corrections(JOBS / job_id, counts)
    render_pages(JOBS / job_id / "part.pdf", data, JOBS / job_id / "pages")
    return results_view(job_id)


@app.callback(Output("set-upload-label", "children"), Input("set-upload", "filename"))
def show_set_files(names):
    if not names:
        return "Drop the parts' PDFs here, or click to choose"
    return f"Selected {len(names)} parts: " + ", ".join(names[:6]) + ("…" if len(names) > 6 else "")


@app.callback(Output("set-job-id", "data"), Output("set-tick", "disabled"),
              Output("set-error", "children"),
              Input("set-start", "n_clicks"), Input("set-example", "n_clicks"),
              State("set-upload", "contents"), State("set-upload", "filename"),
              State("set-totals", "value"), State("set-use-agent", "value"),
              State("set-add-cues", "value"), prevent_initial_call=True)
def start_set(_start, _example, contents, filenames, totals, use_agent, add_cues):
    from dash import ctx
    agent, cue = bool(use_agent), bool(add_cues)
    if ctx.triggered_id == "set-example":
        return (start_set_job(example=True, totals_text=totals, use_agent=agent, add_cues=cue),
                False, "")
    if not contents:
        return no_update, no_update, "Choose the parts' PDFs first."
    if len(contents) > MAX_SET_PARTS:
        return no_update, no_update, f"That's {len(contents)} files; the limit is {MAX_SET_PARTS}."
    files = []
    for content, name in zip(contents, filenames):
        pdf_bytes = base64.b64decode(content.partition(",")[2])
        if not pdf_bytes.startswith(b"%PDF"):
            return no_update, no_update, f"{name} is not a PDF."
        try:
            pages = pymupdf.open(stream=pdf_bytes, filetype="pdf").page_count
        except Exception:
            return no_update, no_update, f"{name} could not be opened as a PDF."
        if pages > MAX_PAGES:
            return no_update, no_update, f"{name} has {pages} pages; the limit is {MAX_PAGES}."
        files.append((re.sub(r"\.pdf$", "", name, flags=re.I), pdf_bytes))
    if not audiveris_path():
        return no_update, no_update, "Audiveris isn't installed here; try the example instead."
    with jobs_lock:
        running = sum(j.get("step") not in ("done", "error") for j in jobs.values())
    if running >= MAX_RUNNING:
        return no_update, no_update, "The app is busy; please try again in a few minutes."
    return start_set_job(files=files, totals_text=totals, use_agent=agent, add_cues=cue), False, ""


SET_STEPS = [("counting", "Number every part"), ("reviewing", "Look into disagreements"),
             ("cueing", "Add cues"), ("stamping", "Pack the numbered parts"), ("done", "Done")]


@app.callback(Output("set-progress", "children"), Output("set-progress-card", "style"),
              Output("set-results-card", "style"), Output("set-tick", "disabled", allow_duplicate=True),
              Output("set-findings", "children"), Output("set-table", "children"),
              Output("set-cues", "children"), Output("nav-parts", "options"),
              Input("set-tick", "n_intervals"), State("set-job-id", "data"),
              prevent_initial_call=True)
def poll_set(_n, job_id):
    with jobs_lock:
        status = dict(jobs.get(job_id, {}))
    shown, hidden = {**CARD, "display": "block"}, {**CARD, "display": "none"}
    step = status.get("step")
    if step == "error":
        return (html.Div(f"Something went wrong: {status['error']}", style={"color": "#b00020"}),
                shown, hidden, True, no_update, no_update, no_update, no_update)
    names = [s for s, _ in SET_STEPS]
    current = names.index(step) if step in names else -1
    rows = []
    for k, (name, label) in enumerate(SET_STEPS[:-1]):
        mark = "✓" if k < current or step == "done" else ("…" if k == current else "·")
        detail = status.get("detail") if k == current else ""
        rows.append(html.Div(f"{mark}  {label}" + (f": {detail}" if detail else ""),
                             style={"padding": "3px 0", "fontWeight": 600 if k == current else 400}))
    if step != "done":
        return rows, shown, hidden, False, no_update, no_update, no_update, no_update
    return (rows, shown, shown, True, *set_results_view(job_id))


def set_results_view(job_id):
    result = json.loads((JOBS / job_id / "set_result.json").read_text())
    names = {p["id"]: p["name"] for p in result["parts"]}
    cards = []
    for v in result["verdicts"]:
        title, colour = VERDICTS.get(v["verdict"], VERDICTS["unclear"])
        cards.append(html.Div(style={"border": f"1px solid {colour}", "borderLeft": f"6px solid {colour}",
                                     "borderRadius": "8px", "padding": "12px 16px",
                                     "marginBottom": "12px"}, children=[
            html.Div(f"{title}: {names.get(v['part'], v['part'])}",
                     style={"fontWeight": 700, "color": colour}),
            html.P(v["note"], style={"margin": "6px 0"}),
            html.Img(src=f"/jobs/{job_id}/findings/{v['image']}",
                     style={"width": "100%", "border": "1px solid #eee"}) if v.get("image") else None,
            html.Details([html.Summary("Evidence"), html.P(v["evidence"], style={"color": "#555"})])
            if v.get("evidence") else None,
        ]))
    source = ("the score" if result["totals_from"] == "score" else "most parts")
    letters = ", ".join(f"{k} = m.{m}" for k, m in result["letters"].items())
    if not cards:
        cards = [html.P("✓ Every part agrees with the others: movement lengths and rehearsal "
                        "letters line up.", style={"fontWeight": 600, "color": "#1b6e3a"})]
    intro = [html.P(f"Movement lengths ({source}): " + ", ".join(
                 f"{mv} = {n}" for mv, n in result["totals"].items()), style={"color": "#555"}),
             html.P(f"Rehearsal letters found across the parts: {letters}.",
                    style={"color": "#555", "fontSize": "13px"})]
    traces = [t for t in (result.get("trace"), result.get("cue_trace")) if t]
    if traces:
        calls = sum(t["model_calls"] for t in traces)
        tokens = sum(t["tokens_in"] + t["tokens_out"] for t in traces)
        cached = sum(t.get("tokens_cached", 0) for t in traces)
        cost = sum(t.get("cost_usd", 0) for t in traces)
        intro.append(html.P(f"AI use this run: {calls} model calls, {tokens:,} tokens "
                            f"({cached:,} cached), about ${cost:.4f}.",
                            style={"color": "#555", "fontSize": "13px"}))
    rows = [html.Tr([html.Th("Part"), html.Th("Measures per movement"), html.Th("Flagged staves")])]
    for p in result["parts"]:
        cells = []
        for mv, n in p["movements"].items():
            ok = n == result["totals"].get(mv)
            cells.append(html.Span(f"{mv}: {n}{'' if ok else ' ✗'}",
                                   style={"color": "#222" if ok else "#b00020",
                                          "display": "inline-block", "minWidth": "74px"}))
        rows.append(html.Tr([html.Td(p["name"]), html.Td(cells), html.Td(p["flagged_staves"] or "")]))
    table = html.Table(rows, style={"borderSpacing": "16px 3px", "fontSize": "14px"})

    cue_rows = result.get("cues") or []
    if cue_rows:
        on_page = sum(c["where"] == "on the page" for c in cue_rows)
        by_part = {}
        for c in cue_rows:
            by_part.setdefault(names.get(c["part"], c["part"]), []).append(c)
        cue_view = [html.P(f"{on_page} cues printed just before entrances after long rests. "
                           f"{len(cue_rows) - on_page} more had no clear space beside the "
                           "entrance, so they were left out. No cues after fermatas or G.P.s, "
                           "where the conductor restarts the music.", style={"color": "#555"})]
        for part in sorted(by_part):
            items = [html.Li(f"Movement {['', 'I', 'II', 'III', 'IV', 'V'][c['movement']]}, "
                             f"m.{c['measure']} (after {c['rest']} bars rest): {c['cue']}"
                             + (f", {c['bars']} bars" if c.get("bars") else "")
                             + (" (no room on the page)" if c["where"] == "no room" else ""))
                     for c in by_part[part]]
            cue_view.append(html.Details([html.Summary(f"{part}: {len(items)} cues"),
                                          html.Ul(items)]))
    else:
        cue_view = [html.P("No cues were added.", style={"color": "#555"})]
    options = [{"label": p["name"], "value": p["id"]} for p in result["parts"]]
    return intro + cards, table, cue_view, options


@app.callback(Output("nav-result", "children"),
              Input("nav-go", "n_clicks"), Input("nav-query", "n_submit"),
              State("nav-query", "value"), State("nav-parts", "value"), State("set-job-id", "data"),
              prevent_initial_call=True)
def navigate(_go, _submit, query, chosen, job_id):
    if not job_id or not query:
        return no_update
    work = JOBS / job_id
    result = json.loads((work / "set_result.json").read_text())
    numerals = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    letters = {(numerals.index(k.split()[0]), k.split()[1]): m for k, m in result["letters"].items()}
    totals = {numerals.index(k): n for k, n in result["totals"].items()}
    titles = {int(k): t for k, t in result["movement_titles"].items()}
    names = {p["id"]: p["name"] for p in result["parts"]}
    parts = {s: json.loads((work / "parts" / s / "result.json").read_text()) for s in names}
    answer = navigator.parse(query, letters, titles)
    how = "worked out directly"
    if answer is None:
        key = portkey_key()
        if not key:
            return html.P("Try a letter or bar number, e.g. \u201c2 before D in the Ballade\u201d "
                          "(the AI navigator needs a PORTKEY_API_KEY).", style={"color": "#b00020"})
        words = {s: [tuple(w) for w in ws] for s, ws in result.get("words", {}).items()}
        try:
            answer, trace = navigator.ask(query, parts, names, letters, totals, titles, key, words)
        except Exception as exc:
            return html.P(f"The navigator could not run ({type(exc).__name__}).",
                          style={"color": "#b00020"})
        how = (f"AI navigator, {trace['model_calls']} step(s), "
               f"{trace['tokens_in'] + trace['tokens_out']:,} tokens, ${trace['cost_usd']:.4f}")
        if answer is None:
            return html.P("The navigator couldn't find that place; try a letter or bar number.",
                          style={"color": "#b00020"})
    movement, measure, explanation = answer
    if not 1 <= measure <= totals.get(movement, 0):
        return html.P(f"Movement {numerals[movement]} has measures 1-{totals.get(movement)}.",
                      style={"color": "#b00020"})
    tag = uuid.uuid4().hex[:6]
    cards = []
    for s in (chosen or sorted(names, key=lambda s: names[s])):
        out = navigator.bar_image(work / "parts" / s / "part.pdf", parts[s], movement, measure,
                                  work / f"nav_{s}_{tag}.png")
        if out:
            cards.append(html.Div([html.Div(names[s], style={"fontWeight": 600, "fontSize": "13px"}),
                                   html.Img(src=f"/jobs/{job_id}/findings/{out.name}",
                                            style={"maxWidth": "100%", "border": "1px solid #eee"})],
                                  style={"marginBottom": "10px"}))
    title = titles.get(movement, f"Movement {numerals[movement]}").rstrip(".")
    return [html.P([html.B(f"{title}, m.{measure}"), f" \u2014 {explanation} ({how})"]),
            html.Div(cards, style={"maxHeight": "560px", "overflowY": "auto"})]


@app.callback(Output("set-download", "data"), Input("set-download-button", "n_clicks"),
              State("set-job-id", "data"), prevent_initial_call=True)
def download_set(_n, job_id):
    path = JOBS / (job_id or "") / "numbered_parts.zip"
    if not job_id or not path.exists():
        return no_update
    return dcc.send_file(str(path), filename="numbered_parts.zip")


@app.callback(Output("download", "data"), Input("download-button", "n_clicks"),
              State("job-id", "data"), prevent_initial_call=True)
def download(_n, job_id):
    path = JOBS / (job_id or "") / "numbered.pdf"
    if not job_id or not path.exists():
        return no_update
    return dcc.send_file(str(path), filename="numbered_part.pdf")


JOBS.mkdir(exist_ok=True)

if __name__ == "__main__":
    app.run(debug=False, port=8050)
