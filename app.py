"""batutti: a web app that stamps measure numbers onto scanned parts.

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


def start_set_job(files=(), example=False, totals_text="", use_agent=True, add_cues=False):
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

app = Dash(__name__, title="batutti", external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap"])
server = app.server
app.index_string = app.index_string.replace("<head>", """<head>
        <script>
          (function () {
            // dark unless the toggle saved light
            document.documentElement.dataset.theme =
              localStorage.getItem("batutti-theme") || "dark";
          })();
        </script>""", 1)


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


SHOWN, HIDDEN = {"display": "block"}, {"display": "none"}

agent_note = ("" if portkey_key() else
              "No PORTKEY_API_KEY found, so parts are numbered without the agent.")
audiveris_note = ("" if audiveris_path() else
                  " Audiveris isn't installed here, so only the examples can run.")
setup_note = (agent_note + audiveris_note).strip()

VERDICTS = {"misprint": "Misprint in the part",
            "reading_error": "Possible misreading by the app",
            "unclear": "Needs a look"}

# the two whole-set pages: id prefix, whether cues are added, and their wording
SET_PAGES = {
    "set": {"cues": False, "agent": "Let the agent look into disagreements between parts",
            "button": "Number and check the set", "zip": "numbered_parts.zip",
            "steps": [("counting", "Number every part"), ("reviewing", "Look into disagreements"),
                      ("stamping", "Pack the numbered parts"), ("done", "Done")],
            "finished": "All done! Every part is numbered and ready to download."},
    "cue": {"cues": True, "agent": "Let the agent choose which instrument to cue",
            "button": "Number and add cues", "zip": "cued_parts.zip",
            "steps": [("counting", "Number every part"), ("reviewing", "Look into disagreements"),
                      ("cueing", "Choose and print cues"), ("stamping", "Pack the parts"),
                      ("done", "Done")],
            "finished": "All done! Every part is numbered and cued."},
}


def cat(size):
    return html.Img(src=app.get_asset_url("cat.svg"), className=f"cat cat-{size}", alt="")


def prose(text):
    """`text` with the app's name in italics, as it is set in sentences."""
    return [html.Em(piece) if piece == "batutti" else piece
            for piece in re.split(r"(batutti)", text) if piece]


def hero(title, text):
    """A page's title and intro."""
    return html.Div([html.H2(title, className="page-title"),
                     html.P(prose(text), className="intro")])


def step_title(number, text):
    return html.H3([html.Span(str(number), className="step-num"), text], className="step-title")


def drop_zone(id_, label_id, text, multiple):
    return dcc.Upload(id=id_, accept=".pdf", multiple=multiple,
                      max_size=MAX_PAGES_MB * 1024 * 1024, className="drop",
                      children=html.Span(text, id=label_id, className="drop-text"))


def toggle(id_, label, on):
    return dcc.Checklist(id=id_, options=[{"label": label, "value": "yes"}],
                         value=["yes"] if on else [], className="toggle")


def set_sections(p, lead, results):
    """Upload, progress and results for a whole-set page; `p` prefixes the ids."""
    page = SET_PAGES[p]
    return [
        html.Div(className="section", children=[
            step_title(1, "All the parts of a piece"),
            html.P(prose(lead), className="lead"),
            drop_zone(f"{p}-upload", f"{p}-upload-label",
                      "Drop the parts' PDFs here, or click to choose", True),
            html.Label(["Measures per movement from the score ",
                        html.Span("(optional; otherwise the parts are compared with each other)")],
                       className="field-label"),
            dcc.Input(id=f"{p}-totals", placeholder="e.g. 157, 162, 168", debounce=True,
                      className="text-input totals"),
            html.Div(className="options", children=[
                toggle(f"{p}-use-agent", page["agent"], portkey_key())]),
            html.P(setup_note, className="note") if setup_note else None,
            html.Div(className="buttons", children=[
                html.Button(page["button"], id=f"{p}-start", className="btn btn-primary"),
                html.Button("Try an example: Karelia Suite, all 29 parts", id=f"{p}-example",
                            className="btn btn-soft")]),
            html.Div(id=f"{p}-error", className="error"),
        ]),
        html.Div(id=f"{p}-progress-card", className="section", style=HIDDEN, children=[
            step_title(2, "Working on it"),
            html.Div(className="working", children=[html.Div(id=f"{p}-progress"), cat("small")]),
        ]),
        html.Div(id=f"{p}-results-card", className="section", style=HIDDEN, children=[
            step_title(3, "What the parts say"),
            html.Div(id=f"{p}-findings"),
            *results,
            html.Div(className="buttons", children=[
                html.Button("Download all parts (.zip)", id=f"{p}-download-button",
                            className="btn btn-primary")]),
            dcc.Download(id=f"{p}-download"),
        ]),
        dcc.Store(id=f"{p}-job-id"),
        dcc.Interval(id=f"{p}-tick", interval=1000, disabled=True),
    ]


def set_tab():
    return set_sections(
        "set", "Drop in every part at once. batutti numbers them all, then checks them against "
               "each other: movement lengths and rehearsal letters must agree, which catches its "
               "own slips and misprints in the parts.",
        [html.Div("Find a place in the music", className="section-title"),
         html.P("Type what the conductor says, like “two before D in the Ballade”, "
                "“m.98 Alla marcia” or “the cor anglais solo”, and every part "
                "shows that bar.", className="lead"),
         html.Div(className="nav-row", children=[
             dcc.Input(id="nav-query", placeholder="two before D in the Ballade",
                       debounce=False, n_submit=0, className="text-input"),
             html.Button("Find", id="nav-go", className="btn btn-primary")]),
         dcc.Dropdown(id="nav-parts", multi=True, placeholder="All parts", className="nav-parts"),
         dcc.Loading(html.Div(id="nav-result"), color="#2f5fb3", type="dot"),
         html.Div("Every part", className="section-title"),
         html.Div(id="set-table")])


def cue_page():
    return html.Div([
        hero("cues", "Before each entrance after a long rest, batutti prints a few small bars "
                     "of a line you'll hear from another part, so you know when to come in. It "
                     "needs every part of the piece, since the cues come from the other parts."),
        *set_sections(
            "cue", "Drop in every part at once. Each part comes back numbered, with cues before "
                   "entrances after 4 or more bars of rest (none after a fermata or G.P., where "
                   "the conductor restarts the music).",
            [html.Div("Cues", className="section-title"),
             html.Div(id="cue-list")]),
    ])


def part_tab():
    return [
        html.Div(className="section", children=[
            step_title(1, "Your part"),
            html.P(prose("One part on its own: batutti numbers it and asks you about "
                         "anything it isn't sure of."), className="lead"),
            drop_zone("upload", "upload-label", "Drop a PDF here, or click to choose", False),
            html.Label(["Measures per movement from the score ",
                        html.Span("(optional, helps catch mistakes)")], className="field-label"),
            dcc.Input(id="totals", placeholder="e.g. 157, 162, 168", debounce=True,
                      className="text-input totals"),
            html.Div(className="options", children=[
                toggle("use-agent", "Let the agent review flagged bars", portkey_key())]),
            html.P(setup_note, className="note") if setup_note else None,
            html.Div(className="buttons", children=[
                html.Button("Number this part", id="start", className="btn btn-primary"),
                html.Button("Try an example: Karelia Suite, Violin I", id="example",
                            className="btn btn-soft")]),
            html.Div(id="error", className="error"),
        ]),
        html.Div(id="progress-card", className="section", style=HIDDEN, children=[
            step_title(2, "Working on it"),
            html.Div(className="working", children=[html.Div(id="progress"), cat("small")]),
        ]),
        html.Div(id="results-card", className="section", style=HIDDEN, children=[
            step_title(3, "Check and download"),
            html.Div(id="summary"),
            html.Div(id="questions"),
            html.Div("Pages", className="section-title"),
            html.P(prose("Each staff shows its measure numbers. Orange labels with a ? mark "
                         "staves batutti was unsure about."), className="lead"),
            html.Div(id="pages", className="page-viewer"),
            html.Div("Corrections", className="section-title"),
            html.P("If a staff's count is wrong, change it below and apply. Count the bars on the "
                   "staff; bars of a 2nd ending don't count (they reuse the 1st ending's numbers).",
                   className="lead"),
            dash_table.DataTable(
                id="staff-table", editable=False, page_size=15,
                columns=[{"name": "Movement", "id": "movement"},
                         {"name": "Page", "id": "page"}, {"name": "Staff", "id": "staff"},
                         {"name": "Numbers", "id": "numbers"},
                         {"name": "Measures (edit)", "id": "measures", "editable": True,
                          "type": "numeric"},
                         {"name": "Why it was flagged", "id": "why"}],
                style_as_list_view=True,
                style_header={"fontWeight": 800, "color": "var(--ink)",
                              "backgroundColor": "var(--bg)",
                              "borderBottom": "2px solid var(--line)"},
                style_cell={"fontFamily": "Nunito, -apple-system, Helvetica, Arial, sans-serif",
                            "fontSize": "14px", "textAlign": "left", "color": "var(--ink)",
                            "backgroundColor": "var(--bg)",
                            "padding": "8px", "whiteSpace": "normal", "height": "auto",
                            "borderBottom": "1px solid var(--line)"},
                style_data_conditional=[{"if": {"filter_query": "{why} != ''"},
                                         "backgroundColor": "var(--peach)"},
                                        {"if": {"column_id": "measures"},
                                         "backgroundColor": "var(--accent-soft)",
                                         "fontWeight": 700}],
            ),
            html.Div(className="buttons", children=[
                html.Button("Apply corrections", id="apply", className="btn btn-soft"),
                html.Button("Download numbered PDF", id="download-button",
                            className="btn btn-primary")]),
            dcc.Download(id="download"),
        ]),
        dcc.Store(id="job-id"),
        dcc.Interval(id="tick", interval=1000, disabled=True),
    ]


def numbers_page():
    return html.Div([
        hero("measure numbers",
             "Upload scanned orchestral parts. batutti numbers every measure on your own pages, "
             "checks the parts against each other, and asks you only about what it isn't sure of."),
        dcc.Tabs(id="tabs", value="one", className="tabs", children=[
            dcc.Tab(label="One part", value="one", className="tab",
                    selected_className="tab--selected", children=part_tab()),
            dcc.Tab(label="Whole set of parts", value="set", className="tab",
                    selected_className="tab--selected", children=set_tab()),
        ]),
    ])


app.layout = html.Div(className="page", children=[
    dcc.Location(id="url"),
    html.Div(className="header", children=[
        dcc.Link(["batutti", cat("logo")], href="/", className="wordmark"),
        html.Div(className="header-right", children=[
            dcc.Link("measure numbers", href="/", id="link-numbers", className="nav-link"),
            dcc.Link("cues", href="/cues", id="link-cues", className="nav-link"),
            html.Button("☾", id="theme-toggle", className="theme-toggle",
                        title="Switch light / dark")]),
    ]),
    # both pages stay in the layout (so a running job keeps its progress); the
    # address decides which one shows
    html.Div(id="page-numbers", children=numbers_page()),
    html.Div(id="page-cues", style=HIDDEN, children=cue_page()),
    html.Footer("batutti · count less, play more", className="footer"),
])


@app.callback(Output("page-numbers", "style"), Output("page-cues", "style"),
              Output("link-numbers", "className"), Output("link-cues", "className"),
              Input("url", "pathname"))
def show_page(path):
    cues = (path or "").rstrip("/") == "/cues"
    return (HIDDEN if cues else SHOWN, SHOWN if cues else HIDDEN,
            "nav-link" + ("" if cues else " active"), "nav-link" + (" active" if cues else ""))


app.clientside_callback(
    """function (clicks) {
        var root = document.documentElement;
        var dark = root.dataset.theme ? root.dataset.theme === "dark"
                 : window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (clicks) {
            dark = !dark;
            root.dataset.theme = dark ? "dark" : "light";
            localStorage.setItem("batutti-theme", root.dataset.theme);
        }
        return dark ? "\u2600" : "\u263e";
    }""",
    Output("theme-toggle", "children"), Input("theme-toggle", "n_clicks"))


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


def step_list(steps, step, detail, finished="All done!"):
    """The job's steps as rows: done ones ticked, the current one hopping."""
    names = [s for s, _ in steps]
    current = names.index(step) if step in names else -1
    rows = []
    for k, (name, label) in enumerate(steps[:-1]):
        if k < current or step == "done":
            state, mark = "done", "\u2713"
        elif k == current:
            state, mark = "now", "\u266a"
        else:
            state, mark = "", str(k + 1)
        rows.append(html.Div(className=f"stepline {state}", children=[
            html.Span(mark, className="dot"), html.Span(label),
            html.Span(f"\u00b7 {detail}", className="detail") if state == "now" and detail else None]))
    if step == "done":
        rows.append(html.Div([finished, html.Span([html.Span("\u266a"), html.Span("\u266b"),
                                                   html.Span("\u266a")], className="confetti")],
                             className="all-done"))
    return html.Div(rows, className="steps")


@app.callback(Output("progress", "children"), Output("progress-card", "style"),
              Output("results-card", "style"), Output("tick", "disabled", allow_duplicate=True),
              Output("summary", "children"), Output("questions", "children"),
              Output("pages", "children"), Output("staff-table", "data"),
              Input("tick", "n_intervals"), State("job-id", "data"),
              prevent_initial_call=True)
def poll(_n, job_id):
    with jobs_lock:
        status = dict(jobs.get(job_id, {}))
    if status.get("step") == "error":
        return (html.Div(f"Something went wrong: {status['error']}", className="error"),
                SHOWN, HIDDEN, True, no_update, no_update, no_update, no_update)
    if status.get("step") != "done":
        return (step_list(STEPS, status.get("step"), status.get("detail")), SHOWN, HIDDEN, False,
                no_update, no_update, no_update, no_update)
    return (step_list(STEPS, "done", "", "All done! Your part is numbered."), SHOWN, SHOWN, True,
            *results_view(job_id))


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
            ok = html.Span(f"{staves} to check" if unsure else "")
        elif ref != m["measures"]:
            ok = html.Span(f"\u2717 score says {ref}", className="off")
        elif unsure:
            ok = html.Span(f"total matches the score; check the {staves}")
        else:
            ok = html.Span("\u2713 matches the score", className="ok")
        rows.append(html.Tr([html.Td(html.B(m["title"].rstrip("."))),
                             html.Td(f"{m['measures']} measures"), html.Td(ok)]))
    summary = html.Table(rows, className="summary-table")

    trace_path = work / "agent_trace.json"
    asked = [html.Li(q["question"]) for q in questions]
    agent = []
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        agent.append(html.P(f"Agent review: {trace.get('summary') or 'done.'}",
                            className="ai-review"))
    if asked:
        agent.append(html.Div([html.B("Please check:"), html.Ul(asked)], className="questions"))

    pages = [html.Img(src=f"/jobs/{job_id}/pages/{p.name}?v={uuid.uuid4().hex[:6]}")
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


def findings_view(job_id, result, names):
    """Movement lengths, letters, agent use and a card per disagreement."""
    cards = []
    for v in result["verdicts"]:
        kind = v["verdict"] if v["verdict"] in VERDICTS else "unclear"
        cards.append(html.Div(className=f"finding {kind}", children=[
            html.Div(f"{VERDICTS[kind]}: {names.get(v['part'], v['part'])}",
                     className="finding-title"),
            html.P(v["note"]),
            html.Img(src=f"/jobs/{job_id}/findings/{v['image']}") if v.get("image") else None,
            html.Details([html.Summary("Evidence"), html.P(v["evidence"])])
            if v.get("evidence") else None,
        ]))
    source = ("the score" if result["totals_from"] == "score" else "most parts")
    if not cards:
        cards = [html.Div("✓ Every part agrees with the others: movement lengths and "
                          "rehearsal letters line up.", className="happy")]
    intro = [html.Div(f"Movement lengths ({source})", className="small"),
             html.Div([html.Span(f"{mv} = {n} bars", className="chip")
                       for mv, n in result["totals"].items()], className="chips"),
             html.Div("Rehearsal letters found across the parts", className="small"),
             html.Div([html.Span(f"{k} = m.{m}", className="chip letter")
                       for k, m in result["letters"].items()], className="chips")]
    traces = [t for t in (result.get("trace"), result.get("cue_trace")) if t]
    if traces:
        calls = sum(t["model_calls"] for t in traces)
        tokens = sum(t["tokens_in"] + t["tokens_out"] for t in traces)
        cached = sum(t.get("tokens_cached", 0) for t in traces)
        cost = sum(t.get("cost_usd", 0) for t in traces)
        intro.append(html.Div(f"Agent use this run: {calls} model calls, {tokens:,} tokens "
                              f"({cached:,} cached), about ${cost:.4f}", className="ai-use"))
    return intro + cards


def parts_table(result):
    rows = [html.Tr([html.Th("Part"), html.Th("Measures per movement"), html.Th("Flagged staves")])]
    for p in result["parts"]:
        cells = []
        for mv, n in p["movements"].items():
            ok = n == result["totals"].get(mv)
            cells.append(html.Span(f"{mv}: {n}{'' if ok else ' ✗'}",
                                   className="mv" if ok else "mv off"))
        rows.append(html.Tr([html.Td(html.B(p["name"])), html.Td(cells),
                             html.Td(p["flagged_staves"] or "")]))
    return html.Table(rows, className="parts-table")


def cue_list(result, names):
    cue_rows = result.get("cues") or []
    if not cue_rows:
        return [html.P("No entrance needed a cue.", className="lead")]
    on_page = sum(c["where"] == "on the page" for c in cue_rows)
    missed = len(cue_rows) - on_page
    by_part = {}
    for c in cue_rows:
        by_part.setdefault(names.get(c["part"], c["part"]), []).append(c)
    boxes = []
    for part in sorted(by_part):
        items = [html.Li(f"Movement {['', 'I', 'II', 'III', 'IV', 'V'][c['movement']]}, "
                         f"m.{c['measure']} (after {c['rest']} bars rest): {c['cue']}"
                         + (f", {c['bars']} bars" if c.get("bars") else "")
                         + (" (no room on the page)" if c["where"] == "no room" else ""))
                 for c in by_part[part]]
        boxes.append(html.Details([html.Summary([f"{part} ",
                                                 html.Span(f"· {len(items)} "
                                                           + ("cue" if len(items) == 1 else "cues"))]),
                                   html.Ul(items)], className="cue-part"))
    return [html.P(f"{on_page} cues printed just before entrances after long rests"
                   + (f"; {missed} more had no clear space, so they were left out." if missed
                      else "."), className="lead"),
            html.Div(boxes, className="cue-list")]


def register_set_page(p):
    """Callbacks for one whole-set page (ids prefixed with `p`)."""
    page = SET_PAGES[p]

    @app.callback(Output(f"{p}-upload-label", "children"), Input(f"{p}-upload", "filename"))
    def show_files(names):
        if not names:
            return "Drop the parts' PDFs here, or click to choose"
        return (f"Selected {len(names)} parts: " + ", ".join(names[:6])
                + ("…" if len(names) > 6 else ""))

    @app.callback(Output(f"{p}-job-id", "data"), Output(f"{p}-tick", "disabled"),
                  Output(f"{p}-error", "children"),
                  Input(f"{p}-start", "n_clicks"), Input(f"{p}-example", "n_clicks"),
                  State(f"{p}-upload", "contents"), State(f"{p}-upload", "filename"),
                  State(f"{p}-totals", "value"), State(f"{p}-use-agent", "value"),
                  prevent_initial_call=True)
    def start(_start, _example, contents, filenames, totals, use_agent):
        from dash import ctx
        agent = bool(use_agent)
        if ctx.triggered_id == f"{p}-example":
            return (start_set_job(example=True, totals_text=totals, use_agent=agent,
                                  add_cues=page["cues"]), False, "")
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
            return no_update, no_update, prose("batutti is busy; please try again in a few minutes.")
        return (start_set_job(files=files, totals_text=totals, use_agent=agent,
                              add_cues=page["cues"]), False, "")

    results = ([Output("set-table", "children"), Output("nav-parts", "options")] if p == "set"
               else [Output("cue-list", "children")])

    @app.callback(Output(f"{p}-progress", "children"), Output(f"{p}-progress-card", "style"),
                  Output(f"{p}-results-card", "style"),
                  Output(f"{p}-tick", "disabled", allow_duplicate=True),
                  Output(f"{p}-findings", "children"), *results,
                  Input(f"{p}-tick", "n_intervals"), State(f"{p}-job-id", "data"),
                  prevent_initial_call=True)
    def poll(_n, job_id):
        with jobs_lock:
            status = dict(jobs.get(job_id, {}))
        step = status.get("step")
        waiting = [no_update] * (1 + len(results))
        if step == "error":
            return (html.Div(f"Something went wrong: {status['error']}", className="error"),
                    SHOWN, HIDDEN, True, *waiting)
        rows = step_list(page["steps"], step, status.get("detail"), page["finished"])
        if step != "done":
            return (rows, SHOWN, HIDDEN, False, *waiting)
        result = json.loads((JOBS / job_id / "set_result.json").read_text())
        names = {q["id"]: q["name"] for q in result["parts"]}
        found = findings_view(job_id, result, names)
        if p == "set":
            extra = [parts_table(result),
                     [{"label": q["name"], "value": q["id"]} for q in result["parts"]]]
        else:
            extra = [cue_list(result, names)]
        return (rows, SHOWN, SHOWN, True, found, *extra)

    @app.callback(Output(f"{p}-download", "data"), Input(f"{p}-download-button", "n_clicks"),
                  State(f"{p}-job-id", "data"), prevent_initial_call=True)
    def download(_n, job_id):
        path = JOBS / (job_id or "") / "numbered_parts.zip"
        if not job_id or not path.exists():
            return no_update
        return dcc.send_file(str(path), filename=page["zip"])


for _page in SET_PAGES:
    register_set_page(_page)


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
                          "(the agent needs a PORTKEY_API_KEY).", className="error")
        words = {s: [tuple(w) for w in ws] for s, ws in result.get("words", {}).items()}
        try:
            answer, trace = navigator.ask(query, parts, names, letters, totals, titles, key, words)
        except Exception as exc:
            return html.P(f"The navigator could not run ({type(exc).__name__}).",
                          className="error")
        how = (f"found by the agent, {trace['model_calls']} step(s), "
               f"{trace['tokens_in'] + trace['tokens_out']:,} tokens, ${trace['cost_usd']:.4f}")
        if answer is None:
            return html.P("The navigator couldn't find that place; try a letter or bar number.",
                          className="error")
    movement, measure, explanation = answer
    if not 1 <= measure <= totals.get(movement, 0):
        return html.P(f"Movement {numerals[movement]} has measures 1-{totals.get(movement)}.",
                      className="error")
    tag = uuid.uuid4().hex[:6]
    cards = []
    for s in (chosen or sorted(names, key=lambda s: names[s])):
        out = navigator.bar_image(work / "parts" / s / "part.pdf", parts[s], movement, measure,
                                  work / f"nav_{s}_{tag}.png")
        if out:
            cards.append(html.Div([html.Div(names[s], className="name"),
                                   html.Img(src=f"/jobs/{job_id}/findings/{out.name}")],
                                  className="nav-card"))
    title = titles.get(movement, f"Movement {numerals[movement]}").rstrip(".")
    return [html.Div([html.B(f"{title}, m.{measure}"), f" \u2014 {explanation} ",
                      html.Span(f"({how})", className="small")], className="nav-answer"),
            html.Div(cards, className="nav-grid")]


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
