"""Run Audiveris (optical music recognition) on a scanned part."""

import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

# Where to find Audiveris: AUDIVERIS env var, else a copy in tools/, else the
# usual install locations (the Docker image installs it under /opt)
CANDIDATES = [
    str(Path(__file__).resolve().parent.parent / "tools/Audiveris.app/Contents/MacOS/Audiveris"),
    "/Applications/Audiveris.app/Contents/MacOS/Audiveris",
    "/opt/audiveris/bin/Audiveris",
    "audiveris",
]
# Audiveris reads pages one after another; this many copies read a part's
# pages side by side (each copy is itself multi-threaded)
WORKERS = int(os.getenv("AUDIVERIS_WORKERS", "3"))


def audiveris_path():
    """Path to the Audiveris program, or None if it isn't installed."""
    for candidate in [os.getenv("AUDIVERIS")] + CANDIDATES:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    return None


def _recognise(program, pdf, out_dir, sheets=None):
    """One Audiveris run over the given pages (all if None); its .omr path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [program, "-batch", "-transcribe", "-save", "-output", str(out_dir)]
    if sheets:
        command += ["-sheets", f"{sheets[0]}-{sheets[-1]}"]
    with open(out_dir / "audiveris.log", "w") as log:
        subprocess.run(command + ["--", str(pdf)], stdout=log, stderr=log)
    return out_dir / f"{pdf.stem}.omr"


def _read_pages(omr):
    """The pages an .omr holds music for: {page: {zip entry: bytes}} and its
    book.xml. A page Audiveris gave up on (it times out when the machine is
    busy) is left out."""
    pages, book = {}, ""
    if not omr.exists():
        return pages, book
    with zipfile.ZipFile(omr) as z:
        book = z.read("book.xml").decode()
        for name in z.namelist():
            match = re.match(r"sheet#(\d+)/", name)
            if match:
                pages.setdefault(int(match.group(1)), {})[name] = z.read(name)
    for page, entries in list(pages.items()):
        if b"<system" not in entries.get(f"sheet#{page}/sheet#{page}.xml", b""):
            del pages[page]
    return pages, book


def _book_sheet(book, page):
    match = re.search(rf'<sheet number="{page}"[\s>].*?</sheet>', book, re.S)
    return match.group(0) if match else None


def run_audiveris(pdf_path, out_dir, workers=None):
    """Recognise every page; return the path of the .omr project Audiveris saves.

    Takes about 25 seconds a page, so the pages are split among `workers`
    copies of Audiveris running side by side and the results merged into one
    project. A page that comes back without music is read again on its own,
    so a busy machine can't silently drop a page. The project file holds
    staves, barlines and notes per page, which counting.py reads.
    """
    program = audiveris_path()
    if program is None:
        raise RuntimeError("Audiveris is not installed (set AUDIVERIS to its path)")
    pdf = Path(pdf_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    omr = out_dir / f"{pdf.stem}.omr"
    count = pymupdf.open(pdf).page_count
    workers = min(workers or WORKERS, count)
    if workers <= 1:
        _recognise(program, pdf, out_dir)
        if not omr.exists():
            raise RuntimeError(f"Audiveris produced no result; see {out_dir / 'audiveris.log'}")
        return omr

    # contiguous runs of pages, as even as possible
    size, extra = divmod(count, workers)
    chunks, start = [], 1
    for k in range(workers):
        end = start + size + (k < extra) - 1
        chunks.append(list(range(start, end + 1)))
        start = end + 1
    parts = out_dir / "pages"
    with ThreadPoolExecutor(workers) as pool:
        omrs = list(pool.map(lambda k: _recognise(program, pdf, parts / str(k), chunks[k]),
                             range(workers)))
    pages, books = {}, {}
    for chunk, path in zip(chunks, omrs):
        found, book = _read_pages(path)
        for page in chunk:
            if page in found:
                pages[page], books[page] = found[page], book
    for page in range(1, count + 1):
        if page not in pages:  # read again alone
            found, book = _read_pages(_recognise(program, pdf, parts / f"again{page}", [page]))
            if page in found:
                pages[page], books[page] = found[page], book
    if not pages:
        raise RuntimeError(f"Audiveris produced no result; see {parts}")

    # one project: the first run's book, each page's entry from the run that read it
    book = books[min(pages)]
    for page in pages:
        mine = _book_sheet(books[page], page)
        if mine and _book_sheet(book, page):
            book = book.replace(_book_sheet(book, page), mine)
    with zipfile.ZipFile(omr, "w", zipfile.ZIP_DEFLATED) as z:
        for page in sorted(pages):
            for name, data in pages[page].items():
                z.writestr(name, data)
        z.writestr("book.xml", book)
    with open(out_dir / "audiveris.log", "w") as log:
        for k, chunk in enumerate(chunks):
            log.write(f"pages {chunk[0]}-{chunk[-1]}: {parts / str(k) / 'audiveris.log'}\n")
        missing = sorted(set(range(1, count + 1)) - set(pages))
        if missing:
            log.write(f"no music read on pages {missing}\n")
    return omr
