"""Run Audiveris (optical music recognition) on a scanned part."""

import os
import shutil
import subprocess
from pathlib import Path

# Where to find Audiveris: AUDIVERIS env var, else a copy in tools/, else the
# usual install locations (the Docker image installs it under /opt)
CANDIDATES = [
    str(Path(__file__).resolve().parent.parent / "tools/Audiveris.app/Contents/MacOS/Audiveris"),
    "/Applications/Audiveris.app/Contents/MacOS/Audiveris",
    "/opt/audiveris/bin/Audiveris",
    "audiveris",
]


def audiveris_path():
    """Path to the Audiveris program, or None if it isn't installed."""
    for candidate in [os.getenv("AUDIVERIS")] + CANDIDATES:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    return None


def run_audiveris(pdf_path, out_dir):
    """Recognise every page; return the path of the .omr project Audiveris saves.

    Takes about 25 seconds a page. The project file holds staves, barlines and
    notes per page, which counting.py reads.
    """
    program = audiveris_path()
    if program is None:
        raise RuntimeError("Audiveris is not installed (set AUDIVERIS to its path)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    omr = out_dir / f"{Path(pdf_path).stem}.omr"
    command = [program, "-batch", "-transcribe", "-save", "-output", str(out_dir),
               "--", str(Path(pdf_path).resolve())]
    with open(out_dir / "audiveris.log", "w") as log:
        subprocess.run(command, stdout=log, stderr=log)
    if not omr.exists():
        raise RuntimeError(f"Audiveris produced no result; see {out_dir / 'audiveris.log'}")
    return omr
