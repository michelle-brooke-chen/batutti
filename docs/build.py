"""Build report.html and slides.html in the repo folder from docs/*_src.html,
with every image embedded so each file opens on its own.

    python docs/build.py
"""

import base64
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def embed(match):
    path = DOCS / match.group(1)
    data = base64.b64encode(path.read_bytes()).decode()
    return f'src="data:{TYPES[path.suffix.lower()]};base64,{data}"'


for name in ("report", "slides"):
    html = (DOCS / f"{name}_src.html").read_text()
    html = re.sub(r'src="(img/[^"]+)"', embed, html)
    (DOCS.parent / f"{name}.html").write_text(html)
    print(f"{name}.html: {len(html) // 1024} KB")
