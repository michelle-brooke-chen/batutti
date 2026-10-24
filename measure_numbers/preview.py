"""Page images for review: each staff labelled with the measure numbers on it."""

from pathlib import Path

import cv2
import numpy as np
import pymupdf

RED, ORANGE, WHITE = (0, 0, 200), (0, 120, 240), (255, 255, 255)


def render_pages(pdf_path, data, out_dir, scale=0.5):
    """Write page_<n>.png for every page with staves; return the paths in order.

    Each staff gets "m.<first>-<last>"; staves with a flag get a "?" in orange,
    and staves where the 1st/2nd-ending rule applied are marked too.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = {(b["page"], b["staff"]) for b in data["bars"] if b.get("flags")}
    endings = {(n["page"], n["staff"]) for n in data.get("notes", []) if "ending" in n["note"]}
    doc = pymupdf.open(pdf_path)
    paths = []
    for page in sorted({s["page"] for s in data["staves"]}):
        pix = doc[page - 1].get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for st in (s for s in data["staves"] if s["page"] == page):
            where = (st["page"], st["staff"])
            label = f"{st['staff']}: m.{st['first_measure']}-{st['last_measure']}"
            if where in endings:
                label += " (endings)"
            color = ORANGE if where in flagged else RED
            if where in flagged:
                label += " ?"
            x, y = max(5, int(st["left"]) - 300), int(st["top"] - 1.2 * st["space"])
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 8)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        path = out_dir / f"page_{page}.png"
        cv2.imwrite(str(path), cv2.resize(img, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_AREA))
        paths.append(path)
    return paths
