"""Map who plays where across the evaluated parts, with entrances after long rests.

Writes evaluation/results/cue_map.html. Run evaluate.py first.
"""

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from measure_numbers import cues  # noqa: E402

PARTS = {"vln1": "Violin I", "vln2": "Violin II", "vla": "Viola", "vc": "Cello", "cb": "Bass"}
STYLE = """
.viz-root { --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --plays:#2a78d6; --rests:#e6e5e1; --mark:#0b0b0b; --grid:#d4d3cf;
  font:14px/1.4 system-ui, sans-serif; color:var(--text-primary);
  background:var(--surface-1); padding:24px; }
@media (prefers-color-scheme: dark) { .viz-root { --surface-1:#1a1a19;
  --text-primary:#f0efec; --text-secondary:#c3c2b7; --plays:#3987e5;
  --rests:#2e2e2c; --mark:#f0efec; --grid:#3a3a37; } }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.sub { color:var(--text-secondary); margin:0 0 8px; }
.legend { display:flex; gap:18px; color:var(--text-secondary); margin:8px 0 0; }
.key { display:inline-block; width:12px; height:12px; border-radius:2px;
  vertical-align:-2px; margin-right:6px; }
.row { display:flex; align-items:center; height:22px; }
.name { width:80px; flex:none; color:var(--text-secondary); }
.bars { position:relative; display:flex; flex:1; min-width:0; }
.m { flex:1; height:14px; background:var(--rests); }
.m.p { background:var(--plays); }
.m:hover { outline:2px solid var(--mark); }
.e { position:absolute; top:-6px; width:0; height:0; margin-left:-5px; border-left:5px solid transparent;
  border-right:5px solid transparent; border-top:7px solid var(--mark); }
.axis { position:relative; height:16px; margin-left:80px; color:var(--text-secondary);
  font-size:11px; }
.axis span { position:absolute; transform:translateX(-50%); }
table { border-collapse:collapse; margin-top:8px; }
td, th { text-align:left; padding:3px 14px 3px 0; border-bottom:1px solid var(--grid); }
th { color:var(--text-secondary); font-weight:500; }
#tip { position:fixed; pointer-events:none; background:var(--text-primary);
  color:var(--surface-1); padding:3px 8px; border-radius:4px; font-size:12px;
  display:none; }
"""

SCRIPT = """
const tip = document.getElementById('tip');
document.addEventListener('mousemove', ev => {
  const t = ev.target.dataset && ev.target.dataset.tip;
  tip.style.display = t ? 'block' : 'none';
  if (t) { tip.textContent = t; tip.style.left = ev.clientX + 12 + 'px';
           tip.style.top = ev.clientY + 12 + 'px'; }
});
"""


def main():
    parts = {}
    for key, name in PARTS.items():
        path = ROOT / "evaluation" / "results" / f"karelia_{key}" / "measure_count.json"
        parts[name] = json.loads(path.read_text())
    first = next(iter(parts.values()))
    out = [f"<meta charset='utf-8'><style>{STYLE}</style><div class='viz-root'>",
           "<h1>Who plays where: Karelia Suite strings</h1>",
           "<p class='sub'>Each column is one measure, lined up by measure number. "
           "▼ marks an entrance after at least "
           f"{cues.MIN_REST} bars of rest.</p>",
           "<div class='legend'><span><i class='key' style='background:var(--plays)'></i>"
           "plays</span><span><i class='key' style='background:var(--rests)'></i>rests"
           "</span><span>▼ entrance</span></div>"]
    rows = []
    for index, mv in enumerate(first["movements"]):
        out.append(f"<h2>{html.escape(mv['title'])} ({mv['measures']} measures)</h2>")
        for name, data in parts.items():
            plays = cues.playing(data)[index]
            cells = "".join(
                f"<div class='m{' p' if plays.get(n) else ''}' "
                f"data-tip='{name} m.{n}: {'plays' if plays.get(n) else 'rests'}'></div>"
                for n in range(1, mv["measures"] + 1))
            marks = ""
            for e in (e for e in cues.entrances(data) if e["movement"] == index):
                left = (e["measure"] - 0.5) / mv["measures"] * 100
                marks += (f"<div class='e' style='left:{left:.2f}%' data-tip='{name} enters "
                          f"m.{e['measure']} after {e['rest']} bars rest'></div>")
                others = cues.lead_in({k: v for k, v in parts.items() if k != name},
                                      index, e["measure"])
                rows.append((mv["title"], name, e["measure"], e["rest"],
                             ", ".join(others) or "no strings"))
            out.append(f"<div class='row'><span class='name'>{name}</span>"
                       f"<div class='bars'>{cells}{marks}</div></div>")
        ticks = "".join(f"<span style='left:{(n - 0.5) / mv['measures'] * 100:.2f}%'>{n}</span>"
                        for n in [1] + list(range(10, mv["measures"] + 1, 10)))
        out.append(f"<div class='axis'>{ticks}</div>")
    out.append("<h2>Entrances after long rests</h2><table><tr><th>Movement</th>"
               "<th>Part</th><th>Enters at</th><th>Bars of rest</th>"
               f"<th>Other strings playing in the {cues.LEAD_IN} bars before</th></tr>")
    out += [f"<tr><td>{html.escape(t)}</td><td>{p}</td><td>m.{m}</td><td>{r}</td>"
            f"<td>{w}</td></tr>" for t, p, m, r, w in rows]
    out.append(f"</table><div id='tip'></div></div><script>{SCRIPT}</script>")
    target = ROOT / "evaluation" / "results" / "cue_map.html"
    target.write_text("\n".join(out) + "\n")
    print(target)


if __name__ == "__main__":
    main()
