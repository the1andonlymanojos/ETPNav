#!/usr/bin/env python3
"""Experiment 3 page: what about a persisted map hurts the planner? Reads docs/experiments/exp3_diagnosis via exp3_data."""
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402
import exp3_data as X  # noqa: E402

data, order = X.load()
S = X.stats(data, order)
assert all(S[a]["n"] == 45 for a in X.ARMS), {a: S[a]["n"] for a in S}
N = S["none"]["n"]
pct = lambda a: 100 * S[a]["k"] / S[a]["n"]
LBL = {a: X.ARMS[a][0] for a in X.ARMS}

# ---------- chart 1: success rate per arm, Wilson interval, filled = all frontiers, outlined = old frontiers hidden ----------
rows = ["none", "w10", "w20", "w40", "full", "w20cur", "fullcur"]
W, RH, ml, mr, mt = 820, 40, 268, 70, 8
H = mt + RH * len(rows) + 30
xs = lambda v: ml + (W - ml - mr) * v / 100
parts = [f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{mt}" y2="{H - 26}" stroke="var(--grid)"/>'
         f'<text x="{xs(v):.1f}" y="{H - 8}" font-size="10.5" text-anchor="middle">{v}%</text>' for v in (0, 20, 40, 60, 80, 100)]
base_x = xs(pct("none"))
for i, a in enumerate(rows):
    st = S[a]
    y0 = mt + i * RH
    lo, hi = X.wilson(st["k"], st["n"]) if hasattr(X, "wilson") else L.wilson(st["k"], st["n"])
    col = {"none": "dim", "window": "s1", "full": "s2"}[X.ARMS[a][1]]
    hollow = X.ARMS[a][2] == "current"
    tip = (f"<b>{LBL[a]}</b><br><span class='m'>{st['k']} of {st['n']} reached · 95% interval {100 * lo:.0f}–{100 * hi:.0f}%<br>"
           + ("" if a == "none" else f"vs map wiped: {st['wins']} gained, {st['losses']} lost (sign test p = {st['p']:.3f})<br>")
           + f"planner saw {st['old']:.0f} old nodes on average</span>")
    fill = f'fill="var(--surface)" stroke="var(--{col})" stroke-width="2"' if hollow else f'fill="var(--{col})"'
    parts.append(f'<text x="{ml - 12}" y="{y0 + RH / 2 + 4:.1f}" font-size="11.5" text-anchor="end" style="fill:var(--ink)">{html.escape(LBL[a])}</text>')
    parts.append(f'<rect data-tip="{html.escape(tip, quote=True)}" x="{ml}" y="{y0 + 8}" width="{xs(pct(a)) - ml:.1f}" height="{RH - 16}" rx="3" {fill}/>')
    parts.append(f'<line x1="{xs(100 * lo):.1f}" x2="{xs(100 * hi):.1f}" y1="{y0 + RH / 2}" y2="{y0 + RH / 2}" stroke="var(--ink)" stroke-width="1.4" opacity=".7" pointer-events="none"/>')
    parts.append(f'<text x="{W - mr + 10}" y="{y0 + RH / 2 + 4:.1f}" font-size="12" style="fill:var(--ink);font-weight:600">{pct(a):.0f}%</text>')
parts.append(f'<line x1="{base_x:.1f}" x2="{base_x:.1f}" y1="{mt}" y2="{H - 26}" stroke="var(--ink)" stroke-dasharray="4 4" stroke-width="1.1" opacity=".55"/>')
chart1 = f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Success rate for each map arm">{"".join(parts)}</svg>'

# ---------- chart 2: dose-response, success vs old nodes the planner sees ----------
W2, H2, ml2, mr2, mt2, mb2 = 820, 300, 46, 130, 14, 40
xmax = 35.0
x2 = lambda v: ml2 + (W2 - ml2 - mr2) * v / xmax
y2 = lambda v: mt2 + (H2 - mt2 - mb2) * (1 - v / 100)
g2 = "".join(f'<line x1="{ml2}" x2="{W2 - mr2}" y1="{y2(v):.1f}" y2="{y2(v):.1f}" stroke="var(--grid)"/>'
              f'<text x="{ml2 - 8}" y="{y2(v) + 3.5:.1f}" font-size="10.5" text-anchor="end">{v}%</text>' for v in (0, 20, 40, 60, 80, 100))
g2 += "".join(f'<text x="{x2(v):.1f}" y="{H2 - mb2 + 17}" font-size="10.5" text-anchor="middle">{v}</text>' for v in (0, 10, 20, 30))
g2 += f'<text x="{ml2}" y="{H2 - 4}" font-size="10.5">old nodes the planner sees, on average</text>'
line_all = ["none", "w10", "w20", "w40", "full"]
pts2 = " ".join("%.1f,%.1f" % (x2(S[a]["old"]), y2(pct(a))) for a in line_all)
g2 += '<polyline points="%s" fill="none" stroke="var(--s1)" stroke-width="2.2" stroke-linejoin="round"/>' % pts2
for a in line_all:
    col = "dim" if a == "none" else ("s2" if a == "full" else "s1")
    tip = f"<b>{LBL[a]}</b><br><span class='m'>{S[a]['old']:.1f} old nodes seen · {pct(a):.0f}% reached</span>"
    dx, dy, anc = {"none": (10, -10, "start"), "w10": (0, 20, "middle"), "w20": (0, -12, "middle"), "w40": (-2, -12, "middle"), "full": (8, 20, "middle")}[a]
    g2 += (f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{x2(S[a]["old"]):.1f}" cy="{y2(pct(a)):.1f}" r="10" fill="transparent"/>'
           f'<circle cx="{x2(S[a]["old"]):.1f}" cy="{y2(pct(a)):.1f}" r="4.6" fill="var(--{col})" stroke="var(--surface)" stroke-width="1.6" pointer-events="none"/>'
           f'<text x="{x2(S[a]["old"]) + dx:.1f}" y="{y2(pct(a)) + dy:.1f}" text-anchor="{anc}" font-size="11" style="fill:var(--ink)">{html.escape(LBL[a])}</text>')
for a in ("w20cur", "fullcur"):
    col = "s1" if a == "w20cur" else "s2"
    tip = f"<b>{LBL[a]}</b><br><span class='m'>{S[a]['old']:.1f} old nodes seen · {pct(a):.0f}% reached</span>"
    g2 += (f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{x2(S[a]["old"]):.1f}" cy="{y2(pct(a)):.1f}" r="10" fill="transparent"/>'
           f'<circle cx="{x2(S[a]["old"]):.1f}" cy="{y2(pct(a)):.1f}" r="4.6" fill="var(--surface)" stroke="var(--{col})" stroke-width="2.2" pointer-events="none"/>'
           f'<text x="{x2(S[a]["old"]):.1f}" y="{y2(pct(a)) + 20:.1f}" text-anchor="middle" font-size="11" style="fill:var(--ink)">{"window 20" if a == "w20cur" else "full graph"}, hidden</text>')
chart2 = f'<svg viewBox="0 0 {W2} {H2}" width="100%" role="img" aria-label="Success rate against old nodes seen">{g2}</svg>'

# ---------- table: everything per arm ----------
def row(a):
    s = S[a]
    lab = html.escape(LBL[a])
    vs = "" if a == "none" else f'+{s["wins"]} / −{s["losses"]}'
    p = "" if a == "none" else (f'{s["p"]:.3f}' if s["p"] >= 0.001 else "&lt;0.001")
    pt = " · ".join(f'{s["per_tour"][t]}' for t in X.TOURS)
    return (f'<tr><td>{lab}</td><td>{s["k"]}/{s["n"]}</td><td>{pt}</td><td>{s["spl"]:.2f}</td><td>{s["ne"]:.1f} m</td>'
            f'<td>{s["steps"]:.0f}</td><td>{s["long"]}</td><td>{s["stop1"]}</td><td>{s["old"]:.0f}</td><td>{vs}</td><td>{p}</td></tr>')
table = ('<div class="panel"><div class="scroll"><table class="eps t3"><thead><tr><th>arm</th><th>reached</th><th>per tour</th><th>SPL</th><th>miss</th>'
         '<th>steps</th><th>&gt;150</th><th>1-step</th><th>old seen</th><th>vs wiped</th><th>p</th></tr></thead><tbody>'
         + "".join(row(a) for a in rows) + '</tbody></table></div></div>')
table = table.replace("table.eps th:nth-child(-n+3)", "")

extra_css = """
table.ref{width:100%;border-collapse:collapse;font-size:.86rem}
table.ref th{font:500 .66rem/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);text-align:left;padding:0 12px 9px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
table.ref td{padding:10px 12px 10px 0;vertical-align:top;border-bottom:1px solid var(--soft)} table.ref tbody tr:last-child td{border-bottom:0}
.prose ul+p{margin-top:.8em}
.tbl2 td:first-child{text-align:left!important}
table.eps th:nth-child(2),table.eps td:nth-child(2),table.eps th:nth-child(3),table.eps td:nth-child(3){text-align:right}
table.eps th:first-child,table.eps td:first-child{text-align:left}
table.eps tbody{border-bottom:0} table.eps tbody tr:not(:last-child) td{border-bottom:1px solid var(--soft)}
table.eps.t3 td,table.eps.t3 th{padding-left:6px;padding-right:6px} table.eps.t3 td:first-child{white-space:normal;min-width:150px;font-size:.78rem} table.eps.t3{font-size:.8rem}
.ring{display:inline-block;width:11px;height:11px;border-radius:3px;border:2px solid var(--dim);background:var(--surface)}
"""

BODY = {}   # filled by the narrative module below (written after the numbers were seen)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp3_text as T  # noqa: E402

sections = T.sections(S, N, chart1, chart2, table)
out = L.page(T.TITLE, 3, [1, 2], T.lead(S, N), sections, L.TIP_JS, T.REPRO)
out = out.replace("</style>", extra_css + "</style>", 1)
open(sys.argv[1] if len(sys.argv) > 1 else "exp3.html", "w").write(out)
print("built", len(out), "bytes")
