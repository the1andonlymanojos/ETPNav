#!/usr/bin/env python3
"""Experiment 2 page: does a persisted topological map connect up and feed the planner valid input?
Reads docs/experiments/exp1_baseline + exp2_mechanics/{none,window,full}; needs the navmesh (habitat_sim.PathFinder, no GPU)."""
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X = f"{REPO}/docs/experiments"
R = {m: json.load(open(f"{X}/exp2_mechanics/{m}/results.json")) for m in ("none", "window", "full")}
D = {m: json.load(open(f"{X}/exp2_mechanics/{m}/persist_debug.json")) for m in ("window", "full")}
base1 = json.load(open(f"{X}/exp1_baseline/baseline/results.json"))["episodes"]
order = R["none"]["intended_order"]
n = len(order)

# ---- check 1: feature off == stock code (Exp 1 ran before any of the new code existed) ----
KEYS = ["steps_taken", "distance_to_goal", "success", "oracle_success", "path_length", "collisions", "spl", "ndtw", "sdtw", "ghost_cnt"]
diffs = sum(1 for e in order for k in KEYS if abs(R["none"]["episodes"][e][k] - base1[e][k]) > 1e-6)

# ---- per-arm summaries ----
def summ(m):
    E = R[m]["episodes"]
    return dict(k=sum(E[e]["success"] > .5 for e in order), spl=sum(E[e]["spl"] for e in order) / n,
                ne=sum(E[e]["distance_to_goal"] for e in order) / n,
                steps=sum(E[e]["steps_taken"] for e in order) / n, wall=R[m]["wall_seconds"])
S = {m: summ(m) for m in R}
Ew, Ef = R["window"]["episodes"], R["full"]["episodes"]
saw_old = {m: sum(1 for e in order[1:] if R[m]["episodes"][e]["g_old_visible_mean"] > 0) for m in ("window", "full")}
last = D["window"][order[-1]]
n_nodes, n_ghosts, n_links = len(last["nodes"]), len(last["ghosts"]), len(last["reloc_links"])
n_nodes_full = len(D["full"][order[-1]]["nodes"])
long_eps = {m: sum(1 for e in order if R[m]["episodes"][e]["steps_taken"] > 150) for m in R}
stop1 = [i + 1 for i, e in enumerate(order) if Ef[e]["steps_taken"] <= 1]

# ---- floor plan ----
import habitat_sim  # noqa: E402
pf = habitat_sim.PathFinder()
pf.load_nav_mesh(f"{REPO}/data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.navmesh")
top = pf.get_topdown_view(0.1, 0.17)
xmin, zmin = float(pf.get_bounds()[0][0]), float(pf.get_bounds()[0][2])
SC = 4.0  # svg units per 0.1 m cell
rects = []
for i in range(top.shape[0]):
    row, j = top[i], 0
    while j < top.shape[1]:
        if row[j]:
            s = j
            while j < top.shape[1] and row[j]:
                j += 1
            rects.append(f"M{s * SC:.0f} {i * SC:.0f}h{(j - s) * SC:.0f}v{SC:.0f}h-{(j - s) * SC:.0f}z")
        else:
            j += 1
px = lambda x: (x - xmin) / 0.1 * SC + SC / 2
pz = lambda z: (z - zmin) / 0.1 * SC + SC / 2
FW, FH = top.shape[1] * SC, top.shape[0] * SC
els = [f'<path d="{"".join(rects)}" fill="var(--floor)"/>']
for gx, gz in last["ghosts"]:
    els.append(f'<circle cx="{px(gx):.1f}" cy="{pz(gz):.1f}" r="3.6" fill="none" stroke="var(--dim)" stroke-width="1.1" opacity=".75"/>')
for a, b in last["reloc_links"]:
    els.append(f'<line x1="{px(a[0]):.1f}" y1="{pz(a[2]):.1f}" x2="{px(b[0]):.1f}" y2="{pz(b[2]):.1f}" stroke="var(--link)" stroke-width="3.2" stroke-linecap="round"/>')
for x, z, ep in last["nodes"]:
    pct = round(100 * ep / (n - 1))
    tip = f"<b>Episode {ep + 1}</b><br><span class='m'>map node · {x:.1f}, {z:.1f} m</span>"
    els.append(f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{px(x):.1f}" cy="{pz(z):.1f}" r="5.2" '
               f'style="fill:color-mix(in srgb, var(--seq-late) {pct}%, var(--seq-early))" stroke="var(--surface)" stroke-width="1.6"/>')
floor = (f'<svg viewBox="0 0 {FW:.0f} {FH:.0f}" width="100%" role="img" aria-label="Top-down floor plan with the persisted map">'
         + "".join(els) + "</svg>")

# ---- old nodes the planner sees, per episode ----
W, H, ml, mr, mt, mb = 820, 270, 40, 104, 14, 34
ymax = 50.0
xs = lambda i: ml + (W - ml - mr) * i / (n - 1)
ys = lambda v: mt + (H - mt - mb) * (1 - v / ymax)
grid = "".join(f'<line x1="{ml}" x2="{W - mr}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="var(--grid)"/>'
               f'<text x="{ml - 8}" y="{ys(v) + 3.5:.1f}" font-size="10.5" text-anchor="end">{v}</text>' for v in (0, 10, 20, 30, 40, 50))
xt = "".join(f'<text x="{xs(i):.1f}" y="{H - mb + 17}" font-size="10.5" text-anchor="middle">{i + 1}</text>' for i in range(n))
lines, dots = [], []
for m, col, name in (("window", "s1", "window"), ("full", "s2", "full graph")):
    vals = [R[m]["episodes"][e]["g_old_visible_mean"] for e in order]
    pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(vals))
    lines.append(f'<polyline points="{pts}" fill="none" stroke="var(--{col})" stroke-width="2.2" stroke-linejoin="round"/>')
    lines.append(f'<text x="{xs(n - 1) + 9:.1f}" y="{ys(vals[-1]) + 4:.1f}" font-size="11.5" style="fill:var(--{col});font-weight:600">{name}</text>')
    for i, v in enumerate(vals):
        tip = f"<b>Episode {i + 1}</b> · {name}<br><span class='m'>planner saw {v:.1f} old nodes on average</span>"
        dots.append(f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="9" fill="transparent"/>'
                    f'<circle cx="{xs(i):.1f}" cy="{ys(v):.1f}" r="3.2" fill="var(--{col})" stroke="var(--surface)" stroke-width="1.4" pointer-events="none"/>')
seen = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Old map nodes visible to the planner per episode">{grid}{xt}'
        f'{"".join(lines)}{"".join(dots)}<text x="{ml}" y="{H - 2}" font-size="10.5">episode in tour order</text></svg>')
seen_rows = "".join(f'<tr><td>{i + 1}</td><td>{R["window"]["episodes"][e]["g_old_visible_mean"]:.1f}</td>'
                    f'<td>{R["full"]["episodes"][e]["g_old_visible_mean"]:.1f}</td></tr>' for i, e in enumerate(order))

# ---- outcome strips: one row per arm, one cell per episode ----
def strip(m, label):
    cells = []
    for i, e in enumerate(order):
        s = R[m]["episodes"][e]
        ok = s["success"] > .5
        tip = (f"<b>Episode {i + 1}</b> · {label}<br><span class='m'>{'reached' if ok else 'missed'} · {s['distance_to_goal']:.1f} m from goal · "
               f"{s['steps_taken']:.0f} sim steps</span>")
        cells.append(f'<span class="cell {"ok" if ok else "miss"}" data-tip="{html.escape(tip, quote=True)}"></span>')
    return f'<div class="strip"><span class="rl">{label}</span><div class="cells">{"".join(cells)}</div><span class="rv">{S[m]["k"]} of {n}</span></div>'

strips = strip("none", "map wiped") + strip("window", "window") + strip("full", "full graph")
extra_css = """<style>
.strip{display:grid;grid-template-columns:96px 1fr 64px;align-items:center;gap:12px;margin:7px 0}
.strip .rl{font:500 .78rem "IBM Plex Mono",monospace;color:var(--dim)} .strip .rv{font:600 .82rem "IBM Plex Mono",monospace;text-align:right}
.cells{display:grid;grid-template-columns:repeat(15,1fr);gap:3px}
.cell{height:26px;border-radius:4px} .cell.ok{background:var(--ok)} .cell.miss{background:var(--miss)}
.ramp{display:inline-block;width:120px;height:9px;border-radius:5px;background:linear-gradient(90deg,var(--seq-early),var(--seq-late));vertical-align:middle}
.ring{display:inline-block;width:11px;height:11px;border-radius:50%;border:1.3px solid var(--dim)} .lk{display:inline-block;width:16px;height:3px;border-radius:2px;background:var(--link)}
.compare{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px 28px;margin:0}
.compare div{padding-left:12px;border-left:2px solid var(--rule)} .compare dt{font:500 .72rem "IBM Plex Mono",monospace;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
.compare dd{margin:2px 0 0;font:600 1.45rem "IBM Plex Mono",monospace} .compare dd small{font-size:.78rem;color:var(--dim);font-weight:500;margin-left:4px}
details summary{cursor:pointer;color:var(--dim);font-size:.84rem;margin-top:10px} details summary:focus-visible{outline:2px solid var(--ok);outline-offset:3px}
@media (max-width:560px){.strip{grid-template-columns:1fr 54px}.strip .rl{grid-column:1/-1;margin-bottom:-4px}}
</style>"""

new = f"""<dl class="kv">
<dt>Window</dt><dd>The planner sees this episode's nodes plus the nearest earlier nodes by walking distance, about 20 in total.</dd>
<dt>Full graph</dt><dd>The planner sees every node connected to where the agent stands, however many there are.</dd>
<dt>Joining</dt><dd>A new node within 1 m of an earlier node is linked to it. That link is the only thing that joins a new episode to the old map.</dd>
<dt>Kept</dt><dd>Places the agent has been and what it saw there. Instructions, goals and stop scores are reset every episode, so no episode sees another's answer.</dd>
</dl>"""

parity = f"""<div class="panel"><dl class="readout" style="margin-bottom:8px">
<div><dt>Differences</dt><dd>{diffs}</dd></div><div><dt>Episodes</dt><dd>{n}</dd></div><div><dt>Metrics each</dt><dd>{len(KEYS)}</dd></div></dl>
<p class="sub" style="margin:0">Same 15 episodes, feature off, new code against the stock code from Experiment 1: every number matches to six decimals.
That includes episode order, which Experiment 1 could not control. Anything that changes later comes from the map, not from the plumbing.</p></div>"""

mapp = f"""<div class="panel"><h3>The map after 15 episodes, window arm</h3>
<p class="sub">{n_nodes} nodes, {n_ghosts} unexplored frontiers and {n_links} joining links, all from one continuous run. Hover a node to see which episode placed it.</p>
{floor}
<div class="legend"><span><i class="ramp"></i>episode 1 to 15</span><span><i class="ring"></i>unexplored frontier</span><span><i class="lk"></i>join to an earlier episode</span></div></div>
<p class="dim" style="margin-top:10px;max-width:66ch">The full-graph arm walked different routes and ended with {n_nodes_full} nodes. In the window arm the planner saw an earlier episode's
nodes in {saw_old['window']} of the {n - 1} later episodes; in the full arm, {saw_old['full']} of {n - 1}. The other episodes started far from anything already mapped, so they began with an empty map, exactly as in the baseline.</p>"""

seenp = f"""<div class="panel"><h3>Old nodes the planner could see</h3>
<p class="sub">Average per decision. The window stays near its cap; the full graph keeps growing as the tour goes on.</p>{seen}
<div class="legend"><span><i class="sw s1"></i>window</span><span><i class="sw s2"></i>full graph</span></div>
<details><summary>Show the numbers</summary><div class="scroll"><table class="eps"><thead><tr><th>episode</th><th>window</th><th>full</th></tr></thead><tbody>{seen_rows}</tbody></table></div></details></div>"""

bug = """<div class="callout"><b>The first window run crashed on episode 7.</b> The planner was handed a node with no known distance from the agent. Nodes from this episode were always kept,
but the route to one of them could run through an older node that the size cap had dropped. The fix keeps every node on those routes, so the cap is slightly soft.
A synthetic test now builds exactly that shape and fails without the fix, and a 3,400-state random test checks every node the planner sees can be reached.</div>"""

acc = f"""<div class="panel"><h3>Reached the goal, per episode</h3>
<p class="sub">Same 15 episodes, same order, one pass each. Hover a cell for details.</p>{strips}
<div class="legend"><span><i class="sw ok"></i>reached</span><span><i class="sw miss"></i>missed</span></div></div>
<dl class="compare" style="margin-top:16px">
<div><dt>Map wiped</dt><dd>{S['none']['spl']:.2f}<small>SPL · {S['none']['ne']:.1f} m miss</small></dd></div>
<div><dt>Window</dt><dd>{S['window']['spl']:.2f}<small>SPL · {S['window']['ne']:.1f} m miss</small></dd></div>
<div><dt>Full graph</dt><dd>{S['full']['spl']:.2f}<small>SPL · {S['full']['ne']:.1f} m miss</small></dd></div></dl>"""

reading = f"""<div class="prose"><p><b>This is a first look, not a result.</b> It is one house, one ordering, 15 episodes. Persisting the map made things worse here.
Both arms match the baseline on episodes 1 and 2, where nothing is shared, then fall behind.</p>
<p>Two things stand out. Episodes ran much longer: {long_eps['none']} baseline episodes needed more than 150 simulator steps, against {long_eps['window']} for the window and {long_eps['full']} for the full graph,
and one window episode took {max(Ew[e]['steps_taken'] for e in order):.0f}. And in the full-graph arm the agent stopped on its very first move in episodes {' and '.join(map(str, stop1))}.</p>
<p>Three explanations I had at this point. Experiment 3 has since tested them:</p><ul>
<li><b>Stale frontiers.</b> Unexplored frontiers left over from earlier episodes tempt the agent to walk off toward places the current instruction never mentioned.</li>
<li><b>An unfamiliar map.</b> The planner trained on maps of at most 15 steps. It now sees 20 to 50 nodes at once.</li>
<li><b>A misleading age.</b> Every old node is marked as visited at step 1, which the planner has never seen.</li></ul></div>"""

nxt = """<div class="prose"><p>Experiment 3 asks the same question properly: 45 episodes in three tours, so one lucky or unlucky order can't decide it,
with the first explanation switched off by hiding earlier episodes' frontiers. <a href="https://claude.ai/artifact/DwGo95P8inYAXTPhupcBVF" target="_blank" rel="noopener">Read Experiment 3</a>.</p></div>"""

lead = ("Keeping the map between episodes <b>works mechanically</b>: a new episode's graph joins the old one and the planner gets valid input. "
        "The payoff we hoped for, better navigation, did not show up. That is less surprising than it first looked: ETPNav's planner can only choose "
        "unexplored frontiers, so a remembered map gave it no way to help.")
sections = [("What was changed", new), ("Check 1 · Feature off behaves like stock ETPNav", parity),
            ("Check 2 · The map joins up", mapp), ("Check 3 · What the planner sees", seenp),
            ("A bug the run found", bug), ("A first look at accuracy", acc), ("Reading it", reading), ("Next", nxt)]
repro = ("Reproduce: <code>python scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --mode {none,window,full} --out docs/experiments/exp2_mechanics/&lt;mode&gt;</code> "
         "· tests: <code>python scripts/selftest_persist_graph.py</code> · data: <code>docs/experiments/exp2_mechanics/</code>")
out = L.page("Does the Map Connect?", 2, [1, 2], lead, sections, L.TIP_JS, repro)
out = out.replace("</style>", extra_css.replace("<style>", "").replace("</style>", "") + "</style>", 1)   # keep <title> first
open(sys.argv[1] if len(sys.argv) > 1 else "exp2.html", "w").write(out)
print(f"diffs={diffs} S={ {m: (v['k'], round(v['spl'], 3)) for m, v in S.items()} } saw_old={saw_old} long={long_eps} stop1={stop1} nodes={n_nodes}/{n_nodes_full} bytes={len(out)}")
