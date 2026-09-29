#!/usr/bin/env python3
"""Experiment 4 page: is it old NODE tokens or old FRONTIER tokens (and how they're made) that hurts?
Reads docs/experiments/exp3_diagnosis (w20/w20cur/fullcur) + exp4_split (nodes hidden, and the absorb-fixed clean control)."""
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X = f"{REPO}/docs/experiments"
TOURS = (0, 1, 2)
base = {s: json.load(open(f"{X}/exp3_diagnosis/s{s}/none/results.json"))["episodes"] for s in TOURS}


def load(path):
    return {s: json.load(open(f"{path.format(s=s)}/results.json"))["episodes"] for s in TOURS}


ARMS = [
    ("map wiped", None, base),
    ("nodes shown, frontiers shown", "s1", load(f"{X}/exp3_diagnosis/s{{s}}/w20")),
    ("nodes HIDDEN, frontiers shown", "s1", load(f"{X}/exp4_split/s{{s}}/nc_gall_w20")),
    ("nodes shown, frontiers HIDDEN", "s2", load(f"{X}/exp3_diagnosis/s{{s}}/w20cur")),
    ("nodes HIDDEN, frontiers HIDDEN\n(absorption still default)", "s2", load(f"{X}/exp4_split/s{{s}}/nc_gcur_w20")),
    ("nodes+frontiers HIDDEN,\nabsorption fixed", "ok", load(f"{X}/exp4_split/s{{s}}/clean_control_w20")),
]
N = 45


def summ(tours):
    rows = [(s, e, st) for s in TOURS for e, st in tours[s].items()]
    k = sum(st["success"] > .5 for _, _, st in rows)
    w = sum(1 for s, e, st in rows if st["success"] > .5 and base[s][e]["success"] <= .5)
    l = sum(1 for s, e, st in rows if st["success"] <= .5 and base[s][e]["success"] > .5)
    return k, w, l


S = [(lab, col, summ(t)) for lab, col, t in ARMS]
diffs = 0
KEYS = ["steps_taken", "distance_to_goal", "success", "spl", "path_length", "ndtw"]
cc = load(f"{X}/exp4_split/s{{s}}/clean_control_w20")
for s in TOURS:
    for e in base[s]:
        for k in KEYS:
            diffs += abs(base[s][e][k] - cc[s][e][k]) > 1e-6

# ---- chart: horizontal bars, one per arm ----
W, RH, ml, mr, mt = 820, 58, 300, 70, 8
H = mt + RH * len(ARMS) + 26
xs = lambda v: ml + (W - ml - mr) * v / 100
parts = [f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{mt}" y2="{H - 22}" stroke="var(--grid)"/>'
         f'<text x="{xs(v):.1f}" y="{H - 6}" font-size="10.5" text-anchor="middle">{v}%</text>' for v in (0, 20, 40, 60, 80, 100)]
base_x = xs(100 * S[0][2][0] / N)
for i, (lab, col, (k, w, l)) in enumerate(S):
    y0 = mt + i * RH
    pct = 100 * k / N
    fill = f'var(--{col})' if col in ("ok",) else (f'var(--s1)' if col == "s1" else (f'var(--s2)' if col == "s2" else 'var(--dim)'))
    tip = f"<b>{html.escape(lab.replace(chr(10), ' '))}</b><br><span class='m'>{k} of {N} reached" + (f" · +{w}/-{l} vs wiped</span>" if col else "</span>")
    for j, line in enumerate(lab.split("\n")):
        parts.append(f'<text x="{ml - 12}" y="{y0 + RH / 2 + 4 + (j - (len(lab.split(chr(10))) - 1) / 2) * 13:.1f}" font-size="11.3" text-anchor="end" style="fill:var(--ink)">{html.escape(line)}</text>')
    parts.append(f'<rect data-tip="{html.escape(tip, quote=True)}" x="{ml}" y="{y0 + 12}" width="{xs(pct) - ml:.1f}" height="{RH - 24}" rx="3" fill="{fill}"/>')
    parts.append(f'<text x="{xs(pct) + 8:.1f}" y="{y0 + RH / 2 + 4:.1f}" font-size="12" style="fill:var(--ink);font-weight:600">{k}/{N}</text>')
parts.append(f'<line x1="{base_x:.1f}" x2="{base_x:.1f}" y1="{mt}" y2="{H - 22}" stroke="var(--ink)" stroke-dasharray="4 4" opacity=".5"/>')
chart = f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Reached goal for each node/frontier visibility combination">{"".join(parts)}</svg>'

setup = """<dl class="kv">
<dt>Question</dt><dd>Experiment 3 found that persisting the map hurts. Is it the old <b>visited-node</b> tokens the planner sees, the old
<b>frontier</b> tokens, or something else?</dd>
<dt>Method</dt><dd>Independently hide old nodes and old frontiers from the planner's input (they still shape routes/distances either way),
on the same 45 episodes used throughout.</dd>
</dl>"""

lead = ("Hiding old visited-node tokens from the planner made <b>no difference at all</b>. The entire effect traced back to one thing: "
        "walking onto already-mapped ground silently absorbed the agent's own proposed waypoints into old places instead of making "
        "them into frontiers, so the episode never got its normal set of options.")

reading = f"""<div class="prose"><p>Rows 2 and 3 are identical whether or not old node tokens are shown (both {S[2][2][0]}/{N}). Rows 4 and 5 are
identical too ({S[4][2][0]}/{N}). Node tokens are simply not the mechanism.</p>
<p>The fix was elsewhere: <b>absorption</b>. A proposed waypoint landing within 0.5 m of an existing node has never made a frontier, in
stock ETPNav or persisted. On fresh ground each episode always got a fresh set of frontiers. On <i>already-mapped</i> ground, that same rule
now silently ate the current episode's own options, because "existing node" included every earlier episode's nodes too. Scoping absorption
to the current episode's own nodes restores exactly what a fresh episode would have had, and the result is <b>{S[5][2][0]}/{N}</b>, a bit-for-bit
match to the map-wiped baseline in 43 of 45 episodes (the other 2 take a real, tiny shortcut through the still-connected old graph when
computing path distances — no effect on any outcome here, but worth knowing about).</p></div>"""

parity = f"""<div class="panel"><dl class="readout"><div><dt>Differences</dt><dd>{diffs}</dd></div>
<div><dt>Episodes</dt><dd>{N}</dd></div><div><dt>Metrics each</dt><dd>{len(KEYS)}</dd></div></dl>
<p class="sub" style="margin:0">Map wiped vs. the absorption-fixed control: {diffs} of {N * len(KEYS)} values differ.</p></div>"""

sections = [("Setup", setup),
            ("What each combination reaches",
             f'<div class="panel"><h3>Reached the goal, {N} episodes</h3><p class="sub">Dashed line: map-wiped baseline. Hover a bar for the paired comparison.</p>{chart}</div>'),
            ("Reading it", reading),
            ("How close is the fix to a true baseline?", parity),
            ("Next", '<div class="prose"><p>With absorption fixed, giving the planner memory it can actually use means reintroducing some old '
                     'places <b>as frontiers</b> — the only thing it can select. '
                     '<a href="https://claude.ai/artifact/VcubWoBV9d2DymS3sSNTeh" target="_blank" rel="noopener">Read Experiment 5</a>.</p></div>')]
repro = ("Reproduce: <code>scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --sample-seed {0,1,2} --mode window --window-nodes 20 "
         "--nodes {all,current} --ghosts {all,current} --absorb {all,current}</code> · data: <code>docs/experiments/exp3_diagnosis/</code>, "
         "<code>docs/experiments/exp4_split/</code>")
out = L.page("Nodes or Frontiers?", 4, [1, 2, 3], lead, sections, L.TIP_JS, repro)
open(sys.argv[1] if len(sys.argv) > 1 else "exp4.html", "w").write(out)
print("built", len(out), "bytes; S=", [(lab.replace(chr(10), ' '), k) for lab, _, (k, w, l) in S], "diffs=", diffs)
