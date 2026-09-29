#!/usr/bin/env python3
"""Experiment 5 page: reopen nearby earlier-episode places as frontier tokens (a radius sweep) on top of the
absorption-fixed clean baseline from Experiment 4. Reads docs/experiments/exp5_reopen + exp4_split via exp5_data."""
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402
import exp5_data as X  # noqa: E402

data, base = X.load()
S = X.stats(data, base)
N = S[-1.0]["n"]
LBL = {-1.0: "reopening off", 0.0: "0 m (exact only)", 1.5: "1.5 m", 3.0: "3 m", 6.0: "6 m", 30.0: "30 m (whole floor)"}
hits, checked = X.exact_zero_hits()

# ---- chart: reached vs radius, x on a stretched scale so 0/1.5/3/6/30 are all legible ----
XPOS = {-1.0: 0, 0.0: 1, 1.5: 2, 3.0: 3, 6.0: 4, 30.0: 5}
W, H, ml, mr, mt, mb = 820, 300, 46, 20, 14, 50
xs = lambda r: ml + (W - ml - mr) * XPOS[r] / 5
ys = lambda v: mt + (H - mt - mb) * (1 - v / 100)
grid = "".join(f'<line x1="{ml}" x2="{W - mr}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="var(--grid)"/>'
               f'<text x="{ml - 8}" y="{ys(v) + 3.5:.1f}" font-size="10.5" text-anchor="end">{v}%</text>' for v in (0, 20, 40, 60, 80, 100))
xt = "".join(f'<text x="{xs(r):.1f}" y="{H - mb + 18}" font-size="10.5" text-anchor="middle">{LBL[r].split(" (")[0]}</text>' for r in X.RADII)
pts = " ".join(f"{xs(r):.1f},{ys(100 * S[r]['k'] / N):.1f}" for r in X.RADII)
line = f'<polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="2.2" stroke-linejoin="round"/>'
dots = []
for r in X.RADII:
    v = 100 * S[r]["k"] / N
    tip = (f"<b>{html.escape(LBL[r])}</b><br><span class='m'>{S[r]['k']} of {N} reached ({v:.0f}%) · SPL {S[r]['spl']:.2f} · "
           f"{S[r]['ne']:.1f} m miss<br>+{S[r]['wins']}/-{S[r]['losses']} vs map wiped</span>")
    col = "dim" if r < 0 else "s1"
    dots.append(f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{xs(r):.1f}" cy="{ys(v):.1f}" r="11" fill="transparent"/>'
                f'<circle cx="{xs(r):.1f}" cy="{ys(v):.1f}" r="4.6" fill="var(--{col})" stroke="var(--surface)" stroke-width="1.7" pointer-events="none"/>')
base_y = ys(100 * S[-1.0]["k"] / N)
chart = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Success rate against reopen radius">{grid}'
         f'<line x1="{ml}" x2="{W - mr}" y1="{base_y:.1f}" y2="{base_y:.1f}" stroke="var(--ink)" stroke-dasharray="4 4" opacity=".45"/>'
         f'<text x="{W - mr}" y="{base_y - 6:.1f}" font-size="10.5" text-anchor="end" style="fill:var(--ink)">reopening off / map wiped</text>'
         f'{line}{"".join(dots)}{xt}<text x="{ml}" y="{H - 4}" font-size="10.5">how far away a place can be and still get reopened</text></svg>')

rows = "".join(f'<tr><td>{html.escape(LBL[r])}</td><td>{S[r]["k"]}/{N}</td><td>{S[r]["spl"]:.2f}</td><td>{S[r]["ne"]:.1f} m</td>'
              f'<td>{S[r]["steps"]:.0f}</td><td>+{S[r]["wins"]}/−{S[r]["losses"]}</td></tr>' for r in X.RADII)
table = ('<div class="panel"><div class="scroll"><table class="eps t3"><thead><tr><th>radius</th><th>reached</th><th>SPL</th><th>miss</th>'
         '<th>sim steps</th><th>vs wiped</th></tr></thead><tbody>' + rows + '</tbody></table></div></div>')

setup = f"""<dl class="kv">
<dt>Built on</dt><dd>Experiment 4's absorption-fixed baseline: hiding old node and frontier tokens, with absorption scoped
to this episode. Alone, that matches the map-wiped baseline exactly.</dd>
<dt>The change</dt><dd>An earlier episode's node within <i>radius</i> of the agent is reintroduced as a frontier: the only
kind of token ETPNav's planner can actually choose. Chosen once, it's gone for the rest of the episode.</dd>
<dt>Sweep</dt><dd>Radius 0 (only an exact position match), 1.5, 3, 6, and 30 m (the whole floor), on the same 45 episodes.</dd>
</dl>"""

lead = (f"Once the planner could actually act on memory, it made things <b>worse</b>, and the harm grew with how much of it was reachable: "
        f"from {S[-1.0]['k']}/{N} with reopening off to {S[6.0]['k']}/{N} at 6 m and beyond.")

reading = f"""<div class="prose"><p>The curve drops the moment reopening does anything at all, then flattens: 6 m and 30 m land on the same
{S[6.0]['k']}/{N}, so exposing the <i>whole</i> mapped floor is no worse than exposing a 6 m radius of it. That rules out "there's simply too much
old memory" as the cause — a little reopened memory is already enough to do most of the damage.</p>
<p>The 0 m point needed a second look before trusting it. Reopening at an exact-zero-metre threshold should almost never fire — but it
did: <b>{hits} of {checked}</b> later-episode nodes in that run sit bit-for-bit on top of an earlier one. That isn't coincidence; ETPNav places
nodes from a fixed grid of step sizes and turn angles on a fixed navmesh, so different episodes' paths land on the exact same coordinates
more often than continuous 3D space would suggest. 0 m is a real, if very sparse, sample of reopening — not a second copy of "off" — and
its result ({S[0.0]['k']}/{N}) sits right where a first small step should: indistinguishable from doing nothing.</p>
<p>The likely reason a reopened place hurts once it's reachable: the planner has no way to judge whether a remembered place is <i>relevant</i>
to the instruction it's currently reading. It only knows "this looked like somewhere I could stand." Handing back an unrelated old place is
functionally a distraction, no different in kind from the stale frontiers ruled out in Experiment 3 — except this time the agent can act on
it, and that turns out to be worse, not better.</p></div>"""

nxt = """<div class="prose"><p>The missing piece isn't reach, it's relevance: a way to prefer a remembered place that actually matches
what the current instruction describes, rather than reopening whatever is simply nearby. That likely needs either a trained retrieval signal or
fine-tuning the planner on tours where it has to use memory to do better — not memory added zero-shot to a model that never saw it in training.</p></div>"""

sections = [("Setup", setup),
            ("Result", f'<div class="panel"><h3>Reached the goal against reopen radius</h3><p class="sub">Hover a point for the full comparison.</p>{chart}</div>'),
            ("Reading it", reading), ("Every radius", table), ("Next", nxt)]
repro = ("Reproduce: <code>scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --sample-seed {0,1,2} --mode window --window-nodes 20 "
         "--nodes current --ghosts current --absorb current --reopen-radius R</code> · analysis: <code>docs/experiments/exp5_data.py</code> "
         "· data: <code>docs/experiments/exp5_reopen/</code>")
out = L.page("Does Reachable Memory Help?", 5, [1, 2, 3, 4], lead, sections, L.TIP_JS, repro)
extra_css = "<style>table.eps.t3 td,table.eps.t3 th{padding-left:10px;padding-right:10px}</style>"
out = out.replace("</style>", extra_css.replace("<style>", "").replace("</style>", "") + "</style>", 1)
open(sys.argv[1] if len(sys.argv) > 1 else "exp5.html", "w").write(out)
print("built", len(out), "bytes")
