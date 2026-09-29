#!/usr/bin/env python3
"""Experiment 1 page: stock ETPNav (map wiped every episode) on 15 real R2R-CE val_unseen episodes in one house.
Reads docs/experiments/exp1_baseline/baseline/results.json (+ the split for instruction text); writes an HTML fragment."""
import gzip
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
res = json.load(open(f"{REPO}/docs/experiments/exp1_baseline/baseline/results.json"))
split = "tour_zsNo4H_n15_s0_o0"
eps_meta = {str(e["episode_id"]): e for e in json.load(gzip.open(
    f"{REPO}/data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz"))["episodes"]}
raw = json.load(gzip.open(f"{REPO}/data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen.json.gz"))["episodes"]
n_house = sum(1 for e in raw if e["scene_id"].endswith("zsNo4HB9uLZ/zsNo4HB9uLZ.glb"))

order = json.load(open(f"{REPO}/docs/experiments/exp2_mechanics/none/results.json"))["intended_order"]   # tour order (same results)
E = res["episodes"]
n = len(order)
ok = [E[e]["success"] > 0.5 for e in order]
k = sum(ok)
lo, hi = L.wilson(k, n)
mean = lambda key: sum(E[e][key] for e in order) / n
spl, ne, ndtw = mean("spl"), mean("distance_to_goal"), mean("ndtw")
near = [e for e in order if not E[e]["success"] and E[e]["distance_to_goal"] < 4.0]

# ---- chart: final distance to goal per episode, 3 m success line ----
W, H, ml, mr, mt, mb = 820, 300, 44, 12, 14, 34
ymax = 20.0
bw = (W - ml - mr) / n
y = lambda v: mt + (H - mt - mb) * (1 - min(v, ymax) / ymax)
bars = []
for i, e in enumerate(order):
    d = E[e]["distance_to_goal"]
    cls = "ok" if ok[i] else "miss"
    tip = (f"<b>Episode {i + 1}</b> · id {e}<br><span class='m'>{'reached goal' if ok[i] else 'missed'} · "
           f"{d:.1f} m from goal · path {E[e]['path_length']:.1f} m</span><br>{html.escape(eps_meta[e]['instruction']['instruction_text'][:170])}…")
    x0 = ml + i * bw + bw * 0.16
    bars.append(f'<rect data-tip="{html.escape(tip, quote=True)}" x="{x0:.1f}" y="{y(d):.1f}" width="{bw * 0.68:.1f}" '
                f'height="{H - mb - y(d):.1f}" rx="3" fill="var(--{cls})"/>')
    bars.append(f'<text x="{x0 + bw * 0.34:.1f}" y="{H - mb + 16}" font-size="10.5" text-anchor="middle">{i + 1}</text>')
grid = "".join(f'<line x1="{ml}" x2="{W - mr}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
               f'<text x="{ml - 8}" y="{y(v) + 3.5:.1f}" font-size="10.5" text-anchor="end">{v}</text>' for v in (0, 5, 10, 15, 20))
chart = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Final distance to goal for each of 15 episodes">'
         f'{grid}<line x1="{ml}" x2="{W - mr}" y1="{y(3):.1f}" y2="{y(3):.1f}" stroke="var(--ink)" stroke-dasharray="4 4" stroke-width="1.2"/>'
         f'<text x="{W - mr}" y="{y(3) - 6:.1f}" font-size="11" text-anchor="end" style="fill:var(--ink)">3 m: counts as reached</text>'
         f'{"".join(bars)}<text x="{ml}" y="{H - 2}" font-size="10.5">episode in tour order</text></svg>')

# ---- table ----
rows = []
for i, e in enumerate(order):
    s = E[e]
    chip = '<span class="chip ok">reached</span>' if ok[i] else '<span class="chip miss">missed</span>'
    rows.append(f'<tbody><tr><td>{i + 1}</td><td class="mono">{e}</td><td>{chip}</td><td>{s["distance_to_goal"]:.1f} m</td>'
                f'<td>{s["path_length"]:.1f} m</td><td>{s["spl"]:.2f}</td><td>{s["ndtw"]:.2f}</td></tr>'
                f'<tr class="i"><td colspan="7">{html.escape(eps_meta[e]["instruction"]["instruction_text"])}</td></tr></tbody>')
table = ('<div class="panel"><div class="scroll"><table class="eps"><thead><tr><th>#</th><th>id</th><th>outcome</th>'
         '<th>miss</th><th>path</th><th>SPL</th><th>nDTW</th></tr></thead>' + "".join(rows) + '</table></div></div>')

setup = f"""<dl class="kv">
<dt>House</dt><dd>Matterport3D <code>zsNo4HB9uLZ</code>, from the R2R-CE <b>val_unseen</b> split: houses the model never saw in training.</dd>
<dt>Episodes</dt><dd>{n} real R2R-CE episodes sampled (fixed seed) from the {n_house} that take place in this house. Human-written instructions, no edits.</dd>
<dt>Model</dt><dd>Released ETPNav R2R checkpoint, unmodified. Runs on one GPU at about 5 s per episode.</dd>
<dt>Map</dt><dd>Rebuilt from nothing at the start of every episode. This is how ETPNav ships.</dd>
<dt>Success</dt><dd>The agent stops within 3 m of the goal, measured along the floor.</dd>
</dl>"""

readout = f"""<dl class="readout">
<div><dt>Reached goal</dt><dd>{k}<small>of {n}</small></dd></div>
<div><dt>Success rate</dt><dd>{100 * k / n:.0f}<small>%</small></dd></div>
<div><dt>SPL</dt><dd>{spl:.2f}</dd></div>
<div><dt>Mean miss</dt><dd>{ne:.1f}<small>m</small></dd></div>
<div><dt>nDTW</dt><dd>{ndtw:.2f}</dd></div></dl>"""

lead = (f"With its map wiped between episodes, stock ETPNav reached the goal in <b>{k} of {n}</b> episodes in this one house. "
        f"That is the number every later experiment has to beat, or at least match.")

sections = [
    ("Setup", setup),
    ("Result", readout + f'<p class="dim" style="margin-top:12px;max-width:66ch">SPL rewards short routes, so it sits below the success rate. '
                         f'nDTW scores how closely the path follows the human reference route.</p>'),
    ("How far each episode ended from its goal",
     f'<div class="panel"><h3>Final distance to goal</h3><p class="sub">Teal bars reached the goal; red bars missed. Hover a bar for the instruction.</p>{chart}'
     f'<div class="legend"><span><i class="sw ok"></i>reached (within 3 m)</span><span><i class="sw miss"></i>missed</span></div></div>'),
    ("Reading it", f"""<div class="prose"><p>{len(near)} of the {n - k} misses ended within 4 m of the goal, so they were near-misses that stopped a little early
or a little far. The other {n - k - len(near)} were real wrong turns, ending 8 m or more away.</p></div>"""),
    ("Caveats", f"""<div class="prose"><ul>
<li><b>Small sample.</b> {n} episodes in one house. The 95% interval on {100 * k / n:.0f}% is {100 * lo:.0f}–{100 * hi:.0f}%, which is wide.
It agrees with the paper's 57% on all of val_unseen and with our earlier 50-episode run (68%), but it can't distinguish them.</li>
<li><b>Order.</b> The dataset loader shuffles episodes, so this run went through them in a random order. Each episode's result is identical to the
ordered re-run in Experiment 2, so the chart shows tour order, the order later experiments use.</li>
<li><b>Why one house.</b> Persisting a map only makes sense inside one house. Every episode here shares the same floor plan.</li></ul></div>"""),
    ("Every episode", table),
]
repro = ("Reproduce: <code>python scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --mode none --out docs/experiments/exp1_baseline/baseline</code> "
         "· data: <code>docs/experiments/exp1_baseline/</code> in the repo.")
out = L.page("Baseline in One House", 1, [1], lead, sections, L.TIP_JS, repro)
open(sys.argv[1] if len(sys.argv) > 1 else "exp1.html", "w").write(out)
print(f"k={k}/{n} spl={spl:.3f} ne={ne:.2f} ndtw={ndtw:.3f} ci=({lo:.2f},{hi:.2f}) near={len(near)} house_eps={n_house} bytes={len(out)}")
