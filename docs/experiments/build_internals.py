#!/usr/bin/env python3
"""'Inside ETPNav': a code-grounded walkthrough of how the agent turns an instruction into movement.
Worked example = the real first episode of the Exp 1/2 tour (map starts empty, so it is exactly the stock agent)."""
import gzip
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_lib as L  # noqa: E402
from floorplan import Floor  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EP = "1197"
dbg = json.load(open(f"{REPO}/docs/experiments/exp2_mechanics/window/persist_debug.json"))[EP]
res = json.load(open(f"{REPO}/docs/experiments/exp1_baseline/baseline/results.json"))["episodes"][EP]
ep = json.load(gzip.open(f"{REPO}/data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/tour_zsNo4H_n15_s0_o0/tour_zsNo4H_n15_s0_o0_bertidx.json.gz"))["episodes"][0]
assert str(ep["episode_id"]) == EP
start, goal = ep["start_position"], ep["goals"][0]["position"]
ref = ep["reference_path"]
nodes, ghosts, trace = dbg["nodes"], dbg["ghosts"], dbg["trace"]
n_dec = len(trace)

# ---------------- figure: the map one episode builds ----------------
fl = Floor("zsNo4HB9uLZ", crop=(-2.6, -8.6, 10.6, -0.6))
els = [f'<path d="{fl.path()}" fill="var(--floor)"/>']
els.append(f'<circle cx="{fl.px(goal[0]):.1f}" cy="{fl.pz(goal[2]):.1f}" r="120" fill="none" stroke="var(--ink)" stroke-width="1.3" stroke-dasharray="5 5" opacity=".5"/>')
els.append('<polyline points="' + " ".join(f"{fl.px(p[0]):.1f},{fl.pz(p[2]):.1f}" for p in ref) +
           '" fill="none" stroke="var(--dim)" stroke-width="2" stroke-dasharray="2 6" stroke-linecap="round"/>')
els.append('<polyline points="' + " ".join(f"{fl.px(x):.1f},{fl.pz(z):.1f}" for x, z, _ in nodes) +
           '" fill="none" stroke="var(--s1)" stroke-width="2.6" stroke-linejoin="round" opacity=".55"/>')
for gx, gz in ghosts:
    els.append(f'<circle cx="{fl.px(gx):.1f}" cy="{fl.pz(gz):.1f}" r="6" fill="none" stroke="var(--s2)" stroke-width="2"/>')
for i, (x, z, _) in enumerate(nodes):
    t = trace[i]
    tip = (f"<b>Decision {i + 1}</b><br><span class='m'>the planner saw {1 + t['active'] + t['ghosts_visible']} tokens here: "
           f"1 stop + {t['active']} visited node{'s' if t['active'] != 1 else ''} + {t['ghosts_visible']} frontier{'s' if t['ghosts_visible'] != 1 else ''}</span>")
    els.append(f'<circle data-tip="{html.escape(tip, quote=True)}" cx="{fl.px(x):.1f}" cy="{fl.pz(z):.1f}" r="12" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>'
               f'<text x="{fl.px(x):.1f}" y="{fl.pz(z) + 4.2:.1f}" font-size="12" font-weight="600" text-anchor="middle" style="fill:#fff" pointer-events="none">{i + 1}</text>')
sx, sz = fl.px(start[0]), fl.pz(start[2])
els.append(f'<text x="{sx + 22:.1f}" y="{sz + 5:.1f}" font-size="13" style="fill:var(--ink);font-weight:600">start</text>')
els.append(f'<text x="{fl.px(goal[0]):.1f}" y="{fl.pz(goal[2]) - 132:.1f}" font-size="13" text-anchor="middle" style="fill:var(--ink);font-weight:600">goal (3 m circle)</text>')
els.append(f'<rect x="{fl.px(goal[0]) - 7:.1f}" y="{fl.pz(goal[2]) - 7:.1f}" width="14" height="14" transform="rotate(45 {fl.px(goal[0]):.1f} {fl.pz(goal[2]):.1f})" fill="var(--ink)"/>')
fig_map = f'<svg viewBox="0 0 {fl.w:.0f} {fl.h:.0f}" width="100%" role="img" aria-label="The map built in one episode">{"".join(els)}</svg>'

decision_rows = "".join(
    f'<tr><td>{i + 1}</td><td>{t["active"]}</td><td>{t["ghosts_visible"]}</td><td><b>{1 + t["active"] + t["ghosts_visible"]}</b></td></tr>'
    for i, t in enumerate(trace))

# ---------------- pipeline overview (SVG) ----------------
def box(x, y, w, h, num, title, sub, kind):
    stroke = {"train": "var(--ok)", "frozen": "var(--dim)", "plain": "var(--rule)"}[kind]
    dash = ' stroke-dasharray="5 4"' if kind == "frozen" else ""
    sw = 2.2 if kind == "train" else 1.5
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="var(--surface)" stroke="{stroke}" stroke-width="{sw}"{dash}/>'
            f'<text x="{x + 14}" y="{y + 26}" font-size="15" font-weight="700" style="fill:var(--ok)">{num}</text>'
            f'<text x="{x + 36}" y="{y + 26}" font-size="13.5" font-weight="600" style="fill:var(--ink);font-family:\'IBM Plex Sans\',sans-serif">{title}</text>'
            f'<text x="{x + 14}" y="{y + 50}" font-size="10.8">{sub[0]}</text><text x="{x + 14}" y="{y + 66}" font-size="10.8">{sub[1]}</text>')

def arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="4 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--dim)" stroke-width="1.6"{d} marker-end="url(#ah)"/>'

BW, BH, GX = 196, 84, 30
xs = [10 + i * (BW + GX) for i in range(4)]
ya, yb = 46, 178
pipe = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 0L10 5L0 10z" fill="var(--dim)"/></marker></defs>',
        '<text x="10" y="24" font-size="11.5" style="fill:var(--dim)">one pass = one decision, at most 15 per episode</text>']
row1 = [("1", "Look around", ("12 RGB + 12 depth", "views, every 30°"), "plain"), ("2", "Encode views", ("CLIP ViT-B/32 (RGB)", "ResNet-50 (depth)"), "frozen"),
        ("3", "Propose waypoints", ("depth-only transformer", "up to 5 (angle, dist)"), "frozen"), ("4", "Panorama tokens", ("2-layer transformer,", "a 768-d token per view"), "train")]
row2 = [("5", "Update the map", ("new node + frontiers", "shortest paths redone"), "plain"), ("6", "Build planner input", ("stop, nodes, frontiers", "+ pairwise distances"), "plain"),
        ("7", "Plan", ("4-layer transformer", "reads the instruction"), "train"), ("8", "Act", ("walk to a frontier,", "or walk back and stop"), "plain")]
for i, (n, t, s, k) in enumerate(row1):
    pipe.append(box(xs[i], ya, BW, BH, n, t, s, k))
    if i < 3:
        pipe.append(arrow(xs[i] + BW + 3, ya + BH / 2, xs[i + 1] - 3, ya + BH / 2))
for i, (n, t, s, k) in enumerate(row2):
    pipe.append(box(xs[i], yb, BW, BH, n, t, s, k))
    if i < 3:
        pipe.append(arrow(xs[i] + BW + 3, yb + BH / 2, xs[i + 1] - 3, yb + BH / 2))
pipe.append(f'<path d="M{xs[3] + BW / 2} {yb - 2} V{ya + BH + 30} H{xs[3] + BW / 2}" fill="none"/>')
pipe.append(f'<path d="M{xs[3] + BW / 2} {ya + BH + 3} V{yb - 3}" stroke="var(--dim)" stroke-width="1.6" fill="none" marker-end="url(#ah)"/>')
pipe.append(f'<path d="M{xs[0] + BW / 2} {yb + BH + 3} V{yb + BH + 22} H{xs[0] - 6 + 0} " fill="none"/>')
# repeat arrow: box 8 bottom -> around -> box 1 (drawn along the left edge)
pipe.append(f'<path d="M{xs[3] + BW / 2} {yb + BH + 3} V{yb + BH + 20} H2 V{ya + BH / 2} H{xs[0] - 3}" fill="none" stroke="var(--dim)" stroke-width="1.6" stroke-dasharray="4 4" marker-end="url(#ah)"/>')
pipe.append(f'<text x="{xs[0] + 8}" y="{yb + BH + 38}" font-size="11.5" style="fill:var(--dim)">repeat until step 8 says stop</text>')
# instruction lane
yi = yb + BH + 62
pipe.append(box(xs[2] - 60, yi, BW + 120, 60, "0", "Encode the instruction (once)", ("9-layer BERT, up to 80 tokens", ""), "train").replace(f'y="{yi + 66}"', f'y="{yi + 200}"'))
pipe.append(f'<path d="M{xs[2] + BW / 2} {yi - 3} V{yb + BH + 4}" stroke="var(--dim)" stroke-width="1.6" fill="none" marker-end="url(#ah)"/>')
pipe_h = yi + 82
fig_pipe = (f'<svg viewBox="0 0 {xs[3] + BW + 12} {pipe_h}" width="100%" role="img" aria-label="ETPNav decision loop">{"".join(pipe)}</svg>'
            '<div class="legend"><span><i class="sw" style="border:2px solid var(--ok);background:transparent"></i>trained</span>'
            '<span><i class="sw" style="border:2px dashed var(--dim);background:transparent"></i>pretrained, frozen</span>'
            '<span><i class="sw" style="border:1.5px solid var(--rule);background:transparent"></i>no learned parameters</span></div>')

# ---------------- text ----------------
def step(n, title, io, body, ptr):
    return (f'<li id="s{n}"><div class="num">{n}</div><div><h3>{title}</h3><p class="io">{io}</p>{body}<p class="ptr">{ptr}</p></div></li>')

STEPS = "".join([
    step(1, "Look around", "<b>in</b> the agent's pose &nbsp;→&nbsp; <b>out</b> 12 RGB and 12 depth images",
         "<p>The simulator renders 12 virtual cameras around the agent, one every 30°, each with a 90° field of view. That is 12 RGB images (224×224) and 12 depth images (256×256). "
         "The agent never turns its body to look; the extra cameras are placed at the same spot.</p>",
         "<code>ss_trainer_ETP.py</code> adds the cameras via <code>get_camera_orientations12()</code> in <code>vlnce_baselines/utils.py</code>"),
    step(2, "Encode each view", "<b>in</b> 12 RGB + 12 depth &nbsp;→&nbsp; <b>out</b> one 512-d CLIP vector and one 128-d depth vector per view",
         "<p>RGB goes through the CLIP ViT-B/32 image encoder, and only its single global vector per view is kept. Depth goes through a ResNet-50 pretrained with DD-PPO. "
         "Both are frozen. Everything the planner will ever know about a place is 12 CLIP vectors and 12 depth summaries, so a view is not searchable for <i>where inside it</i> something is.</p>",
         "<code>Policy_ViewSelection_ETP.py</code> <code>CLIPEncoder</code>, <code>VlnResnetDepthEncoder</code>"),
    step(3, "Propose waypoints", "<b>in</b> 12 depth features &nbsp;→&nbsp; <b>out</b> up to 5 candidates, each an angle and a distance",
         "<p>A small 2-layer transformer looks at the 12 depth features, each view attending to its neighbours, and paints a heatmap over 120 headings (3° each) by 12 distances (0.25 m to 3.0 m). "
         "It uses <b>depth only</b>: <code>vis_x = depth_x</code> in the code. The heatmap is softmaxed, then non-maximum suppression picks the best cell, blanks out its neighbourhood "
         "(5 heading bins, 15°, each side) and repeats up to 5 times. These candidates are every option the agent will have at this decision.</p>"
         "<p>Trained separately and frozen here (<code>data/wp_pred/check_cwp_bestdist_hfov90</code> for R2R, <code>hfov63</code> for RxR).</p>",
         "<code>waypoint_pred/TRM_net.py</code> · <code>waypoint_pred/utils.py</code> <code>nms</code> · <code>Policy_ViewSelection_ETP.py</code> <code>mode='waypoint'</code>"),
    step(4, "Turn views into tokens", "<b>in</b> 12 view features + candidates &nbsp;→&nbsp; <b>out</b> one 768-d token per candidate and per remaining view",
         "<p>Each token is the sum of a linear projection of the CLIP vector, a projection of the depth vector, a projection of the view's direction (sin and cos of heading and elevation), "
         "a flag for whether a candidate points that way, and a token-type embedding. A 2-layer transformer then mixes the tokens.</p>"
         "<p>Two outputs matter. The tokens of the candidates become the <b>frontier embeddings</b>. The <b>mean of all tokens</b> becomes the embedding of the place the agent stands in. "
         "The instruction is never seen here, so these embeddings do not depend on it. That is why they stay valid from one episode to the next in the same house.</p>",
         "<code>vilmodel_cmt.py</code> <code>forward_panorama</code> · <code>ss_trainer_ETP.py</code> <code>_vp_feature_variable</code>"),
    step(5, "Update the map", "<b>in</b> position, candidates, tokens &nbsp;→&nbsp; <b>out</b> a graph with a new node and updated frontiers",
         "<p>A new <b>node</b> is created where the agent actually stands, holding the mean token and the decision number, and joined to the node it walked from. "
         "Each candidate is placed at <code>position − distance·(sin, cos)(heading + angle)</code>. This is an estimate: it is not checked against walls or the floor plan. Then for each candidate:</p>"
         "<ul><li>within 0.5 m of an existing node: no frontier is made, the current node is joined to that node instead;</li>"
         "<li>else within 0.5 m of an existing frontier: <b>merge</b> into it. Its position and embedding become the average, and the current node is added to its list of <i>fronts</i>;</li>"
         "<li>else: a new frontier is created.</li></ul>"
         "<p>Finally all-pairs shortest paths over the graph are recomputed.</p>",
         "<code>models/graph_utils.py</code> <code>GraphMap.identify_node</code>, <code>update_graph</code>"),
    step(6, "Build the planner's input", "<b>in</b> the graph &nbsp;→&nbsp; <b>out</b> a list of tokens plus a matrix of distances between them",
         "<p>The list is: one <b>stop</b> token (a zero image embedding), every <b>visited node</b>, every <b>frontier</b>. Each token is the sum of three things: the place's image embedding, "
         "a <b>step embedding</b> (the decision number for visited nodes, 0 for frontiers and stop), and a <b>position embedding</b> made from 7 numbers: sin and cos of the heading "
         "relative to where the agent faces, sin and cos of the elevation, straight-line distance ÷ 30, path distance ÷ 30, and number of graph hops ÷ 10. "
         "A frontier's path distance goes through its nearest front node.</p>",
         "<code>ss_trainer_ETP.py</code> <code>_nav_gmap_variable</code> · <code>graph_utils.py</code> <code>get_pos_fts</code>"),
    step(7, "Plan", "<b>in</b> instruction embeddings + graph tokens &nbsp;→&nbsp; <b>out</b> one score per token",
         "<p>Four identical layers. In each, the graph tokens first <b>cross-attend to the instruction</b>, then <b>attend to each other</b>, with a learned multiple of the pairwise path distance added to the attention scores, "
         "then pass through a feed-forward block. A small head turns each token into one logit. <b>Visited nodes and padding are then set to −∞</b>, so the only things that can win are "
         "an unexplored frontier or the stop token. Softmax, argmax.</p>"
         "<p>That mask is important: the planner scores old nodes but cannot choose them.</p>",
         "<code>vilmodel_cmt.py</code> <code>GraphLXRTXLayer</code>, <code>forward_navigation</code> (mask at lines 743–744)"),
    step(8, "Act", "<b>in</b> the chosen token &nbsp;→&nbsp; <b>out</b> movement in the simulator",
         "<p><b>A frontier was chosen.</b> The agent walks along the graph's shortest path to the frontier's nearest front node, node by node, then walks to the frontier's estimated position "
         "in 0.25 m steps with 15° turns. The frontier is deleted (\"consumed\"). Wherever the agent really ends up becomes the next node, which may be short of, or beside, the frontier.</p>"
         "<p><b>Stop was chosen</b>, or the 15th decision came, or no frontier is left. The agent walks back to the visited node whose stop probability was highest, "
         "measured when it stood there, and stops.</p>"
         "<p>A wall-following recovery (\"tryout\") exists but is disabled whenever <code>ALLOW_SLIDING</code> is true, as in our runs.</p>",
         "<code>ss_trainer_ETP.py</code> <code>rollout</code> · <code>common/environments.py</code> <code>single_step_control</code>, <code>step</code>"),
])

STATE = """<div class="prose"><p>There is <b>no recurrent hidden state</b>. What passes from one decision to the next is only:</p><ul>
<li>the <b>graph</b>: node positions, node embeddings, decision numbers, frontier positions/embeddings/fronts, and the path matrices;</li>
<li>the <b>stop probability</b> recorded at each visited node, used at the end to choose where to stop;</li>
<li>the <b>instruction embeddings</b>, computed once and reused at every decision;</li>
<li>which node the agent walked from, so the next node is joined to it.</li></ul>
<p>Nothing records which part of the instruction has been followed. The transformer has to infer progress from the map and the decision numbers.</p></div>"""

TOKENS = """<div class="panel"><div class="scroll"><table class="ref"><thead><tr><th>token</th><th>image embedding</th><th>step embedding</th><th>position (7 numbers)</th><th>selectable?</th></tr></thead><tbody>
<tr><td><b>stop</b> (first)</td><td>zeros</td><td>0</td><td>zeros</td><td>yes: means "stop"</td></tr>
<tr><td><b>visited node</b></td><td>mean of that place's 12 tokens</td><td>decision number (1, 2, …)</td><td>relative to the agent, from the graph</td><td>never, masked to −∞</td></tr>
<tr><td><b>frontier</b></td><td>mean of the candidate tokens that proposed it</td><td>0</td><td>relative to the agent, via its nearest front node</td><td>yes</td></tr></tbody></table></div></div>"""

CARRY_ROWS = [
    ("node_pos / node_embeds / node_stepId", "where each visited place is, its 768-d embedding, and the decision it was made at"),
    ("ghost_pos → ghost_mean_pos / ghost_embeds / ghost_fronts", "each frontier: all positions proposed for it, their average, its running embedding sum and count, and the nodes that saw it"),
    ("graph_nx", "the graph (a networkx object) whose edge weights are metres"),
    ("shortest_path / shortest_dist", "all-pairs paths, rebuilt every decision; used for the position features and to plan the walk"),
    ("node_stop_scores", "stop probability recorded at each node when the agent stood there"),
]
STATE_TBL = ('<div class="panel"><div class="scroll"><table class="ref"><thead><tr><th>field of <code>GraphMap</code></th><th>what it holds</th></tr></thead><tbody>'
             + "".join(f"<tr><td><code>{a}</code></td><td>{b}</td></tr>" for a, b in CARRY_ROWS) + "</tbody></table></div></div>")

TRAIN = """<div class="prose">
<p><b>Pretraining</b> (<code>pretrain_src/</code>). The cross-modal transformer is first pretrained on R2R-CE graph data. The code supports masked language modelling, masked region classification and single-step action prediction; the released checkpoint is named <code>mlm.sap_r2r</code>.</p>
<p><b>Fine-tuning</b> (this repo, trainer <code>SS-ETP</code>, scheduled sampling). The episode is rolled out in the simulator. At each decision the executed action is the teacher's with probability <code>0.75^(iter // 3000 + 1)</code>, otherwise the model's own sample. The loss is cross-entropy between the model's scores and the teacher's choice, summed over decisions.</p>
<p><b>The teacher</b> says stop if the agent is within 1.5 m of the goal, otherwise it picks the frontier whose true simulator position is closest to the goal by walking distance. That needs true frontier positions (<code>ghost_real_pos</code>), so they exist only in training and in video rendering. An alternative teacher follows the reference path (<code>expert_policy: ndtw</code>).</p>
<p><b>Frozen:</b> CLIP, the depth encoder and the waypoint predictor. <b>Trained:</b> the language encoder, the panorama transformer and the graph planner. AdamW, learning rate 1e-5, 15,000 iterations.</p></div>"""

NUMS = [
    ("Views per decision", "12 RGB (224²) and 12 depth (256²), 90° FOV, every 30°"),
    ("Candidates per decision", "at most 5, at 0.25–3.0 m, 120 heading bins of 3°"),
    ("Feature sizes", "CLIP 512 · depth 128 · angle 4 · hidden 768"),
    ("Language", "9-layer BERT, 80 tokens (R2R); XLM-R, 200 tokens (RxR)"),
    ("Panorama transformer", "2 layers"),
    ("Planner", "4 graph cross-modal layers, 768 hidden, learned distance bias"),
    ("Merge radius", "0.5 m (<code>loc_noise</code>) for both nodes and frontiers"),
    ("Episode limit", "15 decisions (<code>max_traj_len</code>)"),
    ("Motion", "0.25 m forward step, 15° turn"),
    ("Success", "stop within 3 m of goal"),
]
NUM_TBL = ('<div class="panel"><div class="scroll"><table class="ref"><tbody>' + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in NUMS) + "</tbody></table></div></div>")

IMPROVE = [
    ("Visual features", "CLIP ViT-B/32, one global vector per view", "Nothing inside a view is addressable, so \"left of the table\" is not representable. <i>(from code)</i>",
     "Patch-level or region features, a stronger backbone, or detected-object tokens", "Panorama transformer and planner"),
    ("Waypoint predictor", "Depth only, ≤5 candidates, 0.25–3 m, frozen", "Every option the agent ever has comes from here, and positions are open-loop estimates. <i>(from code)</i> A bad candidate set caps the whole system. <i>(my inference)</i>",
     "Measure the ceiling with oracle candidates first; add RGB; more or longer candidates; check candidates against depth", "The predictor itself, separately"),
    ("Node embedding", "Mean of the place's tokens", "Averaging discards which direction things are in. <i>(my inference)</i>",
     "Keep per-view tokens on the node; add room or object labels", "Planner"),
    ("Memory across episodes", "Map wiped each episode; only this episode's nodes can be stop targets", "Our Experiments 2 and 3: naive persistence hurts, and the planner cannot pick old nodes anyway.",
     "Let old nodes be selectable targets; retrieve instead of dump", "Possibly none, to test zero-shot"),
    ("Planner decisions", "Greedy argmax over frontiers + stop; no lookahead", "One wrong frontier costs a whole detour, and nothing tracks instruction progress. <i>(from code; the second half is my inference)</i>",
     "Beam or lookahead over frontiers; a progress signal", "Some"),
    ("Stop rule", "Stop at the node whose stop probability, recorded at the time, was highest", "A node's score reflects only what was known while standing there. <i>(from code)</i>",
     "Re-score with the final map; calibrate", "None to test"),
    ("Instruction", "BERT-base, 80 tokens, one embedding sequence", "Long instructions are cut at 80 tokens. <i>(from config)</i>",
     "Split into sub-goals (for example with an LLM) and feed them in order", "Depends"),
    ("Controller", "Heuristic walker; recovery off when sliding is on", "Collisions and stuck cases are not learned. <i>(from code)</i>", "A better recovery policy", "None"),
    ("Training signal", "Nearest-frontier-to-goal teacher, scheduled sampling", "The teacher optimises reaching the goal, not following the instruction's route. <i>(from code)</i>",
     "The nDTW teacher that already exists; DAgger-style data; RL fine-tuning", "Yes"),
]
IMP_TBL = ('<div class="panel"><div class="scroll"><table class="ref wide"><thead><tr><th>part</th><th>today</th><th>limit</th><th>idea</th><th>retrain?</th></tr></thead><tbody>'
           + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in IMPROVE) + "</tbody></table></div></div>")

WHERE = [
    ("Episode loop, one decision step, action execution", "vlnce_baselines/ss_trainer_ETP.py", "rollout"),
    ("Planner input assembly", "vlnce_baselines/ss_trainer_ETP.py", "_nav_gmap_variable"),
    ("Encoders, waypoint call, panorama/navigation modes", "vlnce_baselines/models/Policy_ViewSelection_ETP.py", "ETP.forward"),
    ("Panorama transformer and graph planner", "vlnce_baselines/models/etp/vilmodel_cmt.py", "forward_panorama, forward_navigation"),
    ("Waypoint predictor", "vlnce_baselines/waypoint_pred/TRM_net.py", "BinaryDistPredictor_TRM"),
    ("The map", "vlnce_baselines/models/graph_utils.py", "GraphMap"),
    ("Simulator control", "vlnce_baselines/common/environments.py", "single_step_control, step"),
    ("Config defaults", "run_r2r/iter_train.yaml, vlnce_baselines/config/default.py", ""),
    ("Pretraining", "pretrain_src/", ""),
]
WHERE_TBL = ('<div class="panel"><div class="scroll"><table class="ref"><thead><tr><th>what</th><th>file</th><th>look for</th></tr></thead><tbody>'
             + "".join(f"<tr><td>{a}</td><td><code>{b}</code></td><td>{'<code>' + c + '</code>' if c else ''}</td></tr>" for a, b, c in WHERE) + "</tbody></table></div></div>")

EXTRA_CSS = """
.toc{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0;padding:0;list-style:none;font:500 .76rem/1.3 "IBM Plex Mono",monospace}
.toc a{color:var(--ok);text-decoration:none;border-bottom:1px solid var(--ok-soft)} .toc a:hover{border-bottom-color:var(--ok)}
ol.steps{list-style:none;margin:0;padding:0}
.prose ul+p{margin-top:.8em}
ol.steps>li{display:grid;grid-template-columns:40px 1fr;gap:0 14px;padding:20px 0;border-top:1px solid var(--rule)}
ol.steps>li:first-child{border-top:0;padding-top:4px}
ol.steps .num{font:600 1.7rem/1 "IBM Plex Mono",monospace;color:var(--ok)}
ol.steps h3{margin:0 0 4px;font:650 1.08rem/1.3 "IBM Plex Sans",sans-serif}
ol.steps .io{font:400 .8rem/1.5 "IBM Plex Mono",monospace;color:var(--dim);margin:0 0 9px} ol.steps .io b{color:var(--ink);font-weight:600}
ol.steps li>div>p:not(.io):not(.ptr){max-width:68ch} ol.steps li p+p{margin-top:.6em} ol.steps ul{margin:.5em 0 .5em;padding-left:1.15em;max-width:66ch} ol.steps li li+li{margin-top:.3em}
ol.steps .ptr{margin-top:10px;font-size:.78rem;color:var(--dim)}
@media (max-width:560px){ol.steps>li{grid-template-columns:1fr}}
table.ref{width:100%;border-collapse:collapse;font-size:.86rem}
table.ref th{font:500 .66rem/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);text-align:left;padding:0 12px 9px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
table.ref td{padding:9px 12px 9px 0;vertical-align:top;border-bottom:1px solid var(--soft)} table.ref tbody tr:last-child td{border-bottom:0}
table.ref.wide{min-width:820px} table.ref.wide td{font-size:.82rem;line-height:1.45}
.mini{width:100%;max-width:340px;border-collapse:collapse;font:400 .82rem "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.mini th{font-weight:500;font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);text-align:right;padding:0 10px 6px 0;border-bottom:1px solid var(--rule)}
.mini td{text-align:right;padding:5px 10px 5px 0;border-bottom:1px solid var(--soft)}
.ex{display:grid;grid-template-columns:minmax(0,1fr);gap:18px}
.ex .instr{font-size:.92rem;max-width:66ch;color:var(--ink)}
.ring2{display:inline-block;width:11px;height:11px;border-radius:50%;border:2px solid var(--s2)}
"""

toc = ('<ul class="toc"><li><a href="#big">the loop</a></li><li><a href="#steps">one decision, step by step</a></li><li><a href="#state">what carries over</a></li>'
       '<li><a href="#example">a real episode</a></li><li><a href="#train">training</a></li><li><a href="#nums">numbers</a></li>'
       '<li><a href="#improve">where to improve it</a></li><li><a href="#where">where it lives</a></li></ul>')

lead = ("ETPNav navigates by <b>building a map as it walks</b>. At every decision it looks around, proposes a few places it could go, stores what it saw in a graph, "
        "and lets a transformer read the instruction against that graph to pick the next unexplored place, or stop. This page follows one decision from pixels to footsteps, "
        "then shows the map a real episode built.")

sections = [
    ("The loop", f'<div id="big" class="panel">{fig_pipe.replace("</svg>", "</svg>", 1)}</div>'
                 '<div class="prose" style="margin-top:14px"><p>Steps 1–4 are perception and produce candidate places. Steps 5–7 are where the map and the instruction meet, and step 7 is the main trained part. '
                 'Step 8 is a hand-written controller. Only steps 4 and 7 (and the instruction encoder) are trained here; the rest is frozen or plain code.</p></div>'),
    ("One decision, in order", f'<div id="steps"><ol class="steps">{STEPS}</ol></div>'),
    ("What carries from one decision to the next", f'<div id="state">{STATE}{STATE_TBL}</div>'),
    ("A real episode", f'''<div id="example" class="ex"><p class="instr"><b>Instruction.</b> “{html.escape(ep["instruction"]["instruction_text"].strip())}”</p>
<div class="panel"><h3>The map this episode built</h3>
<p class="sub">Episode 1 of the tour, so the map started empty and this is the stock agent. It made {n_dec} decisions ({n_dec - 1} frontier moves, then stop) and ended {res["distance_to_goal"]:.1f} m from the goal, which counts as reached.
Hover a node to see how many tokens the planner saw there.</p>{fig_map}
<div class="legend"><span><i class="sw s1"></i>visited node, numbered by decision</span><span><i class="ring2"></i>frontier never chosen</span>
<span><i class="sw" style="background:transparent;border-top:2px dotted var(--dim);height:0;width:18px;border-radius:0"></i>human reference route</span></div></div>
<div class="panel"><h3>How big the planner's input got</h3><p class="sub">Tokens = 1 stop + visited nodes + frontiers. Attention runs over all of them at every layer.</p>
<table class="mini"><thead><tr><th>decision</th><th>nodes</th><th>frontiers</th><th>tokens</th></tr></thead><tbody>{decision_rows}</tbody></table></div>
<p class="dim" style="max-width:66ch">The simulator took {res["steps_taken"]:.0f} low-level steps (turns and 0.25 m moves) for these {n_dec} decisions, about {res["steps_taken"] / n_dec:.0f} per decision, and the path was {res["path_length"]:.1f} m long.
Which frontier was chosen at each decision is not recorded by the stock code; the logging added for Experiment 3 will show it.</p></div>'''),
    ("Training", f'<div id="train">{TRAIN}</div>'),
    ("Numbers to keep at hand", f'<div id="nums">{NUM_TBL}</div>'),
    ("Where to improve it", '<div id="improve"><div class="prose" style="margin-bottom:12px"><p>Each row says what the part does now, what limits it, an idea, and whether it needs retraining. '
                            'Limits marked <i>from code</i> I read directly; those marked <i>my inference</i> are judgement and should be tested before being trusted.</p></div>' + IMP_TBL + '</div>'),
    ("Where it lives in the code", f'<div id="where">{WHERE_TBL}</div>'),
]
body = "".join(f'<section><h2>{html.escape(h)}</h2>{inner}</section>' for h, inner in sections)
out = (f'<title>Inside ETPNav</title>{L.FONTS}<style>{L.CSS}{EXTRA_CSS}</style>'
       f'<div class="page"><header><div class="eyebrow">ETPNav · how it works</div><h1 style="margin-top:14px">Inside ETPNav</h1></header>{toc}'
       f'<p class="lead">{lead}</p>{body}'
       f'<div class="foot">Read from the working tree of this repo. Line numbers are omitted except where stable; search the function names above. '
       f'Persisted-map code from our experiments lives in <code>graph_utils.py</code> and <code>ss_trainer_ETP.py</code> too, but the walkthrough describes the stock behaviour.</div></div>'
       f'<div class="tip" id="tip" role="tooltip"></div><script>{L.TIP_JS}</script>')
open(sys.argv[1] if len(sys.argv) > 1 else "internals.html", "w").write(out)
print("built", len(out), "bytes; decisions", n_dec, "tokens", [1 + t["active"] + t["ghosts_visible"] for t in trace])
