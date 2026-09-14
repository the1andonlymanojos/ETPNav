import json
import os

with open('/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/spatial_results.json') as f:
    orig = json.load(f)
with open('/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/hybrid_results.json') as f:
    hyb = json.load(f)

TEMPLATE_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/hybrid_template.html"
OUT_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/etpnav_hybrid.html"

files_map = {}
pairs = []
for o, h in zip(orig, hyb):
    before_path = f"videos/before{o['episode_id']}.mp4"
    after_path = f"videos/after{h['episode_id']}.mp4"
    files_map[before_path] = o["video_path"]
    files_map[after_path] = h["video_path"]
    pairs.append({
        "episode_id": o["episode_id"],
        "category": o["category"],
        "human_name": o["human_name"],
        "relation": o["relation"],
        "instruction": o["instruction"],
        "before_dist": o["stats"]["distance_to_goal"],
        "after_dist": h["final_distance"],
        "before_path": before_path,
        "after_path": after_path,
    })

total_before = sum(os.path.getsize(v) for k, v in files_map.items() if k.startswith("videos/before"))
total_after = sum(os.path.getsize(v) for k, v in files_map.items() if k.startswith("videos/after"))
print(f"before videos: {total_before/1e6:.2f} MB, after videos: {total_after/1e6:.2f} MB, total: {(total_before+total_after)/1e6:.2f} MB")

with open(TEMPLATE_PATH) as f:
    template = f.read()


def render_pair(p):
    return f'''
    <article class="pair">
      <div class="pair-head">
        <span class="badge">{p['relation']}</span>
        <span class="pair-instr">{p['instruction']}</span>
      </div>
      <div class="pair-cols">
        <div class="col before">
          <div class="col-label"><span>before (ETPNav)</span><span class="dist">{p['before_dist']:.2f}m</span></div>
          <video controls preload="none"><source src="{p['before_path']}" type="video/mp4"></video>
        </div>
        <div class="col after">
          <div class="col-label"><span>after (hybrid)</span><span class="dist">{p['after_dist']:.2f}m</span></div>
          <video controls preload="none"><source src="{p['after_path']}" type="video/mp4"></video>
        </div>
      </div>
    </article>'''


pairs.sort(key=lambda p: (p["relation"], p["category"]))
pairs_html = "\n".join(render_pair(p) for p in pairs)

html = template.replace("<!--PAIRS-->", pairs_html)
with open(OUT_PATH, "w") as f:
    f.write(html)
print("wrote", OUT_PATH)

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/hybrid_files_map.json", "w") as f:
    json.dump(files_map, f, indent=2)
print("files_map entries:", len(files_map))
