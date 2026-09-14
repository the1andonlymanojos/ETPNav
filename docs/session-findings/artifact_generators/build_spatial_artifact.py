import json
import os
import statistics

RESULTS_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/spatial_results.json"
TEMPLATE_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/spatial_template.html"
OUT_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/etpnav_spatial.html"

with open(RESULTS_PATH) as f:
    results = json.load(f)

print("results:", len(results))

records = []
files_map = {}
skipped_large = []
for r in results:
    dist = r["stats"]["distance_to_goal"]
    if dist <= 1.5:
        outcome = "pass"
    elif dist <= 3.0:
        outcome = "close"
    else:
        outcome = "miss"

    size = os.path.getsize(r["video_path"])
    if size > 14 * 1024 * 1024:
        skipped_large.append((r["episode_id"], size))
        continue
    published_path = f"videos/ep{r['episode_id']}.mp4"
    files_map[published_path] = r["video_path"]

    records.append({
        **r,
        "outcome": outcome,
        "distance": dist,
        "video_path_published": published_path,
    })

print("records:", len(records), "skipped_large:", skipped_large)
print("total video bytes:", sum(os.path.getsize(f) for f in files_map.values()) / 1e6, "MB")

order = {"pass": 0, "close": 1, "miss": 2}
records.sort(key=lambda r: (order[r["outcome"]], r["category"], r["relation"]))

n = len(records)
n_pass = sum(1 for r in records if r["outcome"] == "pass")
n_close = sum(1 for r in records if r["outcome"] == "close")
n_miss = sum(1 for r in records if r["outcome"] == "miss")
avg_dist = statistics.mean(r["distance"] for r in records)
n_landmarks = len(set((r["category"], tuple(r["object_center"])) for r in records))

with open(TEMPLATE_PATH) as f:
    template = f.read()


def render_card(r):
    s = r["stats"]
    return f'''
    <article class="card {r['outcome']}" data-outcome="{r['outcome']}" data-relation="{r['relation']}">
      <div class="card-head">
        <span class="badge outcome {r['outcome']}">{r['outcome']}</span>
        <span class="badge relation">{r['relation']}</span>
        <span class="badge cat">{r['human_name']}</span>
      </div>
      <div class="card-instr">{r['instruction']}</div>
      <video controls preload="none">
        <source src="{r['video_path_published']}" type="video/mp4">
      </video>
      <dl class="readout">
        <div><dt>dist to target</dt><dd>{r['distance']:.2f}m</dd></div>
        <div><dt>steps</dt><dd>{int(s['steps_taken'])}</dd></div>
        <div><dt>path len</dt><dd>{s['path_length']:.1f}m</dd></div>
      </dl>
    </article>'''


cards_html = "\n".join(render_card(r) for r in records)

html = (template
        .replace("<!--CARDS-->", cards_html)
        .replace("<!--COUNT-->", str(n))
        .replace("<!--NLANDMARKS-->", str(n_landmarks))
        .replace("<!--N_PASS-->", str(n_pass))
        .replace("<!--TOTAL_N-->", str(n))
        .replace("<!--N_CLOSE-->", str(n_close))
        .replace("<!--N_MISS-->", str(n_miss))
        .replace("<!--AVG_DIST-->", f"{avg_dist:.2f}"))

with open(OUT_PATH, "w") as f:
    f.write(html)

print("wrote", OUT_PATH, os.path.getsize(OUT_PATH) / 1e6, "MB (index only, videos separate)")

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/spatial_files_map.json", "w") as f:
    json.dump(files_map, f, indent=2)
print("files_map entries:", len(files_map))

# quick summary by relation, for the chat reply
from collections import defaultdict
by_rel = defaultdict(lambda: [0, 0])
for r in records:
    by_rel[r["relation"]][0] += 1
    if r["outcome"] == "pass":
        by_rel[r["relation"]][1] += 1
print("by relation (n, n_pass):", dict(by_rel))
