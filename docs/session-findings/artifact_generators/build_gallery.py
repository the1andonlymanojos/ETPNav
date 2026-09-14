import gzip
import json
import os
import statistics

REPO = "/home/storms-end/ETPNav"
EXP = "diverse50_rxr"
VIDEO_DIR = os.path.join(REPO, "data/logs/video", EXP)
STATS_DIR = os.path.join(REPO, "data/logs/eval_results", EXP)

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/selection50.json") as f:
    selection = {e["episode_id"]: e for e in json.load(f)}

stats_file = [f for f in os.listdir(STATS_DIR) if f.startswith("stats_ep_ckpt_")][0]
with open(os.path.join(STATS_DIR, stats_file)) as f:
    stats = json.load(f)

print("selection:", len(selection), "stats:", len(stats))
missing = set(selection.keys()) - set(stats.keys())
if missing:
    print("WARNING missing from stats:", missing)

records = []
files_map = {}
skipped_large = []

for eid, meta in selection.items():
    if eid not in stats:
        continue
    s = stats[eid]
    scene_short = meta["scene_id"].split("/")[1]
    mp4_candidates = [f for f in os.listdir(VIDEO_DIR) if f.startswith(f"{scene_short}-{eid}-") and f.endswith(".mp4")]
    if not mp4_candidates:
        print("no video for", eid)
        continue
    mp4_path = os.path.join(VIDEO_DIR, mp4_candidates[0])
    size = os.path.getsize(mp4_path)
    if size > 14 * 1024 * 1024:
        skipped_large.append((eid, size))
        continue
    published_path = f"videos/{scene_short}-{eid}.mp4"
    files_map[published_path] = mp4_path

    wc = meta["word_count"]
    length_bucket = "short" if wc <= 40 else ("medium" if wc <= 90 else "long")

    records.append({
        "episode_id": eid,
        "scene": scene_short,
        "language": meta["language"],
        "instruction": meta["instruction"],
        "word_count": wc,
        "length_bucket": length_bucket,
        "success": s["success"] == 1.0,
        "spl": s["spl"],
        "ndtw": s["ndtw"],
        "steps": s["steps_taken"],
        "path_length": s["path_length"],
        "video_path": published_path,
        "video_size": size,
    })

print("records:", len(records), "skipped_large:", skipped_large)
print("total video bytes:", sum(r["video_size"] for r in records) / 1e6, "MB")

records.sort(key=lambda r: (not r["success"], r["scene"], r["word_count"]))

n = len(records)
n_succ = sum(1 for r in records if r["success"])
avg_spl = statistics.mean(r["spl"] for r in records)
avg_ndtw = statistics.mean(r["ndtw"] for r in records)
avg_steps = statistics.mean(r["steps"] for r in records)
n_scenes = len(set(r["scene"] for r in records))

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/gallery_records.json", "w") as f:
    json.dump({
        "records": records,
        "n": n, "n_succ": n_succ, "avg_spl": avg_spl, "avg_ndtw": avg_ndtw,
        "avg_steps": avg_steps, "n_scenes": n_scenes,
    }, f, indent=2)

TEMPLATE_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/gallery_template.html"
OUT_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/etpnav_rxr_gallery.html"

with open(TEMPLATE_PATH) as f:
    template = f.read()


def render_card(r):
    outcome = "succ" if r["success"] else "fail"
    outcome_label = "success" if r["success"] else "fail"
    search_blob = (r["instruction"] + " " + r["scene"]).lower().replace('"', "&quot;")
    return f'''
    <article class="card {outcome}" data-outcome="{outcome}" data-len="{r['length_bucket']}" data-search="{search_blob}">
      <div class="card-head">
        <span class="badge outcome {outcome}">{outcome_label}</span>
        <span class="badge scene">{r['scene']}</span>
        <span class="badge len">{r['word_count']}w &middot; {r['length_bucket']}</span>
      </div>
      <div class="card-instr">{r['instruction']}</div>
      <video controls preload="none">
        <source src="{r['video_path']}" type="video/mp4">
      </video>
      <dl class="readout">
        <div><dt>steps</dt><dd>{int(r['steps'])}</dd></div>
        <div><dt>SPL</dt><dd>{r['spl']:.2f}</dd></div>
        <div><dt>nDTW</dt><dd>{r['ndtw']:.2f}</dd></div>
        <div><dt>path</dt><dd>{r['path_length']:.1f}m</dd></div>
      </dl>
    </article>'''


cards_html = "\n".join(render_card(r) for r in records)

html = (template
        .replace("<!--CARDS-->", cards_html)
        .replace("<!--COUNT-->", str(n))
        .replace("<!--NSCENES-->", str(n_scenes))
        .replace("<!--SUCC_N-->", str(n_succ))
        .replace("<!--TOTAL_N-->", str(n))
        .replace("<!--AVG_SPL-->", f"{avg_spl:.2f}")
        .replace("<!--AVG_NDTW-->", f"{avg_ndtw:.2f}")
        .replace("<!--AVG_STEPS-->", f"{avg_steps:.0f}"))

with open(OUT_PATH, "w") as f:
    f.write(html)

print("wrote", OUT_PATH, os.path.getsize(OUT_PATH) / 1e6, "MB (index only, videos separate)")

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/files_map.json", "w") as f:
    json.dump(files_map, f, indent=2)
print("files_map entries:", len(files_map))
