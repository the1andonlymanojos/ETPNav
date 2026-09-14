import json
import os

RESULTS_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/phrasing_results.json"
TEMPLATE_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/phrasing_template.html"
OUT_PATH = "/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/etpnav_phrasing.html"

TIER_LABELS = {
    "bare": "bare command",
    "verb_stop": "verb + stop",
    "two_step": "two-step narrated",
    "anchored": "landmark-anchored",
    "narrated": "full narration",
}
TIER_ORDER = ["bare", "verb_stop", "two_step", "anchored", "narrated"]

with open(RESULTS_PATH) as f:
    results = json.load(f)
print("results:", len(results))

files_map = {}
by_case = {}
for r in results:
    dist = r["stats"]["distance_to_goal"]
    outcome = "pass" if dist <= 1.5 else ("close" if dist <= 3.0 else "miss")
    size = os.path.getsize(r["video_path"])
    published_path = f"videos/run{r['run_id']}.mp4"
    files_map[published_path] = r["video_path"]
    r["outcome"] = outcome
    r["distance"] = dist
    r["video_path_published"] = published_path
    by_case.setdefault(r["base_episode_id"], []).append(r)

print("total video bytes:", sum(os.path.getsize(f) for f in files_map.values()) / 1e6, "MB")

with open(TEMPLATE_PATH) as f:
    template = f.read()


def render_tier(r):
    s = r["stats"]
    return f'''
      <div class="tier-card {r['outcome']}">
        <div class="tier-head">
          <span class="tier-name">{TIER_LABELS[r['tier']]}</span>
          <span class="tier-outcome {r['outcome']}">{r['outcome']}</span>
        </div>
        <div class="tier-instr">{r['instruction']}</div>
        <video controls preload="none"><source src="{r['video_path_published']}" type="video/mp4"></video>
        <div class="tier-stats">
          <span>dist <b>{r['distance']:.2f}m</b></span>
          <span>steps <b>{int(s['steps_taken'])}</b></span>
        </div>
      </div>'''


def render_case(ep_id, runs):
    runs_by_tier = {r["tier"]: r for r in runs}
    first = runs[0]
    tiers_html = "\n".join(render_tier(runs_by_tier[t]) for t in TIER_ORDER if t in runs_by_tier)
    return f'''
  <section class="case">
    <div class="case-head">
      <span class="badge">{first['relation']}</span>
      <h2>{first['human_name']}</h2>
      <span class="orig-note">original miss: {first['orig_distance']:.2f}m{f", {first['orig_steps']} steps" if first.get('orig_steps') else ""}</span>
    </div>
    <div class="tiers">
      {tiers_html}
    </div>
  </section>'''


cases_html = "\n".join(render_case(ep_id, runs) for ep_id, runs in by_case.items())

# print a quick distance table for building the hypothesis text by hand
print("\ndistance by case/tier:")
for ep_id, runs in by_case.items():
    print(f"  {runs[0]['human_name']} ({runs[0]['relation']}):", {r["tier"]: round(r["distance"], 2) for r in runs})

html = template.replace("<!--CASES-->", cases_html)

with open(OUT_PATH, "w") as f:
    f.write(html)
print("\nwrote", OUT_PATH, "(hypothesis section still needs to be filled in by hand)")

with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/phrasing_files_map.json", "w") as f:
    json.dump(files_map, f, indent=2)
print("files_map entries:", len(files_map))
