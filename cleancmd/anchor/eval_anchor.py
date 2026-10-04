"""
Object-picking benchmark: which instance does "next to the sofa (in the living room / near the window)" mean?

Inputs (this directory): <scene>_anchor_cmds.jsonl (make_anchor_cmds.py) and <scene>_detections.json
(vlmaps_cc/application/extract_anchor_detections.py: VLMaps' own islands per category, in world x/z and VLMaps' grid).
Every method picks one VLMaps detection of the command's category (those VLMaps' size filter keeps, as
move_to_object uses) or answers "ask":

  A vlmaps_front   VLMaps' rule: Map.select_front_objs (+-45 deg of the heading, on VLMaps' grid), then the nearest
                   (Map.select_nearest_obj: distance to the island's bbox). Nothing in front -> ask.
  B vlmaps_nearest the same nearest rule over every detection, no front filter.
  C room_first     room = the one named in the command, else the robot's current room; keep detections whose centre
                   lies in that room (MP3D region polygons); nearest of those; none -> ask.
  G c_gated        C when the command names a room, otherwise B.
  D llm            an LLM gets the command and a numbered candidate list (category, room, distance, direction,
                   what is near it) and answers a number or "ask". Prompt fixed below, temperature 0, seed 0.

Score: a pick is correct when the chosen detection's centre is within 1 m of the annotated instance's centre.
Every non-correct row is attributed to: missed (no kept detection within 1 m of the instance, so no method could
pick it), wrong (it was detected but the method picked another) or ask (it was detected and the method asked).

python cleancmd/anchor/eval_anchor.py --scenes zsNo4HB9uLZ --methods A,B,C,D
"""
import argparse
import hashlib
import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from make_anchor_cmds import SCENE_DIR, parse_house, region_of, say

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CORRECT_M = 1.0
NEAR_M = 1.5           # what counts as "near" a candidate in D's description
METHODS = {"A": "vlmaps_front", "B": "vlmaps_nearest", "C": "room_first", "G": "c_gated", "D": "llm"}

# ---- method D: fixed before any run ----
LLM_MODEL = os.environ.get("ANCHOR_LLM", "qwen3.5:latest")
LLM_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
LLM_OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 8, "num_ctx": 2048}
PROMPT = """You are a home robot. The user said: "{command}"
You are in the {robot_room}.
Objects in your map that could be what the user means (number. object, room, distance and direction from you, what is near it):
{candidates}
{memory}
Which object does the user mean? Reply with only its number. If the command does not let you tell which one, reply with only the word ask."""


def load_scene(scene, detector="vlmaps"):
    """detector: "vlmaps" (VLMaps' argmax islands, <scene>_detections.json) or "ov" (open-vocabulary,
    ov/<scene>_ov_detections.json from ov_detect.py)."""
    rows = [json.loads(l) for l in open(HERE / f"{scene}_anchor_cmds.jsonl")]
    det = json.load(open(HERE / f"{scene}_detections.json" if detector == "vlmaps"
                         else HERE / "ov" / f"{scene}_ov_detections.json"))
    house = scene.split("_")[0]   # scene may be a dataset tag such as JmbYfDe2QKZ_2
    _, regions, _ = parse_house(SCENE_DIR / house / f"{house}.house")
    level = rows[0]["level"]
    room_cache = {}

    def room_of(xz):
        key = (round(xz[0], 2), round(xz[1], 2))
        if key not in room_cache:
            rid = region_of(xz, regions, level)
            room_cache[key] = (rid, regions[rid]["name"] if rid is not None else None)
        return room_cache[key]

    for cat, ds in det["detections"].items():
        for d in ds:
            d["category"] = cat
            d["region"], d["room"] = room_of(d["center_xz"])
    return {"scene": scene, "rows": rows, "det": det, "regions": regions, "room_of": room_of}


def candidates(S, cat):
    return [d for d in S["det"]["detections"].get(cat, []) if d["size_ok"]]


def dist_to_bbox(rc, bbox):
    """Map.select_nearest_obj's measure: distance from the robot cell to the island's bbox (0 inside)."""
    r, c = rc
    rmin, rmax, cmin, cmax = bbox
    dr = max(rmin - r, 0, r - rmax)
    dc = max(cmin - c, 0, c - cmax)
    return math.hypot(dr, dc)


def front_ids(cands, pose):
    """Map.select_front_objs (fov 90) on VLMaps' grid, same angle convention and wrap handling."""
    row_org, col_org, ang = pose
    theta, half, pi_2 = ang * np.pi / 180, np.pi / 4, np.pi / 2
    out = []
    for i, d in enumerate(cands):
        row, col = d["center_rc"]
        a = np.arctan2(-col + col_org, -row + row_org)
        if (abs(a - theta) < half or (theta > pi_2 and a < -pi_2 and abs(2 * np.pi - theta + a) < half)
                or (theta < -pi_2 and a > pi_2 and abs(2 * np.pi - a + theta) < half)):
            out.append(i)
    return out


def nearest(cands, pose):
    return min(cands, key=lambda d: dist_to_bbox(pose[:2], d["bbox_rc"])) if cands else None


def bearing(row, target_xz):
    """Direction word of target_xz seen from the start pose (yaw about +y, forward -z)."""
    p, yaw = row["start"]["position"], math.radians(row["start"]["yaw_deg"])
    vx, vz = target_xz[0] - p[0], target_xz[1] - p[2]
    fwd = vx * -math.sin(yaw) + vz * -math.cos(yaw)
    right = vx * math.cos(yaw) + vz * -math.sin(yaw)
    deg = math.degrees(math.atan2(-right, fwd))  # + is left
    words = ["ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right"]
    return words[int(((deg + 22.5) % 360) // 45)], math.hypot(vx, vz)


def near_words(S, d):
    out = []
    for cat, ds in S["det"]["detections"].items():
        if cat == d["category"]:
            continue
        dm = min((math.dist(d["center_xz"], o["center_xz"]) for o in ds if o["size_ok"]), default=99)
        if dm <= NEAR_M:
            out.append((dm, say(cat)))
    return [w for _, w in sorted(out)][:3]


def describe(S, row, cands):
    lines = []
    for i, d in enumerate(cands, 1):
        word, dist = bearing(row, d["center_xz"])
        near = near_words(S, d)
        lines.append(f"{i}. {say(d['category'])}, {d['room'] or 'unlabelled area'}, {dist:.1f} m {word}"
                     + (f", near {', '.join(near)}" if near else ""))
    return "\n".join(lines)


class LLM:
    def __init__(self, cache_path=HERE / "results" / "llm_cache.jsonl"):
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for l in open(cache_path):
                e = json.loads(l)
                self.cache[e["key"]] = e["reply"]
        self.calls = 0

    def __call__(self, prompt):
        key = hashlib.sha1(json.dumps([LLM_MODEL, LLM_OPTIONS, prompt]).encode()).hexdigest()
        if key not in self.cache:
            body = {"model": LLM_MODEL, "think": False, "stream": False, "options": LLM_OPTIONS,
                    "messages": [{"role": "user", "content": prompt}]}
            req = urllib.request.Request(f"http://{LLM_HOST}/api/chat", json.dumps(body).encode())
            reply = json.load(urllib.request.urlopen(req, timeout=600))["message"]["content"]
            self.cache[key] = reply
            self.calls += 1
            os.makedirs(self.cache_path.parent, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "model": LLM_MODEL, "prompt": prompt, "reply": reply}) + "\n")
        return self.cache[key]


def parse_reply(reply, n):
    m = re.search(r"\d+", reply)
    if m and 1 <= int(m.group()) <= n:
        return int(m.group()) - 1
    return None  # "ask", or anything unusable (counted as ask; raw reply kept in the prediction)


def run_llm(S, row, llm, cands=None, memory_text="", command=None):
    cands = candidates(S, row["category"]) if cands is None else cands
    if not cands:
        return None, {"reason": "no_candidates"}
    # numbered nearest first, so "1" is never an arbitrary map-order choice
    cands = sorted(cands, key=lambda d: bearing(row, d["center_xz"])[1])
    _, robot_room = S["room_of"]([row["start"]["position"][0], row["start"]["position"][2]])
    prompt = PROMPT.format(command=command or row["command"], robot_room=robot_room or "unlabelled area",
                           candidates=describe(S, row, cands), memory=memory_text)
    reply = llm(prompt)
    k = parse_reply(reply, len(cands))
    return (cands[k] if k is not None else None), {"reply": reply, "n_candidates": len(cands)}


def pick(S, row, method, llm=None):
    """(detection or None for ask, meta)."""
    cands = candidates(S, row["category"])
    pose = S["det"]["start_map_pose"][f"{row['command_id']}/{row['pose_idx']}"]
    if method == "A":
        ids = front_ids(cands, pose)
        return nearest([cands[i] for i in ids], pose), {"n_front": len(ids)}
    if method == "B":
        return nearest(cands, pose), {}
    if method == "C":
        if row["room_named"]:
            rids = [rid for rid, r in S["regions"].items() if r["name"] == row["room_named"] and r["level"] == row["level"]]
            rid = rids[0] if len(rids) == 1 else None
        else:
            rid = row["start"]["region"]
        if rid is None:
            return None, {"reason": "no_room"}
        inroom = [d for d in cands if d["region"] == rid]
        return nearest(inroom, pose), {"room_region": rid, "n_in_room": len(inroom)}
    if method == "G":
        return pick(S, row, "C" if row["room_named"] else "B", llm)
    if method == "D":
        return run_llm(S, row, llm)
    raise ValueError(method)


def detected(S, row):
    return any(math.dist(d["center_xz"], row["gt"]["center_xz"]) <= CORRECT_M for d in candidates(S, row["category"]))


def outcome(S, row, d):
    if d is not None and math.dist(d["center_xz"], row["gt"]["center_xz"]) <= CORRECT_M:
        return "correct"
    if not detected(S, row):
        return "missed"
    return "ask" if d is None else "wrong"


def table(preds, key):
    """Markdown rows: per (method, key) accuracy, ask rate, and failure attribution."""
    agg = defaultdict(lambda: defaultdict(int))
    for p in preds:
        for k in (p[key], "all"):
            a = agg[(p["method"], k)]
            a["n"] += 1
            a[p["outcome"]] += 1
            a["asked"] += p["pick"] is None
    keys = sorted({k for _, k in agg}, key=lambda k: (k == "all", k))
    out = [f"| method | {key} | n | accuracy | ask rate | missed det. | wrong choice | asked (target detected) |",
           "|---|---|---|---|---|---|---|---|"]
    for m in METHODS.values():
        for k in keys:
            a = agg.get((m, k))
            if not a:
                continue
            out.append(f"| {m} | {k} | {a['n']} | {a['correct'] / a['n']:.1%} | {a['asked'] / a['n']:.1%} | "
                       f"{a['missed']} | {a['wrong']} | {a['ask']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True, help="comma list, e.g. zsNo4HB9uLZ")
    ap.add_argument("--methods", default="A,B,C,D")
    ap.add_argument("--tag", default="")
    ap.add_argument("--detector", default="vlmaps", choices=["vlmaps", "ov"])
    args = ap.parse_args()
    llm = LLM() if "D" in args.methods else None
    os.makedirs(RESULTS, exist_ok=True)
    allp = []
    for scene in args.scenes.split(","):
        S = load_scene(scene, args.detector)
        for mk in args.methods.split(","):
            preds = []
            for row in S["rows"]:
                d, meta = pick(S, row, mk, llm)
                preds.append({"scene": scene, "command_id": row["command_id"], "pose_idx": row["pose_idx"],
                              "command": row["command"], "form": row["form"], "category": row["category"],
                              "method": METHODS[mk], "pick": d["det_id"] if d else None,
                              "pick_xz": d["center_xz"] if d else None, "gt_instance": row["gt"]["instance_id"],
                              "gt_xz": row["gt"]["center_xz"], "outcome": outcome(S, row, d), "meta": meta})
            suffix = "" if args.detector == "vlmaps" else "_ov"
            with open(RESULTS / f"{scene}_pred_{METHODS[mk]}{suffix}.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            allp += preds
            acc = sum(p["outcome"] == "correct" for p in preds) / len(preds)
            print(f"{scene} {METHODS[mk]}: {acc:.1%} of {len(preds)}" + (f" ({llm.calls} new LLM calls)" if llm else ""),
                  flush=True)
    by_form = table(allp, "form")
    by_scene = table(allp, "scene")
    print(by_form + "\n\n" + by_scene)
    with open(RESULTS / f"anchor_tables{args.tag}.md", "w") as f:
        f.write(f"# Object-picking benchmark ({args.scenes})\n\nCorrect = chosen detection centre within {CORRECT_M} m "
                f"of the annotated instance centre. LLM: {LLM_MODEL}, options {LLM_OPTIONS}.\n\n"
                f"## Per command form\n{by_form}\n\n## Per scene\n{by_scene}\n")


if __name__ == "__main__":
    main()
