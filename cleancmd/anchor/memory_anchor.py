"""
Memory learning curve for the object-picking benchmark: method C (room-first) with and without a per-scene memory.
No LLM: the base picker is eval_anchor's method C, and the user's answers are applied as candidate filters.

Per (scene, seed): each distinct command phrase gets one intended instance (the household's convention: in this
home "next to the table" always means the same table), drawn at random from the instances that phrase was
generated for. POOL phrases are drawn; each "day" is DAY_CMDS of them (no repeats within a day, repeats across days),
each from one of its 3 start poses at random.

When the robot asks or picks wrong, a simulated user answers from the annotations:
  "the one in the <room>"      if no other instance of that category is in a room of that name, else
  "the one near the <thing>"   if the instance has a singling-out landmark (make_anchor_cmds' rule), else
  (leads the robot to it)      no words.
An answer is a filter on the candidates: "in the <room>" keeps detections in a room of that name, "near the
<thing>" keeps detections within NEAR_M of a detected <thing>; then the nearest one (VLMaps' measure).
With memory, the robot retries once with the answer (wrongly picked candidates removed); if that is right it
stores phrase -> that detection, otherwise phrase -> where the user led it. Every wrong pick also goes on the
phrase's not-this list. Before choosing, a remembered phrase is resolved from memory (nearest kept detection
within 1 m of the stored place, else the stored place itself); a phrase with only a hint / not-this list is
picked with the hint's filter over the remaining candidates.
Without memory nothing is kept, so every day is the plain method C.

Per day: accuracy = first-attempt picks within 1 m of the instance; questions = the robot answered "ask";
corrections = the robot picked wrong and the user had to say so.

python cleancmd/anchor/memory_anchor.py --scenes zsNo4HB9uLZ,... --days 5 --seeds 0,1,2
"""
import argparse
import json
import math
import random
from collections import defaultdict

from eval_anchor import CORRECT_M, NEAR_M, RESULTS, candidates, load_scene, nearest, pick
from make_anchor_cmds import say

POOL = 15
DAY_CMDS = 10


def user_hint(gt):
    """(what the user says, filter kind, filter value) or None when they can only lead the robot there."""
    if gt["room_distinct"]:
        return (f"the one in the {gt['room']}", "room", gt["room"])
    if gt["landmark"]:
        return (f"the one near the {say(gt['landmark'])}", "near", gt["landmark"])
    return None


def choose(S, row, cands, hint):
    """Method C over `cands`, or, given a user hint, the nearest candidate that satisfies it (None = ask)."""
    if hint is None:
        d, _ = pick(dict(S, det=dict(S["det"], detections={row["category"]: cands})), row, "C")
        return d
    _, kind, value = hint
    if kind == "room":
        keep = [d for d in cands if d["room"] == value]
    else:
        marks = [o for o in S["det"]["detections"].get(value, []) if o["size_ok"]]
        keep = [d for d in cands if any(math.dist(d["center_xz"], o["center_xz"]) <= NEAR_M for o in marks)]
    return nearest(keep, S["det"]["start_map_pose"][f"{row['command_id']}/{row['pose_idx']}"])


def is_correct(xz, row):
    return xz is not None and math.dist(xz, row["gt"]["center_xz"]) <= CORRECT_M


def run(S, seed, days, use_memory):
    rng = random.Random(f"{S['scene']}-{seed}")
    by_text = defaultdict(lambda: defaultdict(list))
    for r in S["rows"]:
        by_text[r["command"]][r["gt"]["instance_id"]].append(r)
    texts = sorted(by_text)
    pool = rng.sample(texts, min(POOL, len(texts)))
    meant = {t: by_text[t][rng.choice(sorted(by_text[t]))] for t in pool}   # phrase -> its 3 rows (one instance)
    memory = {}
    log = []
    for day in range(1, days + 1):
        for text in rng.sample(pool, min(DAY_CMDS, len(pool))):
            row = rng.choice(meant[text])
            entry = memory.get(text) if use_memory else None
            cands = candidates(S, row["category"])
            if entry and entry.get("not"):
                cands = [d for d in cands if all(math.dist(d["center_xz"], x) > 0.5 for x in entry["not"])]
            source = "pick"
            if entry and entry.get("loc"):
                near = [d for d in cands if math.dist(d["center_xz"], entry["loc"]) <= CORRECT_M]
                xz = min(near, key=lambda d: math.dist(d["center_xz"], entry["loc"]))["center_xz"] if near else entry["loc"]
                source = "memory_" + entry["how"]   # memory_detection | memory_led
            else:
                d = choose(S, row, cands, entry["hint"] if entry else None)
                xz = d["center_xz"] if d else None
            ok = is_correct(xz, row)
            log.append({"scene": S["scene"], "seed": seed, "memory": use_memory, "day": day, "command": text,
                        "pose_idx": row["pose_idx"], "source": source, "correct": ok, "asked": xz is None,
                        "wrong": xz is not None and not ok})
            if ok or not use_memory:
                continue
            # the user steps in; the robot learns from it
            entry = memory.setdefault(text, {"loc": None, "not": [], "hint": None, "how": None})
            if xz is not None:
                entry["not"].append(xz)
            entry["hint"] = user_hint(row["gt"])
            if entry["hint"]:
                cands2 = [d for d in candidates(S, row["category"])
                          if all(math.dist(d["center_xz"], x) > 0.5 for x in entry["not"])]
                d2 = choose(S, row, cands2, entry["hint"])
                if d2 is not None and is_correct(d2["center_xz"], row):
                    entry["loc"], entry["how"] = d2["center_xz"], "detection"
                    continue
                if d2 is not None:
                    entry["not"].append(d2["center_xz"])
            entry["loc"], entry["how"] = row["gt"]["center_xz"], "led"   # the user leads the robot there
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    log = []
    for scene in args.scenes.split(","):
        S = load_scene(scene)
        for seed in map(int, args.seeds.split(",")):
            for use_memory in (False, True):
                log += run(S, seed, args.days, use_memory)
    with open(RESULTS / "memory_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")

    # per (memory, day): mean over (scene, seed) runs, and the spread over seeds of the all-scene mean
    agg = defaultdict(lambda: defaultdict(list))
    for e in log:
        k = (e["memory"], e["day"], e["seed"])
        agg[k]["correct"].append(e["correct"])
        agg[k]["asked"].append(e["asked"])
        agg[k]["wrong"].append(e["wrong"])
    n_scenes = len(args.scenes.split(","))
    summary = []
    for (mem, day, seed), v in sorted(agg.items()):
        summary.append({"memory": mem, "day": day, "seed": seed, "accuracy": sum(v["correct"]) / len(v["correct"]),
                        "questions_per_day": sum(v["asked"]) / n_scenes, "corrections_per_day": sum(v["wrong"]) / n_scenes,
                        "n": len(v["correct"])})
    with open(RESULTS / "memory_summary.json", "w") as f:
        json.dump({"scenes": args.scenes.split(","), "pool": POOL, "day_cmds": DAY_CMDS, "rows": summary}, f, indent=1)
    lines = ["| memory | day | accuracy (mean over seeds) | questions / day / scene | corrections / day / scene |",
             "|---|---|---|---|---|"]
    for mem in (False, True):
        for day in range(1, args.days + 1):
            rs = [s for s in summary if s["memory"] == mem and s["day"] == day]
            m = lambda k: sum(s[k] for s in rs) / len(rs)
            lines.append(f"| {'with' if mem else 'without'} | {day} | {m('accuracy'):.1%} | {m('questions_per_day'):.2f} | "
                         f"{m('corrections_per_day'):.2f} |")
    print("\n".join(lines))
    with open(RESULTS / "memory_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
