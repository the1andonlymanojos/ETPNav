"""
Memory learning curve, controls: method G (C-gated) on the open-vocabulary detector, with and without memory, for
three simulated users who answer the robot's questions and corrections:

  consistent    always points to the intended instance (memory_anchor.py's user)
  inconsistent  30% of corrections point to a different annotated instance of the same category (same floor)
  random        every correction points to a random annotated instance of the category (may be the right one)

Protocol as memory_anchor.py: per (scene, seed) each pooled phrase has one intended instance; 5 days x 10 commands;
3 seeds. A correction is about the instance the user points to: they say "the one in the <room>" / "the one near
the <thing>" when that singles it out, else lead the robot to it; memory stores what was taught (right or not).
Scoring is always against the phrase's intended instance.

Logged per command: whether the intended instance is detected at all (some candidate within 1 m), so accuracy can
be split into all rows vs rows where the target was detected (picking separated from perception), and whether a
memory hit replays a spot whose intended instance was detected vs never detected.

python cleancmd/anchor/memory_users.py --scenes zsNo4HB9uLZ,... --detector ov
"""
import argparse
import json
import math
import random
from collections import defaultdict

from eval_anchor import CORRECT_M, NEAR_M, RESULTS, candidates, detected, load_scene, nearest, pick
from memory_anchor import DAY_CMDS, POOL, user_hint

USERS = ("consistent", "inconsistent", "random")
P_INCONSISTENT = 0.3
BASE = "G"


def choose(S, row, cands, hint):
    """Method G over `cands`, or, given a user hint, the nearest candidate that satisfies it (None = ask)."""
    if hint is None:
        d, _ = pick(dict(S, det=dict(S["det"], detections={row["category"]: cands})), row, BASE)
        return d
    _, kind, value = hint
    if kind == "room":
        keep = [d for d in cands if d["room"] == value]
    else:
        marks = [o for o in S["det"]["detections"].get(value, []) if o["size_ok"]]
        keep = [d for d in cands if any(math.dist(d["center_xz"], o["center_xz"]) <= NEAR_M for o in marks)]
    return nearest(keep, S["det"]["start_map_pose"][f"{row['command_id']}/{row['pose_idx']}"])


def run(S, seed, days, use_memory, user):
    rng = random.Random(f"{S['scene']}-{seed}")                 # same commands for every user and condition
    noise = random.Random(f"{S['scene']}-{seed}-{user}-noise")  # the user's own randomness
    by_text = defaultdict(lambda: defaultdict(list))
    instances = defaultdict(dict)  # category -> {instance_id: gt}
    for r in S["rows"]:
        by_text[r["command"]][r["gt"]["instance_id"]].append(r)
        instances[r["category"]][r["gt"]["instance_id"]] = r["gt"]
    texts = sorted(by_text)
    pool = rng.sample(texts, min(POOL, len(texts)))
    meant = {t: by_text[t][rng.choice(sorted(by_text[t]))] for t in pool}
    memory, log = {}, []
    for day in range(1, days + 1):
        for text in rng.sample(pool, min(DAY_CMDS, len(pool))):
            row = rng.choice(meant[text])
            target_detected = detected(S, row)
            entry = memory.get(text) if use_memory else None
            cands = candidates(S, row["category"])
            if entry and entry["not"]:
                cands = [d for d in cands if all(math.dist(d["center_xz"], x) > 0.5 for x in entry["not"])]
            source = "pick"
            if entry and entry["loc"]:
                near = [d for d in cands if math.dist(d["center_xz"], entry["loc"]) <= CORRECT_M]
                xz = min(near, key=lambda d: math.dist(d["center_xz"], entry["loc"]))["center_xz"] if near else entry["loc"]
                source = "memory"
            else:
                d = choose(S, row, cands, entry["hint"] if entry else None)
                xz = d["center_xz"] if d else None
            ok = xz is not None and math.dist(xz, row["gt"]["center_xz"]) <= CORRECT_M
            log.append({"scene": S["scene"], "seed": seed, "user": user, "memory": use_memory, "day": day,
                        "command": text, "source": source, "correct": ok, "asked": xz is None,
                        "wrong": xz is not None and not ok, "target_detected": target_detected})
            if ok or not use_memory:
                continue
            # which instance the user's correction is about
            others = [g for iid, g in instances[row["category"]].items() if iid != row["gt"]["instance_id"]]
            if user == "random":
                taught = noise.choice(list(instances[row["category"]].values()))
            elif user == "inconsistent" and others and noise.random() < P_INCONSISTENT:
                taught = noise.choice(others)
            else:
                taught = row["gt"]
            entry = memory.setdefault(text, {"loc": None, "not": [], "hint": None})
            if xz is not None:
                entry["not"].append(xz)
            entry["hint"] = user_hint(taught)
            if entry["hint"]:
                cands2 = [d for d in candidates(S, row["category"])
                          if all(math.dist(d["center_xz"], x) > 0.5 for x in entry["not"])]
                d2 = choose(S, row, cands2, entry["hint"])
                # the user confirms the retry when it is the instance they meant to point at
                if d2 is not None and math.dist(d2["center_xz"], taught["center_xz"]) <= CORRECT_M:
                    entry["loc"] = d2["center_xz"]
                    continue
                if d2 is not None:
                    entry["not"].append(d2["center_xz"])
            entry["loc"] = taught["center_xz"]  # the user leads the robot there
    return log


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--detector", default="ov", choices=["vlmaps", "ov"])
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    scenes = args.scenes.split(",")
    log = []
    for scene in scenes:
        S = load_scene(scene, args.detector)
        for seed in map(int, args.seeds.split(",")):
            for user in USERS:
                for use_memory in (False, True):
                    log += run(S, seed, args.days, use_memory, user)
    with open(RESULTS / "memory_users_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")

    # per (user, memory, day, seed); the table averages over seeds, the plot shows the seed range
    agg = defaultdict(list)
    for e in log:
        agg[(e["user"], e["memory"], e["day"], e["seed"])].append(e)
    summary = []
    for (user, mem, day, seed), es in sorted(agg.items()):
        det = [e for e in es if e["target_detected"]]
        hits = [e for e in es if e["source"] == "memory"]
        summary.append({"user": user, "memory": mem, "day": day, "seed": seed, "n": len(es),
                        "accuracy": mean([e["correct"] for e in es]),
                        "accuracy_detected": mean([e["correct"] for e in det]), "n_detected": len(det),
                        "questions_per_day": sum(e["asked"] for e in es) / len(scenes),
                        "corrections_per_day": sum(e["wrong"] for e in es) / len(scenes),
                        "memory_hits_detected": sum(e["target_detected"] for e in hits),
                        "memory_hits_never_detected": sum(not e["target_detected"] for e in hits),
                        "memory_hit_acc_detected": mean([e["correct"] for e in hits if e["target_detected"]]),
                        "memory_hit_acc_never_detected": mean([e["correct"] for e in hits if not e["target_detected"]])})
    with open(RESULTS / "memory_users_summary.json", "w") as f:
        json.dump({"scenes": scenes, "detector": args.detector, "base": BASE, "pool": POOL, "day_cmds": DAY_CMDS,
                   "p_inconsistent": P_INCONSISTENT, "rows": summary}, f, indent=1)

    def cell(user, mem, day, k, fmt):
        rs = [s for s in summary if s["user"] == user and s["memory"] == mem and s["day"] == day]
        return fmt(mean([s[k] for s in rs]))

    pct, num = (lambda v: f"{v:.1%}"), (lambda v: f"{v:.2f}")
    lines = ["| user | memory | day | accuracy | accuracy (target detected) | questions / day / scene | "
             "corrections / day / scene | memory hits: detected (acc) | memory hits: never detected (acc) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for user in USERS:
        for mem in (False, True):
            for day in range(1, args.days + 1):
                rs = [s for s in summary if s["user"] == user and s["memory"] == mem and s["day"] == day]
                hd, hn = sum(s["memory_hits_detected"] for s in rs), sum(s["memory_hits_never_detected"] for s in rs)
                hits = (f"{hd} ({cell(user, mem, day, 'memory_hit_acc_detected', pct)}) | "
                        f"{hn} ({cell(user, mem, day, 'memory_hit_acc_never_detected', pct)})") if mem else "- | -"
                lines.append(f"| {user} | {'with' if mem else 'without'} | {day} | {cell(user, mem, day, 'accuracy', pct)} | "
                             f"{cell(user, mem, day, 'accuracy_detected', pct)} | {cell(user, mem, day, 'questions_per_day', num)} | "
                             f"{cell(user, mem, day, 'corrections_per_day', num)} | {hits} |")
    print("\n".join(lines))
    with open(RESULTS / "memory_users_table.md", "w") as f:
        f.write(f"# Memory controls: method {BASE} on the {args.detector} detector, {len(scenes)} scenes, 3 seeds\n\n"
                "Memory-hit counts are summed over seeds; (acc) is their accuracy. nan = no such rows.\n\n"
                + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
