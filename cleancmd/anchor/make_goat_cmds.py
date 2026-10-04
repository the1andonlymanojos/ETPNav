"""
GOAT-Bench language goals as object-picking commands (same row format as make_anchor_cmds.py), for HM3D floors
that have a VLMaps dataset made by vlmaps_cc/dataset/generate_from_navmesh.py.

Ground truth is GOAT-Bench's own: each goal category lists every annotated instance (object_id, centre) and some
instances carry a language description. Kept: instances on this floor (centre 0.3 m below to 2.7 m above the floor)
whose category has >= 2 instances on this floor, so the description has to pick one. Each gives
  goat_lang  the GOAT description, verbatim (e.g. "white pillow near the clothes and iron board.")
  bare       "next to the <category>"
with N_POSES random start poses (navigable, this floor, near the dataset's poses), deduplicated over the
val_seen / val_seen_synonyms / val_unseen splits.

HM3D regions have ids but no room names; a point's region is the majority region of its 3 nearest annotated objects
on the floor (objects.json, written when the dataset was rendered), shown as "room #<id>".

python cleancmd/anchor/make_goat_cmds.py --dataset TEEsavR23oF_1
"""
import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from make_anchor_cmds import DATA_DIR, N_POSES, OUT_DIR, POSE_COVER_M

GOAT_DIR = Path.home() / "datasets/goat_bench/data/datasets/goat_bench/hm3d/v1"
HM3D_DIR = Path.home() / "datasets/minival"
SPLITS = ["val_seen", "val_seen_synonyms", "val_unseen"]
BELOW_M, ABOVE_M = 0.3, 2.7
STRUCTURE = {"wall", "floor", "ceiling", "door", "door frame", "window", "window frame", "stairs", "unknown", ""}


def on_floor(y, floor_y):
    return -BELOW_M <= y - floor_y <= ABOVE_M


def region_finder(objects, floor_y):
    objs = [o for o in objects if o["region"] is not None and on_floor(o["center"][1], floor_y)
            and o["category"].lower() not in STRUCTURE]
    xz = np.array([[o["center"][0], o["center"][2]] for o in objs])
    reg = [o["region"] for o in objs]

    def room_of(p):
        d = np.hypot(xz[:, 0] - p[0], xz[:, 1] - p[1])
        rid = Counter(reg[i] for i in np.argsort(d)[:3]).most_common(1)[0][0]
        return rid, f"room #{rid}"

    return room_of


def goat_goals(name):
    """{category: {object_id: goal}} merged over the val splits; goal['splits'] lists where it has a description."""
    out = defaultdict(dict)
    for sp in SPLITS:
        f = GOAT_DIR / sp / "content" / f"{name}.json.gz"
        if not f.exists():
            continue
        for insts in json.load(gzip.open(f))["goals"].values():
            for g in insts:
                e = out[g["object_category"]].setdefault(g["object_id"], {"position": g["position"], "descs": {}})
                if g.get("lang_desc"):
                    e["descs"].setdefault(g["lang_desc"].strip(), []).append(sp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. TEEsavR23oF_1 (made by generate_from_navmesh.py)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ddir = DATA_DIR / args.dataset
    src = json.load(open(ddir / "source.json"))
    name, floor_y = args.dataset.split("_")[0], src["floor_y"]
    objects = json.load(open(ddir / "objects.json"))
    room_of = region_finder(objects, floor_y)
    region_ids = {o["id"]: o["region"] for o in objects}
    rng = np.random.default_rng(args.seed)

    import habitat_sim  # PathFinder only

    pf = habitat_sim.nav.PathFinder()
    assert pf.load_nav_mesh(str(HM3D_DIR / src["hm3d"] / f"{name}.basis.navmesh"))
    pf.seed(args.seed)
    pose_xz = np.loadtxt(ddir / "poses.txt")[:, [0, 2]]

    def start_poses(n):
        out = []
        while len(out) < n:
            p = np.array(pf.get_random_navigable_point())
            if abs(p[1] - floor_y) > 0.5 or np.min(np.linalg.norm(pose_xz - p[[0, 2]], axis=1)) > POSE_COVER_M:
                continue
            rid, rname = room_of(p[[0, 2]])
            out.append({"position": p.round(3).tolist(), "yaw_deg": round(float(rng.uniform(-180, 180)), 1),
                        "region": rid, "room": rname})
        return out

    rows, k, kept = [], 0, Counter()
    for cat, insts in sorted(goat_goals(name).items()):
        here = {oid: g for oid, g in insts.items() if on_floor(g["position"][1], floor_y)}
        if len(here) < 2:
            continue
        for oid, g in sorted(here.items()):
            if not g["descs"]:
                continue
            x, y, z = g["position"]
            num = int(oid.split("_")[-1])
            rid = region_ids.get(num)
            gt = {"instance_id": oid, "category": cat, "center_xz": [round(x, 3), round(z, 3)], "center_y": round(y, 3),
                  "region": rid, "room": f"room #{rid}" if rid is not None else None, "room_distinct": False,
                  "landmark": None, "n_on_floor": len(here)}
            cmds = [("goat_lang", d, sps) for d, sps in sorted(g["descs"].items())] + [("bare", f"next to the {cat}", [])]
            for form, text, sps in cmds:
                k += 1
                kept[form] += 1
                for pi, st in enumerate(start_poses(N_POSES)):
                    rows.append({"scene": args.dataset, "dataset": args.dataset, "command_id": f"{name[:4]}{args.dataset[11:]}-g{k:04d}",
                                 "pose_idx": pi, "command": text, "form": form, "category": cat, "room_named": None,
                                 "landmark": None, "goat_splits": sps, "gt": gt, "start": st,
                                 "level": src["floor_index"], "floor_y": round(floor_y, 3)})
    out = OUT_DIR / f"{args.dataset}_anchor_cmds.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cats = Counter(r["category"] for r in rows if r["pose_idx"] == 0 and r["form"] == "goat_lang")
    print(f"{args.dataset}: commands {dict(kept)} x {N_POSES} poses = {len(rows)} rows | goat_lang by category {dict(cats)} -> {out}")


if __name__ == "__main__":
    main()
