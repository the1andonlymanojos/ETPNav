"""
Object-picking ("which one?") commands with ground truth from MP3D's own annotations (no human labels).

For every object category with >= 2 annotated instances on the floor the VLMaps dataset covers, each instance gets:
  bare      "next to the sofa"                       (always; ambiguous by construction)
  room      "next to the sofa in the living room"    (only when that room label occurs once in the scene and the
                                                      instance is the only one of its category in that room)
  landmark  "next to the table near the window"      (only when the instance is within LANDMARK_NEAR_M of a landmark
                                                      and every other instance of its category is >= LANDMARK_FAR_M away
                                                      from all landmarks of that category)
and every command gets N_POSES random start poses (navigable, same floor, inside the area the VLMaps poses cover).
One row per (command, start pose). Ground truth is the annotated instance (.house object index).

Coordinates are Habitat world (x, y up, z): .house (x, y, z-up) -> habitat (x, z, -y).

python cleancmd/anchor/make_anchor_cmds.py --scene zsNo4HB9uLZ_1        (vlmaps_3.9 env: needs habitat_sim's PathFinder,
                                                                         navmesh only, no renderer / GPU)
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

SCENE_DIR = Path.home() / "vlmaps_cc/mp3d_habitat/mp3d"
DATA_DIR = Path.home() / "vlmaps_dataset/vlmaps_dataset"
OUT_DIR = Path(__file__).resolve().parent

N_POSES = 3
LANDMARK_NEAR_M = 1.0
LANDMARK_FAR_M = 1.5
MIN_EXTENT_M = 0.25        # drop annotation fragments whose larger horizontal side is shorter than this
POSE_COVER_M = 1.0         # a start pose must be within this of some pose the VLMaps map was built from

# mpcat40 categories used as targets: furniture-scale and queryable by name in VLMaps' mp3dcat list.
# (cushion / picture / towel / clothes / lighting are small and numerous; wall, door, window etc. are structure.)
TARGETS = ["chair", "table", "sofa", "bed", "cabinet", "sink", "toilet", "tv_monitor", "counter", "stool", "plant",
           "chest_of_drawers", "bathtub", "fireplace", "shower", "seating", "shelving"]
LANDMARKS = TARGETS + ["window", "door", "picture", "mirror", "curtain"]
SAY = {"tv_monitor": "tv", "chest_of_drawers": "chest of drawers", "shelving": "shelf", "seating": "bench"}

# MP3D region label codes -> spoken room names (None: not a nameable room)
ROOMS = {"a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room", "e": "entryway", "f": "family room",
         "g": "garage", "h": "hallway", "i": "library", "j": "laundry room", "k": "kitchen", "l": "living room",
         "m": "meeting room", "n": "lounge", "o": "office", "p": "porch", "r": "game room", "s": None,
         "t": "half bath", "u": "utility room", "v": "tv room", "w": "gym", "x": None, "y": "balcony",
         "z": None, "B": "bar", "C": "classroom", "D": "dining booth", "S": "spa", "Z": None, "-": None}


def say(cat):
    return SAY.get(cat, cat)


def parse_house(path):
    cats, regions, levels, objects = {}, {}, {}, []
    surf_region, surf_verts = {}, defaultdict(list)
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "L":
            levels[int(t[1])] = float(t[9])            # level bbox lo z = floor height (house z)
        elif t[0] == "C":
            cats[int(t[1])] = t[5]                      # mpcat40 name
        elif t[0] == "R":
            regions[int(t[1])] = {"level": int(t[2]), "code": t[5], "zlo": float(t[11])}
        elif t[0] == "S":
            surf_region[int(t[1])] = int(t[2])
        elif t[0] == "V":
            surf_verts[int(t[2])].append((float(t[4]), -float(t[5])))
        elif t[0] == "O":
            f = list(map(float, t[4:19]))
            c, a0, a1, r = np.array(f[0:3]), np.array(f[3:6]), np.array(f[6:9]), np.array(f[9:12])
            # footprint: convex hull of the oriented box's 8 corners projected on the floor
            # (either box axis may be the vertical one; house xy -> habitat x, -y)
            axes = np.stack([a0, a1, np.cross(a0, a1)]) * r[:, None]
            corners = np.array([c + (2 * np.array(s) - 1) @ axes for s in np.ndindex(2, 2, 2)])
            hull = MultiPoint([(x, -y) for x, y in corners[:, :2]]).convex_hull
            minx, miny, maxx, maxy = hull.bounds
            fp = list(hull.exterior.coords)[:-1] if hull.geom_type == "Polygon" else [(c[0], -c[1])]
            objects.append({"id": int(t[1]), "region": int(t[2]), "cat_idx": int(t[3]),
                            "center": [c[0], c[2], -c[1]], "footprint": fp, "extent": float(max(maxx - minx, maxy - miny))})
    for o in objects:
        o["category"] = cats.get(o["cat_idx"], "")
    polys = defaultdict(list)
    for s, rid in surf_region.items():
        if len(surf_verts[s]) >= 3:
            p = Polygon(surf_verts[s])
            polys[rid].append(p if p.is_valid else p.buffer(0))
    for rid, r in regions.items():
        r["name"] = ROOMS.get(r["code"])
        r["polys"] = polys.get(rid, [])
    return objects, regions, levels


def region_of(xz, regions, level):
    """Region index containing point xz on `level` (smallest polygon wins), else the nearest one within 0.5 m."""
    p, best, dbest = Point(xz), None, 0.5
    inside = []
    for rid, r in regions.items():
        if r["level"] != level:
            continue
        for poly in r["polys"]:
            d = poly.distance(p)
            if d == 0:
                inside.append((poly.area, rid))
            elif d < dbest:
                best, dbest = rid, d
    return min(inside)[1] if inside else best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="e.g. zsNo4HB9uLZ_1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scene = args.scene.split("_")[0]
    # file / scene tag: the scene id, or the dataset name when the scene has several datasets (e.g. two floors)
    tag = scene if len(list(DATA_DIR.glob(f"{scene}_*"))) == 1 else args.scene
    rng = np.random.default_rng(args.seed)

    objects, regions, levels = parse_house(SCENE_DIR / scene / f"{scene}.house")

    # which level(s) the VLMaps dataset walks on: habitat y of its poses vs. each level's floor height
    poses = np.loadtxt(DATA_DIR / args.scene / "poses.txt")
    pose_y = np.median(poses[:, 1])
    # .house level bboxes overlap in height, so vote: the level of the region whose floor polygon contains a pose
    # and whose floor height matches it
    votes = Counter()
    for x, y, z in poses[::5, :3]:
        for r in regions.values():
            if abs(r["zlo"] - y) < 0.5 and any(p.distance(Point(x, z)) == 0 for p in r["polys"]):
                votes[r["level"]] += 1
                break
    level = votes.most_common(1)[0][0]
    floor_y = float(np.median([r["zlo"] for r in regions.values() if r["level"] == level]))
    pose_xz = poses[:, [0, 2]]

    import habitat_sim  # PathFinder only: loads the navmesh, no simulator / renderer

    pf = habitat_sim.nav.PathFinder()
    assert pf.load_nav_mesh(str(SCENE_DIR / scene / f"{scene}.navmesh"))
    pf.seed(args.seed)

    def room_of(xz):
        rid = region_of(xz, regions, level)
        return (rid, regions[rid]["name"]) if rid is not None else (None, None)

    # instances on this level, attributed to their annotated region
    inst = [o for o in objects if o["region"] in regions and regions[o["region"]]["level"] == level
            and o["category"] and o["extent"] >= MIN_EXTENT_M]
    by_cat = defaultdict(list)
    for o in inst:
        by_cat[o["category"]].append(o)
    room_name_count = Counter(r["name"] for r in regions.values() if r["level"] == level and r["name"])
    fps = {o["id"]: Polygon(o["footprint"]).buffer(0) if len(o["footprint"]) >= 3 else Point(o["footprint"][0]) for o in inst}

    def start_poses(n):
        out = []
        while len(out) < n:
            p = np.array(pf.get_random_navigable_point())
            if abs(p[1] - pose_y) > 0.5 or np.min(np.linalg.norm(pose_xz - p[[0, 2]], axis=1)) > POSE_COVER_M:
                continue
            yaw = float(rng.uniform(-180, 180))
            rid, rname = room_of(p[[0, 2]])
            out.append({"position": p.round(3).tolist(), "yaw_deg": round(yaw, 1), "region": rid, "room": rname})
        return out

    rows, summary, k = [], {}, 0
    for cat in TARGETS:
        group = by_cat.get(cat, [])
        if len(group) < 2:
            continue
        summary[cat] = len(group)
        per_room = Counter(o["region"] for o in group)
        for o in group:
            reg = regions[o["region"]]
            gt = {"instance_id": o["id"], "category": cat, "center_xz": [round(o["center"][0], 3), round(o["center"][2], 3)],
                  "center_y": round(o["center"][1], 3), "region": o["region"], "room": reg["name"],
                  "footprint_xz": np.round(o["footprint"], 3).tolist()}
            cmds = [("bare", f"next to the {say(cat)}", None, None)]
            if reg["name"] and room_name_count[reg["name"]] == 1 and per_room[o["region"]] == 1:
                cmds.append(("room", f"next to the {say(cat)} in the {reg['name']}", reg["name"], None))
            # landmark: nearest landmark category that singles this instance out
            best = None
            for lcat in LANDMARKS:
                if lcat == cat or not by_cat.get(lcat):
                    continue
                d = lambda a: min(fps[a["id"]].distance(fps[l["id"]]) for l in by_cat[lcat])
                dt = d(o)
                if dt > LANDMARK_NEAR_M:
                    continue
                if all(d(other) >= max(LANDMARK_FAR_M, dt + 1.0) for other in group if other is not o):
                    if best is None or dt < best[1]:
                        best = (lcat, dt)
            # what a simulated user can say to single this instance out (used by the memory study)
            gt["landmark"] = best[0] if best else None
            gt["room_distinct"] = bool(reg["name"]) and all(regions[x["region"]]["name"] != reg["name"]
                                                            for x in group if x is not o)
            if best:
                cmds.append(("landmark", f"next to the {say(cat)} near the {say(best[0])}", None, best[0]))
            for form, text, room, lm in cmds:
                k += 1
                for pi, st in enumerate(start_poses(N_POSES)):
                    rows.append({"scene": tag, "dataset": args.scene, "command_id": f"{tag[:4]}{tag[11:]}-a{k:04d}",
                                 "pose_idx": pi, "command": text, "form": form, "category": cat,
                                 "room_named": room, "landmark": lm, "gt": gt, "start": st, "level": level, "floor_y": round(floor_y, 3)})

    out = OUT_DIR / f"{tag}_anchor_cmds.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    forms = Counter(r["form"] for r in rows if r["pose_idx"] == 0)
    print(f"{tag}: level {level} | categories with >=2 instances: {summary} | commands {dict(forms)} "
          f"x {N_POSES} poses = {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
