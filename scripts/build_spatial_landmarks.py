#!/usr/bin/env python3
"""Find real landmark objects in an MP3D scene (via habitat-sim's semantic
scene graph) and compute start pose + geometric target zone for spatial
relation instructions ("left of the sofa", "behind the table", ...).

Heading convention (empirically verified against the simulator, not assumed):
  forward(theta) = (-sin(theta), -cos(theta))   in the world XZ plane
  quaternion (x,y,z,w) = (0, sin(theta/2), 0, cos(theta/2))
  right_vec = (-forward_z, forward_x)   given forward_vec = (forward_x, forward_z)
"""
import argparse
import json
import math

import habitat_sim
import numpy as np

LANDMARK_CATEGORIES = [
    "sofa", "table", "bed", "shelving", "counter",
    "chest_of_drawers", "tv_monitor",
]
HUMAN_NAME = {
    "sofa": "sofa",
    "table": "table",
    "bed": "bed",
    "shelving": "bookshelf",
    "counter": "counter",
    "chest_of_drawers": "dresser",
    "tv_monitor": "TV",
}

RELATIONS = ["left", "right", "behind", "next to"]

STANDOFF_BUFFER = 1.4      # extra clearance beyond the object's own radius, in meters
START_DIST_BUFFER = 2.2    # how far back the agent starts from the object's edge
SNAP_TOL = 1.2             # max allowed distance between a computed target and its navmesh snap
MIN_OBJ_SEPARATION = 1.5   # skip objects too close to an already-picked one (dedupe clutter)


def heading_to_quat(theta):
    return [0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)]


def forward_from_heading(theta):
    return np.array([-math.sin(theta), -math.cos(theta)])


def heading_from_forward(fx, fz):
    return math.atan2(-fx, -fz)


def right_from_forward(fx, fz):
    return np.array([-fz, fx])


def load_objects(scene_glb):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb
    backend_cfg.gpu_device_id = 0
    backend_cfg.load_semantic_mesh = True
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.resolution = [32, 32]
    agent_cfg.sensor_specifications = [sensor_spec]
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    objs = []
    for o in sim.semantic_scene.objects:
        if o is None or o.category is None:
            continue
        cat = o.category.name()
        if cat not in LANDMARK_CATEGORIES:
            continue
        c = np.array(o.aabb.center, dtype=np.float64)
        s = np.array(o.aabb.sizes, dtype=np.float64)
        objs.append({"category": cat, "center": c, "sizes": s})
    return sim, objs


def dedupe(objs):
    kept = []
    for o in objs:
        if any(np.linalg.norm(o["center"][[0, 2]] - k["center"][[0, 2]]) < MIN_OBJ_SEPARATION
               and o["category"] == k["category"] for k in kept):
            continue
        kept.append(o)
    return kept


def snap(sim, point_xz, y):
    p = np.array([point_xz[0], y, point_xz[1]], dtype=np.float32)
    snapped = sim.pathfinder.snap_point(p)
    return snapped


def find_start_pose(sim, obj_center_xz, obj_radius, floor_y):
    dist = obj_radius + START_DIST_BUFFER
    for angle_deg in range(0, 360, 30):
        a = math.radians(angle_deg)
        candidate_xz = obj_center_xz + dist * np.array([math.cos(a), math.sin(a)])
        snapped = snap(sim, candidate_xz, floor_y)
        if snapped is None:
            continue
        d = math.hypot(snapped[0] - candidate_xz[0], snapped[2] - candidate_xz[1])
        if d > 0.8:
            continue
        # face the object from here
        forward = obj_center_xz - np.array([snapped[0], snapped[2]])
        norm = np.linalg.norm(forward)
        if norm < 1e-3:
            continue
        forward = forward / norm
        theta = heading_from_forward(forward[0], forward[1])
        return snapped, theta, forward
    return None, None, None


def build_targets(obj_center_xz, forward, obj_radius, floor_y, sim):
    right = right_from_forward(forward[0], forward[1])
    offset = obj_radius + STANDOFF_BUFFER
    raw = {
        "left": obj_center_xz - right * offset,
        "right": obj_center_xz + right * offset,
        "behind": obj_center_xz + forward * offset,
        "next to": obj_center_xz + right * (obj_radius * 0.7 + 0.7),
    }
    out = {}
    for rel, xz in raw.items():
        snapped = snap(sim, xz, floor_y)
        if snapped is None:
            continue
        d = math.hypot(snapped[0] - xz[0], snapped[2] - xz[1])
        if d > SNAP_TOL:
            continue
        out[rel] = {"target": [float(snapped[0]), float(snapped[1]), float(snapped[2])],
                     "snap_error": d}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="e.g. mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-landmarks", type=int, default=8)
    args = parser.parse_args()

    scene_glb = f"data/scene_datasets/{args.scene}"
    sim, objs = load_objects(scene_glb)
    objs = dedupe(objs)
    print(f"found {len(objs)} deduped landmark candidates")

    # prefer variety of categories, largest objects first within each category
    by_cat = {}
    for o in objs:
        by_cat.setdefault(o["category"], []).append(o)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda o: -(o["sizes"][0] * o["sizes"][2]))

    chosen = []
    cats_cycle = list(by_cat.keys())
    i = 0
    while len(chosen) < args.max_landmarks and any(by_cat.values()):
        cat = cats_cycle[i % len(cats_cycle)]
        if by_cat[cat]:
            chosen.append(by_cat[cat].pop(0))
        i += 1
        if i > 200:
            break

    results = []
    for idx, o in enumerate(chosen):
        center_xz = o["center"][[0, 2]]
        radius = max(o["sizes"][0], o["sizes"][2]) / 2

        # Snap the object's own center to the navmesh to get a reliable LOCAL
        # floor height -- using an AABB-derived height guess instead can land
        # on a disconnected navmesh island (different floor/closet/stairwell),
        # which silently makes every target geodesically unreachable (inf).
        obj_snap = sim.pathfinder.snap_point(np.array(o["center"], dtype=np.float32))
        if obj_snap is None:
            print(f"  skip {o['category']} #{idx}: object center doesn't snap to navmesh")
            continue
        obj_snap_dist = math.hypot(obj_snap[0] - center_xz[0], obj_snap[2] - center_xz[1])
        if obj_snap_dist > 2.5:
            print(f"  skip {o['category']} #{idx}: nearest navmesh point is {obj_snap_dist:.1f}m away")
            continue
        floor_y = float(obj_snap[1])

        start, theta, forward = find_start_pose(sim, center_xz, radius, floor_y)
        if start is None:
            print(f"  skip {o['category']} #{idx}: no valid start pose found")
            continue
        targets = build_targets(center_xz, forward, radius, floor_y, sim)
        if not targets:
            print(f"  skip {o['category']} #{idx}: no valid targets")
            continue

        # Reject any target that isn't actually reachable from the start pose
        # on the navmesh graph (same building, but could be a separate
        # island -- e.g. across a gap the agent can't cross).
        reachable = {}
        for rel, tinfo in targets.items():
            path = habitat_sim.ShortestPath()
            path.requested_start = np.array(start, dtype=np.float32)
            path.requested_end = np.array(tinfo["target"], dtype=np.float32)
            found = sim.pathfinder.find_path(path)
            gd = path.geodesic_distance if found else float("inf")
            if math.isfinite(gd):
                tinfo["geodesic_distance"] = gd
                reachable[rel] = tinfo
            else:
                print(f"    drop {o['category']} #{idx} relation={rel}: unreachable (inf geodesic distance)")
        targets = reachable
        if not targets:
            print(f"  skip {o['category']} #{idx}: no reachable targets")
            continue

        results.append({
            "category": o["category"],
            "human_name": HUMAN_NAME[o["category"]],
            "object_center": o["center"].tolist(),
            "object_radius": radius,
            "start_position": [float(start[0]), float(start[1]), float(start[2])],
            "start_rotation": heading_to_quat(theta),
            "targets": targets,
        })
        print(f"  kept {o['category']} #{idx}: {sorted(targets.keys())}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} landmarks -> {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
