#!/usr/bin/env python3
"""Turn the 25 Phase 1 spatial probes on zsNo4HB9uLZ into a shared pilot for CleanCmd:

  cleancmd/pilot/zsNo4HB9uLZ_probes.jsonl    commands + start pose, read by every method's runner
  cleancmd/pilot/zsNo4HB9uLZ_gt_proxy.jsonl  PROXY ground truth: cells within 0.75 m of the annotation-derived target

The proxy ground truth comes from geometry, not people, and its left/right are relative to the robot's start view
(how Phase 1 computed them). It is a diagnostic split for wiring and first comparisons, never a headline result.
Plain Python + numpy; run from the repo root: python scripts/make_pilot_probes.py
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from cleancmd.grid import Grid  # noqa: E402
from cleancmd.io import POINT_RADIUS_M, save_jsonl  # noqa: E402

SCENE = "zsNo4HB9uLZ"
TARGET_RADIUS_M = POINT_RADIUS_M   # proxy region = what a point answer covers, so a perfect point answer scores 1
MARGIN_M = 4.0          # grid margin around every probe point


def main():
    res = json.load(open(os.path.join(REPO, "docs/session-findings/results/spatial_results.json")))
    lms = json.load(open(os.path.join(REPO, "docs/session-findings/results/landmarks_zsNo4HB9uLZ.json")))

    def rotation_for(p):
        for lm in lms:
            if lm["object_center"] == p["object_center"] and lm["start_position"] == p["start_position"]:
                return lm["start_rotation"]
        raise KeyError("no landmark for probe %d" % p["episode_id"])

    xs = [v for p in res for v in (p["start_position"][0], p["target_position"][0], p["object_center"][0])]
    zs = [v for p in res for v in (p["start_position"][2], p["target_position"][2], p["object_center"][2])]
    grid = Grid.from_bounds(min(xs) - MARGIN_M, min(zs) - MARGIN_M, max(xs) + MARGIN_M, max(zs) + MARGIN_M,
                            cell=0.5, floor_y=res[0]["start_position"][1])

    probes, gts = [], []
    for p in res:
        cid = "%s-p%02d" % (SCENE, p["episode_id"])
        rot = rotation_for(p)                       # habitat quaternion [x, y, z, w]
        yaw_deg = math.degrees(2 * math.atan2(rot[1], rot[3]))
        probes.append({"scene": SCENE, "command_id": cid, "command": p["instruction"],
                       "anchor_category": p["category"], "relation": p["relation"],
                       "start_position": p["start_position"], "start_rotation": rot, "start_yaw_deg": yaw_deg})

        tx, tz = p["target_position"][0], p["target_position"][2]
        ox, oz = p["object_center"][0], p["object_center"][2]
        d = math.hypot(tx - ox, tz - oz) or 1.0
        mask = sorted(grid.cells_within(tx, tz, TARGET_RADIUS_M))
        gts.append({"scene": SCENE, "command_id": cid, "command": p["instruction"], "grid": grid.to_dict(),
                    "masks": [[list(c) for c in mask]], "asks": [None], "frames": ["robot_start"],
                    "anchor": {"center": [ox, oz], "side": [(tx - ox) / d, (tz - oz) / d]},
                    "proxy": True})

    out = os.path.join(REPO, "cleancmd/pilot")
    os.makedirs(out, exist_ok=True)
    save_jsonl(os.path.join(out, SCENE + "_probes.jsonl"), probes)
    save_jsonl(os.path.join(out, SCENE + "_gt_proxy.jsonl"), gts)
    print("wrote %d probes and proxy ground truth to %s (grid %s)" % (len(probes), out, grid.to_dict()))


if __name__ == "__main__":
    main()
