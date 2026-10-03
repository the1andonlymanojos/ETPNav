#!/usr/bin/env python3
"""Run ETPNav (release R2R checkpoint) on a CleanCmd probe file and write predictions in the cleancmd format.
Each command becomes kind=path: the agent's floor trajectory, which the scorer turns into cells.

Inside `conda activate vlnce`, from the repo root (on a shared machine, through scripts/guarded_run.sh):
    python scripts/run_probes_etpnav.py --probes cleancmd/pilot/zsNo4HB9uLZ_probes.jsonl \
        --out cleancmd/pilot/zsNo4HB9uLZ_pred_etpnav.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_spatial_batch import REPO_ROOT, run_one  # noqa: E402
from cleancmd.io import load_jsonl  # noqa: E402

# a goal the eval code needs to compute its own metrics; unused by cleancmd scoring. Any distinct navigable point works.
DUMMY_GOAL_OFFSET = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp-prefix", default="cleancmd_etpnav")
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):          # resumable: skip commands already written
        done = {p["command_id"] for p in load_jsonl(a.out)}
    probes = load_jsonl(a.probes)
    with open(a.out, "a") as f:
        for i, p in enumerate(probes):
            if p["command_id"] in done:
                continue
            exp = "%s_%s" % (a.exp_prefix, p["command_id"])
            scene = "mp3d/{0}/{0}.glb".format(p["scene"])
            sx, sy, sz = p["start_position"]
            goal = [sx + DUMMY_GOAL_OFFSET, sy, sz]
            t0 = time.time()
            print("[%d/%d] %s: %r" % (i + 1, len(probes), p["command_id"], p["command"]), flush=True)
            try:
                run_one(scene, p["start_position"], p["start_rotation"], goal, p["command"], 0, exp)
                paths = json.load(open(os.path.join(REPO_ROOT, "data/logs/eval_results", exp, "paths_xz.json")))
                path = next(iter(paths.values()))
                pred = {"scene": p["scene"], "command_id": p["command_id"], "method": "etpnav_r2r",
                        "kind": "path", "path": path, "meta": {"exp": exp, "seconds": round(time.time() - t0, 1)}}
            except Exception as e:  # recorded, not skipped: the scorer counts it as a failure
                print("  FAILED: %s" % e, flush=True)
                pred = {"scene": p["scene"], "command_id": p["command_id"], "method": "etpnav_r2r",
                        "kind": "region", "cells": [], "meta": {"error": str(e)}}
            f.write(json.dumps(pred) + "\n")
            f.flush()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
