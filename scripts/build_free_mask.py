#!/usr/bin/env python3
"""Free-floor mask for a CleanCmd scene grid: the cells a floor-cleaning robot can actually reach.

A cell is free when at least half of an n x n lattice of points inside it is navigable on the grid's floor
(habitat-sim pathfinder, navmesh of the scene). Needs habitat_sim, e.g. inside `conda activate vlmaps_3.9`:
    python scripts/build_free_mask.py --gt cleancmd/pilot/zsNo4HB9uLZ_gt_proxy.jsonl \
        --out cleancmd/pilot/zsNo4HB9uLZ_free.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleancmd.grid import Grid  # noqa: E402
from cleancmd.io import load_jsonl  # noqa: E402

NAVMESH = os.path.expanduser("~/vlmaps_cc/mp3d_habitat/mp3d/{0}/{0}.navmesh")
SAMPLES = 5          # 5 x 5 points per cell
MIN_FRACTION = 0.5   # free if at least half of them are navigable


def free_cells(pathfinder, grid, samples=SAMPLES, min_fraction=MIN_FRACTION):
    offs = (np.arange(samples) + 0.5) / samples * grid.cell
    free = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            x0, z0 = grid.x0 + c * grid.cell, grid.z0 + r * grid.cell
            hits = sum(pathfinder.is_navigable(np.array([x0 + dx, grid.floor_y, z0 + dz], dtype=np.float32))
                       for dx in offs for dz in offs)
            if hits >= min_fraction * samples * samples:
                free.append([r, c])
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="ground-truth jsonl; its (single) grid is the one masked")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import habitat_sim

    rows = load_jsonl(a.gt)
    scenes = {g["scene"] for g in rows}
    grids = {json.dumps(g["grid"], sort_keys=True) for g in rows}
    assert len(scenes) == 1 and len(grids) == 1, "expected one scene and one grid in %s" % a.gt
    scene, grid = scenes.pop(), Grid.from_dict(rows[0]["grid"])
    assert grid.floor_y is not None, "grid has no floor_y"

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(NAVMESH.format(scene))
    free = free_cells(pf, grid)
    with open(a.out, "w") as f:
        json.dump({"scene": scene, "grid": grid.to_dict(), "free": free,
                   "rule": "cell free if >= %g of %dx%d points navigable at floor_y" % (MIN_FRACTION, SAMPLES, SAMPLES)}, f)
    print("%s: %d of %d cells free -> %s" % (scene, len(free), grid.rows * grid.cols, a.out))


if __name__ == "__main__":
    main()
