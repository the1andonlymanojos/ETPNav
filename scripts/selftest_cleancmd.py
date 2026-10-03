#!/usr/bin/env python3
"""CPU-only test of the CleanCmd grid, format and scorer on a synthetic room, with numbers checkable by hand.
Run: python scripts/selftest_cleancmd.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleancmd.grid import Grid  # noqa: E402
from cleancmd.io import Truth, to_cells  # noqa: E402
from cleancmd.score import bootstrap_ci, mcnemar_exact, score_file, score_one, summarize  # noqa: E402

# 6 m x 4 m room, 0.5 m cells; x origin snapped down from -0.2 to -0.5 -> 8 rows (z) x 13 cols (x)
g = Grid.from_bounds(-0.2, 0.0, 5.8, 4.0, cell=0.5)
assert g.to_dict() == {"origin_xz": [-0.5, 0.0], "shape": [8, 13], "cell": 0.5, "floor_y": None}, g.to_dict()
assert g.to_cell(0.1, 0.1) == (0, 1) and g.center(0, 1) == (0.25, 0.25)
assert Grid.from_dict(g.to_dict()) == g
print("grid: origin snapping, cell <-> world round trip  OK")

# a point answer covers the cells within 0.75 m of it; a point exactly on a cell centre gets that cell + its 4 neighbours
pt = to_cells({"kind": "point", "point": [2.25, 2.25]}, g)
assert (4, 5) in pt and {(3, 5), (5, 5), (4, 4), (4, 6)} <= pt and (2, 5) not in pt, sorted(pt)
# a path answer covers the last metre widened by 0.5 m: far start is excluded, the end is included
path = to_cells({"kind": "path", "path": [[0.25, 0.25], [4.25, 0.25], [4.25, 2.25]]}, g)
assert g.to_cell(4.25, 2.25) in path and g.to_cell(4.25, 1.25) in path and g.to_cell(0.25, 0.25) not in path
assert to_cells({"kind": "ask", "ask": {"field": "frame"}}, g) == set()
print("format: point / path / ask -> cells  OK")

# ground truth: "mop in front of the fridge". Fridge at x~1.25, z~0.25 facing +z; 4 people.
core4 = [(1, 2), (1, 3), (2, 2), (2, 3)]
gt = {"scene": "synthA", "command_id": "c1", "command": "mop in front of the fridge", "grid": g.to_dict(),
      "masks": [core4 + [(3, 2)], core4, core4 + [(1, 4)], core4[:3]],
      "asks": [None, None, None, None], "anchor": {"center": [1.25, 0.25], "side": [0.0, 1.0]}}
t = Truth(gt)
assert t.core_min == 3 and t.core == set(core4), t.core   # (2,3) chosen by 3 of 4 -> core
assert t.envelope == set(core4) | {(3, 2), (1, 4)} and not t.should_ask
print("truth: core = >=3 of 4 people, envelope = anyone, no ask  OK")

exact = score_one({"kind": "region", "cells": core4}, t)
assert exact["recall"] == 1.0 and exact["precision"] == 1.0 and exact["iou"] == 1.0
assert exact["success"] and exact["wrong_side"] is False
half = score_one({"kind": "region", "cells": [(1, 2), (1, 3), (6, 10), (6, 11)]}, t)
assert half["recall"] == 0.5 and half["precision"] == 0.5 and not half["success"]
# anchor at z=1.0 facing +z: row-0 cells (z centre 0.25) lie behind it
behind = score_one({"kind": "region", "cells": [(0, 2), (0, 3)]}, Truth(dict(gt, anchor={"center": [1.25, 1.0], "side": [0.0, 1.0]})))
assert behind["wrong_side"] is True and behind["recall"] == 0.0
asked_wrongly = score_one({"kind": "ask", "ask": {"field": "frame"}}, t)
assert not asked_wrongly["ask_correct"] and not asked_wrongly["success"] and asked_wrongly["recall"] is None
print("score: exact / half / wrong side / needless question  OK")

# an ambiguous command: 2 of 3 listeners would ask about the frame -> the robot should ask, about the frame
amb = dict(gt, command_id="c2", asks=[None, "frame", "frame", None])
ta = Truth(amb)
assert ta.should_ask and ta.ask_field == "frame"
assert score_one({"kind": "ask", "ask": {"field": "frame"}}, ta)["success"]
assert not score_one({"kind": "ask", "ask": {"field": "extent"}}, ta)["success"]   # right to ask, wrong reason
assert not score_one({"kind": "region", "cells": core4}, ta)["success"]           # should have asked
print("ask: right field / wrong field / executed instead of asking  OK")

# whole-file scoring: a missing prediction is a failure, never dropped
res = score_file([gt, amb], [{"command_id": "c1", "kind": "region", "cells": core4}])
s = summarize(res)
assert s["n"] == 2 and s["missing"] == 1 and s["success"] == 0.5, s
print("file: missing prediction counted as failure  OK")

# stats: 10 commands, method B wins 8 discordant pairs and loses none -> p = 2 * 0.5^8
a = [False] * 8 + [True, True]
b = [True] * 10
assert abs(mcnemar_exact(a, b) - 2 * 0.5 ** 8) < 1e-12 and mcnemar_exact(b, b) == 1.0
lo, hi = bootstrap_ci([{"scene": s, "success": v} for s, v in zip("aabbccdd", [1, 1, 0, 0, 1, 0, 1, 1])])
assert 0.0 <= lo <= 0.625 <= hi <= 1.0
print("stats: exact McNemar, scene-clustered bootstrap  OK")
print("ALL CLEANCMD SELF-TESTS PASSED")
