"""File formats shared by every method, the annotation tool and the scorer (one JSON object per command).

Prediction (written by ETPNav, VLMaps, ours, ...):
    {"scene": "zsNo4HB9uLZ", "command_id": "zsNo-0007", "method": "vlmaps",
     "kind": "region" | "point" | "path" | "ask",
     "cells": [[r, c], ...],          # kind=region, cells of the scene grid
     "point": [x, z],                 # kind=point, Habitat world coordinates
     "path":  [[x, z], ...],          # kind=path, in visiting order (ETPNav's trajectory)
     "ask":   {"field": "anchor_multiple" | "anchor_missing" | "frame" | "extent" | "underspecified",
               "question": "Which sofa?"},
     "meta": {...}}                   # anything else (start pose, timings, raw output)

Ground truth (written by the annotation tool):
    {"scene": ..., "command_id": ..., "command": "mop in front of the fridge", "grid": Grid.to_dict(),
     "masks": [[[r, c], ...], ...],   # one per person; masks[0] is the speaker, the rest are listeners
     "asks":  [null, null, "frame", null],   # per person: null = would clean, else the field they would ask about
     "frames": ["object", "object", "user", "object"],
     "anchor": {"center": [x, z], "side": [dx, dz]} | null}   # side = intended direction from the anchor, if any

Point and path answers are turned into cells by fixed rules (to_cells) chosen before any results are seen.
"""
import json
from collections import Counter

from .grid import Grid

ASK_FIELDS = ("anchor_multiple", "anchor_missing", "frame", "extent", "underspecified")

# Fixed conversion rules for methods that answer with a point or a trajectory, set before scoring anything.
POINT_RADIUS_M = 0.75     # a spot clean around a point goal: cells within 0.75 m of it
PATH_TAIL_M = 1.0         # for a trajectory, the last 1 m of travel ...
PATH_RADIUS_M = 0.5       # ... widened by 0.5 m


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _path_tail(path, tail_m):
    """Points of `path` covering its last `tail_m` metres (always at least the final point)."""
    out = [path[-1]]
    left = tail_m
    for i in range(len(path) - 1, 0, -1):
        (x1, z1), (x0, z0) = path[i], path[i - 1]
        seg = ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5
        if seg >= left:
            f = left / seg if seg > 0 else 0.0
            out.append((x1 + (x0 - x1) * f, z1 + (z0 - z1) * f))
            break
        out.append((x0, z0))
        left -= seg
    return out


def to_cells(pred, grid):
    """The set of grid cells a prediction asks to clean (empty for kind=ask)."""
    kind = pred["kind"]
    if kind == "ask":
        return set()
    if kind == "region":
        return {(int(r), int(c)) for r, c in pred["cells"] if grid.inside(int(r), int(c))}
    if kind == "point":
        x, z = pred["point"]
        return grid.cells_within(x, z, POINT_RADIUS_M)
    if kind == "path":
        pts = _path_tail(pred["path"], PATH_TAIL_M)
        # sample the tail densely enough that no cell between samples is skipped
        cells = set()
        for (xa, za), (xb, zb) in zip(pts, pts[1:] + pts[-1:]):
            n = max(1, int(((xb - xa) ** 2 + (zb - za) ** 2) ** 0.5 / (grid.cell / 4)))
            for i in range(n + 1):
                t = i / float(n)
                cells |= grid.cells_within(xa + (xb - xa) * t, za + (zb - za) * t, PATH_RADIUS_M)
        return cells
    raise ValueError("unknown prediction kind %r" % kind)


class Truth(object):
    """Ground truth for one command, derived from several people's masks."""

    def __init__(self, gt, core_min=None):
        self.raw = gt
        self.grid = Grid.from_dict(gt["grid"])
        masks = [{(int(r), int(c)) for r, c in m} for m in gt["masks"]]
        n = len(masks)
        # core = cells most people chose: at least 3 of 4 by default, i.e. ceil(3n/4)
        self.core_min = core_min if core_min is not None else max(1, -(-3 * n // 4))
        votes = Counter(cell for m in masks for cell in m)
        self.core = {cell for cell, v in votes.items() if v >= self.core_min}
        self.envelope = set(votes)
        asks = gt.get("asks") or [None] * n
        listeners = asks[1:] if n > 1 else asks
        asked = [a for a in listeners if a]
        # the robot should ask when most listeners would; the field is the one they named most
        self.should_ask = len(asked) * 2 > len(listeners)
        self.ask_field = Counter(asked).most_common(1)[0][0] if self.should_ask else None
        self.anchor = gt.get("anchor")
