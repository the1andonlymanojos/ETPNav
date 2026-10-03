"""Score predictions against human ground truth, and compare methods.

    python -m cleancmd.score --gt gt.jsonl --pred vlmaps.jsonl [--pred ours.jsonl ...] [--free scene_free.json ...]

Per command: recall against the core mask (primary), precision against the envelope, IoU with the core,
wrong-side (directional anchors only), ask correctness, and success. Thresholds are fixed here, before any results.

--free (optional, one file per scene) restricts scoring to open floor a vacuum can reach: cells not in the scene's
free set are removed from the ground truth (core and envelope) and from every prediction before anything is computed.
Without it, scoring is unchanged.
"""
import argparse
import copy
import json
import random
from collections import defaultdict

from .grid import Grid
from .io import Truth, load_jsonl, to_cells

RECALL_MIN = 0.8      # success needs at least 80% of the core cleaned ...
PRECISION_MIN = 0.5   # ... and at least half of what was cleaned inside the envelope


def load_free(path):
    """A free-floor file {"scene": ..., "grid": Grid.to_dict(), "free": [[r, c], ...]} -> (scene, Grid, set of cells)."""
    with open(path) as f:
        d = json.load(f)
    return d.get("scene"), Grid.from_dict(d["grid"]), {(int(r), int(c)) for r, c in d["free"]}


def restrict_truth(truth, free):
    """Copy of `truth` with core and envelope limited to the free cells (same as filtering every person's mask)."""
    t = copy.copy(truth)
    t.core = truth.core & free
    t.envelope = truth.envelope & free
    return t


def score_one(pred, truth, free=None):
    cells = to_cells(pred, truth.grid)
    if free is not None:
        cells &= free
        truth = restrict_truth(truth, free)
    executed = pred["kind"] != "ask"
    out = {"command_id": truth.raw["command_id"], "scene": truth.raw["scene"], "executed": executed,
           "should_ask": truth.should_ask, "recall": None, "precision": None, "iou": None, "wrong_side": None}

    if truth.should_ask:
        field = (pred.get("ask") or {}).get("field") if not executed else None
        out["ask_correct"] = (not executed) and field == truth.ask_field
    else:
        out["ask_correct"] = executed

    if executed and truth.core:
        hit = len(cells & truth.core)
        out["recall"] = hit / float(len(truth.core))
        out["precision"] = len(cells & truth.envelope) / float(len(cells)) if cells else 0.0
        out["iou"] = hit / float(len(cells | truth.core))
        side = (truth.anchor or {}).get("side")
        if side and cells:
            ax, az = truth.anchor["center"]
            dots = []
            for r, c in cells:
                x, z = truth.grid.center(r, c)
                dots.append((x - ax) * side[0] + (z - az) * side[1])
            # wrong side = most of the cleaned cells lie behind the anchor relative to the intended direction
            out["wrong_side"] = sum(d < 0 for d in dots) * 2 > len(dots)

    if truth.should_ask:
        out["success"] = out["ask_correct"]
    else:
        out["success"] = bool(executed and out["recall"] is not None
                              and out["recall"] >= RECALL_MIN and out["precision"] >= PRECISION_MIN)
    return out


def score_file(gt_rows, pred_rows, free=None):
    """Score one method. A command with no prediction counts as a failure, never silently dropped.
    free: optional {scene: (Grid, set of free cells)}; a scene missing from it is an error, never scored unrestricted."""
    preds = {p["command_id"]: p for p in pred_rows}
    results = []
    for gt in gt_rows:
        truth = Truth(gt)
        scene_free = None
        if free is not None:
            if gt["scene"] not in free:
                raise KeyError("no free-floor mask for scene %r" % gt["scene"])
            free_grid, scene_free = free[gt["scene"]]
            if free_grid != truth.grid:
                raise ValueError("free-floor grid for %r does not match the ground-truth grid" % gt["scene"])
        p = preds.get(gt["command_id"])
        if p is None:
            results.append({"command_id": gt["command_id"], "scene": gt["scene"], "missing": True, "success": False,
                            "executed": False, "should_ask": truth.should_ask, "ask_correct": False,
                            "recall": None, "precision": None, "iou": None, "wrong_side": None})
        else:
            results.append(score_one(p, truth, scene_free))
    return results


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / float(len(vals)) if vals else None


def summarize(results):
    return {
        "n": len(results),
        "missing": sum(1 for r in results if r.get("missing")),
        "success": _mean([float(r["success"]) for r in results]),
        "recall": _mean([r["recall"] for r in results]),
        "precision": _mean([r["precision"] for r in results]),
        "iou": _mean([r["iou"] for r in results]),
        "wrong_side": _mean([float(r["wrong_side"]) for r in results if r["wrong_side"] is not None]),
        # meaningless when no command should be asked about (e.g. the proxy pilot): every executed answer is "correct"
        "ask_accuracy": (_mean([float(r["ask_correct"]) for r in results])
                         if any(r["should_ask"] for r in results) else None),
        "n_should_ask": sum(1 for r in results if r["should_ask"]),
    }


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value for paired successes a[i], b[i] (same commands, two methods)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = n01 + n10, min(n01, n10)
    if n == 0:
        return 1.0
    from math import factorial  # math.comb is 3.8+; this also runs in the Python 3.6 ETPNav env
    comb = lambda n_, i: factorial(n_) // (factorial(i) * factorial(n_ - i))  # noqa: E731
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def bootstrap_ci(results, key="success", n_boot=2000, seed=0):
    """95% interval of the mean of `key`, resampling whole scenes (commands in one house are not independent)."""
    by_scene = defaultdict(list)
    for r in results:
        if r.get(key) is not None:
            by_scene[r["scene"]].append(float(r[key]))
    scenes = sorted(by_scene)
    if len(scenes) < 2:   # resampling one scene always returns the same mean: no interval exists
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        vals = [v for s in (rng.choice(scenes) for _ in scenes) for v in by_scene[s]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", action="append", required=True, help="one jsonl per method; repeat to compare")
    ap.add_argument("--free", action="append", default=None, help="free-floor json per scene: score open floor only")
    a = ap.parse_args()
    gt = load_jsonl(a.gt)
    free = None
    if a.free:
        free = {}
        for path in a.free:
            scene, grid, cells = load_free(path)
            free[scene] = (grid, cells)
        print("scoring open floor only (--free):", ", ".join("%s %d cells" % (s, len(c)) for s, (_, c) in free.items()))
    scored = {}
    for path in a.pred:
        scored[path] = score_file(gt, load_jsonl(path), free)
        s = summarize(scored[path])
        ci = bootstrap_ci(scored[path])
        print(path, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()},
              "success 95% CI", None if ci is None else (round(ci[0], 3), round(ci[1], 3)))
    paths = list(scored)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a_s = [r["success"] for r in scored[paths[i]]]
            b_s = [r["success"] for r in scored[paths[j]]]
            print("McNemar", paths[i], "vs", paths[j], "p =", round(mcnemar_exact(a_s, b_s), 4))


if __name__ == "__main__":
    main()
