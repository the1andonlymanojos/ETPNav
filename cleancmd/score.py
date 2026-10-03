"""Score predictions against human ground truth, and compare methods.

    python -m cleancmd.score --gt gt.jsonl --pred vlmaps.jsonl [--pred ours.jsonl ...]

Per command: recall against the core mask (primary), precision against the envelope, IoU with the core,
wrong-side (directional anchors only), ask correctness, and success. Thresholds are fixed here, before any results.
"""
import argparse
import random
from collections import defaultdict

from .io import Truth, load_jsonl, to_cells

RECALL_MIN = 0.8      # success needs at least 80% of the core cleaned ...
PRECISION_MIN = 0.5   # ... and at least half of what was cleaned inside the envelope


def score_one(pred, truth):
    cells = to_cells(pred, truth.grid)
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


def score_file(gt_rows, pred_rows):
    """Score one method. A command with no prediction counts as a failure, never silently dropped."""
    preds = {p["command_id"]: p for p in pred_rows}
    results = []
    for gt in gt_rows:
        truth = Truth(gt)
        p = preds.get(gt["command_id"])
        if p is None:
            results.append({"command_id": gt["command_id"], "scene": gt["scene"], "missing": True, "success": False,
                            "executed": False, "should_ask": truth.should_ask, "ask_correct": False,
                            "recall": None, "precision": None, "iou": None, "wrong_side": None})
        else:
            results.append(score_one(p, truth))
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
        "ask_accuracy": _mean([float(r["ask_correct"]) for r in results]),
    }


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value for paired successes a[i], b[i] (same commands, two methods)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = n01 + n10, min(n01, n10)
    if n == 0:
        return 1.0
    from math import comb  # noqa: E402  (py3.8+; scorer runs in the new env, not the frozen ETPNav one)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def bootstrap_ci(results, key="success", n_boot=2000, seed=0):
    """95% interval of the mean of `key`, resampling whole scenes (commands in one house are not independent)."""
    by_scene = defaultdict(list)
    for r in results:
        if r.get(key) is not None:
            by_scene[r["scene"]].append(float(r[key]))
    scenes = sorted(by_scene)
    if not scenes:
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
    a = ap.parse_args()
    gt = load_jsonl(a.gt)
    scored = {}
    for path in a.pred:
        scored[path] = score_file(gt, load_jsonl(path))
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
