#!/usr/bin/env python3
"""Tables for the CleanCmd zsNo4HB9uLZ PILOT (geometric proxy ground truth, not human annotation).

Per-relation successes, distance from each answer's centre to the proxy target, and the object-choice audit
(right object = chosen object centre within 1 m of the annotated anchor centre). Uses cleancmd.score unchanged.
    python scripts/report_cleancmd_pilot.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleancmd.io import Truth, load_jsonl, to_cells  # noqa: E402
from cleancmd.score import load_free, mcnemar_exact, score_file  # noqa: E402

P = "cleancmd/pilot/zsNo4HB9uLZ"
METHODS = ["etpnav", "vlmaps", "vlmaps_nofilter", "ours_v0", "vlmaps_oracleobj", "ours_v0_oracleobj"]
RELS = ["left", "right", "next to", "behind"]
RIGHT_OBJECT_M = 1.0


def answer_centre(pred, grid):
    if pred["kind"] == "path":
        return np.array(pred["path"][-1])
    if pred["kind"] == "point":
        return np.array(pred["point"])
    cells = to_cells(pred, grid)
    return np.mean([grid.center(r, c) for r, c in cells], axis=0) if cells else None


def main():
    gt = load_jsonl(P + "_gt_proxy.jsonl")
    rel = {p["command_id"]: p["relation"] for p in load_jsonl(P + "_probes.jsonl")}
    _, fgrid, fcells = load_free(P + "_free.json")
    preds = {m: {p["command_id"]: p for p in load_jsonl("%s_pred_%s.jsonl" % (P, m))} for m in METHODS}

    for tag, free in (("all floor", None), ("open floor only (--free)", {"zsNo4HB9uLZ": (fgrid, fcells)})):
        print("\n## Per relation, %s: successes / n (mean recall)" % tag)
        print("| method | " + " | ".join(RELS) + " | total |")
        print("|---" * (len(RELS) + 2) + "|")
        succ = {}
        for m in METHODS:
            res = score_file(gt, list(preds[m].values()), free)
            succ[m] = [r["success"] for r in res]
            by = defaultdict(list)
            for r in res:
                by[rel[r["command_id"]]].append(r)
            cells = []
            for k in RELS:
                rec = [r["recall"] for r in by[k] if r["recall"] is not None]
                cells.append("%d/%d (%.2f)" % (sum(r["success"] for r in by[k]), len(by[k]), np.mean(rec) if rec else 0))
            print("| %s | %s | %d/25 |" % (m, " | ".join(cells), sum(succ[m])))
        a, b = succ["vlmaps_oracleobj"], succ["ours_v0_oracleobj"]
        print("McNemar vlmaps_oracleobj vs ours_v0_oracleobj (%s): p = %.4f  (discordant: vlmaps-only %d, ours-only %d)"
              % (tag, mcnemar_exact(a, b), sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))))

    print("\n## Distance from answer centre to proxy target centre (answered commands)")
    print("| method | answered | median m | within 1 m | within 2 m |")
    print("|---|---|---|---|---|")
    dist = {}
    for m in METHODS:
        dist[m] = {}
        for g in gt:
            t = Truth(g)
            c = answer_centre(preds[m][g["command_id"]], t.grid)
            if c is not None:
                tc = np.mean([t.grid.center(r, cc) for r, cc in t.core], axis=0)
                dist[m][g["command_id"]] = float(np.linalg.norm(c - tc))
        d = np.array(list(dist[m].values()))
        print("| %s | %d/25 | %.2f | %d | %d |" % (m, len(d), np.median(d), (d < 1).sum(), (d < 2).sum()))
    common = sorted(set(dist["vlmaps_oracleobj"]) & set(dist["ours_v0_oracleobj"]))
    dv = np.array([dist["vlmaps_oracleobj"][k] for k in common])
    do = np.array([dist["ours_v0_oracleobj"][k] for k in common])
    print("correct object, both answered (%d): VLMaps median %.2f, ours v0 median %.2f, ours closer on %d/%d"
          % (len(common), np.median(dv), np.median(do), (do < dv).sum(), len(common)))
    for k in RELS:
        ks = [c for c in common if rel[c] == k]
        if ks:
            print("  %-8s n=%d  VLMaps median %.2f  ours v0 median %.2f  ours closer %d/%d" % (
                k, len(ks), np.median([dist["vlmaps_oracleobj"][c] for c in ks]),
                np.median([dist["ours_v0_oracleobj"][c] for c in ks]),
                sum(dist["ours_v0_oracleobj"][c] < dist["vlmaps_oracleobj"][c] for c in ks), len(ks)))

    print("\n## Object-choice audit (right object = chosen centre within %.0f m of the annotated centre)" % RIGHT_OBJECT_M)
    anchors = json.load(open(P + "_anchor_choices.json"))
    print("| method | right object | wrong object | not found | unsupported relation | share of distance error on wrong objects |")
    print("|---|---|---|---|---|---|")
    audit = {}
    for m, key in (("vlmaps", "vlmaps"), ("vlmaps_nofilter", "vlmaps_nofilter"), ("ours_v0", "vlmaps")):
        counts = defaultdict(int)
        audit[m] = {}
        for g in gt:
            cid = g["command_id"]
            a = anchors[cid]
            if m != "ours_v0" and rel[cid] == "behind":
                cls = "unsupported"
            elif "center_xz" not in a[key]:
                cls = "not found"
            else:
                err = np.linalg.norm(np.array(a[key]["center_xz"]) - np.array(a["annotated_center_xz"]))
                cls = "right" if err <= RIGHT_OBJECT_M else "wrong"
            counts[cls] += 1
            audit[m][cid] = cls
        tot = sum(dist[m].values())
        wrong = sum(d for cid, d in dist[m].items() if audit[m][cid] == "wrong")
        print("| %s | %d | %d | %d | %d | %.0f%% (%.1f of %.1f m) |" % (
            m, counts["right"], counts["wrong"], counts["not found"], counts["unsupported"], 100 * wrong / tot, wrong, tot))
    print("per command:", " ".join("%s:%s/%s/%s" % (cid[-3:], *(audit[m][cid][0] for m in ("vlmaps", "vlmaps_nofilter", "ours_v0")))
                                   for cid in sorted(audit["vlmaps"])), "(r=right w=wrong n=not found u=unsupported)")


if __name__ == "__main__":
    main()
