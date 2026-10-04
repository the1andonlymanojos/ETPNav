"""
Open-vocabulary detector for the object-picking benchmark, from vlmaps_cc/application/extract_openvocab_scores.py's
score maps (ov/<tag>_ovscores.npz): threshold the per-cell score for the command's object phrase, close small gaps
(binary closing, CLOSE_ITERS), take 8-connected components, keep those with >= min_cells cells (cell = 5 cm).
Each kept component is a candidate instance in the same schema as <tag>_detections.json (centre = bbox middle,
as VLMaps' get_segment_islands_pos), written to ov/<tag>_ov_detections.json.

Score families (all fixed thresholds):
  score   raw cosine to the phrase
  margin  cosine minus the best of LERF's generic negatives
  z       raw cosine standardised per phrase over the scene's mapped cells ((s - median) / (1.4826 * MAD))

The family, threshold and min cluster size are chosen ONCE, on the tuning scene only (--sweep), by instance-level
F1: a candidate is a true positive when it lies within 1 m of an annotated instance of its category, recall is the
share of annotated instances with a candidate within 1 m. The choice is then frozen in CHOSEN.

python cleancmd/anchor/ov_detect.py --sweep zsNo4HB9uLZ          # pick the parameters
python cleancmd/anchor/ov_detect.py --scenes zsNo4HB9uLZ,...     # write ov/<tag>_ov_detections.json with CHOSEN
"""
import argparse
import json
import math

import numpy as np
from scipy import ndimage

from eval_anchor import HERE, CORRECT_M
from make_anchor_cmds import say

OV = HERE / "ov"
CLOSE_ITERS = 2
TUNING_SCENE = "zsNo4HB9uLZ"
CHOSEN = ("score", 0.89, 50)  # best F1 of --sweep zsNo4HB9uLZ (ov/sweep_zsNo4HB9uLZ.json); frozen before any other scene


def score_map(z, family, phrase):
    if family == "z":
        s = z[f"score/{phrase}"].astype(float)
        v = s[s > -1]
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        return np.where(s > -1, (s - med) / mad, -np.inf)
    s = z[f"{family}/{phrase}"].astype(float)
    return np.where(s > -1, s, -np.inf)


def detect(z, family, thr, min_cells, phrase):
    mask = ndimage.binary_closing(score_map(z, family, phrase) >= thr, iterations=CLOSE_ITERS)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    r0, c0 = z["offset"]
    A = z["affine"]
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        size = int((lab[sl] == i + 1).sum())
        if size < min_cells:
            continue
        rmin, rmax = int(sl[0].start + r0), int(sl[0].stop - 1 + r0)
        cmin, cmax = int(sl[1].start + c0), int(sl[1].stop - 1 + c0)
        ctr = [(rmin + rmax) / 2, (cmin + cmax) / 2]
        out.append({"center_rc": ctr, "bbox_rc": [rmin, rmax, cmin, cmax],
                    "center_xz": (A @ [ctr[0], ctr[1], 1.0]).round(3).tolist(), "area_cells": size, "size_ok": True})
    return out


def gt_instances(rows):
    inst = {}
    for r in rows:
        inst[r["gt"]["instance_id"]] = (r["category"], r["gt"]["center_xz"])
    return inst


def evaluate(dets, inst):
    """Instance-level precision / recall / F1 of {category: [candidates]} against the annotated instances."""
    tp = n = 0
    for cat, ds in dets.items():
        gts = [xz for c, xz in inst.values() if c == cat]
        for d in ds:
            n += 1
            tp += any(math.dist(d["center_xz"], g) <= CORRECT_M for g in gts)
    found = sum(any(math.dist(d["center_xz"], xz) <= CORRECT_M for d in dets.get(c, [])) for c, xz in inst.values())
    p = tp / n if n else 0.0
    r = found / len(inst)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), n


def build(tag, family, thr, min_cells):
    rows = [json.loads(l) for l in open(HERE / f"{tag}_anchor_cmds.jsonl")]
    z = np.load(OV / f"{tag}_ovscores.npz")
    dets = {}
    for cat in sorted({r["category"] for r in rows}):
        dets[cat] = detect(z, family, thr, min_cells, say(cat))
        for i, d in enumerate(dets[cat]):
            d["det_id"] = f"{cat}#ov{i}"
    return rows, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", help="tuning scene tag")
    ap.add_argument("--scenes", help="comma list of tags to write with CHOSEN")
    args = ap.parse_args()
    if args.sweep:
        grids = {"score": np.round(np.arange(0.80, 0.97, 0.01), 2), "margin": np.round(np.arange(0.0, 0.161, 0.01), 2),
                 "z": np.arange(1.0, 6.01, 0.5)}
        res = []
        for family, thrs in grids.items():
            for thr in thrs:
                for mc in (10, 25, 50, 100, 200, 400):
                    rows, dets = build(args.sweep, family, float(thr), mc)
                    p, r, f, n = evaluate(dets, gt_instances(rows))
                    res.append({"family": family, "thr": float(thr), "min_cells": mc, "precision": p, "recall": r,
                                "f1": f, "candidates": n})
        res.sort(key=lambda x: -x["f1"])
        OV.mkdir(exist_ok=True)
        json.dump(res, open(OV / f"sweep_{args.sweep}.json", "w"), indent=1)
        for fam in grids:
            b = next(x for x in res if x["family"] == fam)
            print(f"best {fam:6s}: thr {b['thr']:.2f} min_cells {b['min_cells']:3d} | P {b['precision']:.2f} "
                  f"R {b['recall']:.2f} F1 {b['f1']:.3f} | {b['candidates']} candidates")
        return
    family, thr, mc = CHOSEN
    for tag in args.scenes.split(","):
        rows, dets = build(tag, family, thr, mc)
        old = json.load(open(HERE / f"{tag}_detections.json"))
        out = {"scene": tag, "detector": {"family": family, "thr": thr, "min_cells": mc, "close_iters": CLOSE_ITERS,
                                          "tuned_on": TUNING_SCENE},
               "detections": dets, "start_map_pose": old["start_map_pose"]}
        json.dump(out, open(OV / f"{tag}_ov_detections.json", "w"))
        p, r, f, n = evaluate(dets, gt_instances(rows))
        print(f"{tag}: {n} candidates | P {p:.2f} R {r:.2f} F1 {f:.3f}")


if __name__ == "__main__":
    main()
