#!/usr/bin/env python3
"""Build a single-house "tour" split from REAL R2R-CE val_unseen episodes: a fixed random sample of N episodes
from one scene, presented in a chosen order. Same --sample-seed => same episode set; --order-seed only permutes it.

Writes (relative to repo root):
  data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/<name>/<name>_bertidx.json.gz   (episodes, in tour order)
  data/datasets/R2R_VLNCE_v1-2_preprocessed/<name>/<name>_gt.json.gz                 (matching GT subset)
Then evaluate with:  EVAL.SPLIT <name>   (the default R2R DATA_PATH / GT_PATH templates pick these files up).
"""
import argparse
import gzip
import json
import os
import random

DS = "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{s}/{s}_bertidx.json.gz"
GT = "data/datasets/R2R_VLNCE_v1-2_preprocessed/{s}/{s}_gt.json.gz"


def build(scene, n, sample_seed, order_seed, name):
    ds = json.load(gzip.open(DS.format(s="val_unseen")))
    gt = json.load(gzip.open(GT.format(s="val_unseen")))
    pool = [e for e in ds["episodes"] if e["scene_id"].split("/")[1] == scene and str(e["episode_id"]) in gt]
    picked = random.Random(sample_seed).sample(pool, n)
    if order_seed:
        random.Random(order_seed).shuffle(picked)

    out_ds = {"episodes": picked, "instruction_vocab": ds["instruction_vocab"]}
    out_gt = {str(e["episode_id"]): gt[str(e["episode_id"])] for e in picked}
    for path, obj in ((DS.format(s=name), out_ds), (GT.format(s=name), out_gt)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(obj))
    return [str(e["episode_id"]) for e in picked]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    ids = build(a.scene, a.n, a.sample_seed, a.order_seed, a.name)
    print(f"split '{a.name}': {len(ids)} episodes in {a.scene}, order: {ids}")
