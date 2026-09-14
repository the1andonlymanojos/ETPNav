#!/usr/bin/env python3
"""RxR variant of build_custom_episode.py: builds a single-episode custom
RxR-VLN-CE dataset (+ matching dummy GT file) so a free-text instruction can
be run through the release_rxr checkpoint. See
/home/storms-end/.claude/plans/squishy-popping-cupcake.md for the R2R version
this mirrors; RxR differs in tokenizer (XLM-R, not BERT), dataset schema
(role-sharded files, no instruction_vocab), and instruction sensor uuid
handling.
"""
import argparse
import gzip
import json
import os

from transformers import AutoTokenizer

import build_custom_episode as r2r  # reuse the known-good scene/pose defaults

CUSTOM_DATASET_PATH = "data/datasets/RxR_VLNCE_v0_enc_xlmr/custom/custom_guide.json.gz"
CUSTOM_GT_PATH = "data/datasets/RxR_VLNCE_v0_enc_xlmr/custom/custom_guide_gt.json.gz"

DEFAULT_SCENE = r2r.DEFAULT_SCENE
DEFAULT_START = r2r.DEFAULT_START
DEFAULT_ROT = r2r.DEFAULT_ROT
DEFAULT_GOAL = r2r.DEFAULT_GOAL
DEFAULT_GEODESIC_DISTANCE = r2r.DEFAULT_GEODESIC_DISTANCE


def build(instruction, scene, start_position, start_rotation, episode_id, language="en-IN"):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", do_lower_case=True)
    token_ids = tokenizer(instruction)["input_ids"]

    episode = {
        "episode_id": str(episode_id),
        "trajectory_id": episode_id,
        "scene_id": scene,
        "start_position": start_position,
        "start_rotation": start_rotation,
        "goals": [{"position": DEFAULT_GOAL, "radius": 3.0}],
        "reference_path": [start_position, DEFAULT_GOAL],
        "instruction": {
            "instruction_id": str(episode_id),
            "instruction_text": instruction,
            "language": language,
            "annotator_id": None,
            "edit_distance": None,
            "instruction_tokens": token_ids,
        },
    }
    dataset = {"episodes": [episode]}

    os.makedirs(os.path.dirname(CUSTOM_DATASET_PATH), exist_ok=True)
    with gzip.open(CUSTOM_DATASET_PATH, "wt") as f:
        f.write(json.dumps(dataset))

    gt = {
        str(episode_id): {
            "locations": [start_position, DEFAULT_GOAL],
            "forward_steps": 0,
            "actions": [0],
        }
    }
    os.makedirs(os.path.dirname(CUSTOM_GT_PATH), exist_ok=True)
    with gzip.open(CUSTOM_GT_PATH, "wt") as f:
        f.write(json.dumps(gt))

    print(f"instruction: {instruction!r}")
    print(f"xlm-r tokens ({len(token_ids)}): {token_ids}")
    print(f"scene: {scene}")
    print(f"start_position: {start_position}")
    print(f"wrote dataset -> {CUSTOM_DATASET_PATH}")
    print(f"wrote gt       -> {CUSTOM_GT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--start", type=float, nargs=3, default=DEFAULT_START)
    parser.add_argument("--rot", type=float, nargs=4, default=DEFAULT_ROT)
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--episode-id", type=int, default=0)
    args = parser.parse_args()
    build(args.instruction, args.scene, args.start, args.rot, args.episode_id, args.language)


if __name__ == "__main__":
    main()
