#!/usr/bin/env python3
"""Build a single-episode custom dataset (+ matching dummy GT file) so an
arbitrary free-text instruction can be run through the existing ETPNav eval
pipeline (see /home/storms-end/.claude/plans/squishy-popping-cupcake.md).
"""
import argparse
import copy
import gzip
import json
import os

from transformers import BertTokenizer

VOCAB_PATH = "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/vocab.txt"
VOCAB_SOURCE_SPLIT = "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"

CUSTOM_DATASET_PATH = "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/custom/custom_bertidx.json.gz"
CUSTOM_GT_PATH = "data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/custom_gt.json.gz"

MAX_INPUT = 80

# Default pose: mp3d/zsNo4HB9uLZ, a real navmesh-valid start position whose
# ground-truth R2R instructions describe a dining table / coffee table /
# living room right from this spot.
DEFAULT_SCENE = "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
DEFAULT_START = [15.068599700927734, 0.17162801325321198, -4.4848198890686035]
DEFAULT_ROT = [0, 0.6198966754439885, 0, -0.7846834468583432]
# A real, distinct navigable point in the same scene, used only as a dummy
# "goal" so SPL/SUCCESS/DISTANCE_TO_GOAL measures don't divide by zero
# (goal == start would make geodesic_distance 0). The model never sees this.
DEFAULT_GOAL = [13.04640007019043, 0.17162801325321198, 1.8739700317382812]
DEFAULT_GEODESIC_DISTANCE = 7.9608235359191895


def pad_instr_tokens(instr_tokens, maxlength=MAX_INPUT):
    if len(instr_tokens) > maxlength - 2:  # -2 for [CLS] and [SEP]
        instr_tokens = instr_tokens[: maxlength - 2]
    instr_tokens = ["[CLS]"] + instr_tokens + ["[SEP]"]
    instr_tokens += ["[PAD]"] * (maxlength - len(instr_tokens))
    assert len(instr_tokens) == maxlength
    return instr_tokens


def build(instruction, scene, start_position, start_rotation, episode_id, goal_position=None):
    goal_position = goal_position if goal_position is not None else DEFAULT_GOAL
    tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH, do_lower_case=True)
    tokens = tokenizer.tokenize(instruction)
    padded = pad_instr_tokens(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(padded)

    with gzip.open(VOCAB_SOURCE_SPLIT) as f:
        source = json.load(f)
    instruction_vocab = copy.deepcopy(source["instruction_vocab"])

    episode = {
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "scene_id": scene,
        "start_position": start_position,
        "start_rotation": start_rotation,
        "info": {"geodesic_distance": DEFAULT_GEODESIC_DISTANCE},  # decorative only; SPL uses the live POSITION measure
        "goals": [{"position": goal_position, "radius": 3.0}],
        "instruction": {
            "instruction_text": instruction,
            "instruction_tokens": token_ids,
        },
        "reference_path": [start_position, goal_position],
    }
    dataset = {"episodes": [episode], "instruction_vocab": instruction_vocab}

    os.makedirs(os.path.dirname(CUSTOM_DATASET_PATH), exist_ok=True)
    with gzip.open(CUSTOM_DATASET_PATH, "wt") as f:
        f.write(json.dumps(dataset))

    gt = {
        str(episode_id): {
            "locations": [start_position, goal_position],
            "forward_steps": 0,
            "actions": [0],
        }
    }
    os.makedirs(os.path.dirname(CUSTOM_GT_PATH), exist_ok=True)
    with gzip.open(CUSTOM_GT_PATH, "wt") as f:
        f.write(json.dumps(gt))

    print(f"instruction: {instruction!r}")
    print(f"tokens ({len(tokens)} words -> padded to {MAX_INPUT}): {tokens}")
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
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--goal", type=float, nargs=3, default=None)
    args = parser.parse_args()
    build(args.instruction, args.scene, args.start, args.rot, args.episode_id, args.goal)


if __name__ == "__main__":
    main()
