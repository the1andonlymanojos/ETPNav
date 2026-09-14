#!/usr/bin/env python3
"""Run one free-text instruction through ETPNav in a chosen MP3D scene and
produce a video of the rollout. See
/home/storms-end/.claude/plans/squishy-popping-cupcake.md for how this works.

Must run inside `conda activate vlnce`. Example:
    python scripts/run_instruction.py --instruction "move to the left of the table"
"""
import argparse
import glob
import os
import subprocess
import sys

import build_custom_episode as bce

CKPT_PATH = "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
PRETRAINED_PATH = "data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene", default=bce.DEFAULT_SCENE)
    parser.add_argument("--start", type=float, nargs=3, default=bce.DEFAULT_START)
    parser.add_argument("--rot", type=float, nargs=4, default=bce.DEFAULT_ROT)
    parser.add_argument("--exp-name", default="custom_demo")
    args = parser.parse_args()
    EXP_NAME = args.exp_name

    bce.build(args.instruction, args.scene, args.start, args.rot, episode_id=0)

    opts = [
        "SIMULATOR_GPU_IDS", "[0]",
        "TORCH_GPU_IDS", "[0]",
        "GPU_NUMBERS", "1",
        "NUM_ENVIRONMENTS", "1",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "True",
        "EVAL.CKPT_PATH_DIR", CKPT_PATH,
        "EVAL.SPLIT", "custom",
        "EVAL.EPISODE_COUNT", "1",
        "TASK_CONFIG.DATASET.DATA_PATH",
        "data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz",
        "TASK_CONFIG.TASK.NDTW.GT_PATH",
        "data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/{split}_gt.json.gz",
        "TASK_CONFIG.TASK.SDTW.GT_PATH",
        "data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/{split}_gt.json.gz",
        "VIDEO_OPTION", "['disk']",
        "IL.back_algo", "control",
        "MODEL.pretrained_path", PRETRAINED_PATH,
    ]

    cmd = [
        sys.executable, "run.py",
        "--exp_name", EXP_NAME,
        "--run-type", "eval",
        "--exp-config", "run_r2r/iter_train.yaml",
    ] + opts

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    video_dir = f"data/logs/video/{EXP_NAME}"
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if videos:
        print(f"video(s) written to {video_dir}:")
        for v in videos:
            print(" ", v)
    else:
        print(f"no video found in {video_dir} -- check logs above for errors")


if __name__ == "__main__":
    main()
