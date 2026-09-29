#!/usr/bin/env python3
"""RxR variant of run_instruction.py -- runs one free-text instruction
through the release_rxr checkpoint and produces a video of the rollout.
Must run inside `conda activate vlnce`. Example:
    python scripts/run_instruction_rxr.py --instruction "move to the left of the table"
"""
import argparse
import glob
import os
import subprocess
import sys

import build_custom_episode_rxr as bce

CKPT_PATH = "data/logs/checkpoints/release_rxr/ckpt.iter19600.pth"
PRETRAINED_PATH = "data/pretrained/ETP/mlm.sap_rxr/ckpts/model_step_90000.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene", default=bce.DEFAULT_SCENE)
    parser.add_argument("--start", type=float, nargs=3, default=bce.DEFAULT_START)
    parser.add_argument("--rot", type=float, nargs=4, default=bce.DEFAULT_ROT)
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--exp-name", default="custom_demo_rxr")
    args = parser.parse_args()
    EXP_NAME = args.exp_name

    bce.build(args.instruction, args.scene, args.start, args.rot, episode_id=0, language=args.language)

    opts = [
        "SIMULATOR_GPU_IDS", "[0]",
        "TORCH_GPU_IDS", "[0]",
        "GPU_NUMBERS", "1",
        "NUM_ENVIRONMENTS", "1",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "False",
        "EVAL.CKPT_PATH_DIR", CKPT_PATH,
        "EVAL.SPLIT", "custom",
        "EVAL.EPISODE_COUNT", "1",
        "EVAL.LANGUAGES", "['*']",
        "TASK_CONFIG.DATASET.DATA_PATH",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}.json.gz",
        "IL.RECOLLECT_TRAINER.gt_file",
        "data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz",
        "VIDEO_OPTION", "['disk']",
        "IL.back_algo", "control",
        "MODEL.pretrained_path", PRETRAINED_PATH,
    ]

    cmd = [
        sys.executable, "run.py",
        "--exp_name", EXP_NAME,
        "--run-type", "eval",
        "--exp-config", "run_rxr/iter_train.yaml",
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
