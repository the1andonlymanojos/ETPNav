#!/usr/bin/env python3
"""Run several rephrasings of the same spatial instruction against the same
fixed start pose + geometric target, to isolate whether misses are a
phrasing/grounding problem or a genuine navigation failure.

Must run inside `conda activate vlnce`. Example:
    python scripts/run_phrasing_experiment.py \
        --cases /tmp/.../phrasing_cases.json \
        --scene mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb \
        --out-prefix phrasing \
        --results-out /tmp/.../phrasing_results.json
"""
import argparse
import json
import os
import subprocess
import sys

import build_custom_episode as bce

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
PRETRAINED_PATH = "data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt"


def run_one(scene, start_position, start_rotation, goal, instruction, episode_id, exp_name):
    bce.build(instruction, scene, start_position, start_rotation, episode_id=episode_id, goal_position=goal)

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
    cmd = [sys.executable, "run.py", "--exp_name", exp_name, "--run-type", "eval",
           "--exp-config", "run_r2r/iter_train.yaml"] + opts
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    stats_dir = os.path.join(REPO_ROOT, "data/logs/eval_results", exp_name)
    stats_file = [f for f in os.listdir(stats_dir) if f.startswith("stats_ckpt_") and f.endswith("_custom.json")][0]
    with open(os.path.join(stats_dir, stats_file)) as f:
        stats = json.load(f)

    video_dir = os.path.join(REPO_ROOT, "data/logs/video", exp_name)
    mp4 = [f for f in os.listdir(video_dir) if f.endswith(".mp4")][0]
    return stats, os.path.join(video_dir, mp4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--results-out", required=True)
    args = parser.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)

    results = []
    run_id = 0
    for c in cases:
        exp_name = f"{args.out_prefix}_{run_id}"
        print(f"[{run_id + 1}/{len(cases)}] ep{c['base_episode_id']} tier={c['tier']}: {c['instruction']!r}")
        try:
            stats, video_path = run_one(
                args.scene, c["start_position"], c["start_rotation"], c["target"],
                c["instruction"], run_id, exp_name,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            run_id += 1
            continue
        results.append({**c, "run_id": run_id, "video_path": video_path, "stats": stats})
        run_id += 1

    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} results -> {args.results_out}")


if __name__ == "__main__":
    main()
