#!/usr/bin/env python3
"""Hybrid-grounding baseline: instead of asking ETPNav's language model to
resolve "left of the sofa" itself, use the same semantic-scene landmark
lookup + geometric target computation from build_spatial_landmarks.py, then
walk straight there with habitat-sim's own pathfinder. No VLN model, no GPU
inference -- this tests whether the failure is fixable by skipping language
grounding entirely for this narrow instruction class, not whether the
learned model itself got better.

Must run inside `conda activate vlnce`. Example:
    python scripts/run_hybrid_grounding.py \
        --landmarks /tmp/.../landmarks_zsNo4HB9uLZ.json \
        --scene mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb \
        --video-dir /tmp/.../hybrid_videos \
        --results-out /tmp/.../hybrid_results.json
"""
import argparse
import json
import math
import os

import habitat_sim
import imageio
import numpy as np
import quaternion  # noqa: F401 -- registers np.quaternion

STEP_SIZE = 0.15  # meters between interpolated frames along the path
SUCCESS_RADIUS = 1.5  # same pass threshold used in the spatial-relation artifact


def heading_to_quat(theta):
    return np.quaternion(math.cos(theta / 2), 0, math.sin(theta / 2), 0)


def heading_from_forward(fx, fz):
    return math.atan2(-fx, -fz)


def make_sim(scene_glb):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_glb
    backend_cfg.gpu_device_id = 0
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensor_spec = habitat_sim.SensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.resolution = [256, 256]
    sensor_spec.hfov = 90
    agent_cfg.sensor_specifications = [sensor_spec]
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def walk_and_record(sim, start_position, start_rotation_xyzw, target, video_path):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start_position, dtype=np.float32)
    path.requested_end = np.array(target, dtype=np.float32)
    found = sim.pathfinder.find_path(path)
    if not found:
        return None, None

    waypoints = list(path.points)
    frames = []
    agent = sim.get_agent(0)

    # face the first waypoint before moving, same convention as build_spatial_landmarks
    state = habitat_sim.AgentState()
    state.position = np.array(start_position, dtype=np.float32)
    x, y, z, w = start_rotation_xyzw
    state.rotation = np.quaternion(w, x, y, z)
    agent.set_state(state)
    frames.append(np.asarray(sim.get_sensor_observations()["color_sensor"])[:, :, :3])

    cur = np.array(start_position, dtype=np.float64)
    for wp in waypoints[1:]:
        wp = np.array(wp, dtype=np.float64)
        seg = wp - cur
        seg_xz = np.array([seg[0], seg[2]])
        dist = np.linalg.norm(seg_xz)
        if dist < 1e-4:
            cur = wp
            continue
        n_steps = max(1, int(math.ceil(dist / STEP_SIZE)))
        theta = heading_from_forward(seg_xz[0] / dist, seg_xz[1] / dist)
        rot = heading_to_quat(theta)
        for i in range(1, n_steps + 1):
            frac = i / n_steps
            pos = cur + seg * frac
            s = habitat_sim.AgentState()
            s.position = pos.astype(np.float32)
            s.rotation = rot
            agent.set_state(s)
            frames.append(np.asarray(sim.get_sensor_observations()["color_sensor"])[:, :, :3])
        cur = wp

    final_pos = cur
    final_dist = math.hypot(final_pos[0] - target[0], final_pos[2] - target[2])

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=10, macro_block_size=1)
    return final_dist, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--results-out", required=True)
    args = parser.parse_args()

    scene_glb = f"data/scene_datasets/{args.scene}"
    sim = make_sim(scene_glb)

    with open(args.landmarks) as f:
        landmarks = json.load(f)

    results = []
    ep_id = 0
    total = sum(len(l["targets"]) for l in landmarks)
    for lm in landmarks:
        for relation, tinfo in lm["targets"].items():
            video_path = os.path.join(args.video_dir, f"hybrid_{ep_id}.mp4")
            print(f"[{ep_id + 1}/{total}] {lm['category']} relation={relation}")
            final_dist, n_frames = walk_and_record(
                sim, lm["start_position"], lm["start_rotation"], tinfo["target"], video_path
            )
            results.append({
                "episode_id": ep_id,
                "category": lm["category"],
                "human_name": lm["human_name"],
                "relation": relation,
                "object_center": lm["object_center"],
                "start_position": lm["start_position"],
                "target_position": tinfo["target"],
                "final_distance": final_dist,
                "n_frames": n_frames,
                "video_path": video_path,
            })
            ep_id += 1

    sim.close()
    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} results -> {args.results_out}")

    n_pass = sum(1 for r in results if r["final_distance"] is not None and r["final_distance"] <= SUCCESS_RADIUS)
    print(f"pass: {n_pass}/{len(results)}")


if __name__ == "__main__":
    main()
