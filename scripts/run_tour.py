#!/usr/bin/env python3
"""Run one "tour" (N real R2R-CE episodes in one house, in a fixed order) through the R2R checkpoint and save the
per-episode results. Wraps everything in scripts/guarded_run.sh so it is polite on a shared machine.

  python scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --mode none   --out docs/experiments/exp1/baseline
  python scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --mode window --order-seed 1 --out ...

--mode none = stock behaviour (map wiped every episode). window/full = keep the topological map across episodes
(IL.persist_graph, see vlnce_baselines/models/graph_utils.py). Writes <out>/results.json (+ run.log).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import make_tour_split

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--mode", choices=["none", "window", "full"], default="none")
    p.add_argument("--window-nodes", type=int, default=20)
    p.add_argument("--reloc-radius", type=float, default=1.0)
    p.add_argument("--ghosts", choices=["all", "current"], default="all")
    p.add_argument("--nodes", choices=["all", "current"], default="all")
    p.add_argument("--absorb", choices=["all", "current"], default="all")
    p.add_argument("--reopen-radius", type=float, default=-1.0)
    p.add_argument("--oracle-reopen", action="store_true", help="diagnostic: anchor reopening on the true goal, not the agent")
    p.add_argument("--out", required=True)
    p.add_argument("--ram-cap", default="9G")
    a = p.parse_args()

    name = f"tour_{a.scene[:6]}_n{a.n}_s{a.sample_seed}_o{a.order_seed}"
    exp = f"{name}_{a.mode}" + (f"_w{a.window_nodes}" if a.mode == "window" else "") + ("_ghostscur" if a.ghosts == "current" else "") + ("_nodescur" if a.nodes == "current" else "") + ("_abscur" if a.absorb == "current" else "") + (f"_reopen{a.reopen_radius:g}" if a.reopen_radius >= 0 else "") + ("_oraclegoal" if a.oracle_reopen else "")
    ids = make_tour_split.build(a.scene, a.n, a.sample_seed, a.order_seed, name)
    print(f"[tour] {exp}: intended order {ids}")

    opts = [
        "SIMULATOR_GPU_IDS", "[0]", "TORCH_GPU_IDS", "[0]", "GPU_NUMBERS", "1", "NUM_ENVIRONMENTS", "1",
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING", "True",
        "EVAL.CKPT_PATH_DIR", "data/logs/checkpoints/release_r2r/ckpt.iter12000.pth",
        "EVAL.SPLIT", name, "EVAL.EPISODE_COUNT", str(a.n), "IL.back_algo", "control",
        "MODEL.pretrained_path", "data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt",
    ]
    if a.mode != "none":
        opts += ["IL.persist_graph", a.mode, "IL.persist_window_nodes", str(a.window_nodes),
                 "IL.persist_reloc_radius", str(a.reloc_radius), "IL.persist_ghosts", a.ghosts, "IL.persist_nodes", a.nodes, "IL.persist_absorb", a.absorb,
                 "IL.persist_reopen_radius", str(a.reopen_radius), "IL.persist_reopen_oracle_goal", str(a.oracle_reopen)]
    cmd = ["scripts/guarded_run.sh", sys.executable, "run.py", "--exp_name", exp, "--run-type", "eval",
           "--exp-config", "run_r2r/iter_train.yaml"] + opts

    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ, RAM_CAP=a.ram_cap, ETP_KEEP_EPISODE_ORDER="1")  # dataset loader shuffles otherwise
    t0 = time.time()
    with open(os.path.join(a.out, "run.log"), "w") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        print(f"[tour] run failed rc={rc}; see {a.out}/run.log")
        sys.exit(rc)

    ep_file = glob.glob(os.path.join(REPO, "data/logs/eval_results", exp, "stats_ep_ckpt_*_r0_w1.json"))[0]
    per_ep = json.load(open(ep_file))          # insertion order == evaluation order
    result = {"exp": exp, "scene": a.scene, "mode": a.mode, "n": a.n, "sample_seed": a.sample_seed,
              "order_seed": a.order_seed, "window_nodes": a.window_nodes, "reloc_radius": a.reloc_radius, "ghosts": a.ghosts, "nodes": a.nodes, "absorb": a.absorb,
              "reopen_radius": a.reopen_radius, "oracle_reopen": a.oracle_reopen,
              "intended_order": ids, "eval_order": list(per_ep.keys()), "wall_seconds": round(wall, 1),
              "episodes": per_ep}
    json.dump(result, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    dbg = os.path.join(REPO, "data/logs/eval_results", exp, "persist_debug.json")   # per-step graph trace + reloc links
    if os.path.exists(dbg):
        import shutil
        shutil.copy(dbg, os.path.join(a.out, "persist_debug.json"))
    print(f"[tour] done in {wall:.0f}s -> {a.out}/results.json  (eval order matches intended: "
          f"{result['eval_order'] == ids})")


if __name__ == "__main__":
    main()
