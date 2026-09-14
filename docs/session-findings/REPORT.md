# Session findings: free-text instruction navigation on ETPNav

This directory is a complete archive of one working session (2026-09-11 to 2026-09-15) spent
extending this ETPNav checkout so it can execute arbitrary free-text instructions in the
Habitat simulator, evaluate it against the real R2R-CE/RxR-CE benchmarks, and probe its
spatial-reasoning limits. The machine this was done on was scheduled for destruction, so
everything needed to pick this up on a different machine is here: reusable scripts (in
`scripts/` at the repo root), raw results, videos, run logs, the exact heading-convention math,
every bug hit and how it was fixed, and links to the published (Anthropic-hosted, independent of
this machine) result pages.

**Read this file first if you're an agent picking this up cold.**

## TL;DR

- `scripts/*.py` (repo root `scripts/`) let you feed ETPNav an arbitrary instruction + scene +
  start pose and get a video + real distance-to-goal metrics back, for both the R2R and RxR
  checkpoints. See "Reusable scripts" below.
- ETPNav's pretrained checkpoints (`release_r2r`, `release_rxr`) roughly reproduce the paper's
  numbers when evaluated properly (see "Benchmark validation" below) — the pipeline is correctly
  configured, not silently broken.
- Both checkpoints **fail badly** (~8% pass) at short egocentric spatial commands like "move to
  the left of the sofa" — this is a real, well-evidenced finding, not a bug. Rephrasing the
  instruction (more explicit, longer, narrated) **does not fix it** — see "Spatial reasoning
  findings" below for the full chain of experiments that established this.
- A "hybrid grounding" bypass (look up the object's real position from the scene's semantic
  annotations, skip the language model, walk straight there) trivially gets 100% — but this is a
  privileged-coordinates pathfinding demo, not an improved model. See the caveats in that
  section; two concrete non-trivial follow-ups are listed and were **not** built yet.

## Environment (what's already set up on the original machine)

- Conda env `vlnce` (Python 3.6, habitat-sim 0.1.7, habitat-lab v0.1.7 at
  `~/habitat-lab-v0.1.7`, torch 1.9.1+cu111). Activate with:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlnce
  ```
- GPU: single NVIDIA T1000, 8GB. Everything in this session ran on `SIMULATOR_GPU_IDS [0]
  TORCH_GPU_IDS [0]` — no multi-GPU needed for inference/eval (only the paper's *training* run
  used 2x RTX 3090; we never trained anything, only ran the released checkpoints).
- Data already downloaded under `data/` (git-ignored, `du -sh`: `data/logs/checkpoints` 5.8G,
  `data/pretrained` 2.7G, `data/datasets` 1.1G, `data/ddppo-models` 584M, `data/wp_pred` 386M) —
  all 90 MP3D scenes present under `data/scene_datasets/mp3d/`. See `SETUP.md` and the main
  `README.md` for how this was originally populated (ModelScope for checkpoints, MP3D's own
  download script for scenes — see `download_mp3d.py` in this directory, copied from
  `~/dataset_download.py` on the original machine; it's the official, public, unmodified MP3D
  downloader, no embedded credentials — run it from a **Python 2** environment: `python
  download_mp3d.py -o data/scene_datasets -task habitat --id ALL`, and see `SETUP.md`/README for
  the exact folder layout it expects).
- `conda env export -n vlnce` is saved as `vlnce_environment.yml` in this directory for exact
  reproducibility (note: this pins exact build strings, which can be platform-fragile — treat it
  as a reference, not something to blindly `conda env create -f` on a different OS/driver combo;
  `environment.yaml` at the repo root is the portable one to actually use, per SETUP.md).
- `gh` (GitHub CLI) was authenticated as `the1andonlymanojos` at the *start* of this session but
  the token had gone invalid by the end (`gh auth status` → "token is invalid"). `git remote -v`:
  `origin` = `https://github.com/the1andonlymanojos/ETPNav.git` (this user's fork — push here),
  `upstream` = `https://github.com/MarSaKi/ETPNav.git` (the original paper repo — **never push
  here**). If you're continuing this work and pushing fails, re-auth with `gh auth login`.

## Reusable scripts (repo root `scripts/`)

All of these must run inside `conda activate vlnce`, from the repo root. They all follow the same
pattern: build a single fabricated episode (custom instruction + scene + start pose + a goal
position used only so the eval code's distance/SPL measures don't crash), then shell out to the
repo's own `run.py --run-type eval` to get a real forward pass + video, then read the resulting
stats JSON back.

- **`build_custom_episode.py`** — core builder for the R2R checkpoint's tokenizer/dataset format
  (BERT, local vocab at `data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/vocab.txt`). Writes a
  fabricated single-episode dataset + GT file under `data/datasets/.../custom/`. `build()` takes
  an explicit `goal_position` — pass the real target you want distance-to-goal computed against
  (see "the goal trick" below); omitted, it falls back to a known-good default point in scene
  `zsNo4HB9uLZ`.
- **`run_instruction.py`** — CLI wrapper: `python scripts/run_instruction.py --instruction "move
  to the left of the table" [--scene mp3d/X/X.glb --start x y z --rot w x y z --exp-name NAME]`.
  Uses the `release_r2r` checkpoint. Produces `data/logs/video/<exp-name>/*.mp4` and
  `data/logs/eval_results/<exp-name>/stats_ckpt_*_custom.json`.
- **`build_custom_episode_rxr.py`** / **`run_instruction_rxr.py`** — same idea for the
  `release_rxr` checkpoint. Key differences: XLM-RoBERTa tokenizer (downloads
  `xlm-roberta-base` from HuggingFace on first use — **needs internet**, unlike the R2R BERT
  tokenizer which is fully local), role-sharded dataset files (`{split}_{role}.json.gz`, role
  always `guide`), `episode_id` **must be a string** (see bug list), `INSTRUCTION_SENSOR_UUID` is
  still `"instruction"` even though the sensor class is `RxRInstructionSensor`.
- **`build_spatial_landmarks.py`** — given an MP3D scene, loads its real semantic-object graph
  via habitat-sim (`load_semantic_mesh=True`), finds landmark objects (sofa/table/bed/shelving/
  counter/chest_of_drawers/tv_monitor), and for each one computes a start pose + geometric target
  point for "left/right/behind/next to" relative to that object — all validated for navmesh
  reachability first. `python scripts/build_spatial_landmarks.py --scene
  mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb --out landmarks.json --max-landmarks 10`.
- **`run_spatial_batch.py`** — consumes that landmarks JSON, runs one R2R-checkpoint eval per
  landmark×relation with the geometric target as the goal, collects real distance-to-target.
- **`run_phrasing_experiment.py`** — same idea but takes an explicit list of
  `{start/rotation/target, instruction}` cases (see `build_phrasing_cases.py` for the format) so
  you can rerun the *same* start/goal with *different instruction wordings*.
- **`run_hybrid_grounding.py`** — **does not use the VLN model at all**. Given the same landmarks
  JSON, walks the agent from start to target using habitat-sim's own `ShortestPath` pathfinder,
  interpolating position directly and recording frames. Pure geometry/pathfinding baseline — see
  the "hybrid grounding" section for what this does and doesn't demonstrate.

### The "goal" trick, and why it's needed

The repo's `eval` run-type computes `success`/`SPL`/`distance_to_goal` from
`episode.goals[0].position` via the `POSITION` habitat measure — there's no way to just "run an
instruction and watch," the eval code needs *some* goal or it errors/produces garbage. Early on
we used a fixed real point in the scene as a dummy goal just to avoid a crash (division by zero
when goal == start — see bug list). Later, for the spatial-relation experiments, we **reused this
same mechanism on purpose**: by setting `goal_position` to the actual geometrically-computed
target ("2m to the left of the sofa"), `distance_to_goal` becomes a real, meaningful measurement
instead of a throwaway. This is why `build_custom_episode.build()` takes an explicit
`goal_position` parameter.

### Heading/quaternion convention (verified empirically, don't re-derive from scratch)

Habitat's `start_rotation` is `[x, y, z, w]`, yaw-only about Y. Verified by placing an agent at a
known heading and checking which way `move_forward` actually moves it:

```
forward(theta) = (-sin(theta), -cos(theta))   in the world XZ plane
quaternion (x, y, z, w) = (0, sin(theta/2), 0, cos(theta/2))
heading_from_forward(fx, fz) = atan2(-fx, -fz)
right_vec = (-forward_z, forward_x)   given forward_vec = (forward_x, forward_z)
```

This is implemented in `build_spatial_landmarks.py` and `run_hybrid_grounding.py`. Do not
re-guess this — it was wrong on the first attempt (calibrating against a real R2R episode's
first-step displacement, which turned out unreliable since agents can turn before moving) and
only became trustworthy once verified directly against `sim.step("move_forward")`.

## Bugs found and fixed this session (so nobody re-discovers them)

1. **`BertTokenizer.from_pretrained` needs a local file path, not a HF shortcut name**, on this
   old `transformers` version — the repo's own preprocessing script
   (`data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/process_instrs_to_bert_idx.py`) uses an
   import path (`transformers.pytorch_transformers.BertTokenizer`) that no longer exists in the
   installed `transformers==4.12.5`; use `from transformers import BertTokenizer` and pass the
   local `vocab.txt` path.
2. **`habitat.VectorEnv` (multiprocessing) silently swallows worker exceptions** — the main
   process just sees `EOFError` on `recv_bytes()` with no traceback. To get the real error,
   force single-process `ThreadedVectorEnv` by making `sys.gettrace()` truthy (the codebase
   checks this to decide which VectorEnv class to use) — e.g. `sys.settrace(lambda *a: None)`
   before importing/running `run.py`. Also make sure to run with unbuffered output
   (`PYTHONUNBUFFERED=1 python -u ...`) piped to a file you read after — buffered stdout can hide
   the last, most important lines of a crash if the process is killed by a timeout.
3. **`ZeroDivisionError` in the SPL calc** when a fabricated goal position equals the start
   position (geodesic distance 0). Fix: always use a real, *distinct* navigable point in the same
   scene as the dummy goal when you don't care about the metric.
4. **RxR custom episodes silently vanish (0 episodes loaded, no error) if `episode_id` is an
   int** instead of a string. `habitat.core.dataset.Dataset`'s `EPISODES_ALLOWED` filter (built
   from GT-file JSON keys, which are always strings) does a set-difference against the raw
   `episode.episode_id` value; if that's an `int` (as JSON naturally parses `"episode_id": 0`),
   it never matches the string filter and the episode gets purged with no warning. The R2R path
   happens not to hit this (unclear why — possibly a different code path); for RxR, always write
   `episode_id` as `str(...)`.
5. **A naive AABB-derived floor-height guess for snapping a start pose to the navmesh can land on
   a disconnected navmesh island** (different floor/closet/stairwell), making every subsequent
   target geodesically unreachable (`distance_to_goal: inf`, no crash, just silently wrong
   metrics). Fix: snap the *landmark object's own center* to the navmesh first to get a reliable
   local floor height, and explicitly validate reachability (`habitat_sim.ShortestPath` +
   `find_path`) before accepting a start/target pair — see `build_spatial_landmarks.py`.
6. **`os.listdir()` order is not guaranteed** — a video-file lookup that matched by filename
   prefix only (not extension) occasionally grabbed a multi-MB `.png` topdown-map thumbnail
   instead of the intended `.mp4` (both share the same prefix), bloating one artifact to 23MB
   before the extension filter was added. Always filter by extension explicitly when multiple
   file types share a naming prefix.
7. **habitat-sim doesn't load semantic object annotations by default** — even though every MP3D
   scene has a matching `.house` file with real object positions/categories, you get "the active
   scene does not contain semantic annotations" and an empty `sim.semantic_scene.objects` unless
   you explicitly set `SimulatorConfiguration.load_semantic_mesh = True`.

## Benchmark validation against the paper

Two real-data batches were run (not fabricated instructions) to check the pipeline reproduces the
paper's own numbers — see the ETPNav paper (TPAMI 2024, sections 4.1–4.2, Tables 2/3/9; a copy
should be sourced separately, e.g. `2210.05714v4.pdf` if present on the original machine's home
directory — **not copied here, check paper licensing before redistributing**).

- **RxR-CE `val_unseen`, 20 episodes (single scene, English only)**: SR 50.0%, SPL 42.9%, NDTW
  47.1%.
- **RxR-CE `val_unseen`, 51 episodes (11 scenes, English only, stratified by instruction
  length)**: SR 54.9%, SPL 46.8%, NDTW 59.9%.

These sit close to the paper's own **Table 9 "Heuristic w/ Tryout" controller ablation** row (the
exact controller config used throughout this session — `IL.back_algo control`, sliding forbidden,
0.18m chassis): SR 54.79%, SPL 44.89%, NDTW 61.90%. The paper's *headline* Table 3 number for the
full model is higher (SR 61.46%, SPL 50.83%, NDTW 66.41%) — the gap is well explained by: (a)
English-only filtering here (paper's average includes Hindi/Telugu), (b) small n (20–51 vs.
thousands), (c) the 51-episode batch was deliberately stratified toward a spread of instruction
lengths rather than drawn uniformly at random. **Conclusion: the eval pipeline is correctly
configured and reproducing genuine paper-consistent behavior**, not silently broken.

No R2R-CE benchmark batch was run *in this session* — only custom out-of-distribution instructions
were run on that checkpoint here. However, `results/release_r2r/stats_ckpt_59_val_unseen.json`
(dated 2026-08-27, i.e. **predates this session** — this is the repo owner's own prior
sanity-check run, not something generated in this session, so don't attribute it here) already has
a real 50-episode R2R-CE `val_unseen` result on this machine: SR 68%, SPL 61.4%, NDTW 73.2% — close
to the paper's Table 2 headline (SR 66.19%, SPL 59.37%). Kept here since it's directly relevant
and was just sitting in `data/logs/eval_results/` (git-ignored, so it would otherwise have been
lost with the machine). If picking this up again and wanting a fresh/larger R2R-CE check:
```bash
python run.py --exp_name r2r_benchmark_check --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT 100 \
  VIDEO_OPTION "['disk']" IL.back_algo control \
  MODEL.pretrained_path data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
```

## Spatial reasoning findings (the main result of this session)

Chain of experiments, each motivating the next:

1. **Free-text instructions work end-to-end** (both checkpoints), but short instructions on the
   RxR checkpoint collapse to *identical* low-level trajectories regardless of wording — greedy
   (argmax) decoding + weak language grounding for short commands means the model falls back to
   one fixed exploration pattern. → "Instruction Rollouts" / "RxR Instruction Rollouts" artifacts.
2. **Real benchmark batches** confirm the pipeline itself is sound (see above).
3. **Custom Extremely Spatial Instructions Test** — 25 instructions ("move to the left of the
   X" / "right of" / "behind" / "next to", X = a real landmark object found via the scene's
   semantic graph) against the R2R checkpoint, scored by real geometric distance to a computed
   target (not the engine's built-in 3m success radius, which is too loose to distinguish
   relations only 2–3m apart around the same object — pass threshold used: ≤1.5m).
   **Result: 2/25 passed (8%)**. Raw data: `results/spatial_results.json`,
   `results/landmarks_zsNo4HB9uLZ.json`. Generator: `artifact_generators/build_spatial_artifact.py`.
4. **Spatial Prompt Reformulation Test** — took 4 of the clearest misses from (3) and reran each
   with 5 phrasings (bare command → verb+stop → two-step narrated → landmark-anchored → full
   R2R-style narration), same start pose and target each time. **Result: for 3 of 4 cases, final
   distance-to-target was identical to two decimal places across all five phrasings** — rewording
   does not change the outcome. The one case with any spread showed no monotonic trend (the
   longest, most explicit phrasing tied the *worst* result). One case (dresser/"behind") turned
   out to be an unrelated deadlock — 210+ steps to cover a net 1.68m, collision rate >10 — the
   same Tryout-controller deadlock failure mode the paper documents qualitatively in its own
   §4.3.4/Fig. 6. Raw data: `results/phrasing_results.json`, `results/phrasing_cases.json`.
   Generator: `artifact_generators/build_phrasing_artifact.py`.
   **Conclusion: the failure is not primarily a prompt-clarity problem.** The model finds the
   right *neighborhood* (visible in the videos — it heads toward the correct landmark) but
   doesn't resolve "which side," most likely because R2R/RxR training instructions essentially
   never ask an agent to stop at a specific offset from a named object, so the model never learned
   that distinction — no amount of rephrasing recovers a distinction that was never trained.
5. **Hybrid Grounding Before/After** — re-ran the same 25 cases from (3), this time bypassing
   ETPNav's language model entirely: parse relation + object, look up the real position (same
   semantic-scene lookup as (3)), walk straight there via `habitat_sim.ShortestPath`.
   **Result: 25/25 (100%)**, expected and mechanically guaranteed since the walked-to coordinate
   *is* the scoring target. **Explicitly flagged as not meaningful on its own** (both in the
   artifact and when the user called this out directly mid-session): it uses privileged
   ground-truth coordinates no real system has, and it isn't even testing a controller — it's
   direct position interpolation, not discrete actions or obstacle avoidance. What it *does*
   show: a cheap, practical mitigation for this narrow instruction class if you're willing to add
   an object-lookup step, not a research result. **Two concrete, harder follow-ups were proposed
   but not built** — if continuing this thread, these are the next real steps:
   - Replace the privileged coordinate lookup with actual vision-based grounding (e.g. an
     open-vocabulary detector over the agent's RGB frames) so the target has to be *found*, not
     looked up.
   - Replace the pathfinder-interpolation "walk" with ETPNav's own PointGoal or Heuristic+Tryout
     controller (both already in this codebase, same ones from the paper's Table 9) driving
     toward the found target, so real obstacle avoidance/collision handling is actually exercised
     instead of teleport-interpolation.

## Published artifacts (hosted by Anthropic, independent of this machine)

These are private Claude artifacts under this user's account — they survive regardless of what
happens to this machine, but the URLs are recorded here as a durable index. All were built by the
generator scripts + templates in `artifact_generators/` (three early ones' generator scripts were
lost to a scratchpad rotation between session teleports — see note below — but their source
videos/stats are preserved in `videos/` and `results/` and the artifacts themselves remain live).

| # | Artifact | URL | Generator preserved? |
|---|----------|-----|----|
| 1 | Instruction Rollouts (3 R2R custom runs) | https://claude.ai/code/artifact/418cc0f1-6f8e-4d5d-b848-46a0a7096150 | No — videos in `videos/custom_demo*` |
| 2 | RxR Instruction Rollouts (4 RxR custom runs) | https://claude.ai/code/artifact/579c5555-a92d-4ecd-adeb-a9dc6d43e205 | No — videos in `videos/rxr_*` |
| 3 | RxR Val Unseen Results (10 real episodes, success/fail) | https://claude.ai/code/artifact/756cf027-48c5-44c2-a014-d92d1b558e37 | No — videos in `videos/real_batch_rxr` |
| 4 | RxR Task Gallery (51 diverse real episodes, 11 scenes) | https://claude.ai/code/artifact/74ab085d-4e40-4b5d-9985-961ccd258bf1 | Yes — `build_gallery.py` |
| 5 | Custom Extremely Spatial Instructions Test (25 landmark probes) | https://claude.ai/code/artifact/8b48e08f-580d-4d4b-ae01-19c6fc2c7d59 | Yes — `build_spatial_artifact.py` |
| 6 | Spatial Prompt Reformulation Test (20 phrasing runs) | https://claude.ai/code/artifact/5a6869a9-e1b7-4293-bcae-f0731a4fed47 | Yes — `build_phrasing_artifact.py` |
| 7 | Hybrid Grounding Before/After (25 pairs) | https://claude.ai/code/artifact/71b771b8-fd48-4637-85a0-0b950dc10331 | Yes — `build_hybrid_artifact.py` |

Note on "generator preserved": for #1–3, the Python scripts that stitched the raw videos/stats
into the published HTML were written to a `/tmp` scratchpad directory that rotated away between
two session teleports mid-conversation and were not recovered. The *raw material* (every video,
every stats JSON) for all seven is fully preserved under `videos/` and `results/` in this
directory — regenerating equivalent pages for #1–3 would mean writing a new (simple) templating
script following the exact pattern already demonstrated by #4–7's preserved generators.

## Directory layout of this archive

```
docs/session-findings/
  REPORT.md                    <- this file
  download_mp3d.py             <- official MP3D dataset downloader (run under Python 2)
  vlnce_environment.yml        <- `conda env export -n vlnce` snapshot
  squishy-popping-cupcake.md   <- original implementation plan (R2R free-text pipeline design)
  videos/<exp-name>/*.mp4      <- every rollout video generated this session, 136 files, ~83MB
  results/                     <- every raw stats/selection JSON, eval_results/ contents, ~1.9MB
  run_logs/                    <- full stdout logs of the three big batch runs
  artifact_generators/         <- Python + HTML template pairs that build 4 of the 7 artifacts,
                                   plus copies of the 4 published HTML pages themselves
```
