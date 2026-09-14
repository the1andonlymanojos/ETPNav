# Free-text instruction → ETPNav sim rollout

## Context

The user wants to type an arbitrary instruction ("move to the left of the table", "move behind the sofa") and watch the ETPNav agent execute it in Habitat-Sim, rather than only running the fixed R2R/RxR benchmark episodes. ETPNav is a Vision-Language-Navigation model trained on step-by-step paragraph instructions (R2R/RxR), driven entirely through `run.py` against pre-built dataset JSONs — there's no existing "type a sentence, watch it move" entry point. This plan adds one.

Verified in this environment: GPU (`T1000`, 8GB) works, `conda env vlnce` has working `habitat-sim` 0.1.7 with headless/EGL rendering (smoke-tested a live sim + navmesh + render), all 90 MP3D scenes and both release checkpoints (`release_r2r`, `release_rxr`) are already downloaded, and `ffmpeg` is available for video encoding.

**Expectation to set with the user up front:** ETPNav was trained on long, step-by-step paragraph instructions, not short spatial referring expressions. It has no explicit notion of "to the left of X" — it will do its best to ground the sentence and pick waypoints, but precise relative-object positioning is out of its training distribution. Treat the first runs as an experiment to see how well it generalizes, not a guaranteed-correct spatial planner.

## How the model is actually invoked (traced through the code)

- Entry point `run.py` → `Trainer` registered as `SS-ETP` (`vlnce_baselines/ss_trainer_ETP.py`), driven by `run_r2r/iter_train.yaml` (`MODEL.policy_name: PolicyViewSelectionETP`).
- Episodes are never constructed in Python — they're read from a gzip JSON matching `TASK_CONFIG.DATASET.DATA_PATH`, format confirmed from `data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/...`:
  ```
  {"episodes": [{episode_id, trajectory_id, scene_id, start_position, start_rotation,
                 info:{geodesic_distance}, goals:[{position, radius}],
                 instruction:{instruction_text, instruction_tokens}, reference_path:[...]}],
   "instruction_vocab": {...}}
  ```
  `instruction_tokens` must be produced the same way training data was: `data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/process_instrs_to_bert_idx.py` shows it's `BertTokenizer(vocab.txt, do_lower_case=True)` → `[CLS] tokens [SEP] [PAD...]` padded to length 80, converted to ids. The vocab file is local (`.../vocab.txt`), so this needs no network access.
- `run_exp()`/`trainer.eval()` (in `vlnce_baselines/common/base_il_trainer.py`) → `_eval_checkpoint()` (overridden in `ss_trainer_ETP.py`) → `rollout('eval')`. This is the path to reuse, **not** `--run-type inference`: `inference()` hardcodes `TASK.MEASUREMENTS = ['POSITION_INFER']`, which omits `SPL`/`SUCCESS`, and the video writer (`vlnce_baselines/common/environments.py`) reads `info["spl"]` unconditionally when `VIDEO_OPTION` is set — that combination throws. `_eval_checkpoint()` correctly appends `TOP_DOWN_MAP_VLNCE, DISTANCE_TO_GOAL, SUCCESS, SPL` to `MEASUREMENTS` whenever `VIDEO_OPTION` is truthy, so eval mode is video-safe as-is.
- `eval()`'s `collect_val_traj()` requires `TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=...)` to exist and uses its keys as the allowed-episode filter, and `rollout()` indexes `self.gt_data[str(episode_id)]['locations']` directly (not gated behind the NDTW measure being enabled) — so a matching GT file with our episode id must exist, even though we don't care about its metrics. Real GT schema confirmed from `data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen_gt.json.gz`: `{"<episode_id>": {"locations": [[x,y,z],...], "forward_steps": int, "actions": [int,...]}}`. A 2-point dummy (`[start, start]`) is sufficient since it's only read, not validated against the model's actual path.
- Video is produced by `vlnce_baselines/common/environments.py::VLNCEDaggerEnv` calling `generate_video(...)` from `habitat_extensions/utils.py` when `done`, writing to `config.VIDEO_DIR` (`data/logs/video/<exp_name>/`) as an mp4 showing the RGB view + the topological node/ghost-waypoint overlay ETPNav actually planned over.
- Default scene/start pose: grepped raw R2R episodes for instruction text containing "table" — `mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb`, `start_position [15.0686, 0.1716, -4.4848]`, `start_rotation [0, 0.6199, 0, -0.7847]` is a real, navmesh-valid pose in a scene whose ground-truth instructions describe a dining table, coffee table, and living room right from that spot — good default for "table"/"sofa"-style prompts.

## Implementation

1. **`scripts/build_custom_episode.py`** — given `--instruction TEXT [--scene SCENE_ID --start x y z --rot w x y z --episode-id ID]`:
   - Tokenizes `TEXT` with `transformers.BertTokenizer(vocab_file="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/vocab.txt", do_lower_case=True)`, pads/truncates to 80 tokens exactly like `process_instrs_to_bert_idx.py`.
   - Writes a single-episode dataset gzip to `data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/custom/custom_bertidx.json.gz`, copying `instruction_vocab` verbatim from the existing `val_unseen_bertidx.json.gz` (dataset loader requires this key).
   - Writes a matching dummy GT gzip to `data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/custom_gt.json.gz` (`locations: [start, start]`, `forward_steps: 0`, `actions: [0]`), keyed by the same episode id (string).
   - Defaults scene/start/rot to the `zsNo4HB9uLZ` pose above when not given.

2. **`scripts/run_instruction.py`** — thin wrapper that calls step 1, then shells out to:
   ```
   python run.py --exp_name custom_demo --run-type eval --exp-config run_r2r/iter_train.yaml \
     SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
     TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
     EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
     EVAL.SPLIT custom EVAL.EPISODE_COUNT 1 \
     TASK_CONFIG.DATASET.DATA_PATH data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz \
     TASK_CONFIG.TASK.NDTW.GT_PATH data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/{split}_gt.json.gz \
     TASK_CONFIG.TASK.SDTW.GT_PATH data/datasets/R2R_VLNCE_v1-2_preprocessed/custom/{split}_gt.json.gz \
     VIDEO_OPTION [\"disk\"] IL.back_algo control \
     MODEL.pretrained_path data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
   ```
   (mirrors `run_r2r/main.bash`'s `eval` flags, minus multi-GPU launch since this is a single episode on one GPU.) Must run inside `conda activate vlnce`.
   - Finds the resulting mp4 under `data/logs/video/custom_demo/` and reports it back (send via `SendUserFile`), plus prints the final stats json from `data/logs/eval_results/custom_demo/`.

3. Smoke-test with the working example instruction ("move to the left of the table") against the default scene, confirm it runs end to end without crashing and produces a playable video, then hand it to the user to try their own phrasing/scenes.

4. If the user wants a different scene than the default, they can pass `--scene <id>` — for now we won't build automatic scene search (e.g. parsing `.house` semantic files for "which of the 90 scenes has a sofa"); we can add that later if picking scenes by hand becomes a pain point.

## Verification

- Run `scripts/run_instruction.py --instruction "move to the left of the table"` end-to-end inside `conda activate vlnce`; confirm no exceptions, a non-empty mp4 is produced, and the stats json shows a sane step count (not 0, not stuck at max steps).
- Visually inspect the video (send to user) to judge whether the agent's motion plausibly relates to the instruction, and report that honestly rather than inferring success from the SPL/SUCCESS numbers (which are meaningless here since the "goal" is a dummy point equal to the start position).
- Try a second instruction on the same scene to confirm the script is reusable without leftover state issues (stale `custom_demo` results dir, etc.).
