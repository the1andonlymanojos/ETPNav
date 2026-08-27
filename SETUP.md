# Local setup notes

This documents how this repo was actually gotten running on a single-GPU
machine (1x NVIDIA T1000, 8GB VRAM), including the dependency-pinning issues
that had to be fixed along the way. The README's install steps are correct
in spirit but pin some now-dead package versions and have a couple of
release-script bugs — this fills in the gaps.

## 0. Prerequisites assumed present

- conda/miniconda
- NVIDIA driver + CUDA-capable GPU (the exact driver/CUDA runtime version
  doesn't need to match `cu111` below — the older PyTorch CUDA 11.1 wheel
  runs fine on newer drivers)
- Matterport3D habitat scene meshes already downloaded somewhere as
  `{scene}/{scene}.glb` (90 scenes) — see step 4 if you need to source them

## 1. Create the `vlnce` conda environment

```bash
conda env create -f environment.yaml
conda activate vlnce
```

**Pinning issue found:** `environment.yaml` pinned
`tb-nightly==2.8.0a20220108`, an exact TensorBoard nightly build that is no
longer published on PyPI (`pip install` fails with "No matching
distribution found"). It also directly conflicts with the file's own
`tensorboard==1.13.1` pin. **Fix applied:** removed the `tb-nightly` line
entirely; `tensorboard==1.13.1` alone is sufficient (a separate `tb-nightly`
gets reinstalled anyway by `habitat-lab`'s own requirements in step 3, and
that later install works fine since it doesn't pin an exact dead build).

## 2. Install `habitat-sim` 0.1.7, headless build

```bash
conda install -n vlnce -c aihabitat -c conda-forge \
  habitat-sim=0.1.7=py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
```

**Gotcha found:** installing with the loose spec from the README
(`habitat-sim=0.1.7 headless`) resolved on a modern conda (26.x) to the
**plain GLX-linked build** (`py3.6_linux_...`), not the headless/EGL one —
even though the `headless` metapackage and `habitat-sim-mutex=headless_nobullet`
were correctly present in the solved environment. The mutex/metapackage
matched, but the solver still picked the wrong concrete `habitat-sim` build.
This silently breaks any machine without an X display, since the loaded
`.so` links `libGLX` instead of `libEGL` (verify with
`ldd .../habitat_sim_bindings*.so | grep -i gl`). **Fix:** pin the exact
build string above rather than the loose `headless` label.

## 3. Install `habitat-lab` v0.1.7

```bash
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
# comment out the `tensorflow==1.13.1` line in this file first (per README):
#   habitat_baselines/rl/requirements.txt
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

This step's own `requirements.txt` bumps `gym` to a newer version and
pulls in a generic `tb-nightly` (a *different*, still-published build, not
the dead pin from step 1) — both get corrected in step 4.

## 4. Install exact torch / CLIP / gym versions

**Must run in this order**, after steps 1–3, since habitat-lab's installer
overwrites some of these with the wrong versions:

```bash
cd ETPNav
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/openai/CLIP.git
pip install gym==0.21.0
```

Notes:
- `environment.yaml`'s pip section installs a PyPI package that is *also*
  named `clip` (unrelated — a leftover, unmaintained package) as a
  transitive dependency. Installing the real OpenAI CLIP from git in this
  step correctly overwrites it.
- Skipping the explicit `gym==0.21.0` re-pin leaves whatever version
  habitat-lab's installer chose (currently 0.26.2), which is not what
  habitat-lab v0.1.7 / this codebase's action-space code expects.

## 5. Get the Matterport3D scenes into `data/scene_datasets/mp3d`

If you already have the official MP3D habitat release (`mp3d_habitat.zip`
from `download_mp.py --task habitat`) extracted somewhere as
`{scene}/{scene}.glb` x90, skip the official downloader entirely and just
symlink it in:

```bash
mkdir -p data/scene_datasets
ln -s /path/to/existing/mp3d data/scene_datasets/mp3d
```

Verify all 90 scenes are visible: `ls data/scene_datasets/mp3d | wc -l`.

## 6. Download ETPNav weights + preprocessed datasets

`modelscope` requires Python >= 3.10, but the `vlnce` env is Python 3.6 —
install it into a separate throwaway venv (using the system Python) rather
than fighting the conda env's version:

```bash
python3 -m venv ~/.modelscope_dl_venv
~/.modelscope_dl_venv/bin/pip install modelscope
~/.modelscope_dl_venv/bin/modelscope download --model admagic/ETPNav --local_dir ./data
```

This pulls ~11GB into `data/`: `connectivity_graphs.pkl`, `datasets/`
(R2R-CE + RxR-CE, raw and preprocessed), `ddppo-models/` (depth encoder
weights), `pretrained/ETP/` (both R2R and RxR pretrained checkpoints),
`wp_pred/` (waypoint predictor weights), `logs/checkpoints/` (released
finetuned checkpoints for both tasks).

## 7. Bug fix: `MODEL.pretrained_path` missing from eval/infer

`run_r2r/main.bash` and `run_rxr/main.bash`'s `train` flag correctly sets
`MODEL.pretrained_path`, but their `eval` and `infer` flags never did —
even though the model architecture is initialized identically regardless of
run mode (the eval checkpoint gets loaded *on top of* this base init). Left
as-is, `eval`/`infer` crash with:

```
FileNotFoundError: [Errno 2] No such file or directory: 'pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt'
```

(missing the `data/` prefix — it's silently falling back to the config
file's un-prefixed default). **Fixed** by adding the same
`MODEL.pretrained_path data/pretrained/ETP/mlm.sap_{r2r,rxr}/ckpts/model_step_{82500,90000}.pt`
line to the `eval` and `infer` flags in both `main.bash` scripts, matching
`train`.

## 8. Running on a single GPU

`main.bash`'s flags are hardcoded for the paper's multi-GPU setup
(`GPU_NUMBERS 2`/`4`, `torch.distributed.launch --nproc_per_node=2`/`4`).
On a single GPU, call `run.py` directly (skip `torch.distributed.launch`
entirely — `GPU_NUMBERS 1` makes the trainer skip NCCL init, so no
distributed launcher is needed) with the GPU-count flags overridden:

```bash
conda activate vlnce
python run.py \
  --exp_name release_r2r \
  --run-type eval \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" \
  TORCH_GPU_IDS "[0]" \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 1 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
  IL.back_algo control \
  MODEL.pretrained_path data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
```

(The last line becomes unnecessary if you've applied the `main.bash` fix
from step 7 and run through that script, but is included here since running
`run.py` directly bypasses `main.bash` altogether.)

Note: `IL.back_algo control`'s trial-and-error obstacle-recovery heuristic
(`single_step_control`'s `tryout` path in
`vlnce_baselines/common/environments.py`) is only active when
`ALLOW_SLIDING=False`. R2R's released eval/infer flags use
`ALLOW_SLIDING=True` (so that heuristic never actually runs for R2R), while
RxR's use `ALLOW_SLIDING=False` (so it does run there). Worth knowing before
trying to tune it.

## 9. Sanity check

A 50-episode eval on R2R-CE `val_unseen` with the released `ckpt.iter12000.pth`
checkpoint should land close to:

| success | oracle_success | spl | ndtw | sdtw |
|---|---|---|---|---|
| ~0.70 | ~0.72 | ~0.64 | ~0.74 | ~0.59 |

which is consistent with the paper's reported ~72% SR / 0.61 SPL on the full
1839-episode split — confirms the environment, weights, and scene data are
all wired up correctly before committing to a full ~2 hour eval run.
