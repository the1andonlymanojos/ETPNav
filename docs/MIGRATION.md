# Migrating this setup to another machine

Written 2026-09-20 from an inventory of the original machine (`storms-end@192.168.83.210`, Ubuntu,
NVIDIA T1000 — the GPU was detached when this was written). Everything code-shaped is already on
GitHub; this doc covers the ~38 GiB that is **not** (weights, datasets, MP3D scenes, the conda env).

## 1. What lives where

| Item | Size | Old-machine path | On GitHub? | Action |
|---|---|---|---|---|
| Code, `scripts/`, `docs/session-findings/` (report, results, 136 videos, generators) | ~125 MB | `~/ETPNav` | yes | `git clone` |
| **MP3D scenes** (90 houses: `.glb .navmesh .house _semantic.ply`) | **21 GB** | `/home/storms-end/vlmaps_cc/mp3d_habitat/mp3d` (symlinked from `ETPNav/data/scene_datasets/mp3d`) | no | rsync, **resolve the symlink** |
| Release checkpoints (R2R 2.1 GB, RxR 4.1 GB) | 6.2 GB | `ETPNav/data/logs/checkpoints/release_{r2r,rxr}/` | no | rsync |
| Pretrained MLM+SAP weights | 2.7 GB | `ETPNav/data/pretrained/` | no | rsync |
| DD-PPO depth encoders | 584 MB | `ETPNav/data/ddppo-models/` | no | rsync |
| Waypoint predictor weights | 386 MB | `ETPNav/data/wp_pred/` | no | rsync |
| R2R / RxR datasets + preprocessed | 1.1 GB | `ETPNav/data/datasets/` | no | rsync |
| `connectivity_graphs.pkl`, `configuration.json` | 2 MB | `ETPNav/data/` | no | rsync |
| habitat-lab v0.1.7 (editable install, 1 local edit) | 34 MB | `~/habitat-lab-v0.1.7` | upstream tag + edit below | rsync |
| conda env `vlnce` | 6.4 GB | `~/miniconda3/envs/vlnce` | spec only (`docs/session-findings/vlnce_environment.yml`) | see section 4 |
| CLIP ViT-B/32 weights (optional, auto-downloads) | 338 MB | `~/.cache/clip/` | n/a | rsync |

Total ≈ 38 GiB with the env, ≈ 32 GiB without. Keep ~50 GB free on the new machine.

**Don't copy:** `data/logs/video` and `data/logs/eval_results` (the mp4s/JSON are already in
`docs/session-findings/`; the PNG thumbnails are regenerable), the ~70 empty per-run dirs under
`data/logs/checkpoints/` (the trainer recreates them), `tensorboard_dirs`, `running_log`, the `py2`
env (only needed to re-run the MP3D downloader), `~/.cache/huggingface` (unrelated models). Never
copy `~/.config/gh` or `~/.ssh/id_*` — run `gh auth login` and `ssh-keygen` fresh on the new machine.

The one local edit in habitat-lab (comment out the TensorFlow pin, per the main README):
```diff
--- a/habitat_baselines/rl/requirements.txt
-tensorflow==1.13.1
+# tensorflow==1.13.1
```

## 2. Transfer (pull mode — run everything below on the NEW machine)

The old machine has `sshd` running. `rsync` beats `scp` here: resumable, and re-runs skip files
already copied. No `-z`: weights and meshes are already compressed, it only burns CPU. Run it in
`tmux` so a dropped SSH session doesn't matter.

```bash
# one-time: let the new machine log into the old one
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519      # skip if you already have a key
ssh-copy-id storms-end@192.168.83.210                 # needs password auth on the old box, or paste the
                                                      # .pub into its ~/.ssh/authorized_keys
SRC=storms-end@192.168.83.210
R="rsync -a --partial --info=progress2 -h"

# code, scripts, docs, results, videos (already on GitHub)
git clone https://github.com/the1andonlymanojos/ETPNav.git ~/ETPNav
mkdir -p ~/ETPNav/data/scene_datasets ~/ETPNav/data/logs/checkpoints/{release_r2r,release_rxr}

# MP3D scenes, 21 GB — copied as a REAL directory (not the symlink) straight to where the repo expects it
$R $SRC:/home/storms-end/vlmaps_cc/mp3d_habitat/mp3d/ ~/ETPNav/data/scene_datasets/mp3d/

# datasets, pretrained, ddppo-models, wp_pred, pkl/json  (~4.8 GB; skips scenes + logs)
$R --exclude='/scene_datasets' --exclude='/logs' $SRC:/home/storms-end/ETPNav/data/ ~/ETPNav/data/

# the two release checkpoints (6.2 GB)
$R $SRC:/home/storms-end/ETPNav/data/logs/checkpoints/release_r2r/ckpt.iter12000.pth ~/ETPNav/data/logs/checkpoints/release_r2r/
$R $SRC:/home/storms-end/ETPNav/data/logs/checkpoints/release_rxr/ckpt.iter19600.pth ~/ETPNav/data/logs/checkpoints/release_rxr/

# habitat-lab (editable install source, keeps the local edit)
$R $SRC:/home/storms-end/habitat-lab-v0.1.7/ ~/habitat-lab-v0.1.7/

# optional: CLIP weights, paper PDF
$R $SRC:/home/storms-end/.cache/clip/ ~/.cache/clip/
$R $SRC:/home/storms-end/2210.05714v4.pdf ~/
```

Verify any of them: re-run the same command with `-n --checksum` added — no filenames listed
means byte-identical.

**Push mode instead** (run on the old machine): same commands with source and destination swapped,
e.g. `rsync -a --partial --info=progress2 -h /home/storms-end/vlmaps_cc/mp3d_habitat/mp3d/ NEWUSER@NEWIP:~/ETPNav/data/scene_datasets/mp3d/`.

## 3. Other things on the old machine that are NOT this project

`bulk.json` (3.4 GB), `collection.tar.gz` (1 GB), `hm3d-minival-*.tar` (~0.9 GB), `~/vlmaps2` (3 GB),
`~/vlmaps_cc` minus the MP3D scenes (~3 GB), `~/datasets` (HM3D minival, 1.4 GB), `~/IRE`,
`~/hab2`, `~/Hierarchical-Localization`, `~/elastic-start-local`. Copy them the same way if you
want them; none are needed for ETPNav.

## 4. The conda env (pick one)

- **A. Same username + `~/miniconda3` path on the new machine → just rsync it (simplest):**
  `$R $SRC:/home/storms-end/miniconda3/envs/vlnce/ ~/miniconda3/envs/vlnce/`
  Conda envs embed absolute prefixes, so this only works if the path is identical
  (`/home/storms-end/miniconda3/envs/vlnce`). Install Miniconda first.
- **B. Different username/path → `conda-pack` (relocatable).** On the old machine:
  `pip install conda-pack && conda pack -n vlnce -o /tmp/vlnce.tar.gz`, copy it over, then on the new one:
  `mkdir -p ~/miniconda3/envs/vlnce && tar -xzf vlnce.tar.gz -C ~/miniconda3/envs/vlnce && ~/miniconda3/envs/vlnce/bin/conda-unpack`
- **C. Rebuild from spec (needs internet, least certain).** `conda env create -f docs/session-findings/vlnce_environment.yml`
  (pinned build strings — can fail on a different OS). Fallback: follow the main README install steps
  (`environment.yaml`, then `conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless`,
  `pip install torch==1.9.1+cu111 ... gym==0.21.0`, `pip install git+https://github.com/openai/CLIP.git`).

habitat-lab is an **editable install pointing at an absolute path** (`/home/storms-end/habitat-lab-v0.1.7`).
If the new path differs (options B/C): `cd ~/habitat-lab-v0.1.7 && python setup.py develop --all`
inside the activated env.

## 5. Gotchas

1. **The MP3D symlink.** `rsync -a ETPNav/data/ ...` copies only the *link*, which points at a path
   that won't exist on the new machine. The commands above copy the scenes explicitly.
2. **GPU generation matters.** The env is `torch 1.9.1+cu111` + `habitat-sim 0.1.7 (headless)`. That's
   fine on Turing/Ampere (RTX 20/30-series, T4, A100). On RTX 40-series it may work via PTX JIT;
   on RTX 50-series (Blackwell) expect torch to have no kernels — that would mean a newer torch,
   which is its own migration.
3. **The sim needs a real GPU + NVIDIA driver + EGL.** Confirmed on the old machine: with no driver,
   habitat-sim fails at `WindowlessContext: Unable to create windowless context — no EGL devices found`.
   No CPU fallback exists in this build.
4. **RxR needs internet once**: `build_custom_episode_rxr.py` downloads the `xlm-roberta-base`
   tokenizer from HuggingFace on first use (R2R's BERT vocab is fully local).
5. **Old machine's IP may change** (DHCP) — recheck with `hostname -I`.

## 6. Verify the migration

```bash
nvidia-smi                                            # driver + GPU visible
conda activate vlnce
python -c "import torch, habitat, habitat_sim; print(torch.cuda.is_available(), habitat_sim.__version__)"
cd ~/ETPNav
python scripts/run_instruction.py --instruction "move to the left of the table" --exp-name migration_check
```
Takes ~15 s and prints `video(s) written to data/logs/video/migration_check`. The run is
deterministic (greedy decoding), so the stats should match (or be very close to) what the old
machine recorded in `docs/session-findings/results/custom_demo/stats_ckpt_59_custom.json`:
`steps_taken 79`, `path_length 12.72`, `distance_to_goal 4.70`. Small float drift across
different GPUs is possible; a totally different trajectory means something is mis-copied.

## 7. As executed: migration to `winterfell` (2026-09-20)

Target: `ned@192.168.83.125`, Ubuntu 24.04 x86_64, NVIDIA T1000 8GB (driver 595.84 — same GPU as the
old machine), external ext4 drive mounted at `/data` (933 GB), everything placed under `/data/manoj`.
Transfer used SSH key auth from the old machine (no passwords stored anywhere) and took 218 s
for 39 GB (~220-350 MB/s over the LAN). All trees verified by rsync dry-run diff; both release
checkpoints verified by sha256.

Layout on the target:
```
/data/manoj/
  ETPNav/                     # full repo incl. .git; data/ with datasets, pretrained, ddppo-models, wp_pred,
                              #   logs/checkpoints/release_{r2r,rxr}/, scene_datasets/mp3d/ (real dir, 90 houses)
  habitat-lab-v0.1.7/         # editable-install source, includes the 1-line requirements.txt edit
  envs/vlnce/                 # unpacked conda-pack env;  envs/vlnce_env.tar = the tarball (safe to delete)
  cache/clip/ViT-B-32.pt      # ~/.cache/clip on the target is a symlink to this
  2210.05714v4.pdf            # the ETPNav paper
```

Use it:
```bash
source /data/manoj/envs/vlnce/bin/activate      # conda-pack's activate; `conda activate /data/manoj/envs/vlnce` also works
cd /data/manoj/ETPNav
python scripts/run_instruction.py --instruction "move to the left of the table"
```

What was needed beyond copying (in case you redo this elsewhere):
1. `conda-pack` was run with `--ignore-editable-packages --ignore-missing-files` (habitat-lab is an
   editable install, and some conda-managed files were overwritten by pip; conda-pack refuses both by default).
2. After extracting the tarball: `source envs/vlnce/bin/activate && conda-unpack` (rewrites hardcoded prefixes).
3. The editable-install pointers survive packing and still name the old path — repointed with
   `sed -i 's#/home/storms-end/habitat-lab-v0.1.7#/data/manoj/habitat-lab-v0.1.7#g'` on
   `lib/python3.6/site-packages/easy-install.pth` and `habitat.egg-link`.
4. `ETPNav/data/scene_datasets/mp3d` is a real directory on the target (copied from the old machine's
   `~/vlmaps_cc/mp3d_habitat/mp3d`), not a symlink.
5. `clip.load("ViT-B/32")` reads `~/.cache/clip`; that was symlinked to `/data/manoj/cache/clip`.

Verification result on the target (deterministic run from section 6): `steps_taken 79`,
`distance_to_goal 4.700018`, `path_length 12.723581`, mp4 335,138 bytes — identical to the old machine.
