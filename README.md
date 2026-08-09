# lingbot-studio

A pipeline for turning ordinary video into **walkable 3D scenes** and **Gaussian splats** — reconstruct,
fuse, level, skin, and deploy to a web gallery. Built on top of the `lingbot-map` streaming reconstructor
(a DUSt3R/VGGT-lineage feed-forward model) plus a hand-rolled fuse/train/skin toolchain and, experimentally,
Tencent's HunyuanWorld-Mirror.

Live gallery: **https://walk.bluekittymeow.com** · Viewer source: the sibling `lingbot-viewer` repo.

---

## Where things run (and the sync model)

- **This repo (MysteryOfGlass, `~/Documents/Git/lingbot-studio`) is the source of truth** — version control,
  docs, backup, GitHub origin.
- **The scripts actually execute inside WSL on MarshLair** (the RTX 4070 Ti box) where the models + CUDA live.
- **Sync is handled by the agent, not by hand.** Today: `tar` the WSL scripts → pull into this repo → commit
  (WSL → repo); `base64`-into-a-heredoc to push a repo-edited script back to WSL (repo → WSL). Once a GitHub
  remote is wired, WSL becomes a plain `git clone` and sync is just `git pull` / `git push` — at which point a
  WSL VM crash loses nothing, because everything lives on GitHub.
- GitHub remote is **deferred until ready**; auth will be headless (SSH deploy key or fine-grained PAT — the
  human pastes one key/token, no browser needed on the machine).

## Pipeline (the arc)

1. **Reconstruct** — `lingbot-map`'s `demo_render/batch_demo.py --video_path X --fps N --save_predictions`
   → per-frame NPZs (depth, W2C extrinsic OpenCV, intrinsic, image, confidence). Run inside WSL via
   `scripts/reconstruct/*.sh` (systemd-run units survive SSH drops). Config: `config/wsl16gb.yaml`.
2. **Fuse** — back-project depth → world points, high-confidence subsample per frame, concat, write tiered
   PLYs (1.5M/3M/6M/max). `scripts/fuse/fuse_generic.py` (+ scene-specific variants).
3. **Level** — bake an average-camera-up rotation so scenes stand vertical: `scripts/fuse/*l_fuse.py`
   (e.g. `catacombs20l_fuse.py`, `kwlnv2_fuse.py`).
4. **Skin** *(surreal)* — keep the geometry, swap the colour source to a different video's frames:
   `scripts/fuse/skin_fuse.py`.
5. **Splat** — train a Gaussian splat on a fused/skinned cloud: `scripts/splat/train_splat.py`; sky-masked
   variant `train_splat_sky.py` + `gen_sky_masks.py` (kills sky blowout via skyseg.onnx).
6. **World-Mirror** *(experimental)* — feed-forward splats + pose-conditioning research:
   `scripts/worldmirror/build_infer_cond.py`, `wm_close_splat.py` (the "Kobayashi Maru" reprojection that
   closes WM's folded geometry using lingbot poses), `wm_dump_poses.py`, `wm_eval_cond.py`.
7. **Deploy** — relay tiers to Factotum `/var/www/walk/`, add a scene to `lingbot-viewer/scenes.json`.

## Key learnings

The full knowledge base — hard-won recipes, the fps sweet-spot, density-vs-drift-scatter, the VM crash
envelope, sky-masking, World-Mirror's loop-closure limitation and the pose-conditioning findings, and the
two-window stitching recipe — lives in **`docs/lingbot_map_brief.md`**. Read that first.

## Layout

```
scripts/     all pipeline scripts (fuse / splat / worldmirror / reconstruct / install)
docs/        lingbot_map_brief.md  — the knowledge base
config/      wsl16gb.yaml           — the 16GB-VRAM reconstruct config
```

*(scripts/ is currently flat — a snapshot of the WSL working dir. Categorization into subdirs is a nice-to-have.)*
