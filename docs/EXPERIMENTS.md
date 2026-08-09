# Experiments — running list

A living backlog of what we want to try. Check things off, add freely. Findings from completed
experiments migrate to `FINDING_AID.md` + `lingbot_map_brief.md`.

**Baseline settings (as of 2026-08-08):** reconstruct at `--fps 10`, `--use_sdpa`,
`--kv_cache_sliding_window 48`, `--num_scale_frames 4`. **`sw48/nsf4` is our proven 16GB ceiling**
(renders 200 clean frames; only a benign teardown segfault). Fuse with leveling. Deploy to
walk.bluekittymeow.com.

**VRAM ceiling — the hard boundary (mapped 2026-08-08, throttled runs, SDPA, forced 518):**
- ✅ `sw48/nsf4` fits. ❌ `sw64/nsf4` OOMs. ❌ `sw48/nsf8` & `sw64/nsf8` OOM → **`nsf8` and window>48 both overflow.**
- **`--image_size` is LOCKED to 518** — the checkpoint's `pos_embed` is hard-baked to the 37×37/1370-token
  grid; `--image_size 448` fails weight-loading (`size mismatch [1,1370,1024] vs [1,1025,1024]`). The quadratic
  memory lever is unavailable without a pos_embed-interpolation code patch. (The "size sacrifice" in good early
  runs was cropping/downscaling the *source footage*, NOT the model's internal res — it always resamples to 518.)
- **`--offload_to_cpu` and `--camera_num_iterations` do NOT exist in `batch_demo.py`** (demo.py-only) — the
  paper's biggest camera-KV lever (4× cut) is useless for MP4 flythroughs.
- CPU throttle (CPUQuota 400% + OMP=3 + nice) fully defeats the *MCE crash* but not the *VRAM* ceiling —
  it's parallelism-in-memory (whole KV cache held at once), not sequential compute you can slow down.
- **Only path past sw48/nsf4: FlashInfer** (paged KV cache, more memory-efficient than SDPA + quantizable) —
  blocked by no `nvcc` in WSL. Install `cuda-toolkit-12-8` to unlock it and push the window higher. See infra list.

---

## 🎨 Creative / skinning backlog

- [ ] **★ Mirrored + reversed Kowloon (ceiling↔floor, palindrome)** — the one we keep not getting to!
  Two moves that combine: (a) **mirror the frame vertically** (ceiling→floor) to synthesize a symmetric
  floor/ceiling and fake a *box/tube* instead of Kowloon's thin straight-ahead depth; (b) **forward + reverse
  (palindrome) stitch** the clip so the mirror seam is hidden AND the camera covers the corridor in both
  directions (denser reconstruction). Fake geometry, deliberately so — a structured surreal tunnel.
- [ ] **Skin combos** (geometry + foreign colour, `skin_fuse.py`):
  - [ ] Courthouse → Doom (`MnqLJpgq7jc`, crop top notif strip + bottom HUD)
  - [ ] Courthouse → "pretty" cathedral (`k5CMR5b1cIU` @1:02)
  - [ ] Catacombs → waterlilies / Doom / pretty
  - [ ] **Cathedral → Doom** (`MnqLJpgq7jc`) — gothic bones in hellfire; the sacred-profane clash
  - [ ] **Cathedral → waterlilies/flowers** (`heR36dG8qh0`) — the nave dissolving into a Monet pond
  - [ ] **Cathedral bones + Kowloon skin** → "walk the Walled City's ghost on cathedral bones" (also sidesteps
    Kowloon's own hard-footage limits by borrowing good geometry). "A pretty of each."
- [ ] **Spoof the lower half** — fill/fake the cropped-out lower frame region; or the deferred-texturing route
  (good geometry from the clean crop, then texture with the FULL original frame incl. overlays) for a glitchy
  "floating timecode smeared on walls" look.

## 🔜 Up next (queued)

- [ ] **gsplat-skin** — train a Gaussian *splat* of the waterlily courthouse (splatted skin, not just points).
- [ ] **Real mesh export** — Poisson / TSDF the courthouse (or catacombs) into a solid textured `.glb` you fly
  around as an *object*, skin baked into the texture. (We have depth+poses+normals → straight into Open3D.)
- [ ] **Full cathedral run** — the whole Cologne nave walk (Lara bets it'll shine; test at 20s looked great).
- [ ] **Full catacombs run** — now that we know the settings.

## 🔧 Pipeline / infrastructure

- [ ] **Two-window stitching** — implement the Umeyama-Sim(3)-on-shared-camera-centres recipe (banked in the
  brief) to reconstruct corridors longer than the ~320-frame VM ceiling at full fps-10. Unlocks the *whole*
  Kowloon tape / 5k-frame catacomb mega-scenes.
- [ ] **Potree viewer** — octree-LOD web renderer for buttery mega-scenes on the Iris Xe (no moving/still swap
  seam). Do it before the mega-scenes. (Recipe in the brief.)
- [ ] **World-Mirror closed splat** — hook `rasterization.py` for per-view gaussians (pre-fusion), apply the
  per-view Sim(3) → a *closed feed-forward* splat. Marginal ROI vs the trained gsplat; optional.
- [ ] **gsplat pose-optimization** — enable pose+intrinsics optimization / bundle adjustment during splat
  training (per lingbot issue #35) to reduce splat overfit-to-trajectory.
- [x] ~~Fresh-VM sw48/nsf4 attempt~~ — **DONE: sw48/nsf4 fits, sw64 & nsf8 don't. sw48/nsf4 is the ceiling.**
- [ ] **FlashInfer unlock** — install `cuda-toolkit-12-8` in WSL → paged KV cache → push window past 48 / afford
  nsf8. The single infra job that raises the VRAM ceiling.
- [ ] **pos_embed interpolation patch** — patch `load_model` to interpolate the 518 `pos_embed` to a smaller
  grid, unlocking `--image_size 448/384` (the quadratic lever) for lower-res-but-higher-window runs. Optional.
- [ ] **Re-render catacombs2 hero at sw48/nsf4** and eyeball vs the shipped sw32/nsf4 (`catacombs2q`) — confirm
  the wider window visibly helps before adopting sw48 as the default (screenshots are source of truth).

## 🗺️ Big / ambitious

- [ ] **Massive maps** — the whole Kowloon tape, a 5000-frame catacomb (needs stitching + Potree).
- [ ] **Progression gallery site** — a `walk`-style page cataloguing the iteration story (point-maps + POV
  captures per stage). Pairs with `FINDING_AID.md`.

## ✅ Done (see FINDING_AID.md for detail)

Kowloon whole-walk (cropped, leveled) · catacombs / catacombs2 / catacombs20 (fps sweet-spot + dense pass) ·
courthouse density A/B (sparse wins) · sky-masked trained gsplat (the beauty) · World-Mirror install +
pose-conditioning + Kobayashi-Maru closure proof · first skin (waterlily courthouse) · cathedral 20s test ·
the studio repo + finding aid + gallery.
