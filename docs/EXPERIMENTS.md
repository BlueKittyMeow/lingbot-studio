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
- [x] ~~Fresh-VM sw48/nsf4 attempt~~ — **DONE: sw48/nsf4 fits, sw64 & nsf8 don't (confirmed on a clean GPU).
  sw48/nsf4 is the ceiling at 518 on BOTH backends.**
- [x] ~~FlashInfer unlock~~ — **DONE (built + working), but it does NOT raise the VRAM ceiling.** sw64/nsf4 and
  sw56/nsf4 OOM on paged KV too (the pool is pre-sized to the window); nsf8 OOMs from an upfront scale-phase
  activation spike no backend touches. **FlashInfer's real value = SPEED** (~12%: 7.77 vs 6.95 it/s at sw48/nsf4;
  scales better on long sequences → matters for future mega-maps/stitching), and it's the prerequisite for the
  pos_embed patch below. Toolchain build recipe banked in the brief (cuda-nvcc 12.8 + gcc-14 + conda-forge
  sysroot 2.28 to dodge Ubuntu-26.04 glibc 2.41 + WSL libcuda linking; all MCE-safe via single-thread throttle).
- [x] ~~pos_embed interpolation patch~~ — **DONE & WINS. The FULL model defaults `sw64/nsf8` now fit at
  `--image_size 448`** (rendered 200/200 frames; OOM'd every way at 518). Resolution IS the juggle.
  - **Patch:** `demo_render/demo.py` `load_model` — `_adapt_pos_embed()` bicubic-interpolates the checkpoint's
    518 pos_embed (1370 tok, 37×37) down to the target grid before `load_state_dict` (backup: `demo.py.bak_lowres`).
    Works because the model is resolution-agnostic everywhere except that one DINOv2 param (RoPE + DPT head +
    FlashInfer KV are all runtime-sized; the ViT even keeps `interpolate_pos_encoding` live every forward).
  - **Cleaner alt (agent-recommended, not yet applied):** add `--model_img_size 518` (build at native 518,
    load clean, NO checkpoint edit) + drive input via `--image_size 448` → the forward interpolates ONCE
    (vs my load-time patch's negligible double-resample). Same VRAM (KV sized by *actual* tokens, not img_size).
  - **Legal `--image_size`: multiples of 14 only — 448 (32×14), 392 (28×14), 378 (27×14). NOT 384.**
  - **Quality:** 518→448 ≈13% linear downscale, DINOv2/VGGT "graceful regime"; coarse drift metrics unchanged
    so far → **needs eyes-on** (fuse + walk vs the shipped 518 scene) before adopting as default.
  - **Recipe (full defaults @448):** FlashInfer env (no `--use_sdpa`) + `--kv_cache_sliding_window 64
    --num_scale_frames 8 --image_size 448`, throttled. Fits ~clean on 16GB.
- [ ] **FP8 KV cache (2nd path to sw64, keeps 518)** — fork `ureeey/lingbot-map-rtx4060-8g@rtx4060_8g` adds
  `--kv_cache_fp8` (FlashInfer-only; we have it). Halves the KV pool → `sw64@518` should fit. BUT the fork's own
  benches flag FP8 KV as **"significant" pose/trajectory degradation** — real cost for geometry. Weight-quant
  `--quant_wa mix` is "minor" but doesn't free the KV pool. Trade: 448-lower-detail (our patch) vs 518-noisier-pose.
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
