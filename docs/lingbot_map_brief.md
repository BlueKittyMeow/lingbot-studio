# LingBot-Map on MarshLair — side-session config brief

**For:** a fresh Claude session tasked with getting lingbot-map running.
**Owner:** Lara. **Written:** 2026-08-05 (desk-3D session).

---
## ★ SESSION STATE — 2026-08-08 (READ THIS FIRST; it's a whole studio now) ★

**What exists:** `https://walk.bluekittymeow.com` — a LIVE gallery of walkable 3D reconstructions,
hosted on Factotum (Caddy `:8093` → `/var/www/walk/`, cloudflared). Gallery = `index.html` (reads
`scenes.json`); point-cloud viewer = `view.html?scene=NAME` (Three.js first-person fly, density
tiers via dropdown + number keys, `rotation.x=π` for Y-up); splat viewer = `splat.html?scene=NAME`
(GaussianSplats3D, `sharedMemoryForWorkers:false`). Source repo: `~/Documents/Git/lingbot-viewer/`
on MysteryOfGlass (index/view/splat.html + scenes.json). Add a scene = stream `NAME_*_fused.ply`
tiers to `/var/www/walk/` + add an entry to `scenes.json` + scp it.

**Scenes live:** `kowloon` (walkable corridor — hard-won from 1991 fisheye VHS), `courthouse`
(lingbot sample + has a trained splat), and `catacombs` (clean 4K, DEPLOYING as of compaction —
see in-flight). `scenes.json` in the repo already has the catacombs entry prepped (2nd, after kowloon).

**THE PIPELINE (per scene):** (1) reconstruct video → per-frame NPZ (depth+pose+image) via
`batch_demo.py` in the `lingbot-map` conda env (WSL on MarshLair). (2) fuse NPZs → colored PLY tiers
(pure numpy). (3) stream tiers WSL→Factotum, add to gallery.

**HARD-WON RECIPES & GOTCHAS (do not relearn these):**
- **Reconstruct command (the fix):** `python demo_render/batch_demo.py --video_path X --fps 4
  --first_k 240 --use_sdpa --kv_cache_sliding_window 16 --num_scale_frames 1 --output_folder
  ~/renders/NAME --model_path ~/models/lingbot-map/lingbot-map.pt --config
  demo_render/config/wsl16gb.yaml --save_predictions`. Launch via `systemd-run --unit=NAME
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ...`. Pre-downscale big inputs to ~640px
  wide (`ffmpeg -vf scale=640:-2`).
- **OOM ROOT CAUSE (diagnosed by 2 Opus agents):** the `CUDA driver error: out of memory` that fires
  INSTANTLY at inference start = FlashInfer paged-KV pool, sized by **patches-per-frame
  (resolution × aspect squareness), NOT frame count** (sliding window caps slots). Big/square frames
  OOM; cutting frames does nothing. FIX = `--use_sdpa` (FlashInfer off → torch allocator) +
  `--kv_cache_sliding_window 16`. Alt near-lossless: `--kv_cache_sliding_window 32
  --kv_cache_scale_frames 2 --max_frame_num 300` (keeps FlashInfer). Corroborated by lingbot issue #32.
  **Corollary: frame count is cheap → huge multi-thousand-frame maps ARE feasible now.**
- **TRACKING BREAKTHROUGH (Kowloon):** static burned-in graphics (subtitles, timecode, channel logos)
  AND pillarbox/vignette bars are screen-FIXED → the tracker thinks the camera is stationary →
  reconstruction collapses to a flat plane (net cam displacement ~0.1, thin depth slab). **CROP ALL
  static frame junk before reconstructing.** Cropping the bottom overlays fixed Kowloon's tracking
  (net 0.1→1.6); cropping the sidebars ALSO de-flattened the depth (0.79-0.91 → 0.45-1.52). For new
  footage: shave any logo/subtitle/timecode + pillarbox/letterbox bars first. Lara diagnoses these
  by eye — trust her calls.
- **FPS MATTERS — use `--fps 10` (repo default), NOT 3-4 (Lara's catch, 2026-08-08):** every README
  example uses `--fps 10`. The model is a STREAMING model trained on smooth 320-frame video-RoPE
  windows, so sparse frames (our earlier fps 3-4) = big inter-frame jumps = (a) worse tracking → more
  drift/spiral/tilt, and (b) fewer overlapping views per surface → SPARSER, lower-res-looking cloud.
  fps 10 = denser + cleaner + straighter maps. **CONFIRMED 2026-08-08:** the catacombs 7:00-9:00
  ossuary section reconstructed at fps 10 (300 frames) dropped end up-drift from **~26° (fps 4) to 6.1°**
  — a 4× reduction in spiral, and visibly denser. So fps WAS a major cause of the "low-res + tilted"
  look. This dense pass is deployed live as scene **catacombs2** ("Catacombs — Ossuary Wall") at
  walk.bluekittymeow.com (tiers 1.5M/3M/6M/13.5M, default 3M). Note: fps 10 over 30s of slow close-up
  panning covers little DISTANCE (net walk 0.83 units vs the fps-4 corridor's 4.25) — dense but short.
  Trade-off vs our ~300-frame VM budget: fps 10 covers
  ~30s per run (300 frames, conveniently < the 320-view training limit → no `--keyframe_interval`
  needed). For LONGER walks: `--keyframe_interval N` (repo's tool for >320 frames — non-keyframes still
  get predictions, aren't cached) — but inference TIME still scales with frames, so the VM-crash ceiling
  caps a single run at ~300-350 frames regardless. To cover a long corridor densely: multiple fps-10
  ~30s windows, stitch/fuse together. Also `--image_stride 1` = every frame.
- **★ DENSITY vs DRIFT-SCATTER — SPARSE wins on long paths (courthouse A/B, 2026-08-08):** more frames
  buy COVERAGE, but each frame's pose carries drift; many overlapping frames re-estimating the SAME
  surface with inconsistent poses SCATTER points into a thick blurry shell instead of one crisp surface.
  Courthouse orbit reconstructed dense (stride 1, 286 frames, 12.9M pts) vs sparse (stride 2, 143, 6.4M):
  at equal 6M view budget the DENSE was visibly MUSHIER (smeared columns/edges), SPARSE crisp — Lara
  called sparse "completely superior." Reconciles with the catacombs fps20 win: that was dense over a
  TINY 15s span (poses barely drift → density helped, richer/enclosed); over a long orbit drift dominates
  → density HURTS. HOUSE RULE (corrected 2026-08-08 after Lara flagged the overstatement): there is ONE
  sweet spot in the MIDDLE, missable from either side — TOO SPARSE (fps 3-4) = jumps too big, tracking
  breaks; TOO DENSE (fps 20 / courthouse's 286-frame orbit) = drift piles up, surfaces smear; SWEET SPOT
  ≈ **fps 10** for our video. "Sparse won" on the courthouse meant only "don't OVER-sample past the sweet
  spot" (286→143 moved back toward the middle) — NOT "go minimal." So: video → target fps 10; image seqs
  → don't use every frame if the orbit is dense (stride toward ~fps-10-equivalent). Bonus: staying near
  the sweet spot (not maxing frames) also keeps us under the VM crash ceiling.
- **★ STITCHING TWO WINDOWS (recipe banked 2026-08-08, agent-researched — this UNBLOCKS long full-fps walks):**
  to cover a corridor longer than the ~320-frame VM ceiling, reconstruct TWO windows with a DELIBERATE ~30-50
  frame OVERLAP, then align: (1) camera centre of each shared frame `C = -R.T @ t` (W2C extrinsic) for both
  windows; (2) **Umeyama Sim(3)** (scale+rot+trans, closed-form SVD w/ det-reflection guard) mapping window-B
  centres → window-A centres — copy `evo.core.geometry.umeyama_alignment` (~30 lines) or VGGT-Long's
  `loop_utils/sim3utils.py` (`weighted_estimate_sim3` + `robust_weighted_estimate_sim3` = Huber-IRLS, tuning-light,
  preferred over discrete RANSAC); (3) apply `(s,R,t)` to ALL of B's points+poses (and depths ×s if TSDF-ing);
  (4) MERGE the overlap without doubling: simplest = hard-cut at the overlap midpoint frame; best = Open3D
  `ScalableTSDFVolume.integrate` over all frames (seam vanishes, free mesh). CORRIDOR GOTCHA: dead-straight
  camera centres are COLLINEAR → scale + roll ill-conditioned; put the overlap across a bend, or add a few
  high-confidence OFF-AXIS 3D points (walls/floor) to the correspondence set. Sanity-check the fit: inlier frac
  >80%, scale s plausible (~0.5-2), low centre residual. Open3D FPFH+RANSAC(`with_scaling=True`)+colored-ICP is
  the no-shared-frames fallback (ICP is rigid-only → Sim3 first for scale, then ICP refine). ~few hundred lines total.
- **VM crash envelope:** the WSL VM kernel-panics (MCE) on long/heavy load. Safe reconstruction zone
  ≈ 240-340 frames. Launch when GPU is free (`nvidia-smi` — the OTHER Claude's ComfyUI on E: bursts
  VRAM and can collide). `.wslconfig` cap must stay (see below). Restart `WSLKeepalive` schtask after
  any crash. Heavy nvcc compiles crash it too (`TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=1 nice`).
- **Fuse (crash-safe):** per-frame keep top-conf ~45k pts → concat (~9-15M) → center on median →
  write max PLY → **random-subsample (np.random.permutation) for 6M/3M/1.5M tiers**. DO NOT voxel
  (np.unique) on >10M pts in the same process — it silently dies; random subsample is fine. Scripts:
  `~/kowloon_fuse_tight.py`, `~/catacombs_fuse.py` (sed one from the other, change SRC + output names).
- **Watcher scripts:** bash exit code = last command's; a flaky final `ssh` to MarshLair returns 255
  → task shows "failed" even on success. FIX (Opus): compute a `verdict` var in the loop, retry the
  tail fetch 3×, `exit 0` when the watched condition is detected. Use this pattern for all watchers.
- **SKINNING / deferred texturing (the creative core):** geometry and color are DIVORCED. Reconstruct
  scene A → in the fuse, color each point from a DIFFERENT video B (sampled to A's frame count, resized
  to the depth grid) instead of A's own image. Walk A's architecture wearing B's skin. Not yet
  implemented — it's a small change to the fuse (point the color lookup at B's frames).
- **SPLAT:** overfits to the capture trajectory → looks great from training-ish views, white-blowout
  floaters from novel/orbit angles (narrow forward capture is the cause). Best recipe (v4): dense init
  from our depth points, CLEAN L1 loss, tight scales `log(0.002*extent)`, ~8000 iters, NO densification/
  SSIM (they added speckles). Post-prune huge/faint/far Gaussians. Env: `gsplat` (torch 2.4.0+cu124 +
  PREBUILT gsplat wheel — torch 2.8 can't build gsplat). Scripts: `~/train_splat.py`,
  `~/render_splat_test.py`. ALWAYS verify a splat from an OFF-AXIS camera, not training views.
- **SPLAT UPGRADE PATH (from 2026-08-08 community research — the fix for our floater/overfit problem):**
  lingbot's own issue **#35** (someone hit our exact splat-collapse) — maintainer says: our poses AND
  per-frame intrinsics are only APPROXIMATE, so **do NOT freeze them — enable pose+intrinsics
  optimization in the trainer**, run **bundle adjustment** to refine intrinsics before training, and
  **sky-mask**. That (frozen slightly-wrong poses) is the real cause of the trajectory-overfit floaters,
  not just narrow capture. lingbot has NO native splat yet but a "GS head" is planned (#5); PR #82 adds
  a confidence-filtered PLY export (good splat init). **Best trainer:** gsplat `simple_trainer` with
  `pose_opt=True` + our dense point cloud as init + a depth loss (reuses our `gsplat` env). Alt:
  nerfstudio `ns-train splatfacto`. **BRUSH** (github.com/ArthurBrussee/brush) is WebGPU/no-CUDA →
  could train splats on MysteryOfGlass's Intel GPU, off the MarshLair queue entirely. Forward-walk
  tips: sky-mask, scale-reg, depth-reg, DropGaussian-style dropout, modest SH degree (1–2), aggressive
  floater pruning. Feed-forward-splat to watch: **HunyuanWorld-Mirror** (takes poses+depth priors →
  emits Gaussians in one pass — we already have the priors).

**IN-FLIGHT / QUEUED at compaction:**
- **Catacombs deploy:** fuse running (`catacombs-fuse` unit), watcher auto-streams tiers to Factotum.
  After tiers land, DEPLOY `scenes.json` (`scp ~/Documents/Git/lingbot-viewer/scenes.json
  bluekitty@192.168.1.201:/var/www/walk/`) + verify `view.html?scene=catacombs` loads. THEN it's live.
- **Splat-community research agent** running (Opus) — what people do with splatting on lingbot-map.
- **Kowloon v2 (queued):** re-crop bottom deeper (~220px kept, was 250) to kill leftover static-overlay
  "checkerboard" patches on the floor Lara spotted → re-reconstruct → fuse → deploy (overwrite).
- **Cathedral:** Cologne 4K [mzOwOSm2ubE] — grab clean segment → reconstruct RAW → deploy → then skin.
- **PLAN: raw ("plain vanilla") FIRST, then skin.** Skin library (all YouTube, transformative use ok'd):
  🌸 pretty `k5CMR5b1cIU` @1:02 · 💀 Doom `MnqLJpgq7jc` (crop top notif strip + bottom HUD) · 🪷
  waterlilies `heR36dG8qh0`. First skin target: catacombs, once raw is standing.
- Footage lives in WSL `~/footage/` (catacombs*.mp4) and `~/kowloon/`; yt-dlp is in the lingbot env.

**★ TILT FIX (queued — scenes render canted):** the reconstruction anchors its world frame to camera
frame 0's orientation; handheld cameras aren't level → the whole cloud is tilted, and the viewer's
fixed `rotation.x=π` only handles the 180° flip, not arbitrary tilt/roll. FIX (bake into the fuse, no
re-reconstruction): estimate true up = average of every frame's camera up-vector (world up ≈ mean of
`-(R[1,:])` over frames, i.e. negative 2nd row of each world-to-cam R, since OpenCV cam +Y is down),
build a rotation mapping that up → +Y, apply to points before centering. Then either drop the viewer's
`rotation.x=π` for leveled scenes or fold it into the baked rotation. Re-fuse + redeploy catacombs (and
kowloon has a subtler tilt too). Optionally also align walking direction to +Z and RANSAC the floor plane.

**★ LARA'S CREATIVE BACKLOG (her ideas — do these, they're great):**
- **Mirror ceiling→floor to fake a box/tube:** Kowloon's depth is thin (we mostly see straight ahead).
  Mirror the frame vertically to synthesize a symmetric floor/ceiling → a structured surreal tunnel
  instead of flat cards. Fake geometry, but deliberately so.
- **Forward + reverse (palindrome) stitch:** play the clip forward then reversed, stitched, to (a) hide
  the obvious mirror seam and (b) DOUBLE the camera's coverage of the same corridor (walk it both
  directions → more points, denser reconstruction). Her idea to make the mirror less obvious.
- **Deferred SKINNING combos (once raw scenes stand):** skin catacombs/cathedral with pretty 🌸 / Doom 💀
  / waterlilies 🪷. Inverse too: reconstruct a CLEAN scene (catacombs/cathedral) then skin it with
  KOWLOON footage → walk the Walled City's ghost on cathedral bones (this also SIDESTEPS Kowloon's own
  hard-footage limits by borrowing good geometry). "A pretty of each" scene.
- **Spoof the lower half + map lousy footage back:** she floated filling/faking the cropped-out lower
  frame region. Note: fake pixels don't add real geometry — but the deferred-texturing route (good
  geometry from the clean crop, then texture with the FULL original frame incl. overlays) gives the
  glitchy "floating timecode smeared on walls" look she's after.
- **Massive maps:** now that we understand the KV memory (frame count is cheap post-SDPA), reconstruct
  a LONG walk — the whole Kowloon tape, or a 5k-frame catacombs — for an explorable mega-scene.
- **★ ROADMAP: Potree viewer (TABLED 2026-08-08, Lara wants it — do it before mega-scenes).** The current
  viewer draws the WHOLE cloud every frame; we made 13M navigable with a motion-adaptive LOD hack (fly on
  the light tier + drop resolution while moving, snap to full detail when still — `proxyMat`/`detailMat`,
  `applyMotion()`, `stillDPR()` in view.html) but the moving→still SWAP has a visible seam/lag on the
  Iris Xe. Potree = octree-LOD web renderer that streams only the visible points at the right detail
  continuously (no swap, no seam), handles 100M+ pts on integrated GPUs — the real fix and the enabler for
  mega-scenes. Build = (1) PotreeConverter on each max PLY → octree tiles, (2) host tiles on Factotum
  `/var/www/walk/`, (3) swap in the Potree renderer (keep current view.html as fallback), + a converter
  step after each fuse. A few hours. Alt ceiling if even Potree isn't enough: server-side pixel-streaming
  from MarshLair (cloud-gaming style) — but Potree keeps the simple web arch and no latency, so prefer it.
- **★ HunyuanWorld-Mirror (INVESTIGATING 2026-08-08):** Tencent feed-forward reconstructor with a NATIVE
  Gaussian-splat head (lingbot-map has none — maintainer confirmed in map issue #84). ~1B params, single
  16GB GPU (cast F32 weights→bf16), takes video/multiview + optional pose/intrinsic PRIORS (could feed our
  lingbot extrinsics), outputs point cloud + depth + normals + GAUSSIANS in one pass; integrates gsplat.
  The real upgrade for the SPLAT track. Repo github.com/Tencent-Hunyuan/HunyuanWorld-Mirror, weights
  huggingface.co/tencent/HunyuanWorld-Mirror. CRITICAL open question: does install need nvcc compilation?
  (WSL has driver only, NO cuda toolkit — heavy nvcc compiles MCE-crash this VM). Also Hunyuan3D-Paint
  (already in comfy3d) = object-scale skinning (geometry + reference-image appearance).
- Her philosophy this session: "no true failures, just exploration" — the whole skinning studio only
  became thinkable BECAUSE Kowloon failed (tracking collapse → brightening → color-swap → geometry/
  texture are separable → skin anything on anything). Failures were the path.

**★★ SPLAT PIPELINE + WORLD-MIRROR (built 2026-08-08 night — a whole studio) ★★**
Three ways to make a Gaussian splat, all deployed & comparable on the courthouse:
- **Trained gsplat (BEST quality)** — `~/train_splat.py`. Runs in the **`gsplat` conda env** (torch
  2.4.0+cu124 + PREBUILT gsplat 1.5.3 wheel → NO nvcc → VM-safe; only *compiling* gsplat crashes the VM,
  training on prebuilt kernels is fine). Recipe: unproject lingbot depth → init cloud, L1 loss, tight
  scales log(0.002*extent), NO densification. 6000 iters ≈ 66s, 1.5M gaussians. Edit SRC/OUT to point at
  any lingbot NPZ folder. Deployed `courthouse-fresh` (trained on the crisp SPARSE courthouse — beats the
  old Aug-7 `courthouse` splat).
- **Sky-masked trained gsplat** — `~/train_splat_sky.py` + `~/gen_sky_masks.py`. gen_sky_masks.py (WM env)
  runs `segment_sky` (skyseg.onnx, 255=non-sky) on each NPZ frame → masks at `~/renders/courthouse_sky143/`;
  train_splat_sky.py excludes sky from BOTH the init AND the L1 loss → sharp building, clean empty sky.
  Deployed `courthouse-sky`. (Nit: roofline floater specks the segmenter kept — tighten init conf or
  opacity-cull later.)
- **World-Mirror (feed-forward, seconds, no training)** — Tencent HunyuanWorld-Mirror at
  `~/HunyuanWorld-Mirror`, env `hunyuanworld-mirror` (cloned from gsplat env + onnxruntime + huggingface_hub
  1.x). Weights `ckpts/` (5GB; **`hf download tencent/HunyuanWorld-Mirror`** — CLI is `hf`, NOT
  huggingface-cli in hub 1.x). Run: `python infer.py --input_path <dir|video> --output_path <out>
  --target_size 518 [--apply_sky_mask]`. Outputs gaussians.ply (native splat!) + pts_from_pointmap.ply +
  depth + normals + rendered.mp4 + COLMAP. NO nvcc (gsplat prebuilt is the only CUDA dep). **Memory scales
  with # frames**: 16 safe; 24 OOMs on the rendered-video step (splat still saves first). It's a feed-forward
  GUESS → pose-uncertainty "spray" + sky blowout (--apply_sky_mask cleans the POINT CLOUD, not the
  gaussians). Value = SPEED + PRIOR-CONDITIONING, not final polish. Deployed `courthouse-wm`.

**Splat viewer:** `splat.html?scene=X` → loads `X_splat_web.ply` (GaussianSplats3D CDN, SH0/DC-color).
Splat PLY = standard 3DGS (x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3, ~68 B/gaussian). Add a
splat: drop `NAME_splat_web.ply` in /var/www/walk/, browse `splat.html?scene=NAME`.

**WM POSE-CONDITIONING (IN PROGRESS — priors ARE consumed, convention not nailed):** infer.py's
`cond_flags` is hardcoded `[0,0,0]` and `--cond_pose/--cond_intrinsics` are declared but NOT wired. To feed
priors, inject into `views` (see `~/HunyuanWorld-Mirror/infer_cond.py` + builder `~/build_infer_cond.py`):
`views['camera_poses']` [B,S,4,4], optional `views['camera_intrs']` [B,S,3,3], `cond_flags=[depth,rays,camera]`.
Feed images+poses+intrinsics all from the SAME lingbot NPZs (native 294×518) so they're consistent.
FINDINGS: priors are definitely used (banner positions shift every run). Tried C2W (banners duplicated on
multiple sides = rotational error), W2C+intrinsics (banners front but asymmetric L/R distance = suspected
intrinsics/projection warp), W2C pose-only (latest, scene `courthouse-wmc`). Per the model's own decoder
(`transform_camera_vector` builds w2c then inverts for output) **W2C is the right pose convention**. **SMART
NEXT STEP — don't keep guessing: run WM UNCONDITIONED, read the poses IT outputs (its COLMAP sparse/cameras),
and align them to our lingbot poses to DERIVE the exact transform** (handedness / normalize_poses / scale)
rather than coin-flipping conventions.
**★ RESOLVED (2026-08-08, overnight):** convention is **`views['camera_poses'] = np.linalg.inv(lb_w2c)`**
(C2W, OpenCV, no flip/transpose) + `cond_flags=[0,0,1]` (pose-only). VERIFIED by an Opus agent: since both
recons anchor frame 0 at identity, their world frames are pre-co-registered → compare camera forward-axes
DIRECTLY (Umeyama is misleading — WM's own 16-frame poses are half-garbage, giving a fake 22% residual). Only
inv(lb_w2c) OpenCV matches WM's good frames to 0.1–0.5°; every other convention throws frame 1 off by 90–180°.
`~/HunyuanWorld-Mirror/infer_cond.py` now has this config (built by `~/build_infer_cond.py`); run = `~/run_wm_cond4.sh`.
Result deployed at scene **courthouse-wmc** — the fly-through is COHERENT (clean columned front, NO false banners,
posters on the correct annex walls) where every wrong-convention attempt warped. GAINS ARE MODEST though: WM's
uncond guess was already ~half-decent, conditioning mainly fixes its ~5 flipped/garbage frames. **KEY GOTCHA:**
WM's OUTPUT camera-pose head IGNORES the conditioning (always re-estimates its own) — only the GEOMETRY is
conditioned. So auto-eval MUST be geometry-based (render fly-through & look, or photometric/reproj error), NOT
output-pose residual (that metric is invalid — it read identical for all configs). NEXT IDEAS: try pose+intrinsics
with the intrinsics in the code's expected normalized form; feed MORE frames (priors reduce the memory penalty of
pose uncertainty?); or condition + then train a gsplat on the improved geometry.
**★★ DECISIVE VERDICT (2026-08-08, Lara's "it doesn't make a square" catch):** rendered the point maps top-down
(matplotlib, headless — `~/*_pointmap.png` via /mnt/e/wsl_deploy). SAME 16 courthouse frames: **LINGBOT makes a
clean CLOSED rectangular footprint** with the camera centers ringing it in a proper orbit; **World-Mirror (even
with correct C2W pose conditioning) folds the walls into an open, near-flat "nested" shape that never closes.**
So WM's geometry is the weak link — NOT the poses (verified) and NOT the footage (lingbot closes it fine). WM
feed-forward can't hold cross-view depth/scale consistency well enough to close a loop here. **CONCLUSION: for
faithful geometry, lingbot's reconstruction → trained gsplat (courthouse-sky) WINS; World-Mirror's niche is speed
or scenes lingbot can't do, not quality.** The conditioning quest was still worth it: we cracked the convention
rigorously AND definitively bounded WM's ceiling. Diagnostic tool banked: render point maps top-down to check
loop closure (matplotlib scatter, clip 2–98 pct to kill spray, overlay camera centers = -R^T·t from C2W).
**★★★ KOBAYASHI MARU — the fold is FIXABLE externally (2026-08-08 overnight, autonomous):** three Opus agents
confirmed (a) WM's pose priors are effectively INERT in the OSS release (open bug #32 — outputs bitwise-identical
with/without poses; two users report external poses make geometry WORSE), and (b) loop-non-closure is the
DEFINING architectural weakness of the whole DUSt3R/VGGT/WorldMirror feed-forward-pointmap class (no bundle
adjustment / global alignment / loop closure in the forward pass — hence VGGT-Long, MASt3R-SfM, NoDrift3R exist to
patch it). **THE CHEAT THAT WORKS:** WM's per-view geometry is LOCALLY GOOD; the fold is purely global placement.
Take WM's per-view DEPTH predictions (`predictions['depth']`, saved to `<out>/depth/*.npy`), rescale each view to
lingbot's metric depth (per-view `s_i = median(lb_depth/wm_depth)`), and unproject with LINGBOT's poses+intrinsics
(the proven W2C `(cam - extr[:,3]) @ extr[:,:3]` unproject) → the courthouse snaps into a CLEAN CLOSED footprint,
virtually identical to lingbot's own (see `kobayashi.png`). So: supply the global frame externally and WM closes.
**Closed feed-forward SPLAT (next thread, optional):** WM's `predictions['splats']` is PRE-FUSED+filtered
(a length-1 list, ~1.5M gaussians, no per-view split) so you can't per-view-transform it directly; you'd hook
`src/models/models/rasterization.py`'s GaussianSplatRenderer to grab per-view gaussian params (means/scales/quats/
opacity/sh) BEFORE fusion, then apply the same per-view Sim3 (rotate quats by `R_lb_i·R_wm_iᵀ`, scale by `s_i`).
Marginal ROI though — training a gsplat on the closed point cloud ≈ the existing `courthouse-sky` beauty. VERDICT
stands: lingbot geometry → trained gsplat for quality; WM for speed. But we DID wrestle the fold to the ground.

---

> **CURRENT STATE — 2026-08-07 (READ FIRST; supersedes the render-blocker notes below).**
> The offline MP4 renderer was ABANDONED (WSL2 Open3D/EGL is dead) and SIDESTEPPED with something
> better: a **web point-cloud walkthrough**. Reconstruct in WSL → fuse depth maps into a colored
> PLY (pure numpy, no GPU/Open3D — `~/export_tiers.py` + `~/downsample_more.py` in WSL) → serve a
> single-file Three.js first-person fly-through. **LIVE at https://walk.bluekittymeow.com** (hosted
> on Factotum: gallery at `/`, viewer at `/view.html?scene=NAME`; density tiers 1.5/3/6/13M via
> dropdown or number keys 1–4; soft round points; source repo `~/Documents/Git/lingbot-viewer/` on
> MysteryOfGlass; tier PLYs live in `/var/www/walk/` on Factotum). This is the delivered interactive
> path — better than a fixed-camera MP4. Test scene = lingbot-map's bundled "courthouse" sample
> (NOT Kowloon). Point-cloud math: `lingbot_map.utils.geometry.depth_to_world_coords_points`
> (OpenCV world-to-cam); viewer applies `rotation.x=π` for Y-up display.
>
> **Level 2 — 3D Gaussian Splatting: DONE & LIVE (2026-08-07).** Splat viewer at
> `https://walk.bluekittymeow.com/splat.html?scene=courthouse` (also linked from each gallery card
> that has `"hasSplat": true`). Uses GaussianSplats3D via CDN, `sharedMemoryForWorkers:false` so it
> needs no COOP/COEP headers; loads `<scene>_splat_web.ply` (INRIA format, DC-color/SH-deg-0).
> **DEDICATED ENV (torch 2.8 was too new for gsplat):** conda env `gsplat` = torch 2.4.0+cu124 +
> **PREBUILT** gsplat 1.5.3 wheel (`--index-url https://docs.gsplat.studio/whl/pt24cu124`, `--no-deps`,
> then `pip install rich typeguard numpy imageio`). Prebuilt = no nvcc compile = no VM crash. (The
> torch-2.8 env can't build gsplat: JIT-compile crashes the capped VM AND gsplat 1.5.3 is incompatible
> with torch 2.8's `_jit_compile` signature. Don't fight it — use the torch-2.4 env.)
> **WINNING RECIPE (`~/train_splat.py` in WSL, run in `gsplat` env):** dense-init ~3M Gaussians from
> our depth points (native OpenCV frame, no centering), CLEAN L1 loss, tight scales (`log(0.002*extent)`),
> ~8000 iters, Adam. NO densification, NO SSIM — both *regressed* quality on this data (densification
> barely fired because the depth init is already dense; SSIM+opacity-resets added floater speckles).
> Trained in ~3 min on the 4070 Ti. Export = INRIA `.ply` (f_dc=(rgb-0.5)/0.2820948, opacity=raw logit,
> scale=log, rot=wxyz quat). Web version = random-subsample to ~1M Gaussians (`courthouse_splat_web.ply`,
> 68MB); full 3M is `courthouse_splat.ply` (199MB). Verify a splat by re-rendering from a training
> camera with `~/render_splat_test.py` and eyeballing vs the GT frame — caught v3's speckles that way.
> Ceiling is set by the 518×294 input resolution.
>
> **BIG LESSON (2026-08-07): splats OVERFIT to the capture trajectory.** A splat that looks perfect
> from the ~120 training cameras can be a white-blowout/spike mess from *novel* off-trajectory views
> (orbit). ALWAYS verify a splat from an OFF-AXIS camera, never only training views (`render_splat_test.py`
> renders training views → misleading; v5 renders an off-axis view too). Root cause: our footage is a
> narrow *forward walk toward* the building, so nothing constrains Gaussians from the sides/above and
> they sprawl into sheets where the cameras never looked. Online "Courthouse" splats use an ORBITING
> capture; ours doesn't. Mitigations tried: post-hoc pruning of big/faint/far Gaussians (helped a bit,
> not enough); v5 recipe = DefaultStrategy densification+pruning (prune_opa=0.02) + a scale-reg penalty
> `5*relu(max_scale-0.05*extent)` (kills spike Gaussians) → pruned 400k→259k, cleaner but still rough
> off-axis. **Takeaways:** (1) the POINT CLOUD is the robust free-viewpoint experience (no view-dependent
> sprawl); the splat is best appreciated from the front / along the path. (2) For a genuinely clean
> orbitable splat you need orbit-style capture OR heavier regularization/masking. (3) For Kowloon, expect
> the same constraint unless the footage circles a subject. Current live web splat = v5 259k
> (`courthouse_splat_web.ply`, 18MB).
>
> **Kowloon E03** staged at WSL `~/kowloon/kwc_e03_full.mp4` (one continuous ~6.5-min take, zero
> cuts) — reconstruct once the pipeline's dialed in.
>
> **STATUS 2026-08-06: RECONSTRUCTION WORKS. OFFLINE MP4 RENDERER BLOCKED on
> WSL2 EGL.** Smoke test passed — courthouse (120 frames) reconstructed in 62 s
> (~2 it/s), served via viser. CUDA toolkit 12.8 installed, FlashInfer works.
> See CLAUDE.md "WSL2 on MarshLair" (CRITICAL `.wslconfig` cap; systemd-unit
> pattern) and "LingBot-Map".
>
> **PAUSED 2026-08-07 mid-render-debug.** `demo_render/batch_demo.py` reproducibly
> segfaults (SIGSEGV/139) ~23 s into "Processing scenes". ROOT CAUSE FOUND (2 Opus
> research agents + gdb + code read): Open3D's Filament `OffscreenRenderer`
> (`demo_render/rgbd_render/renderer.py:114`) cannot init EGL headless on WSL2 —
> `eglInitialize failed` → segfault. Known-open upstream: isl-org/Open3D#7066.
> The renderer's `_suppress_c_output()` dup2's fd1/fd2 to /dev/null, which is why
> faulthandler was silent (I patched a `LINGBOT_UNMASK=1` bypass into renderer.py
> to reveal the error). Things ALREADY TRIED that did NOT fix it: open3d-cpu wheel
> swap; env vars `OPEN3D_CPU_RENDERING=true` / `EGL_PLATFORM=surfaceless|x11` /
> `LIBGL_ALWAYS_SOFTWARE=1`; `XDG_RUNTIME_DIR` set; mesa/EGL/xvfb installed;
> num_workers 16→1; res 1920×1080→960×540. `eglinfo`: Mesa advertises surfaceless
> at client level but device enumeration returns empty (no working EGL device).
>
> **2026-08-07 UPDATE — WSL software render CONFIRMED DEAD.** Tested the exact
> failing `OffscreenRenderer(640,480)` call in isolation with the bundled software
> renderer, clean env (`OPEN3D_CPU_RENDERING=true`, interfering EGL_PLATFORM/
> LIBGL overrides removed, XDG_RUNTIME_DIR set): segfaults at `eglInitialize
> failed` every time. WSL2 route abandoned. (College try #2, 2026-08-07:
> `open3d-cpu` ships NO bundled libEGL so `OPEN3D_CPU_RENDERING=true` was a silent
> no-op; swapped back to full `open3d==0.19.0` — STILL fails `eglInitialize`, no
> bundled software libs in the wheel; forced-llvmpipe/surfaceless also fail. Truly
> dead — do NOT retry any WSL render path.) Also confirmed: **the RENDER stage
> needs only Open3D + torch (torch just for a guarded `torch.cuda.empty_cache()`)
> — NO render_cuda_ext / CUDA needed.** And reconstruction is already saved:
> **120 prediction NPZ files (350MB) live in `~/renders/courthouse/courthouse/`**
> in WSL, loadable via `batch_demo.py --load_predictions`.
>
> **PLAN OF RECORD: render on native Windows (MarshLair) — Lara's call ("the beast";
> NEVER MysteryOfGlass, see [[heavy-compute-marshlair-not-mysteryofglass]]).**
> Native Windows Open3D uses the GPU driver directly (no EGL headless problem).
> Steps: (a) Windows Python env with open3d==0.19.0 + numpy==1.26.4 + opencv +
> pyyaml + tqdm + torch (benchmark venv on E: already has torch 2.6+cu124, or
> system Py310). (b) Get the render code on Windows (copy ~/lingbot-map/demo_render
> from WSL to E:, or git clone). (c) Copy the 120 NPZs from WSL (/mnt/e/...) to E:.
> (d) Run `batch_demo.py --load_predictions <npz_dir> ...` → MP4. **WATCH-OUT:**
> MarshLair is HEADLESS Windows (no monitor) — Open3D GPU context needs the
> interactive CONSOLE session, so launch via the schtasks-interactive pattern like
> ComfyUI (NOT a Session-0 background service — that has no GPU). Verify numpy==1.26.4.
**The idea:** feed Kowloon Walled City archival footage into lingbot-map and see
what a modern streaming-reconstruction model makes of genuinely hard footage.

## What lingbot-map is

https://github.com/Robbyant/lingbot-map — feed-forward 3D foundation model for
STREAMING scene reconstruction from video (Apache 2.0, ~16k stars, active 2026).
Input: video file or image folder. Output: point-cloud reconstruction + camera
trajectory; offline rendering pipeline produces MP4 flythroughs with trajectory
overlays. Claims ~20 FPS at 518×378 over 10,000+ frame sequences (Geometric
Context Transformer, anchor context + trajectory memory, FlashInfer paged-KV).

## Target hardware & current blockers

- **MarshLair** (Windows, 192.168.1.156, RTX 4070 Ti SUPER 16 GB) — see
  ~/.claude/CLAUDE.md for SSH conventions (username has a space; cmd.exe shell).
- **Blocker 1 — driver:** repo wants CUDA 12.8 + PyTorch 2.8; MarshLair's NVIDIA
  driver is 561.09 (CUDA 12.6 ceiling). We hit this exact ceiling today with a
  stock ComfyUI portable (torch cu130 → "CUDA not available" + access violation;
  fixed by downgrading to cu126). For lingbot-map, **the driver needs updating
  first** — owner is aware and warm to it; it's her gaming box, so schedule with
  her, expect a reboot. FIRST TRY cu126 torch anyway (2.13.0+cu126 exists and
  works today) — lingbot may run on torch>=2.8 built for cu126-compatible stacks
  or with minor pin loosening; only force the driver update if FlashInfer/torch
  genuinely refuse.
- **Blocker 2 — OS:** Linux-focused (FlashInfer, ffmpeg pipeline). Plan on
  **WSL2 + Ubuntu on MarshLair** with CUDA-in-WSL (needs the updated Windows
  driver; no separate Linux driver inside WSL). Native Windows is unproven.
- 16 GB VRAM: repo ships `--offload_to_cpu` and reduced-scale flags — expect it
  to fit with those; start at default 518×378 res.

## Hard-won MarshLair gotchas (today's session, will save you an hour each)

1. **pip temp lives on C: which is nearly full** — big wheel installs die with
   ENOSPC mid-download. Always `set TMP=E:\ai\tmp&& set TEMP=E:\ai\tmp` (create
   the dir FIRST and verify — a nonexistent TMP silently falls back to C:).
2. Put everything on **E:** (E:\ai\… has ~260 GB free). Suggested home:
   `E:\ai\lingbot\` (or inside the WSL filesystem if WSL2 route — faster IO).
3. cmd-over-SSH quoting: outer single quotes, `&` chains run sequentially,
   `2>nul` inside compound commands is flaky — prefer separate SSH calls and
   verify each step (dir the thing you just made).
4. GPU-over-SSH works fine for CUDA compute (proven: Cycles OptiX farm renders,
   ComfyUI). The old "Session 0 has no GPU" note in CLAUDE.md applies to
   OpenGL/display, not CUDA.
5. There are now TWO ComfyUI instances: production (E:\ComfyUI_windows_portable,
   port 8188, Victoriana — DO NOT TOUCH) and the 3D forge
   (E:\ai\comfy3d\ComfyUI_windows_portable, port 8189, torch 2.13.0+cu126).
   Don't disturb either; lingbot gets its own env.

## Suggested plan of attack

1. Chat with Lara about the driver update window (Studio driver recommended,
   current stable ≥ CUDA 12.8 support), do it, reboot, verify `nvidia-smi`.
2. `wsl --install -d Ubuntu` (needs its own reboot possibly); inside WSL verify
   `nvidia-smi` sees the 4070 Ti (CUDA-in-WSL comes free with the Windows driver).
3. Clone lingbot-map inside WSL, conda/venv per README, `pip install -e .`
   (temp dir inside WSL is fine; disk lives on C: by default — consider moving
   the WSL vhdx to E: if space complains).
4. Smoke test on their sample data FIRST (README examples) before Kowloon.
5. Kowloon footage: Lara supplies (yt-dlp candidates: 1980s-90s KWC documentary
   walkthroughs). Preprocess: pick continuous handheld/walking segments, avoid
   cuts (streaming model wants continuous motion); trim with ffmpeg; start with
   a 30-60 s clip before feeding long sequences.
6. Deliverables she'll love: the MP4 flythrough render with trajectory overlay,
   plus the point cloud in something Blender can import (PLY) for poking around.

## Comms

House norms: chat decisions with Lara, procedure just runs. Courier bot
(@TheCourierPingsTwiceBot, creds ~/.config/claude-fling/telegram.env) for pings
with sample outputs when she's away; long-poll getUpdates in a background
watcher for replies (never a token poll loop). Traditional greeting applies.
