# Experiments — running list

A living backlog of what we want to try. Check things off, add freely. Findings from completed
experiments migrate to `FINDING_AID.md` + `lingbot_map_brief.md`.

**Baseline settings (as of 2026-08-08):** reconstruct at `--fps 10`, `--use_sdpa`,
`--kv_cache_sliding_window 32`, `--num_scale_frames 4`. That's our *VM-safe ceiling* (the model's 64/8
defaults MCE-crash this box; even sw48 crashed). Fuse with leveling. Deploy to walk.bluekittymeow.com.

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
- [ ] **Fresh-VM sw48/nsf4 attempt** — push the quality knobs one rung past our safe 32/4 (borderline; crashed
  once, might hold on a clean VM).

## 🗺️ Big / ambitious

- [ ] **Massive maps** — the whole Kowloon tape, a 5000-frame catacomb (needs stitching + Potree).
- [ ] **Progression gallery site** — a `walk`-style page cataloguing the iteration story (point-maps + POV
  captures per stage). Pairs with `FINDING_AID.md`.

## ✅ Done (see FINDING_AID.md for detail)

Kowloon whole-walk (cropped, leveled) · catacombs / catacombs2 / catacombs20 (fps sweet-spot + dense pass) ·
courthouse density A/B (sparse wins) · sky-masked trained gsplat (the beauty) · World-Mirror install +
pose-conditioning + Kobayashi-Maru closure proof · first skin (waterlily courthouse) · cathedral 20s test ·
the studio repo + finding aid + gallery.
