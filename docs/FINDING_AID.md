# Finding Aid — what each scene & experiment was testing

A key to the collection: every deployed scene, the footage + settings behind it, and what we were
actually *testing* as we iterated. Live gallery: **https://walk.bluekittymeow.com**.
Companion image gallery with the "green radiographic" point-maps: `docs/gallery/`.

---

## Point-cloud walkthroughs (`view.html?scene=<id>`)

| Scene id | Footage | Key settings | What it was testing / the finding |
|---|---|---|---|
| `kowloon` | WSJ 1991 Kowloon Walled City corridor (E03), cropped to remove burned-in subtitles + fisheye vignette, **whole 40s walk** | fps 8, sw32/nsf4, leveled | The hard-footage hero. **Breakthrough: cropping the burned-in overlays fixed a total tracking collapse.** Whole-walk v3 = ~3° drift, 13.5M pts. |
| `catacombs` | 4K catacombs tour (clean, empty) | fps 10 | The model's *dream* input — sharp, textured, deserted → walked deep and true. Our early proof it works. |
| `catacombs2` | Catacombs 7:00–9:00 ossuary wall, right-crop + downscale | fps 10, sw32/nsf4, leveled | **fps + quality-param sweet spot.** vs the sparse first pass: drift 6.1°→5.2°, and `num_scale_frames` nearly doubled the recovered walk (better-proportioned geometry). |
| `catacombs20` | Same ossuary, **fps 20** | fps 20, sw32/nsf4, leveled | **"Too dense" test.** fps20 → drift 22°, net walk collapsed 1.5→0.29 — the degenerate side of the curve. BUT visually rich/enclosed over its shorter span, so reframed from "collapse" to "dense pass" (Lara's catch). |
| `courthouse` | lingbot-map bundled sample orbit | (original) | Baseline test-bed before real footage. |
| `courthouse-sparse` | Courthouse orbit, **stride 2** (143 views) | sw32/nsf4 | **Density A/B — SPARSE.** |
| `courthouse-dense` | Courthouse orbit, **stride 1** (286 views) | sw32/nsf4 | **Density A/B — DENSE.** Came out *mushier* (drift-scatter: 2× frames = 2× accumulated drift, surfaces smear). Sparse won → **house rule: don't over-sample a long path.** |
| `courthouse-waterlilies` | Courthouse geometry + **waterlily-pond colour** | `skin_fuse.py` | **First skin.** Geometry keeps its true positions; colour sampled from Monet-esque footage. Proof of concept for "geometry from one world, appearance from another." |
| `cathedral` | Cologne Cathedral 4K tour (mzOwOSm2ubE), 14:14–14:34 nave walk, downscaled | fps 10, sw32/nsf4, leveled | **Cathedral settings test** (20s) before the full run. 200 frames, net walk 1.85, drift 8.4°. |

## Gaussian splats (`splat.html?scene=<id>`)

| Scene id | Method | What it was testing / the finding |
|---|---|---|
| `courthouse` | Old trained gsplat (Aug 7) | Baseline trained splat. |
| `courthouse-wm` | World-Mirror **feed-forward** | Fast splat in seconds, but *rough*: pose-uncertainty spray + sky blowout. Feed-forward has a roughness ceiling. |
| `courthouse-fresh` | Trained gsplat on the **crisp sparse** cloud | Beats the old blob — better reconstruction → better splat. |
| `courthouse-sky` | **Sky-masked** trained gsplat | **The beauty champion.** Sky segmented out of the training loss → sharp building, clean empty sky. |
| `courthouse-wmc` | World-Mirror **pose-conditioned** (C2W lingbot poses) | The pose-conditioning quest: convention rigorously cracked, but WM's geometry still *folds* (loop-closure is an architecture-class limit; and its pose priors are inert per open bug #32). |

## The experiment arc (chronological, the "why we iterated")

1. **Kowloon tracking collapse** → cropping the burned-in timecode/logo/subtitles + fisheye vignette was the whole breakthrough (static overlays fool the tracker into "camera stationary").
2. **fps sweet spot** → a valley: too sparse (fps 3–4) breaks tracking with big jumps; too dense (fps 20 on a long path) drift-smears; **~fps 10 is the middle.**
3. **Density vs drift-scatter** → more frames buy coverage but each carries drift; over-sampling smears surfaces (courthouse dense mushier than sparse).
4. **Quality params** → `num_scale_frames`/`kv_cache_sliding_window` toward defaults improve geometry, BUT 8/64 MCE-crashes *this* VM; **sw32/nsf4 is our safe ceiling.**
5. **Sky-masked training** → segment sky out of the gsplat loss to kill outdoor blowout.
6. **World-Mirror** → installs clean (no nvcc), splats in seconds, but *cannot close a 360° loop* (DUSt3R/VGGT-class limit) and its pose priors are inert (bug #32). **"Kobayashi Maru": place WM's per-view depth in lingbot's already-closed frame → it snaps shut.** WM's local geometry was good; only global placement folded.
7. **Stitching** → two overlapping windows + Umeyama Sim(3) on shared camera centres = long walks at full fps-10 (recipe in the brief).
8. **Skinning** → keep geometry, swap the colour source → surreal (`skin_fuse.py`).

---

*Full technical detail, recipes, and gotchas: `docs/lingbot_map_brief.md`. Pipeline scripts: `scripts/`.*

---

## Update 2026-08-08 — the VRAM ceiling, mapped & broken

We set out to push the ancillary knobs (`kv_cache_sliding_window`, `num_scale_frames`) past our safe `sw32/nsf4`. Full arc:

1. **Mapped the ceiling.** At native 518 res, `sw48/nsf4` is the hard max on 16GB — `sw56`/`sw64` OOM (KV pool), `nsf8` OOMs (upfront scale-phase activation spike). Confirmed on a clean GPU, on **both** SDPA and FlashInfer. The CPU-throttle defeats the *MCE crash* but not the *VRAM* wall (it's parallelism-in-memory, not slow-able compute).
2. **Built FlashInfer** (was blocked by no nvcc): cuda-nvcc 12.8 + gcc-14 + conda-forge sysroot 2.28 (dodges Ubuntu-26.04 glibc 2.41) + WSL libcuda linking, all MCE-safe. Payoff = ~12% speed, not headroom — it does NOT raise the ceiling (paged pool pre-sized to window). Recipe in the brief.
3. **Broke the ceiling with a code patch.** `--image_size` is the quadratic memory lever but was locked (checkpoint `pos_embed` hard-baked to 518). Patched `load_model` to bicubic-interpolate the pretrained pos_embed to a smaller grid → **at `image_size 448` the FULL model defaults `sw64/nsf8` finally fit** (impossible at 518). The model is resolution-agnostic everywhere else (RoPE + DPT head + KV all runtime-sized; the ViT already interpolates pos_embed each forward). Legal sizes: multiples of 14 (448/392/378, not 384).

| Scene id | Footage | Key settings | What it was testing / the finding |
|---|---|---|---|
| `catacombs2-max` | Same ossuary wall, first-200 segment | **FULL defaults sw64/nsf8 @ image_size 448**, pos_embed patch + FlashInfer, leveled | **The ceiling-break.** Proof the model's max knobs run on 16GB via the resolution trade. Coarse drift metrics = dead heat vs `catacombs2` (sw32/nsf4@518) → the A/B is a visual/surface judgement, deployed side-by-side on the wall. |

**Open follow-ups:** (a) full-1500-frame render at 448 hit a `32 vs 22` patch-grid error (very-long-seq path; 200-frame renders clean) — investigate; (b) FP8-KV path (`ureeey` fork `--kv_cache_fp8`) = 2nd route to sw64 keeping 518, at a pose-quality cost; (c) adopt the cleaner `--model_img_size` build-at-518 patch (agent-recommended, avoids a negligible double-resample).

---

## RESOLVED — the max-knobs fair test (eyeball, 2026-08-08)

The `catacombs2-max` question settled. Fair test: **identical first 300 frames**, only settings differ
(`cata2max300` sw64/nsf8@448 vs `catacombs2q` sw32/nsf4@518). Top-down topography + drift metrics were a
**dead heat** — the numbers said "wash." But walking BOTH in the web viewer at 6M pts flipped it: the
**max-knobs reconstruction is visibly cleaner and more complete** — filled surfaces, legible lit chamber, and
**far fewer floating-streak artifacts**; the lighter config leaves vertical floaters hanging in the void and more
holes. **Max knobs (`sw64/nsf8@448`) WIN — adopted as the reconstruction default.**

**Lesson (Lara):** eyeballs are required for ANY visual 3D/2D comparison. Metrics measure the *trajectory*; they
do NOT see surface completeness or floaters. Metrics verify; they don't see. (The whole VRAM-ceiling → FlashInfer
→ pos_embed-patch adventure was worth it precisely to be *able* to run the max knobs and discover this.)
