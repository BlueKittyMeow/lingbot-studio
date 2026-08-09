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
