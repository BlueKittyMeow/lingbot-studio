# Progression Gallery

Diagnostic renders and footage skims from building this pipeline — the "green radiographic" top-down/oblique
point maps and the fly-throughs that show what each stage was actually testing. (More to come — POV
walkthrough captures, per-stage before/afters.)

## Geometry: does the reconstruction close the loop?

- **`courthouse_geometry_lingbot-closed_pointmap.png`** — lingbot-map's reconstruction of the courthouse
  orbit, top-down + oblique. A CLEAN CLOSED rectangular footprint with the red camera centres ringing it in a
  proper orbit. This is what good geometry looks like.
- **`courthouse_geometry_worldmirror-folded_pointmap.png`** — HunyuanWorld-Mirror, fed the SAME 16 frames:
  the walls splay open into a folded, near-flat "nested" shape that never closes. Feed-forward pointmap models
  (DUSt3R/VGGT lineage) have no loop-closure in the forward pass — a known architecture-class limitation.
- **`kobayashi-maru_wm-depth-in-lingbot-frame-closes.png`** — the fix ("changing the test, Kirk-style"): take
  WM's per-view DEPTH and place it in lingbot's already-closed coordinate frame with a per-view scale
  correction → it snaps into a clean closed building (left), matching lingbot's reference (right). WM's *local*
  geometry was good all along; only its *global placement* folded.

## World-Mirror splat fly-throughs

- **`worldmirror_unconditioned_flythrough.png`** — WM's own pose estimate: mostly coherent, a couple garbage
  frames where its 16-frame guess broke down.
- **`worldmirror_pose-conditioned_flythrough.png`** — WM handed our trusted lingbot poses as priors: more
  uniformly coherent (though WM's output-pose head ignores the priors per open bug #32 — only geometry shifts).

## Source-footage skims (picking clean, walkable segments)

- **`courthouse_source-footage_orbit.png`** — the real courthouse (lingbot's bundled sample), 6 frames around
  the full orbit — a multi-sided building with posters on the annex, which fooled the eye during debugging.
- **`cathedral_source-footage_skim.png`** — Cologne Cathedral tour, timestamped skim to find a walkable
  nave-pass (we chose 14:14).
- **`waterlilies_skin-source_skim.png`** — the Monet-esque waterlily-pond footage whose colour we draped over
  the courthouse geometry (the first "skin").
