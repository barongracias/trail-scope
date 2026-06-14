# Build progress log

Append a dated entry per work session: what was built, decisions made, what's verified,
and what's next. Keep newest at the bottom.

## Phasing (from agents/spec.md §Phasing)

- [x] **P1** — backend scaffold, vendoring + `VENDOR_MANIFEST.md`, checkpoint copy +
  SHA gate (startup refusal on mismatch), `/health`, `/model`. (frontend/compose deferred
  to P3/P4 per the walking-skeleton order.)
- [x] **P2** — `preprocess.py` dispatch (FITS/PNG), pixel-scale/tier logic, patch-budget
  check after resample, + tests. **Milestone DONE: NAVSTAR-70 DECam frame end-to-end →
  mask.png + stats.json + input_8bit.png on CPU, reproducing the thesis result.**
- [x] **P3** — `/infer` (sync, 64-patch cap, 413 over), `artifacts.py` (overlay/mask/
  input_8bit/stats), frontend (input/processing/output states).
- [ ] **P4** — Docker/CI, demo assets + `download_demo_assets.py`, README with scope
  sentence + disclaimer.

## Log

### 2026-06-14 — scaffolding created (no code yet)
- `trail-scope/` created as a sibling of `bg492` with `CLAUDE.md` (rules), `agents/spec.md`
  (frozen v1 spec), and `agents/build_context.md` (checkpoint SHA `ff680804…`, vendoring
  source paths, demo source, architecture constants). No application code written yet.
- GitHub remote not yet created; commit locally until the author confirms it exists.
- Next: P1.

### 2026-06-14 — P1 complete (backend skeleton + checkpoint SHA gate + /health, /model)
- **.venv** created in-repo with Python 3.11 (gitignored), CPU torch wheels
  (`--index-url .../whl/cpu`), `opencv-python-headless`, `astropy`, `pillow`, `scipy`,
  `fastapi`/`uvicorn`, `pytest`. Sibling `../astro_venv` untouched.
- **Vendored core** at `backend/trailscope/vendored/` with `VENDOR_MANIFEST.md`
  (source commit `b9a4e602`): `unet.py` (verbatim), `loading.py` (adapted — dropped the
  `attention_unet` branch + rewrote import to local `.unet`), `hough_runner.py`
  (verbatim; `_apply_hough` used for the overlay), `preprocess_core.py` (the 10 listed
  functions, bodies verbatim; `PATCH_SIZE=528` + `LOCKED_NORMALISATION` re-declared
  locally). No `src.*` imports — enforced by `tests/test_no_src_imports.py`.
- **Checkpoint** `model-best.pth` copied to `backend/checkpoints/` (committed; public per
  thesis code-availability statement). On-disk SHA verified
  `ff680804…` == `EXPECTED_CHECKPOINT_SHA256`.
- **SHA gate**: `config.checkpoint_sha_ok()` + `inference.ModelService.load()` refuse to
  load on mismatch (raise `CheckpointError`); lifespan disables inference and `/health`
  reports `model_sha_ok=false` rather than crashing. After instantiating the U-Net,
  **485,673 trainable params asserted** (matches `EXPECTED_PARAM_COUNT`).
- **config.py** holds every locked invariant (threshold 0.45, Hough 0.1/50/100/250/3,
  PATCH_SIZE 528, target 0.56 arcsec/px, 64-patch budget, 64 MB cap), the verbatim scope
  sentence + disclaimer. **main.py is thin** (routes/CORS/lifespan/errors only),
  serving `GET /health` and `GET /model` (disclaimer + neutral tiers present).
- Tests: `test_model.py` (param count, SHA match, ModelService load, corrupted-checkpoint
  refusal, /health + /model honesty) and `test_no_src_imports.py`.
- Next: **P2** — `preprocess.py` (FITS/PNG dispatch, tier/pixel-scale, post-resample
  patch budget) + tests, then the DECam-frame end-to-end milestone.

### 2026-06-14 — P2 complete + RISK-RETIRING MILESTONE met
- **preprocess.py**: single `preprocess_image()` entry wrapping the vendored core.
  FITS (astropy, explicit `hdu_index` or first 2-D HDU, lists HDUs on error) vs display
  (PIL preserving bit depth; RGB→luminance ITU-R 601 with a provenance flag). Order:
  load+clean (non-finite→median) → resolve pixel scale (override→header→unknown) →
  resample `scale/0.56` INTER_AREA when known → stretch (8-bit passthrough; else ZScale
  with percentile fallback) → **patch budget AFTER resample, 413 over 64**. Neutral tier:
  `recipe_matched` (FITS+known scale), `in_domain_like` (8-bit display), `best_effort`
  (unknown scale / other). Provenance carries all spec keys + checkpoint SHA + vendored
  commit. `PreprocessError(status, detail)` carries the 413/400 to the route layer.
- **inference.py**: added `hough_segments()` (segment list for stats, mirrors the vendored
  `_apply_hough` call); `hough_overlay()` still uses the vendored function (the path
  validated against thesis `hough_pixel_count`).
- **artifacts.py**: `run_inference()` → prob canvas (threshold 0.45), binary mask,
  connected-component stats (`cv2.connectedComponentsWithStats`; ellipse fit only with
  ≥5 boundary points, else null axis/orientation — no fake values), optional Hough,
  and writes `input_8bit.png` (always), `mask.png`, `overlay.png`, `stats.json`.
  "predicted mask/component" vocabulary throughout.
- **schemas.py**: full `InferStats` (+ `Provenance/ImageStats/ModelOutput/PredictedComponent/
  HoughStats/TimingStats`) and `InferResponse`. Produced stats.json validates against it.
- **scripts/download_demo_assets.py**: NOIRLab retrieval (httpx) for the 9 predeclared
  public DECam frames; downloads to gitignored `.demo_cache/` (never committed; NOIRLab
  acknowledgement emitted). MeerLICHT never touched.
- **MILESTONE**: NAVSTAR-70 (exp 1134933 det 5) end-to-end on CPU in ~1.6 s →
  `predicted_mask_pixel_count=13113` vs thesis **13111** (Δ2, cv2/torch float diffs),
  `max_model_probability=0.9995` (exact), processed **1926×962**, **8 patches**,
  tier `recipe_matched`; streak cleanly overlaid. The vendored pipeline reproduces the
  locked detector path — risk retired.
- Tests: 23 passing (preprocess contract ×12, synthetic inference/ellipse-null ×4,
  e2e DECam ×1 [skips without the frame], P1 model/health ×6, no-src-imports). Used the
  fixed DECam plate scale 0.2634 as override to match the thesis (header reads ≈0.2623;
  the general path uses the header value).
- Next: **P3** — `/infer` route (sync, 64-patch cap, 413) + results file serving +
  the Next.js frontend (input/processing/output states).

### 2026-06-14 — P3 complete (/infer + results serving + frontend)
- **main.py** (still thin): `POST /infer` (multipart `file`, `hough` default true,
  optional `pixel_scale_arcsec`/`hdu_index`) — upload validation lifted from InterPyApp
  (extension allowlist incl. `.fits.fz`, 64 MB streamed cap, UUID storage), runs
  preprocess→artifacts via `run_in_threadpool`, deletes the upload in `finally`, maps
  `PreprocessError`→413/400, returns `{result_id, stats}`; 503 when the checkpoint gate
  failed. `GET /results/{id}/{file}` serves the four artifacts via `FileResponse` with a
  32-hex id guard + filename allowlist (404 otherwise); result dirs cleared on startup.
- **test_api.py** (9 tests): unsupported-extension 400, schema-valid sync infer with all
  four honesty fields, all four result files served, post-resample 413 path, results 404s
  (garbage id / unknown artifact / before-infer), health ok, and corrupted-checkpoint →
  503 startup refusal. Backend suite now **32 passing**.
- **Frontend** (Next 14 + Tailwind v4, chassis lifted from InterPyApp): `lib/api.ts`
  typed client (`ApiError`, env base URL, `infer`/`resultUrl`/`healthCheck`/`getModelInfo`).
  `app/page.tsx` is the three-state tool — **input** (drag-drop, accepted types + 64 MB,
  DECam demo picker, pixel-scale + HDU overrides, Hough toggle default on, locked model
  card, caveat), **processing** (filename, spinner, staged text, the 64-patch note +
  caveat), **output** (neutral blue/grey tier banner [same style all tiers] + warnings;
  `input_8bit.png` vs a `<canvas>` overlay with opacity slider + independent Mask/Hough
  toggles; result summary; provenance panel; predicted-components table with `—` for null
  axis/orientation; downloads; disclaimer footer). `CanvasCompare.tsx` composes input +
  recoloured mask + Hough segments (from stats) client-side via CORS-clean blob bitmaps.
- Disclaimer present on input page, output page, `/model`, stats.json (README next, P4).
  Vocabulary stays "predicted mask/component" (a frontend test forbids "detection").
- **Demo asset**: `frontend/public/demo/decam_navstar70_crop.png` — a 700×662 8-bit crop
  derived from the public NAVSTAR-70 DECam frame (streak clearly visible; NOIRLab ack in
  footer). No MeerLICHT data.
- Verified: `npm run lint` clean, `npm test` (node --test ×4) pass, `npm run build` OK;
  live backend `/infer` on the demo crop → tier in_domain_like, mask 8793 px, 1 component,
  16 Hough segments, 4 patches; all four artifacts serve 200; garbage id → 404.
- Next: **P4** — docker-compose, CI workflow, `scripts/{docker_*,run_local}.sh`, README
  with scope sentence + disclaimer + NOIRLab acknowledgement, `requirements.lock`/Docker
  checkpoint SHA verification at build.
