# Build progress log

Append a dated entry per work session: what was built, decisions made, what's verified,
and what's next. Keep newest at the bottom.

> v1 (P1–P4) is complete and verified end-to-end in a browser. Post-v1 enhancement plan
> lives in **`agents/roadmap.md`** (v1.1–v1.4, with guardrails). The frozen spec is
> `agents/spec.md`.

## Phasing (from agents/spec.md §Phasing)

- [x] **P1** — backend scaffold, vendoring + `VENDOR_MANIFEST.md`, checkpoint copy +
  SHA gate (startup refusal on mismatch), `/health`, `/model`. (frontend/compose deferred
  to P3/P4 per the walking-skeleton order.)
- [x] **P2** — `preprocess.py` dispatch (FITS/PNG), pixel-scale/tier logic, patch-budget
  check after resample, + tests. **Milestone DONE: NAVSTAR-70 DECam frame end-to-end →
  mask.png + stats.json + input_8bit.png on CPU, reproducing the thesis result.**
- [x] **P3** — `/infer` (sync, 64-patch cap, 413 over), `artifacts.py` (overlay/mask/
  input_8bit/stats), frontend (input/processing/output states).
- [x] **P4** — Docker/compose/CI, scripts, demo assets + `download_demo_assets.py`,
  README with scope sentence + disclaimer + NOIRLab acknowledgement.

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

### 2026-06-14 — P4 complete (Docker / compose / CI / scripts / README) — v1 DONE
- **backend/Dockerfile**: `python:3.11-slim`, venv, system libs (libgomp1, libglib2.0-0,
  curl); installs CPU torch from the PyTorch CPU index **then** `requirements.lock` (so the
  pinned torch is already satisfied and never pulls CUDA). **Build-time checkpoint SHA-256
  gate** fails the build on mismatch (verified the snippet runs: `ff680804…`). Non-root
  `app` user; uvicorn CMD. `backend/.dockerignore` excludes venv/caches/uploads/results/
  tests (keeps `checkpoints/`).
- **frontend/Dockerfile** (P3) + **docker-compose.yml**: backend (8000, healthcheck on
  /health, `ALLOWED_ORIGINS`) + frontend (3000, `NEXT_PUBLIC_API_URL=localhost:8000` so
  the browser hits the host-published backend; baked default matches). `docker compose
  config` validates.
- **scripts**: `docker_build.sh` / `docker_up.sh` / `docker_down.sh` and `run_local.sh`
  (in-repo `.venv`, CPU torch, both servers) — all chmod +x.
- **CI** (`.github/workflows/ci.yml`): backend job (CPU torch + lock + ruff + pytest),
  frontend job (npm install + lint + test + build), and a docker-build job (needs both;
  builds both images, exercising the build-time SHA gate). No GPU anywhere.
- **ruff**: `backend/ruff.toml` (line-length 100, select E/F/W/I, excludes the frozen
  `vendored/`). Fixed import order in main.py + an unused import in a test; `ruff check`
  clean.
- **README**: scope sentence + disclaimer + tier table + locked-knobs note + Docker/local
  quick-starts + API list + demo-data section with the **NOIRLab acknowledgement** and the
  no-MeerLICHT statement.
- Final validation: backend 32 tests pass + ruff clean; frontend lint clean, node tests
  (4) pass, `npm run build` OK; compose config valid; build-time SHA gate verified.
  Docker images not built locally (daemon off) — the CI docker-build job covers that.
- Disclaimer + neutral tiers now present in all five required places: input page, output
  page, README, `/model`, `stats.json`. **v1 build complete (P1–P4).**

### 2026-06-15 — v1.1–v1.3 enhancements (built + browser-verified)
Implemented the non-scope-breaking roadmap items (see `agents/roadmap.md`); the frozen v1
contract and all honesty guardrails are unchanged.
- **Backend** (40 tests passing, ruff clean):
  - Demo-result cache: `sha256(bytes + options) → result_id`, bounded LRU, self-healing,
    cleared on startup (`infer.cache_hit`).
  - Concurrency guard: `MAX_CONCURRENT_INFER` (default 2) → **429** on overflow.
  - Result-dir **TTL cleanup** (`RESULTS_TTL_SECONDS`, default 3600) swept per request.
  - `prob.png` grayscale model-confidence map + `original_preview.png` (pre-resample "as
    uploaded", only when resampling changed geometry); `stats.artifacts` lists what's
    present; serving allowlist + a `bundle.zip` of all artifacts.
  - `POST /inspect` lists FITS HDUs (index/type/shape/is_2d_image) for the picker.
  - CPU speedups: `torch.set_num_threads(cpu_count)` at load; `torch.inference_mode()`
    wrapping the (frozen, untouched) vendored inference call.
  - New tests: `tests/test_api_v11.py` (prob/original artifacts, cache reuse, 429, /inspect,
    bundle.zip, TTL cleanup).
- **Frontend** (lint/test/build clean): prob-heatmap toggle with a colourmapped layer +
  **cursor probability readout** (samples prob.png); **click-a-component-to-highlight**
  (bbox on the canvas); **crop-to-fit** rubber-band on the 413 path (raster only);
  **FITS HDU picker** dropdown via `/inspect`; **original-vs-input** 3-panel for resampled
  inputs; **Hough off/on side-by-side** compare; **zoom/pan** on the overlay; paste-from-
  clipboard upload; copy-provenance button; "what's this?" tier tooltip; Confidence + .zip
  downloads.
- **Verified in-browser (Chrome via Playwright), zero console errors:** demo flow + all
  toggles; component highlight; Hough compare (2 canvases); FITS upload → HDU dropdown
  (`[1] CompImageHDU 4094×2046`) → `recipe_matched`, 3-panel, 13113 mask px; cursor readout
  `p=…`; cache hit (identical result_id, logged); `bundle.zip` (5 files).
- **Deferred (L):** #11 async job queue and #12 full-frame >64 patches (depends on #11 and
  relaxes the locked budget) — left out to keep the verified sync path stable.

### 2026-06-16 — remaining roadmap items: #10, #11, #12, #14, v1.4 (built + verified)
Completed every outstanding roadmap item. Sync /infer path unchanged; honesty guardrails
and locked invariants intact (the async ceiling and confidence map are not tuning knobs).
- **v1.4 vendored checksum guard**: `vendored/CHECKSUMS.sha256` + `test_vendored_integrity.py`
  (re-hashes the 4 frozen files; loud failure on accidental edit). Manifest documents it.
- **#10 TorchScript**: `_maybe_torchscript` traces the locked model at load,
  `optimize_for_inference`, and **verifies equivalence** (max abs diff < 1e-4 on a random
  528² patch) before use; eager fallback on any mismatch/error (`USE_TORCHSCRIPT` env).
  `ModelService.backend` reports "torchscript"/"eager"; param count read from eager weights.
- **#14 CI Playwright**: committed `frontend/e2e/smoke.mjs` (drives demo → output, asserts
  overlay pixels + zero console errors) + a CI `e2e` job (boots backend+frontend, installs
  chromium, runs it). `npm run test:e2e`; unit `test` still scoped so it ignores e2e.
- **#11 async jobs + #12 full-frame** (additive): `trailscope/jobs.py` `JobManager` (one CPU
  worker, `asyncio.Semaphore`, status mirrored to `status.json`); `POST /jobs` →
  `{job_id}`, `GET /jobs/{id}/status` (queued→preprocessing→inferring→rendering→done|error);
  `preprocess_image(max_patch_budget=…)` lets the jobs path use `MAX_JOB_PATCH_BUDGET`
  (256) while sync stays at 64; `artifacts.run_inference(on_stage=…)` reports coarse
  progress. Pending-jobs cap → 429. Frontend: "Process large images" toggle drives the
  async submit+poll flow; processing view shows stage · patch count.
- **Verified**: backend 48 tests + ruff clean; frontend lint + 5 unit tests + build clean;
  e2e smoke passes (system Chrome). Live: 81-patch image → sync 413, async job completes
  (preprocessing→inferring·81→rendering→done, 30363 mask px, artifacts served), UI async
  flow reaches the full-frame output; zero console errors. TorchScript backend active.
- **All roadmap items now complete** (v1.1–v1.4). Remaining future ideas are only the
  explicitly out-of-scope ones (threshold slider, benchmarking, etc. — never to build).

### 2026-06-16 — v1.5 interpretability + UX (built + browser-verified)
Seven additive items (roadmap v1.5: 1,2,3,7,8,9,10); guardrails intact.
- **#3 per-component confidence**: `_component_stats` now samples the prob canvas →
  `mean_probability`/`max_probability` per component (schema + table columns), with an
  honest caption ("not a likelihood that a real object is present"). +tests.
- **#7 cancel job**: `JobManager.cancel` + `DELETE /jobs/{id}` + new terminal `cancelled`
  state (CancelledError handled; in-flight threadpool work finishes in background, result
  dropped). Frontend Cancel button in the large-mode processing view. +tests.
- **#1 confidence colourbar**: `ConfidenceLegend` shares the exact canvas colourmap;
  shown when "Model confidence" is on.
- **#2 About panel**: collapsible method/tiers/thesis-link explainer on the input page.
- **#9 tabbed output + synced split-view**: Overview/Components/Provenance/Downloads tabs;
  model-input and overlay are both `CanvasCompare` sharing a lifted `view` → zoom/pan in
  sync (verified: identical transforms). Wheel-zoom moved to a non-passive native listener
  (kills the passive-preventDefault console warning).
- **#8 a11y**: tab roles/aria-selected, slider + icon-button aria-labels, keyboard-operable
  dropzone and component rows, focus outlines, `role=status` processing announcer.
- **#10 demo gallery**: 3 small DECam crops (NAVSTAR-70, STARLINK-2600, DELTA-2; ~564 KB
  total) located by the predicted-mask bbox; picker lists all three.
- **Verified**: backend 52 tests + ruff clean; frontend lint + 5 tests + build clean; live
  browser run of every feature (tabs, legend, synced zoom, per-component conf., cancel,
  gallery) with **zero console errors**. Deferred: #4 shape descriptors, #5 arcsec units,
  #6 confidence histogram.
