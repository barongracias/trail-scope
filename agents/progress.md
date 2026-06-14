# Build progress log

Append a dated entry per work session: what was built, decisions made, what's verified,
and what's next. Keep newest at the bottom.

## Phasing (from agents/spec.md §Phasing)

- [x] **P1** — backend scaffold, vendoring + `VENDOR_MANIFEST.md`, checkpoint copy +
  SHA gate (startup refusal on mismatch), `/health`, `/model`. (frontend/compose deferred
  to P3/P4 per the walking-skeleton order.)
- [ ] **P2** — `preprocess.py` dispatch (FITS/PNG), pixel-scale/tier logic, patch-budget
  check after resample, + tests. **Milestone: one DECam demo frame end-to-end through the
  backend → mask + stats JSON on CPU.**
- [ ] **P3** — `/infer` (sync, 64-patch cap, 413 over), `artifacts.py` (overlay/mask/
  input_8bit/stats), frontend results page.
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
