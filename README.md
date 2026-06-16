# trail-scope

A single-image **qualitative inference demo** for satellite-trail detection: upload an
astronomical image, run a locked U-Net (+ optional probabilistic Hough), and get back an
overlay, a binary predicted mask, and a stats summary. FastAPI backend + Next.js frontend
+ Docker Compose.

> **Scope.** trail-scope is a qualitative, single-image inference demo of the locked
> thesis detector. Not a benchmark, not a validated cross-domain tool, no training, no
> tunable thresholds. Qualitative inference only; no performance claims are made for
> uploaded images.

> **Disclaimer.** This is not a validated detector for this input unless it is from the
> original MeerLICHT-style domain.

The detector is the locked winner from an MPhil dissertation (a PyTorch replication of an
ASTA-style U-Net + Hough pipeline), trained on **MeerLICHT 8-bit display PNG patches**.
It is loaded from a single committed checkpoint whose SHA-256 is asserted at startup (and
at Docker build); the app refuses to serve on mismatch.

## What it does

1. Accepts one image (`.fits`, `.fit`, `.fits.fz`, `.png`, `.jpg`, `.jpeg`, `.tif`),
   64 MB max.
2. Preprocesses it through the **locked recipe**: load (FITS via astropy / display via
   PIL, preserving bit depth; RGB→luminance), clean non-finite pixels, resolve pixel
   scale (your override → FITS header → unknown), resample to ~0.56″/px when the scale is
   known, stretch (ZScale for scientific data with a percentile fallback; 8-bit display
   passes through), then check the **patch budget after resampling** (images over 64
   patches are rejected).
3. Runs the locked U-Net (full-image z-score normalisation, stride-528 tiling, threshold
   **0.45**), with optional probabilistic Hough overlay.
4. Returns `input_8bit.png` (exactly what the model saw — the key honesty artifact),
   `mask.png`, `overlay.png`, and `stats.json`.

Stats use **"predicted mask / predicted component"** language — never "trail" or
"detection". Every result is tagged with a neutral, informational tier:

| Tier | Meaning |
|---|---|
| `in_domain_like` | 8-bit display image at a plausible scale. |
| `recipe_matched` | FITS with a header-resolved pixel scale (the validated DECam-style recipe). |
| `best_effort` | Everything else (e.g. unknown pixel scale — the model is not scale-invariant). |

The only user-facing knobs are: **Hough overlay on/off**, an optional **pixel-scale
override**, and an optional **FITS HDU index**. The model, threshold, normalisation, Hough
parameters, stretch, and patch size are locked constants.

## Quick start (Docker)

```bash
./scripts/docker_build.sh   # builds backend (CPU torch) + frontend images
./scripts/docker_up.sh      # frontend → http://localhost:3000  ·  backend → http://localhost:8000
./scripts/docker_down.sh
```

## Quick start (local, no Docker)

Requires Python 3.11 and Node 20+.

```bash
./scripts/run_local.sh      # creates ./.venv (CPU torch), installs deps, runs both servers
```

Or manually:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.lock
(cd backend && uvicorn main:app --port 8000)
# in another shell:
(cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev)
```

## API

- `GET /health` → `{status, version, model_sha_ok}`
- `GET /model` → static metadata (architecture, 485,673 params, threshold 0.45, patch
  size, training domain, tier definitions, thesis-repo link, full disclaimer).
- `POST /infer` — multipart `file` (required), `hough: bool = true`,
  `pixel_scale_arcsec: float | null`, `hdu_index: int | null`. Synchronous (≤ 64 patches,
  else 413). Returns `{result_id, stats}`.
- `POST /inspect` — multipart `file`; lists a FITS file's HDUs for the HDU picker.
- `POST /jobs` + `GET /jobs/{id}/status` — **opt-in async path** for large/full-frame
  images (over 64 patches, up to a hard `MAX_JOB_PATCH_BUDGET` ceiling, default 256).
  `/jobs` returns `{job_id}` immediately; poll the status file (`queued → preprocessing →
  inferring → rendering → done | error`). The model, threshold, and recipe are identical to
  `/infer`. **Memory:** a full-budget (256-patch) job peaks at **~1 GB RSS** on CPU
  (compose caps the backend at 2 GB) — lower `TRAILSCOPE_MAX_JOB_PATCH_BUDGET` on small
  hosts.
- `GET /results/{id}/{input_8bit.png | overlay.png | mask.png | prob.png |
  original_preview.png | stats.json | bundle.zip}` — per-result artifacts (cleared on
  startup, swept by TTL). `prob.png` is a qualitative confidence map; `original_preview.png`
  appears only when resampling changed the geometry.

## Demo data

The repo ships only a small cropped 8-bit PNG derived from a public DECam frame
(`frontend/public/demo/`). To fetch a full detector frame on demand:

```bash
python scripts/download_demo_assets.py --expnum 1134933 --detector 5 --out-dir .demo_cache
```

This caches large FITS locally (gitignored, never committed). **No MeerLICHT imagery is
distributed** — that is collaboration data and is out of scope here.

> **Acknowledgement.** DECam demo frames are public products of the NSF NOIRLab Astro Data
> Archive. Based on observations at Cerro Tololo Inter-American Observatory, NSF's NOIRLab,
> managed by AURA under a cooperative agreement with the U.S. National Science Foundation.

## Development

- Backend tests: `cd backend && ../.venv/bin/python -m pytest tests` (the end-to-end DECam
  test self-skips when the frame is absent). Lint: `ruff check .`.
- Frontend: `cd frontend && npm run lint && npm test && npm run build`.
- The minimal inference core is **vendored** (a frozen copy) under
  `backend/trailscope/vendored/` with a `VENDOR_MANIFEST.md`; there are no imports from
  the thesis repo (enforced by a test).
- Project rules: `CLAUDE.md`. Build spec: `agents/spec.md`. Build log: `agents/progress.md`.
