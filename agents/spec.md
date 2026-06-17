# trail-scope — Final v1 Specification (locked 2026-06-12)

Single-image **qualitative inference demo** of the locked thesis satellite-trail
detector. FastAPI + Next.js + Docker Compose, architecture lifted from
InterPyApp. Reviewed three times (Claude plan → Codex review → Claude rescope →
Codex second review); this file is the agreed build spec. Scope position:
post-M9.4 / week-3 work — the re-annotation audit outranks it.

**Scope sentence (verbatim in README, `/model`, and stats.json):** a
qualitative, single-image inference demo of the locked thesis detector. Not a
benchmark, not a validated cross-domain tool, no training, no tunable
thresholds. Qualitative inference only; no performance claims are made for
uploaded images.

## Locked invariants (constants, never API parameters)

- Checkpoint: `results/checkpoints/model-best.pth` from the thesis repo at a
  pinned commit, fetched at Docker build, **SHA-256 asserted at startup**
  (`EXPECTED_CHECKPOINT_SHA256` in `config.py`); app refuses to serve on
  mismatch.
- `THRESHOLD = 0.45`; Hough `(input_threshold=0.1, votes=50, min_len=100,
  max_gap=250, draw_px=3)`; `PATCH_SIZE = 528`; normalisation `full_image`;
  `TARGET_ARCSEC_PER_PX = 0.56`; `MAX_PATCH_BUDGET = 64`;
  `MAX_UPLOAD_BYTES = 64 MB`.
- User-facing knobs ONLY: Hough overlay on/off, optional
  `pixel_scale_arcsec` override, optional `hdu_index`. No threshold, model,
  Hough-parameter, stretch, or batch options anywhere.

## Repository layout

```
trail-scope/
  backend/
    main.py                  # THIN: routes, validation, CORS, lifespan, errors
                             # (lift InterPyApp scaffold: log_event,
                             # error_response, upload validation pattern)
    trailscope/
      config.py
      preprocess.py          # format dispatch, tier assignment, provenance (new)
      inference.py           # model loaded once at lifespan; SHA gate
      artifacts.py           # overlay.png, mask.png, input_8bit.png, stats.json
      schemas.py             # pydantic models incl. the stats schema
      vendored/
        VENDOR_MANIFEST.md   # source repo, commit SHA, file, function list, date
        unet.py              # from src/models/unet.py
        loading.py           # load_segmentation_model
        hough_runner.py      # from src/classical/hough_runner
        preprocess_core.py   # INDIVIDUAL FUNCTIONS (not the whole script) from
                             # scripts/figures/decam_cold_inference.py:
                             #   select_image_hdu, header_pixel_scale_arcsec,
                             #   clean_fits_image, downsample_linear_area,
                             #   zscale_to_uint8, reflect_pad_to_multiple,
                             #   iter_stride_tiles, full_image_stats,
                             #   normalise_uint8_patch_array,
                             #   infer_probability_canvas
    tests/
    Dockerfile  requirements.lock  .env.example
  frontend/
    app/page.tsx  lib/api.ts  (+ lifted Next.js shell, tests, Dockerfile)
  scripts/download_demo_assets.py
  docker-compose.yml  .github/workflows/ci.yml  README.md
```

Vendoring rules: no `src.*` imports anywhere (enforced by a test); each
vendored file headed by a provenance comment; the app must read as an app, not
a thesis figure script — `preprocess.py`/`inference.py` wrap the vendored core
behind clean functions.

## Endpoints (complete list — nothing else in v1)

- `GET /health` → `{status, version, model_sha_ok}`
- `GET /model` → static metadata: architecture summary, 485,673 params,
  threshold 0.45, patch size, training-domain statement, thesis-repo link,
  full disclaimer text.
- `POST /infer` — multipart `file` (required), `hough: bool = true`,
  `pixel_scale_arcsec: float | null`, `hdu_index: int | null`. Synchronous
  (`run_in_threadpool`). Returns `200 {result_id, stats}`.
- `GET /results/{id}/input_8bit.png | overlay.png | mask.png | stats.json` —
  `FileResponse` from per-result UUID dir, cleared on lifespan startup.

No `/jobs`, no polling, no training, no threshold options.

## Preprocessing contract (`preprocess.py` around `preprocess_core.py`)

Order matters — **the patch budget is checked AFTER preprocessing**, because
resampling changes the patch count:

1. Upload validation (size cap, extension allowlist
   `{.fits,.fit,.fits.fz,.png,.jpg,.jpeg,.tif}`, UUID storage).
2. Load: FITS via astropy (`hdu_index` or first 2-D image HDU; error listing
   HDUs if ambiguous). PNG/TIFF via PIL preserving bit depth; RGB converted by
   luminance with a provenance flag. Non-finite → image median
   (`clean_fits_image`).
3. Pixel scale: user override → FITS header (`header_pixel_scale_arcsec`) →
   unknown. Known: resample by `scale/0.56` with INTER_AREA
   (`downsample_linear_area`). Unknown: no resample, tier `best_effort`,
   warning "pixel scale unknown — model is not scale-invariant".
4. Stretch: FITS/16-bit → `zscale_to_uint8`; fallback percentile stretch
   (0.5–99.5) when ZScale degenerate; 8-bit input → passthrough. Recorded as
   `stretch` + `stretch_fallback` in provenance; never a UI knob.
5. Pad-size calculation (`reflect_pad_to_multiple`) → compute `n_patches` →
   **if > 64, reject 413**: `"Image too large for this demo: N patches > 64.
   Crop or downsample the image and retry."`
6. Locked path: `full_image_stats` z-score excluding padding, stride-528
   tiling, inference (`infer_probability_canvas`), threshold 0.45, optional
   Hough.
7. Always write `input_8bit.png` — the user must see exactly what the model
   saw. This is the single most important honesty feature.

Tier assignment (neutral, informational): `in_domain_like` (8-bit PNG,
plausible scale) / `recipe_matched` (FITS with header-resolved scale, i.e. the
validated DECam-style recipe) / `best_effort` (everything else). Every tier
shows the constant sentence: "This is not a validated detector for this input
unless it is from the original MeerLICHT-style domain."

## stats.json schema (final names)

```json
{
  "qualitative_only": true, "benchmark": false, "threshold_tuned": false,
  "training_domain": "MeerLICHT 8-bit display PNG patches",
  "tier": "recipe_matched", "warnings": [],
  "provenance": {"format":"fits","hdu":1,"stretch":"zscale","stretch_fallback":false,
                 "pixel_scale_source":"header","pixel_scale_arcsec":0.2634,
                 "resample_factor":0.470,"checkpoint_sha256":"…","threshold":0.45,
                 "vendored_source_commit":"<thesis repo sha>"},
  "image": {"input_shape":[],"processed_shape":[],"n_patches":8},
  "model_output": {"predicted_mask_pixel_count":0,"predicted_mask_fraction":0.0,
                   "predicted_component_count":0,"max_model_probability":0.0,
                   "predicted_components":[{"index":0,"pixel_count":0,"bbox":[],
                                            "major_axis_px":null,"orientation_deg":null}]},
  "hough": {"enabled":true,"segment_count":0,"segments":[]},
  "timing_ms": {"preprocess":0,"inference":0,"hough":0}
}
```

Component stats: `cv2.connectedComponentsWithStats`; ellipse fit only when the
component has ≥5 boundary points — otherwise `major_axis_px: null`,
`orientation_deg: null` (no crash, no fake values). No field may use
"trail"/"detection" wording.

## Frontend (three states, one page)

**Input page (the actual tool, not a landing page):** drag-drop upload;
accepted types + 64 MB max stated; DECam demo picker; optional pixel-scale
override (arcsec/px); optional HDU index; Hough toggle (default on); fixed
model card (Locked thesis U-Net · Threshold 0.45 · Patch 528×528 · Training
domain: MeerLICHT 8-bit display PNG patches); `Run inference` button; caveat
line.

**Processing state (synchronous, honest):** filename, spinner, stage text
("Preparing image" → "Running locked U-Net" → "Rendering outputs"), static
note "Large images may take up to a minute on CPU. Images over 64 patches are
rejected in this demo.", caveat "The model and threshold are fixed; this run
does not tune parameters or estimate accuracy." Rejection shows the 413
message verbatim.

**Output page:** neutral tier banner (same blue/grey style all tiers, text
differs) + preprocessing warnings; main comparison `input_8bit.png` vs
`overlay.png` with opacity slider and Mask/Hough toggles; result summary
(predicted mask pixels, fraction, component count, max model probability,
Hough segment count, processed shape, patch count, runtime); predicted
components table (index, pixels, bbox, major axis, orientation — nulls render
as "—"); provenance panel (format, HDU, stretch, fallback, pixel-scale source,
resample factor, checkpoint SHA, threshold, vendored commit); downloads
(overlay, mask, model-input PNG, stats JSON); disclaimer footer.

Disclaimer placement (all five, non-negotiable): input page, output page,
README, `/model`, `stats.json`.

## Demo assets and data policy

- Repo ships only **small cropped 8-bit PNG examples** derived from the public
  DECam frames (a few hundred KB total), with the NOIRLab acknowledgement in
  README and the demo picker UI.
- `scripts/download_demo_assets.py` fetches the full detector FITS from the
  NOIRLab archive on demand (reuse the retrieval logic from
  `decam_cold_inference.py`); optionally cache processed demo outputs.
- **MeerLICHT: narrow approved exception only.** Originally "never any MeerLICHT imagery".
  Updated 2026-06-16 (owner-authorised; consortium approval in progress): the demo may also
  ship the small thesis Fig. 5.4/5.5 MeerLICHT display-PNG patches (`frontend/public/demo/
  ml1_*.png`) with a MeerLICHT acknowledgement everywhere they appear. No raw/full-frame
  MeerLICHT, no GT masks, nothing beyond those acknowledged example patches.

## Build/dependencies

- `opencv-python-headless` (not `opencv-python`).
- Torch CPU wheels (`--index-url https://download.pytorch.org/whl/cpu`).
- `python:3.11-slim`; checkpoint downloaded + SHA-checked at build AND
  asserted at startup.
- CI (lift InterPyApp `ci.yml`): backend pytest + Ruff; frontend lint + tests;
  docker build job. No GPU anywhere.

## Tests / acceptance criteria

- `test_preprocess.py`: 16-bit synthetic PNG → [0,1] with percentile fallback
  exercised; FITS with `CD1_1` → resample factor reproduces DECam 0.4704 to
  1e-3; unknown scale → `best_effort` + warning; non-finite cleaned; RGB
  luminance-converted with flag; patch budget computed post-resample (a file
  that passes pre-resample but fails post must 413, and vice versa).
- `test_inference_synthetic.py`: drawn 2 px line recovered ≥80 %; empty frame
  `predicted_mask_fraction < 1e-4`; tiny-component ellipse returns nulls.
  (Sanity harness, not a benchmark — say so in docstrings.)
- `test_api.py`: upload validation; sync infer returns schema-valid stats with
  all four honesty fields; 413 path; results 404 before/garbage id; no-`src.*`
  import test; corrupted checkpoint → startup refusal.
- End-to-end: one demo example through compose in < 60 s CPU.

## Phasing (~2 days)

- **P1 (½ d):** scaffold lift, vendoring + manifest, checkpoint fetch + SHA
  gate, `/health`, `/model`.
- **P2 (¾ d):** `preprocess.py` dispatch/tier/budget logic + tests.
- **P3 (½ d):** `/infer` + `artifacts.py` + frontend results page.
- **P4 (¼ d):** compose, CI, demo assets + download script, README.

## v2 candidates (explicitly out of v1)

Async jobs via status files in per-result dirs; promoting the vendored core to
a clean `src/inference/full_image.py` in the thesis repo at a tagged release
(then pip-pin and delete `vendored/`); full-frame support beyond 64 patches;
side-by-side Hough-off/on comparison view.
