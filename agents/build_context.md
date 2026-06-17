# Build context — concrete facts for trail-scope

Companion to `agents/spec.md` (the frozen spec) and `../CLAUDE.md` (the rules). Everything
here is read from the sibling thesis repo `../bg492` (read-only — never modify it).

## Locked checkpoint

- Source (read-only): `../bg492/results/checkpoints/model-best.pth`
- This **is** the locked thesis winner (`unet_paper_arch_noise_topk_t44_s2804`); it is
  public per the thesis code-availability statement, so it may be copied into this repo.
- **Size:** 5,894,411 bytes.
- **SHA-256 (bake into `config.py` as `EXPECTED_CHECKPOINT_SHA256`, assert at startup):**
  `ff680804f6cf66d6948dcd76af4958c4427099ecdb45bab0140ac80314b8e55b`
- Recommended handling: copy it into the repo (or a `scripts/fetch_checkpoint.py`) and
  verify the SHA at build *and* at startup; refuse to serve on mismatch.

## Files to vendor (copy into `backend/trailscope/vendored/`, stamp provenance)

Copy these from `../bg492` (do not import them); adapt only as needed to stand alone.

| Purpose | Source (read-only) |
|---|---|
| U-Net architecture (`UNet`, `base_channels=8`) | `../bg492/src/models/unet.py` |
| Checkpoint loader (`load_segmentation_model`) | `../bg492/src/models/loading.py` |
| Hough runner | `../bg492/src/classical/hough_runner.py` (see also `hough.py`, `run_hough.py`) |
| Full-image preprocessing/inference functions | `../bg492/scripts/figures/decam_cold_inference.py` — copy the individual functions, NOT the whole script: `select_image_hdu`, `header_pixel_scale_arcsec`, `clean_fits_image`, `downsample_linear_area`, `zscale_to_uint8`, `reflect_pad_to_multiple`, `iter_stride_tiles`, `full_image_stats`, `normalise_uint8_patch_array`, `infer_probability_canvas` |

`VENDOR_MANIFEST.md` should record, per file: source path, the `../bg492` git commit it
was copied at, the date, and (for `decam_cold_inference.py`) the exact function list.

## Architecture constants (verified against the code, for self-checks)

- Encoder widths `8 → 16 → 32 → 64`, 128-channel bottleneck; AvgPool down,
  transposed-conv up, skip-concat.
- `DoubleConv` = Conv3×3 → LeakyReLU(0.3) → Dropout → Conv3×3 → LeakyReLU(0.3); no BatchNorm.
- Dropout schedule stem→final: `(0.1, 0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1)`.
- Init: He-normal (kaiming) for 3×3 convs; Glorot/xavier-uniform for transposed convs and
  the final 1×1 head; biases zero.
- **485,673 trainable params** at `base_channels=8` — assert this after instantiation as a
  vendoring sanity check.
- Output is logits; sigmoid applied explicitly in the threshold/Hough paths.

## Locked inference recipe (matches the thesis DECam path)

1. Load (FITS via astropy first 2-D image HDU or `hdu_index`; PNG/TIFF via PIL preserving
   bit depth). Clean non-finite → image median.
2. Pixel scale: user override → FITS header → unknown. If known, resample by
   `scale / 0.56` with `cv2.INTER_AREA` (DECam factor `0.2634/0.56 = 0.470357…`). If
   unknown, no resample → tier `best_effort` + warning.
3. Stretch: ZScale → 8-bit (percentile 0.5–99.5 fallback when ZScale degenerate); 8-bit
   input passthrough.
4. Compute patch budget AFTER resample/pad; reject (413) if > 64 patches.
5. `full_image` z-score on the final 8-bit image excluding reflect padding; stride-528
   tiling; U-Net; threshold **0.45**; optional Hough (`0.1/50/100/250/3`).
6. Always emit `input_8bit.png` (what the model saw).

## Demo data (public DECam + narrow MeerLICHT exception)

- Manifest (read-only): `../bg492/results/classical/decam_cold_manifest.json` — nine
  predeclared NOIRLab DECam measured-streak frames (expnum/detector/object).
- Ship small **cropped 8-bit PNG** examples in-repo, and/or a
  `scripts/download_demo_assets.py` that fetches the full FITS from the NOIRLab archive on
  demand (retrieval logic in `decam_cold_inference.py`).
- **MeerLICHT exception (2026-06-16, owner-authorised; consortium approval in progress):**
  the small thesis Fig. 5.4/5.5 MeerLICHT display-PNG patches (`frontend/public/demo/
  ml1_*.png`) may be shipped with a MeerLICHT acknowledgement. Do **not** commit any other
  MeerLICHT data (raw/full-frame imagery, GT masks, `data/patches` content).
- Raw DECam linear FITS are downsampled before stretch by `0.470357…` (area-averaging,
  not flux-preserving); one fixed ZScale stretch for display.

## Environment

- Python 3.11; CPU torch wheels (`--index-url https://download.pytorch.org/whl/cpu`);
  `opencv-python-headless`; `astropy` for FITS; `pillow`, `numpy`, `scipy`.
- A full 10,560² MeerLICHT frame is 400 patches (rejected by the 64-patch budget in v1);
  a resampled DECam detector is ~8 patches and runs in seconds on CPU.
