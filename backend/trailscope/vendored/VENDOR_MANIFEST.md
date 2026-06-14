# Vendored inference core — provenance manifest

These files are a **frozen copy** of the minimal inference core from the MPhil thesis
repository (`bg492`). They are vendored, not imported: there are **no `src.*` imports**
anywhere in this package (enforced by `backend/tests/test_no_src_imports.py`). Each file
carries a provenance header; this manifest is the authoritative record.

- **Source repository:** `bg492` (the thesis submission repo; checkpoint and these files
  are public per the thesis code-availability statement).
- **Source commit:** `b9a4e602e8ae570b016b5ed3a08d7bdb11b4055f`
- **Copied on:** 2026-06-14

| Vendored file | Source path (in bg492) | Contents copied | Adaptation |
|---|---|---|---|
| `unet.py` | `src/models/unet.py` | Whole module: `DROPOUT_RATES`, `DoubleConv`, `DownBlock`, `UpBlock`, `UNet` | Verbatim (provenance header only). 485,673 trainable params at `base_channels=8`. |
| `loading.py` | `src/models/loading.py` | `load_segmentation_model` | Import rewritten from `src.models.unet` → local `.unet`; the `attention_unet` dispatch branch replaced with an explicit error (trail-scope ships only the plain locked UNet). Behaviour for the locked `model_type="unet"` checkpoint is identical. |
| `hough_runner.py` | `src/classical/hough_runner.py` | Whole module: `_apply_hough`, `HoughCanvasResult`, `run_hough_on_canvas` | Verbatim (provenance header only). trail-scope calls `_apply_hough` for the optional overlay; the GT-overlap `run_hough_on_canvas` is unused (the demo has no ground truth). |
| `preprocess_core.py` | `scripts/figures/decam_cold_inference.py` | Individual functions only: `select_image_hdu`, `header_pixel_scale_arcsec`, `clean_fits_image`, `downsample_linear_area`, `zscale_to_uint8`, `reflect_pad_to_multiple`, `iter_stride_tiles`, `full_image_stats`, `normalise_uint8_patch_array`, `infer_probability_canvas` | Function bodies verbatim. The two constants the script imported from `src` (`PATCH_SIZE` from `src/config/constants.py`; `LOCKED_NORMALISATION`) are re-declared locally with unchanged values (528, `"full_image"`). The figure/montage/manifest/retrieval code was NOT copied. |

## Locked checkpoint

- Vendored at `backend/checkpoints/model-best.pth` (copied from
  `bg492/results/checkpoints/model-best.pth`).
- **SHA-256:** `ff680804f6cf66d6948dcd76af4958c4427099ecdb45bab0140ac80314b8e55b`
  (5,894,411 bytes). Asserted at startup by `config.py` / `inference.py`; the app
  refuses to serve on mismatch.
- This is the locked thesis winner `unet_paper_arch_noise_topk_t44_s2804`.

## v2 note

The clean exit from vendoring is to promote this core to a tagged `src/inference/`
module in the thesis repo, then pip-pin it and delete this directory. Out of scope for v1.
