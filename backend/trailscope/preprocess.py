"""Preprocessing contract — format dispatch, tier assignment, provenance.

This is the one genuinely hard part (spec §Preprocessing). It wraps the vendored
`preprocess_core` functions behind a single clean entry point, `preprocess_image`,
producing the exact 8-bit array the locked U-Net will see (`image_u8`) plus the neutral
tier, warnings, and provenance that the honesty story depends on.

Order matters (spec): load → resolve pixel scale → resample (INTER_AREA) → stretch →
**patch budget AFTER resample** (reject 413 if > 64) → locked inference path (elsewhere).
The patch budget is checked post-resample because resampling changes the patch count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .vendored import preprocess_core as core

# Extension families (lowercased). `.fits.fz` is handled by the compound check below.
_FITS_SUFFIXES = (".fits.fz", ".fits", ".fit", ".fz")
_DISPLAY_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

_PIXEL_SCALE_UNKNOWN_WARNING = "pixel scale unknown — model is not scale-invariant"


class PreprocessError(Exception):
    """A client-facing preprocessing failure carrying an HTTP status + detail."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class PreprocessResult:
    image_u8: np.ndarray            # exactly what the model sees (the honesty artifact)
    tier: str                       # in_domain_like | recipe_matched | best_effort
    warnings: list[str]
    provenance: dict
    input_shape: tuple[int, int]    # (H, W) as loaded, pre-resample
    processed_shape: tuple[int, int]  # (H, W) fed to the model
    n_patches: int
    preprocess_ms: float = field(default=0.0)
    # "As uploaded" display render of the pre-resample image, downsized for transport.
    # None when it would be identical to image_u8 (8-bit passthrough, no resample).
    original_preview_u8: np.ndarray | None = field(default=None)


def is_fits_filename(filename: str) -> bool:
    name = filename.lower()
    return name.endswith(_FITS_SUFFIXES)


def list_fits_hdus(path: str | Path, filename: str) -> list[dict]:
    """Return [{index, type, shape, is_2d_image}] for a FITS file (for the HDU picker)."""
    if not is_fits_filename(filename):
        raise PreprocessError(400, "HDU inspection is only available for FITS files.")
    from astropy.io import fits

    out: list[dict] = []
    with fits.open(path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            shape = list(data.shape) if data is not None and hasattr(data, "shape") else None
            out.append(
                {
                    "index": idx,
                    "type": type(hdu).__name__,
                    "shape": shape,
                    "is_2d_image": data is not None and getattr(data, "ndim", 0) == 2,
                }
            )
    return out


def _downsize_for_display(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Area-downsample so the longest edge is <= max_edge (no-op if already small)."""
    import cv2

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _hdu_summary(hdul) -> str:
    parts = []
    for idx, hdu in enumerate(hdul):
        shape = getattr(getattr(hdu, "data", None), "shape", None)
        parts.append(f"[{idx}] {type(hdu).__name__} shape={shape}")
    return "; ".join(parts)


def _load_fits(path: Path, hdu_index: int | None):
    """Return (float32 image, header, hdu_index_used). Lists HDUs on error."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        if hdu_index is not None:
            if hdu_index < 0 or hdu_index >= len(hdul):
                raise PreprocessError(
                    400,
                    f"hdu_index {hdu_index} out of range; file has {len(hdul)} HDUs: "
                    f"{_hdu_summary(hdul)}",
                )
            data = hdul[hdu_index].data
            if data is None or getattr(data, "ndim", 0) != 2:
                raise PreprocessError(
                    400,
                    f"HDU {hdu_index} is not a 2-D image. Available HDUs: {_hdu_summary(hdul)}",
                )
            return np.asarray(data, dtype=np.float32), hdul[hdu_index].header.copy(), hdu_index

        for idx, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and getattr(data, "ndim", 0) == 2:
                return np.asarray(data, dtype=np.float32), hdu.header.copy(), idx
        raise PreprocessError(
            400, f"No 2-D image HDU found. Available HDUs: {_hdu_summary(hdul)}"
        )


def _load_display(path: Path, filename: str):
    """Load PNG/JPEG/TIFF via PIL preserving bit depth.

    Returns (array, is_8bit, rgb_to_luminance, format_label). RGB is converted to
    luminance (ITU-R 601) and flagged in provenance.
    """
    from PIL import Image

    ext = Path(filename).suffix.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    elif ext == "tiff":
        ext = "tif"

    with Image.open(path) as img:
        img.load()
        mode = img.mode
        rgb_to_luminance = False
        if mode in ("RGB", "RGBA", "P"):
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            # ITU-R 601 luma; keep 8-bit display depth.
            arr = (
                0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            )
            arr = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
            rgb_to_luminance = True
        else:
            arr = np.asarray(img)

    if arr.ndim != 2:
        raise PreprocessError(400, f"Unsupported image with shape {arr.shape}; expected 2-D.")

    is_8bit = arr.dtype == np.uint8
    return arr, is_8bit, rgb_to_luminance, ext


def preprocess_image(
    path: str | Path,
    *,
    filename: str,
    pixel_scale_arcsec: float | None = None,
    hdu_index: int | None = None,
    max_patch_budget: int | None = None,
) -> PreprocessResult:
    """Run the full preprocessing contract and return the model-ready 8-bit image.

    Raises ``PreprocessError(413, …)`` when the post-resample patch budget exceeds the
    budget (default ``config.MAX_PATCH_BUDGET`` = 64 for the synchronous path; the async
    jobs path passes the higher ``config.MAX_JOB_PATCH_BUDGET`` ceiling).
    """
    budget = config.MAX_PATCH_BUDGET if max_patch_budget is None else max_patch_budget
    t0 = time.perf_counter()
    path = Path(path)
    warnings: list[str] = []

    if pixel_scale_arcsec is not None and pixel_scale_arcsec <= 0:
        raise PreprocessError(400, "pixel_scale_arcsec must be positive.")

    # 1) Load + clean (non-finite → image median for scientific data).
    if is_fits_filename(filename):
        raw, header, used_hdu = _load_fits(path, hdu_index)
        work, nonfinite, _fill = core.clean_fits_image(raw)
        is_8bit_display = False
        rgb_to_luminance = False
        header_scale = core.header_pixel_scale_arcsec(header)
        fmt_label = "fits"
    else:
        if hdu_index is not None:
            warnings.append("hdu_index ignored for non-FITS input")
        arr, is_8bit_display, rgb_to_luminance, fmt_label = _load_display(path, filename)
        used_hdu = None
        header_scale = None
        if is_8bit_display:
            work, nonfinite = arr, 0
        else:
            work, nonfinite, _fill = core.clean_fits_image(arr)

    input_shape = (int(work.shape[0]), int(work.shape[1]))
    pre_resample = work  # keep a reference for the "as uploaded" preview

    # 2) Resolve pixel scale: user override → FITS header → unknown.
    if pixel_scale_arcsec is not None:
        scale: float | None = float(pixel_scale_arcsec)
        scale_source = "user_override"
    elif header_scale is not None:
        scale = float(header_scale)
        scale_source = "header"
    else:
        scale = None
        scale_source = "unknown"

    # 3) Resample by scale / 0.56 (INTER_AREA) when the scale is known.
    if scale is not None:
        resample_factor = scale / config.TARGET_ARCSEC_PER_PX
        work = core.downsample_linear_area(work, factor=resample_factor)
        resampled = True
    else:
        resample_factor = 1.0
        resampled = False
        # Only scientific inputs need the scale warning; an 8-bit display PNG is
        # assumed to already be in the display domain (plausible scale).
        if not is_8bit_display:
            warnings.append(_PIXEL_SCALE_UNKNOWN_WARNING)

    # 4) Stretch: 8-bit display passthrough; otherwise ZScale (percentile fallback).
    if is_8bit_display:
        image_u8 = np.ascontiguousarray(work, dtype=np.uint8)
        stretch_name = "passthrough"
        stretch_fallback = False
    else:
        image_u8, stretch_meta = core.zscale_to_uint8(work)
        stretch_name = "zscale"
        stretch_fallback = bool(stretch_meta["fallback_percentile_used"])

    processed_shape = (int(image_u8.shape[0]), int(image_u8.shape[1]))

    # 5) Patch budget AFTER resample — reject 413 over 64.
    padded, _pad_shape, _orig = core.reflect_pad_to_multiple(image_u8)
    n_patches = (padded.shape[0] // config.PATCH_SIZE) * (padded.shape[1] // config.PATCH_SIZE)
    if n_patches > budget:
        raise PreprocessError(
            413,
            f"Image too large for this demo: {n_patches} patches > {budget}. "
            "Crop or downsample the image and retry.",
        )

    # Tier assignment (neutral, informational).
    if scale_source == "unknown" and not is_8bit_display:
        tier = "best_effort"
    elif fmt_label == "fits":
        tier = "recipe_matched"
    elif is_8bit_display:
        tier = "in_domain_like"
    else:
        tier = "best_effort"

    # "As uploaded" preview (pre-resample, downsized) — only meaningful when resampling
    # changed the geometry; otherwise it would duplicate input_8bit.
    original_preview_u8: np.ndarray | None = None
    if resampled:
        prev = _downsize_for_display(pre_resample, config.PREVIEW_MAX_EDGE)
        if is_8bit_display:
            original_preview_u8 = np.ascontiguousarray(prev, dtype=np.uint8)
        else:
            original_preview_u8, _ = core.zscale_to_uint8(prev)

    provenance = {
        "format": fmt_label,
        "hdu": used_hdu,
        "stretch": stretch_name,
        "stretch_fallback": stretch_fallback,
        "pixel_scale_source": scale_source,
        "pixel_scale_arcsec": scale,
        "resample_factor": float(resample_factor),
        "rgb_to_luminance": rgb_to_luminance,
        "nonfinite_pixels_cleaned": int(nonfinite),
        "checkpoint_sha256": config.EXPECTED_CHECKPOINT_SHA256,
        "threshold": config.THRESHOLD,
        "vendored_source_commit": config.VENDORED_SOURCE_COMMIT,
    }

    return PreprocessResult(
        image_u8=image_u8,
        tier=tier,
        warnings=warnings,
        provenance=provenance,
        input_shape=input_shape,
        processed_shape=processed_shape,
        n_patches=int(n_patches),
        preprocess_ms=(time.perf_counter() - t0) * 1000.0,
        original_preview_u8=original_preview_u8,
    )
