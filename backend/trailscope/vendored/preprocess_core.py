# VENDORED — frozen copy, do not edit to "improve". See VENDOR_MANIFEST.md.
# Source: bg492 scripts/figures/decam_cold_inference.py @ commit
#   b9a4e602e8ae570b016b5ed3a08d7bdb11b4055f. Copied 2026-06-14.
# Copied as INDIVIDUAL FUNCTIONS (not the whole figure script):
#   select_image_hdu, header_pixel_scale_arcsec, clean_fits_image,
#   downsample_linear_area, zscale_to_uint8, reflect_pad_to_multiple,
#   iter_stride_tiles, full_image_stats, normalise_uint8_patch_array,
#   infer_probability_canvas
# Function bodies are verbatim. ADAPTATION: the two module-level constants the script
# imported from src (`PATCH_SIZE` from src.config.constants; `LOCKED_NORMALISATION`)
# are re-declared here so this file stands alone with no src.* imports. Values are
# unchanged (PATCH_SIZE=528, LOCKED_NORMALISATION="full_image").
"""Full-image preprocessing + inference primitives for the locked detector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Vendored from bg492 src/config/constants.py (PATCH_SIZE = 528) and the locked
# normalisation mode used by the DECam cold-inference path.
PATCH_SIZE: int = 528
LOCKED_NORMALISATION: str = "full_image"

# DECam reference resample factor (0.2634 / 0.56); used only as the default for
# downsample_linear_area. trail-scope's preprocess.py passes the factor computed
# from the actual input pixel scale, so this default is a documentation anchor.
DECAM_PIXEL_SCALE_ARCSEC = 0.2634
TARGET_ARCSEC_PER_PX = 0.56
RESAMPLE_FACTOR = DECAM_PIXEL_SCALE_ARCSEC / TARGET_ARCSEC_PER_PX


def select_image_hdu(path: str | Path) -> tuple[np.ndarray, Any, int]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        for idx, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and getattr(data, "ndim", 0) == 2:
                return np.asarray(data, dtype=np.float32), hdu.header.copy(), idx
    raise ValueError(f"No 2D image HDU found in {path}")


def header_pixel_scale_arcsec(header: Any) -> float | None:
    if "PIXSCAL" in header:
        try:
            return float(header["PIXSCAL"])
        except Exception:
            return None
    if all(key in header for key in ("CD1_1", "CD2_1", "CD1_2", "CD2_2")):
        x_scale = math.hypot(float(header["CD1_1"]), float(header["CD2_1"])) * 3600.0
        y_scale = math.hypot(float(header["CD1_2"]), float(header["CD2_2"])) * 3600.0
        return float((x_scale + y_scale) / 2.0)
    if "CDELT1" in header and "CDELT2" in header:
        return float((abs(float(header["CDELT1"])) + abs(float(header["CDELT2"]))) * 1800.0)
    return None


def clean_fits_image(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    arr = np.asarray(image, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("FITS image has no finite pixels")
    fill_value = float(np.median(arr[finite]))
    replaced = int((~finite).sum())
    arr[~finite] = fill_value
    return arr, replaced, fill_value


def downsample_linear_area(image: np.ndarray, factor: float = RESAMPLE_FACTOR) -> np.ndarray:
    import cv2

    if factor <= 0:
        raise ValueError("resample factor must be positive")
    h, w = image.shape
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def zscale_to_uint8(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from astropy.visualization import ZScaleInterval

    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Cannot stretch an image with no finite pixels")
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(image[finite])
    fallback = False
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.percentile(image[finite], [0.5, 99.5])
        fallback = True
    if vmax <= vmin:
        vmax = vmin + 1.0
        fallback = True
    scaled = np.clip((image - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8), {
        "name": "zscale",
        "implementation": "astropy.visualization.ZScaleInterval",
        "vmin": float(vmin),
        "vmax": float(vmax),
        "fallback_percentile_used": bool(fallback),
    }


def reflect_pad_to_multiple(
    image: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, dict[str, int], tuple[int, int]]:
    h, w = image.shape
    pad_h = (math.ceil(h / patch_size) * patch_size) - h
    pad_w = (math.ceil(w / patch_size) * patch_size) - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, {"bottom": int(pad_h), "right": int(pad_w)}, (int(h), int(w))


def iter_stride_tiles(image: np.ndarray, patch_size: int = PATCH_SIZE):
    h, w = image.shape
    if h % patch_size or w % patch_size:
        raise ValueError("iter_stride_tiles expects a padded image with patch-size multiple shape")
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            yield y, x, image[y : y + patch_size, x : x + patch_size]


def full_image_stats(image_u8: np.ndarray) -> dict[str, float]:
    image01 = image_u8.astype(np.float32) / 255.0
    return {"mean": float(image01.mean()), "std": float(image01.std())}


def normalise_uint8_patch_array(
    patch_u8: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Apply the locked full-image z-score to a uint8 patch as [1, H, W]."""
    patch = patch_u8.astype(np.float32) / 255.0
    return ((patch - float(mean)) / (float(std) + 1e-6))[None, :, :]


def infer_probability_canvas(
    image_u8: np.ndarray,
    model: Any,
    device: Any,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    stats = full_image_stats(image_u8)
    padded, pad_shape, original_shape = reflect_pad_to_multiple(image_u8)
    prob_padded = np.zeros(padded.shape, dtype=np.float32)
    tiles = list(iter_stride_tiles(padded))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start : start + batch_size]
            batch_np = np.stack([
                normalise_uint8_patch_array(tile, stats["mean"], stats["std"])
                for _, _, tile in batch_tiles
            ])
            batch = torch.from_numpy(batch_np).to(device)
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()
            for (y, x, _), prob in zip(batch_tiles, probs):
                prob_padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE] = prob.astype(np.float32)
    h, w = original_shape
    return prob_padded[:h, :w], {
        "normalisation": LOCKED_NORMALISATION,
        "full_image_mean": stats["mean"],
        "full_image_std": stats["std"],
        "pad_shape": pad_shape,
        "padded_shape": [int(v) for v in padded.shape],
        "n_patches": int(len(tiles)),
    }
