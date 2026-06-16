"""Run the locked inference path and render the output artifacts.

Given a `PreprocessResult` and the loaded `ModelService`, this produces:
  - `input_8bit.png` — exactly what the model saw (the single most important honesty
    feature; spec step 7),
  - `mask.png` — the binary predicted mask at threshold 0.45,
  - `overlay.png` — input with the predicted mask (and optional Hough) composited,
  - `stats.json` — the full stats schema.

Vocabulary is "predicted mask/component", never "trail"/"detection" (CLAUDE.md rule 6).
This is qualitative inference only; no benchmark numbers are produced.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .inference import ModelService
from .preprocess import PreprocessResult

# Overlay colours (RGB): predicted mask in pink, Hough overlay in cyan — matching the
# thesis montage palette.
_MASK_RGB = (255, 47, 146)
_HOUGH_RGB = (0, 200, 255)


def _component_stats(
    binary: np.ndarray, prob_canvas: np.ndarray
) -> tuple[int, list[dict[str, Any]]]:
    """Connected-component stats with an ellipse fit only when ≥5 boundary points.

    Returns (component_count, components) sorted by pixel_count descending. Components
    that cannot be ellipse-fit get `major_axis_px=None`, `orientation_deg=None` — no
    crash, no fabricated values.
    """
    import cv2

    mask_u8 = (binary > 0).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    components: list[dict[str, Any]] = []
    for label in range(1, num_labels):  # 0 is background
        x, y, w, h, area = (int(v) for v in stats[label])
        major_axis_px: float | None = None
        orientation_deg: float | None = None

        contours, _ = cv2.findContours(
            (labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        points = np.concatenate(contours, axis=0) if contours else np.empty((0, 1, 2))
        if len(points) >= 5:
            (_cx, _cy), (axis_a, axis_b), angle = cv2.fitEllipse(points)
            major_axis_px = float(max(axis_a, axis_b))
            orientation_deg = float(angle)

        # Model confidence within this component's pixels (sampled from the prob canvas).
        comp_probs = prob_canvas[labels == label]
        mean_probability = float(comp_probs.mean()) if comp_probs.size else 0.0
        max_probability = float(comp_probs.max()) if comp_probs.size else 0.0

        components.append(
            {
                "pixel_count": area,
                "bbox": [x, y, w, h],
                "major_axis_px": major_axis_px,
                "orientation_deg": orientation_deg,
                "mean_probability": mean_probability,
                "max_probability": max_probability,
            }
        )

    components.sort(key=lambda c: c["pixel_count"], reverse=True)
    for idx, comp in enumerate(components):
        comp["index"] = idx
    # Reorder keys so `index` leads (cosmetic; schema is order-insensitive).
    ordered = [
        {
            "index": c["index"],
            "pixel_count": c["pixel_count"],
            "bbox": c["bbox"],
            "major_axis_px": c["major_axis_px"],
            "orientation_deg": c["orientation_deg"],
            "mean_probability": c["mean_probability"],
            "max_probability": c["max_probability"],
        }
        for c in components
    ]
    return num_labels - 1, ordered


def _write_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def _build_overlay(image_u8: np.ndarray, binary: np.ndarray, hough: np.ndarray | None) -> np.ndarray:
    # Draw the classical Hough aid first, then the model's predicted mask on top, so the
    # primary model output stays visible where the two overlap. Kept consistent with the
    # interactive canvas (CanvasCompare.tsx).
    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1).astype(np.uint8)
    if hough is not None:
        rgb[hough > 0] = _HOUGH_RGB
    rgb[binary > 0] = _MASK_RGB
    return rgb


def run_inference(
    pre: PreprocessResult,
    service: ModelService,
    *,
    hough: bool,
    out_dir: str | Path,
    on_stage: Any = None,
) -> dict[str, Any]:
    """Execute inference + rendering for one image, writing artifacts into `out_dir`.

    `on_stage("rendering")` (optional) is called after inference, before artifacts are
    written, so the async jobs path can report the render phase. (The "inferring"
    transition is owned by the caller, which knows the patch count.) Returns the stats dict.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_u8 = pre.image_u8

    t0 = time.perf_counter()
    # inference_mode is a touch faster than the vendored function's inner no_grad and
    # nests harmlessly; we wrap the call site rather than edit the frozen vendored core.
    with torch.inference_mode():
        prob_canvas, _inf_meta = service.infer_probability_canvas(image_u8)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    if on_stage:
        on_stage("rendering")
    binary = prob_canvas >= config.THRESHOLD
    component_count, components = _component_stats(binary, prob_canvas)

    hough_ms = 0.0
    hough_mask: np.ndarray | None = None
    segments: list[list[int]] = []
    if hough:
        th0 = time.perf_counter()
        hough_mask = service.hough_overlay(prob_canvas)
        segments = service.hough_segments(prob_canvas)
        hough_ms = (time.perf_counter() - th0) * 1000.0

    # Artifacts (always write input_8bit.png — spec step 7).
    _write_png(out_dir / "input_8bit.png", image_u8)
    _write_png(out_dir / "mask.png", (binary.astype(np.uint8) * 255))
    _write_png(out_dir / "overlay.png", _build_overlay(image_u8, binary, hough_mask))
    # Grayscale model-confidence map (prob*255); the frontend colourmaps it for display
    # and samples the raw value for the cursor readout. Labelled "confidence", not a knob.
    _write_png(out_dir / "prob.png", np.clip(prob_canvas * 255.0 + 0.5, 0, 255).astype(np.uint8))

    artifact_files = ["input_8bit.png", "mask.png", "overlay.png", "prob.png", "stats.json"]
    if pre.original_preview_u8 is not None:
        _write_png(out_dir / "original_preview.png", pre.original_preview_u8)
        artifact_files.insert(0, "original_preview.png")

    stats = {
        "qualitative_only": True,
        "benchmark": False,
        "threshold_tuned": False,
        "training_domain": config.TRAINING_DOMAIN,
        "scope": config.SCOPE_SENTENCE,
        "disclaimer": config.DISCLAIMER,
        "tier": pre.tier,
        "warnings": pre.warnings,
        "provenance": pre.provenance,
        "artifacts": artifact_files,
        "image": {
            "input_shape": list(pre.input_shape),
            "processed_shape": list(pre.processed_shape),
            "n_patches": pre.n_patches,
        },
        "model_output": {
            "predicted_mask_pixel_count": int(binary.sum()),
            "predicted_mask_fraction": float(binary.mean()),
            "predicted_component_count": int(component_count),
            "max_model_probability": float(prob_canvas.max()),
            "predicted_components": components,
        },
        "hough": {
            "enabled": bool(hough),
            "segment_count": len(segments),
            "segments": segments,
        },
        "timing_ms": {
            "preprocess": round(pre.preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "hough": round(hough_ms, 2),
        },
    }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
