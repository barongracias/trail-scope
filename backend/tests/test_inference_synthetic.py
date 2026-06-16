"""Synthetic inference sanity checks (spec §Tests test_inference_synthetic.py).

This is a SANITY HARNESS, not a benchmark: it confirms the locked pipeline fires on an
obvious bright line, stays quiet on an empty frame, and that component stats degrade
gracefully — it makes no precision/recall or accuracy claim about the detector.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from trailscope import artifacts, config
from trailscope.inference import ModelService


@pytest.fixture(scope="module")
def service() -> ModelService:
    return ModelService.load()


def test_synthetic_2px_line_recovered(service: ModelService) -> None:
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(40, 5, (528, 528)), 0, 255).astype(np.uint8)
    line = np.zeros((528, 528), np.uint8)
    cv2.line(line, (60, 80), (470, 440), 255, 2)
    img[line > 0] = 240

    prob, _ = service.infer_probability_canvas(img)
    binary = prob >= config.THRESHOLD
    gt = line > 0
    recovered = float((binary & gt).sum() / gt.sum())
    assert recovered >= 0.80, f"recovered only {recovered:.2%} of the drawn line"


def test_empty_frame_predicts_almost_nothing(service: ModelService) -> None:
    rng = np.random.default_rng(1)
    empty = np.clip(rng.normal(40, 5, (528, 528)), 0, 255).astype(np.uint8)
    prob, _ = service.infer_probability_canvas(empty)
    fraction = float((prob >= config.THRESHOLD).mean())
    assert fraction < 1e-4, f"empty frame predicted fraction {fraction}"


def test_tiny_component_returns_null_ellipse() -> None:
    # A 3-pixel blob has < 5 boundary points → no ellipse fit (nulls, not fake values).
    binary = np.zeros((100, 100), dtype=bool)
    binary[10, 10:13] = True
    prob = binary.astype(np.float32) * 0.6
    count, comps = artifacts._component_stats(binary, prob)
    assert count == 1
    assert comps[0]["major_axis_px"] is None
    assert comps[0]["orientation_deg"] is None
    # Confidence within the component reflects the prob canvas there.
    assert abs(comps[0]["mean_probability"] - 0.6) < 1e-5
    assert abs(comps[0]["max_probability"] - 0.6) < 1e-5


def test_elongated_component_fits_ellipse() -> None:
    binary = np.zeros((200, 200), dtype=bool)
    cv2.line(
        (img := np.zeros((200, 200), np.uint8)), (20, 20), (180, 160), 255, 3
    )
    binary |= img > 0
    prob = binary.astype(np.float32) * 0.9
    count, comps = artifacts._component_stats(binary, prob)
    assert count == 1
    assert comps[0]["major_axis_px"] is not None
    assert comps[0]["orientation_deg"] is not None
    assert comps[0]["major_axis_px"] > 100  # roughly the line length
    assert comps[0]["max_probability"] > 0.0
