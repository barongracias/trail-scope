"""Pydantic response models, including the full `stats.json` schema."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    model_sha_ok: bool


class ModelResponse(BaseModel):
    """Static metadata for `GET /model` (CLAUDE.md rule 6: disclaimer must appear here)."""

    architecture: str
    param_count: int
    threshold: float
    patch_size: int
    normalisation: str
    training_domain: str
    tier_definitions: dict[str, str]
    thesis_repo: str
    checkpoint_sha256: str
    scope: str
    disclaimer: str
    user_facing_knobs: list[str]


# --------------------------------------------------------------------------------------
# stats.json schema (spec §stats.json). Vocabulary is "predicted mask/component" — never
# "trail"/"detection". Null axis/orientation render as "—" client-side (no fake values).
# --------------------------------------------------------------------------------------
class Provenance(BaseModel):
    format: str
    hdu: Optional[int]
    stretch: str
    stretch_fallback: bool
    pixel_scale_source: str
    pixel_scale_arcsec: Optional[float]
    resample_factor: float
    rgb_to_luminance: bool
    nonfinite_pixels_cleaned: int
    checkpoint_sha256: str
    threshold: float
    vendored_source_commit: str


class ImageStats(BaseModel):
    input_shape: list[int]
    processed_shape: list[int]
    n_patches: int


class PredictedComponent(BaseModel):
    index: int
    pixel_count: int
    bbox: list[int]                       # [x, y, w, h]
    major_axis_px: Optional[float]
    orientation_deg: Optional[float]


class ModelOutput(BaseModel):
    predicted_mask_pixel_count: int
    predicted_mask_fraction: float
    predicted_component_count: int
    max_model_probability: float
    predicted_components: list[PredictedComponent]


class HoughStats(BaseModel):
    enabled: bool
    segment_count: int
    segments: list[list[int]]


class TimingStats(BaseModel):
    preprocess: float
    inference: float
    hough: float


class InferStats(BaseModel):
    qualitative_only: bool
    benchmark: bool
    threshold_tuned: bool
    training_domain: str
    scope: str
    disclaimer: str
    tier: str
    warnings: list[str]
    provenance: Provenance
    image: ImageStats
    model_output: ModelOutput
    hough: HoughStats
    timing_ms: TimingStats


class InferResponse(BaseModel):
    result_id: str
    stats: InferStats
