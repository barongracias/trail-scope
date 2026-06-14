"""Pydantic response models.

The full `stats.json` schema (`InferStats`) is added in P3 alongside `/infer`; P1 ships
only the `/health` and `/model` models.
"""

from __future__ import annotations

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
