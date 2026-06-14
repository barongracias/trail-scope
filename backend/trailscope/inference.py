"""Model lifecycle + locked inference path.

The checkpoint SHA-256 gate lives here and at `config.checkpoint_sha_ok`: the app
refuses to construct a `ModelService` if the on-disk checkpoint does not match
`EXPECTED_CHECKPOINT_SHA256` (CLAUDE.md hard rule 4). The U-Net is loaded exactly
once, at lifespan startup, and the 485,673-param sanity check (build_context.md) runs
immediately after instantiation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from .vendored import preprocess_core
from .vendored.hough_runner import _apply_hough
from .vendored.loading import load_segmentation_model


class CheckpointError(RuntimeError):
    """Raised when the checkpoint is missing, corrupt, or SHA-mismatched."""


def _count_trainable_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class ModelService:
    """Holds the single locked U-Net instance and runs the locked inference path."""

    def __init__(self, model: Any, device: Any) -> None:
        self.model = model
        self.device = device
        self.param_count = _count_trainable_params(model)
        self.checkpoint_sha256 = config.EXPECTED_CHECKPOINT_SHA256

    @classmethod
    def load(cls) -> "ModelService":
        """Run the SHA gate, load the U-Net once, and assert the param count."""
        import torch

        if not config.CHECKPOINT_PATH.exists():
            raise CheckpointError(f"Checkpoint not found at {config.CHECKPOINT_PATH}")
        actual_sha = config.sha256_file(config.CHECKPOINT_PATH)
        if actual_sha != config.EXPECTED_CHECKPOINT_SHA256:
            raise CheckpointError(
                "Checkpoint SHA-256 mismatch — refusing to serve. "
                f"expected={config.EXPECTED_CHECKPOINT_SHA256} actual={actual_sha}"
            )

        device = torch.device("cpu")
        model, normalisation = load_segmentation_model(config.CHECKPOINT_PATH, device)

        params = _count_trainable_params(model)
        if params != config.EXPECTED_PARAM_COUNT:
            raise CheckpointError(
                f"Vendored U-Net has {params} trainable params; "
                f"expected {config.EXPECTED_PARAM_COUNT}. Vendoring is out of sync."
            )
        if normalisation != config.NORMALISATION:
            raise CheckpointError(
                f"Locked checkpoint must use normalisation={config.NORMALISATION!r}; "
                f"got {normalisation!r}."
            )
        return cls(model, device)

    # -- locked inference primitives (used by preprocess.py / artifacts.py) -----------

    def infer_probability_canvas(
        self, image_u8: np.ndarray, batch_size: int = 16
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the locked stride-528 full-image U-Net path, returning [0,1] probs."""
        return preprocess_core.infer_probability_canvas(
            image_u8, self.model, self.device, batch_size=batch_size
        )

    def hough_overlay(self, prob_canvas: np.ndarray) -> np.ndarray:
        """Draw the optional locked-parameter Hough overlay; returns a boolean mask."""
        hough_input = (prob_canvas >= config.HOUGH_INPUT_THRESHOLD).astype(np.uint8) * 255
        drawn = _apply_hough(
            hough_input,
            hough_threshold=config.HOUGH_THRESHOLD,
            min_line_length=config.HOUGH_MIN_LINE_LENGTH,
            max_line_gap=config.HOUGH_MAX_LINE_GAP,
            line_thickness=config.HOUGH_LINE_THICKNESS,
        )
        return drawn > 0
