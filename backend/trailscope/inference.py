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


def _maybe_torchscript(model: Any, device: Any) -> tuple[Any, str]:
    """Optionally TorchScript-trace the model for a CPU speedup.

    The traced module is **verified equivalent** to the eager model on a random patch
    before use (max abs diff < 1e-4); on any exception or mismatch we fall back to the
    eager model. The locked U-Net has no input-dependent control flow at the 528×528 patch
    size (the UpBlock interpolate branch never fires), so tracing is faithful. The
    checkpoint SHA gate is unaffected — it guards the source weights, not this in-memory
    derivation.
    """
    import logging
    import warnings

    import torch

    log = logging.getLogger("trailscope")
    if not config.USE_TORCHSCRIPT:
        return model, "eager"
    try:
        example = torch.zeros(1, 1, config.PATCH_SIZE, config.PATCH_SIZE)
        with torch.no_grad(), warnings.catch_warnings():
            # The UpBlock shape check is constant-false at the fixed 528×528 patch size,
            # so the trace bakes it out; equivalence is verified explicitly below.
            warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
            traced = torch.jit.trace(model, example)
            traced = torch.jit.optimize_for_inference(traced)
            probe = torch.randn(1, 1, config.PATCH_SIZE, config.PATCH_SIZE)
            max_diff = float((model(probe) - traced(probe)).abs().max())
        if max_diff < 1e-4:
            log.info("TorchScript trace verified (max abs diff %.2e); using torchscript.", max_diff)
            return traced, "torchscript"
        log.warning("TorchScript output diverged (max abs diff %.2e); using eager.", max_diff)
    except Exception as exc:  # pragma: no cover - environment-dependent
        log.warning("TorchScript trace failed (%s); using eager.", exc)
    return model, "eager"


class ModelService:
    """Holds the single locked U-Net instance and runs the locked inference path."""

    def __init__(
        self, model: Any, device: Any, *, param_count: int | None = None, backend: str = "eager"
    ) -> None:
        self.model = model
        self.device = device
        # Param count must come from the eager weights; a traced module reports 0.
        self.param_count = param_count if param_count is not None else _count_trainable_params(model)
        self.checkpoint_sha256 = config.EXPECTED_CHECKPOINT_SHA256
        self.backend = backend  # "eager" | "torchscript"

    @classmethod
    def load(cls) -> "ModelService":
        """Run the SHA gate, load the U-Net once, and assert the param count."""
        import os

        import torch

        # Use all CPU cores for the locked inference path (no GPU anywhere).
        torch.set_num_threads(max(1, os.cpu_count() or 1))

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

        run_model, backend = _maybe_torchscript(model, device)
        return cls(run_model, device, param_count=params, backend=backend)

    # -- locked inference primitives (used by preprocess.py / artifacts.py) -----------

    def infer_probability_canvas(
        self, image_u8: np.ndarray, batch_size: int = 16
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run the locked stride-528 full-image U-Net path, returning [0,1] probs."""
        return preprocess_core.infer_probability_canvas(
            image_u8, self.model, self.device, batch_size=batch_size
        )

    def hough_overlay(self, prob_canvas: np.ndarray) -> np.ndarray:
        """Draw the optional locked-parameter Hough overlay; returns a boolean mask.

        Uses the vendored `_apply_hough` verbatim — this is the path validated against
        the thesis DECam `hough_pixel_count`.
        """
        hough_input = (prob_canvas >= config.HOUGH_INPUT_THRESHOLD).astype(np.uint8) * 255
        drawn = _apply_hough(
            hough_input,
            hough_threshold=config.HOUGH_THRESHOLD,
            min_line_length=config.HOUGH_MIN_LINE_LENGTH,
            max_line_gap=config.HOUGH_MAX_LINE_GAP,
            line_thickness=config.HOUGH_LINE_THICKNESS,
        )
        return drawn > 0

    def hough_segments(self, prob_canvas: np.ndarray) -> list[list[int]]:
        """Enumerate Hough line segments [x1,y1,x2,y2] for stats.

        Mirrors `_apply_hough`'s exact `cv2.HoughLinesP` call (locked params), but
        returns the segment list rather than a drawn canvas.
        """
        import cv2

        hough_input = (prob_canvas >= config.HOUGH_INPUT_THRESHOLD).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(
            hough_input,
            rho=1,
            theta=np.pi / 180.0,
            threshold=config.HOUGH_THRESHOLD,
            minLineLength=config.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=config.HOUGH_MAX_LINE_GAP,
        )
        if lines is None:
            return []
        return [[int(a), int(b), int(c), int(d)] for a, b, c, d in lines[:, 0]]
