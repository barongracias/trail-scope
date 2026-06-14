# VENDORED — frozen copy, do not edit to "improve". See VENDOR_MANIFEST.md.
# Source: bg492 src/models/loading.py @ commit b9a4e602e8ae570b016b5ed3a08d7bdb11b4055f
# Copied 2026-06-14. ADAPTATION: the original dispatched on config.model_type to
# either UNet or AttentionUNet, importing both from src.models.*. trail-scope ships
# ONLY the locked plain-UNet winner, so the AttentionUNet branch is replaced with an
# explicit error and the import is rewritten to the local vendored .unet. Behaviour
# for the locked checkpoint (model_type "unet") is identical to the source.
"""Single source of truth for loading a trained segmentation checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .unet import UNet


def load_segmentation_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, str]:
    """Load the locked U-Net checkpoint, dispatching on ``config.model_type``.

    Returns the eval-mode model on ``device`` and its normalisation mode. The
    original supported ``attention_unet`` as well; trail-scope vendors only the
    plain ``UNet`` (the locked winner) and rejects any other ``model_type``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model_type", "unet")
    if model_type != "unet":
        raise ValueError(
            f"trail-scope vendors only the plain UNet; checkpoint model_type={model_type!r} "
            "is not supported in this demo."
        )
    model = UNet(base_channels=cfg.get("base_channels", 8)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg.get("normalisation", "fixed")
