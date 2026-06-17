"""Locked configuration for trail-scope.

Everything in the "Locked invariants" block is a CONSTANT, never an API parameter
(see CLAUDE.md hard rule 4). The only user-facing knobs are Hough on/off, an optional
pixel-scale override, and an optional FITS HDU index — all handled in the request
layer, none of them defined here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Version / scope
# --------------------------------------------------------------------------------------
VERSION = "1.0.0"

# Verbatim scope sentence — must match README, /model, and stats.json (CLAUDE.md rule 6).
SCOPE_SENTENCE = (
    "trail-scope is a qualitative, single-image inference demo of the locked thesis "
    "satellite-trail detector. Not a benchmark, not a validated cross-domain tool, no "
    "training, no tunable thresholds. Qualitative inference only; no performance claims "
    "are made for uploaded images."
)

# Constant tier sentence shown for EVERY tier (CLAUDE.md rule 6 / spec tier block).
DISCLAIMER = (
    "This is not a validated detector for this input unless it is from the original "
    "MeerLICHT-style domain."
)

TRAINING_DOMAIN = "MeerLICHT 8-bit display PNG patches"
THESIS_REPO_LINK = "https://github.com/barongracias/satellite_trail_detection"

# --------------------------------------------------------------------------------------
# Locked invariants (constants, never API parameters)
# --------------------------------------------------------------------------------------
# Locked thesis winner checkpoint: unet_paper_arch_noise_topk_t44_s2804.
EXPECTED_CHECKPOINT_SHA256 = (
    "ff680804f6cf66d6948dcd76af4958c4427099ecdb45bab0140ac80314b8e55b"
)
EXPECTED_CHECKPOINT_BYTES = 5_894_411

# The bg492 commit the vendored inference core was copied at (VENDOR_MANIFEST.md);
# surfaced in stats.json provenance.
VENDORED_SOURCE_COMMIT = "b9a4e602e8ae570b016b5ed3a08d7bdb11b4055f"

THRESHOLD = 0.45
NORMALISATION = "full_image"
PATCH_SIZE = 528
TARGET_ARCSEC_PER_PX = 0.56
MAX_PATCH_BUDGET = 64
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB

# Locked Hough parameters: (input_threshold, votes, min_len, max_gap, draw_px).
HOUGH_INPUT_THRESHOLD = 0.1
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 250
HOUGH_LINE_THICKNESS = 3

# U-Net sanity constant — asserted after instantiation (build_context.md).
EXPECTED_PARAM_COUNT = 485_673

# Upload contract.
ALLOWED_EXTENSIONS = {".fits", ".fit", ".fits.fz", ".png", ".jpg", ".jpeg", ".tif"}

# Architecture summary for /model (static metadata).
ARCHITECTURE_SUMMARY = (
    "U-Net, base_channels=8, encoder widths 8→16→32→64, 128-channel bottleneck; "
    "AvgPool down, transposed-conv up, skip-concat; DoubleConv = Conv3×3 → "
    "LeakyReLU(0.3) → Dropout → Conv3×3 → LeakyReLU(0.3); no BatchNorm. Outputs logits."
)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/

CHECKPOINT_PATH = Path(
    os.getenv("TRAILSCOPE_CHECKPOINT", str(_BACKEND_DIR / "checkpoints" / "model-best.pth"))
)
UPLOAD_DIR = Path(os.getenv("TRAILSCOPE_UPLOAD_DIR", str(_BACKEND_DIR / "uploads")))
RESULTS_DIR = Path(os.getenv("TRAILSCOPE_RESULTS_DIR", str(_BACKEND_DIR / "results")))

# --------------------------------------------------------------------------------------
# Operational settings (v1.1+; env-overridable — NOT locked invariants)
# --------------------------------------------------------------------------------------
# Per-result dirs are cleared on startup and also swept by age on each request.
RESULTS_TTL_SECONDS = int(os.getenv("TRAILSCOPE_RESULTS_TTL", "3600"))
# CPU inference is serial-ish; cap concurrent /infer runs and reject the overflow (429).
MAX_CONCURRENT_INFER = int(os.getenv("TRAILSCOPE_MAX_CONCURRENT_INFER", "2"))
# Best-effort cache: identical (file bytes + options) → reuse the existing result dir.
DEMO_CACHE_SIZE = int(os.getenv("TRAILSCOPE_DEMO_CACHE_SIZE", "64"))
# Async jobs path (opt-in, for full-frame work beyond the 64-patch synchronous limit).
# The synchronous /infer default stays at MAX_PATCH_BUDGET; the async path lifts it to
# this hard ceiling (RAM/time-bounded), never unbounded.
MAX_JOB_PATCH_BUDGET = int(os.getenv("TRAILSCOPE_MAX_JOB_PATCH_BUDGET", "256"))
MAX_CONCURRENT_JOBS = int(os.getenv("TRAILSCOPE_MAX_CONCURRENT_JOBS", "1"))
MAX_PENDING_JOBS = int(os.getenv("TRAILSCOPE_MAX_PENDING_JOBS", "8"))
# Longest display edge for the "as uploaded" preview (downsized for transport only).
PREVIEW_MAX_EDGE = 1200
# Optionally TorchScript-trace the locked model at load for a CPU speedup; verified
# equivalent to the eager model before use, with eager fallback on any mismatch/error.
USE_TORCHSCRIPT = os.getenv("TRAILSCOPE_USE_TORCHSCRIPT", "1") not in ("0", "false", "False")


def sha256_file(path: str | Path) -> str:
    """Stream a SHA-256 over a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_sha_ok() -> bool:
    """True iff the checkpoint exists and matches the locked SHA-256."""
    try:
        return sha256_file(CHECKPOINT_PATH) == EXPECTED_CHECKPOINT_SHA256
    except OSError:
        return False
