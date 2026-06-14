"""trail-scope API — THIN entrypoint.

Routes, validation, CORS, lifespan, and error shaping only. All real logic lives in
`trailscope/` (config, inference, preprocess, artifacts) and the frozen `vendored/`
core. Scaffold (structured `log_event`, `error_response`, env-driven CORS, lifespan
dir-clearing) is lifted from the author's InterPyApp; the training/predict machinery
was stripped — trail-scope ships one locked checkpoint, no training, no tunable knobs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from trailscope import config
from trailscope.inference import CheckpointError, ModelService
from trailscope.schemas import HealthResponse, ModelResponse

# --------------------------------------------------------------------------------------
# Structured logging (lifted from InterPyApp)
# --------------------------------------------------------------------------------------
def get_app_logger() -> logging.Logger:
    logger = logging.getLogger("trailscope")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s",
                datefmt="%d/%m/%y %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger


app_logger = get_app_logger()


def log_event(action: str, **fields) -> None:
    """Emit a structured log line with a consistent prefix."""
    try:
        payload = json.dumps(fields, default=str, sort_keys=True)
    except Exception as exc:  # pragma: no cover - defensive
        payload = json.dumps({"log_error": str(exc)})
    app_logger.info(f"event={action} {payload}")


def error_response(status_code: int, detail: str) -> HTTPException:
    """Consistent error responses."""
    return HTTPException(status_code=status_code, detail=detail)


def _clear_dir(path) -> None:
    os.makedirs(path, exist_ok=True)
    for entry in os.scandir(path):
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        except OSError as exc:  # pragma: no cover - defensive
            app_logger.warning(f"Could not remove {entry.path}: {exc}")


# --------------------------------------------------------------------------------------
# Lifespan: clear per-run dirs, run the checkpoint SHA gate, load the U-Net once
# --------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _clear_dir(config.UPLOAD_DIR)
    _clear_dir(config.RESULTS_DIR)
    try:
        app.state.model_service = ModelService.load()
        log_event(
            "startup.model_loaded",
            params=app.state.model_service.param_count,
            checkpoint_sha_ok=True,
        )
    except CheckpointError as exc:
        # Refuse to serve inference on SHA mismatch / corruption, but stay up so
        # /health can report the failure honestly (CLAUDE.md rule 4).
        app.state.model_service = None
        app.state.model_error = str(exc)
        app_logger.error(f"Checkpoint gate failed — inference disabled: {exc}")
    yield


app = FastAPI(
    title="trail-scope API",
    description=config.SCOPE_SENTENCE,
    version=config.VERSION,
    contact={"name": "Baron Gracias"},
    lifespan=lifespan,
)


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [x.strip() for x in raw.split(",") if x.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=config.VERSION,
        model_sha_ok=getattr(app.state, "model_service", None) is not None,
    )


@app.get("/model", response_model=ModelResponse)
async def model() -> ModelResponse:
    """Static metadata for the locked detector. Disclaimer present (CLAUDE.md rule 6)."""
    return ModelResponse(
        architecture=config.ARCHITECTURE_SUMMARY,
        param_count=config.EXPECTED_PARAM_COUNT,
        threshold=config.THRESHOLD,
        patch_size=config.PATCH_SIZE,
        normalisation=config.NORMALISATION,
        training_domain=config.TRAINING_DOMAIN,
        tier_definitions={
            "in_domain_like": "8-bit display PNG at a plausible scale.",
            "recipe_matched": "FITS with a header-resolved pixel scale (the validated "
            "DECam-style recipe).",
            "best_effort": "Everything else (e.g. unknown pixel scale).",
        },
        thesis_repo=config.THESIS_REPO_LINK,
        checkpoint_sha256=config.EXPECTED_CHECKPOINT_SHA256,
        scope=config.SCOPE_SENTENCE,
        disclaimer=config.DISCLAIMER,
        user_facing_knobs=["hough (on/off)", "pixel_scale_arcsec (optional)", "hdu_index (optional)"],
    )
