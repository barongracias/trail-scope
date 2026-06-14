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
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from trailscope import artifacts, config
from trailscope.inference import CheckpointError, ModelService
from trailscope.preprocess import PreprocessError, preprocess_image
from trailscope.schemas import HealthResponse, InferResponse, ModelResponse


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


# --------------------------------------------------------------------------------------
# /infer — upload validation pattern lifted from InterPyApp (extension allowlist + size
# cap + UUID storage), then the locked preprocess → inference → artifacts path.
# --------------------------------------------------------------------------------------
_RESULT_FILES = {"input_8bit.png", "mask.png", "overlay.png", "stats.json"}
_RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _matched_extension(filename: str) -> str | None:
    """Return the longest allowed suffix matching `filename` (handles `.fits.fz`)."""
    name = filename.lower()
    for ext in sorted(config.ALLOWED_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return None


async def _store_upload(file: UploadFile) -> tuple[Path, str, str]:
    """Validate (extension + 64 MB cap) and store the upload under a UUID. Returns
    (stored_path, result_id, original_filename)."""
    original = os.path.basename(file.filename or "")
    ext = _matched_extension(original)
    if ext is None:
        raise error_response(
            400,
            f"Unsupported file type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )
    result_id = uuid.uuid4().hex
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOAD_DIR / f"{result_id}{ext}"
    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                raise error_response(413, "File too large. Limit is 64 MB.")
            fh.write(chunk)
    return dest, result_id, original


def _run_inference_job(
    stored_path: Path,
    original_filename: str,
    service: ModelService,
    *,
    hough: bool,
    pixel_scale_arcsec: float | None,
    hdu_index: int | None,
    result_id: str,
) -> dict:
    pre = preprocess_image(
        stored_path,
        filename=original_filename,
        pixel_scale_arcsec=pixel_scale_arcsec,
        hdu_index=hdu_index,
    )
    return artifacts.run_inference(
        pre, service, hough=hough, out_dir=config.RESULTS_DIR / result_id
    )


@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    hough: bool = Form(True),
    pixel_scale_arcsec: Optional[float] = Form(None),
    hdu_index: Optional[int] = Form(None),
) -> InferResponse:
    """Synchronous single-image inference. The model and threshold are fixed; this run
    does not tune parameters or estimate accuracy."""
    service = getattr(app.state, "model_service", None)
    if service is None:
        raise error_response(503, "Model unavailable: checkpoint failed the integrity gate.")

    stored_path, result_id, original = await _store_upload(file)
    try:
        stats = await run_in_threadpool(
            _run_inference_job,
            stored_path,
            original,
            service,
            hough=hough,
            pixel_scale_arcsec=pixel_scale_arcsec,
            hdu_index=hdu_index,
            result_id=result_id,
        )
    except PreprocessError as exc:
        log_event("infer.rejected", status=exc.status_code, detail=exc.detail)
        raise error_response(exc.status_code, exc.detail)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log_event("infer.error", error=str(exc))
        raise error_response(500, f"Inference failed: {exc}")
    finally:
        stored_path.unlink(missing_ok=True)

    log_event(
        "infer.completed",
        result_id=result_id,
        tier=stats["tier"],
        n_patches=stats["image"]["n_patches"],
    )
    return InferResponse(result_id=result_id, stats=stats)


@app.get("/results/{result_id}/{filename}")
async def get_result(result_id: str, filename: str):
    """Serve a per-result artifact. Results live in a per-UUID dir cleared on startup."""
    if filename not in _RESULT_FILES:
        raise error_response(404, f"Unknown artifact: {filename}")
    if not _RESULT_ID_RE.match(result_id):
        raise error_response(404, "Unknown result id")
    path = config.RESULTS_DIR / result_id / filename
    if not path.exists():
        raise error_response(404, "Result not found")
    media_type = "application/json" if filename.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media_type)
