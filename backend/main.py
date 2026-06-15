"""trail-scope API — THIN entrypoint.

Routes, validation, CORS, lifespan, and error shaping only. All real logic lives in
`trailscope/` (config, inference, preprocess, artifacts) and the frozen `vendored/`
core. Scaffold (structured `log_event`, `error_response`, env-driven CORS, lifespan
dir-clearing) is lifted from the author's InterPyApp; the training/predict machinery
was stripped — trail-scope ships one locked checkpoint, no training, no tunable knobs.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from trailscope import artifacts, config
from trailscope.inference import CheckpointError, ModelService
from trailscope.jobs import JobManager
from trailscope.preprocess import (
    PreprocessError,
    is_fits_filename,
    list_fits_hdus,
    preprocess_image,
)
from trailscope.schemas import (
    HealthResponse,
    InferResponse,
    InspectResponse,
    JobCreateResponse,
    JobStatus,
    ModelResponse,
)


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
    _RESULT_CACHE.clear()
    try:
        app.state.model_service = ModelService.load()
        app.state.job_manager = JobManager(app.state.model_service)
        log_event(
            "startup.model_loaded",
            params=app.state.model_service.param_count,
            backend=app.state.model_service.backend,
            checkpoint_sha_ok=True,
        )
    except CheckpointError as exc:
        # Refuse to serve inference on SHA mismatch / corruption, but stay up so
        # /health can report the failure honestly (CLAUDE.md rule 4).
        app.state.model_service = None
        app.state.job_manager = None
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
    # The API is stateless multipart/form-data — no cookies, auth headers, or sessions —
    # so credentialed cross-origin requests are neither needed nor advertised.
    allow_credentials=False,
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
_RESULT_FILES = {
    "input_8bit.png",
    "mask.png",
    "overlay.png",
    "prob.png",
    "original_preview.png",
    "stats.json",
}
_RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Best-effort result cache: (content digest + options) → result_id. Bounded LRU; entries
# are self-healing (a hit is re-validated against the result dir before reuse).
_RESULT_CACHE: "OrderedDict[str, str]" = OrderedDict()
# In-flight counts for the concurrency guards (event-loop-serialised ints).
_active_infers = 0
_active_inspects = 0


def _cache_get(key: str) -> str | None:
    rid = _RESULT_CACHE.get(key)
    if rid is None:
        return None
    if not (config.RESULTS_DIR / rid / "stats.json").exists():
        _RESULT_CACHE.pop(key, None)  # stale (TTL-swept) — drop it
        return None
    _RESULT_CACHE.move_to_end(key)
    return rid


def _cache_put(key: str, result_id: str) -> None:
    _RESULT_CACHE[key] = result_id
    _RESULT_CACHE.move_to_end(key)
    while len(_RESULT_CACHE) > config.DEMO_CACHE_SIZE:
        _RESULT_CACHE.popitem(last=False)


def _cleanup_old_results() -> None:
    """Remove per-result dirs older than RESULTS_TTL_SECONDS (best-effort)."""
    if not config.RESULTS_DIR.exists():
        return
    cutoff = time.time() - config.RESULTS_TTL_SECONDS
    for d in config.RESULTS_DIR.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:  # pragma: no cover - defensive
            pass


def _matched_extension(filename: str) -> str | None:
    """Return the longest allowed suffix matching `filename` (handles `.fits.fz`)."""
    name = filename.lower()
    for ext in sorted(config.ALLOWED_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return ext
    return None


async def _store_upload(file: UploadFile) -> tuple[Path, str, str, str]:
    """Validate (extension + 64 MB cap) and store the upload under a UUID. Returns
    (stored_path, result_id, original_filename, content_sha256)."""
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
    hasher = hashlib.sha256()
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
            hasher.update(chunk)
            fh.write(chunk)
    return dest, result_id, original, hasher.hexdigest()


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
    global _active_infers
    service = getattr(app.state, "model_service", None)
    if service is None:
        raise error_response(503, "Model unavailable: checkpoint failed the integrity gate.")

    _cleanup_old_results()
    stored_path, result_id, original, digest = await _store_upload(file)

    # Best-effort cache: identical (bytes + options) → reuse the existing result dir.
    cache_key = f"{digest}:{hough}:{pixel_scale_arcsec}:{hdu_index}"
    cached = _cache_get(cache_key)
    if cached is not None:
        stored_path.unlink(missing_ok=True)
        stats = json.loads((config.RESULTS_DIR / cached / "stats.json").read_text())
        log_event("infer.cache_hit", result_id=cached, tier=stats["tier"])
        return InferResponse(result_id=cached, stats=stats)

    # Concurrency guard: CPU inference is heavy; reject the overflow rather than queue it.
    if _active_infers >= config.MAX_CONCURRENT_INFER:
        stored_path.unlink(missing_ok=True)
        raise error_response(429, "Server busy: too many concurrent inferences. Retry shortly.")

    _active_infers += 1
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
        _active_infers -= 1
        stored_path.unlink(missing_ok=True)

    _cache_put(cache_key, result_id)
    log_event(
        "infer.completed",
        result_id=result_id,
        tier=stats["tier"],
        n_patches=stats["image"]["n_patches"],
    )
    return InferResponse(result_id=result_id, stats=stats)


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)) -> InspectResponse:
    """List the HDUs of an uploaded FITS file (for the HDU picker). Stored briefly, then
    deleted; no inference is run."""
    global _active_inspects
    original = os.path.basename(file.filename or "")
    if not is_fits_filename(original):
        raise error_response(400, "HDU inspection is only available for FITS files.")
    stored_path, _rid, _orig, _digest = await _store_upload(file)
    # Guard against memory-exhaustion from a burst of concurrent FITS reads (each holds
    # up to 64 MB while astropy parses it). Check+increment are adjacent (no await
    # between) so they are atomic under asyncio; the parse is fast so the window is narrow.
    if _active_inspects >= config.MAX_CONCURRENT_INFER:
        stored_path.unlink(missing_ok=True)
        raise error_response(429, "Server busy: too many concurrent inspections. Retry shortly.")
    _active_inspects += 1
    try:
        hdus = await run_in_threadpool(list_fits_hdus, stored_path, original)
    except PreprocessError as exc:
        raise error_response(exc.status_code, exc.detail)
    except Exception as exc:  # pragma: no cover - defensive
        raise error_response(400, f"Could not read FITS HDUs: {exc}")
    finally:
        _active_inspects -= 1
        stored_path.unlink(missing_ok=True)
    return InspectResponse(hdus=hdus)


# --------------------------------------------------------------------------------------
# Async jobs path (opt-in) — full-frame inference beyond the 64-patch synchronous limit.
# Additive: the synchronous /infer above is unchanged.
# --------------------------------------------------------------------------------------
@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    file: UploadFile = File(...),
    hough: bool = Form(True),
    pixel_scale_arcsec: Optional[float] = Form(None),
    hdu_index: Optional[int] = Form(None),
) -> JobCreateResponse:
    """Submit a large image for background inference (up to MAX_JOB_PATCH_BUDGET patches).
    Returns immediately; poll GET /jobs/{id}/status. Model and threshold are still fixed."""
    service = getattr(app.state, "model_service", None)
    job_manager = getattr(app.state, "job_manager", None)
    if service is None or job_manager is None:
        raise error_response(503, "Model unavailable: checkpoint failed the integrity gate.")

    _cleanup_old_results()
    if job_manager.active_count() >= config.MAX_PENDING_JOBS:
        raise error_response(429, "Too many queued jobs. Retry shortly.")

    stored_path, job_id, original, _digest = await _store_upload(file)
    job_manager.submit(
        job_id,
        stored_path,
        original,
        hough=hough,
        pixel_scale_arcsec=pixel_scale_arcsec,
        hdu_index=hdu_index,
    )
    log_event("job.submitted", job_id=job_id)
    return JobCreateResponse(job_id=job_id)


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    if not _RESULT_ID_RE.match(job_id):
        raise error_response(404, "Unknown job id")
    job_manager = getattr(app.state, "job_manager", None)
    status = job_manager.get_status(job_id) if job_manager else None
    if status is None:
        raise error_response(404, "Job not found")
    return JobStatus(**status)


@app.get("/results/{result_id}/{filename}")
async def get_result(result_id: str, filename: str):
    """Serve a per-result artifact (or a .zip bundle of all of them). Results live in a
    per-UUID dir cleared on startup and swept by age."""
    if not _RESULT_ID_RE.match(result_id):
        raise error_response(404, "Unknown result id")
    result_dir = config.RESULTS_DIR / result_id

    if filename == "bundle.zip":
        if not result_dir.is_dir():
            raise error_response(404, "Result not found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(result_dir.iterdir()):
                if f.is_file() and f.name in _RESULT_FILES:
                    zf.write(f, arcname=f.name)
        headers = {"Content-Disposition": f'attachment; filename="trail-scope-{result_id[:8]}.zip"'}
        return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)

    if filename not in _RESULT_FILES:
        raise error_response(404, f"Unknown artifact: {filename}")
    path = result_dir / filename
    if not path.exists():
        raise error_response(404, "Result not found")
    media_type = "application/json" if filename.endswith(".json") else "image/png"
    return FileResponse(path, media_type=media_type)
