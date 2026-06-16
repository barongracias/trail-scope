"""Async jobs path — full-frame inference beyond the synchronous 64-patch limit.

ADDITIVE: the synchronous ``/infer`` path is unchanged. Large images opt in to a
background job — ``POST /jobs`` returns a ``job_id`` immediately, a single CPU worker
processes it (bounded by ``MAX_CONCURRENT_JOBS``), and coarse progress is written to a
per-job ``status.json`` (polled via ``GET /jobs/{id}/status``). This realises the spec v2
candidate "async jobs via status files in per-result dirs" and the full-frame ceiling
(``MAX_JOB_PATCH_BUDGET``), without ever making a request unbounded.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

from . import artifacts, config
from .inference import ModelService
from .preprocess import PreprocessError, preprocess_image

_STAGE_DETAIL = {
    "queued": "Queued",
    "preprocessing": "Preparing image",
    "inferring": "Running locked U-Net",
    "rendering": "Rendering outputs",
    "done": "Done",
    "error": "Failed",
    "cancelled": "Cancelled",
}
_TERMINAL = {"done", "error", "cancelled"}


class JobManager:
    """Tracks background inference jobs; one CPU worker, status mirrored to disk."""

    def __init__(self, service: ModelService) -> None:
        self.service = service
        self._status: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(config.MAX_CONCURRENT_JOBS)

    def active_count(self) -> int:
        return sum(1 for s in self._status.values() if s.get("state") not in _TERMINAL)

    def _write(self, job_id: str, **fields: Any) -> dict[str, Any]:
        st = self._status.setdefault(job_id, {"job_id": job_id})
        st.update(fields)
        if "state" in fields:
            st["detail"] = _STAGE_DETAIL.get(fields["state"], st.get("detail", ""))
        try:
            d = config.RESULTS_DIR / job_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "status.json").write_text(json.dumps(st) + "\n")
            # Bump the DIR mtime (overwriting status.json does not) so the TTL sweep
            # keeps an active job's dir fresh; TTL then counts from the last write.
            os.utime(d, None)
        except OSError:  # pragma: no cover - defensive
            pass
        return st

    def submit(
        self,
        job_id: str,
        stored_path: Path,
        filename: str,
        *,
        hough: bool,
        pixel_scale_arcsec: float | None,
        hdu_index: int | None,
    ) -> dict[str, Any]:
        self._prune()
        st = self._write(
            job_id, state="queued", n_patches=None, result_id=None, stats=None,
            error=None, status_code=None,
        )
        task = asyncio.create_task(
            self._run(job_id, stored_path, filename, hough=hough,
                      pixel_scale_arcsec=pixel_scale_arcsec, hdu_index=hdu_index)
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(job_id, None))
        return st

    async def _run(
        self, job_id: str, stored_path: Path, filename: str, *,
        hough: bool, pixel_scale_arcsec: float | None, hdu_index: int | None,
    ) -> None:
        try:
            async with self._sem:
                self._write(job_id, state="preprocessing")
                pre = await run_in_threadpool(
                    preprocess_image, stored_path, filename=filename,
                    pixel_scale_arcsec=pixel_scale_arcsec, hdu_index=hdu_index,
                    max_patch_budget=config.MAX_JOB_PATCH_BUDGET,
                )
                self._write(job_id, state="inferring", n_patches=pre.n_patches)

                def on_stage(name: str) -> None:
                    # Called from the worker thread; dict update + small disk write.
                    self._write(job_id, state=name, n_patches=pre.n_patches)

                stats = await run_in_threadpool(
                    artifacts.run_inference, pre, self.service,
                    hough=hough, out_dir=config.RESULTS_DIR / job_id, on_stage=on_stage,
                )
                self._write(job_id, state="done", result_id=job_id, stats=stats,
                            n_patches=pre.n_patches)
        except asyncio.CancelledError:
            # Cancellation is delivered at the next await boundary; the in-flight
            # threadpool computation finishes in the background but its result is dropped.
            self._write(job_id, state="cancelled", error="Cancelled by user")
            raise
        except PreprocessError as exc:
            self._write(job_id, state="error", error=exc.detail, status_code=exc.status_code)
        except Exception as exc:  # pragma: no cover - defensive
            self._write(job_id, state="error", error=str(exc), status_code=500)
        finally:
            Path(stored_path).unlink(missing_ok=True)

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a running job. Returns True if it was cancellable.

        The asyncio task is cancelled (CancelledError raises at its next await); any
        CPU work already running in the worker thread runs to completion in the background
        but its result is discarded. Status flips to 'cancelled' optimistically.
        """
        task = self._tasks.get(job_id)
        st = self._status.get(job_id)
        if task is None or task.done() or (st and st.get("state") in _TERMINAL):
            return False
        task.cancel()
        self._write(job_id, state="cancelled", error="Cancelled by user", status_code=None)
        return True

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        if job_id in self._status:
            return self._status[job_id]
        f = config.RESULTS_DIR / job_id / "status.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
                return None
        return None

    def _prune(self) -> None:
        # Drop in-memory entries whose result dir was TTL-swept (self-healing).
        for jid in list(self._status):
            if not (config.RESULTS_DIR / jid).exists():
                self._status.pop(jid, None)
