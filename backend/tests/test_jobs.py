"""Async jobs path (#11) + full-frame ceiling (#12).

The synchronous /infer path is unchanged (covered elsewhere). These tests drive the
opt-in background path: submit → poll status → done/error, and confirm the jobs path
lifts the 64-patch budget to MAX_JOB_PATCH_BUDGET.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image

import main
from trailscope import config


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _fits_bytes(arr: np.ndarray, cards: dict | None = None) -> bytes:
    buf = io.BytesIO()
    hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32))
    for k, v in (cards or {}).items():
        hdu.header[k] = v
    hdu.writeto(buf)
    return buf.getvalue()


def _poll(client, job_id, timeout=60.0):
    deadline = time.time() + timeout
    st = {}
    while time.time() < deadline:
        st = client.get(f"/jobs/{job_id}/status").json()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.1)
    return st


def test_job_completes_and_serves_artifacts(client):
    arr = (np.random.default_rng(20).random((300, 300)) * 255).astype(np.uint8)
    job_id = client.post("/jobs", files={"file": ("j.png", _png_bytes(arr), "image/png")}).json()[
        "job_id"
    ]
    st = _poll(client, job_id)
    assert st["state"] == "done", st
    assert st["result_id"] == job_id
    assert st["stats"]["model_output"]["predicted_mask_pixel_count"] >= 0
    assert st["stats"]["qualitative_only"] is True
    # Artifacts served from the same per-id dir as /infer.
    assert client.get(f"/results/{job_id}/overlay.png").status_code == 200
    assert client.get(f"/results/{job_id}/stats.json").status_code == 200


def test_jobs_path_lifts_the_64_patch_budget(client, monkeypatch):
    # Sync /infer would 413 (budget 1), but the jobs path uses MAX_JOB_PATCH_BUDGET.
    monkeypatch.setattr(config, "MAX_PATCH_BUDGET", 1)
    monkeypatch.setattr(config, "MAX_JOB_PATCH_BUDGET", 16)
    P = config.PATCH_SIZE
    side = int(P * 1.2)  # 2×2 = 4 patches → 413 at sync budget 1, OK at job budget 16
    payload = _png_bytes((np.random.default_rng(21).random((side, side)) * 255).astype(np.uint8))

    # Sync path rejects it.
    assert client.post("/infer", files={"file": ("big.png", payload, "image/png")}).status_code == 413
    # Jobs path completes.
    job_id = client.post("/jobs", files={"file": ("big.png", payload, "image/png")}).json()["job_id"]
    st = _poll(client, job_id, timeout=120)
    assert st["state"] == "done", st
    assert st["n_patches"] == 4


def test_job_over_ceiling_ends_in_error(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_PATCH_BUDGET", 1)
    P = config.PATCH_SIZE
    side = int(P * 1.2)  # 4 patches > ceiling 1
    payload = _png_bytes((np.random.default_rng(22).random((side, side)) * 255).astype(np.uint8))
    job_id = client.post("/jobs", files={"file": ("x.png", payload, "image/png")}).json()["job_id"]
    st = _poll(client, job_id)
    assert st["state"] == "error"
    assert st["status_code"] == 413
    assert "patches >" in st["error"]


def test_job_status_404_for_unknown_id(client):
    assert client.get("/jobs/" + "a" * 32 + "/status").status_code == 404
    assert client.get("/jobs/not-hex/status").status_code == 404


def test_jobs_503_when_model_unavailable(tmp_path, monkeypatch):
    bad = tmp_path / "model-best.pth"
    bad.write_bytes(b"corrupt")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", bad)
    with TestClient(main.app) as c:
        arr = (np.random.default_rng(23).random((64, 64)) * 255).astype(np.uint8)
        resp = c.post("/jobs", files={"file": ("j.png", _png_bytes(arr), "image/png")})
        assert resp.status_code == 503
