"""API-layer tests (spec §Tests test_api.py): upload validation, a schema-valid sync
infer with all four honesty fields, the 413 path, results 404s, and the startup refusal
on a corrupted checkpoint.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image

import main
from trailscope import config
from trailscope.schemas import InferStats


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _fits_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    fits.PrimaryHDU(np.asarray(arr, dtype=np.float32)).writeto(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------------------
def test_rejects_unsupported_extension(client):
    resp = client.post("/infer", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_infer_returns_schema_valid_stats_with_honesty_fields(client):
    arr = (np.random.default_rng(0).random((300, 300)) * 255).astype(np.uint8)
    resp = client.post(
        "/infer",
        files={"file": ("demo.png", _png_bytes(arr), "image/png")},
        data={"hough": "true"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "result_id" in body
    stats = InferStats.model_validate(body["stats"])  # schema-valid
    # The four honesty fields.
    assert stats.qualitative_only is True
    assert stats.benchmark is False
    assert stats.threshold_tuned is False
    assert "not a validated detector" in stats.disclaimer
    assert stats.tier == "in_domain_like"


def test_results_files_served_after_infer(client):
    arr = (np.random.default_rng(1).random((300, 300)) * 255).astype(np.uint8)
    rid = client.post(
        "/infer", files={"file": ("d.png", _png_bytes(arr), "image/png")}
    ).json()["result_id"]
    for name, ctype in [
        ("input_8bit.png", "image/png"),
        ("mask.png", "image/png"),
        ("overlay.png", "image/png"),
        ("stats.json", "application/json"),
    ]:
        r = client.get(f"/results/{rid}/{name}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith(ctype)


def test_413_when_too_many_patches(client, monkeypatch):
    # A FITS with a known scale that upsamples past the (lowered) budget → 413.
    monkeypatch.setattr(config, "MAX_PATCH_BUDGET", 4)
    P = config.PATCH_SIZE
    arr = np.random.default_rng(2).random((P, P)).astype(np.float32)
    # Embed CD matrix for scale 1.4 arcsec/px → factor 2.5 → 3×3 patches.
    buf = io.BytesIO()
    hdu = fits.PrimaryHDU(arr)
    deg = (0.56 * 2.5) / 3600.0
    for k, v in {"CD1_1": deg, "CD2_1": 0.0, "CD1_2": 0.0, "CD2_2": deg}.items():
        hdu.header[k] = v
    hdu.writeto(buf)
    resp = client.post("/infer", files={"file": ("big.fits", buf.getvalue(), "application/octet-stream")})
    assert resp.status_code == 413
    assert "patches >" in resp.json()["detail"]


# --------------------------------------------------------------------------------------
# Results 404s
# --------------------------------------------------------------------------------------
def test_results_404_for_garbage_id(client):
    assert client.get("/results/not-a-valid-id/stats.json").status_code == 404


def test_results_404_for_unknown_artifact(client):
    rid = "0" * 32
    assert client.get(f"/results/{rid}/secret.txt").status_code == 404


def test_results_404_before_any_infer(client):
    rid = "a" * 32
    assert client.get(f"/results/{rid}/stats.json").status_code == 404


# --------------------------------------------------------------------------------------
# Health / model gate
# --------------------------------------------------------------------------------------
def test_health_reports_model_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_sha_ok"] is True


def test_corrupted_checkpoint_startup_refuses_inference(tmp_path, monkeypatch):
    bad = tmp_path / "model-best.pth"
    bad.write_bytes(b"corrupted")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", bad)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_sha_ok"] is False
        arr = (np.random.default_rng(3).random((100, 100)) * 255).astype(np.uint8)
        resp = c.post("/infer", files={"file": ("d.png", _png_bytes(arr), "image/png")})
        assert resp.status_code == 503
