"""API tests for the v1.1–v1.3 additions: artifacts list (prob/original_preview),
demo-result caching, the concurrency 429, /inspect, the .zip bundle, and TTL cleanup.

Sanity harness, not a benchmark.
"""

from __future__ import annotations

import io
import time
import zipfile

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


def test_infer_lists_prob_artifact_and_serves_it(client):
    arr = (np.random.default_rng(10).random((300, 300)) * 255).astype(np.uint8)
    body = client.post("/infer", files={"file": ("a.png", _png_bytes(arr), "image/png")}).json()
    rid, stats = body["result_id"], body["stats"]
    assert "prob.png" in stats["artifacts"]
    # 8-bit, no resample → no original_preview.
    assert "original_preview.png" not in stats["artifacts"]
    assert client.get(f"/results/{rid}/prob.png").status_code == 200
    assert client.get(f"/results/{rid}/original_preview.png").status_code == 404


def test_fits_resample_emits_original_preview(client):
    deg = 0.2634 / 3600.0
    cards = {"CD1_1": deg, "CD2_1": 0.0, "CD1_2": 0.0, "CD2_2": deg}
    arr = np.random.default_rng(11).random((900, 900)).astype(np.float32)
    body = client.post(
        "/infer", files={"file": ("d.fits", _fits_bytes(arr, cards), "application/octet-stream")}
    ).json()
    rid, stats = body["result_id"], body["stats"]
    assert stats["provenance"]["resample_factor"] != 1.0
    assert "original_preview.png" in stats["artifacts"]
    assert client.get(f"/results/{rid}/original_preview.png").status_code == 200


def test_demo_cache_returns_same_result_id(client):
    arr = (np.random.default_rng(12).random((256, 256)) * 255).astype(np.uint8)
    payload = _png_bytes(arr)
    r1 = client.post("/infer", files={"file": ("c.png", payload, "image/png")}).json()
    r2 = client.post("/infer", files={"file": ("c.png", payload, "image/png")}).json()
    assert r1["result_id"] == r2["result_id"]  # cache hit reuses the dir
    # Different options must NOT collide.
    r3 = client.post(
        "/infer", files={"file": ("c.png", payload, "image/png")}, data={"hough": "false"}
    ).json()
    assert r3["result_id"] != r1["result_id"]


def test_bundle_zip_contains_all_artifacts(client):
    arr = (np.random.default_rng(13).random((256, 256)) * 255).astype(np.uint8)
    rid = client.post("/infer", files={"file": ("z.png", _png_bytes(arr), "image/png")}).json()[
        "result_id"
    ]
    resp = client.get(f"/results/{rid}/bundle.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    names = set(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())
    assert {"input_8bit.png", "mask.png", "overlay.png", "prob.png", "stats.json"} <= names


def test_inspect_lists_fits_hdus(client):
    arr = np.ones((64, 64), dtype=np.float32)
    resp = client.post(
        "/inspect", files={"file": ("h.fits", _fits_bytes(arr), "application/octet-stream")}
    )
    assert resp.status_code == 200
    hdus = resp.json()["hdus"]
    assert hdus[0]["is_2d_image"] is True
    assert hdus[0]["shape"] == [64, 64]


def test_inspect_rejects_non_fits(client):
    arr = (np.random.default_rng(14).random((32, 32)) * 255).astype(np.uint8)
    resp = client.post("/inspect", files={"file": ("x.png", _png_bytes(arr), "image/png")})
    assert resp.status_code == 400


def test_concurrency_guard_returns_429(client, monkeypatch):
    # Force the "busy" branch without real concurrency by pinning the active counter.
    monkeypatch.setattr(config, "MAX_CONCURRENT_INFER", 1)
    monkeypatch.setattr(main, "_active_infers", 1)
    arr = (np.random.default_rng(15).random((64, 64)) * 255).astype(np.uint8)
    resp = client.post("/infer", files={"file": ("b.png", _png_bytes(arr), "image/png")})
    assert resp.status_code == 429


def test_ttl_cleanup_removes_aged_result_dirs(client, monkeypatch):
    arr = (np.random.default_rng(16).random((128, 128)) * 255).astype(np.uint8)
    rid = client.post("/infer", files={"file": ("t.png", _png_bytes(arr), "image/png")}).json()[
        "result_id"
    ]
    result_dir = config.RESULTS_DIR / rid
    assert result_dir.exists()
    # Age the dir past the TTL, then trigger cleanup via another request.
    monkeypatch.setattr(config, "RESULTS_TTL_SECONDS", 0)
    import os as _os

    old = time.time() - 10
    _os.utime(result_dir, (old, old))
    arr2 = (np.random.default_rng(17).random((128, 128)) * 255).astype(np.uint8)
    client.post("/infer", files={"file": ("t2.png", _png_bytes(arr2), "image/png")})
    assert not result_dir.exists()
