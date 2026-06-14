"""P1 acceptance: vendored U-Net instantiates with the locked param count, the SHA
gate works, and `/health` / `/model` serve the disclaimer and honest metadata.

This is a sanity harness, not a benchmark.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trailscope import config
from trailscope.inference import CheckpointError, ModelService
from trailscope.vendored.unet import UNet


def test_unet_param_count_matches_locked_constant() -> None:
    model = UNet(base_channels=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == config.EXPECTED_PARAM_COUNT == 485_673


def test_checkpoint_on_disk_matches_locked_sha() -> None:
    assert config.CHECKPOINT_PATH.exists(), "locked checkpoint must be present"
    assert config.checkpoint_sha_ok()


def test_model_service_loads_and_asserts_params() -> None:
    service = ModelService.load()
    assert service.param_count == 485_673
    assert service.checkpoint_sha256 == config.EXPECTED_CHECKPOINT_SHA256


def test_corrupted_checkpoint_refuses_to_load(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "model-best.pth"
    bad.write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(config, "CHECKPOINT_PATH", bad)
    with pytest.raises(CheckpointError):
        ModelService.load()


def test_health_and_model_endpoints() -> None:
    with TestClient(__import__("main").app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model_sha_ok"] is True

        meta = client.get("/model").json()
        assert meta["param_count"] == 485_673
        assert meta["threshold"] == 0.45
        assert meta["patch_size"] == 528
        # Honesty surfaces (CLAUDE.md rule 6).
        assert "not a validated detector" in meta["disclaimer"]
        assert "qualitative" in meta["scope"].lower()
        assert "MeerLICHT" in meta["training_domain"]
