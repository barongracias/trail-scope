"""End-to-end milestone (spec §Tests, §Phasing P2 milestone).

One public DECam demo frame through the backend → mask.png + stats.json + input_8bit.png
on CPU, reproducing the recorded thesis result. The full FITS is fetched on demand and
not committed, so this test SKIPS when the frame is absent (e.g. in CI):

    python scripts/download_demo_assets.py --expnum 1134933 --detector 5 --out-dir .demo_cache

NAVSTAR-70 (exp 1134933 det 5): the thesis recorded predicted_pixel_count=13111,
max prob 0.9995, resampled 1926×962, 8 patches. Sanity reproduction, not a benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trailscope import artifacts
from trailscope.inference import ModelService
from trailscope.preprocess import preprocess_image

_FRAME = (
    Path(__file__).resolve().parents[2]
    / ".demo_cache"
    / "decam_exp1134933_det5_instcal.fits"
)


@pytest.mark.skipif(not _FRAME.exists(), reason="DECam demo frame not downloaded")
def test_decam_navstar70_end_to_end(tmp_path) -> None:
    service = ModelService.load()
    # The thesis path used the fixed DECam plate scale 0.2634 (header ≈ 0.2623).
    pre = preprocess_image(
        _FRAME, filename=_FRAME.name, pixel_scale_arcsec=0.2634
    )
    assert pre.tier == "recipe_matched"
    assert pre.processed_shape == (1926, 962)
    assert pre.n_patches == 8
    assert abs(pre.provenance["resample_factor"] - 0.470357) < 1e-4

    stats = artifacts.run_inference(pre, service, hough=True, out_dir=tmp_path)

    for name in ("input_8bit.png", "mask.png", "overlay.png", "stats.json"):
        assert (tmp_path / name).exists(), name

    mo = stats["model_output"]
    assert abs(mo["predicted_mask_pixel_count"] - 13111) <= 20  # version-tolerant
    assert abs(mo["max_model_probability"] - 0.9995) < 1e-3
    assert stats["hough"]["enabled"] is True
    assert stats["hough"]["segment_count"] >= 1
    # Honesty fields present (CLAUDE.md rule 6).
    assert stats["qualitative_only"] is True and stats["benchmark"] is False
    assert "not a validated detector" in stats["disclaimer"]
