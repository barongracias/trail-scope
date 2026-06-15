"""Vendored-core integrity guard (roadmap v1.4).

The inference core under `trailscope/vendored/` is a FROZEN verbatim copy (a frozen
upstream cannot drift — see VENDOR_MANIFEST.md). To turn an *accidental local edit* of
that copy into a loud failure rather than silent drift, each vendored source's SHA-256 is
recorded in `vendored/CHECKSUMS.sha256`; this test re-hashes the files and compares.

If you intentionally re-vendor (new upstream commit), regenerate the file with:
    cd backend/trailscope/vendored && shasum -a 256 \\
        unet.py loading.py hough_runner.py preprocess_core.py > CHECKSUMS.sha256
and update VENDOR_MANIFEST.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

VENDORED = Path(__file__).resolve().parent.parent / "trailscope" / "vendored"
CHECKSUMS = VENDORED / "CHECKSUMS.sha256"


def _parse_checksums() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in CHECKSUMS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


def test_vendored_files_match_recorded_checksums() -> None:
    recorded = _parse_checksums()
    assert recorded, "CHECKSUMS.sha256 is empty"
    for name, expected in recorded.items():
        path = VENDORED / name
        assert path.exists(), f"vendored file missing: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"vendored/{name} changed (sha256 {actual} != recorded {expected}). "
            "If this was an intentional re-vendor, regenerate CHECKSUMS.sha256."
        )


def test_all_vendored_modules_are_checksummed() -> None:
    # Guard against adding a new vendored module without recording its hash.
    recorded = set(_parse_checksums())
    modules = {
        p.name
        for p in VENDORED.glob("*.py")
        if p.name != "__init__.py"
    }
    assert modules <= recorded, f"vendored modules missing from CHECKSUMS.sha256: {modules - recorded}"
