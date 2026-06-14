#!/usr/bin/env python
"""Fetch a public DECam detector frame from the NOIRLab archive on demand.

Retrieval logic is adapted from the thesis figure script
`bg492/scripts/figures/decam_cold_inference.py` (NOIRLab advanced-search → per-detector
HDU retrieve), rewritten to use `httpx` instead of `requests`. The nine predeclared
RECA/NOIRLab measured-streak frames are public DECam data — **never** MeerLICHT imagery
(CLAUDE.md rule 5). Include the NOIRLab acknowledgement wherever these appear.

Downloaded FITS are large and are NOT committed: write them to a gitignored cache. The
repo ships only small cropped 8-bit PNG examples derived from these frames.

Usage:
    python scripts/download_demo_assets.py --expnum 1134933 --detector 5 \
        --out-dir .demo_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

NOIRLAB_ROOT = "https://astroarchive.noirlab.edu"
NOIRLAB_ACKNOWLEDGEMENT = (
    "Based on observations at Cerro Tololo Inter-American Observatory, NSF's NOIRLab, "
    "managed by AURA under a cooperative agreement with the U.S. National Science "
    "Foundation."
)

# The nine predeclared RECA measured-streak frames (object, expnum, detector).
PREDECLARED = {
    (1134933, 5): "NAVSTAR-70",
    (1138498, 23): "STARLINK-2600",
    (1033925, 17): "STARLINK-2559",
    (1103448, 4): "STARLINK-3758",
    (1103448, 20): "STARLINK-3772",
    (1103448, 45): "STARLINK-3771",
    (1103448, 34): "STARLINK-3765",
    (1072590, 5): "DELTA-2 R/B",
    (1125268, 21): "SORCE",
}


def noirlab_search(expnum: int, proc_type: str = "instcal", timeout: float = 180.0) -> list[dict]:
    import httpx

    qspec = {
        "outfields": [
            "md5sum", "url", "archive_filename", "original_filename",
            "instrument", "proc_type", "EXPNUM", "prod_type",
        ],
        "search": [
            ["instrument", "decam"],
            ["proc_type", proc_type],
            ["EXPNUM", int(expnum), int(expnum)],
            ["prod_type", "image"],
        ],
    }
    url = f"{NOIRLAB_ROOT}/api/adv_search/find/?limit=20"
    resp = httpx.post(url, json=qspec, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if len(payload) >= 2 and isinstance(payload[0], dict) and "HEADER" in payload[0]:
        return [row for row in payload[1:] if isinstance(row, dict)]
    if len(payload) >= 2 and isinstance(payload[0], list):
        columns = payload[0]
        return [dict(zip(columns, row)) for row in payload[1:]]
    raise RuntimeError(f"NOIRLab search returned no image rows for expnum={expnum}")


def retrieve_detector_hdu(
    expnum: int, detector: int, out_dir: Path, proc_type: str = "instcal", timeout: float = 180.0
) -> Path:
    import httpx

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"decam_exp{expnum}_det{detector}_{proc_type}.fits"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Cached: {out_path}")
        return out_path

    rows = noirlab_search(expnum, proc_type, timeout=timeout)
    if "md5sum" not in rows[0]:
        raise RuntimeError(f"NOIRLab search row lacks md5sum for expnum={expnum}")
    base_url = rows[0].get("url") or f"{NOIRLAB_ROOT}/api/retrieve/{rows[0]['md5sum']}/"
    sep = "&" if "?" in base_url else "?"
    retrieve_url = f"{base_url}{sep}hdus={detector}"
    print(f"Downloading exp {expnum} det {detector} → {out_path}")
    with httpx.stream("GET", retrieve_url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                fh.write(chunk)
    print(f"Saved {out_path.stat().st_size} bytes. {NOIRLAB_ACKNOWLEDGEMENT}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expnum", type=int, default=1134933)
    parser.add_argument("--detector", type=int, default=5)
    parser.add_argument("--out-dir", default=".demo_cache")
    args = parser.parse_args()

    key = (args.expnum, args.detector)
    if key in PREDECLARED:
        print(f"Object: {PREDECLARED[key]} (predeclared RECA measured-streak frame)")
    retrieve_detector_hdu(args.expnum, args.detector, Path(args.out_dir))


if __name__ == "__main__":
    main()
