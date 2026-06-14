"""Enforce the vendoring rule: NO `src.*` imports anywhere in the backend.

The inference core is a frozen copy under `trailscope/vendored/` (see VENDOR_MANIFEST.md).
If any module reaches back into the thesis repo's `src.*` packages, vendoring has been
violated (CLAUDE.md hard rule 3).
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Matches `import src...` / `from src...` / `from src.models import ...` etc.,
# but NOT identifiers that merely contain "src" (e.g. a variable named `srcs`).
_SRC_IMPORT = re.compile(r"^\s*(?:import\s+src\b|from\s+src\b)", re.MULTILINE)


def _python_files() -> list[Path]:
    return [
        p
        for p in BACKEND_ROOT.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    ]


def test_no_src_star_imports() -> None:
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if _SRC_IMPORT.search(text):
            offenders.append(str(path.relative_to(BACKEND_ROOT)))
    assert not offenders, f"Found forbidden src.* imports in: {offenders}"
