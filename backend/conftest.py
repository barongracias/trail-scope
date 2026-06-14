"""Ensure the backend root is importable as `main` / `trailscope` during tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
