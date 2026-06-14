#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building trail-scope-backend:latest"
docker build -t trail-scope-backend:latest -f backend/Dockerfile backend

echo "Building trail-scope-frontend:latest"
docker build -t trail-scope-frontend:latest -f frontend/Dockerfile frontend

echo "Build complete."
