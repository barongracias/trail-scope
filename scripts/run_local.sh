#!/usr/bin/env bash
# Run trail-scope locally without Docker: CPU-only backend (uvicorn) + Next.js dev server.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"
VENV_DIR="${ROOT_DIR}/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN=python3

echo "Setting up the in-repo .venv with ${PYTHON_BIN}..."
if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip >/dev/null

echo "Installing CPU torch + backend deps..."
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r "${BACKEND_DIR}/requirements.lock"

echo "Installing frontend deps..."
cd "${FRONTEND_DIR}"
npm install

export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:8000}"

echo "Starting backend (uvicorn) on :8000..."
cd "${BACKEND_DIR}"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
trap "echo 'Stopping backend (PID ${BACKEND_PID})'; kill ${BACKEND_PID} 2>/dev/null || true" EXIT

cd "${FRONTEND_DIR}"
echo "Starting frontend (Next.js) on :3000..."
npm run dev
