#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export VITE_ORIGIN="http://localhost:5173"
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
