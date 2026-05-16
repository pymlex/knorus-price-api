#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiles

py-spy record \
  -o profiles/knorus_api.svg \
  --format svg \
  --rate 100 \
  -- python -m uvicorn app.main:app --host 0.0.0.0 --port 8000