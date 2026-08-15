#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "→ creating .venv"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "→ wrote .env from .env.example — add your keys, then re-run"
fi

PORT="${SM_PORT:-8137}"
HOST="${SM_HOST:-127.0.0.1}"

echo "→ http://${HOST}:${PORT}"
exec ./.venv/bin/uvicorn backend.main:app --host "$HOST" --port "$PORT" "$@"
