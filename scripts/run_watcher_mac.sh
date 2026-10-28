#!/usr/bin/env bash
set -euo pipefail

# Load env file if present (keeps secrets out of the repo)
ENV_FILE="$HOME/.config/trading-watcher.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

# Change to repo root
cd "$(dirname "$0")/.."

# Run the watcher script
exec "${PYTHON:-python3}" watch_mode.py
