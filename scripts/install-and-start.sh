#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/agent"

python3 -m pip install -e . --quiet
browser-memory install
browser-memory serve --open --seed-demo --sync &
AGENT_PID=$!
echo "Browser Memory agent starting (pid $AGENT_PID)"
echo "UI: http://127.0.0.1:8765"
echo "Auto-start on login: enabled"
