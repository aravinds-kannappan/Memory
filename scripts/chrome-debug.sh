#!/usr/bin/env bash
# Optional: launch Chrome with remote debugging so the agent can list/close/open tabs.
# Does not replace your normal Chrome profile — uses a dedicated debug profile.

set -euo pipefail
PROFILE="${HOME}/.browser-memory/chrome-debug-profile"
mkdir -p "$PROFILE"

if [[ "$(uname)" == "Darwin" ]]; then
  open -a "Google Chrome" --args --remote-debugging-port=9222 --user-data-dir="$PROFILE"
else
  CHROME="$(command -v google-chrome || command -v chrome || command -v chromium || true)"
  if [[ -z "$CHROME" ]]; then
    echo "Chrome/Chromium not found"
    exit 1
  fi
  "$CHROME" --remote-debugging-port=9222 --user-data-dir="$PROFILE" >/dev/null 2>&1 &
fi
echo "Chrome debugging on http://127.0.0.1:9222"
