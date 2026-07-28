#!/usr/bin/env bash
# Desktop launcher: start the backend and open the dashboard as a standalone app window
# (a lightweight desktop wrap — no Electron build needed; uses a Chromium-family browser in
# --app mode, falling back to the default browser). Ctrl-C stops the attached backend.
#
#   ./desktop.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${SWM_PORT:-8723}"; HOST="${SWM_HOST:-127.0.0.1}"; URL="http://${HOST}:${PORT}"
JULIA="${JULIA:-julia}"
JULIA_THREADS="${SWM_JULIA_THREADS:-4}"

command -v "$JULIA" >/dev/null 2>&1 || { echo "error: '$JULIA' not found (install Julia 1.12.6+)." >&2; exit 1; }
if ! "$JULIA" --startup-file=no -e 'exit(VERSION >= v"1.12.6" ? 0 : 1)' >/dev/null 2>&1; then
  echo "error: the monitor app requires Julia 1.12.6 or newer." >&2
  exit 1
fi

echo "Instantiating + starting backend…"
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()' >/dev/null
# Per-invocation private log file (0600, unpredictable name) so a shared host cannot pre-plant
# /tmp/swm_desktop.out to hijack the redirect or make the launcher follow a symlink.
OUT="$(mktemp "${TMPDIR:-/tmp}/swm_desktop.XXXXXX")"
nohup "$JULIA" --threads="$JULIA_THREADS" --project=. src/server.jl > "$OUT" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null || true; rm -f "$OUT" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Wait for readiness (bounded: 60 s), exit fast if the server dies.
ready=0
for _ in $(seq 1 60); do
  if curl -fs "${URL}/api/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "backend exited during startup:"; cat "$OUT"; exit 1; }
  sleep 1
done
[ "$ready" = 1 ] || { echo "backend did not become ready in 60 s."; cat "$OUT"; exit 1; }

open_window() {
  local c
  for c in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    google-chrome chromium chromium-browser microsoft-edge; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then
      "$c" --app="$URL" --window-size=1280,920 >/dev/null 2>&1 &
      return 0
    fi
  done
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  else echo "Open ${URL} in your browser."; fi
}

echo "Space-Weather Threat Monitor running at ${URL}  (press Ctrl-C here to stop the backend)."
open_window
wait "$SRV"
