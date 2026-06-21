#!/usr/bin/env bash
# One-command launch for the Space-Weather Threat Monitor.
#   ./run.sh                 # start on http://127.0.0.1:8723
#   SWM_PORT=9000 ./run.sh   # custom port
#   SWM_HOST=0.0.0.0 ./run.sh # bind all interfaces (LAN access)
#   SOLARSINDY_LOG=/path/to/live_forecast_log.csv ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
JULIA="${JULIA:-julia}"

if ! command -v "$JULIA" >/dev/null 2>&1; then
  echo "error: '$JULIA' not found. Install Julia 1.10+ (https://julialang.org/downloads/)." >&2
  exit 1
fi

echo "Instantiating Julia environment (first run may take a moment)…"
"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate()'

exec "$JULIA" --project=. src/server.jl
