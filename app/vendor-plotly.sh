#!/usr/bin/env bash
# Optional: vendor Plotly.js locally so the dashboard works fully offline.
# Without this, index.html falls back to the official Plotly CDN automatically.
set -euo pipefail
cd "$(dirname "$0")"
VER="${PLOTLY_VERSION:-2.35.2}"
mkdir -p public/vendor
echo "Downloading Plotly ${VER} → public/vendor/plotly.min.js …"
curl -fsSL "https://cdn.plot.ly/plotly-${VER}.min.js" -o public/vendor/plotly.min.js
echo "Done ($(du -h public/vendor/plotly.min.js | cut -f1)). The dashboard will now use the local copy."
