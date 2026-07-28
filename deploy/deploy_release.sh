#!/usr/bin/env bash
# Snapshot the package at a committed git revision into a versioned, read-only run directory and
# (optionally) point the launchd services at it. This separates the operational code path from the
# development working tree, so a KeepAlive restart can no longer silently deploy uncommitted or
# mid-edit state into the live forecast path.
#
# Usage:
#   deploy/deploy_release.sh [REVISION] [RELEASES_ROOT]
#     REVISION       git revision to pin (default: HEAD)
#     RELEASES_ROOT  where releases live (default: ~/.solarsindy/releases)
#
# Env:
#   SOLARSINDY_MONITOR_DIR  canonical state dir shared across releases
#                           (default: ~/.solarsindy/var/monitor). Keep this OUTSIDE the release so
#                           the locked-live forecast log persists across deploys.
#   SOLARSINDY_INSTANTIATE=0  skip `Pkg.instantiate` of the snapshot
#   SOLARSINDY_ACTIVATE=1     after snapshotting, run install_launchd.sh against the release
#
# The release is a clean `git archive` export: it contains only committed content at REVISION, never
# the working tree. State (var/monitor) is intentionally not copied; it stays at the canonical dir.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_CLONE="$(cd "$SELF_DIR/.." && pwd)"
REVISION="${1:-HEAD}"
RELEASES_ROOT="${2:-$HOME/.solarsindy/releases}"
MONITOR_DIR="${SOLARSINDY_MONITOR_DIR:-$HOME/.solarsindy/var/monitor}"

command -v git >/dev/null || { echo "error: git not found" >&2; exit 1; }
cd "$SRC_CLONE"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "error: $SRC_CLONE is not a git repo" >&2; exit 1; }
SHA="$(git rev-parse --short "$REVISION")"
FULL_SHA="$(git rev-parse "$REVISION")"
REL_DIR="$RELEASES_ROOT/$SHA"

echo "source     = $SRC_CLONE"
echo "revision   = $REVISION ($FULL_SHA)"
echo "release    = $REL_DIR"
echo "state dir  = $MONITOR_DIR"

mkdir -p "$REL_DIR" "$MONITOR_DIR/logs"
# Clean export of committed content only (no uncommitted/working-tree files).
git archive "$REVISION" | tar -x -C "$REL_DIR"
# Record provenance in the release for auditability.
printf '{"revision":"%s","full_sha":"%s","deployed_utc":"%s","source":"%s"}\n' \
  "$REVISION" "$FULL_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SRC_CLONE" \
  > "$REL_DIR/RELEASE.json"

if [ "${SOLARSINDY_INSTANTIATE:-1}" = "1" ]; then
  JULIA_BIN="${SOLARSINDY_JULIA:-$HOME/.juliaup/bin/julia}"
  [ -x "$JULIA_BIN" ] || JULIA_BIN="$(command -v julia || true)"
  if [ -n "$JULIA_BIN" ]; then
    echo "instantiating $REL_DIR ..."
    JULIA_NUM_THREADS=2 "$JULIA_BIN" --startup-file=no --project="$REL_DIR" -e 'using Pkg; Pkg.instantiate()'
    JULIA_NUM_THREADS=2 "$JULIA_BIN" --startup-file=no --project="$REL_DIR/app" -e 'using Pkg; Pkg.instantiate()' || true
  fi
fi

ln -sfn "$REL_DIR" "$RELEASES_ROOT/current"
echo "current -> $REL_DIR"

if [ "${SOLARSINDY_ACTIVATE:-0}" = "1" ]; then
  echo "activating launchd services against the release ..."
  SOLARSINDY_MONITOR_DIR="$MONITOR_DIR" "$REL_DIR/deploy/install_launchd.sh" "$REL_DIR"
else
  echo
  echo "Release staged (not activated). To point the services at it:"
  echo "  SOLARSINDY_MONITOR_DIR='$MONITOR_DIR' '$REL_DIR/deploy/install_launchd.sh' '$REL_DIR'"
fi
