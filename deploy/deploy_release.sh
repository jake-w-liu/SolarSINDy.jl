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
#   SOLARSINDY_ENV_FILE       operator config (default: <source clone>/solarsindy.env), loaded like
#                             bin/solarsindy does; variables already set in the environment win
#
# The release is a clean `git archive` export: it contains only committed content at REVISION, never
# the working tree. State (var/monitor) is intentionally not copied; it stays at the canonical dir.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_CLONE="$(cd "$SELF_DIR/.." && pwd)"
# Operator config: load the source clone's solarsindy.env exactly like bin/solarsindy does (env
# vars already set WIN over the file). The git-archive release never contains this gitignored
# file, and install_launchd.sh reads the keys only from its environment, so without this an
# activation silently drops the webhook/port/monitor-dir configuration.
CONFIG="${SOLARSINDY_ENV_FILE:-$SRC_CLONE/solarsindy.env}"
if [ -f "$CONFIG" ]; then
  # Same parser as bin/solarsindy (keep in sync): KEY=VALUE lines only, malformed lines skipped
  # with a warning, CRLF tolerated, reserved __solarsindy_cfg_ loop-variable prefix.
  while IFS= read -r __solarsindy_cfg_line || [ -n "$__solarsindy_cfg_line" ]; do
    __solarsindy_cfg_line="${__solarsindy_cfg_line%$'\r'}"
    case "$__solarsindy_cfg_line" in
      ''|\#*) continue ;;
      [A-Za-z_]*=*) ;;
      *) printf 'warning: ignoring non-KEY=VALUE line in %s: %.40s\n' "$CONFIG" "$__solarsindy_cfg_line" >&2
         continue ;;
    esac
    __solarsindy_cfg_name="${__solarsindy_cfg_line%%=*}"
    case "$__solarsindy_cfg_name" in
      *[!A-Za-z0-9_]*)
        printf 'warning: ignoring invalid variable name in %s: %.40s\n' "$CONFIG" "$__solarsindy_cfg_name" >&2
        continue ;;
      __solarsindy_cfg_*)
        printf 'warning: ignoring reserved variable name in %s: %.40s\n' "$CONFIG" "$__solarsindy_cfg_name" >&2
        continue ;;
    esac
    if eval "[ -n \"\${$__solarsindy_cfg_name+x}\" ]"; then
      continue                              # env wins over the file
    fi
    case "${__solarsindy_cfg_line#*=}" in
      [\"\']*) ;;
      *[[:space:]]*)
        printf 'warning: unquoted whitespace in %s value of %s — quote the value; only the first word applies\n' "$CONFIG" "$__solarsindy_cfg_name" >&2 ;;
    esac
    if "${BASH:-bash}" -nc "export $__solarsindy_cfg_line" 2>/dev/null; then
      eval "export $__solarsindy_cfg_line"
    else
      printf 'warning: skipping malformed line in %s: %s=…\n' "$CONFIG" "$__solarsindy_cfg_name" >&2
    fi
  done < "$CONFIG"
  unset __solarsindy_cfg_line __solarsindy_cfg_name
elif [ -n "${SOLARSINDY_ENV_FILE:-}" ] && [ ! -e "$CONFIG" ]; then
  printf 'warning: SOLARSINDY_ENV_FILE points to a missing file, no config loaded: %s\n' "$CONFIG" >&2
fi
REVISION="${1:-HEAD}"
RELEASES_ROOT="${2:-$HOME/.solarsindy/releases}"
# Canonical absolute path, resolved against the caller's cwd before the `cd "$SRC_CLONE"` below: a
# relative root would otherwise land inside the clone and `ln -sfn` would store a relative target
# that resolves against the link's own directory (releases/releases/<sha>), i.e. a dangling link.
mkdir -p "$RELEASES_ROOT"
RELEASES_ROOT="$(cd "$RELEASES_ROOT" && pwd -P)"
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
  if [ -n "${SOLARSINDY_JULIA:-}" ]; then
    JULIA_BIN="$SOLARSINDY_JULIA"
  elif [ -x "$HOME/.juliaup/bin/julia" ]; then
    JULIA_BIN="$HOME/.juliaup/bin/julia"
  else
    JULIA_BIN="julia"
  fi
  command -v "$JULIA_BIN" >/dev/null 2>&1 || {
    echo "error: Julia not found: $JULIA_BIN" >&2; exit 1
  }
  "$JULIA_BIN" --startup-file=no -e 'exit(VERSION >= v"1.12.6" ? 0 : 1)' || {
    echo "error: Julia 1.12.6 or later is required" >&2; exit 1
  }
  echo "instantiating $REL_DIR ..."
  JULIA_NUM_THREADS=2 "$JULIA_BIN" --startup-file=no --project="$REL_DIR" -e 'using Pkg; Pkg.instantiate()'
  JULIA_NUM_THREADS=2 "$JULIA_BIN" --startup-file=no --project="$REL_DIR/app" -e 'using Pkg; Pkg.instantiate()'
fi

ln -sfn "$REL_DIR" "$RELEASES_ROOT/current"
echo "current -> $REL_DIR"

if [ "${SOLARSINDY_ACTIVATE:-0}" = "1" ]; then
  echo "activating launchd services against the release ..."
  SOLARSINDY_MONITOR_DIR="$MONITOR_DIR" "$REL_DIR/deploy/install_launchd.sh" "$REL_DIR"
else
  echo
  echo "Release staged (not activated). To point the services at it (export the solarsindy.env"
  echo "keys first, or re-run with SOLARSINDY_ACTIVATE=1, which loads $CONFIG):"
  echo "  SOLARSINDY_MONITOR_DIR='$MONITOR_DIR' '$REL_DIR/deploy/install_launchd.sh' '$REL_DIR'"
fi
