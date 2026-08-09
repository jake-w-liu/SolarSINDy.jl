#!/bin/bash
# Render + install the SolarSINDy launchd services from the tracked templates, so the deployed
# ~/Library/LaunchAgents plists are reproducible instead of hand-edited drift.
#
# Usage:
#   deploy/install_launchd.sh [CLONE_DIR] [service ...]
#     CLONE_DIR   package clone root (default: the parent of this script's directory)
#     service     one or more of: monitor dashboard watchdog  (default: all three)
#
# Env:
#   SOLARSINDY_ORG        reverse-DNS org segment for the installed Label (default: empire)
#   SOLARSINDY_MONITOR_DIR  monitor state dir (default: CLONE_DIR/var/monitor)
#   SOLARSINDY_JULIA      julia launcher (default: ~/.juliaup/bin/julia if present, else `which julia`)
#   SOLARSINDY_LOAD=0     render + install the plists but do not bootstrap them
#
# The installed plists use the STABLE juliaup shim, not a version-pinned juliaup directory, so a
# `juliaup gc`/`update` cannot delete the interpreter out from under the service.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="${1:-$(cd "$SELF_DIR/.." && pwd)}"
shift || true
SERVICES=("$@")
[ "${#SERVICES[@]}" -eq 0 ] && SERVICES=(monitor dashboard watchdog)

ORG="${SOLARSINDY_ORG:-empire}"
MONITOR_DIR="${SOLARSINDY_MONITOR_DIR:-$CLONE_DIR/var/monitor}"
APP_DIR="$CLONE_DIR/app"
LA_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

if [ -n "${SOLARSINDY_JULIA:-}" ]; then
  JULIA_BIN="$SOLARSINDY_JULIA"
elif [ -x "$HOME/.juliaup/bin/julia" ]; then
  JULIA_BIN="$HOME/.juliaup/bin/julia"
else
  JULIA_BIN="$(command -v julia || true)"
fi
[ -n "$JULIA_BIN" ] || { echo "error: no julia launcher found (set SOLARSINDY_JULIA)" >&2; exit 1; }
case "$JULIA_BIN" in
  */.julia/juliaup/julia-*) echo "warning: $JULIA_BIN is a version-pinned juliaup path that juliaup gc can delete; prefer ~/.juliaup/bin/julia" >&2 ;;
esac

mkdir -p "$LA_DIR" "$MONITOR_DIR/logs"

bootstrap_service() {
  local label="$1" dst="$2" attempt output=""
  for attempt in 1 2 3; do
    if output="$(launchctl bootstrap "$DOMAIN" "$dst" 2>&1)"; then
      return 0
    fi
    [ "$attempt" -eq 3 ] || sleep 1
  done
  printf 'error: launchctl bootstrap failed for %s after 3 attempts\n%s\n' \
    "$label" "$output" >&2
  return 1
}

render() {
  # $1 = service short name; maps to the template + installed label suffix.
  local svc="$1" suffix tmpl label dst
  case "$svc" in
    monitor)   suffix="live-monitor" ;;
    dashboard) suffix="dashboard" ;;
    watchdog)  suffix="watchdog" ;;
    *) echo "error: unknown service '$svc'" >&2; exit 1 ;;
  esac
  tmpl="$SELF_DIR/com.example.solarsindy.$suffix.plist"
  label="com.$ORG.solarsindy.$suffix"
  dst="$LA_DIR/$label.plist"
  [ -f "$tmpl" ] || { echo "error: no template $tmpl" >&2; exit 1; }
  sed -e "s#__JULIA_BIN__#$JULIA_BIN#g" \
      -e "s#__CLONE_DIR__#$CLONE_DIR#g" \
      -e "s#__APP_DIR__#$APP_DIR#g" \
      -e "s#__MONITOR_DIR__#$MONITOR_DIR#g" \
      -e "s#com\.example\.solarsindy#com.$ORG.solarsindy#g" \
      "$tmpl" > "$dst.tmp"
  plutil -lint "$dst.tmp" >/dev/null
  mv "$dst.tmp" "$dst"
  echo "rendered $dst"
  if [ "${SOLARSINDY_LOAD:-1}" = "1" ]; then
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    bootstrap_service "$label" "$dst"
    launchctl enable "$DOMAIN/$label"
    # RunAtLoad normally starts the new job during bootstrap. A non-killing kickstart also
    # starts a previously disabled job without terminating a healthy process and invoking
    # launchd's restart throttle.
    launchctl kickstart "$DOMAIN/$label"
    echo "bootstrapped + kickstarted $label"
  fi
}

echo "clone=$CLONE_DIR"
echo "julia=$JULIA_BIN"
echo "monitor_dir=$MONITOR_DIR"
echo "org=$ORG  services=${SERVICES[*]}"
for svc in "${SERVICES[@]}"; do
  case "$svc" in
    monitor|dashboard|watchdog) render "$svc" ;;
    *) echo "error: unknown service '$svc' (want monitor|dashboard|watchdog)" >&2; exit 1 ;;
  esac
done
echo "done"
