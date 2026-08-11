#!/bin/bash
# Render + install the SolarSINDy launchd services from the tracked templates, so the deployed
# ~/Library/LaunchAgents plists are reproducible instead of hand-edited drift.
#
# Usage:
#   deploy/install_launchd.sh [CLONE_DIR] [service ...]
#     CLONE_DIR   package clone root (default: the parent of this script's directory)
#     service     one or more of: monitor dashboard watchdog collector
#                 (default: the three V2.1 services; collector is explicit only)
#
# Env:
#   SOLARSINDY_ORG        reverse-DNS org segment for the installed Label (default: empire)
#   SOLARSINDY_MONITOR_DIR  monitor state dir (default: CLONE_DIR/var/monitor)
#   SOLARSINDY_V22_RECEIPT_DIR  V2.2 immutable receipt root
#   SOLARSINDY_V22_RECEIPT_LOG_DIR  separate collector log directory
#   SOLARSINDY_JULIA      julia launcher (default: ~/.juliaup/bin/julia if present, else `which julia`)
#   SOLARSINDY_LOAD=0     render + install the plists but do not bootstrap them
#                        collector-containing invocations default to 0; set 1 explicitly to start
#
# The installed plists use the STABLE juliaup shim, not a version-pinned juliaup directory, so a
# `juliaup gc`/`update` cannot delete the interpreter out from under the service.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLONE_DIR="$(cd "$SELF_DIR/.." && pwd)"
case "${1:-}" in
  monitor|dashboard|watchdog|collector) CLONE_DIR="$DEFAULT_CLONE_DIR" ;;
  *)
    CLONE_DIR="${1:-$DEFAULT_CLONE_DIR}"
    [ "$#" -eq 0 ] || shift
    ;;
esac
SERVICES=("$@")
[ "${#SERVICES[@]}" -eq 0 ] && SERVICES=(monitor dashboard watchdog)

ORG="${SOLARSINDY_ORG:-empire}"
MONITOR_DIR="${SOLARSINDY_MONITOR_DIR:-$CLONE_DIR/var/monitor}"
RECEIPT_DIR="${SOLARSINDY_V22_RECEIPT_DIR:-$CLONE_DIR/var/v2_2_l1_receipts}"
RECEIPT_LOG_DIR="${SOLARSINDY_V22_RECEIPT_LOG_DIR:-$CLONE_DIR/var/v2_2_l1_logs}"
APP_DIR="$CLONE_DIR/app"
LA_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

reject_control_input() {
  local name="$1" value="$2"
  case "$value" in
    *$'\n'*|*$'\r'*) echo "error: $name contains a newline or carriage return" >&2; exit 1 ;;
  esac
  if printf '%s' "$value" | LC_ALL=C grep '[[:cntrl:]]' >/dev/null; then
    echo "error: $name contains a control character" >&2
    exit 1
  fi
}

xml_sed_escape() {
  # XML-escape element text, then escape sed replacement metacharacters.
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    -e "s/'/\&apos;/g" \
    -e 's/[\\&#]/\\&/g'
}

service_suffix() {
  case "$1" in
    monitor)   printf '%s' "live-monitor" ;;
    dashboard) printf '%s' "dashboard" ;;
    watchdog)  printf '%s' "watchdog" ;;
    collector) printf '%s' "v22-receipt-collector" ;;
    *) return 1 ;;
  esac
}

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

for input_name in JULIA_BIN CLONE_DIR APP_DIR MONITOR_DIR RECEIPT_DIR RECEIPT_LOG_DIR ORG; do
  reject_control_input "$input_name" "${!input_name}"
done
case "$ORG" in
  ''|*[!A-Za-z0-9.-]*) echo "error: SOLARSINDY_ORG must contain only letters, digits, dots, or hyphens" >&2; exit 1 ;;
esac

MONITOR_REQUESTED=0
COLLECTOR_REQUESTED=0
for svc in "${SERVICES[@]}"; do
  if ! suffix="$(service_suffix "$svc")"; then
    echo "error: unknown service '$svc' (want monitor|dashboard|watchdog|collector)" >&2
    exit 1
  fi
  tmpl="$SELF_DIR/com.example.solarsindy.$suffix.plist"
  [ -f "$tmpl" ] || { echo "error: no template $tmpl" >&2; exit 1; }
  case "$svc" in
    collector) COLLECTOR_REQUESTED=1 ;;
    *) MONITOR_REQUESTED=1 ;;
  esac
done

if [ -n "${SOLARSINDY_LOAD+x}" ]; then
  LOAD_SERVICES="$SOLARSINDY_LOAD"
elif [ "$COLLECTOR_REQUESTED" -eq 1 ]; then
  LOAD_SERVICES=0
else
  LOAD_SERVICES=1
fi
case "$LOAD_SERVICES" in
  0|1) ;;
  *) echo "error: SOLARSINDY_LOAD must be 0 or 1" >&2; exit 1 ;;
esac

mkdir -p "$LA_DIR"
[ "$MONITOR_REQUESTED" -eq 0 ] || mkdir -p "$MONITOR_DIR/logs"
[ "$COLLECTOR_REQUESTED" -eq 0 ] || mkdir -p "$RECEIPT_DIR" "$RECEIPT_LOG_DIR"

JULIA_RENDER="$(xml_sed_escape "$JULIA_BIN")"
CLONE_RENDER="$(xml_sed_escape "$CLONE_DIR")"
APP_RENDER="$(xml_sed_escape "$APP_DIR")"
MONITOR_RENDER="$(xml_sed_escape "$MONITOR_DIR")"
RECEIPT_RENDER="$(xml_sed_escape "$RECEIPT_DIR")"
RECEIPT_LOG_RENDER="$(xml_sed_escape "$RECEIPT_LOG_DIR")"
ORG_PREFIX_RENDER="$(xml_sed_escape "com.$ORG.solarsindy")"

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
  suffix="$(service_suffix "$svc")"
  tmpl="$SELF_DIR/com.example.solarsindy.$suffix.plist"
  label="com.$ORG.solarsindy.$suffix"
  dst="$LA_DIR/$label.plist"
  sed -e "s#__JULIA_BIN__#$JULIA_RENDER#g" \
      -e "s#__CLONE_DIR__#$CLONE_RENDER#g" \
      -e "s#__APP_DIR__#$APP_RENDER#g" \
      -e "s#__MONITOR_DIR__#$MONITOR_RENDER#g" \
      -e "s#__RECEIPT_DIR__#$RECEIPT_RENDER#g" \
      -e "s#__RECEIPT_LOG_DIR__#$RECEIPT_LOG_RENDER#g" \
      -e "s#com\.example\.solarsindy#$ORG_PREFIX_RENDER#g" \
      "$tmpl" > "$dst.tmp"
  plutil -lint "$dst.tmp" >/dev/null
  mv "$dst.tmp" "$dst"
  echo "rendered $dst"
}

load_service() {
  local svc="$1" suffix label dst
  suffix="$(service_suffix "$svc")"
  label="com.$ORG.solarsindy.$suffix"
  dst="$LA_DIR/$label.plist"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  bootstrap_service "$label" "$dst"
  launchctl enable "$DOMAIN/$label"
  # RunAtLoad normally starts the new job during bootstrap. A non-killing kickstart also
  # starts a previously disabled job without terminating a healthy process and invoking
  # launchd's restart throttle.
  launchctl kickstart "$DOMAIN/$label"
  echo "bootstrapped + kickstarted $label"
}

echo "clone=$CLONE_DIR"
echo "julia=$JULIA_BIN"
echo "monitor_dir=$MONITOR_DIR"
echo "v22_receipt_dir=$RECEIPT_DIR"
echo "v22_receipt_log_dir=$RECEIPT_LOG_DIR"
echo "org=$ORG  services=${SERVICES[*]}  load=$LOAD_SERVICES"
for svc in "${SERVICES[@]}"; do
  render "$svc"
done
[ "$LOAD_SERVICES" -eq 0 ] || for svc in "${SERVICES[@]}"; do
  load_service "$svc"
done
echo "done"
