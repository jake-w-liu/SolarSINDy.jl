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
# Configuration propagation: a launchd job inherits nothing from an interactive shell, so every
# setting a service needs has to be rendered into its plist. This script reads the allow-listed
# solarsindy.env keys from its own environment and renders each into the services that actually read
# it (see service_env_keys); a set key that no selected service reads is reported at install time
# instead of being silently dropped. `bin/solarsindy install-service` sources solarsindy.env before
# exec-ing this script, so the file's keys are already exported here; a direct invocation should
# source the file first (`set -a; . ./solarsindy.env; set +a`) or export the keys.
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

# Rendering scratch space. Under `set -euo pipefail` any failure between the first write and the
# final rename aborts the script immediately, so intermediates must not live in ~/Library/LaunchAgents
# where launchd and the operator both read: a failed `awk` or `plutil -lint` would otherwise leave
# half-rendered `.env`/`.pre`/`.tmp` files sitting beside the real plists. Everything is staged here
# and removed by the EXIT trap on success and on failure alike. Only the atomic install rename
# touches LA_DIR, and its in-flight name is registered so the trap can remove that too.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/solarsindy-launchd.XXXXXX")"
STAGED_PLIST=""
cleanup_render_temporaries() {
  rm -rf "$WORK_DIR"
  [ -z "$STAGED_PLIST" ] || rm -f "$STAGED_PLIST"
}
trap cleanup_render_temporaries EXIT

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

xml_escape() {
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    -e "s/'/\&apos;/g"
}

xml_sed_escape() {
  # XML-escape element text, then escape sed replacement metacharacters.
  printf '%s' "$(xml_escape "$1")" | sed -e 's/[\\&#]/\\&/g'
}

# Allow-listed solarsindy.env keys each installed service actually reads, beyond the ones that have
# their own placeholder (paths, org label, SWM_HOST/SWM_PORT, the watchdog URL and staleness).
# Keeping this per service means a monitor-only key is not rendered into the dashboard job, and a
# key absent from every list is reported rather than silently ignored.
service_env_keys() {
  case "$1" in
    monitor)
      printf '%s\n' \
        SOLARSINDY_V2_CALIBRATION \
        LIVE_MONITOR_INTERVAL_SEC \
        LIVE_MONITOR_DEADMAN_CYCLES \
        LIVE_MONITOR_MAX_LOG_ROWS \
        LIVE_MONITOR_LOG_MAX_BYTES \
        LIVE_MONITOR_LOG_MAX_FILES \
        LIVE_FUTURE_CLOCK_TOLERANCE_MIN \
        LIVE_MAX_FUTURE_CLOCK_SKEW_MIN \
        SOLARSINDY_V2_4_DEPLOY_DIR \
        SOLARSINDY_V2_3_SHADOW_DIR \
        SOLARSINDY_V2_2_STACK \
        SOLARSINDY_V2_2_STACK_SHA256 \
        SOLARSINDY_ALLOW_UNPINNED_STACK
      ;;
    dashboard)
      printf '%s\n' SWM_WEBHOOK_URL
      ;;
    watchdog)
      printf '%s\n' \
        SWM_WEBHOOK_URL \
        SOLARSINDY_WATCHDOG_DATA_URL \
        SOLARSINDY_WATCHDOG_DATA_TIMEOUT \
        SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES
      ;;
    collector) : ;;
    *) return 1 ;;
  esac
}

# Keys the CLI documents in solarsindy.env. Anything here that is set but reaches no selected
# service is named at install time (some are CLI-only by design).
KNOWN_CONFIG_KEYS="SWM_HOST SWM_PORT SWM_WEBHOOK_URL SOLARSINDY_MONITOR_DIR
SOLARSINDY_V2_CALIBRATION LIVE_MONITOR_INTERVAL_SEC LIVE_MONITOR_DEADMAN_CYCLES
LIVE_MONITOR_MAX_LOG_ROWS LIVE_MONITOR_LOG_MAX_BYTES LIVE_MONITOR_LOG_MAX_FILES
LIVE_FUTURE_CLOCK_TOLERANCE_MIN LIVE_MAX_FUTURE_CLOCK_SKEW_MIN
SOLARSINDY_V2_2_STACK SOLARSINDY_V2_2_STACK_SHA256 SOLARSINDY_ALLOW_UNPINNED_STACK
SOLARSINDY_V2_4_DEPLOY_DIR SOLARSINDY_V2_3_SHADOW_DIR SOLARSINDY_WATCHDOG_STALE_SEC
SOLARSINDY_WATCHDOG_DASH_URL SOLARSINDY_WATCHDOG_DATA_URL SOLARSINDY_WATCHDOG_DATA_TIMEOUT
SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES SOLARSINDY_V22_RECEIPT_DIR SOLARSINDY_V22_RECEIPT_LOG_DIR
JULIA SOLARSINDY_JULIA_THREADS SOLARSINDY_NO_OPEN SOLARSINDY_IGNORE_SERVICE"

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

SWM_HOST_VALUE="${SWM_HOST:-127.0.0.1}"
SWM_PORT_VALUE="${SWM_PORT:-8723}"
case "$SWM_PORT_VALUE" in
  ''|*[!0-9]*) echo "error: SWM_PORT must be a port number, got '$SWM_PORT_VALUE'" >&2; exit 1 ;;
esac
[ "$SWM_PORT_VALUE" -ge 1 ] && [ "$SWM_PORT_VALUE" -le 65535 ] || {
  echo "error: SWM_PORT must be between 1 and 65535, got '$SWM_PORT_VALUE'" >&2; exit 1
}
# 0.0.0.0 / :: are bind addresses, not reachable probe destinations: the watchdog health check has
# to go over loopback or it reports a healthy dashboard as down.
PROBE_HOST="$SWM_HOST_VALUE"
case "$SWM_HOST_VALUE" in
  0.0.0.0|::|'[::]') PROBE_HOST="127.0.0.1" ;;
esac
WATCHDOG_DASH_URL_VALUE="${SOLARSINDY_WATCHDOG_DASH_URL:-http://$PROBE_HOST:$SWM_PORT_VALUE/api/health}"
WATCHDOG_STALE_SEC_VALUE="${SOLARSINDY_WATCHDOG_STALE_SEC:-7200}"
case "$WATCHDOG_STALE_SEC_VALUE" in
  ''|*[!0-9]*) echo "error: SOLARSINDY_WATCHDOG_STALE_SEC must be a whole number of seconds, got '$WATCHDOG_STALE_SEC_VALUE'" >&2; exit 1 ;;
esac

# The live engine's source-clock skew band. The daemon treats a malformed value as a warning and
# falls back to its documented default, because a forecast daemon that refuses to start is a worse
# outage than one running on the default band. Install time is where a typo should be fatal instead:
# the operator is present, the message is read, and the plist that would carry the bad value has not
# been written yet.
LIVE_CLOCK_TOLERANCE_MIN_DEFAULT=2
LIVE_CLOCK_SKEW_MIN_DEFAULT=15
for minutes_key in LIVE_FUTURE_CLOCK_TOLERANCE_MIN LIVE_MAX_FUTURE_CLOCK_SKEW_MIN; do
  [ -n "${!minutes_key+x}" ] || continue
  case "${!minutes_key}" in
    ''|*[!0-9]*)
      echo "error: $minutes_key must be a nonnegative whole number of minutes, got '${!minutes_key}'" >&2
      exit 1
      ;;
  esac
done
LIVE_CLOCK_TOLERANCE_CHECK="${LIVE_FUTURE_CLOCK_TOLERANCE_MIN:-$LIVE_CLOCK_TOLERANCE_MIN_DEFAULT}"
LIVE_CLOCK_SKEW_CHECK="${LIVE_MAX_FUTURE_CLOCK_SKEW_MIN:-$LIVE_CLOCK_SKEW_MIN_DEFAULT}"
[ "$LIVE_CLOCK_SKEW_CHECK" -ge "$LIVE_CLOCK_TOLERANCE_CHECK" ] || {
  echo "error: LIVE_MAX_FUTURE_CLOCK_SKEW_MIN ($LIVE_CLOCK_SKEW_CHECK) must be at least" >&2
  echo "error: LIVE_FUTURE_CLOCK_TOLERANCE_MIN ($LIVE_CLOCK_TOLERANCE_CHECK)" >&2
  exit 1
}

for input_name in JULIA_BIN CLONE_DIR APP_DIR MONITOR_DIR RECEIPT_DIR RECEIPT_LOG_DIR ORG \
                  SWM_HOST_VALUE SWM_PORT_VALUE WATCHDOG_DASH_URL_VALUE WATCHDOG_STALE_SEC_VALUE; do
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

# Values of the allow-listed keys must be plist-safe before anything is written.
for svc in "${SERVICES[@]}"; do
  while IFS= read -r env_key; do
    [ -n "$env_key" ] || continue
    [ -n "${!env_key+x}" ] || continue
    reject_control_input "$env_key" "${!env_key}"
  done <<EOF
$(service_env_keys "$svc")
EOF
done

# Report every documented key that is set here but reaches no selected service, so the config file
# cannot appear to configure a daemon that never sees it.
UNAPPLIED_KEYS=""
for known_key in $KNOWN_CONFIG_KEYS; do
  [ -n "${!known_key+x}" ] || continue
  case "$known_key" in
    SOLARSINDY_MONITOR_DIR|SOLARSINDY_V22_RECEIPT_DIR|SOLARSINDY_V22_RECEIPT_LOG_DIR) continue ;;
  esac
  key_applied=0
  for svc in "${SERVICES[@]}"; do
    case "$svc" in
      dashboard|watchdog)
        case "$known_key" in
          SWM_HOST|SWM_PORT) key_applied=1 ;;
        esac
        ;;
    esac
    case "$svc" in
      watchdog)
        case "$known_key" in
          SOLARSINDY_WATCHDOG_DASH_URL|SOLARSINDY_WATCHDOG_STALE_SEC) key_applied=1 ;;
        esac
        ;;
    esac
    while IFS= read -r env_key; do
      [ "$env_key" = "$known_key" ] && key_applied=1
    done <<EOF
$(service_env_keys "$svc")
EOF
  done
  [ "$key_applied" -eq 1 ] || UNAPPLIED_KEYS="$UNAPPLIED_KEYS $known_key"
done
if [ -n "$UNAPPLIED_KEYS" ]; then
  echo "warning: these configuration keys are set but are not rendered into the selected service(s)" >&2
  echo "warning: and therefore do not apply to them (JULIA, SOLARSINDY_JULIA_THREADS," >&2
  echo "warning: SOLARSINDY_NO_OPEN and SOLARSINDY_IGNORE_SERVICE are bin/solarsindy-only by design):" >&2
  for unapplied_key in $UNAPPLIED_KEYS; do
    echo "warning:   $unapplied_key" >&2
  done
fi

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
SWM_HOST_RENDER="$(xml_sed_escape "$SWM_HOST_VALUE")"
SWM_PORT_RENDER="$(xml_sed_escape "$SWM_PORT_VALUE")"
WATCHDOG_DASH_URL_RENDER="$(xml_sed_escape "$WATCHDOG_DASH_URL_VALUE")"
WATCHDOG_STALE_SEC_RENDER="$(xml_sed_escape "$WATCHDOG_STALE_SEC_VALUE")"

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

# Write the <key>/<string> pairs for one service's allow-listed keys to $2. Only keys that are
# actually set in this environment are emitted, so an unset key leaves the daemon on its own
# documented default instead of being pinned to an empty string.
write_extra_env_block() {
  local svc="$1" out="$2" env_key
  : > "$out"
  while IFS= read -r env_key; do
    [ -n "$env_key" ] || continue
    [ -n "${!env_key+x}" ] || continue
    printf '    <key>%s</key>\n' "$(xml_escape "$env_key")" >> "$out"
    printf '    <string>%s</string>\n' "$(xml_escape "${!env_key}")" >> "$out"
    echo "  $svc env: $env_key"
  done <<EOF
$(service_env_keys "$svc")
EOF
}

render() {
  # $1 = service short name; maps to the template + installed label suffix.
  local svc="$1" suffix tmpl label dst env_block pre staged
  suffix="$(service_suffix "$svc")"
  tmpl="$SELF_DIR/com.example.solarsindy.$suffix.plist"
  label="com.$ORG.solarsindy.$suffix"
  dst="$LA_DIR/$label.plist"
  env_block="$WORK_DIR/$label.env"
  pre="$WORK_DIR/$label.pre"
  staged="$WORK_DIR/$label.plist"
  write_extra_env_block "$svc" "$env_block"
  # The template's header comment is documentation for the manual procedure: it spells every
  # placeholder in prose and shows the example label. Substituting inside it produced an installed
  # plist whose "every one of them must be replaced" list had already been replaced, so the file
  # documented values instead of the contract. Only the body from `<plist` onward is substituted;
  # the rendered artifact gets a fixed provenance header naming the template it came from.
  local body_start
  body_start="$(grep -n '<plist' "$tmpl" | head -1 | cut -d: -f1)"
  tail -n "+$body_start" "$tmpl" > "$WORK_DIR/$label.body"
  sed -e "s#__JULIA_BIN__#$JULIA_RENDER#g" \
      -e "s#__CLONE_DIR__#$CLONE_RENDER#g" \
      -e "s#__APP_DIR__#$APP_RENDER#g" \
      -e "s#__MONITOR_DIR__#$MONITOR_RENDER#g" \
      -e "s#__RECEIPT_DIR__#$RECEIPT_RENDER#g" \
      -e "s#__RECEIPT_LOG_DIR__#$RECEIPT_LOG_RENDER#g" \
      -e "s#__SWM_HOST__#$SWM_HOST_RENDER#g" \
      -e "s#__SWM_PORT__#$SWM_PORT_RENDER#g" \
      -e "s#__WATCHDOG_DASH_URL__#$WATCHDOG_DASH_URL_RENDER#g" \
      -e "s#__WATCHDOG_STALE_SEC__#$WATCHDOG_STALE_SEC_RENDER#g" \
      -e "s#com\.example\.solarsindy#$ORG_PREFIX_RENDER#g" \
      "$WORK_DIR/$label.body" > "$pre"
  # Splice the multi-line env block in place of its marker with awk: sed replacements cannot carry
  # newlines portably, and the marker must disappear even when the block is empty. The match is the
  # whole marker LINE, not a substring: the header comment documents the marker by name for the
  # manual procedure, and a substring match spliced the env keys into that sentence and deleted the
  # documentation with it.
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"\n'
    printf ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    printf '%s\n' '<!--'
    printf '  Rendered by deploy/install_launchd.sh from deploy/com.example.solarsindy.%s.plist.\n' "$suffix"
    printf '  Do not edit in place: re-render from that template, whose own header states the\n'
    printf '  placeholder contract and the manual procedure.\n'
    printf '%s\n' '-->'
    awk -v block="$env_block" '
      /^[[:space:]]*<!--[[:space:]]*__EXTRA_ENV__[[:space:]]*-->[[:space:]]*$/ {
        while ((getline line < block) > 0) print line
        close(block)
        next
      }
      { print }
    ' "$pre"
  } > "$staged"
  rm -f "$pre" "$env_block" "$WORK_DIR/$label.body"
  plutil -lint "$staged" >/dev/null
  # A rendered plist that still carries a placeholder would install a service whose environment is
  # the literal marker; the daemons reject that at startup, but an install-time failure names the
  # missing substitution instead of leaving a crash loop to diagnose. Only the plist body is checked:
  # the header comment documents the placeholder names for the manual procedure and legitimately
  # spells `__PLACEHOLDER__` in prose.
  local leftovers
  leftovers="$(sed -n '/<plist/,$p' "$staged" | LC_ALL=C grep -o '__[A-Za-z0-9_]*__' | sort -u || true)"
  if [ -n "$leftovers" ]; then
    echo "error: $tmpl left unrendered placeholders in the rendered plist body:" >&2
    printf '%s\n' "$leftovers" | sed 's/^/error:   /' >&2
    exit 1
  fi
  # Atomic install: the rename has to happen inside LA_DIR so a reader never observes a partial
  # plist. The staged copy is registered before it is created, so the EXIT trap removes it if
  # anything between the copy and the rename fails.
  STAGED_PLIST="$dst.tmp"
  cp "$staged" "$STAGED_PLIST"
  mv "$STAGED_PLIST" "$dst"
  STAGED_PLIST=""
  rm -f "$staged"
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
echo "dashboard_bind=$SWM_HOST_VALUE:$SWM_PORT_VALUE"
echo "watchdog_health_url=$WATCHDOG_DASH_URL_VALUE"
echo "org=$ORG  services=${SERVICES[*]}  load=$LOAD_SERVICES"
for svc in "${SERVICES[@]}"; do
  render "$svc"
done
[ "$LOAD_SERVICES" -eq 0 ] || for svc in "${SERVICES[@]}"; do
  load_service "$svc"
done
echo "done"
