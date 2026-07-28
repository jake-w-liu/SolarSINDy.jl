#!/usr/bin/env bash
# External liveness watchdog for the SolarSINDy live monitor + dashboard.
#
# Runs as a one-shot on a launchd StartInterval schedule (see deploy/com.example.solarsindy.
# watchdog.plist), independent of both supervised processes. It detects a dead or unloaded forecast
# daemon (stale forecast log) and a down dashboard/alerting server (unreachable health endpoint) —
# the exact failure that previously went unnoticed for four days — and surfaces it two ways:
#   1. writes/refreshes the OUTAGE.md sentinel beside the forecast log so the dashboard API shows the
#      outage even when the daemon is not running (nothing else writes the sentinel on process death);
#   2. POSTs the existing webhook (Slack/Discord/generic) on a state change, so push alerting does
#      not depend on the dashboard process being alive.
# It also size-bounds the launchd console-capture files so no diagnostic stream grows without limit.
#
# Dependencies: bash, curl, stat, date — no Julia startup per probe, no new packages.
#
# Env (all optional except the monitor dir default):
#   SOLARSINDY_MONITOR_DIR        monitor state dir (default: <clone>/var/monitor)
#   SOLARSINDY_WATCHDOG_STALE_SEC forecast-log staleness threshold (default 7200 = 2x hourly cadence)
#   SOLARSINDY_WATCHDOG_DASH_URL  dashboard health URL (default http://127.0.0.1:8723/api/health)
#   SWM_WEBHOOK_URL               webhook (empty = no push, sentinel still written)
#   SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES  rotate launchd .out/.err over this size (default 2097152)
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MONITOR_DIR="$(cd "$SELF_DIR/.." && pwd)/var/monitor"
MONITOR_DIR="${SOLARSINDY_MONITOR_DIR:-$DEFAULT_MONITOR_DIR}"
LOG="$MONITOR_DIR/live_forecast_log.csv"
SENTINEL="$MONITOR_DIR/OUTAGE.md"
LOGS_DIR="$MONITOR_DIR/logs"
WD_LOG="$LOGS_DIR/watchdog.log"
STATE="$LOGS_DIR/watchdog_state"
STALE_SEC="${SOLARSINDY_WATCHDOG_STALE_SEC:-7200}"
DASH_URL="${SOLARSINDY_WATCHDOG_DASH_URL:-http://127.0.0.1:8723/api/health}"
WEBHOOK="${SWM_WEBHOOK_URL:-}"
STREAM_MAX_BYTES="${SOLARSINDY_WATCHDOG_STREAM_MAX_BYTES:-2097152}"

mkdir -p "$LOGS_DIR"

now="$(date -u +%s)"
iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

_size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1" 2>/dev/null || echo 0; }
_mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null || echo 0; }

wd_log() {
  # Bounded watchdog log: trim to the last 500 lines when it exceeds ~256 KB.
  printf '%s  %s\n' "$iso" "$1" >> "$WD_LOG" 2>/dev/null || true
  if [ "$(_size "$WD_LOG")" -gt 262144 ]; then
    tail -n 500 "$WD_LOG" > "$WD_LOG.tmp" 2>/dev/null && mv "$WD_LOG.tmp" "$WD_LOG" 2>/dev/null || true
  fi
}

# Size-bound the launchd console-capture files (monitor + dashboard) so no stream grows without
# limit between restarts. O_APPEND makes in-place truncation safe (next append lands at offset 0).
for f in "$LOGS_DIR"/*.out "$LOGS_DIR"/*.err; do
  [ -f "$f" ] || continue
  if [ "$(_size "$f")" -gt "$STREAM_MAX_BYTES" ]; then
    cp -f "$f" "$f.1" 2>/dev/null || true
    : > "$f" 2>/dev/null || true
    wd_log "rotated oversized stream $(basename "$f")"
  fi
done

problems=""
kinds=""
# $1 = human message (may carry volatile numbers), $2 = stable kind label used only for
# state-change dedup. Keeping the age out of the dedup key stops the webhook from firing on
# every probe while the same outage persists (the age in $1 grows each probe).
add_problem() {
  problems="${problems:+$problems; }$1"
  kinds="${kinds:+$kinds,}$2"
}

# 1) Forecast daemon liveness via forecast-log freshness.
if [ -f "$LOG" ]; then
  age=$(( now - $(_mtime "$LOG") ))
  if [ "$age" -gt "$STALE_SEC" ]; then
    add_problem "forecast log stale ${age}s (>${STALE_SEC}s): daemon not issuing" "stale"
  fi
else
  add_problem "forecast log missing at $LOG: daemon has never issued" "no_log"
fi

# 2) Dashboard/alerting server liveness via its health endpoint.
if ! curl -fsS -m 8 "$DASH_URL" >/dev/null 2>&1; then
  add_problem "dashboard health endpoint unreachable at $DASH_URL" "dash_down"
fi

prev_state=""
[ -f "$STATE" ] && prev_state="$(cat "$STATE" 2>/dev/null || true)"

post_webhook() {
  # $1 = text summary, $2 = kind. Best-effort; never blocks the schedule.
  [ -n "$WEBHOOK" ] || return 0
  local text="$1" kind="$2"
  local payload
  payload=$(printf '{"text":%s,"kind":%s,"source":"watchdog","time_utc":%s}' \
    "\"$(printf '%s' "$text" | sed 's/\\/\\\\/g; s/"/\\"/g')\"" \
    "\"$kind\"" "\"$iso\"")
  curl -fsS -m 10 -X POST -H 'Content-Type: application/json' -d "$payload" "$WEBHOOK" \
    >/dev/null 2>&1 || wd_log "webhook POST failed"
}

if [ -n "$problems" ]; then
  # Ensure the sentinel exists so the dashboard surfaces the outage. Do NOT clobber a richer
  # daemon-authored dead-man sentinel; only create one when absent.
  if [ ! -f "$SENTINEL" ]; then
    {
      printf '# LIVE FORECAST MONITOR OUTAGE\n\n'
      printf 'Source: external watchdog\n'
      printf 'Detected UTC: %s\n' "$iso"
      printf 'Summary: %s\n\n' "$problems"
      printf 'The out-of-process watchdog detected that the live forecast monitor or its dashboard '
      printf 'is not healthy. This sentinel persists until the watchdog observes recovery.\n'
    } > "$SENTINEL" 2>/dev/null || wd_log "could not write sentinel"
  fi
  # Dedup on stable problem KINDS only, not the human summary: the age embedded in the
  # stale message grows every probe, so keying on it re-POSTed the outage each 900 s cycle.
  sig="PROBLEM: $kinds"
  if [ "$prev_state" != "$sig" ]; then
    wd_log "OUTAGE detected: $problems"
    post_webhook "Space-weather monitor outage: $problems" "outage"
    printf '%s' "$sig" > "$STATE" 2>/dev/null || true
  fi
else
  # Healthy. Clear only a watchdog-authored sentinel; leave a daemon dead-man sentinel to the daemon
  # (a fresh log with a daemon sentinel means issuance is failing while the process is alive, which
  # the coarse liveness checks here cannot see).
  if [ -f "$SENTINEL" ] && grep -q "Source: external watchdog" "$SENTINEL" 2>/dev/null; then
    rm -f "$SENTINEL" 2>/dev/null || true
    wd_log "watchdog-authored sentinel cleared on recovery"
  fi
  if [ -f "$SENTINEL" ]; then
    # A daemon-authored dead-man sentinel is still present: issuance is failing while the
    # process is alive, which these coarse checks cannot see. Do NOT claim full recovery —
    # /api/health keeps reporting the outage. Hold a distinct non-OK state so the eventual
    # sentinel clear still fires exactly one genuine recovery webhook. Log the suppression
    # once per transition, not every probe.
    if [ "$prev_state" != "DAEMON_OUTAGE" ]; then
      wd_log "watchdog probes healthy; daemon-authored issuance outage still active — recovery webhook suppressed"
    fi
    printf 'DAEMON_OUTAGE' > "$STATE" 2>/dev/null || true
  else
    if [ -n "$prev_state" ] && [ "$prev_state" != "OK" ]; then
      wd_log "RECOVERY: monitor + dashboard healthy"
      post_webhook "Space-weather monitor recovered: forecast daemon and dashboard are healthy." "recovery"
    fi
    printf 'OK' > "$STATE" 2>/dev/null || true
  fi
fi

exit 0
