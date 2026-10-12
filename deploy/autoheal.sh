#!/bin/sh
# Restart-on-unhealthy sidecar for deploy/docker-compose.yml (service `autoheal`).
#
# Plain `docker compose` restart policies (`restart: always`) fire only when a container's process
# EXITS; an `unhealthy` HEALTHCHECK status merely flips state. This loop closes that gap: every
# AUTOHEAL_INTERVAL_SEC seconds (default 60) it asks the Docker daemon for containers that carry
# the opt-in label AUTOHEAL_LABEL (default com.solarsindy.autoheal=true) AND are health=unhealthy,
# scoped to this compose project (com.docker.compose.project) when that can be determined, and
# runs `docker restart -t 30` on each, logging every action.
#
# Docker's own health semantics decide "unhealthy" (start-period, interval, retries from the
# image HEALTHCHECK), so a container still in its start period is `starting`, never matched.
#
# Environment:
#   AUTOHEAL_INTERVAL_SEC  seconds between passes (positive integer; default 60)
#   AUTOHEAL_LABEL         opt-in label filter (default com.solarsindy.autoheal=true)
#   AUTOHEAL_PROJECT       compose project to scope to; default: read from this container's own
#                          com.docker.compose.project label (via $HOSTNAME = container id)
#   AUTOHEAL_ONCE=1        run a single pass and exit (tests)
#
# A failing docker call is logged and the loop carries on. POSIX sh (busybox ash compatible).
#
# SECURITY: this needs the Docker socket, and access to the Docker socket is root-equivalent on
# the host (a `:ro` bind mount does not restrict the API). Only labelled containers are touched,
# but the script is trusted code. Drop the sidecar when an orchestrator already restarts
# unhealthy containers.

set -u

LABEL="${AUTOHEAL_LABEL:-com.solarsindy.autoheal=true}"
INTERVAL="${AUTOHEAL_INTERVAL_SEC:-60}"
case "$INTERVAL" in
    ''|*[!0-9]*|0) INTERVAL=60 ;;
esac
PROJECT="${AUTOHEAL_PROJECT:-}"

log() {
    printf '%s autoheal: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

resolve_project() {
    [ -n "$PROJECT" ] && return 0
    [ -n "${HOSTNAME:-}" ] || return 0
    p=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project" }}' \
        "$HOSTNAME" 2>/dev/null) || p=""
    case "$p" in
        ''|'<no value>') ;;
        *) PROJECT="$p"; log "scoping to compose project '$PROJECT'" ;;
    esac
}

heal_once() {
    resolve_project
    if [ -n "$PROJECT" ]; then
        ids=$(docker ps -q --filter "label=$LABEL" \
              --filter "label=com.docker.compose.project=$PROJECT" \
              --filter "health=unhealthy") || { log "docker ps failed (status $?); retrying next pass"; return 0; }
    else
        ids=$(docker ps -q --filter "label=$LABEL" \
              --filter "health=unhealthy") || { log "docker ps failed (status $?); retrying next pass"; return 0; }
    fi
    for id in $ids; do
        case "$id" in
            *[!0-9a-f]*) log "ignoring unexpected id '$id'"; continue ;;
        esac
        name=$(docker inspect --format '{{ .Name }}' "$id" 2>/dev/null) || name="?"
        log "container $id ($name) is unhealthy; restarting"
        if docker restart -t 30 "$id" >/dev/null; then
            log "restarted $id ($name)"
        else
            log "restart of $id ($name) failed (status $?); will retry next pass"
        fi
    done
    return 0
}

trap 'log "stopping"; exit 0' TERM INT

log "started: label=$LABEL interval=${INTERVAL}s"
while :; do
    heal_once
    [ "${AUTOHEAL_ONCE:-0}" = 1 ] && exit 0
    sleep "$INTERVAL" &
    wait $! || true
done
