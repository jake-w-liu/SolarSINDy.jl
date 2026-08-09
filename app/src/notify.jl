# notify.jl — alert escalation + outbound webhook for the threat monitor.
#
# Combines the live layers (Dst forecast level, 90%-target-interval watch, SWPC upstream
# indicator, ground-dB/dt threshold band) into one application routing priority, and POSTs to a configured
# webhook ONLY on a level transition (no per-poll spam). Slack/Discord/generic compatible
# (payload carries both a `text` field and the structured fields).
#
# A lost feed or dead forecast cycle is NOT treated as "quiet". Staleness of the primary Dst
# forecast is a first-class alert state ("monitoring data stale — alert state unknown"): the
# webhook never emits a "returned to quiet (all clear)" caused by an outage, and the all-clear
# text is produced only from a fresh cycle that actually shows quiet conditions.
#
# The last delivered state (level + stale flag) is persisted under var/ so a restart across a
# threat escalation still delivers the transition instead of silently re-baselining.
#
# Config: SWM_WEBHOOK_URL (empty = disabled), SWM_ALERT_STATE (optional state-file override).
# Depends on HTTP, JSON3, Dates.

using HTTP, JSON3, Dates

# A stale status (issue time beyond the staleness threshold, or an expired cycle) must not
# escalate the webhook: its threat reflects a forecast issued hours-to-days ago, not now.
_status_stale(status) = hasproperty(status, :stale) && getproperty(status, :stale) == true

# The primary Dst forecast layer is "blind" when a cycle exists but is not fresh (stale,
# expired, or degraded/incomplete feed). This is distinct from "no data yet" (before the first
# write), where build_status returns available=false WITHOUT a forecast_issue_utc. Only the
# blind case must raise the stale/outage alert; a cold start must not.
function _forecast_blind(status)
    getproperty(status, :available) == false || return false
    return hasproperty(status, :forecast_issue_utc) &&
           getproperty(status, :forecast_issue_utc) !== nothing
end

# Overall alert state = max severity across the live layers, with human-readable reasons, plus a
# `stale` flag that is true when the primary forecast feed has gone blind (see `_forecast_blind`).
function compute_alert_state(status, upstream_status, dbdt)
    level = 0; reasons = String[]
    if getproperty(status, :available) == true && !_status_stale(status)
        th = status.threat
        if th.level >= 1; level = max(level, th.level); push!(reasons, "Dst forecast $(th.label)"); end
        if th.watch
            level = max(level, th.watch_level)
            push!(reasons, "a 90% target interval extends into the $(th.watch_label) range")
        end
    end
    if upstream_status !== nothing && getproperty(upstream_status, :available) == true &&
       getproperty(upstream_status, :elevated) == true
        level = max(level, 1); append!(reasons, upstream_status.reasons)
    end
    if dbdt !== nothing && getproperty(dbdt, :available) == true
        dl = dbdt.current_tier.level
        if dl !== nothing && dl >= 1
            # The shared integer is an application notification priority only. It does not
            # physically cross-calibrate a dB/dt threshold band with a Dst storm class.
            level = max(level, dl)
            push!(reasons, "ground dB/dt threshold band $(dbdt.current_tier.label) ($(dbdt.current_dbdt) nT/min)")
        end
    end
    return (level = level, reasons = reasons, stale = _forecast_blind(status))
end

# ---- persistent delivered-state (survives restarts) ------------------------------------
# `_LAST_ALERT_LEVEL` is the last delivered alert level (-1 = uninitialized baseline; do not
# fire on the first evaluation). `_LAST_ALERT_STALE` is whether the last delivered state was the
# stale/outage state. Both are committed only after successful delivery and mirrored to disk.
const _LAST_ALERT_LEVEL = Ref{Int}(-1)
const _LAST_ALERT_STALE = Ref{Bool}(false)
const _ALERT_STATE_LOADED = Ref{Bool}(false)

function reset_notify!()
    _LAST_ALERT_LEVEL[] = -1
    _LAST_ALERT_STALE[] = false
    _ALERT_STATE_LOADED[] = false
    return nothing
end

# Default state-file location: $SWM_ALERT_STATE, else the state.json sibling of the forecast log
# under var/. Empty string disables disk persistence (used by unit tests, which stay in-memory).
function _default_alert_state_path(log_path::AbstractString)
    override = strip(get(ENV, "SWM_ALERT_STATE", ""))
    isempty(override) || return String(override)
    isempty(log_path) && return ""
    return joinpath(dirname(log_path), "alert_notify_state.json")
end

# Load the persisted delivered-state into the in-memory Refs. Best-effort: a missing or corrupt
# file leaves the baseline uninitialized so the first live evaluation re-baselines rather than
# firing spuriously. Idempotent per process (guarded by `_ALERT_STATE_LOADED`).
function _load_alert_state!(state_path::AbstractString)
    _ALERT_STATE_LOADED[] = true
    (isempty(state_path) || !isfile(state_path)) && return nothing
    try
        d = JSON3.read(read(state_path, String))
        lvl = get(d, :level, nothing)
        if lvl isa Integer
            _LAST_ALERT_LEVEL[] = Int(lvl)
            _LAST_ALERT_STALE[] = get(d, :stale, false) == true
        end
    catch e
        e isa InterruptException && rethrow()
        @warn "could not load persisted alert state; re-baselining" state_path
    end
    return nothing
end

# Atomically persist the delivered-state (temp file + rename) so a crash mid-write cannot leave a
# truncated state file that would corrupt the restart baseline.
function _persist_alert_state(state_path::AbstractString, level::Int, stale::Bool)
    isempty(state_path) && return nothing
    try
        dir = dirname(state_path)
        isempty(dir) || isdir(dir) || mkpath(dir)
        tmp = state_path * ".tmp"
        open(tmp, "w") do io
            JSON3.write(io, (level = level, stale = stale,
                             updated_utc = string(now(UTC)) * "Z"))
        end
        mv(tmp, state_path; force = true)
    catch e
        e isa InterruptException && rethrow()
        @warn "could not persist alert state" state_path error_type=string(nameof(typeof(e)))
    end
    return nothing
end

_webhook_host(url) = try
    host = HTTP.URI(url).host
    isempty(host) ? "?" : String(host)
catch e
    e isa InterruptException && rethrow()
    "?"
end

# Injectable transport so the success path is unit-testable without a network. Returns on
# success, throws on delivery failure (2xx enforced via status_exception).
_default_post(url, headers, body) =
    HTTP.post(url, headers, body;
              readtimeout = 10, connect_timeout = 10, retries = 1, status_exception = true)

# Compose the operator-facing message for a delivered transition.
function _alert_message(kind::Symbol, level::Int, reasons::Vector{String}, prev_level::Int)
    joined = isempty(reasons) ? "" : join(reasons, "; ")
    if kind === :stale_onset
        base = "Space-weather monitoring data is STALE — alert state unknown " *
               "(the forecast feed has stopped updating)."
        prev_level >= 1 && (base *= " Last delivered level was L$(prev_level).")
        isempty(joined) || (base *= " Live layers: $(joined).")
        return base
    elseif kind === :stale_escalation
        return "Space-weather alert L$(level) while the forecast feed is STALE: $(joined)"
    elseif kind === :restored
        return level == 0 ?
            "Space-weather monitoring restored; a fresh forecast cycle shows quiet conditions." :
            "Space-weather monitoring restored — alert L$(level): $(joined)"
    elseif kind === :allclear
        return "Space weather returned to quiet (all clear)."
    else # :elevated
        return "Space-weather alert L$(level): $(joined)"
    end
end

"""
    maybe_notify!(state; url, now_utc, state_path, post_fn) -> NamedTuple

Fire the webhook only when the delivered alert state changes. `state` carries `level`, `reasons`,
and (optionally) `stale`. The transition semantics are:

- first evaluation sets a baseline without firing;
- a forecast feed going stale fires a distinct "monitoring data stale" alert, never an all-clear;
- while stale, only a strict escalation of the live-layer level fires again;
- a fresh cycle after staleness fires a "monitoring restored" transition;
- otherwise the webhook fires once per level change, and the all-clear text is emitted only from a
  fresh (non-stale) transition to level 0.

The delivered level+stale is committed (and persisted to `state_path`) only after successful
delivery, so a transient POST failure leaves the transition retryable. Network errors are caught
(never propagate into the caller). `post_fn(url, headers, body)` is injectable for testing.
"""
function maybe_notify!(state; url::AbstractString = get(ENV, "SWM_WEBHOOK_URL", ""),
                       now_utc::AbstractString = "",
                       state_path::AbstractString = "",
                       post_fn = _default_post)
    # Lazily load persisted delivered-state the first time we run against a state file, so a
    # restart compares against the last delivered level instead of re-baselining.
    _ALERT_STATE_LOADED[] || _load_alert_state!(state_path)

    cur_level = state.level
    cur_stale = hasproperty(state, :stale) && getproperty(state, :stale) == true
    prev_level = _LAST_ALERT_LEVEL[]
    prev_stale = _LAST_ALERT_STALE[]

    # commit helper: advance in-memory + persisted delivered-state together.
    commit! = (lvl, stl) -> (_LAST_ALERT_LEVEL[] = lvl; _LAST_ALERT_STALE[] = stl;
                             _persist_alert_state(state_path, lvl, stl))

    if prev_level == -1
        commit!(cur_level, cur_stale)
        return (fired = false, changed = false, level = cur_level, stale = cur_stale,
                reason = "baseline set")
    end

    # Decide whether a transition should be delivered and with what message.
    kind::Symbol = :none
    if cur_stale
        if !prev_stale
            kind = :stale_onset
        elseif cur_level > prev_level
            kind = :stale_escalation      # live layer escalated while the forecast stays blind
        end
        # A stale->stale non-escalation, or any level drop while stale, is NOT an all-clear.
    else
        if prev_stale
            kind = :restored              # fresh cycle after an outage
        elseif cur_level != prev_level
            kind = cur_level == 0 ? :allclear : :elevated
        end
    end

    if kind === :none
        # No delivered transition. Deliberately do NOT mutate the stored level here: when kind is
        # :none the stale flag is unchanged, and the only way the level can differ is a drop below
        # the current episode's high-water while the forecast is stale. Keeping the high-water means
        # a re-crossing of a level already announced during this outage is not re-announced on
        # upstream flapping (mirroring the announce-once/escalate-only horizon-alarm semantics).
        return (fired = false, changed = false, level = cur_level, stale = cur_stale,
                reason = "no change")
    end

    if isempty(url)
        commit!(cur_level, cur_stale)
        return (fired = false, changed = true, level = cur_level, stale = cur_stale,
                reason = "no webhook configured")
    end

    msg = _alert_message(kind, cur_level, state.reasons, prev_level)
    payload = Dict("text" => msg, "level" => cur_level, "reasons" => state.reasons,
                   "stale" => cur_stale, "kind" => string(kind),
                   "previous_level" => prev_level, "time_utc" => now_utc)
    try
        post_fn(url, ["Content-Type" => "application/json"], JSON3.write(payload))
        commit!(cur_level, cur_stale)     # commit only on successful delivery
        return (fired = true, level = cur_level, stale = cur_stale, kind = string(kind),
                previous_level = prev_level, message = msg)
    catch e
        e isa InterruptException && rethrow()
        # Leave the delivered-state at its previous value so the next poll re-attempts this
        # transition rather than swallowing it.
        @warn "alert webhook POST failed" error_type=string(nameof(typeof(e))) webhook_host=_webhook_host(url)
        return (fired = false, changed = true, level = cur_level, stale = cur_stale,
                error = "webhook delivery failed")
    end
end

# Background loop: re-evaluate the alert state every `interval` s and notify on transitions.
# No-op (returns nothing) when no webhook is configured, so it costs nothing by default.
function start_notify_loop(log_path::AbstractString; interval::Int = 300)
    url = get(ENV, "SWM_WEBHOOK_URL", "")
    isempty(url) && return nothing
    host = try
        HTTP.URI(url).host
    catch e
        e isa InterruptException && rethrow()
        "?"
    end
    state_path = _default_alert_state_path(log_path)
    _load_alert_state!(state_path)        # restore delivered-state before the first poll
    @info "alert webhook enabled" url_host=host interval_s=interval state_path=state_path
    return @async begin
        while true
            try
                df = get_log(log_path)
                status = build_status(df)
                snap = swpc_snapshot()
                st = compute_alert_state(
                    status, upstream_assessment(snap),
                    dashboard_dbdt_nowcast(; wait_timeout=USGS_REFRESH_WAIT_S),
                )
                r = maybe_notify!(st; url = url, now_utc = string(now(UTC)) * "Z",
                                  state_path = state_path)
                getproperty(r, :fired) == true &&
                    @info "alert webhook fired" level=st.level stale=st.stale kind=get(r, :kind, "")
            catch e
                e isa InterruptException && rethrow()
                @warn "notify loop iteration failed" exception = (e, catch_backtrace())
            end
            sleep(interval)
        end
    end
end
