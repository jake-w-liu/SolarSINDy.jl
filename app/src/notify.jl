# notify.jl — alert escalation + outbound webhook for the threat monitor.
#
# Combines the three live layers (Dst forecast level, calibrated-band watch, SWPC upstream
# indicator, ground dB/dt tier) into one overall alert level, and POSTs to a configured
# webhook ONLY on a level transition (no per-poll spam). Slack/Discord/generic compatible
# (payload carries both a `text` field and the structured fields).
#
# Config: SWM_WEBHOOK_URL (empty = disabled). Depends on HTTP, JSON3, Dates.

using HTTP, JSON3, Dates

# A stale status (issue time beyond the staleness threshold, or an expired cycle) must not
# escalate the webhook: its threat reflects a forecast issued hours-to-days ago, not now.
_status_stale(status) = hasproperty(status, :stale) && getproperty(status, :stale) == true

# Overall alert state = max severity across the live layers, with human-readable reasons.
function compute_alert_state(status, upstream_status, dbdt)
    level = 0; reasons = String[]
    if getproperty(status, :available) == true && !_status_stale(status)
        th = status.threat
        if th.level >= 1; level = max(level, th.level); push!(reasons, "Dst forecast $(th.label)"); end
        if th.watch; level = max(level, th.watch_level); push!(reasons, "90% band reaches $(th.watch_label)"); end
    end
    if upstream_status !== nothing && getproperty(upstream_status, :available) == true &&
       getproperty(upstream_status, :elevated) == true
        level = max(level, 1); append!(reasons, upstream_status.reasons)
    end
    if dbdt !== nothing && getproperty(dbdt, :available) == true
        dl = dbdt.current_tier.level
        if dl !== nothing && dl >= 1
            level = max(level, dl)
            push!(reasons, "ground dB/dt $(dbdt.current_tier.label) ($(dbdt.current_dbdt) nT/min)")
        end
    end
    return (level = level, reasons = reasons)
end

const _LAST_ALERT_LEVEL = Ref{Int}(-1)        # -1 = uninitialized (baseline, do not fire)
reset_notify!() = (_LAST_ALERT_LEVEL[] = -1; nothing)

_webhook_host(url) = try
    host = HTTP.URI(url).host
    isempty(host) ? "?" : String(host)
catch e
    e isa InterruptException && rethrow()
    "?"
end

"""
    maybe_notify!(state; url, now_utc) -> NamedTuple

Fire the webhook only when the overall alert level changes from the previously recorded
level. The first call sets a baseline without firing. Network errors are caught (never
propagate into the caller). Returns what happened.
"""
function maybe_notify!(state; url::AbstractString = get(ENV, "SWM_WEBHOOK_URL", ""),
                       now_utc::AbstractString = "")
    # Read the previous level WITHOUT committing yet: a transition's level must be recorded
    # only once delivery is resolved, otherwise a transient POST failure advances the stored
    # level and the next poll (same storm level) sees no change and never retries the alert.
    prev = _LAST_ALERT_LEVEL[]
    if prev == -1
        _LAST_ALERT_LEVEL[] = state.level
        return (fired = false, changed = false, level = state.level, reason = "baseline set")
    end
    changed = prev != state.level
    if !changed
        _LAST_ALERT_LEVEL[] = state.level
        return (fired = false, changed = false, level = state.level, reason = "no change")
    end
    if isempty(url)
        _LAST_ALERT_LEVEL[] = state.level
        return (fired = false, changed = true, level = state.level, reason = "no webhook configured")
    end
    msg = state.level == 0 ? "Space weather returned to quiet (all clear)." :
          "Space-weather alert L$(state.level): " * join(state.reasons, "; ")
    payload = Dict("text" => msg, "level" => state.level, "reasons" => state.reasons,
                   "previous_level" => prev, "time_utc" => now_utc)
    try
        HTTP.post(url, ["Content-Type" => "application/json"], JSON3.write(payload);
                  readtimeout = 10, connect_timeout = 10, retries = 1, status_exception = true)
        _LAST_ALERT_LEVEL[] = state.level           # commit only on successful delivery
        return (fired = true, level = state.level, previous_level = prev, message = msg)
    catch e
        e isa InterruptException && rethrow()
        # Leave _LAST_ALERT_LEVEL at prev so the next poll re-attempts this transition.
        @warn "alert webhook POST failed" error_type=string(nameof(typeof(e))) webhook_host=_webhook_host(url)
        return (fired = false, changed = true, level = state.level,
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
    @info "alert webhook enabled" url_host=host interval_s=interval
    return @async begin
        while true
            try
                df = get_log(log_path)
                status = build_status(df)
                snap = swpc_snapshot()
                st = compute_alert_state(status, upstream_assessment(snap), usgs_dbdt())
                r = maybe_notify!(st; url = url, now_utc = string(now(UTC)) * "Z")
                getproperty(r, :fired) == true && @info "alert webhook fired" level=st.level
            catch e
                e isa InterruptException && rethrow()
                @warn "notify loop iteration failed" exception = (e, catch_backtrace())
            end
            sleep(interval)
        end
    end
end
