# Configurable threshold-based geomagnetic storm alarm system

"""
    StormSeverity

Geomagnetic storm severity levels based on Dst* thresholds.
"""
@enum StormSeverity QUIET MODERATE INTENSE SUPERINTENSE

"""
    Alarm

Triggered alarm with severity, predicted Dst*, and context.
"""
struct Alarm
    time::DateTime
    severity::StormSeverity
    dst_predicted::Float64
    dst_ci_05::Float64
    message::String
end

"""
    AlarmConfig

Configuration for the alarm system.

- `thresholds`: Dst* thresholds for each severity level [nT]
- `use_worst_case`: if true, alarm on 5th percentile instead of median
- `callback`: function called when alarm triggers (receives Alarm)
- `cooldown_hours`: suppress repeated alarms within this window
"""
struct AlarmConfig
    thresholds::Dict{StormSeverity, Float64}
    use_worst_case::Bool
    callback::Function
    cooldown_hours::Int
    function AlarmConfig(thresholds::Dict{StormSeverity,Float64},
                         use_worst_case::Bool, callback::Function,
                         cooldown_hours::Int)
        all(isfinite, values(thresholds)) ||
            throw(ArgumentError("alarm thresholds must be finite"))
        cooldown_hours >= 0 ||
            throw(ArgumentError("cooldown_hours must be nonnegative"))
        return new(copy(thresholds), use_worst_case, callback, cooldown_hours)
    end
end

"""
    default_alarm_config(; callback=alarm_print)

Default alarm configuration with standard NOAA-like thresholds.
"""
function default_alarm_config(; callback::Function=alarm_print)
    return AlarmConfig(
        Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
        true,   # alarm on worst-case (5th percentile)
        callback,
        6,
    )
end

"""
    classify_severity(dst, thresholds)

Classify storm severity from Dst* value.
"""
function classify_severity(dst::Float64,
                           thresholds::Dict{StormSeverity, Float64})
    # A non-finite Dst is a data-quality failure, not a severity level. Reject it
    # so an invalid worst-case forecast cannot be silently interpreted as quiet
    # (NaN) or super-intense (-Inf).
    isfinite(dst) || throw(ArgumentError("Dst* severity input must be finite"))
    all(isfinite, values(thresholds)) ||
        throw(ArgumentError("severity thresholds must be finite"))
    # Upgrade cascade from least to most severe so the result is the most severe tier the
    # value actually crosses. This is identical to an ordered if/elseif for the documented
    # monotone thresholds (SUPERINTENSE ≤ INTENSE ≤ MODERATE) but stays correct (most-severe
    # crossed tier wins) even if a caller supplies a non-monotone custom threshold dict.
    sev = QUIET
    dst <= get(thresholds, MODERATE, -50.0) && (sev = MODERATE)
    dst <= get(thresholds, INTENSE, -100.0) && (sev = INTENSE)
    dst <= get(thresholds, SUPERINTENSE, -200.0) && (sev = SUPERINTENSE)
    return sev
end

"""
    check_alarm(config, result, last_alarm_time)

Check if alarm should trigger. Returns (Union{Nothing,Alarm}, last_alarm_time).
"""
function check_alarm(config::AlarmConfig, result::ForecastResult,
                     last_alarm_time::DateTime)
    dst_check = config.use_worst_case ? result.dst_ci_05 : result.dst_predicted
    severity = classify_severity(dst_check, config.thresholds)

    severity == QUIET && return (nothing, last_alarm_time)

    # Check cooldown. Guard the lower bound so a future-dated last_alarm_time (e.g. set by a
    # forecast-horizon alarm whose result.t is ahead of wall clock) cannot make `elapsed`
    # negative and silently suppress a genuine present-time alarm. The upper bound is closed
    # (`<=`): a cooldown of N hours suppresses repeats up to and including exactly N hours
    # apart, so a target that sits precisely `cooldown_hours` from the last alarm does not
    # re-fire on every subsequent poll cycle.
    elapsed = result.t - last_alarm_time
    if Millisecond(0) <= elapsed <= Hour(config.cooldown_hours)
        return (nothing, last_alarm_time)
    end

    msg = _alarm_message(severity, result)
    alarm = Alarm(result.t, severity, result.dst_predicted, result.dst_ci_05, msg)

    config.callback(alarm)
    return (alarm, result.t)
end

"""
    maybe_fire_horizon_alarm!(config, result, seen)

Fire the alarm callback for a forecast-horizon row only when its target hour has
not already been announced at an equal-or-greater severity. `seen` is a persistent
`Dict{DateTime,StormSeverity}` recording the highest severity already announced per
target hour; it is updated in place. Returns the `Alarm` that fired, or `nothing`.

Unlike [`check_alarm`](@ref), this does not consult the present-time cooldown clock:
horizon rows carry future timestamps, so a `result.t`-based cooldown is meaningless and
would either fire every poll cycle (an alarm storm) or block indefinitely. Deduplicating
on `(target hour, severity)` announces each future crossing once and re-announces only on
escalation to a more severe tier. The caller is responsible for pruning stale `seen`
entries (targets earlier than the current issue time) to bound memory.
"""
function maybe_fire_horizon_alarm!(config::AlarmConfig, result::ForecastResult,
                                   seen::Dict{DateTime,StormSeverity})
    dst_check = config.use_worst_case ? result.dst_ci_05 : result.dst_predicted
    severity = classify_severity(dst_check, config.thresholds)
    severity == QUIET && return nothing
    # Already announced this target hour at an equal-or-greater severity → suppress.
    severity > get(seen, result.t, QUIET) || return nothing
    msg = _alarm_message(severity, result)
    alarm = Alarm(result.t, severity, result.dst_predicted, result.dst_ci_05, msg)
    # Commit the dedup state only after delivery succeeds. If the callback throws
    # (network outage, full disk, etc.), the target remains retryable on the next
    # poll rather than being silently marked as announced.
    config.callback(alarm)
    seen[result.t] = severity
    return alarm
end

function _alarm_message(severity::StormSeverity, result::ForecastResult)
    label = severity == SUPERINTENSE ? "SUPER-INTENSE" :
            severity == INTENSE      ? "INTENSE" : "MODERATE"
    return "[$label STORM] Dst* predicted: $(round(result.dst_predicted, digits=1)) nT " *
           "[CI: $(round(result.dst_ci_05, digits=1)) to $(round(result.dst_ci_95, digits=1)) nT] " *
           "at $(Dates.format(result.t, "yyyy-mm-dd HH:MM")) UTC"
end

# --- Built-in callbacks ---

"""
    alarm_print(alarm)

Print alarm to terminal with severity-appropriate formatting.
"""
function alarm_print(alarm::Alarm)
    prefix = alarm.severity == SUPERINTENSE ? "🔴🔴🔴" :
             alarm.severity == INTENSE      ? "🔴🔴" : "🟡"
    println("\n$prefix $(alarm.message)")
end

"""
    alarm_log(alarm; logfile="storm_alarms.log")

Append alarm to log file.
"""
function alarm_log(alarm::Alarm; logfile::String="storm_alarms.log")
    open(logfile, "a") do io
        println(io, "[$(Dates.format(alarm.time, "yyyy-mm-dd HH:MM:SS"))] $(alarm.message)")
    end
end
