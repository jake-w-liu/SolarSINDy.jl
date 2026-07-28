# Configurable threshold-based geomagnetic storm alarm system

"""
    StormSeverity

Geomagnetic storm severity levels based on Dst* thresholds.
"""
@enum StormSeverity QUIET MODERATE INTENSE SUPERINTENSE

@doc """
    QUIET

Storm-severity category used when no configured storm threshold is crossed.
""" QUIET

@doc """
    MODERATE

First nonquiet storm-severity category. The default alarm threshold is
`Dst* = -50 nT`.
""" MODERATE

@doc """
    INTENSE

Intermediate storm-severity category. The default alarm threshold is
`Dst* = -100 nT`.
""" INTENSE

@doc """
    SUPERINTENSE

Highest storm-severity category. The default alarm threshold is
`Dst* = -200 nT`.
""" SUPERINTENSE

"""
    Alarm

Triggered alarm with severity, predicted Dst*, and context.
"""
struct Alarm
    time::DateTime
    severity::StormSeverity
    dst_predicted::Float64
    dst_ci_05::Float64   # lower edge of the 90% predictive band (see AlarmConfig note); a
                         # conservative deep-Dst screen, NOT a calibrated one-sided 5% bound
    message::String
end

"""
    AlarmConfig

Configuration for the alarm system.

- `thresholds`: Dst* thresholds for each severity level [nT]
- `use_worst_case`: if true, alarm on the lower edge of the forecast's 90% predictive band
  (`dst_ci_05`) rather than the point prediction (`dst_predicted`). This is a deliberately
  conservative deep-Dst screen. It is NOT a calibrated one-sided 5%-risk bound: depending on
  the interval source, `dst_ci_05` is either the coefficient-ensemble 5th percentile
  (parameter uncertainty only, no residual-error term) or the lower endpoint of a symmetric
  two-sided 90% conformal/ACI band (two-sided coverage control only). Storm-time residuals are
  asymmetric (the model tends to underpredict deep Dst), so the one-sided exceedance
  `P(observed < dst_ci_05)` during storms can exceed 5%. Treat a `use_worst_case` trigger as a
  low-threshold conservative screen, not a guaranteed 95% no-exceedance statement.
- `callback`: function called when alarm triggers (receives Alarm)
- `cooldown_hours`: suppress repeated alarms within this window (a strict escalation to a more
  severe tier bypasses the cooldown; see [`check_alarm`](@ref))
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
        true,   # conservative screen: alarm on the 90% band's lower edge (dst_ci_05), not a
                # calibrated one-sided 5% bound — see AlarmConfig docstring
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
    AlarmCooldownState(time, severity)

Cooldown bookkeeping for [`check_alarm`](@ref): the timestamp of the alarm that armed the
current cooldown window and the severity it fired at. Threaded by the caller in place of a bare
`DateTime`, so an escalation to a strictly more severe tier can bypass the cooldown. A bare
`DateTime` is still accepted (legacy callers) and is treated as a maximally severe armed state,
which reproduces the previous suppress-every-repeat-within-cooldown behavior exactly.
"""
struct AlarmCooldownState
    time::DateTime
    severity::StormSeverity
end

"""
    check_alarm(config, result, last)

Check if an alarm should trigger. `last` is an [`AlarmCooldownState`](@ref) (or a `DateTime`,
treated as `AlarmCooldownState(time, SUPERINTENSE)`). Returns
`(Union{Nothing,Alarm}, AlarmCooldownState)`; thread the returned state into the next call.

The cooldown suppresses repeats within `cooldown_hours`, EXCEPT a strict escalation to a more
severe tier, which re-announces immediately (mirroring [`maybe_fire_horizon_alarm!`](@ref)). This
prevents a storm that deepens from MODERATE to INTENSE/SUPERINTENSE inside the cooldown window
from being silently swallowed — the exact escalation the alert program exists to deliver.
"""
function check_alarm(config::AlarmConfig, result::ForecastResult,
                     last::AlarmCooldownState)
    dst_check = config.use_worst_case ? result.dst_ci_05 : result.dst_predicted
    severity = classify_severity(dst_check, config.thresholds)

    severity == QUIET && return (nothing, last)

    # Check cooldown. Guard the lower bound so a future-dated armed time (e.g. set by a
    # forecast-horizon alarm whose result.t is ahead of wall clock) cannot make `elapsed`
    # negative and silently suppress a genuine present-time alarm. The upper bound is closed
    # (`<=`): a cooldown of N hours suppresses repeats up to and including exactly N hours apart,
    # so a target that sits precisely `cooldown_hours` from the last alarm does not re-fire on
    # every poll cycle.
    elapsed = result.t - last.time
    within_cooldown = Millisecond(0) <= elapsed <= Hour(config.cooldown_hours)
    # Escalation bypass: inside the cooldown window, suppress only when the new severity is not
    # strictly greater than the severity that armed the cooldown. A strict escalation re-arms the
    # cooldown at the higher tier and fires now.
    if within_cooldown && severity <= last.severity
        return (nothing, last)
    end

    msg = _alarm_message(severity, result)
    alarm = Alarm(result.t, severity, result.dst_predicted, result.dst_ci_05, msg)

    config.callback(alarm)
    return (alarm, AlarmCooldownState(result.t, severity))
end

# Legacy entry point: a bare DateTime arms the cooldown at the maximum severity, so no escalation
# can bypass it and the behavior is identical to the pre-escalation implementation. Callers that
# thread the returned AlarmCooldownState automatically gain the escalation bypass.
check_alarm(config::AlarmConfig, result::ForecastResult, last_alarm_time::DateTime) =
    check_alarm(config, result, AlarmCooldownState(last_alarm_time, SUPERINTENSE))

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
