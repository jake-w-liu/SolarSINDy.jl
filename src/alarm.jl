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
    if dst <= get(thresholds, SUPERINTENSE, -200.0)
        return SUPERINTENSE
    elseif dst <= get(thresholds, INTENSE, -100.0)
        return INTENSE
    elseif dst <= get(thresholds, MODERATE, -50.0)
        return MODERATE
    else
        return QUIET
    end
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

    # Check cooldown
    if result.t - last_alarm_time < Hour(config.cooldown_hours)
        return (nothing, last_alarm_time)
    end

    msg = _alarm_message(severity, result)
    alarm = Alarm(result.t, severity, result.dst_predicted, result.dst_ci_05, msg)

    config.callback(alarm)
    return (alarm, result.t)
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
