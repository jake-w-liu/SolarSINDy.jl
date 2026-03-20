# Real-time storm monitoring loop with display

const BOX_W = 58  # inner width between │ and │

"""
    _boxline(content::String)

Format a single line inside the box, padded to BOX_W.
"""
function _boxline(content::String)
    n = textwidth(content)
    pad = max(0, BOX_W - n)
    return "│ $(content)$(repeat(' ', pad)) │"
end

"""
    _fmt(x::Float64, w::Int=7; digits=1)

Format a float to exactly `w` characters, right-aligned. NaN → "N/A" centered.
"""
function _fmt(x::Float64, w::Int=7; digits::Int=1)
    isnan(x) && return lpad("N/A", w)
    return lpad(string(round(x, digits=digits)), w)
end

"""
    run_monitor(; poll_interval_min=5, forecast_horizon_hr=6,
                  alarm_config, coefficients_csv, ensemble_csv,
                  log_file="monitor.log", display=true)

Main monitoring loop. Fetches real-time solar wind data, runs the SINDy
forecaster, checks alarms, and optionally displays status.

Press Ctrl-C to stop.
"""
function run_monitor(; poll_interval_min::Int=5,
                       forecast_horizon_hr::Int=6,
                       alarm_config::AlarmConfig=default_alarm_config(),
                       coefficients_csv::String,
                       ensemble_csv::String,
                       log_file::String="monitor.log",
                       display::Bool=true)
    # Initial data fetch with retry
    swd, t_tags = _fetch_with_retry(; hours=48, max_retries=3)

    # Find last valid Dst* or use 0.0
    dst0 = 0.0
    for i in length(t_tags):-1:1
        if !isnan(swd.Dst_star[i])
            dst0 = swd.Dst_star[i]
            break
        end
    end

    state = init_forecast(;
        coefficients_csv=coefficients_csv,
        ensemble_csv=ensemble_csv,
        t0=t_tags[end],
        dst0=dst0,
    )

    # Warm up: run through recent history to initialise state
    println("Initialising from $(length(t_tags)) hours of history...")
    for i in eachindex(t_tags)
        (isnan(swd.V[i]) || isnan(swd.Bz[i])) && continue
        V_safe = _safe_val(swd.V[i], 400.0)
        n_safe = _safe_val(swd.n[i], 5.0)
        Pdyn_safe = _safe_val(swd.Pdyn[i], 1.6726e-6 * n_safe * V_safe^2)
        step_forecast!(state, t_tags[i],
                       V_safe, swd.Bz[i], _safe_val(swd.By[i], 0.0),
                       n_safe, Pdyn_safe;
                       dst_observed=swd.Dst_star[i])
    end

    last_alarm_time = DateTime(1970)
    consecutive_failures = 0
    println("Monitor started. Polling every $(poll_interval_min) min. Ctrl-C to stop.\n")

    try
        while true
            # Fetch latest data with error handling
            swd_new, t_new = try
                data = fetch_realtime_solar_wind(; hours=3)
                consecutive_failures = 0
                data
            catch e
                consecutive_failures += 1
                if display
                    println("  [WARN] Data fetch failed (attempt $consecutive_failures): $(sprint(showerror, e))")
                end
                if consecutive_failures >= 10
                    println("  [ERROR] 10 consecutive failures. Check internet connection.")
                end
                sleep(poll_interval_min * 60)
                continue
            end

            # Guard: need at least 1 valid data point
            if isempty(t_new) || all(isnan, swd_new.V)
                sleep(poll_interval_min * 60)
                continue
            end

            # Find last valid index
            latest_idx = findlast(i -> !isnan(swd_new.V[i]), eachindex(swd_new.V))
            if latest_idx === nothing
                sleep(poll_interval_min * 60)
                continue
            end

            # Safe solar wind values (replace NaN with defaults)
            V = _safe_val(swd_new.V[latest_idx], 400.0)
            Bz = _safe_val(swd_new.Bz[latest_idx], 0.0)
            By = _safe_val(swd_new.By[latest_idx], 0.0)
            n_val = _safe_val(swd_new.n[latest_idx], 5.0)
            Pdyn = _safe_val(swd_new.Pdyn[latest_idx], 1.6726e-6 * n_val * V^2)

            # Step forecast
            result = step_forecast!(state, t_new[latest_idx],
                                    V, Bz, By, n_val, Pdyn)

            # Multi-hour forecast (persistence assumption)
            forecast = forecast_ahead(state, V, Bz, By, n_val, Pdyn,
                                       forecast_horizon_hr)

            # Check alarms on current + forecast
            alarm, last_alarm_time = check_alarm(alarm_config, result,
                                                  last_alarm_time)
            for fr in forecast
                _, last_alarm_time = check_alarm(alarm_config, fr,
                                                  last_alarm_time)
            end

            # Display
            if display
                print_status(result, forecast, alarm, V, Bz, n_val)
            end

            # Log
            try
                open(log_file, "a") do io
                    println(io, Dates.format(result.t, "yyyy-mm-dd HH:MM"),
                            ",", round(result.dst_predicted, digits=1),
                            ",", round(result.dst_ci_05, digits=1),
                            ",", round(result.dst_ci_95, digits=1))
                end
            catch
                # Non-critical — don't crash on log write failure
            end

            sleep(poll_interval_min * 60)
        end
    catch e
        if e isa InterruptException
            println("\nMonitor stopped.")
        else
            rethrow(e)
        end
    end
end

"""
    _safe_val(x, default)

Return x if finite, otherwise default.
"""
_safe_val(x::Float64, default::Float64) = (isnan(x) || isinf(x)) ? default : x

"""
    _fetch_with_retry(; hours, max_retries=3, delay_sec=10)

Fetch solar wind data with retries on failure.
"""
function _fetch_with_retry(; hours::Int, max_retries::Int=3, delay_sec::Int=10)
    for attempt in 1:max_retries
        try
            return fetch_realtime_solar_wind(; hours=hours)
        catch e
            if attempt == max_retries
                error("Failed to fetch solar wind data after $max_retries attempts: $(sprint(showerror, e))")
            end
            println("  Fetch attempt $attempt failed, retrying in $(delay_sec)s...")
            sleep(delay_sec)
        end
    end
end

"""
    print_status(result, forecast, alarm, V, Bz, n_val)

Print formatted monitoring status to terminal.
"""
function print_status(result::ForecastResult,
                      forecast::Vector{ForecastResult},
                      alarm::Union{Nothing, Alarm},
                      V::Float64, Bz::Float64, n_val::Float64)
    severity = classify_severity(result.dst_ci_05,
                                 default_alarm_config().thresholds)
    status_str = severity == QUIET         ? "QUIET" :
                 severity == MODERATE      ? "MODERATE STORM" :
                 severity == INTENSE       ? "INTENSE STORM" :
                                             "SUPER-INTENSE"

    fc_end = isempty(forecast) ? result : forecast[end]
    fc_hours = length(forecast)
    time_str = Dates.format(result.t, "yyyy-mm-dd HH:MM")

    # Solar wind strings
    V_s = isnan(V) ? "  N/A" : lpad(Int(round(V)), 5)
    Bz_s = isnan(Bz) ? "   N/A" : lpad(string(round(Bz, digits=1)), 6)
    n_s = isnan(n_val) ? "  N/A" : lpad(string(round(n_val, digits=1)), 5)

    # Forecast trajectory bar
    bar = String[]
    for fr in forecast
        sym = fr.dst_median > -30 ? "▁" :
              fr.dst_median > -50 ? "▃" :
              fr.dst_median > -100 ? "▅" : "▇"
        push!(bar, sym)
    end
    bar_str = isempty(bar) ? "---" : join(bar)

    # Clear terminal
    print("\033[2J\033[H")

    hline = repeat("─", BOX_W + 2)
    println("┌$(hline)┐")
    println(_boxline(""))
    println(_boxline("  SINDy Storm Monitor           $(time_str) UTC"))
    println(_boxline(""))
    println("├$(hline)┤")
    # Build each line as a char buffer of exactly BOX_W characters.
    # Column layout (0-indexed positions within the BOX_W content area):
    #   Label:   pos 0-9    (10 chars: "  Dst*    " or "  90% CI  ")
    #   Col 1:   pos 10-31  (22 chars for Current value)
    #   Col 2:   pos 32-57  (26 chars for Forecast value)
    function _row(label::String, v1::String, v2::String)
        buf = rpad(label, 10) * rpad(v1, 22) * v2
        return _boxline(rpad(buf, BOX_W))
    end

    fc_label = "$(fc_hours)-hr Forecast"
    println(_row("", "Current", fc_label))
    println(_row("  Dst*", "$(_fmt(result.dst_predicted)) nT", "$(_fmt(fc_end.dst_predicted)) nT"))
    println(_row("  90% CI", "[$(_fmt(result.dst_ci_05)),$(_fmt(result.dst_ci_95))]",
                             "[$(_fmt(fc_end.dst_ci_05)),$(_fmt(fc_end.dst_ci_95))]"))
    println(_boxline(""))
    println(_boxline("  Status: $(rpad(status_str, 15)) Trend: $(rpad(bar_str, 8)) ($(fc_hours) hr)"))
    println("├$(hline)┤")
    println(_boxline("  V =$(V_s) km/s    Bz =$(Bz_s) nT    n =$(n_s) cm-3"))
    println("├$(hline)┤")

    # Alarm line
    if alarm !== nothing
        max_msg = BOX_W - 4
        msg = alarm.message
        if length(msg) > max_msg
            msg = first(msg, max_msg - 1) * "…"
        end
        println(_boxline("  $(msg)"))
    else
        println(_boxline("  Alarm: none"))
    end
    println("└$(hline)┘")
end
