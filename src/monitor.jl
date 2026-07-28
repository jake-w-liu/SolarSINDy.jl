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

function _refresh_dst_feed(previous, fetch::Function=fetch_swpc_dst)
    return try
        fetch()
    catch error_value
        error_value isa InterruptException && rethrow()
        previous
    end
end

"""
    run_monitor(; poll_interval_min=5, forecast_horizon_hr=6,
                  alarm_config, coefficients_csv, ensemble_csv,
                  log_file="monitor.log", display=true,
                  history_cap=2000, max_log_bytes=5_000_000)

Main monitoring loop. Fetches real-time solar wind data, runs the SINDy
forecaster, checks alarms, and optionally displays status.

The forecaster advances exactly one hour of ODE dynamics per new hourly data
bin, not once per poll cycle: `poll_interval_min` controls how often the feed is
refreshed and the display redrawn, while model time stays synchronized with wall
clock. Between new bins the last forecast is reused for display and alarms.

`history_cap` bounds the retained per-step forecast history (the monitor never
reads it back, so it is a rolling window sized for daemon memory safety, about
83 days of hourly steps by default). `max_log_bytes` triggers single-generation
rotation of `log_file` so the append-only log cannot grow without bound; set it
to `0` to disable rotation.

Press Ctrl-C to stop.
"""
function run_monitor(; poll_interval_min::Int=5,
                       forecast_horizon_hr::Int=6,
                       alarm_config::AlarmConfig=default_alarm_config(),
                       coefficients_csv::String,
                       ensemble_csv::String,
                       log_file::String="monitor.log",
                       display::Bool=true,
                       clock::Function=() -> now(UTC),
                       staleness_threshold_hr::Real=3.0,
                       history_cap::Int=2000,
                       max_log_bytes::Int=5_000_000)
    poll_interval_min >= 1 || throw(ArgumentError("poll_interval_min must be at least 1"))
    forecast_horizon_hr >= 0 ||
        throw(ArgumentError("forecast_horizon_hr must be nonnegative"))
    isfinite(staleness_threshold_hr) && staleness_threshold_hr > 0 ||
        throw(ArgumentError("staleness_threshold_hr must be finite and positive"))
    history_cap >= 0 || throw(ArgumentError("history_cap must be nonnegative"))
    max_log_bytes >= 0 || throw(ArgumentError("max_log_bytes must be nonnegative"))
    # Observed Dst for anchoring (best-effort; monitor still runs if the feed fails)
    dst_feed = try
        fetch_swpc_dst()
    catch e
        e isa InterruptException && rethrow()
        println("  [WARN] Dst feed unavailable; forecaster will run unanchored: $(sprint(showerror, e))")
        nothing
    end

    # Initial data fetch with retry
    swd, t_tags = _fetch_with_retry(; hours=48, max_retries=3, dst=dst_feed)

    # Warm only the newest contiguous block of measured primary drivers, then
    # initialise (anchored on the latest observed Dst* when present). This same
    # routine is reused to recover the daemon if a per-cycle modeling fault leaves
    # the state stale or outside the fetch window.
    state, last_result, last_obs_time = _warmup_and_init_monitor(
        swd, t_tags; coefficients_csv=coefficients_csv,
        ensemble_csv=ensemble_csv, history_cap=history_cap)

    last_alarm_time = DateTime(1970)
    last_forecast = ForecastResult[]
    last_alarm::Union{Nothing,Alarm} = nothing
    last_horizon_alarm::Union{Nothing,Alarm} = nothing
    # Persistent horizon-alarm dedup: highest severity already announced per target
    # hour, so a future crossing that reappears every poll is announced once (and
    # re-announced only on escalation). Pruned each cycle to bound memory.
    horizon_seen = Dict{DateTime,StormSeverity}()
    consecutive_failures = 0
    cycle_failures = 0
    println("Monitor started. Polling every $(poll_interval_min) min. Ctrl-C to stop.\n")

    try
        while true
            # Fetch latest data with error handling (refresh observed Dst too)
            swd_new, t_new, t_fresh = try
                dst_feed = _refresh_dst_feed(dst_feed)
                # 6 h trailing window: wide enough to contain at least one
                # published Kyoto Dst hour (the feed lags ~1-3 h) so re-anchoring
                # has an observation to lock onto.
                data = fetch_realtime_solar_wind(; hours=6, dst=dst_feed,
                                                 propagate_l1_to_earth=true)
                consecutive_failures = 0
                data
            catch e
                e isa InterruptException && rethrow()
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
            if isempty(t_new) || all(x -> !isfinite(x), swd_new.V)
                sleep(poll_interval_min * 60)
                continue
            end

            # Find last bin where BOTH V and Bz are finite. Matching the warm-up
            # policy (which skips NaN-Bz hours) is essential: selecting on V
            # alone can land on a bin whose Bz is NaN, where _safe_val(Bz, 0.0)
            # injects Bz=0 (no southward driving) and silently produces a quiet
            # forecast during an active storm. If no bin has finite Bz in the
            # window, skip this cycle rather than emit a Bz=0 quiet forecast.
            latest_idx = _latest_finite_VBz_idx(swd_new.V, swd_new.Bz)
            if latest_idx === nothing
                if display
                    println("  [WARN] No bin with finite V and Bz (mag gap); skipping cycle.")
                end
                sleep(poll_interval_min * 60)
                continue
            end

            # Data-staleness guard: a frozen feed (old rows returned 200 OK) must
            # not be displayed as current. Threshold allows Kyoto/SWPC publication
            # lag (~2-3 h) before flagging.
            # Measure freshness from the newest actual sample (t_fresh = t_end), not the last
            # hour-floored bin start, which lags real time by up to ~2 h (within-hour offset +
            # the dropped trailing partial hour) and would falsely trip STALE on a live feed.
            data_age = clock() - t_fresh
            # Use |age|: a future-dated latest timestamp (clock skew or a mislabeled feed) is
            # anomalous too and must not read as "fresh". Threshold in minutes avoids the
            # round-half-to-even surprise of Hour(Int(round(2.5))) == 2 h.
            stale = _monitor_data_stale(data_age, staleness_threshold_hr)

            # Safe solar wind values (replace NaN with defaults)
            V = _safe_val(swd_new.V[latest_idx], 400.0)
            Bz = _safe_val(swd_new.Bz[latest_idx], 0.0)
            By = _safe_val(swd_new.By[latest_idx], 0.0)
            n_val = _safe_val(swd_new.n[latest_idx], 5.0)
            Pdyn = _safe_val(swd_new.Pdyn[latest_idx], 1.6726e-6 * n_val * V^2)

            # Advance the model (anti-compounding new-bin gate + delayed-anchor
            # replay + forecast + alarms) in one testable step. Guard it: the replay
            # bridge raises ArgumentError in reachable feed-gap states (an interior
            # >3 h driver gap between a lagging Dst anchor and the newest bin, or a
            # state that has fallen outside the fetch window after a long outage or
            # host suspension). Without this guard that exception would propagate to
            # the loop-level catch and terminate the daemon for a transient upstream
            # condition. On failure, re-warm from the freshest contiguous block so
            # the monitor recovers instead of wedging, then skip this cycle.
            cyc = try
                _monitor_cycle!(state, swd_new, t_new, latest_idx,
                                V, Bz, By, n_val, Pdyn;
                                forecast_horizon_hr=forecast_horizon_hr,
                                alarm_config=alarm_config,
                                history_cap=history_cap,
                                last_result=last_result,
                                last_forecast=last_forecast,
                                last_alarm=last_alarm,
                                last_alarm_time=last_alarm_time,
                                last_horizon_alarm=last_horizon_alarm,
                                last_obs_time=last_obs_time,
                                horizon_seen=horizon_seen)
            catch e
                e isa InterruptException && rethrow()
                cycle_failures += 1
                if display
                    println("  [WARN] Forecast cycle failed (attempt $cycle_failures): $(sprint(showerror, e))")
                end
                # Best-effort recovery: rebuild the forecaster from the freshest
                # contiguous driver block (Dst re-anchors from the feed). If re-warm
                # also fails, stay alive and retry on the next poll.
                try
                    state, last_result, last_obs_time = _warmup_and_init_monitor(
                        swd_new, t_new; coefficients_csv=coefficients_csv,
                        ensemble_csv=ensemble_csv, history_cap=history_cap)
                    last_forecast = ForecastResult[]
                    last_alarm = nothing
                    last_horizon_alarm = nothing
                    empty!(horizon_seen)
                catch e2
                    e2 isa InterruptException && rethrow()
                    @warn "Monitor re-warm after cycle failure failed; retrying next cycle" exception=(e2, catch_backtrace()) maxlog=1
                end
                sleep(poll_interval_min * 60)
                continue
            end
            # Reset the consecutive-cycle-failure counter only after a clean cycle
            # (the catch above ends in `continue`, so control reaches here only on
            # success). Resetting before the attempt would make a persistent fault
            # report "attempt 1" on every poll instead of escalating.
            cycle_failures = 0

            result = cyc.result
            forecast = cyc.forecast
            last_result = cyc.result
            last_forecast = cyc.forecast
            last_alarm = cyc.last_alarm
            last_alarm_time = cyc.last_alarm_time
            last_horizon_alarm = cyc.last_horizon_alarm
            last_obs_time = cyc.last_obs_time

            # Log only on a new bin so the file is not filled with duplicate
            # same-timestamp rows. Rotate first so the append-only log is bounded.
            if cyc.new_bin
                try
                    _rotate_log!(log_file, max_log_bytes)
                    open(log_file, "a") do io
                        println(io, Dates.format(result.t, "yyyy-mm-dd HH:MM"),
                                ",", round(result.dst_predicted, digits=1),
                                ",", round(result.dst_ci_05, digits=1),
                                ",", round(result.dst_ci_95, digits=1))
                    end
                catch e
                    e isa InterruptException && rethrow()
                    # Forecasting continues, but persistence failure is an
                    # operational health fault and must remain visible.
                    @warn "Monitor log persistence failed" log_file exception=(e, catch_backtrace()) maxlog=1
                end
            end

            # Display. Surface the Dst-anchor age so an unanchored free-run (Dst feed
            # outage) is visibly labeled rather than presented as a current state.
            if display
                anchor_age = last_obs_time === nothing ? nothing : clock() - last_obs_time
                print_status(result, forecast, last_alarm, V, Bz, n_val;
                             data_age=data_age, stale=stale,
                             alarm_config=alarm_config,
                             horizon_alarm=last_horizon_alarm,
                             anchor_age=anchor_age,
                             unanchored=(last_obs_time === nothing))
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
    _warmup_and_init_monitor(swd, t_tags; coefficients_csv, ensemble_csv, history_cap)

Warm the forecaster over the newest strictly-hourly contiguous driver block and
return `(state, last_result, last_obs_time)`. When the block contains an observed
Dst*, the state is anchored on the first such observation; otherwise it starts from
`Dst*=0` (unanchored free-run). Row `k` (the average over `[t[k], t[k+1])`) advances
the state from `t[k]` to `t[k+1]`, so the transition into row `i` uses driver row
`i-1`. Shared by the initial start-up and by the daemon's per-cycle recovery path.
"""
function _warmup_and_init_monitor(swd::SolarWindData,
                                  t_tags::AbstractVector{DateTime};
                                  coefficients_csv::String, ensemble_csv::String,
                                  history_cap::Int)
    warm_start, warm_end, anchor_idx = _monitor_warmup_window(swd, t_tags)
    if anchor_idx !== nothing
        state = init_forecast(;
            coefficients_csv=coefficients_csv,
            ensemble_csv=ensemble_csv,
            t0=t_tags[anchor_idx],
            dst0=swd.Dst_star[anchor_idx],
        )
        observed = swd.Dst_star[anchor_idx]
        last_result::Union{Nothing,ForecastResult} = ForecastResult(
            t_tags[anchor_idx], observed, observed, observed, observed, observed)
        last_obs_time::Union{Nothing,DateTime} = t_tags[anchor_idx]
        first_step = anchor_idx + 1
    else
        println("  [WARN] No observed Dst* in the contiguous driver window; initial Dst*=0 (unanchored free-run).")
        state = init_forecast(;
            coefficients_csv=coefficients_csv,
            ensemble_csv=ensemble_csv,
            t0=t_tags[warm_start],
            dst0=0.0,
        )
        last_result = nothing
        last_obs_time = nothing
        first_step = warm_start + 1
    end

    println("Initialising from contiguous rows $(first_step):$(warm_end)...")
    for i in first_step:warm_end
        # Never use the target row's driver average, which belongs to the following
        # (future) interval.
        driver_idx = i - 1
        V_safe = _safe_val(swd.V[driver_idx], 400.0)
        n_safe = _safe_val(swd.n[driver_idx], 5.0)
        Pdyn_safe = _safe_val(swd.Pdyn[driver_idx],
                              1.6726e-6 * n_safe * V_safe^2)
        last_result = step_forecast!(state, t_tags[i],
                       V_safe, swd.Bz[driver_idx],
                       _safe_val(swd.By[driver_idx], 0.0),
                       n_safe, Pdyn_safe;
                       dst_observed=swd.Dst_star[i])
        isfinite(swd.Dst_star[i]) && (last_obs_time = t_tags[i])
    end
    _cap_history!(state, history_cap)
    return state, last_result, last_obs_time
end

"""
    _monitor_data_stale(data_age, staleness_threshold_hr) -> Bool

Return `true` when the newest data sample is older than `staleness_threshold_hr`.
`|data_age|` is used so a future-dated latest timestamp (clock skew or a mislabeled
feed) is treated as anomalous rather than fresh; the threshold is taken in whole
minutes to avoid the round-half-to-even surprise of `Hour(Int(round(2.5))) == 2 h`.
"""
_monitor_data_stale(data_age::Period, staleness_threshold_hr::Real) =
    abs(data_age) >= Minute(round(Int, staleness_threshold_hr * 60))

"""
    _prune_horizon_seen!(horizon_seen, current_t)

Drop horizon-alarm dedup entries whose target hour is now strictly in the past
relative to `current_t`, bounding the dictionary's memory. Kept-current-hour
semantics: an entry at exactly `current_t` is retained.
"""
function _prune_horizon_seen!(horizon_seen::Dict{DateTime,StormSeverity},
                              current_t::DateTime)
    for target in collect(keys(horizon_seen))
        target < current_t && delete!(horizon_seen, target)
    end
    return horizon_seen
end

"""
    _monitor_cycle!(state, swd_new, t_new, latest_idx, V, Bz, By, n_val, Pdyn; ...)

One poll cycle of model advancement, extracted from `run_monitor`'s loop so the glue
is unit-testable (no fetch, sleep, log, or display side effects).

The anti-compounding gate advances the model only when a genuinely new hourly bin has
appeared: `step_forecast!` integrates a fixed 1 h of dynamics per call, so re-stepping
the same bin every poll (default 5 min) would compound one model-hour per cycle —
during a Dst outage the unanchored forecaster would free-run ~12× faster than wall
clock. On a new bin the forecaster re-anchors on the most recent observed Dst*, replays
each intervening bin (a delayed anchor must not be compressed into one Euler step),
projects the multi-hour forecast, checks alarms, prunes and updates the horizon-alarm
dedup set. On a repeated bin the previous result/forecast/alarms are returned unchanged.

`last_alarm_time` accepts either a bare `DateTime` (the bootstrap value before any alarm
has been evaluated) or the [`AlarmCooldownState`](@ref) returned by [`check_alarm`](@ref);
the loop threads the returned `last_alarm_time` straight back into the next cycle, so after
the first new-bin advance this keyword carries an `AlarmCooldownState`, which is what enables
the strict-escalation cooldown bypass across cycles.

Returns a named tuple `(new_bin, result, forecast, last_alarm, last_alarm_time,
last_horizon_alarm, last_obs_time)`.
"""
function _monitor_cycle!(state::ForecastState, swd_new::SolarWindData,
                         t_new::AbstractVector{DateTime}, latest_idx::Int,
                         V::Float64, Bz::Float64, By::Float64,
                         n_val::Float64, Pdyn::Float64;
                         forecast_horizon_hr::Int,
                         alarm_config::AlarmConfig,
                         history_cap::Int,
                         last_result::Union{Nothing,ForecastResult},
                         last_forecast::Vector{ForecastResult},
                         last_alarm::Union{Nothing,Alarm},
                         last_alarm_time::Union{DateTime,AlarmCooldownState},
                         last_horizon_alarm::Union{Nothing,Alarm},
                         last_obs_time::Union{Nothing,DateTime},
                         horizon_seen::Dict{DateTime,StormSeverity})
    new_bin = last_result === nothing || t_new[latest_idx] > state.t_current
    if !new_bin
        # No new hourly bin: reuse the last forecast rather than re-integrating.
        return (; new_bin=false, result=last_result, forecast=last_forecast,
                last_alarm=last_alarm, last_alarm_time=last_alarm_time,
                last_horizon_alarm=last_horizon_alarm, last_obs_time=last_obs_time)
    end

    obs_idx = findlast(i -> isfinite(swd_new.Dst_star[i]), 1:latest_idx)
    new_obs_time = obs_idx !== nothing ? t_new[obs_idx] : last_obs_time

    result = _replay_monitor_from_anchor!(state, swd_new, t_new, obs_idx, latest_idx)
    _cap_history!(state, history_cap)

    forecast = forecast_ahead(state, V, Bz, By, n_val, Pdyn, forecast_horizon_hr)

    # Only the current-observation alarm advances the cooldown clock; forecast-horizon
    # alarms have future timestamps, so letting them set last_alarm_time would push the
    # cooldown into the future and suppress the next genuine present-time alarm.
    alarm, alarm_time = check_alarm(alarm_config, result, last_alarm_time)

    _prune_horizon_seen!(horizon_seen, result.t)
    horizon_alarm = nothing
    for fr in forecast
        a = maybe_fire_horizon_alarm!(alarm_config, fr, horizon_seen)
        a === nothing && continue
        if horizon_alarm === nothing || a.severity > horizon_alarm.severity
            horizon_alarm = a
        end
    end

    return (; new_bin=true, result=result, forecast=forecast,
            last_alarm=alarm, last_alarm_time=alarm_time,
            last_horizon_alarm=horizon_alarm, last_obs_time=new_obs_time)
end

"""
    _safe_val(x, default)

Return x if finite, otherwise default.
"""
_safe_val(x::Float64, default::Float64) = (isnan(x) || isinf(x)) ? default : x

"""
    _monitor_warmup_window(swd, times)

Return `(start_idx, end_idx, anchor_idx)` for the newest strictly hourly,
contiguous block whose speed and Bz are finite. `anchor_idx` is the first finite
Dst* observation in that block, ensuring warm-up never starts from a future
observation. Throws when no primary-driver row is usable.
"""
function _monitor_warmup_window(swd::SolarWindData,
                                times::AbstractVector{DateTime})
    n = length(times)
    n >= 1 || throw(ArgumentError("monitor warm-up requires at least one timestamp"))
    all(length(v) == n for v in
        (swd.t, swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("driver times and SolarWindData fields must align"))
    latest = _latest_finite_VBz_idx(swd.V, swd.Bz)
    latest === nothing && throw(ArgumentError("monitor warm-up has no finite V/Bz row"))
    start = latest
    while start > 1 && isfinite(swd.V[start - 1]) && isfinite(swd.Bz[start - 1]) &&
          times[start] - times[start - 1] == Hour(1)
        start -= 1
    end
    anchor = findfirst(i -> isfinite(swd.Dst_star[i]), start:latest)
    anchor_idx = anchor === nothing ? nothing : start + anchor - 1
    return start, latest, anchor_idx
end

"""
    _replay_monitor_from_anchor!(state, swd, times, obs_idx, latest_idx)

Anchor `state` at `obs_idx` when available, then advance through every hourly
target timestamp up to `latest_idx`. A row timestamp labels the start of its
hourly driver-average interval, so the transition to target row `j` uses driver
row `j-1`. This prevents both delayed-anchor compression and future-driver
leakage.
"""
function _replay_monitor_from_anchor!(state::ForecastState, swd::SolarWindData,
                                      times::AbstractVector{DateTime},
                                      obs_idx::Union{Nothing,Int}, latest_idx::Int)
    n = length(times)
    1 <= latest_idx <= n || throw(BoundsError(times, latest_idx))
    all(length(v) == n for v in
        (swd.t, swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("driver times and SolarWindData fields must align"))
    all(i -> times[i + 1] - times[i] == Hour(1), 1:(n - 1)) ||
        throw(ArgumentError("monitor driver times must be strictly hourly and contiguous"))
    if obs_idx !== nothing
        1 <= obs_idx <= latest_idx ||
            throw(ArgumentError("observation index must not follow the latest driver index"))
        isfinite(swd.Dst_star[obs_idx]) ||
            throw(ArgumentError("observation anchor must be finite"))
        state.dst_current = swd.Dst_star[obs_idx]
        state.t_current = times[obs_idx]
    end

    # Replay every driver bin strictly after the state/anchor time. In the
    # unanchored case this also catches multiple bins missed during a polling or
    # network outage; taking only the newest row would compress the whole gap to
    # one Euler step.
    replay_start = findfirst(j -> times[j] > state.t_current, 1:latest_idx)
    if replay_start === nothing
        # The newest Dst observation can coincide with the newest driver bin.
        # There is then no elapsed model interval to integrate: expose the
        # observed current state and forecast future horizons from that anchor,
        # rather than applying a one-hour step at the same timestamp.
        state.t_current == times[latest_idx] ||
            throw(ArgumentError("no driver bin follows the current forecast state"))
        current = state.dst_current
        observed = obs_idx === latest_idx ? current : NaN
        result = ForecastResult(times[latest_idx], current, current,
                                current, current, observed)
        push!(state.history, result)
        return result
    end
    replay_start >= 2 && times[replay_start - 1] == state.t_current ||
        throw(ArgumentError(
            "cannot bridge forecast state to the first available target without its preceding driver bin"
        ))
    result = nothing
    for j in replay_start:latest_idx
        driver_idx = j - 1
        isfinite(swd.V[driver_idx]) && isfinite(swd.Bz[driver_idx]) ||
            throw(ArgumentError(
                "cannot bridge Dst anchor across non-finite primary drivers at $(times[driver_idx])"
            ))
        Vj = _safe_val(swd.V[driver_idx], 400.0)
        Bzj = swd.Bz[driver_idx]
        Byj = _safe_val(swd.By[driver_idx], 0.0)
        nj = _safe_val(swd.n[driver_idx], 5.0)
        Pj = _safe_val(swd.Pdyn[driver_idx], dynamic_pressure(nj, Vj))
        result = step_forecast!(state, times[j], Vj, Bzj, Byj, nj, Pj)
    end
    result === nothing && error("monitor failed to produce a forecast step")
    return result
end

"""
    _cap_history!(state, cap)

Bound `state.history` to at most `cap` entries by dropping the oldest, matching the
rolling-window pattern used for the conformal residual history. The monitor never
reads `state.history`, so trimming it prevents unbounded memory growth in a
long-running daemon. A non-positive `cap` disables trimming.
"""
function _cap_history!(state::ForecastState, cap::Int)
    cap > 0 || return state
    excess = length(state.history) - cap
    excess > 0 && deleteat!(state.history, 1:excess)
    return state
end

"""
    _rotate_log!(path, max_bytes)

Rotate `path` to `path * ".1"` (overwriting any prior rotation) once it exceeds
`max_bytes`, so an append-only monitor log cannot grow without bound. Keeps a single
previous generation; total on-disk footprint stays below `2 * max_bytes`. A
non-positive `max_bytes` disables rotation.
"""
function _rotate_log!(path::String, max_bytes::Int)
    (max_bytes > 0 && isfile(path) && filesize(path) > max_bytes) || return nothing
    _atomic_replace_regular(path, path * ".1")
    return nothing
end

"""
    _latest_finite_VBz_idx(V, Bz)

Return the last index where BOTH `V` and `Bz` are finite, or `nothing` if no
such index exists. The live monitor uses this so its issue-time bin selection
matches the warm-up policy (which skips NaN-Bz hours): a bin with finite V but
NaN Bz would otherwise force Bz=0 and suppress a southward-driving storm alarm.
"""
function _latest_finite_VBz_idx(V::AbstractVector, Bz::AbstractVector)
    length(V) == length(Bz) ||
        throw(DimensionMismatch("V and Bz must have equal length"))
    return findlast(i -> isfinite(V[i]) && isfinite(Bz[i]), eachindex(V))
end

"""
    _fetch_with_retry(; hours, max_retries=3, delay_sec=10)

Fetch solar wind data with retries on failure.
"""
function _fetch_with_retry(; hours::Int, max_retries::Int=3, delay_sec::Int=10,
                            dst::Union{Nothing,Tuple}=nothing)
    for attempt in 1:max_retries
        try
            return fetch_realtime_solar_wind(; hours=hours, dst=dst,
                                             propagate_l1_to_earth=true)
        catch e
            e isa InterruptException && rethrow()
            if attempt == max_retries
                error("Failed to fetch solar wind data after $max_retries attempts: $(sprint(showerror, e))")
            end
            println("  Fetch attempt $attempt failed, retrying in $(delay_sec)s...")
            sleep(delay_sec)
        end
    end
end

"""
    print_status(result, forecast, alarm, V, Bz, n_val;
                 alarm_config=default_alarm_config(), horizon_alarm=nothing,
                 anchor_age=nothing, unanchored=false)

Print formatted monitoring status to terminal.

The status banner is classified with the same rule the alarm path uses
(`alarm_config.use_worst_case` selects the 5th percentile or the median, against
`alarm_config.thresholds`), so the displayed severity cannot contradict the alarms
actually raised for a custom-threshold configuration.
"""
function print_status(result::ForecastResult,
                      forecast::Vector{ForecastResult},
                      alarm::Union{Nothing, Alarm},
                      V::Float64, Bz::Float64, n_val::Float64;
                      data_age::Union{Nothing,Period}=nothing,
                      stale::Bool=false,
                      alarm_config::AlarmConfig=default_alarm_config(),
                      horizon_alarm::Union{Nothing,Alarm}=nothing,
                      anchor_age::Union{Nothing,Period}=nothing,
                      unanchored::Bool=false)
    dst_class = alarm_config.use_worst_case ? result.dst_ci_05 : result.dst_predicted
    severity = classify_severity(dst_class, alarm_config.thresholds)
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

    # Header data-age annotation: report how old the newest data hour is, and
    # flag a frozen/stale feed loudly so an old state is never read as current.
    age_str = if data_age === nothing
        ""
    else
        age_hr = data_age / Hour(1)
        stale ? "  [STALE: data $(round(age_hr, digits=1)) h old]" :
                "  (data $(round(age_hr, digits=1)) h old)"
    end

    # Dst-anchor annotation: the displayed state is a projection from the last
    # observed Dst*. Label an unanchored free-run (Dst feed outage) loudly, and
    # otherwise report how old the anchor is, so the nowcast is never mistaken for
    # a fresh observation.
    anchor_str = if unanchored
        "  [UNANCHORED: no observed Dst*; forecaster free-run]"
    elseif anchor_age === nothing
        ""
    else
        "  (Dst anchor $(round(anchor_age / Hour(1), digits=1)) h old)"
    end

    hline = repeat("─", BOX_W + 2)
    println("┌$(hline)┐")
    println(_boxline(""))
    println(_boxline("  SINDy Storm Monitor           $(time_str) UTC"))
    isempty(age_str) || println(_boxline(age_str))
    isempty(anchor_str) || println(_boxline(anchor_str))
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

    # Forecast-horizon alarm line. Rendered from the passed alarm rather than the
    # callback so the strongest upcoming crossing stays visible after the terminal
    # clear (the callback's stdout output is erased on the next redraw).
    if horizon_alarm !== nothing
        lab = horizon_alarm.severity == SUPERINTENSE ? "SUPER-INTENSE" :
              horizon_alarm.severity == INTENSE      ? "INTENSE" : "MODERATE"
        println(_boxline("  Fcst alarm: $(lab) by " *
                         "$(Dates.format(horizon_alarm.time, "yyyy-mm-dd HH:MM")) " *
                         "($(round(horizon_alarm.dst_ci_05, digits=1)) nT)"))
    end
    println("└$(hline)┘")
end
