#!/usr/bin/env julia
# storm_monitor.jl — Real-time geomagnetic storm monitor (Operational V2.1)
#
# Usage:
#   julia --project=SolarSINDy.jl examples/storm_monitor.jl
#
# Fetches live solar wind data from NOAA SWPC, forward-integrates the
# eleven-term SINDy-discovered Dst* equation at hourly cadence, and propagates a
# 500-member coefficient ensemble to produce ensemble prediction intervals with
# configurable storm-severity alarms.
#
# Bounded runs (verification / CI): set the environment variable
#   STORM_MONITOR_MAX_CYCLES=<n>
# to exit after n poll cycles instead of running until Ctrl-C.
#
# Press Ctrl-C to stop.

using SolarSINDy
using Dates

const CORE_VERSION = OPERATIONAL_V2_1_MODEL_VERSION
const CORE_ARTIFACTS = operational_core_artifacts(CORE_VERSION)
const DATA_DIR = get_data_dir()
const COEF_CSV  = CORE_ARTIFACTS.coefficients_csv
const INCL_CSV  = CORE_ARTIFACTS.ensemble_csv
const DRAWS_CSV = CORE_ARTIFACTS.draws_csv

# All three current artifacts are versioned with the package and validated as
# one exact 20-candidate/11-active-term identity before monitoring begins.
for f in (COEF_CSV, INCL_CSV, DRAWS_CSV)
    isfile(f) || error("Missing Operational V2.1 artifact: $f")
end
load_operational_core(CORE_VERSION)

function print_banner()
    println("=" ^ 60)
    println("  SINDy Real-Time Storm Monitor")
    println("  Equation: Operational V2.1 (20 candidates, 11 active terms)")
    println("  Ensemble: 500 coefficient sets for UQ")
    println("  Data: NOAA SWPC (DSCOVR L1)")
    println("=" ^ 60)
end

# Alarm thresholds on the ensemble 5th-percentile (worst-case) Dst*.
const ALARM_CONFIG = AlarmConfig(
    Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
    true,          # alarm on worst-case (5th percentile)
    alarm_print,   # print to terminal
    6,             # 6-hour cooldown between alarms
)

"""
    build_v2_1_state(swd, t_tags; history_cap=2000)

Warm the current V2.1 forecaster over the newest strictly-hourly contiguous
driver block and return `(state, last_result, last_obs_time)`. The versioned
initializer validates and loads the canonical 20/11 point and joint-draw
artifacts. Row `k` advances the state from
`t[k]` to `t[k+1]`, so the transition into row `i` uses driver row `i-1`.
"""
function build_v2_1_state(swd::SolarWindData, t_tags::AbstractVector{DateTime};
                          history_cap::Int=2000)
    warm_start, warm_end, anchor_idx = SolarSINDy._monitor_warmup_window(swd, t_tags)
    if anchor_idx !== nothing
        state = init_operational_forecast(; version=CORE_VERSION,
                                          t0=t_tags[anchor_idx],
                                          dst0=swd.Dst_star[anchor_idx])
        observed = swd.Dst_star[anchor_idx]
        last_result::Union{Nothing,ForecastResult} =
            ForecastResult(t_tags[anchor_idx], observed, observed, observed, observed, observed)
        last_obs_time::Union{Nothing,DateTime} = t_tags[anchor_idx]
        first_step = anchor_idx + 1
    else
        println("  [WARN] No observed Dst* in the contiguous driver window; initial Dst*=0 (unanchored free-run).")
        state = init_operational_forecast(; version=CORE_VERSION,
                                          t0=t_tags[warm_start], dst0=0.0)
        last_result = nothing
        last_obs_time = nothing
        first_step = warm_start + 1
    end

    println("Initialising from contiguous rows $(first_step):$(warm_end)...")
    for i in first_step:warm_end
        driver_idx = i - 1
        V_safe = SolarSINDy._safe_val(swd.V[driver_idx], 400.0)
        n_safe = SolarSINDy._safe_val(swd.n[driver_idx], 5.0)
        Pdyn_safe = SolarSINDy._safe_val(swd.Pdyn[driver_idx],
                                         1.6726e-6 * n_safe * V_safe^2)
        last_result = step_forecast!(state, t_tags[i], V_safe, swd.Bz[driver_idx],
                                     SolarSINDy._safe_val(swd.By[driver_idx], 0.0),
                                     n_safe, Pdyn_safe; dst_observed=swd.Dst_star[i])
        isfinite(swd.Dst_star[i]) && (last_obs_time = t_tags[i])
    end
    SolarSINDy._cap_history!(state, history_cap)
    return state, last_result, last_obs_time
end

"""
    run_v2_1_monitor(; poll_interval_min=5, forecast_horizon_hr=6,
                        alarm_config=ALARM_CONFIG, log_file="storm_monitor.log",
                        display=true, max_cycles=typemax(Int))

Live monitoring loop for the current V2.1 forecaster. Each new hourly bin
advances exactly one hour of ODE dynamics (`step_forecast!` via the shared
per-cycle advance), re-anchors on the most recent observed Dst*, projects the
multi-hour forecast with ensemble prediction intervals, and checks alarms.
`poll_interval_min` controls how often the feed is refreshed; model time stays
synchronized with the wall clock. `max_cycles` bounds the loop for verification.

The per-cycle advance and warm-up reuse the package's tested monitor internals,
so the direct monitor and the locked operational workflow use the same core.
"""
function run_v2_1_monitor(; poll_interval_min::Int=5,
                            forecast_horizon_hr::Int=6,
                            alarm_config::AlarmConfig=ALARM_CONFIG,
                            log_file::String="storm_monitor.log",
                            display::Bool=true,
                            history_cap::Int=2000,
                            max_log_bytes::Int=5_000_000,
                            max_cycles::Int=typemax(Int))
    max_cycles >= 1 || throw(ArgumentError("max_cycles must be at least 1"))
    max_log_bytes >= 0 || throw(ArgumentError("max_log_bytes must be nonnegative"))

    dst_feed = try
        fetch_swpc_dst()
    catch e
        e isa InterruptException && rethrow()
        println("  [WARN] Dst feed unavailable; forecaster will run unanchored: $(sprint(showerror, e))")
        nothing
    end

    swd, t_tags = SolarSINDy._fetch_with_retry(; hours=48, max_retries=3, dst=dst_feed)
    state, last_result, last_obs_time = build_v2_1_state(swd, t_tags; history_cap=history_cap)

    last_alarm_time::Union{DateTime,SolarSINDy.AlarmCooldownState} = DateTime(1970)
    last_forecast = ForecastResult[]
    last_alarm::Union{Nothing,Alarm} = nothing
    last_horizon_alarm::Union{Nothing,Alarm} = nothing
    horizon_seen = Dict{DateTime,StormSeverity}()
    consecutive_failures = 0
    cycle_failures = 0

    println("Monitor started. Polling every $(poll_interval_min) min. Ctrl-C to stop.\n")
    cycles = 0
    try
        while cycles < max_cycles
            cycles += 1

            swd_new, t_new, t_fresh = try
                dst_feed = SolarSINDy._refresh_dst_feed(dst_feed)
                data = fetch_realtime_solar_wind(; hours=6, dst=dst_feed,
                                                 propagate_l1_to_earth=true)
                consecutive_failures = 0
                data
            catch e
                e isa InterruptException && rethrow()
                consecutive_failures += 1
                display && println("  [WARN] Data fetch failed (attempt $consecutive_failures): $(sprint(showerror, e))")
                consecutive_failures >= 10 &&
                    println("  [ERROR] 10 consecutive failures. Check internet connection.")
                cycles >= max_cycles && break
                sleep(poll_interval_min * 60)
                continue
            end

            if isempty(t_new) || all(x -> !isfinite(x), swd_new.V)
                cycles >= max_cycles && break
                sleep(poll_interval_min * 60)
                continue
            end

            latest_idx = SolarSINDy._latest_finite_VBz_idx(swd_new.V, swd_new.Bz)
            if latest_idx === nothing
                display && println("  [WARN] No bin with finite V and Bz (mag gap); skipping cycle.")
                cycles >= max_cycles && break
                sleep(poll_interval_min * 60)
                continue
            end

            data_age = now(UTC) - t_fresh
            stale = SolarSINDy._monitor_data_stale(data_age, 3.0)

            V = SolarSINDy._safe_val(swd_new.V[latest_idx], 400.0)
            Bz = SolarSINDy._safe_val(swd_new.Bz[latest_idx], 0.0)
            By = SolarSINDy._safe_val(swd_new.By[latest_idx], 0.0)
            n_val = SolarSINDy._safe_val(swd_new.n[latest_idx], 5.0)
            Pdyn = SolarSINDy._safe_val(swd_new.Pdyn[latest_idx],
                                        1.6726e-6 * n_val * V^2)

            # Advance the model in one guarded step. The replay bridge raises
            # ArgumentError in reachable feed-gap states (an interior >3 h driver
            # gap between a lagging Dst anchor and the newest bin, or a state that
            # has fallen outside the fetch window after a long outage or host
            # suspension). Guard it so a transient upstream condition re-warms the
            # forecaster instead of propagating to the loop-level catch and
            # terminating the daemon. This mirrors the operational monitor.
            cyc = try
                SolarSINDy._monitor_cycle!(
                    state, swd_new, t_new, latest_idx, V, Bz, By, n_val, Pdyn;
                    forecast_horizon_hr=forecast_horizon_hr, alarm_config=alarm_config,
                    history_cap=history_cap, last_result=last_result,
                    last_forecast=last_forecast, last_alarm=last_alarm,
                    last_alarm_time=last_alarm_time, last_horizon_alarm=last_horizon_alarm,
                    last_obs_time=last_obs_time, horizon_seen=horizon_seen)
            catch e
                e isa InterruptException && rethrow()
                cycle_failures += 1
                display && println("  [WARN] Forecast cycle failed (attempt $cycle_failures): $(sprint(showerror, e))")
                # Best-effort recovery: rebuild the V2.1 forecaster from the
                # freshest contiguous driver block (Dst re-anchors from the feed).
                # If re-warm also fails, stay alive and retry on the next poll.
                try
                    state, last_result, last_obs_time =
                        build_v2_1_state(swd_new, t_new; history_cap=history_cap)
                    last_forecast = ForecastResult[]
                    last_alarm = nothing
                    last_horizon_alarm = nothing
                    empty!(horizon_seen)
                catch e2
                    e2 isa InterruptException && rethrow()
                    @warn "Monitor re-warm after cycle failure failed; retrying next cycle" exception=(e2, catch_backtrace()) maxlog=1
                end
                cycles >= max_cycles && break
                sleep(poll_interval_min * 60)
                continue
            end
            # Reset the cycle-failure counter only after a clean cycle (the catch
            # above ends in `continue`), so a persistent fault escalates instead of
            # reporting "attempt 1" every poll.
            cycle_failures = 0

            last_result = cyc.result
            last_forecast = cyc.forecast
            last_alarm = cyc.last_alarm
            last_alarm_time = cyc.last_alarm_time
            last_horizon_alarm = cyc.last_horizon_alarm
            last_obs_time = cyc.last_obs_time

            if cyc.new_bin
                try
                    SolarSINDy._rotate_log!(log_file, max_log_bytes)
                    open(log_file, "a") do io
                        println(io, Dates.format(cyc.result.t, "yyyy-mm-dd HH:MM"),
                                ",", round(cyc.result.dst_predicted, digits=1),
                                ",", round(cyc.result.dst_ci_05, digits=1),
                                ",", round(cyc.result.dst_ci_95, digits=1))
                    end
                catch e
                    e isa InterruptException && rethrow()
                    @warn "Monitor log persistence failed" log_file exception=(e, catch_backtrace()) maxlog=1
                end
            end

            if display
                anchor_age = last_obs_time === nothing ? nothing : now(UTC) - last_obs_time
                SolarSINDy.print_status(cyc.result, cyc.forecast, cyc.last_alarm,
                                        V, Bz, n_val;
                                        data_age=data_age, stale=stale,
                                        alarm_config=alarm_config,
                                        horizon_alarm=cyc.last_horizon_alarm,
                                        anchor_age=anchor_age,
                                        unanchored=(last_obs_time === nothing))
            end

            cycles >= max_cycles && break
            sleep(poll_interval_min * 60)
        end
    catch e
        if e isa InterruptException
            println("\nMonitor stopped.")
        else
            rethrow(e)
        end
    end
    return nothing
end

function main()
    print_banner()
    max_cycles = let v = get(ENV, "STORM_MONITOR_MAX_CYCLES", "")
        isempty(v) ? typemax(Int) : parse(Int, v)
    end
    run_v2_1_monitor(; alarm_config=ALARM_CONFIG, log_file="storm_monitor.log",
                     display=true, max_cycles=max_cycles)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
