#!/usr/bin/env julia
# Issue a live Dst forecast, then optionally wait for the target observation.
#
# Usage:
#   julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --wait
#
# The script appends a pre-observation forecast row to live_forecasts/live_forecast_log.csv.
# When --wait is enabled, it polls the SWPC Kyoto Dst feed until the target
# observation appears, then updates that same row with the residual.

using SolarSINDy
using CSV
using DataFrames
using Dates
using HTTP
using JSON3
using Statistics

const KYOTO_DST_JSON_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
const DEFAULT_LOG_PATH = joinpath("live_forecasts", "live_forecast_log.csv")

Base.@kwdef struct LiveVerifyConfig
    wait::Bool = false
    poll_seconds::Int = 300
    timeout_hours::Float64 = 4.0
    horizon_hours::Int = 1
    log_path::String = DEFAULT_LOG_PATH
end

function _usage()
    return """
    Usage:
      julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl [options]

    Options:
      --wait                 Poll Dst and verify the forecast when the target arrives.
      --no-wait              Only log the forecast row. This is the default.
      --poll-seconds=N       Poll interval for --wait. Default: 300.
      --timeout-hours=N      Maximum wait time for --wait. Default: 4.
      --horizon-hours=N      Forecast at least N hours beyond latest complete SWPC hour. Default: 1.
      --log=PATH             CSV log path. Default: live_forecasts/live_forecast_log.csv.
      --help                 Print this message.
    """
end

function _parse_args(args)::LiveVerifyConfig
    cfg = LiveVerifyConfig()
    wait = cfg.wait
    poll_seconds = cfg.poll_seconds
    timeout_hours = cfg.timeout_hours
    horizon_hours = cfg.horizon_hours
    log_path = cfg.log_path

    for arg in args
        if arg == "--help"
            println(_usage())
            exit(0)
        elseif arg == "--wait"
            wait = true
        elseif arg == "--no-wait"
            wait = false
        elseif startswith(arg, "--poll-seconds=")
            poll_seconds = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--timeout-hours=")
            timeout_hours = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--horizon-hours=")
            horizon_hours = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--log=")
            log_path = split(arg, "=", limit=2)[2]
        else
            error("Unknown argument: $arg\n$(_usage())")
        end
    end

    poll_seconds > 0 || throw(ArgumentError("--poll-seconds must be positive"))
    timeout_hours > 0 || throw(ArgumentError("--timeout-hours must be positive"))
    horizon_hours > 0 || throw(ArgumentError("--horizon-hours must be positive"))

    return LiveVerifyConfig(;
        wait=wait,
        poll_seconds=poll_seconds,
        timeout_hours=timeout_hours,
        horizon_hours=horizon_hours,
        log_path=log_path,
    )
end

_floor_hour(t::DateTime) = DateTime(year(t), month(t), day(t), hour(t))
_parse_dt(x) = x isa DateTime ? x : DateTime(String(x))

function _fetch_dst()
    resp = HTTP.get(KYOTO_DST_JSON_URL; connect_timeout=15, readtimeout=30)
    resp.status == 200 || error("Kyoto Dst HTTP status $(resp.status)")
    rows = JSON3.read(String(resp.body))
    times = DateTime[DateTime(String(r.time_tag)) for r in rows]
    dst = Float64[Float64(r.dst) for r in rows]
    return times, dst
end

function _window(df::DataFrame, t0::DateTime, t1::DateTime)
    mask = (df.time_tag .>= t0) .& (df.time_tag .< t1)
    return df[mask, :]
end

function _finite_mean(v, default::Float64)
    vals = Float64[Float64(x) for x in v if isfinite(Float64(x))]
    return isempty(vals) ? default : mean(vals)
end

function _drivers_for_window(plasma::DataFrame, mag::DataFrame,
                             t0::DateTime, t1::DateTime; fallback=nothing)
    if fallback === nothing
        v0, n0, bz0, by0 = 400.0, 5.0, 0.0, 0.0
    else
        v0, n0, bz0, by0 = fallback.V, fallback.n, fallback.Bz, fallback.By
    end

    p = _window(plasma, t0, t1)
    m = _window(mag, t0, t1)
    V = _finite_mean(p.speed, v0)
    n = _finite_mean(p.density, n0)
    Bz = _finite_mean(m.bz_gsm, bz0)
    By = _finite_mean(m.by_gsm, by0)
    Pdyn = 1.6726e-6 * n * V^2
    return (; V, n, Bz, By, Pdyn)
end

function _append_forecast!(log_path::String, row::DataFrame)
    dir = dirname(log_path)
    !isempty(dir) && mkpath(dir)
    CSV.write(log_path, row; append=isfile(log_path), writeheader=!isfile(log_path))
    return nrow(CSV.read(log_path, DataFrame))
end

function _update_observation!(log_path::String, row_idx::Int,
                              obs::Float64, residual::Float64, in_ci::Bool)
    df = CSV.read(log_path, DataFrame)
    allowmissing!(df, [:observation_dst_nt, :residual_dst_nt, :observed_in_90ci])
    df.observation_dst_nt[row_idx] = obs
    df.residual_dst_nt[row_idx] = residual
    df.observed_in_90ci[row_idx] = in_ci
    CSV.write(log_path, df)
    return nothing
end

function issue_forecast(cfg::LiveVerifyConfig)
    issue_time = now(UTC)
    plasma = fetch_swpc_plasma(; max_retries=3, retry_delay_sec=1.0)
    mag = fetch_swpc_mag(; max_retries=3, retry_delay_sec=1.0)
    dst_times, dst_vals = _fetch_dst()

    latest_common_sw = min(maximum(plasma.time_tag), maximum(mag.time_tag))
    latest_complete_hour = _floor_hour(latest_common_sw)
    latest_dst_time = dst_times[end]
    latest_dst = dst_vals[end]
    target_time = max(latest_complete_hour + Hour(cfg.horizon_hours),
                      latest_dst_time + Hour(1))

    recent_start = latest_common_sw - Hour(1)
    recent = _drivers_for_window(plasma, mag, recent_start, latest_common_sw)
    anchor_drivers = _drivers_for_window(
        plasma, mag, latest_dst_time, latest_dst_time + Hour(1);
        fallback=recent,
    )
    anchor_dst_star = pressure_correct_dst([latest_dst], [anchor_drivers.Pdyn])[1]

    coef_csv = joinpath(get_data_dir(), "real_sindy_discovery_coefficients.csv")
    ens_csv = joinpath(get_data_dir(), "real_ensemble_inclusion.csv")
    state = init_forecast(;
        coefficients_csv=coef_csv,
        ensemble_csv=ens_csv,
        t0=latest_dst_time,
        dst0=anchor_dst_star,
    )

    result = nothing
    used_drivers = recent
    step_time = latest_dst_time + Hour(1)
    while step_time <= target_time
        source_hour = step_time - Hour(1)
        drivers = if source_hour < latest_complete_hour
            _drivers_for_window(
                plasma, mag, source_hour, source_hour + Hour(1);
                fallback=recent,
            )
        else
            recent
        end
        used_drivers = drivers
        result = step_forecast!(
            state,
            step_time,
            drivers.V,
            drivers.Bz,
            drivers.By,
            drivers.n,
            drivers.Pdyn,
        )
        step_time += Hour(1)
    end

    pressure_term = 7.26 * sqrt(max(used_drivers.Pdyn, 0.0)) - 11.0
    pred_dst = result.dst_predicted + pressure_term
    ci05_dst = result.dst_ci_05 + pressure_term
    ci95_dst = result.dst_ci_95 + pressure_term
    model_steps = Int((target_time - latest_dst_time) / Hour(1))
    wall_horizon = (target_time - issue_time) / Hour(1)

    row = DataFrame(
        issue_time_utc=[string(issue_time)],
        latest_solar_wind_utc=[string(latest_common_sw)],
        latest_dst_time_utc=[string(latest_dst_time)],
        latest_dst_nt=[latest_dst],
        anchor_dst_star_nt=[anchor_dst_star],
        target_time_utc=[string(target_time)],
        horizon_hours=[wall_horizon],
        driver_assumption=["observed_solar_wind_until_latest_complete_hour_then_latest_60min_persistence; model_step_hours=$model_steps"],
        V_kms=[used_drivers.V],
        Bz_nt=[used_drivers.Bz],
        By_nt=[used_drivers.By],
        n_cm3=[used_drivers.n],
        Pdyn_npa=[used_drivers.Pdyn],
        pred_dst_star_nt=[result.dst_predicted],
        pred_dst_nt=[pred_dst],
        pred_dst_ci05_nt=[ci05_dst],
        pred_dst_ci95_nt=[ci95_dst],
        observation_dst_nt=[missing],
        residual_dst_nt=[missing],
        observed_in_90ci=[missing],
    )
    row_idx = _append_forecast!(cfg.log_path, row)

    println("Logged live forecast row $row_idx: $(cfg.log_path)")
    println("Issue UTC: $issue_time")
    println("Latest SWPC solar wind: $latest_common_sw")
    println("Latest observed Kyoto Dst: $latest_dst_time = $latest_dst nT")
    println("Target observation UTC: $target_time")
    println("Forecast Dst*: $(round(result.dst_predicted; digits=2)) nT")
    println(
        "Forecast Dst: $(round(pred_dst; digits=2)) nT; 90% CI " *
        "[$(round(ci05_dst; digits=2)), $(round(ci95_dst; digits=2))]"
    )
    println(
        "Forecast drivers: V=$(round(used_drivers.V; digits=1)) km/s, " *
        "Bz=$(round(used_drivers.Bz; digits=2)) nT, " *
        "By=$(round(used_drivers.By; digits=2)) nT, " *
        "n=$(round(used_drivers.n; digits=2)) cm^-3, " *
        "Pdyn=$(round(used_drivers.Pdyn; digits=3)) nPa"
    )

    return (; row_idx, target_time, pred_dst, ci05_dst, ci95_dst)
end

function wait_for_observation(cfg::LiveVerifyConfig, forecast)
    deadline = now(UTC) + Millisecond(round(Int, cfg.timeout_hours * 3600 * 1000))
    target = forecast.target_time

    while true
        times, dst = _fetch_dst()
        idx = findfirst(==(target), times)
        latest = times[end]

        if idx !== nothing
            obs = dst[idx]
            residual = obs - forecast.pred_dst
            in_ci = min(forecast.ci05_dst, forecast.ci95_dst) <= obs <=
                    max(forecast.ci05_dst, forecast.ci95_dst)
            _update_observation!(cfg.log_path, forecast.row_idx, obs, residual, in_ci)
            println("Observed target Dst arrived: $target = $obs nT")
            println(
                "Prediction: $(round(forecast.pred_dst; digits=2)) nT; " *
                "residual obs-pred = $(round(residual; digits=2)) nT; " *
                "in 90% CI = $in_ci"
            )
            return (; obs, residual, in_ci)
        end

        now(UTC) >= deadline && error(
            "Timed out waiting for target $target; latest Dst currently $latest"
        )
        println(
            "Waiting for target $target; latest Dst currently $latest. " *
            "Next check in $(cfg.poll_seconds) s."
        )
        sleep(cfg.poll_seconds)
    end
end

function main(args=ARGS)
    cfg = _parse_args(args)
    forecast = issue_forecast(cfg)
    cfg.wait && wait_for_observation(cfg, forecast)
    return nothing
end

main()
