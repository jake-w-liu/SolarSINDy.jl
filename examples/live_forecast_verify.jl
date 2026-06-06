#!/usr/bin/env julia
# Issue live Dst forecasts and verify locked predictions against future Dst.
#
# Usage:
#   julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --wait
#
# The forecast row is written before the target observation exists. Verification
# only updates that row after the exact target timestamp appears in the Dst feed.

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
    mode::Symbol = :issue
    poll_seconds::Int = 300
    timeout_hours::Float64 = 4.0
    horizon_hours::Int = 1
    log_path::String = DEFAULT_LOG_PATH
end

function _usage()
    return """
    Usage:
      julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl [options]

    Modes:
      --issue                Issue and lock one future forecast row. This is the default.
      --verify-pending       Score all pending rows whose target Dst is now available.
      --backfill-baselines   Fill baseline forecasts/residuals for existing rows.
      --wait                 Issue one row, then poll until its target observation arrives.
      --summary              Print aggregate live-log scores.

    Options:
      --poll-seconds=N       Poll interval for --wait. Default: 300.
      --timeout-hours=N      Maximum wait time for --wait. Default: 4.
      --horizon-hours=N      Hourly target index after issue time. Default: 1.
      --log=PATH             CSV log path. Default: live_forecasts/live_forecast_log.csv.
      --help                 Print this message.
    """
end

function _parse_args(args)::LiveVerifyConfig
    cfg = LiveVerifyConfig()
    mode = cfg.mode
    poll_seconds = cfg.poll_seconds
    timeout_hours = cfg.timeout_hours
    horizon_hours = cfg.horizon_hours
    log_path = cfg.log_path

    for arg in args
        if arg == "--help"
            println(_usage())
            exit(0)
        elseif arg == "--issue" || arg == "--no-wait"
            mode = :issue
        elseif arg == "--verify-pending"
            mode = :verify_pending
        elseif arg == "--backfill-baselines"
            mode = :backfill_baselines
        elseif arg == "--wait"
            mode = :wait
        elseif arg == "--summary"
            mode = :summary
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
        mode=mode,
        poll_seconds=poll_seconds,
        timeout_hours=timeout_hours,
        horizon_hours=horizon_hours,
        log_path=log_path,
    )
end

_floor_hour(t::DateTime) = DateTime(year(t), month(t), day(t), hour(t))
_parse_dt(x) = x isa DateTime ? x : DateTime(String(x))
_dst_from_dst_star(dst_star::Real, pdyn::Real) = dst_star + 7.26 * sqrt(max(pdyn, 0.0)) - 11.0

function _next_hourly_target(issue_time::DateTime, horizon_hours::Int,
                             latest_dst_time::DateTime)
    target_time = _floor_hour(issue_time) + Hour(horizon_hours)
    while target_time <= issue_time || target_time <= latest_dst_time
        target_time += Hour(1)
    end
    return target_time
end

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
    if isfile(log_path)
        df = CSV.read(log_path, DataFrame)
        df = vcat(df, row; cols=:union)
    else
        df = row
    end
    CSV.write(log_path, df)
    return nrow(df)
end

function _set_value!(df::DataFrame, row_idx::Int, col::Symbol, value)
    if !(String(col) in names(df))
        df[!, col] = fill(missing, nrow(df))
    end
    if eltype(df[!, col]) === Missing
        df[!, col] = Vector{Union{Missing, typeof(value)}}(missing, nrow(df))
    else
        allowmissing!(df, col)
    end
    df[row_idx, col] = value
    return nothing
end

function _optional_float(df::DataFrame, row_idx::Int, col::Symbol)
    String(col) in names(df) || return missing
    value = df[row_idx, col]
    ismissing(value) && return missing
    return Float64(value)
end

function _score_row!(df::DataFrame, row_idx::Int, observed_dst::Float64)
    pred = Float64(df[row_idx, :pred_dst_nt])
    ci05 = Float64(df[row_idx, :pred_dst_ci05_nt])
    ci95 = Float64(df[row_idx, :pred_dst_ci95_nt])
    _set_value!(df, row_idx, :observation_dst_nt, observed_dst)
    _set_value!(df, row_idx, :residual_dst_nt, observed_dst - pred)
    _set_value!(df, row_idx, :observed_in_90ci, min(ci05, ci95) <= observed_dst <= max(ci05, ci95))

    for (pred_col, residual_col) in (
        (:persistence_dst_nt, :persistence_residual_dst_nt),
        (:burton_dst_nt, :burton_residual_dst_nt),
        (:burton_full_dst_nt, :burton_full_residual_dst_nt),
        (:obrien_dst_nt, :obrien_residual_dst_nt),
    )
        baseline_pred = _optional_float(df, row_idx, pred_col)
        ismissing(baseline_pred) || _set_value!(df, row_idx, residual_col, observed_dst - baseline_pred)
    end

    return (; observed_dst, residual=observed_dst - pred)
end

function _advance_baselines(dst_star::Float64, drivers)
    Bs = max(-drivers.Bz, 0.0)
    V = [drivers.V]
    Bs_vec = [Bs]
    dst_vec = [dst_star]
    d_burton = clamp(burton_model(V, Bs_vec, dst_vec)[1], -200.0, 200.0)
    d_burton_full = clamp(burton_model_full(V, Bs_vec, dst_vec)[1], -200.0, 200.0)
    d_obrien = clamp(obrien_mcpherron_model(V, Bs_vec, dst_vec)[1], -200.0, 200.0)
    return (
        burton=clamp(dst_star + d_burton, -2000.0, 50.0),
        burton_full=clamp(dst_star + d_burton_full, -2000.0, 50.0),
        obrien=clamp(dst_star + d_obrien, -2000.0, 50.0),
    )
end

function _model_step_hours(df::DataFrame, row_idx::Int)
    existing = _optional_float(df, row_idx, :model_step_hours)
    ismissing(existing) || return max(1, round(Int, existing))

    target = _parse_dt(df[row_idx, :target_time_utc])
    latest_dst = _parse_dt(df[row_idx, :latest_dst_time_utc])
    return max(1, round(Int, (target - latest_dst) / Hour(1)))
end

function _row_drivers(df::DataFrame, row_idx::Int)
    return (
        V=Float64(df[row_idx, :V_kms]),
        Bz=Float64(df[row_idx, :Bz_nt]),
        By=Float64(df[row_idx, :By_nt]),
        n=Float64(df[row_idx, :n_cm3]),
        Pdyn=Float64(df[row_idx, :Pdyn_npa]),
    )
end

function _baseline_predictions_from_row(df::DataFrame, row_idx::Int)
    drivers = _row_drivers(df, row_idx)
    n_steps = _model_step_hours(df, row_idx)
    burton_star = Float64(df[row_idx, :anchor_dst_star_nt])
    burton_full_star = burton_star
    obrien_star = burton_star

    for _ in 1:n_steps
        baselines = _advance_baselines(burton_star, drivers)
        burton_star = baselines.burton
        baselines = _advance_baselines(burton_full_star, drivers)
        burton_full_star = baselines.burton_full
        baselines = _advance_baselines(obrien_star, drivers)
        obrien_star = baselines.obrien
    end

    return (
        persistence=Float64(df[row_idx, :latest_dst_nt]),
        burton=_dst_from_dst_star(burton_star, drivers.Pdyn),
        burton_full=_dst_from_dst_star(burton_full_star, drivers.Pdyn),
        obrien=_dst_from_dst_star(obrien_star, drivers.Pdyn),
        model_steps=n_steps,
    )
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
    target_time = _next_hourly_target(issue_time, cfg.horizon_hours, latest_dst_time)
    @assert target_time > issue_time
    @assert target_time > latest_dst_time

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
    burton_star = anchor_dst_star
    burton_full_star = anchor_dst_star
    obrien_star = anchor_dst_star

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
        baselines = _advance_baselines(burton_star, drivers)
        burton_star = baselines.burton
        baselines = _advance_baselines(burton_full_star, drivers)
        burton_full_star = baselines.burton_full
        baselines = _advance_baselines(obrien_star, drivers)
        obrien_star = baselines.obrien
        step_time += Hour(1)
    end

    pred_dst = _dst_from_dst_star(result.dst_predicted, used_drivers.Pdyn)
    ci05_dst = _dst_from_dst_star(result.dst_ci_05, used_drivers.Pdyn)
    ci95_dst = _dst_from_dst_star(result.dst_ci_95, used_drivers.Pdyn)
    persistence_dst = latest_dst
    burton_dst = _dst_from_dst_star(burton_star, used_drivers.Pdyn)
    burton_full_dst = _dst_from_dst_star(burton_full_star, used_drivers.Pdyn)
    obrien_dst = _dst_from_dst_star(obrien_star, used_drivers.Pdyn)
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
        wall_clock_lead_hours=[wall_horizon],
        model_step_hours=[model_steps],
        driver_assumption=["observed_solar_wind_until_latest_complete_hour_then_latest_60min_persistence"],
        V_kms=[used_drivers.V],
        Bz_nt=[used_drivers.Bz],
        By_nt=[used_drivers.By],
        n_cm3=[used_drivers.n],
        Pdyn_npa=[used_drivers.Pdyn],
        pred_dst_star_nt=[result.dst_predicted],
        pred_dst_nt=[pred_dst],
        pred_dst_ci05_nt=[ci05_dst],
        pred_dst_ci95_nt=[ci95_dst],
        persistence_dst_nt=[persistence_dst],
        burton_dst_nt=[burton_dst],
        burton_full_dst_nt=[burton_full_dst],
        obrien_dst_nt=[obrien_dst],
        observation_dst_nt=[missing],
        residual_dst_nt=[missing],
        observed_in_90ci=[missing],
        persistence_residual_dst_nt=[missing],
        burton_residual_dst_nt=[missing],
        burton_full_residual_dst_nt=[missing],
        obrien_residual_dst_nt=[missing],
    )
    row_idx = _append_forecast!(cfg.log_path, row)

    println("Logged live forecast row $row_idx: $(cfg.log_path)")
    println("Issue UTC: $issue_time")
    println("Latest SWPC solar wind: $latest_common_sw")
    println("Latest observed Kyoto Dst: $latest_dst_time = $latest_dst nT")
    println("Target observation UTC: $target_time")
    println("Lead time: $(round(wall_horizon; digits=3)) hr wall-clock, $model_steps model steps")
    println("Forecast Dst*: $(round(result.dst_predicted; digits=2)) nT")
    println(
        "SINDy Dst: $(round(pred_dst; digits=2)) nT; 90% CI " *
        "[$(round(ci05_dst; digits=2)), $(round(ci95_dst; digits=2))]"
    )
    println(
        "Baselines Dst: persistence=$(round(persistence_dst; digits=2)), " *
        "Burton=$(round(burton_dst; digits=2)), " *
        "BurtonFull=$(round(burton_full_dst; digits=2)), " *
        "OBrien=$(round(obrien_dst; digits=2))"
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

function verify_pending!(cfg::LiveVerifyConfig; dst_times=nothing, dst_vals=nothing)
    isfile(cfg.log_path) || error("No forecast log exists at $(cfg.log_path)")
    df = CSV.read(cfg.log_path, DataFrame)
    if dst_times === nothing || dst_vals === nothing
        dst_times, dst_vals = _fetch_dst()
    end

    verified = 0
    for row_idx in 1:nrow(df)
        if String(:observation_dst_nt) in names(df) && !ismissing(df[row_idx, :observation_dst_nt])
            continue
        end
        target = _parse_dt(df[row_idx, :target_time_utc])
        idx = findfirst(==(target), dst_times)
        idx === nothing && continue
        _score_row!(df, row_idx, Float64(dst_vals[idx]))
        verified += 1
    end

    verified > 0 && CSV.write(cfg.log_path, df)
    println("Verified $verified pending forecast row(s).")
    return verified
end

function backfill_baselines!(log_path::String)
    isfile(log_path) || error("No forecast log exists at $log_path")
    df = CSV.read(log_path, DataFrame)
    updated = 0

    for row_idx in 1:nrow(df)
        required = (:anchor_dst_star_nt, :latest_dst_nt, :target_time_utc,
                    :latest_dst_time_utc, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa)
        all(String(col) in names(df) for col in required) || continue
        baseline = _baseline_predictions_from_row(df, row_idx)

        _set_value!(df, row_idx, :persistence_dst_nt, baseline.persistence)
        _set_value!(df, row_idx, :burton_dst_nt, baseline.burton)
        _set_value!(df, row_idx, :burton_full_dst_nt, baseline.burton_full)
        _set_value!(df, row_idx, :obrien_dst_nt, baseline.obrien)
        _set_value!(df, row_idx, :model_step_hours, baseline.model_steps)

        observed = _optional_float(df, row_idx, :observation_dst_nt)
        if !ismissing(observed)
            _score_row!(df, row_idx, observed)
        end
        updated += 1
    end

    updated > 0 && CSV.write(log_path, df)
    println("Backfilled baseline forecasts for $updated row(s).")
    return updated
end

function wait_for_observation(cfg::LiveVerifyConfig, forecast)
    deadline = now(UTC) + Millisecond(round(Int, cfg.timeout_hours * 3600 * 1000))
    target = forecast.target_time

    while true
        times, dst = _fetch_dst()
        idx = findfirst(==(target), times)
        latest = times[end]

        if idx !== nothing
            df = CSV.read(cfg.log_path, DataFrame)
            result = _score_row!(df, forecast.row_idx, Float64(dst[idx]))
            CSV.write(cfg.log_path, df)
            in_ci = df[forecast.row_idx, :observed_in_90ci]
            println("Observed target Dst arrived: $target = $(result.observed_dst) nT")
            println(
                "Prediction: $(round(forecast.pred_dst; digits=2)) nT; " *
                "residual obs-pred = $(round(result.residual; digits=2)) nT; " *
                "in 90% CI = $in_ci"
            )
            return result
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

function _metric_rows(df::DataFrame, pred_col::Symbol)
    String(pred_col) in names(df) || return Float64[], Float64[]
    preds = Float64[]
    obs = Float64[]
    for row_idx in 1:nrow(df)
        observed = _optional_float(df, row_idx, :observation_dst_nt)
        predicted = _optional_float(df, row_idx, pred_col)
        if !ismissing(observed) && !ismissing(predicted)
            push!(obs, observed)
            push!(preds, predicted)
        end
    end
    return preds, obs
end

function _print_metric(name::String, preds::Vector{Float64}, obs::Vector{Float64})
    if isempty(preds)
        println(rpad(name, 16), " no verified rows")
        return nothing
    end
    residuals = obs .- preds
    rmse_val = sqrt(mean(residuals .^ 2))
    mae_val = mean(abs.(residuals))
    bias_val = mean(residuals)
    println(
        rpad(name, 16),
        " n=", length(preds),
        " RMSE=", round(rmse_val; digits=2),
        " MAE=", round(mae_val; digits=2),
        " bias=", round(bias_val; digits=2),
    )
    return nothing
end

function summarize_log(log_path::String)
    isfile(log_path) || error("No forecast log exists at $log_path")
    df = CSV.read(log_path, DataFrame)
    println("Live forecast log: $log_path")
    for (name, col) in (
        ("SINDy", :pred_dst_nt),
        ("Persistence", :persistence_dst_nt),
        ("Burton", :burton_dst_nt),
        ("BurtonFull", :burton_full_dst_nt),
        ("OBrien", :obrien_dst_nt),
    )
        preds, obs = _metric_rows(df, col)
        _print_metric(name, preds, obs)
    end
    if String(:observed_in_90ci) in names(df)
        flags = Bool[]
        for value in df.observed_in_90ci
            ismissing(value) || push!(flags, Bool(value))
        end
        isempty(flags) || println("SINDy 90% coverage n=$(length(flags)) coverage=$(round(mean(flags); digits=3))")
    end
    return nothing
end

function main(args=ARGS)
    cfg = _parse_args(args)
    if cfg.mode == :issue
        issue_forecast(cfg)
    elseif cfg.mode == :verify_pending
        verify_pending!(cfg)
    elseif cfg.mode == :backfill_baselines
        backfill_baselines!(cfg.log_path)
    elseif cfg.mode == :wait
        forecast = issue_forecast(cfg)
        wait_for_observation(cfg, forecast)
    elseif cfg.mode == :summary
        summarize_log(cfg.log_path)
    else
        error("Unsupported mode: $(cfg.mode)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
