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
const DEFAULT_REPORT_PATH = joinpath("live_forecasts", "live_comparison_report.md")
const DEFAULT_V2_CALIBRATION_PATH = joinpath(
    "live_forecasts", "operational_v2_calibration.csv"
)

Base.@kwdef struct LiveVerifyConfig
    mode::Symbol = :issue
    model::Symbol = :v1
    poll_seconds::Int = 300
    timeout_hours::Float64 = 4.0
    horizon_hours::Int = 1
    log_path::String = DEFAULT_LOG_PATH
    report_path::String = DEFAULT_REPORT_PATH
    replay_hours::Int = 48
    table_path::String = joinpath("live_forecasts", "live_replay_table.csv")
    table_limit::Int = 24
    v2_calibration_path::String = DEFAULT_V2_CALIBRATION_PATH
    v2_train_fraction::Float64 = 0.70
    v2_ridge::Float64 = 100.0
    v2_interval_coverage::Float64 = 0.90
end

function _usage()
    return """
    Usage:
      julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl [options]

    Modes:
      --issue                Issue and lock one future forecast row. This is the default.
      --verify-pending       Score all pending rows whose target Dst is now available.
      --backfill-baselines   Fill baseline forecasts/residuals for existing rows.
      --replay-recent        Build a recent causal replay table from live feeds.
      --fit-v2-calibration   Fit operational v2 calibration from --table CSV.
      --wait                 Issue one row, then poll until its target observation arrives.
      --summary              Print aggregate live-log scores.
      --comparison-report    Write standard locked-live comparison report.

    Options:
      --model=v1|v2          Forecast model to log/score. Default: v1.
      --poll-seconds=N       Poll interval for --wait. Default: 300.
      --timeout-hours=N      Maximum wait time for --wait. Default: 4.
      --horizon-hours=N      Hourly target index after issue time. Default: 1.
      --log=PATH             CSV log path. Default: live_forecasts/live_forecast_log.csv.
      --report=PATH          Markdown output path for --comparison-report.
                            Default: live_forecasts/live_comparison_report.md.
      --replay-hours=N       Recent hourly anchors for --replay-recent. Default: 48.
      --table=PATH           CSV output path for --replay-recent.
                            Default: live_forecasts/live_replay_table.csv.
      --table-limit=N        Number of recent rows to print for --replay-recent. Default: 24.
      --v2-calibration=PATH  Calibration path for --model=v2 or --fit-v2-calibration.
                            Default: live_forecasts/operational_v2_calibration.csv.
      --v2-train-fraction=N  Chronological fraction used to fit v2 calibration. Default: 0.70.
      --v2-ridge=N           Ridge penalty for v2 residual calibration. Default: 100.
      --v2-coverage=N        Target train coverage for v2 interval inflation. Default: 0.90.
      --help                 Print this message.
    """
end

function _parse_model(s::AbstractString)
    s == "v1" && return :v1
    s == "v2" && return :v2
    throw(ArgumentError("--model must be v1 or v2, got $s"))
end

function _parse_args(args)::LiveVerifyConfig
    cfg = LiveVerifyConfig()
    mode = cfg.mode
    model = cfg.model
    poll_seconds = cfg.poll_seconds
    timeout_hours = cfg.timeout_hours
    horizon_hours = cfg.horizon_hours
    log_path = cfg.log_path
    report_path = cfg.report_path
    replay_hours = cfg.replay_hours
    table_path = cfg.table_path
    table_limit = cfg.table_limit
    v2_calibration_path = cfg.v2_calibration_path
    v2_train_fraction = cfg.v2_train_fraction
    v2_ridge = cfg.v2_ridge
    v2_interval_coverage = cfg.v2_interval_coverage

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
        elseif arg == "--replay-recent"
            mode = :replay_recent
        elseif arg == "--fit-v2-calibration"
            mode = :fit_v2_calibration
        elseif arg == "--wait"
            mode = :wait
        elseif arg == "--summary"
            mode = :summary
        elseif arg == "--comparison-report"
            mode = :comparison_report
        elseif startswith(arg, "--model=")
            model = _parse_model(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--poll-seconds=")
            poll_seconds = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--timeout-hours=")
            timeout_hours = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--horizon-hours=")
            horizon_hours = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--log=")
            log_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--report=")
            report_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--replay-hours=")
            replay_hours = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--table=")
            table_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--table-limit=")
            table_limit = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-calibration=")
            v2_calibration_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--v2-train-fraction=")
            v2_train_fraction = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-ridge=")
            v2_ridge = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-coverage=")
            v2_interval_coverage = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg\n$(_usage())")
        end
    end

    poll_seconds > 0 || throw(ArgumentError("--poll-seconds must be positive"))
    timeout_hours > 0 || throw(ArgumentError("--timeout-hours must be positive"))
    horizon_hours > 0 || throw(ArgumentError("--horizon-hours must be positive"))
    replay_hours > 0 || throw(ArgumentError("--replay-hours must be positive"))
    table_limit >= 0 || throw(ArgumentError("--table-limit must be nonnegative"))
    0 < v2_train_fraction < 1 ||
        throw(ArgumentError("--v2-train-fraction must lie in (0, 1)"))
    v2_ridge >= 0 || throw(ArgumentError("--v2-ridge must be nonnegative"))
    0 < v2_interval_coverage < 1 ||
        throw(ArgumentError("--v2-coverage must lie in (0, 1)"))

    return LiveVerifyConfig(;
        mode=mode,
        model=model,
        poll_seconds=poll_seconds,
        timeout_hours=timeout_hours,
        horizon_hours=horizon_hours,
        log_path=log_path,
        report_path=report_path,
        replay_hours=replay_hours,
        table_path=table_path,
        table_limit=table_limit,
        v2_calibration_path=v2_calibration_path,
        v2_train_fraction=v2_train_fraction,
        v2_ridge=v2_ridge,
        v2_interval_coverage=v2_interval_coverage,
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

function _dst_lookup(dst_times, dst_vals)
    lookup = Dict{DateTime,Float64}()
    for (t, v) in zip(dst_times, dst_vals)
        lookup[_parse_dt(t)] = Float64(v)
    end
    return lookup
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

function _v2_features(latest_dst::Real, drivers)
    return (
        latest_dst_nt=Float64(latest_dst),
        V_kms=Float64(drivers.V),
        Bz_nt=Float64(drivers.Bz),
        By_nt=Float64(drivers.By),
        n_cm3=Float64(drivers.n),
        Pdyn_npa=Float64(drivers.Pdyn),
    )
end

function _load_calibration_for_model(cfg::LiveVerifyConfig)
    cfg.model == :v1 && return nothing
    cfg.model == :v2 || error("Unsupported model: $(cfg.model)")
    isfile(cfg.v2_calibration_path) || error(
        "Operational v2 calibration not found at $(cfg.v2_calibration_path). " *
        "Run --fit-v2-calibration first."
    )
    return read_operational_v2_calibration(cfg.v2_calibration_path)
end

function _select_model_prediction(model::Symbol, calibration,
                                  latest_dst::Real, drivers,
                                  v1_pred_dst::Real,
                                  v1_ci05_dst::Real,
                                  v1_ci95_dst::Real)
    if model == :v1
        return (
            model_version="v1",
            pred_dst=Float64(v1_pred_dst),
            ci05_dst=Float64(v1_ci05_dst),
            ci95_dst=Float64(v1_ci95_dst),
            v2_pred_dst=missing,
            v2_ci05_dst=missing,
            v2_ci95_dst=missing,
            v2_correction=missing,
            v2_interval_scale=missing,
            v2_label=missing,
        )
    elseif model == :v2
        calibration === nothing && error("v2 model requested without calibration")
        v2 = operational_v2_predict(
            calibration,
            v1_pred_dst,
            v1_ci05_dst,
            v1_ci95_dst,
            _v2_features(latest_dst, drivers),
        )
        return (
            model_version="v2",
            pred_dst=v2.pred_dst,
            ci05_dst=v2.ci05_dst,
            ci95_dst=v2.ci95_dst,
            v2_pred_dst=v2.pred_dst,
            v2_ci05_dst=v2.ci05_dst,
            v2_ci95_dst=v2.ci95_dst,
            v2_correction=v2.correction,
            v2_interval_scale=v2.interval_scale,
            v2_label=v2.label,
        )
    else
        error("Unsupported model: $model")
    end
end

function _replay_anchor_hours(plasma::DataFrame, mag::DataFrame, dst_times, replay_hours::Int)
    dst_dt = DateTime[_parse_dt(t) for t in dst_times]
    latest_complete_sw = _floor_hour(min(maximum(plasma.time_tag), maximum(mag.time_tag)))
    first_complete_sw = _floor_hour(max(minimum(plasma.time_tag), minimum(mag.time_tag))) + Hour(1)
    latest_dst_time = maximum(dst_dt)
    earliest_dst_time = minimum(dst_dt)

    last_anchor = min(latest_complete_sw, latest_dst_time - Hour(1))
    first_anchor = max(first_complete_sw, earliest_dst_time)
    first_anchor <= last_anchor || error("No overlapping replay window between SWPC solar wind and Kyoto Dst feeds")

    requested_start = last_anchor - Hour(replay_hours - 1)
    start_anchor = max(first_anchor, requested_start)
    return collect(start_anchor:Hour(1):last_anchor)
end

function _forecast_one_replay(anchor_time::DateTime, target_time::DateTime,
                              anchor_dst::Float64, drivers;
                              model::Symbol=:v1, calibration=nothing)
    anchor_dst_star = pressure_correct_dst([anchor_dst], [drivers.Pdyn])[1]
    coef_csv = joinpath(get_data_dir(), "real_sindy_discovery_coefficients.csv")
    ens_csv = joinpath(get_data_dir(), "real_ensemble_inclusion.csv")
    state = init_forecast(;
        coefficients_csv=coef_csv,
        ensemble_csv=ens_csv,
        t0=anchor_time,
        dst0=anchor_dst_star,
    )
    result = step_forecast!(
        state,
        target_time,
        drivers.V,
        drivers.Bz,
        drivers.By,
        drivers.n,
        drivers.Pdyn,
    )
    baselines = _advance_baselines(anchor_dst_star, drivers)
    v1_pred = _dst_from_dst_star(result.dst_predicted, drivers.Pdyn)
    v1_ci05 = _dst_from_dst_star(result.dst_ci_05, drivers.Pdyn)
    v1_ci95 = _dst_from_dst_star(result.dst_ci_95, drivers.Pdyn)
    selected = _select_model_prediction(
        model,
        calibration,
        anchor_dst,
        drivers,
        v1_pred,
        v1_ci05,
        v1_ci95,
    )

    return (
        model_version=selected.model_version,
        anchor_dst_star=anchor_dst_star,
        pred_dst_star=result.dst_predicted,
        pred_dst=selected.pred_dst,
        ci05_dst=selected.ci05_dst,
        ci95_dst=selected.ci95_dst,
        v1_pred_dst=v1_pred,
        v1_ci05_dst=v1_ci05,
        v1_ci95_dst=v1_ci95,
        v2_pred_dst=selected.v2_pred_dst,
        v2_ci05_dst=selected.v2_ci05_dst,
        v2_ci95_dst=selected.v2_ci95_dst,
        v2_correction=selected.v2_correction,
        v2_interval_scale=selected.v2_interval_scale,
        v2_label=selected.v2_label,
        persistence_dst=anchor_dst,
        burton_dst=_dst_from_dst_star(baselines.burton, drivers.Pdyn),
        burton_full_dst=_dst_from_dst_star(baselines.burton_full, drivers.Pdyn),
        obrien_dst=_dst_from_dst_star(baselines.obrien, drivers.Pdyn),
    )
end

function replay_recent_table(plasma::DataFrame, mag::DataFrame,
                             dst_times, dst_vals; replay_hours::Int=48,
                             model::Symbol=:v1, calibration=nothing)
    dst_map = _dst_lookup(dst_times, dst_vals)
    anchors = _replay_anchor_hours(plasma, mag, dst_times, replay_hours)
    rows = NamedTuple[]

    for anchor_time in anchors
        target_time = anchor_time + Hour(1)
        haskey(dst_map, anchor_time) || continue
        haskey(dst_map, target_time) || continue

        source_start = anchor_time - Hour(1)
        source_end = anchor_time
        drivers = _drivers_for_window(plasma, mag, source_start, source_end)
        if !all(isfinite, (drivers.V, drivers.n, drivers.Bz, drivers.By, drivers.Pdyn))
            continue
        end

        forecast = _forecast_one_replay(
            anchor_time,
            target_time,
            dst_map[anchor_time],
            drivers,
            model=model,
            calibration=calibration,
        )
        observed = dst_map[target_time]
        in_ci = min(forecast.ci05_dst, forecast.ci95_dst) <= observed <=
                max(forecast.ci05_dst, forecast.ci95_dst)

        push!(rows, (
            issue_time_utc=string(anchor_time),
            source_driver_start_utc=string(source_start),
            source_driver_end_utc=string(source_end),
            latest_dst_time_utc=string(anchor_time),
            target_time_utc=string(target_time),
            model_version=forecast.model_version,
            latest_dst_nt=dst_map[anchor_time],
            observation_dst_nt=observed,
            pred_dst_nt=forecast.pred_dst,
            residual_dst_nt=observed - forecast.pred_dst,
            pred_dst_ci05_nt=forecast.ci05_dst,
            pred_dst_ci95_nt=forecast.ci95_dst,
            observed_in_90ci=in_ci,
            v1_pred_dst_nt=forecast.v1_pred_dst,
            v1_pred_dst_ci05_nt=forecast.v1_ci05_dst,
            v1_pred_dst_ci95_nt=forecast.v1_ci95_dst,
            v2_pred_dst_nt=forecast.v2_pred_dst,
            v2_pred_dst_ci05_nt=forecast.v2_ci05_dst,
            v2_pred_dst_ci95_nt=forecast.v2_ci95_dst,
            v2_correction_dst_nt=forecast.v2_correction,
            v2_interval_scale=forecast.v2_interval_scale,
            v2_calibration_label=forecast.v2_label,
            persistence_dst_nt=forecast.persistence_dst,
            persistence_residual_dst_nt=observed - forecast.persistence_dst,
            burton_dst_nt=forecast.burton_dst,
            burton_residual_dst_nt=observed - forecast.burton_dst,
            burton_full_dst_nt=forecast.burton_full_dst,
            burton_full_residual_dst_nt=observed - forecast.burton_full_dst,
            obrien_dst_nt=forecast.obrien_dst,
            obrien_residual_dst_nt=observed - forecast.obrien_dst,
            V_kms=drivers.V,
            Bz_nt=drivers.Bz,
            By_nt=drivers.By,
            n_cm3=drivers.n,
            Pdyn_npa=drivers.Pdyn,
            replay_note="causal_replay_previous_complete_hour_driver_persistence",
        ))
    end

    isempty(rows) && error("No replay rows could be scored from the available feeds")
    return DataFrame(rows)
end

function _markdown_path(table_path::String)
    root, ext = splitext(table_path)
    return isempty(ext) ? table_path * ".md" : root * ".md"
end

function _fmt_cell(value)
    value isa AbstractString && return value
    value isa Bool && return string(value)
    value isa Real && return string(round(Float64(value); digits=2))
    return string(value)
end

function write_markdown_table(path::String, df::DataFrame; limit::Int=24)
    cols = [
        :issue_time_utc,
        :target_time_utc,
        :model_version,
        :observation_dst_nt,
        :pred_dst_nt,
        :residual_dst_nt,
        :observed_in_90ci,
        :v1_pred_dst_nt,
        :v2_pred_dst_nt,
        :persistence_dst_nt,
        :burton_dst_nt,
        :obrien_dst_nt,
    ]
    view_df = limit == 0 || nrow(df) <= limit ? df : last(df, limit)
    open(path, "w") do io
        println(io, "| ", join(String.(cols), " | "), " |")
        println(io, "| ", join(fill("---", length(cols)), " | "), " |")
        for row in eachrow(view_df)
            println(io, "| ", join((_fmt_cell(row[col]) for col in cols), " | "), " |")
        end
    end
    return path
end

function _print_replay_metrics(df::DataFrame)
    println("Recent causal replay rows: $(nrow(df))")
    for (name, col) in (
        ("Selected", :pred_dst_nt),
        ("SINDy-v1", :v1_pred_dst_nt),
        ("Operational-v2", :v2_pred_dst_nt),
        ("Persistence", :persistence_dst_nt),
        ("Burton", :burton_dst_nt),
        ("BurtonFull", :burton_full_dst_nt),
        ("OBrien", :obrien_dst_nt),
    )
        String(col) in names(df) || continue
        vals = df[!, col]
        all(ismissing, vals) && continue
        idx = .!ismissing.(vals)
        _print_metric(name, Float64.(vals[idx]), Float64.(df.observation_dst_nt[idx]))
    end
    println(
        "Selected 90% coverage n=$(nrow(df)) coverage=",
        round(mean(Bool.(df.observed_in_90ci)); digits=3),
    )
    return nothing
end

function run_replay_recent(cfg::LiveVerifyConfig)
    plasma = fetch_swpc_plasma(; max_retries=3, retry_delay_sec=1.0)
    mag = fetch_swpc_mag(; max_retries=3, retry_delay_sec=1.0)
    dst_times, dst_vals = _fetch_dst()
    calibration = _load_calibration_for_model(cfg)
    df = replay_recent_table(plasma, mag, dst_times, dst_vals;
        replay_hours=cfg.replay_hours,
        model=cfg.model,
        calibration=calibration,
    )

    dir = dirname(cfg.table_path)
    !isempty(dir) && mkpath(dir)
    CSV.write(cfg.table_path, df)
    md_path = _markdown_path(cfg.table_path)
    write_markdown_table(md_path, df; limit=cfg.table_limit)

    _print_replay_metrics(df)
    println("Wrote CSV table: $(cfg.table_path)")
    println("Wrote Markdown table: $md_path")
    return df
end

function _chronological_train_test(df::DataFrame, fraction::Float64)
    sort_cols = Symbol[]
    String(:issue_time_utc) in names(df) && push!(sort_cols, :issue_time_utc)
    !isempty(sort_cols) && sort!(df, sort_cols)
    n_train = floor(Int, fraction * nrow(df))
    n_train = clamp(n_train, 1, nrow(df) - 1)
    return df[1:n_train, :], df[n_train + 1:end, :]
end

function _v2_base_prediction_table(df::DataFrame)
    out = copy(df)
    v1_cols = [:v1_pred_dst_nt, :v1_pred_dst_ci05_nt, :v1_pred_dst_ci95_nt]
    if all(String(c) in names(out) for c in v1_cols)
        out[!, :pred_dst_nt] = Float64.(out.v1_pred_dst_nt)
        out[!, :pred_dst_ci05_nt] = Float64.(out.v1_pred_dst_ci05_nt)
        out[!, :pred_dst_ci95_nt] = Float64.(out.v1_pred_dst_ci95_nt)
    end
    return out
end

function fit_v2_calibration!(cfg::LiveVerifyConfig)
    isfile(cfg.table_path) || error("Replay table not found: $(cfg.table_path)")
    df = _v2_base_prediction_table(CSV.read(cfg.table_path, DataFrame))
    nrow(df) >= 4 || error("Need at least 4 replay rows to fit/test v2 calibration")
    train, test = _chronological_train_test(df, cfg.v2_train_fraction)
    cal = fit_operational_v2_calibration(
        train;
        ridge=cfg.v2_ridge,
        interval_coverage=cfg.v2_interval_coverage,
        label="operational_v2_ridge$(cfg.v2_ridge)_n$(nrow(train))",
    )
    write_operational_v2_calibration(cfg.v2_calibration_path, cal)

    train_scored = score_operational_v2(train, cal)
    test_scored = score_operational_v2(test, cal)
    scored = vcat(train_scored, test_scored; cols=:union)
    scored_path = replace(cfg.v2_calibration_path, r"\.csv$" => "_scored.csv")
    scored_path == cfg.v2_calibration_path &&
        (scored_path = cfg.v2_calibration_path * "_scored.csv")
    CSV.write(scored_path, scored)

    println("Fitted operational v2 calibration: $(cfg.v2_calibration_path)")
    println("Scored replay rows: $scored_path")
    println("Train rows: $(nrow(train)); held-out rows: $(nrow(test))")
    _print_metric(
        "Train SINDy-v1",
        Float64.(train.pred_dst_nt),
        Float64.(train.observation_dst_nt),
    )
    _print_metric(
        "Train v2",
        Float64.(train_scored.v2_pred_dst_nt),
        Float64.(train_scored.observation_dst_nt),
    )
    _print_metric(
        "Heldout SINDy-v1",
        Float64.(test.pred_dst_nt),
        Float64.(test.observation_dst_nt),
    )
    _print_metric(
        "Heldout v2",
        Float64.(test_scored.v2_pred_dst_nt),
        Float64.(test_scored.observation_dst_nt),
    )
    println("V2 interval scale: $(round(cal.interval_scale; digits=3))")
    return cal
end

function issue_forecast(cfg::LiveVerifyConfig)
    issue_time = now(UTC)
    plasma = fetch_swpc_plasma(; max_retries=3, retry_delay_sec=1.0)
    mag = fetch_swpc_mag(; max_retries=3, retry_delay_sec=1.0)
    dst_times, dst_vals = _fetch_dst()
    calibration = _load_calibration_for_model(cfg)

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

    v1_pred_dst = _dst_from_dst_star(result.dst_predicted, used_drivers.Pdyn)
    v1_ci05_dst = _dst_from_dst_star(result.dst_ci_05, used_drivers.Pdyn)
    v1_ci95_dst = _dst_from_dst_star(result.dst_ci_95, used_drivers.Pdyn)
    selected = _select_model_prediction(
        cfg.model,
        calibration,
        latest_dst,
        used_drivers,
        v1_pred_dst,
        v1_ci05_dst,
        v1_ci95_dst,
    )
    pred_dst = selected.pred_dst
    ci05_dst = selected.ci05_dst
    ci95_dst = selected.ci95_dst
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
        model_version=[selected.model_version],
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
        v1_pred_dst_nt=[v1_pred_dst],
        v1_pred_dst_ci05_nt=[v1_ci05_dst],
        v1_pred_dst_ci95_nt=[v1_ci95_dst],
        v2_pred_dst_nt=[selected.v2_pred_dst],
        v2_pred_dst_ci05_nt=[selected.v2_ci05_dst],
        v2_pred_dst_ci95_nt=[selected.v2_ci95_dst],
        v2_correction_dst_nt=[selected.v2_correction],
        v2_interval_scale=[selected.v2_interval_scale],
        v2_calibration_label=[selected.v2_label],
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
    println("Selected model: $(selected.model_version)")
    println("Lead time: $(round(wall_horizon; digits=3)) hr wall-clock, $model_steps model steps")
    println("Forecast Dst*: $(round(result.dst_predicted; digits=2)) nT")
    println(
        "Selected Dst: $(round(pred_dst; digits=2)) nT; 90% CI " *
        "[$(round(ci05_dst; digits=2)), $(round(ci95_dst; digits=2))]"
    )
    if selected.model_version == "v2"
        println(
            "SINDy-v1 Dst: $(round(v1_pred_dst; digits=2)) nT; " *
            "v2 correction=$(round(selected.v2_correction; digits=2)) nT; " *
            "interval scale=$(round(selected.v2_interval_scale; digits=2))"
        )
    end
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

    return (; row_idx, target_time, pred_dst, ci05_dst, ci95_dst,
            model_version=selected.model_version)
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

_fmt2(x) = ismissing(x) ? "" : string(round(Float64(x); digits=2))
_fmt3(x) = ismissing(x) ? "" : string(round(Float64(x); digits=3))
_fmt_bool(x) = ismissing(x) ? "" : string(Bool(x))
_fmt_text(x) = ismissing(x) ? "" : string(x)

function _metric_values(preds::Vector{Float64}, obs::Vector{Float64})
    isempty(preds) && return nothing
    residuals = obs .- preds
    return (
        n=length(preds),
        rmse=sqrt(mean(residuals .^ 2)),
        mae=mean(abs.(residuals)),
        bias=mean(residuals),
    )
end

function _metric_markdown_row(name::String, preds::Vector{Float64}, obs::Vector{Float64})
    m = _metric_values(preds, obs)
    m === nothing && return "| $name | 0 |  |  |  |"
    return "| $name | $(m.n) | $(_fmt2(m.rmse)) | $(_fmt2(m.mae)) | $(_fmt2(m.bias)) |"
end

function _verified_indices(df::DataFrame)
    String(:observation_dst_nt) in names(df) || return Int[]
    return [i for i in 1:nrow(df) if !ismissing(df[i, :observation_dst_nt])]
end

function _pending_indices(df::DataFrame)
    String(:observation_dst_nt) in names(df) || return collect(1:nrow(df))
    return [i for i in 1:nrow(df) if ismissing(df[i, :observation_dst_nt])]
end

function _row_model_version(df::DataFrame, row_idx::Int)
    if String(:model_version) in names(df)
        value = df[row_idx, :model_version]
        if !ismissing(value) && !isempty(String(value))
            return String(value)
        end
    end
    return "v1"
end

function _row_is_strictly_future(df::DataFrame, row_idx::Int)
    try
        issue = _parse_dt(df[row_idx, :issue_time_utc])
        target = _parse_dt(df[row_idx, :target_time_utc])
        target > issue || return false
        if String(:latest_dst_time_utc) in names(df) && !ismissing(df[row_idx, :latest_dst_time_utc])
            target > _parse_dt(df[row_idx, :latest_dst_time_utc]) || return false
        end
        return true
    catch
        return false
    end
end

function _coverage_fraction(df::DataFrame, rows::Vector{Int})
    String(:observed_in_90ci) in names(df) || return missing
    flags = Bool[]
    for i in rows
        value = df[i, :observed_in_90ci]
        ismissing(value) || push!(flags, Bool(value))
    end
    isempty(flags) && return missing
    return mean(flags)
end

function _standard_model_columns(df::DataFrame)
    specs = Pair{String,Symbol}[
        "Selected" => :pred_dst_nt,
        "SINDy-v1" => :v1_pred_dst_nt,
        "Operational-v2" => :v2_pred_dst_nt,
        "Persistence" => :persistence_dst_nt,
        "Burton" => :burton_dst_nt,
        "BurtonFull" => :burton_full_dst_nt,
        "OBrien" => :obrien_dst_nt,
    ]
    return [spec for spec in specs if String(last(spec)) in names(df)]
end

function _prediction_value(df::DataFrame, row_idx::Int, pred_col::Symbol)
    if pred_col == :v1_pred_dst_nt
        value = _optional_float(df, row_idx, :v1_pred_dst_nt)
        if ismissing(value) && _row_model_version(df, row_idx) == "v1"
            return _optional_float(df, row_idx, :pred_dst_nt)
        end
        return value
    end
    return _optional_float(df, row_idx, pred_col)
end

function _metric_rows_for_indices(df::DataFrame, pred_col::Symbol, rows::Vector{Int})
    preds = Float64[]
    obs = Float64[]
    if pred_col != :v1_pred_dst_nt && !(String(pred_col) in names(df))
        return preds, obs
    end
    for row_idx in rows
        observed = _optional_float(df, row_idx, :observation_dst_nt)
        predicted = _prediction_value(df, row_idx, pred_col)
        if !ismissing(observed) && !ismissing(predicted)
            push!(obs, observed)
            push!(preds, predicted)
        end
    end
    return preds, obs
end

function write_live_comparison_report(log_path::String, report_path::String)
    isfile(log_path) || error("No forecast log exists at $log_path")
    df = CSV.read(log_path, DataFrame)
    verified = _verified_indices(df)
    valid_verified = [i for i in verified if _row_is_strictly_future(df, i)]
    invalid_verified = setdiff(verified, valid_verified)
    pending = _pending_indices(df)
    model_specs = _standard_model_columns(df)
    coverage = _coverage_fraction(df, valid_verified)

    lines = String[]
    push!(lines, "# Locked Live Forecast Comparison Report")
    push!(lines, "")
    push!(lines, "Source log: `$log_path`")
    push!(lines, "Evidence tier: locked-live forecast, scored only after target observation publication.")
    push!(lines, "Generated UTC: $(Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"))Z")
    push!(lines, "")
    push!(lines, "Verified rows used: $(length(valid_verified))")
    push!(lines, "Invalid verified rows excluded: $(length(invalid_verified))")
    push!(lines, "Pending rows: $(length(pending))")
    if !ismissing(coverage)
        push!(lines, "Selected-model 90% interval coverage: $(_fmt3(coverage))")
    end

    push!(lines, "")
    push!(lines, "## Aggregate Metrics")
    push!(lines, "")
    push!(lines, "| model | n | RMSE nT | MAE nT | bias nT |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: |")
    for spec in model_specs
        preds, obs = _metric_rows_for_indices(df, last(spec), valid_verified)
        push!(lines, _metric_markdown_row(first(spec), preds, obs))
    end

    push!(lines, "")
    push!(lines, "## Verified Rows Used")
    push!(lines, "")
    push!(lines, "| issue UTC | target UTC | model | lead h | observed | selected pred | residual obs-pred | abs error | inside 90% CI | v1 pred | persistence | Burton | OBrien |")
    push!(lines, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for row_idx in valid_verified
        observed = _optional_float(df, row_idx, :observation_dst_nt)
        pred = _optional_float(df, row_idx, :pred_dst_nt)
        residual = _optional_float(df, row_idx, :residual_dst_nt)
        abs_error = ismissing(residual) ? missing : abs(residual)
        lead = _optional_float(df, row_idx, :wall_clock_lead_hours)
        if ismissing(lead)
            lead = _optional_float(df, row_idx, :horizon_hours)
        end
        in_ci = String(:observed_in_90ci) in names(df) ? df[row_idx, :observed_in_90ci] : missing
        model_version = _row_model_version(df, row_idx)
        push!(lines,
            "| $(_fmt_text(df[row_idx, :issue_time_utc])) | " *
            "$(_fmt_text(df[row_idx, :target_time_utc])) | " *
            "$(_fmt_text(model_version)) | " *
            "$(_fmt2(lead)) | $(_fmt2(observed)) | $(_fmt2(pred)) | " *
            "$(_fmt2(residual)) | $(_fmt2(abs_error)) | $(_fmt_bool(in_ci)) | " *
            "$(_fmt2(_prediction_value(df, row_idx, :v1_pred_dst_nt))) | " *
            "$(_fmt2(_optional_float(df, row_idx, :persistence_dst_nt))) | " *
            "$(_fmt2(_optional_float(df, row_idx, :burton_dst_nt))) | " *
            "$(_fmt2(_optional_float(df, row_idx, :obrien_dst_nt))) |"
        )
    end

    if !isempty(invalid_verified)
        push!(lines, "")
        push!(lines, "## Invalid Verified Rows Excluded")
        push!(lines, "")
        push!(lines, "| issue UTC | target UTC | reason |")
        push!(lines, "| --- | --- | --- |")
        for row_idx in invalid_verified
            push!(lines,
                "| $(_fmt_text(df[row_idx, :issue_time_utc])) | " *
                "$(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "target is not strictly after issue/latest observed Dst time |"
            )
        end
    end

    if !isempty(pending)
        push!(lines, "")
        push!(lines, "## Pending Rows")
        push!(lines, "")
        push!(lines, "| issue UTC | target UTC | model | selected pred | CI05 | CI95 |")
        push!(lines, "| --- | --- | --- | ---: | ---: | ---: |")
        for row_idx in pending
            model_version = _row_model_version(df, row_idx)
            push!(lines,
                "| $(_fmt_text(df[row_idx, :issue_time_utc])) | " *
                "$(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "$(_fmt_text(model_version)) | " *
                "$(_fmt2(_optional_float(df, row_idx, :pred_dst_nt))) | " *
                "$(_fmt2(_optional_float(df, row_idx, :pred_dst_ci05_nt))) | " *
                "$(_fmt2(_optional_float(df, row_idx, :pred_dst_ci95_nt))) |"
            )
        end
    end

    if !isempty(valid_verified)
        worst = sort(valid_verified; by=i -> abs(Float64(df[i, :residual_dst_nt])), rev=true)
        push!(lines, "")
        push!(lines, "## Worst Selected-Model Misses")
        push!(lines, "")
        push!(lines, "| target UTC | observed | selected pred | residual obs-pred | abs error | model |")
        push!(lines, "| --- | ---: | ---: | ---: | ---: | --- |")
        for row_idx in worst[1:min(10, length(worst))]
            model_version = _row_model_version(df, row_idx)
            residual = _optional_float(df, row_idx, :residual_dst_nt)
            push!(lines,
                "| $(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "$(_fmt2(_optional_float(df, row_idx, :observation_dst_nt))) | " *
                "$(_fmt2(_optional_float(df, row_idx, :pred_dst_nt))) | " *
                "$(_fmt2(residual)) | $(_fmt2(abs(residual))) | " *
                "$(_fmt_text(model_version)) |"
            )
        end
    end

    push!(lines, "")
    push!(lines, "## Standard Interpretation")
    push!(lines, "")
    push!(lines, "- A row is correct for point accuracy only by its absolute error against the locked target observation.")
    push!(lines, "- A row is correct for probabilistic coverage only if the observation falls inside the locked interval.")
    push!(lines, "- A model upgrade must improve locked-live RMSE and MAE against all mandatory baselines over enough rows, not only one row or a replay window.")
    push!(lines, "- Pending rows are not evidence for or against the model.")

    dir = dirname(report_path)
    !isempty(dir) && mkpath(dir)
    write(report_path, join(lines, "\n") * "\n")
    println("Wrote locked-live comparison report: $report_path")
    println(
        "Verified rows used: $(length(valid_verified)); " *
        "invalid verified rows excluded: $(length(invalid_verified)); " *
        "pending rows: $(length(pending))"
    )
    return report_path
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
    elseif cfg.mode == :replay_recent
        run_replay_recent(cfg)
    elseif cfg.mode == :fit_v2_calibration
        fit_v2_calibration!(cfg)
    elseif cfg.mode == :wait
        forecast = issue_forecast(cfg)
        wait_for_observation(cfg, forecast)
    elseif cfg.mode == :summary
        summarize_log(cfg.log_path)
    elseif cfg.mode == :comparison_report
        write_live_comparison_report(cfg.log_path, cfg.report_path)
    else
        error("Unsupported mode: $(cfg.mode)")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
