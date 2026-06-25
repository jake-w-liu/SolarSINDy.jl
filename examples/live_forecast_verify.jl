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
const DEFAULT_OMNI_EXTRACTED_PATH = normpath(joinpath(
    @__DIR__, "..", "data", "omni_extracted.csv"
))

Base.@kwdef struct LiveVerifyConfig
    mode::Symbol = :issue
    model::Symbol = :v1
    poll_seconds::Int = 300
    timeout_hours::Float64 = 4.0
    horizon_hours::Int = 1
    campaign_horizons::Vector{Int} = Int[1, 2, 3, 6]
    log_path::String = DEFAULT_LOG_PATH
    report_path::String = DEFAULT_REPORT_PATH
    replay_hours::Int = 48
    replay_horizons::Vector{Int} = Int[1]
    table_path::String = joinpath("live_forecasts", "live_replay_table.csv")
    table_limit::Int = 24
    omni_path::String = DEFAULT_OMNI_EXTRACTED_PATH
    omni_year_start::Int = 2024
    omni_year_end::Int = 2025
    v2_calibration_path::String = DEFAULT_V2_CALIBRATION_PATH
    v2_train_fraction::Float64 = 0.70
    v2_validation_fraction::Float64 = 0.15
    v2_ridge::Float64 = 100.0
    v2_ridge_grid::Vector{Float64} = Float64[0.0, 1.0, 10.0, 100.0, 1000.0]
    v2_interval_coverage::Float64 = 0.90
    v2_selector_margin_nt::Float64 = 0.5
    v2_coverage_floor::Float64 = 0.85
end

# Below this many fit rows, do not let a baseline component win the selector on
# noise — serve the corrected v2 center (robustness guard against tiny-sample
# selection).
const V2_MIN_COMPONENT_ROWS = 12

# F5/M3: minimum validation rows before the coverage-floor acceptance gate is
# trusted. With a 1–2 row validation split the empirical coverage collapses to a
# degenerate 0/1 (or 0/0.5/1) estimate, so the gate cannot meaningfully enforce
# the 0.85 floor; below this count the v2 candidate is rejected in favor of the
# v1-equivalent fallback. Set to the smallest validation size already exercised
# by the deployed-calibration regression fixtures so the guard rejects only the
# genuinely degenerate splits and stays a no-op on valid multi-row calibrations.
const V2_MIN_VALIDATION_ROWS = 3

function _usage()
    return """
    Usage:
      julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl [options]

    Modes:
      --issue                Issue and lock one future forecast row. This is the default.
      --verify-pending       Score all pending rows whose target Dst is now available.
      --refresh-observations Reconcile logged observations with the current Dst feed.
      --backfill-baselines   Fill baseline forecasts/residuals for existing rows.
      --replay-recent        Build a recent causal replay table from live feeds.
      --replay-omni          Build a longer causal replay table from local OMNI CSV.
      --fit-v2-calibration   Fit operational v2 calibration from --table CSV.
      --wait                 Issue one row, then poll until its target observation arrives.
      --campaign             Issue multiple operational-v2 horizons, verify, and report.
      --summary              Print aggregate live-log scores.
      --comparison-report    Write standard locked-live comparison report.

    Options:
      --model=v1|v2          Forecast model to log/score. Default: v1, except
                            --campaign defaults to v2.
      --poll-seconds=N       Poll interval for --wait/--campaign. Default: 300.
      --timeout-hours=N      Maximum wait time for --wait/--campaign. Default: 4.
      --horizon-hours=N      Hourly target index after issue time. Default: 1.
      --campaign-horizons=A,B
                            Comma-separated target horizons for --campaign.
                            Default: 1,2,3,6.
      --log=PATH             CSV log path. Default: live_forecasts/live_forecast_log.csv.
      --report=PATH          Markdown output path for --comparison-report.
                            Default: live_forecasts/live_comparison_report.md.
      --replay-hours=N       Recent hourly anchors for --replay-recent. Default: 48.
      --replay-horizons=A,B  Lead times [hr] emitted per anchor in replay tables.
                            Default: 1. Use e.g. 1,2,3,6 to build a multi-horizon
                            conformal calibration table.
      --table=PATH           CSV output path for --replay-recent.
                            Default: live_forecasts/live_replay_table.csv.
      --table-limit=N        Number of recent rows to print for --replay-recent. Default: 24.
      --omni=PATH            Extracted OMNI CSV for --replay-omni.
                            Default: data/omni_extracted.csv.
      --omni-year-start=N    First OMNI year loaded for --replay-omni. Default: 2024.
      --omni-year-end=N      Last OMNI year loaded for --replay-omni. Default: 2025.
      --v2-calibration=PATH  Calibration path for --model=v2 or --fit-v2-calibration.
                            Default: live_forecasts/operational_v2_calibration.csv.
      --v2-train-fraction=N  Chronological fraction used to fit v2 candidates. Default: 0.70.
      --v2-validation-fraction=N
                            Chronological fraction used to select v2 candidates. Default: 0.15.
      --v2-ridge=N           Single ridge penalty for v2 residual calibration.
                            Overrides --v2-ridge-grid when provided. Default: 100.
      --v2-ridge-grid=A,B    Ridge penalties to tune on validation rows.
                            Default: 0,1,10,100,1000.
      --v2-coverage=N        Target train coverage for v2 interval inflation. Default: 0.90.
      --v2-selector-margin=N Guard margin retained in v2 calibration metadata.
                            Default: 0.5 nT.
      --v2-coverage-floor=N  Minimum validation 90% interval coverage a v2
                            candidate must meet to deploy; otherwise a
                            v1-equivalent fallback is used. Default: 0.85.
      --help                 Print this message.
    """
end

function _parse_model(s::AbstractString)
    s == "v1" && return :v1
    s == "v2" && return :v2
    throw(ArgumentError("--model must be v1 or v2, got $s"))
end

function _parse_horizons(s::AbstractString)
    vals = Int[]
    for part in split(s, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(vals, parse(Int, stripped))
    end
    isempty(vals) && throw(ArgumentError("--campaign-horizons must not be empty"))
    any(<=(0), vals) && throw(ArgumentError("--campaign-horizons must be positive integers"))
    return unique(vals)
end

function _parse_float_list(s::AbstractString, name::AbstractString)
    vals = Float64[]
    for part in split(s, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(vals, parse(Float64, stripped))
    end
    isempty(vals) && throw(ArgumentError("$name must not be empty"))
    return unique(vals)
end

function _parse_args(args)::LiveVerifyConfig
    cfg = LiveVerifyConfig()
    mode = cfg.mode
    model = cfg.model
    model_explicit = false
    poll_seconds = cfg.poll_seconds
    timeout_hours = cfg.timeout_hours
    horizon_hours = cfg.horizon_hours
    campaign_horizons = cfg.campaign_horizons
    log_path = cfg.log_path
    report_path = cfg.report_path
    replay_hours = cfg.replay_hours
    replay_horizons = copy(cfg.replay_horizons)
    table_path = cfg.table_path
    table_limit = cfg.table_limit
    omni_path = cfg.omni_path
    omni_year_start = cfg.omni_year_start
    omni_year_end = cfg.omni_year_end
    v2_calibration_path = cfg.v2_calibration_path
    v2_train_fraction = cfg.v2_train_fraction
    v2_validation_fraction = cfg.v2_validation_fraction
    v2_ridge = cfg.v2_ridge
    v2_ridge_grid = copy(cfg.v2_ridge_grid)
    v2_interval_coverage = cfg.v2_interval_coverage
    v2_selector_margin_nt = cfg.v2_selector_margin_nt
    v2_coverage_floor = cfg.v2_coverage_floor

    for arg in args
        if arg == "--help"
            println(_usage())
            exit(0)
        elseif arg == "--issue" || arg == "--no-wait"
            mode = :issue
        elseif arg == "--verify-pending"
            mode = :verify_pending
        elseif arg == "--refresh-observations"
            mode = :refresh_observations
        elseif arg == "--backfill-baselines"
            mode = :backfill_baselines
        elseif arg == "--replay-recent"
            mode = :replay_recent
        elseif arg == "--replay-omni"
            mode = :replay_omni
        elseif arg == "--fit-v2-calibration"
            mode = :fit_v2_calibration
        elseif arg == "--wait"
            mode = :wait
        elseif arg == "--campaign"
            mode = :campaign
        elseif arg == "--summary"
            mode = :summary
        elseif arg == "--comparison-report"
            mode = :comparison_report
        elseif startswith(arg, "--model=")
            model = _parse_model(split(arg, "=", limit=2)[2])
            model_explicit = true
        elseif startswith(arg, "--poll-seconds=")
            poll_seconds = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--timeout-hours=")
            timeout_hours = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--horizon-hours=")
            horizon_hours = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--campaign-horizons=")
            campaign_horizons = _parse_horizons(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--log=")
            log_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--report=")
            report_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--replay-hours=")
            replay_hours = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--replay-horizons=")
            replay_horizons = _parse_horizons(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--table=")
            table_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--table-limit=")
            table_limit = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--omni=")
            omni_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--omni-year-start=")
            omni_year_start = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--omni-year-end=")
            omni_year_end = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-calibration=")
            v2_calibration_path = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--v2-train-fraction=")
            v2_train_fraction = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-validation-fraction=")
            v2_validation_fraction = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-ridge=")
            v2_ridge = parse(Float64, split(arg, "=", limit=2)[2])
            v2_ridge_grid = Float64[v2_ridge]
        elseif startswith(arg, "--v2-ridge-grid=")
            v2_ridge_grid = _parse_float_list(
                split(arg, "=", limit=2)[2],
                "--v2-ridge-grid",
            )
        elseif startswith(arg, "--v2-coverage=")
            v2_interval_coverage = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-selector-margin=")
            v2_selector_margin_nt = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v2-coverage-floor=")
            v2_coverage_floor = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg\n$(_usage())")
        end
    end

    poll_seconds > 0 || throw(ArgumentError("--poll-seconds must be positive"))
    timeout_hours > 0 || throw(ArgumentError("--timeout-hours must be positive"))
    horizon_hours > 0 || throw(ArgumentError("--horizon-hours must be positive"))
    isempty(campaign_horizons) && throw(ArgumentError("--campaign-horizons must not be empty"))
    any(<=(0), campaign_horizons) &&
        throw(ArgumentError("--campaign-horizons must be positive integers"))
    replay_hours > 0 || throw(ArgumentError("--replay-hours must be positive"))
    (isempty(replay_horizons) || any(<=(0), replay_horizons)) &&
        throw(ArgumentError("--replay-horizons must be positive integers"))
    table_limit >= 0 || throw(ArgumentError("--table-limit must be nonnegative"))
    omni_year_start <= omni_year_end ||
        throw(ArgumentError("--omni-year-start must be <= --omni-year-end"))
    0 < v2_train_fraction < 1 ||
        throw(ArgumentError("--v2-train-fraction must lie in (0, 1)"))
    0 < v2_validation_fraction < 1 ||
        throw(ArgumentError("--v2-validation-fraction must lie in (0, 1)"))
    v2_train_fraction + v2_validation_fraction < 1 ||
        throw(ArgumentError("--v2-train-fraction + --v2-validation-fraction must be < 1"))
    v2_ridge >= 0 || throw(ArgumentError("--v2-ridge must be nonnegative"))
    isempty(v2_ridge_grid) && throw(ArgumentError("--v2-ridge-grid must not be empty"))
    all(x -> isfinite(x) && x >= 0.0, v2_ridge_grid) ||
        throw(ArgumentError("--v2-ridge-grid entries must be finite and nonnegative"))
    0 < v2_interval_coverage < 1 ||
        throw(ArgumentError("--v2-coverage must lie in (0, 1)"))
    v2_selector_margin_nt >= 0 ||
        throw(ArgumentError("--v2-selector-margin must be nonnegative"))
    0 < v2_coverage_floor < 1 ||
        throw(ArgumentError("--v2-coverage-floor must lie in (0, 1)"))

    if mode == :campaign && !model_explicit
        model = :v2
    end

    return LiveVerifyConfig(;
        mode=mode,
        model=model,
        poll_seconds=poll_seconds,
        timeout_hours=timeout_hours,
        horizon_hours=horizon_hours,
        campaign_horizons=campaign_horizons,
        log_path=log_path,
        report_path=report_path,
        replay_hours=replay_hours,
        replay_horizons=replay_horizons,
        table_path=table_path,
        table_limit=table_limit,
        omni_path=omni_path,
        omni_year_start=omni_year_start,
        omni_year_end=omni_year_end,
        v2_calibration_path=v2_calibration_path,
        v2_train_fraction=v2_train_fraction,
        v2_validation_fraction=v2_validation_fraction,
        v2_ridge=v2_ridge,
        v2_ridge_grid=v2_ridge_grid,
        v2_interval_coverage=v2_interval_coverage,
        v2_selector_margin_nt=v2_selector_margin_nt,
        v2_coverage_floor=v2_coverage_floor,
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
    times = DateTime[]; dst = Float64[]
    for r in rows
        tt = get(r, :time_tag, nothing); tt === nothing && continue
        t = tryparse(DateTime, String(tt)); t === nothing && continue
        dv = get(r, :dst, nothing)
        v = dv === nothing ? nothing : tryparse(Float64, string(dv))
        (v === nothing || !isfinite(v) || abs(v) > 9000) && continue   # reject null / fill sentinel (real Dst |x| ≪ 9000)
        push!(times, t); push!(dst, v)
    end
    isempty(dst) && error("Kyoto Dst feed returned no finite values")
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

function _window_finite_count(df::DataFrame, col::Symbol, t0::DateTime, t1::DateTime)
    w = _window(df, t0, t1)
    String(col) in names(w) || return 0
    return count(x -> !ismissing(x) && isfinite(Float64(x)), w[!, col])
end

"""
    _step_driver_fallback(plasma, mag, source_hour)

True when the intermediate forecast hour `[source_hour, source_hour+1h)` has no
finite speed OR no finite Bz sample, so `_drivers_for_window` would carry frozen
persistence drivers rather than observed solar wind into that step (P1-2). Pure,
so the per-step fallback bookkeeping in the multi-step issuance loop is testable
without the live feeds.
"""
function _step_driver_fallback(plasma::DataFrame, mag::DataFrame, source_hour::DateTime)
    src_end = source_hour + Hour(1)
    return _window_finite_count(plasma, :speed, source_hour, src_end) == 0 ||
           _window_finite_count(mag, :bz_gsm, source_hour, src_end) == 0
end

"""
    _driver_gap_status(n_speed_finite, n_bz_finite[, n_density_finite, n_by_finite])

Classify solar-wind coverage of the trailing forecast window over all four
drivers that `_drivers_for_window` would otherwise silently replace with quiet
defaults (speed→400, density→5, Bz→0, By→0). `:hard` (no finite speed AND no
finite Bz — refuse to issue), `:partial` (ANY of the four trailing counts is
empty so at least one driver fell back to a default — issue but flag), `:ok`
when all four have at least one finite sample. The density/By counts default to
a nonzero sentinel so legacy two-argument callers keep their original meaning.
Pure function so the data-gap decision is unit-testable without the live feeds.
"""
function _driver_gap_status(n_speed_finite::Int, n_bz_finite::Int,
                            n_density_finite::Int=1, n_by_finite::Int=1)
    (n_speed_finite == 0 && n_bz_finite == 0) && return :hard
    (n_speed_finite == 0 || n_bz_finite == 0 ||
     n_density_finite == 0 || n_by_finite == 0) && return :partial
    return :ok
end

# Atomic CSV write: write to a sibling .tmp then rename (same-filesystem mv is atomic), so a concurrent
# reader (e.g. the dashboard API serving live_forecast_log.csv) never observes a half-written log.
function _atomic_csv(path::AbstractString, df)
    tmp = string(path, ".tmp")
    CSV.write(tmp, df)
    mv(tmp, path; force=true)
    return df
end

const L1_DIST_KM = 1.5e6   # L1 standoff [km]; solar-wind transit lag to Earth = L1_DIST_KM / V / 3600 h

# Minute (sub-hourly) layer: for a frozen-tail forecast step, drive it with the 1-min L1 wind that PROPAGATES
# into that Earth-hour and is ALREADY measured by issue time. Leakage-safe by construction: the source window is
# capped at `latest_common_sw` (the freshest sample available at issue), so only measured wind enters. Reduces
# to the frozen `recent` driver (== v2) when no measured wind maps into the step (continuity). Blends the
# measured sub-window with the frozen driver by the known fraction f, mirroring the validated sub-hourly-A.
function _subhourly_driver(plasma::DataFrame, mag::DataFrame, step_time::DateTime, recent, latest_common_sw::DateTime)
    V = max(recent.V, 1.0)
    lag = Millisecond(round(Int, (L1_DIST_KM / V / 3600.0) * 3_600_000))   # transit L1 -> Earth
    src_lo = step_time - Hour(1) - lag
    src_hi = min(step_time - lag, latest_common_sw)                        # only minutes measured at L1 by issue
    src_hi <= src_lo && return recent                                      # nothing measured maps here -> v2
    f = clamp(Dates.value(src_hi - src_lo) / 3_600_000.0, 0.0, 1.0)
    meas = _drivers_for_window(plasma, mag, src_lo, src_hi; fallback=recent)
    return (V = f*meas.V + (1-f)*recent.V, Bz = f*meas.Bz + (1-f)*recent.Bz,
            By = f*meas.By + (1-f)*recent.By, n = f*meas.n + (1-f)*recent.n,
            Pdyn = f*meas.Pdyn + (1-f)*recent.Pdyn)
end

# Sub-hour MODEL TRAJECTORY for the near term: the served forecast integrated at a sub-hour step (default 15 min)
# with the same per-hour drivers (observed / L1 look-ahead / frozen). This is DISPLAY ONLY and is a model
# trajectory, not a validated sub-hour forecast: Dst is published only hourly (no sub-hour ground truth) and the
# discovered ODE is fit on hourly data, so the curve is the hourly-scale model's own interpolation. It tracks the
# hourly forecast closely because the ring current has little sub-hour structure.
function _subhour_trajectory(coef_csv, ens_csv, latest_dst_time::DateTime, anchor_dst_star::Float64,
                             plasma::DataFrame, mag::DataFrame, recent, latest_complete_hour::DateTime,
                             latest_common_sw::DateTime, pdyn::Float64, v2_correction::Float64;
                             window_h::Int=6, substeps::Int=4)
    st = init_forecast(; coefficients_csv=coef_csv, ensemble_csv=ens_csv, t0=latest_dst_time,
                         dst0=anchor_dst_star, dt=1.0 / substeps)
    pts = NamedTuple[]
    for k in 1:window_h
        hstart = latest_dst_time + Hour(k - 1)
        drv = hstart < latest_complete_hour ?
            _drivers_for_window(plasma, mag, hstart, hstart + Hour(1); fallback=recent) :
            _subhourly_driver(plasma, mag, latest_dst_time + Hour(k), recent, latest_common_sw)
        for s in 1:substeps
            t = latest_dst_time + Millisecond(round(Int, ((k - 1) + s / substeps) * 3_600_000))
            res = step_forecast!(st, t, drv.V, drv.Bz, drv.By, drv.n, drv.Pdyn)
            push!(pts, (t=string(t), dst=_dst_from_dst_star(res.dst_predicted, pdyn) + v2_correction))
        end
    end
    return pts
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
    _atomic_csv(log_path, df)
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

    v2_pred = _optional_float(df, row_idx, :v2_pred_dst_nt)
    if !ismissing(v2_pred)
        _set_value!(df, row_idx, :v2_residual_dst_nt, observed_dst - v2_pred)
        v2_ci05 = _optional_float(df, row_idx, :v2_pred_dst_ci05_nt)
        v2_ci95 = _optional_float(df, row_idx, :v2_pred_dst_ci95_nt)
        if !ismissing(v2_ci05) && !ismissing(v2_ci95)
            _set_value!(
                df,
                row_idx,
                :v2_observed_in_90ci,
                min(v2_ci05, v2_ci95) <= observed_dst <= max(v2_ci05, v2_ci95),
            )
        end
    end

    # Score the promoted served forecast (v2 + L1 look-ahead) alongside v2, so its live skill is tracked.
    served_pred = _optional_float(df, row_idx, :served_pred_dst_nt)
    if !ismissing(served_pred)
        _set_value!(df, row_idx, :served_residual_dst_nt, observed_dst - served_pred)
        s05 = _optional_float(df, row_idx, :served_pred_dst_ci05_nt)
        s95 = _optional_float(df, row_idx, :served_pred_dst_ci95_nt)
        (!ismissing(s05) && !ismissing(s95)) &&
            _set_value!(df, row_idx, :served_observed_in_90ci, min(s05, s95) <= observed_dst <= max(s05, s95))
    end

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

function _v2_expert_features(v1_pred_dst::Real, baselines)
    if baselines === nothing
        return (
            baseline_spread_nt=0.0,
            v1_minus_persistence_nt=0.0,
            obrien_minus_v1_nt=0.0,
            burton_minus_v1_nt=0.0,
        )
    end
    values = Float64[
        Float64(v1_pred_dst),
        Float64(baselines.persistence),
        Float64(baselines.burton),
        Float64(baselines.burton_full),
        Float64(baselines.obrien),
    ]
    return (
        baseline_spread_nt=maximum(values) - minimum(values),
        v1_minus_persistence_nt=Float64(v1_pred_dst) - Float64(baselines.persistence),
        obrien_minus_v1_nt=Float64(baselines.obrien) - Float64(v1_pred_dst),
        burton_minus_v1_nt=Float64(baselines.burton) - Float64(v1_pred_dst),
    )
end

function _zero_v2_memory_features()
    return (
        dst_delta_1h_nt=0.0,
        dst_delta_3h_nt=0.0,
        Bz_delta_1h_nt=0.0,
        VBsouth_delta_1h_mvm=0.0,
        VBsouth_mean_3h_mvm=0.0,
        Bsouth_mean_3h_nt=0.0,
    )
end

function _vb_south(drivers)
    return 1e-3 * Float64(drivers.V) * max(-Float64(drivers.Bz), 0.0)
end

function _live_v2_memory_features(plasma::DataFrame, mag::DataFrame,
                                  dst_times, dst_vals,
                                  latest_dst_time::DateTime,
                                  current_drivers)
    dst_map = _dst_lookup(dst_times, dst_vals)
    latest_dst = get(dst_map, latest_dst_time, Float64(dst_vals[end]))
    prev1_dst = get(dst_map, latest_dst_time - Hour(1), latest_dst)
    prev3_dst = get(dst_map, latest_dst_time - Hour(3), prev1_dst)
    prev_drivers = _drivers_for_window(
        plasma,
        mag,
        latest_dst_time - Hour(1),
        latest_dst_time;
        fallback=current_drivers,
    )
    vb_values = Float64[]
    bs_values = Float64[]
    for h in 0:2
        d = _drivers_for_window(
            plasma,
            mag,
            latest_dst_time - Hour(h + 1),
            latest_dst_time - Hour(h);
            fallback=current_drivers,
        )
        push!(vb_values, _vb_south(d))
        push!(bs_values, max(-Float64(d.Bz), 0.0))
    end
    return (
        dst_delta_1h_nt=latest_dst - prev1_dst,
        dst_delta_3h_nt=latest_dst - prev3_dst,
        Bz_delta_1h_nt=Float64(current_drivers.Bz) - Float64(prev_drivers.Bz),
        VBsouth_delta_1h_mvm=_vb_south(current_drivers) - _vb_south(prev_drivers),
        VBsouth_mean_3h_mvm=mean(vb_values),
        Bsouth_mean_3h_nt=mean(bs_values),
    )
end

function _v2_features(latest_dst::Real, drivers;
                      memory=_zero_v2_memory_features(),
                      baselines=nothing,
                      v1_pred_dst::Real=NaN)
    base = operational_v2_feature_tuple(
        latest_dst,
        drivers.V,
        drivers.Bz,
        drivers.By,
        drivers.n,
        drivers.Pdyn,
    )
    return merge(base, memory, _v2_expert_features(v1_pred_dst, baselines))
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

# Load the conformal interval calibration sidecar if present (v2 only). Returns
# nothing when absent, so the workflow stays backward compatible with logs that
# predate conformal intervals.
function _load_conformal_for_model(cfg::LiveVerifyConfig)
    cfg.model == :v2 || return nothing
    path = _conformal_path(cfg.v2_calibration_path)
    isfile(path) || return nothing
    return read_conformal_calibration(path)
end

# N2: online adaptive-conformal (ACI) interval derived from the verified forecast
# log. The log IS the ACI state (no separate state file): for the requested horizon
# we replay the chronological verified (v2 point, observed) stream to bring the ACI
# miscoverage level to its current value, then read the locked interval for the new
# forecast via a predict-only (gap) step that does not mutate the state. Returns
# (lo, hi), or `nothing` when there is too little verified history for this horizon
# (the caller then keeps the static interval). Fail-safe: any error returns
# `nothing`, so a malformed log can never break issuance.
# Activity regime for the residual pool — same rule and threshold as the stratified
# conformal calibration (SolarSINDy._activity_regime / CONFORMAL_ACTIVITY_THRESHOLD_NT):
# :disturbed when latest_dst ≤ threshold, else :quiet; non-finite Dst → :disturbed.
const _ACI_ACTIVITY_THRESHOLD_NT = -30.0
_aci_regime(latest_dst::Real, thr::Real=_ACI_ACTIVITY_THRESHOLD_NT) =
    (isfinite(latest_dst) && latest_dst > thr) ? :quiet : :disturbed

function _aci_interval_from_log(log_path::AbstractString, center::Real,
                                horizon_steps::Integer;
                                latest_dst::Real=NaN,
                                activity_threshold::Real=_ACI_ACTIVITY_THRESHOLD_NT,
                                target_coverage::Float64=0.90, gamma::Float64=0.03,
                                warmup::Int=30)
    try
        isfile(log_path) || return nothing
        df = CSV.read(log_path, DataFrame)
        cols = names(df)
        # Key on model_step_hours — the lead (target − anchor) the QUERY is keyed on —
        # not horizon_hours (wall-clock target − issue). The two differ whenever the
        # anchor Dst lags the issue time, which silently routed long leads to a
        # different / empty residual pool and through to the over-wide fallback.
        all(c -> c in cols, ("model_step_hours", "v2_pred_dst_nt",
                             "observation_dst_nt", "issue_time_utc", "latest_dst_nt")) || return nothing
        h = Int(horizon_steps)
        cur_regime = _aci_regime(Float64(latest_dst), activity_threshold)
        # Residual indices at this lead; regime_match restricts to the current activity
        # regime so a quiet-period band is not inflated by past storm-time residuals.
        collect_idx(regime_match::Bool) = begin
            out = Int[]
            for i in 1:nrow(df)
                hv = df[i, :model_step_hours]; pv = df[i, :v2_pred_dst_nt]
                ov = df[i, :observation_dst_nt]; ld = df[i, :latest_dst_nt]
                (ismissing(hv) || ismissing(pv) || ismissing(ov)) && continue
                (round(Int, Float64(hv)) == h) || continue
                (isfinite(Float64(pv)) && isfinite(Float64(ov))) || continue
                if regime_match
                    (ismissing(ld) || !isfinite(Float64(ld))) && continue
                    (_aci_regime(Float64(ld), activity_threshold) == cur_regime) || continue
                end
                push!(out, i)
            end
            out
        end
        # Prefer the regime-conditional pool; fall back to the all-regime pool at this
        # lead if regime data is too sparse (graceful — still correctly lead-keyed ACI,
        # never the over-wide static fallback).
        idx = isfinite(Float64(latest_dst)) ? collect_idx(true) : collect_idx(false)
        length(idx) < warmup + 5 && (idx = collect_idx(false))
        length(idx) < warmup + 5 && return nothing
        idx = idx[sortperm([string(df[i, :issue_time_utc]) for i in idx])]
        ac = init_adaptive_conformal(; target_coverage=target_coverage,
                                     gamma=gamma, warmup=warmup)
        for i in idx
            adaptive_conformal_step!(ac, Float64(df[i, :v2_pred_dst_nt]),
                                     Float64(df[i, :observation_dst_nt]))
        end
        s = adaptive_conformal_step!(ac, Float64(center), NaN)  # predict-only (gap path)
        (isfinite(s.lo) && isfinite(s.hi)) || return nothing
        return (s.lo, s.hi)
    catch
        return nothing
    end
end

"""
    _resolve_interval(conformal, center, model_steps, latest_dst, ci05, ci95)

Choose the logged 90% interval. With a deployed conformal calibration, return the
stratified conformal interval around `center` and the tag `"conformal"`;
otherwise pass through the supplied `(ci05, ci95)` and the tag
`"interval_scale"`. Pure, so the interval-source decision is unit-testable
without the network-bound issuance path.
"""
function _resolve_interval(conformal, center::Real, model_steps::Integer,
                           latest_dst::Real, ci05::Real, ci95::Real)
    conformal === nothing && return (Float64(ci05), Float64(ci95), "interval_scale")
    lo, hi = conformal_interval(conformal, center, Float64(model_steps), latest_dst)
    return (lo, hi, "conformal")
end

function _select_model_prediction(model::Symbol, calibration,
                                  latest_dst::Real, drivers,
                                  v1_pred_dst::Real,
                                  v1_ci05_dst::Real,
                                  v1_ci95_dst::Real;
                                  baselines=nothing,
                                  features=nothing)
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
            v2_selected_component=missing,
            v2_selected_component_pred=missing,
        )
    elseif model == :v2
        calibration === nothing && error("v2 model requested without calibration")
        v2 = operational_v2_predict(
            calibration,
            v1_pred_dst,
            v1_ci05_dst,
            v1_ci95_dst,
            features === nothing ?
                _v2_features(
                    latest_dst,
                    drivers;
                    baselines=baselines,
                    v1_pred_dst=v1_pred_dst,
                ) :
                features,
            baselines=baselines,
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
            v2_selected_component=v2.selected_component,
            v2_selected_component_pred=v2.selected_component_pred,
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
                              n_steps::Int=1, model::Symbol=:v1, calibration=nothing)
    n_steps >= 1 || throw(ArgumentError("n_steps must be ≥ 1"))
    anchor_dst_star = pressure_correct_dst([anchor_dst], [drivers.Pdyn])[1]
    coef_csv = joinpath(get_data_dir(), "real_sindy_discovery_coefficients.csv")
    ens_csv = joinpath(get_data_dir(), "real_ensemble_inclusion.csv")
    state = init_forecast(;
        coefficients_csv=coef_csv,
        ensemble_csv=ens_csv,
        t0=anchor_time,
        dst0=anchor_dst_star,
    )
    # Integrate n_steps hours forward holding the anchor drivers (persistence of
    # drivers), matching the issuance assumption beyond the last complete hour.
    # forecast_ahead does not mutate state and propagates the ensemble per member;
    # for n_steps=1 it is numerically identical to a single step_forecast! call.
    fc = forecast_ahead(state, drivers.V, drivers.Bz, drivers.By,
                        drivers.n, drivers.Pdyn, n_steps)
    result = fc[n_steps]
    # Advance the baseline ring-current state the same number of steps.
    burton_star = anchor_dst_star
    burton_full_star = anchor_dst_star
    obrien_star = anchor_dst_star
    for _ in 1:n_steps
        burton_star = _advance_baselines(burton_star, drivers).burton
        burton_full_star = _advance_baselines(burton_full_star, drivers).burton_full
        obrien_star = _advance_baselines(obrien_star, drivers).obrien
    end
    baselines = (burton=burton_star, burton_full=burton_full_star, obrien=obrien_star)
    v1_pred = _dst_from_dst_star(result.dst_predicted, drivers.Pdyn)
    v1_ci05 = _dst_from_dst_star(result.dst_ci_05, drivers.Pdyn)
    v1_ci95 = _dst_from_dst_star(result.dst_ci_95, drivers.Pdyn)
    baseline_predictions = (
        persistence=anchor_dst,
        burton=_dst_from_dst_star(baselines.burton, drivers.Pdyn),
        burton_full=_dst_from_dst_star(baselines.burton_full, drivers.Pdyn),
        obrien=_dst_from_dst_star(baselines.obrien, drivers.Pdyn),
    )
    selected = _select_model_prediction(
        model,
        calibration,
        anchor_dst,
        drivers,
        v1_pred,
        v1_ci05,
        v1_ci95,
        baselines=baseline_predictions,
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
        v2_selected_component=selected.v2_selected_component,
        v2_selected_component_pred=selected.v2_selected_component_pred,
        persistence_dst=baseline_predictions.persistence,
        burton_dst=baseline_predictions.burton,
        burton_full_dst=baseline_predictions.burton_full,
        obrien_dst=baseline_predictions.obrien,
    )
end

function replay_recent_table(plasma::DataFrame, mag::DataFrame,
                             dst_times, dst_vals; replay_hours::Int=48,
                             horizons::Vector{Int}=[1],
                             model::Symbol=:v1, calibration=nothing)
    isempty(horizons) && throw(ArgumentError("horizons must not be empty"))
    any(<=(0), horizons) && throw(ArgumentError("horizons must be positive"))
    dst_map = _dst_lookup(dst_times, dst_vals)
    anchors = _replay_anchor_hours(plasma, mag, dst_times, replay_hours)
    rows = NamedTuple[]

    for anchor_time in anchors
        haskey(dst_map, anchor_time) || continue

        source_start = anchor_time - Hour(1)
        source_end = anchor_time
        drivers = _drivers_for_window(plasma, mag, source_start, source_end)
        if !all(isfinite, (drivers.V, drivers.n, drivers.Bz, drivers.By, drivers.Pdyn))
            continue
        end
        # Mirror the issuance C1 guard: `_drivers_for_window` (no fallback) fills an
        # all-NaN window with finite quiet defaults, so the isfinite check above
        # cannot see a fabricated driver. Drop the anchor whenever ANY of the four
        # solar-wind drivers had no finite trailing sample. A no-op on valid data
        # (every window has finite samples), it only rejects fabricated rows.
        if _driver_gap_status(
                _window_finite_count(plasma, :speed, source_start, source_end),
                _window_finite_count(mag, :bz_gsm, source_start, source_end),
                _window_finite_count(plasma, :density, source_start, source_end),
                _window_finite_count(mag, :by_gsm, source_start, source_end),
            ) != :ok
            continue
        end
        features = _v2_features(dst_map[anchor_time], drivers)

        for h in horizons
            target_time = anchor_time + Hour(h)
            haskey(dst_map, target_time) || continue

            forecast = _forecast_one_replay(
                anchor_time,
                target_time,
                dst_map[anchor_time],
                drivers,
                n_steps=h,
                model=:v1,
                calibration=nothing,
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
                model_step_hours=h,
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
                v2_selected_component=forecast.v2_selected_component,
                v2_selected_component_pred_nt=forecast.v2_selected_component_pred,
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
                Bsouth_nt=features.Bsouth_nt,
                VBsouth_mvm=features.VBsouth_mvm,
                Bperp_nt=features.Bperp_nt,
                clock_angle_sin2=features.clock_angle_sin2,
                sqrt_Pdyn_npa=features.sqrt_Pdyn_npa,
                replay_note="causal_replay_previous_complete_hour_driver_persistence",
            ))
        end
    end

    isempty(rows) && error("No replay rows could be scored from the available feeds")
    out = SolarSINDy.add_operational_v2_features!(DataFrame(rows))
    if model == :v2
        calibration === nothing && error("Operational v2 replay requires calibration")
        scored = score_operational_v2(out, calibration)
        scored[!, :model_version] = fill("v2", nrow(scored))
        scored[!, :pred_dst_nt] = scored.v2_pred_dst_nt
        scored[!, :pred_dst_ci05_nt] = scored.v2_pred_dst_ci05_nt
        scored[!, :pred_dst_ci95_nt] = scored.v2_pred_dst_ci95_nt
        scored[!, :residual_dst_nt] = scored.observation_dst_nt .- scored.pred_dst_nt
        scored[!, :observed_in_90ci] = scored.v2_observed_in_90ci
        return scored
    elseif model == :v1
        return out
    else
        error("Unsupported replay model: $model")
    end
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
        :v2_selected_component,
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
        ("Operational v2", :v2_pred_dst_nt),
        ("SINDy v1", :v1_pred_dst_nt),
        ("Persistence", :persistence_dst_nt),
        ("Burton", :burton_dst_nt),
        ("BurtonFull", :burton_full_dst_nt),
        ("OBrien", :obrien_dst_nt),
        ("Logged model", :pred_dst_nt),
    )
        String(col) in names(df) || continue
        vals = df[!, col]
        all(ismissing, vals) && continue
        idx = .!ismissing.(vals)
        _print_metric(name, Float64.(vals[idx]), Float64.(df.observation_dst_nt[idx]))
    end
    println(
        "Logged-model 90% coverage n=$(nrow(df)) coverage=",
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
        horizons=cfg.replay_horizons,
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

function _omni_replay_inputs(path::String, year_start::Int, year_end::Int)
    isfile(path) || error(
        "OMNI CSV not found: $path. Run the OMNI extraction workflow or pass --omni=PATH."
    )
    df = parse_omni2(path; year_start=year_start, year_end=year_end)
    clean_omni_data!(df)
    required = [:datetime, :V, :Bz, :By, :n, :Pdyn, :Dst]
    _require_live_columns(df, required)
    valid = trues(nrow(df))
    for col in (:V, :Bz, :By, :n, :Pdyn, :Dst)
        valid .&= isfinite.(Float64.(df[!, col]))
    end
    String(:quality) in names(df) && (valid .&= df.quality .== 1)
    df = df[valid, :]
    sort!(df, :datetime)
    nrow(df) >= 3 || error("Need at least 3 finite OMNI rows for replay")
    plasma = DataFrame(
        time_tag=DateTime.(df.datetime),
        speed=Float64.(df.V),
        density=Float64.(df.n),
    )
    mag = DataFrame(
        time_tag=DateTime.(df.datetime),
        bz_gsm=Float64.(df.Bz),
        by_gsm=Float64.(df.By),
    )
    return plasma, mag, DateTime.(df.datetime), Float64.(df.Dst)
end

function run_replay_omni(cfg::LiveVerifyConfig)
    plasma, mag, dst_times, dst_vals =
        _omni_replay_inputs(cfg.omni_path, cfg.omni_year_start, cfg.omni_year_end)
    calibration = _load_calibration_for_model(cfg)
    df = replay_recent_table(plasma, mag, dst_times, dst_vals;
        replay_hours=cfg.replay_hours,
        horizons=cfg.replay_horizons,
        model=cfg.model,
        calibration=calibration,
    )

    dir = dirname(cfg.table_path)
    !isempty(dir) && mkpath(dir)
    CSV.write(cfg.table_path, df)
    md_path = _markdown_path(cfg.table_path)
    write_markdown_table(md_path, df; limit=cfg.table_limit)

    println(
        "OMNI causal replay years: $(cfg.omni_year_start)-$(cfg.omni_year_end); " *
        "finite rows available=$(length(dst_vals)); scored rows=$(nrow(df))"
    )
    _print_replay_metrics(df)
    println("Wrote CSV table: $(cfg.table_path)")
    println("Wrote Markdown table: $md_path")
    return df
end

const V2_FULL_FEATURES = Symbol[
    :latest_dst_nt,
    :V_kms,
    :Bz_nt,
    :By_nt,
    :n_cm3,
    :Pdyn_npa,
    :Bsouth_nt,
    :VBsouth_mvm,
    :Bperp_nt,
    :clock_angle_sin2,
    :sqrt_Pdyn_npa,
]

const V2_MEMORY_FEATURES = vcat(V2_FULL_FEATURES, Symbol[
    :dst_delta_1h_nt,
    :dst_delta_3h_nt,
    :Bz_delta_1h_nt,
    :VBsouth_delta_1h_mvm,
    :VBsouth_mean_3h_mvm,
    :Bsouth_mean_3h_nt,
])

const V2_MEMORY_EXPERT_FEATURES = vcat(V2_MEMORY_FEATURES, Symbol[
    :baseline_spread_nt,
    :v1_minus_persistence_nt,
    :obrien_minus_v1_nt,
    :burton_minus_v1_nt,
])

const V2_BASE_FEATURES = Symbol[
    :latest_dst_nt,
    :V_kms,
    :Bz_nt,
    :By_nt,
    :n_cm3,
    :Pdyn_npa,
]

const V2_COUPLING_FEATURES = Symbol[
    :latest_dst_nt,
    :Bsouth_nt,
    :VBsouth_mvm,
    :Bperp_nt,
    :clock_angle_sin2,
    :sqrt_Pdyn_npa,
]

function _v2_feature_sets()
    return Pair{String,Vector{Symbol}}[
        "memory_expert" => copy(V2_MEMORY_EXPERT_FEATURES),
        "memory" => copy(V2_MEMORY_FEATURES),
        "full" => copy(V2_FULL_FEATURES),
        "base" => copy(V2_BASE_FEATURES),
        "coupling" => copy(V2_COUPLING_FEATURES),
        "latest_dst" => Symbol[:latest_dst_nt],
    ]
end

function _chronological_train_validation_test(df::DataFrame,
                                              train_fraction::Float64,
                                              validation_fraction::Float64)
    sort_cols = Symbol[]
    String(:issue_time_utc) in names(df) && push!(sort_cols, :issue_time_utc)
    !isempty(sort_cols) && sort!(df, sort_cols)
    n = nrow(df)
    n >= 3 || throw(ArgumentError("Need at least 3 rows for fit/validation/holdout split"))

    # F3: split by ANCHOR (issue_time), not by raw row index. With multi-horizon
    # replay an anchor contributes several rows (one per lead); a raw-index cut can
    # straddle one anchor's rows across train/validation/holdout, leaking the same
    # issue time into multiple splits. Partition the unique sorted anchors by
    # fraction and assign each anchor's whole row block to one split, so the
    # issue_time sets are pairwise disjoint. With one row per anchor (the
    # single-horizon table) this reduces to the original index split.
    if String(:issue_time_utc) in names(df)
        anchors = unique(df.issue_time_utc)            # sorted: df is sorted above
        na = length(anchors)
        na >= 3 || throw(ArgumentError("Need at least 3 distinct anchors for fit/validation/holdout split"))
        na_train = clamp(floor(Int, train_fraction * na), 1, na - 2)
        na_validation = clamp(floor(Int, validation_fraction * na), 1, na - na_train - 1)
        split_of = Dict{eltype(anchors),Int}()
        for (i, a) in enumerate(anchors)
            split_of[a] = i <= na_train ? 1 : (i <= na_train + na_validation ? 2 : 3)
        end
        s = [split_of[a] for a in df.issue_time_utc]
        return df[s .== 1, :], df[s .== 2, :], df[s .== 3, :]
    end

    n_train = clamp(floor(Int, train_fraction * n), 1, n - 2)
    n_validation = clamp(floor(Int, validation_fraction * n), 1, n - n_train - 1)
    train = df[1:n_train, :]
    validation = df[n_train + 1:n_train + n_validation, :]
    holdout = df[n_train + n_validation + 1:end, :]
    return train, validation, holdout
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

function _scored_metric(scored::DataFrame, pred_col::Symbol)
    preds, obs = _metric_rows(scored, pred_col)
    m = _metric_values(preds, obs)
    m === nothing && throw(ArgumentError("no finite rows for metric column $pred_col"))
    return m
end

"""
    _paired_gate_metrics(scored, pred_cols)

Compute metrics for several predictor columns over a SINGLE common finite-row
mask: a row counts only when its observation and EVERY listed prediction are
present and finite (`isfinite`, not just `!ismissing`). F6: the validation gate
compares v2 against persistence and O'Brien, so a missing/NaN baseline row must
drop the whole comparison row from all metrics, otherwise the comparison is
unpaired (different n per column) and a baseline can look artificially better or
worse than v2. Returns a `Dict{Symbol,NamedTuple}` keyed by predictor column.
"""
function _paired_gate_metrics(scored::DataFrame, pred_cols::Vector{Symbol})
    cols = Symbol[c for c in pred_cols if String(c) in names(scored)]
    isempty(cols) && throw(ArgumentError("no listed predictor columns present in scored frame"))
    finite(df, i, c) = begin
        v = _optional_float(df, i, c)
        !ismissing(v) && isfinite(Float64(v))
    end
    obs = Float64[]
    series = Dict(c => Float64[] for c in cols)
    for i in 1:nrow(scored)
        (finite(scored, i, :observation_dst_nt) && all(finite(scored, i, c) for c in cols)) || continue
        push!(obs, Float64(_optional_float(scored, i, :observation_dst_nt)))
        for c in cols
            push!(series[c], Float64(_optional_float(scored, i, c)))
        end
    end
    out = Dict{Symbol,NamedTuple}()
    for c in cols
        m = _metric_values(series[c], obs)
        m === nothing && throw(ArgumentError("no common finite rows for paired gate metric $c"))
        out[c] = m
    end
    return out
end

function _require_live_columns(df::DataFrame, cols)
    missing_cols = [String(c) for c in cols if !(String(c) in names(df))]
    isempty(missing_cols) || throw(ArgumentError(
        "missing required live workflow column(s): $(join(missing_cols, ", "))"
    ))
    return nothing
end

function _calibration_with_validation_selector(cal::OperationalV2Calibration,
                                               selector;
                                               label::AbstractString=cal.label)
    return OperationalV2Calibration(
        copy(cal.feature_names),
        copy(cal.feature_mean),
        copy(cal.feature_scale),
        copy(cal.coefficients),
        cal.interval_scale,
        label,
        selector_names=copy(selector.selector_names),
        selector_rmse=copy(selector.selector_rmse),
        selector_mae=copy(selector.selector_mae),
        selector_half_width=copy(selector.selector_half_width),
        selector_weights=selector.selector_weights === nothing ? nothing : copy(selector.selector_weights),
        selected_component=selector.selected_component,
        guard_margin_nt=cal.guard_margin_nt,
    )
end

function _calibration_with_forced_component(cal::OperationalV2Calibration,
                                            component::Symbol;
                                            label::AbstractString=cal.label)
    component in cal.selector_names ||
        throw(ArgumentError("cannot force unavailable v2 component: $component"))
    selector = (
        selector_names=copy(cal.selector_names),
        selector_rmse=copy(cal.selector_rmse),
        selector_mae=copy(cal.selector_mae),
        selector_half_width=copy(cal.selector_half_width),
        selector_weights=nothing,
        selected_component=component,
    )
    return _calibration_with_validation_selector(cal, selector; label=label)
end

function _v2_gate_pass(v2_metric, v1_metric)
    return v2_metric.rmse <= v1_metric.rmse && v2_metric.mae <= v1_metric.mae
end

function _selection_path(calibration_path::String)
    out = replace(calibration_path, r"\.csv$" => "_selection.csv")
    return out == calibration_path ? calibration_path * "_selection.csv" : out
end

function _conformal_path(calibration_path::String)
    out = replace(calibration_path, r"\.csv$" => "_conformal.csv")
    return out == calibration_path ? calibration_path * "_conformal.csv" : out
end

# Lead time [hr] for each scored row: prefer model_step_hours, else 1.0 (the
# replay calibration table is 1-step; multi-horizon stratification needs the
# multi-horizon replay).
function _scored_horizons(scored::DataFrame)
    String(:model_step_hours) in names(scored) || return fill(1.0, nrow(scored))
    return Float64[ismissing(x) ? 1.0 : Float64(x) for x in scored.model_step_hours]
end

"""
    _fit_deployed_conformal(scored, cfg)

Fit a stratified conformal calibration from the DEPLOYED model's residuals on
`scored` (the validation split — out-of-sample for the ridge β, model frozen),
using the v2 point forecast, horizon, and issue-time Dst. Returns `nothing` if
there are no finite calibration rows.
"""
function _fit_deployed_conformal(scored::DataFrame, cfg::LiveVerifyConfig)
    nrow(scored) >= 1 || return nothing
    pts = Float64.(scored.v2_pred_dst_nt)
    obs = Float64.(scored.observation_dst_nt)
    hor = _scored_horizons(scored)
    latest = Float64.(scored.latest_dst_nt)
    any(i -> isfinite(pts[i]) && isfinite(obs[i]), eachindex(pts)) || return nothing
    return fit_conformal(pts, obs, hor, latest; coverage=cfg.v2_interval_coverage)
end

function _select_validated_v2_calibration(train::DataFrame,
                                          validation::DataFrame,
                                          cfg::LiveVerifyConfig)
    # Leakage-free selection: the ridge coefficients AND the component selector
    # are fit on `train` only (inside fit_operational_v2_calibration). The
    # feature set / ridge / component are then chosen by performance on
    # `validation` — rows used to fit neither. The holdout is never seen here.
    candidates = NamedTuple[]
    best = nothing
    for feature_spec in _v2_feature_sets()
        feature_set, feature_names = first(feature_spec), last(feature_spec)
        all(String(c) in names(train) for c in feature_names) || continue
        nrow(train) > length(feature_names) + 1 || continue
        for ridge in cfg.v2_ridge_grid
            label = "operational_v2_$(feature_set)_ridge$(ridge)_fit$(nrow(train))"
            cal = try
                cal0 = fit_operational_v2_calibration(
                    train;
                    feature_names=feature_names,
                    ridge=ridge,
                    interval_coverage=cfg.v2_interval_coverage,
                    guard_margin_nt=cfg.v2_selector_margin_nt,
                    label=label,
                )
                # Robustness guard: on a small fit set a baseline component can
                # win on noise. Force the corrected v2 center instead.
                (nrow(train) < V2_MIN_COMPONENT_ROWS && cal0.selected_component != :v2) ?
                    _calibration_with_forced_component(cal0, :v2; label=label * "_minsample_v2") :
                    cal0
            catch err
                push!(candidates, (
                    feature_set=feature_set,
                    ridge=ridge,
                    selected_component="failed",
                    validation_n=0,
                    validation_rmse_nt=NaN,
                    validation_mae_nt=NaN,
                    validation_bias_nt=NaN,
                    validation_persistence_rmse_nt=NaN,
                    validation_obrien_rmse_nt=NaN,
                    validation_coverage=NaN,
                    beats_persistence=false,
                    beats_obrien=false,
                    coverage_ok=false,
                    gate_pass=false,
                    error=String(nameof(typeof(err))),
                ))
                continue
            end

            # Evaluate purely on the held-out validation split. F6: v2,
            # persistence, and O'Brien metrics share ONE common finite-row mask so
            # the gate comparison is paired (equal n); a missing/NaN baseline row
            # drops from all three rather than only its own column.
            vs = score_operational_v2(validation, cal)
            has_obrien = String(:obrien_dst_nt) in names(vs)
            gate_cols = has_obrien ?
                Symbol[:v2_pred_dst_nt, :latest_dst_nt, :obrien_dst_nt] :
                Symbol[:v2_pred_dst_nt, :latest_dst_nt]
            gate_metrics = _paired_gate_metrics(vs, gate_cols)
            v2m = gate_metrics[:v2_pred_dst_nt]
            # Persistence forecast = latest observed Dst at issue time.
            persm = gate_metrics[:latest_dst_nt]
            obrm = has_obrien ? gate_metrics[:obrien_dst_nt] : nothing
            cov = nrow(vs) == 0 ? NaN : mean(Bool.(vs.v2_observed_in_90ci))
            beats_pers = _v2_gate_pass(v2m, persm)
            beats_obr = obrm === nothing ? true : _v2_gate_pass(v2m, obrm)
            cov_ok = isfinite(cov) && cov >= cfg.v2_coverage_floor
            gate = beats_pers && beats_obr && cov_ok
            row = (
                feature_set=feature_set,
                ridge=ridge,
                selected_component=String(cal.selected_component),
                validation_n=v2m.n,
                validation_rmse_nt=v2m.rmse,
                validation_mae_nt=v2m.mae,
                validation_bias_nt=v2m.bias,
                validation_persistence_rmse_nt=persm.rmse,
                validation_obrien_rmse_nt=(obrm === nothing ? NaN : obrm.rmse),
                validation_coverage=cov,
                beats_persistence=beats_pers,
                beats_obrien=beats_obr,
                coverage_ok=cov_ok,
                gate_pass=gate,
                error="",
            )
            push!(candidates, row)
            # Rank gate-passers first, then by validation RMSE/MAE.
            key = (gate ? 0 : 1, v2m.rmse, v2m.mae, ridge)
            if best === nothing || key < best.key
                best = (;
                    key, cal, row, gate_pass=gate,
                    validation_rmse=v2m.rmse,
                    validation_mae=v2m.mae,
                    validation_coverage=cov,
                )
            end
        end
    end
    best === nothing && throw(ArgumentError("no operational v2 calibration candidate could be fit"))
    return (; best..., candidates=DataFrame(candidates))
end

function fit_v2_calibration!(cfg::LiveVerifyConfig)
    isfile(cfg.table_path) || error("Replay table not found: $(cfg.table_path)")
    df = SolarSINDy.add_operational_v2_features!(
        _v2_base_prediction_table(CSV.read(cfg.table_path, DataFrame))
    )
    nrow(df) >= 8 || error("Need at least 8 replay rows to fit/select/test v2 calibration")
    train, validation, holdout = _chronological_train_validation_test(
        df,
        cfg.v2_train_fraction,
        cfg.v2_validation_fraction,
    )
    selection = _select_validated_v2_calibration(train, validation, cfg)
    candidate_cal = selection.cal

    # F5: the coverage-floor gate is only trustworthy when the validation split is
    # large enough to estimate coverage. With a single-row validation it degenerates
    # into a 0/1 check, so refuse to trust the v2 acceptance gate below this count
    # and deploy the v1-equivalent fallback instead (M3 minimum-validation guard).
    validation_trusted = nrow(validation) >= V2_MIN_VALIDATION_ROWS

    # F1+F2 (M8): the OPERATIONALLY-SERVED interval is the conformal sidecar, not
    # the legacy interval_scale band the validation gate inspects. Fit the sidecar
    # from the v2 CANDIDATE's validation residuals and require its holdout coverage
    # to clear the floor as part of the deploy decision; otherwise the candidate is
    # rejected even when the legacy band over-covered on validation. The candidate
    # splits are scored here only to drive that served-interval gate.
    candidate_validation_scored = score_operational_v2(validation, candidate_cal)
    candidate_holdout_scored = score_operational_v2(holdout, candidate_cal)
    candidate_conformal = _fit_deployed_conformal(candidate_validation_scored, cfg)
    candidate_conformal_holdout_coverage = NaN
    if candidate_conformal !== nothing && nrow(candidate_holdout_scored) >= 1
        candidate_conformal_holdout_coverage = conformal_coverage(
            candidate_conformal,
            Float64.(candidate_holdout_scored.v2_pred_dst_nt),
            Float64.(candidate_holdout_scored.observation_dst_nt),
            _scored_horizons(candidate_holdout_scored),
            Float64.(candidate_holdout_scored.latest_dst_nt),
        )
    end
    # The served (conformal) interval must clear the floor on the held-out split.
    # If it cannot be measured (no holdout rows / no conformal fit), treat it as
    # not satisfied so the gate fails closed rather than shipping an unchecked band.
    conformal_gate_ok = isfinite(candidate_conformal_holdout_coverage) &&
        candidate_conformal_holdout_coverage >= cfg.v2_coverage_floor

    deploy = selection.gate_pass && validation_trusted && conformal_gate_ok
    deploy_block_reason = if !selection.gate_pass
        "validation_acceptance_gate"
    elseif !validation_trusted
        "validation_split_too_thin"
    elseif !conformal_gate_ok
        "conformal_holdout_undercover"
    else
        ""
    end

    # Acceptance gate decides deployment. A v2 candidate that does not beat
    # persistence and O'Brien on validation, has too thin a validation split, or
    # whose SERVED conformal interval under-covers the holdout is NOT shipped; a
    # v1-equivalent (zero-correction) fallback is deployed instead.
    cal = if deploy
        candidate_cal
    else
        default_operational_v2_calibration(
            feature_names=copy(candidate_cal.feature_names),
            label="operational_v2_fallback_v1_equiv",
        )
    end
    write_operational_v2_calibration(cfg.v2_calibration_path, cal)

    train_scored = score_operational_v2(train, cal)
    validation_scored = score_operational_v2(validation, cal)
    # Holdout is never used to select feature set, ridge, component, or any tuning
    # parameter — those are chosen on validation only. It is read for honest
    # out-of-sample reporting and, since F1+F2, as a single final go/no-go safety
    # gate on the served conformal interval's coverage (M8); it does not feed back
    # into model selection.
    holdout_scored = score_operational_v2(holdout, cal)
    holdout_v2 = _scored_metric(holdout_scored, :v2_pred_dst_nt)
    holdout_coverage = nrow(holdout_scored) == 0 ? NaN :
        mean(Bool.(holdout_scored.v2_observed_in_90ci))

    selection_audit = copy(selection.candidates)
    selected_mask = (selection_audit.feature_set .== selection.row.feature_set) .&
                    (selection_audit.ridge .== selection.row.ridge)
    selection_audit[!, :selected_by_validation] = selected_mask
    selection_audit[!, :deployed] = selected_mask .& deploy
    selection_audit[!, :final_component] = fill(String(cal.selected_component), nrow(selection_audit))
    selection_audit[!, :acceptance_gate_pass] = fill(deploy, nrow(selection_audit))
    selection_audit[!, :validation_trusted] = fill(validation_trusted, nrow(selection_audit))
    selection_audit[!, :conformal_holdout_coverage] =
        fill(candidate_conformal_holdout_coverage, nrow(selection_audit))
    selection_audit[!, :conformal_gate_pass] = fill(conformal_gate_ok, nrow(selection_audit))
    selection_audit[!, :deploy_block_reason] = fill(deploy_block_reason, nrow(selection_audit))
    selection_audit[!, :holdout_n] = fill(holdout_v2.n, nrow(selection_audit))
    selection_audit[!, :holdout_rmse_nt] = fill(holdout_v2.rmse, nrow(selection_audit))
    selection_audit[!, :holdout_mae_nt] = fill(holdout_v2.mae, nrow(selection_audit))
    selection_audit[!, :holdout_coverage] = fill(holdout_coverage, nrow(selection_audit))
    selection_path = _selection_path(cfg.v2_calibration_path)
    CSV.write(selection_path, selection_audit)

    train_scored[!, :v2_split] = fill("fit", nrow(train_scored))
    validation_scored[!, :v2_split] = fill("validation", nrow(validation_scored))
    holdout_scored[!, :v2_split] = fill("holdout", nrow(holdout_scored))
    scored = vcat(train_scored, validation_scored, holdout_scored; cols=:union)
    scored_path = replace(cfg.v2_calibration_path, r"\.csv$" => "_scored.csv")
    scored_path == cfg.v2_calibration_path &&
        (scored_path = cfg.v2_calibration_path * "_scored.csv")
    CSV.write(scored_path, scored)

    # Conformal predictive-interval calibration (N1). Fit from the DEPLOYED model's
    # validation residuals (out-of-sample for β), persist a sidecar, and report
    # coverage on the untouched holdout — the honest out-of-sample check. When v2
    # deploys, this reproduces the candidate sidecar already gated above.
    conformal = _fit_deployed_conformal(validation_scored, cfg)
    conformal_path = _conformal_path(cfg.v2_calibration_path)
    holdout_conformal_coverage = NaN
    if conformal === nothing
        @warn "No finite validation rows for conformal calibration; conformal sidecar not written."
    else
        write_conformal_calibration(conformal_path, conformal)
        if nrow(holdout_scored) >= 1
            holdout_conformal_coverage = conformal_coverage(
                conformal,
                Float64.(holdout_scored.v2_pred_dst_nt),
                Float64.(holdout_scored.observation_dst_nt),
                _scored_horizons(holdout_scored),
                Float64.(holdout_scored.latest_dst_nt),
            )
        end
    end

    println("Fitted operational v2 calibration: $(cfg.v2_calibration_path)")
    println("Wrote v2 candidate selection audit: $selection_path")
    conformal === nothing || println("Wrote conformal calibration: $conformal_path")
    println("Scored replay rows: $scored_path")
    println(
        "Fit rows: $(nrow(train)); validation rows: $(nrow(validation)); " *
        "holdout rows: $(nrow(holdout)) (holdout scored once; not used to rank/tune candidates — only as the single-use deploy safety gate)"
    )
    if deploy
        println(
            "Deployed v2 candidate: feature_set=$(selection.row.feature_set), " *
            "ridge=$(selection.row.ridge), component=$(cal.selected_component) " *
            "(passed validation acceptance gate)"
        )
    else
        @warn(
            "Best v2 candidate failed the acceptance gate " *
            "(validation beat + thick-enough validation split + served conformal " *
            "interval holdout coverage ≥ floor); deployed v1-equivalent fallback.",
            block_reason=deploy_block_reason,
            best_validation_rmse=selection.validation_rmse,
            best_validation_coverage=selection.validation_coverage,
            conformal_holdout_coverage=candidate_conformal_holdout_coverage,
            validation_rows=nrow(validation),
            coverage_floor=cfg.v2_coverage_floor,
        )
    end
    _print_metric(
        "Validation v2",
        Float64.(validation_scored.v2_pred_dst_nt),
        Float64.(validation_scored.observation_dst_nt),
    )
    _print_metric(
        "Validation persistence",
        Float64.(validation_scored.latest_dst_nt),
        Float64.(validation_scored.observation_dst_nt),
    )
    _print_metric(
        "Holdout v2 (honest; unused for selection)",
        Float64.(holdout_scored.v2_pred_dst_nt),
        Float64.(holdout_scored.observation_dst_nt),
    )
    println("Holdout v2 90% coverage (interval-scale): $(round(holdout_coverage; digits=3))")
    isnan(holdout_conformal_coverage) ||
        println("Holdout v2 90% coverage (conformal): $(round(holdout_conformal_coverage; digits=3))")
    isnan(candidate_conformal_holdout_coverage) ||
        println(
            "Served-interval (conformal) holdout coverage gated against floor " *
            "$(cfg.v2_coverage_floor): $(round(candidate_conformal_holdout_coverage; digits=3)) " *
            "($(conformal_gate_ok ? "PASS" : "FAIL"))"
        )
    println("V2 interval scale: $(round(cal.interval_scale; digits=3))")
    println("Acceptance gate: $(deploy ? "PASS — v2 deployed" : "FAIL — v1-equivalent fallback deployed")")
    return cal
end

"""
    _assert_issuable_model(model, horizon_hours)

P1-3 guard: multi-step v1 issuance loops `step_forecast!`, whose per-step
ensemble spread is ~5× narrower than the `forecast_ahead` propagation the
calibration/replay path uses, so a v1 row at horizon > 1 would log a badly
under-wide band. Deployment is v2 (which resolves its interval from the
conformal sidecar), so multi-step v1 is latent; reject it explicitly rather
than risk the core integrator. Pure, so the guard is testable without the feeds.
"""
function _assert_issuable_model(model::Symbol, horizon_hours::Integer)
    (model == :v1 && horizon_hours > 1) && throw(ArgumentError(
        "multi-step v1 issuance (horizon_hours=$horizon_hours) is unsupported: the " *
        "per-step ensemble band is structurally too narrow at horizon > 1. Issue v2 " *
        "(conformal interval) or use the forecast_ahead-based replay path instead."
    ))
    return nothing
end

function issue_forecast(cfg::LiveVerifyConfig)
    _assert_issuable_model(cfg.model, cfg.horizon_hours)
    issue_time = now(UTC)
    plasma = fetch_swpc_plasma(; max_retries=3, retry_delay_sec=1.0)
    mag = fetch_swpc_mag(; max_retries=3, retry_delay_sec=1.0)
    dst_times, dst_vals = _fetch_dst()
    calibration = _load_calibration_for_model(cfg)
    conformal = _load_conformal_for_model(cfg)
    # Pairing check: a deployed (non-fallback) v2 calibration should always have its conformal
    # sidecar. If the sidecar is missing (deleted / partial restore), _resolve_interval silently
    # reverts to the static interval_scale band the conformal machinery exists to replace — warn
    # loudly so the operator sees the reversion rather than inferring it from interval_source.
    if cfg.model == :v2 && calibration !== nothing &&
       getfield(calibration, :label) != "operational_v2_fallback_v1_equiv" && conformal === nothing
        @warn("Deployed v2 calibration is missing its conformal sidecar; served interval reverts " *
              "to the static interval_scale band (not the calibrated conformal/ACI interval).",
              expected_sidecar=_conformal_path(cfg.v2_calibration_path))
    end

    latest_common_sw = min(maximum(plasma.time_tag), maximum(mag.time_tag))
    latest_complete_hour = _floor_hour(latest_common_sw)
    latest_dst_time = dst_times[end]
    latest_dst = dst_vals[end]
    target_time = _next_hourly_target(issue_time, cfg.horizon_hours, latest_dst_time)
    @assert target_time > issue_time
    @assert target_time > latest_dst_time

    # Staleness guard. The finite-count gap check below catches a HARD gap (no finite samples) but NOT a
    # FROZEN feed that still returns old-but-finite rows. Compare the feed vintage to the issue time: the
    # L1/DSCOVR solar wind is minute-cadence, so a multi-hour age means the uplink has stalled and the
    # forecast would be anchored to stale drivers (model steps silently balloon). Refuse gross solar-wind
    # staleness (mirrors the :hard refuse); warn on moderate solar-wind or Dst staleness (Kyoto provisional
    # Dst routinely lags a few hours, so that is a warning, not a refusal).
    sw_age_hours  = Dates.value(issue_time - latest_common_sw) / 3_600_000
    dst_age_hours = Dates.value(issue_time - latest_dst_time)  / 3_600_000
    if sw_age_hours > 6
        error(
            "Solar-wind feed stale: latest common sample is $(round(sw_age_hours; digits=1)) h before the " *
            "issue time (frozen/stalled L1 feed). Refusing to issue a forecast anchored to stale drivers."
        )
    end
    (sw_age_hours > 2 || dst_age_hours > 6) && @warn(
        "Stale input feed at issuance; forecast anchored to aged data.",
        sw_age_hours=round(sw_age_hours; digits=1), dst_age_hours=round(dst_age_hours; digits=1)
    )

    recent_start = latest_common_sw - Hour(1)
    recent = _drivers_for_window(plasma, mag, recent_start, latest_common_sw)

    # Data-coverage guard. The trailing solar-wind window backs both `recent`
    # (the persistence driver beyond the last complete hour) and the issued row.
    # An empty window makes `_drivers_for_window` fall back to quiet defaults
    # (V=400, Bz=0, ...), which would silently issue a no-storm forecast during
    # an L1/DSCOVR data gap. Refuse a hard gap; flag a partial gap.
    n_speed_finite = _window_finite_count(plasma, :speed, recent_start, latest_common_sw)
    n_bz_finite = _window_finite_count(mag, :bz_gsm, recent_start, latest_common_sw)
    # density and By back `n`/clock-angle terms; an all-NaN window for either one
    # silently substitutes quiet defaults (n=5, By=0) via `_finite_mean`, which
    # fabricates Pdyn and the clock-angle features. Count them so a fallback in
    # ANY of the four drivers flags `driver_data_gap` rather than passing silently.
    n_density_finite = _window_finite_count(plasma, :density, recent_start, latest_common_sw)
    n_by_finite = _window_finite_count(mag, :by_gsm, recent_start, latest_common_sw)
    gap_status = _driver_gap_status(n_speed_finite, n_bz_finite, n_density_finite, n_by_finite)
    if gap_status == :hard
        error(
            "Solar-wind data gap: no finite speed or Bz samples in the trailing " *
            "hour [$recent_start, $latest_common_sw]. Refusing to issue a forecast " *
            "built on quiet-time fallback drivers."
        )
    end
    driver_data_gap = gap_status != :ok
    driver_data_gap && @warn(
        "Partial solar-wind data gap; some drivers fell back to defaults.",
        n_speed_finite, n_bz_finite, n_density_finite, n_by_finite,
        recent_start, latest_common_sw
    )
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

    # P1-2: count intermediate steps whose driver window had no finite speed or
    # Bz and therefore fell back to frozen persistence drivers. Persisted so the
    # logged row discloses how many multi-step hours were driven by carried-over
    # rather than observed solar wind (silent before).
    n_steps_driver_fallback = 0
    step_time = latest_dst_time + Hour(1)
    while step_time <= target_time
        source_hour = step_time - Hour(1)
        drivers = if source_hour < latest_complete_hour
            n_steps_driver_fallback +=
                _step_driver_fallback(plasma, mag, source_hour) ? 1 : 0
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
    persistence_dst = latest_dst
    burton_dst = _dst_from_dst_star(burton_star, used_drivers.Pdyn)
    burton_full_dst = _dst_from_dst_star(burton_full_star, used_drivers.Pdyn)
    obrien_dst = _dst_from_dst_star(obrien_star, used_drivers.Pdyn)
    baseline_predictions = (
        persistence=persistence_dst,
        burton=burton_dst,
        burton_full=burton_full_dst,
        obrien=obrien_dst,
    )
    memory_features = _live_v2_memory_features(
        plasma,
        mag,
        dst_times,
        dst_vals,
        latest_dst_time,
        used_drivers,
    )
    features = _v2_features(
        latest_dst,
        used_drivers;
        memory=memory_features,
        baselines=baseline_predictions,
        v1_pred_dst=v1_pred_dst,
    )
    selected = _select_model_prediction(
        cfg.model,
        calibration,
        latest_dst,
        used_drivers,
        v1_pred_dst,
        v1_ci05_dst,
        v1_ci95_dst,
        baselines=baseline_predictions,
        features=features,
    )
    pred_dst = selected.pred_dst
    ci05_dst = selected.ci05_dst
    ci95_dst = selected.ci95_dst
    model_steps = Int((target_time - latest_dst_time) / Hour(1))
    wall_horizon = (target_time - issue_time) / Hour(1)

    # N1: when a conformal calibration is deployed, the logged 90% interval comes
    # from the stratified conformal half-width (horizon × activity regime),
    # replacing the v1-ensemble-spread interval-scale machinery that structurally
    # undercovered. The point forecast is unchanged.
    ci05_dst, ci95_dst, interval_source = _resolve_interval(
        conformal, pred_dst, model_steps, latest_dst, ci05_dst, ci95_dst,
    )
    # N2: prefer the ONLINE adaptive-conformal (ACI) interval, which holds ~90%
    # coverage under storm-driven distribution shift where the static stratified
    # interval drops (≈0.84 on broad replay vs 0.89 for ACI). The ACI state is
    # derived per horizon from the verified log. Fail-safe: insufficient history or
    # any error keeps the static interval above; the point forecast is unchanged.
    let aci = _aci_interval_from_log(cfg.log_path, pred_dst, model_steps; latest_dst=latest_dst)
        if aci !== nothing
            ci05_dst, ci95_dst, interval_source = aci[1], aci[2], "aci"
        end
    end
    # Reflect the deployed interval in the v2 CI columns the report scores.
    v2_ci05_log = interval_source in ("conformal", "aci") ? ci05_dst : selected.v2_ci05_dst
    v2_ci95_log = interval_source in ("conformal", "aci") ? ci95_dst : selected.v2_ci95_dst

    # ---- Driver beyond the L1 advection window: frozen-driver default (Direction B removed) ----
    # The multi-hour driver is unobserved and not predictable from the data available at issue time. A
    # regime-aware relaxation of the frozen southward field (Direction B) was evaluated and does not improve
    # broad-population accuracy over freezing — it under-predicts the depth of actively intensifying storms — so
    # the served forecast holds the driver constant (v2). v2 already carries the L1 look-ahead (Direction A) on
    # observed hours, where the solar wind is fresher than Dst. The improved_*/served_* columns are retained
    # equal to v2 so the live log keeps a stable schema and the dashboard/historical rows read consistently.
    # ponytail: improved_*/served_* kept == v2 for CSV-schema stability; drop the columns in a log migration if desired.
    improved_pred_dst = selected.v2_pred_dst; improved_ci05 = v2_ci05_log; improved_ci95 = v2_ci95_log

    # ---- L1 look-ahead layer (Direction A) ----
    # Same discovered ODE and v2 residual correction, but the frozen-driver tail is driven by the 1-min L1 wind
    # propagating into each Earth-hour and already measured by issue time (leakage-safe; reduces to v2 where no
    # measured wind maps in). Backtested on the storm set (live_forecasts/validate_ballistic_subhourly.jl):
    # continuity to v2 = 0, and lower RMSE than v2 at every lead (15.9/28.7/38.8/67.5 vs 18.0/30.2/40.1/68.3 nT),
    # so it is PROMOTED to the served forecast.
    sub_hourly_pred_dst = selected.v2_pred_dst
    try
        sstate = init_forecast(; coefficients_csv=coef_csv, ensemble_csv=ens_csv, t0=latest_dst_time, dst0=anchor_dst_star)
        sresult = nothing; sst = latest_dst_time + Hour(1)
        while sst <= target_time
            sh = sst - Hour(1)
            sdrv = sh < latest_complete_hour ?
                _drivers_for_window(plasma, mag, sh, sh + Hour(1); fallback=recent) :   # observed hour (== v2)
                _subhourly_driver(plasma, mag, sst, recent, latest_common_sw)           # L1 look-ahead on the tail
            sresult = step_forecast!(sstate, sst, sdrv.V, sdrv.Bz, sdrv.By, sdrv.n, sdrv.Pdyn)
            sst += Hour(1)
        end
        if sresult !== nothing && isfinite(sresult.dst_predicted)
            sv1 = _dst_from_dst_star(sresult.dst_predicted, used_drivers.Pdyn)
            sub_hourly_pred_dst = sv1 + selected.v2_correction
        end
    catch e
        @warn "L1 look-ahead forecast failed; serving v2" exception=(e, catch_backtrace())
    end
    sub_hourly_ci05 = sub_hourly_pred_dst + (v2_ci05_log - selected.v2_pred_dst)
    sub_hourly_ci95 = sub_hourly_pred_dst + (v2_ci95_log - selected.v2_pred_dst)

    # ---- PROMOTED served forecast = v2 + L1 look-ahead. Severity (build_status) applies a depth-safe floor
    # min(v2, served) so the look-ahead can only escalate, never under-warn, relative to frozen v2.
    served_pred_dst = sub_hourly_pred_dst
    served_ci05_dst = sub_hourly_ci05
    served_ci95_dst = sub_hourly_ci95

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
        driver_data_gap=[driver_data_gap],
        n_speed_finite_trailing_hour=[n_speed_finite],
        n_bz_finite_trailing_hour=[n_bz_finite],
        n_density_finite_trailing_hour=[n_density_finite],
        n_by_finite_trailing_hour=[n_by_finite],
        n_steps_driver_fallback=[n_steps_driver_fallback],
        interval_source=[interval_source],
        V_kms=[used_drivers.V],
        Bz_nt=[used_drivers.Bz],
        By_nt=[used_drivers.By],
        n_cm3=[used_drivers.n],
        Pdyn_npa=[used_drivers.Pdyn],
        Bsouth_nt=[features.Bsouth_nt],
        VBsouth_mvm=[features.VBsouth_mvm],
        Bperp_nt=[features.Bperp_nt],
        clock_angle_sin2=[features.clock_angle_sin2],
        sqrt_Pdyn_npa=[features.sqrt_Pdyn_npa],
        dst_delta_1h_nt=[features.dst_delta_1h_nt],
        dst_delta_3h_nt=[features.dst_delta_3h_nt],
        Bz_delta_1h_nt=[features.Bz_delta_1h_nt],
        VBsouth_delta_1h_mvm=[features.VBsouth_delta_1h_mvm],
        VBsouth_mean_3h_mvm=[features.VBsouth_mean_3h_mvm],
        Bsouth_mean_3h_nt=[features.Bsouth_mean_3h_nt],
        baseline_spread_nt=[features.baseline_spread_nt],
        v1_minus_persistence_nt=[features.v1_minus_persistence_nt],
        obrien_minus_v1_nt=[features.obrien_minus_v1_nt],
        burton_minus_v1_nt=[features.burton_minus_v1_nt],
        pred_dst_star_nt=[result.dst_predicted],
        pred_dst_nt=[pred_dst],
        pred_dst_ci05_nt=[ci05_dst],
        pred_dst_ci95_nt=[ci95_dst],
        v1_pred_dst_nt=[v1_pred_dst],
        v1_pred_dst_ci05_nt=[v1_ci05_dst],
        v1_pred_dst_ci95_nt=[v1_ci95_dst],
        v2_pred_dst_nt=[selected.v2_pred_dst],
        v2_pred_dst_ci05_nt=[v2_ci05_log],
        v2_pred_dst_ci95_nt=[v2_ci95_log],
        v2_correction_dst_nt=[selected.v2_correction],
        v2_interval_scale=[selected.v2_interval_scale],
        v2_calibration_label=[selected.v2_label],
        v2_selected_component=[selected.v2_selected_component],
        v2_selected_component_pred_nt=[selected.v2_selected_component_pred],
        persistence_dst_nt=[persistence_dst],
        burton_dst_nt=[burton_dst],
        burton_full_dst_nt=[burton_full_dst],
        obrien_dst_nt=[obrien_dst],
        improved_model_version=["v2"],
        improved_pred_dst_nt=[improved_pred_dst],
        improved_pred_dst_ci05_nt=[improved_ci05],
        improved_pred_dst_ci95_nt=[improved_ci95],
        served_pred_dst_nt=[served_pred_dst],
        served_pred_dst_ci05_nt=[served_ci05_dst],
        served_pred_dst_ci95_nt=[served_ci95_dst],
        sub_hourly_model_version=["v2+L1A"],
        sub_hourly_pred_dst_nt=[sub_hourly_pred_dst],
        sub_hourly_pred_dst_ci05_nt=[sub_hourly_ci05],
        sub_hourly_pred_dst_ci95_nt=[sub_hourly_ci95],
        observation_dst_nt=[missing],
        residual_dst_nt=[missing],
        observed_in_90ci=[missing],
        persistence_residual_dst_nt=[missing],
        burton_residual_dst_nt=[missing],
        burton_full_residual_dst_nt=[missing],
        obrien_residual_dst_nt=[missing],
    )
    row_idx = _append_forecast!(cfg.log_path, row)

    # Sub-hour model trajectory (display only) for the latest cycle; overwritten each issue (idempotent).
    try
        traj = _subhour_trajectory(coef_csv, ens_csv, latest_dst_time, anchor_dst_star, plasma, mag, recent,
                                   latest_complete_hour, latest_common_sw, used_drivers.Pdyn, selected.v2_correction)
        open(joinpath(dirname(cfg.log_path), "subhour_trajectory.json"), "w") do io
            JSON3.write(io, Dict("issue_time_utc" => string(issue_time),
                                 "anchor_time_utc" => string(latest_dst_time),
                                 "anchor_dst_nt" => latest_dst, "points" => traj))
        end
    catch e
        @warn "sub-hour trajectory write failed" exception=(e, catch_backtrace())
    end

    println("Logged live forecast row $row_idx: $(cfg.log_path)")
    println("Issue UTC: $issue_time")
    println("Latest SWPC solar wind: $latest_common_sw")
    println("Latest observed Kyoto Dst: $latest_dst_time = $latest_dst nT")
    println("Target observation UTC: $target_time")
    model_label = selected.model_version == "v2" ? "Operational v2" : "SINDy v1"
    println("Forecast model: $model_label")
    println("Lead time: $(round(wall_horizon; digits=3)) hr wall-clock, $model_steps model steps")
    println("Forecast Dst*: $(round(result.dst_predicted; digits=2)) nT")
    println(
        "$model_label Dst: $(round(pred_dst; digits=2)) nT; 90% CI " *
        "[$(round(ci05_dst; digits=2)), $(round(ci95_dst; digits=2))]"
    )
    if selected.model_version == "v2"
        println(
            "SINDy-v1 Dst: $(round(v1_pred_dst; digits=2)) nT; " *
            "v2 correction=$(round(selected.v2_correction; digits=2)) nT; " *
            "interval scale=$(round(selected.v2_interval_scale; digits=2)); " *
            "component=$(selected.v2_selected_component)"
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

    verified > 0 && _atomic_csv(cfg.log_path, df)
    println("Verified $verified pending forecast row(s).")
    return verified
end

function refresh_observations!(cfg::LiveVerifyConfig; dst_times=nothing, dst_vals=nothing)
    isfile(cfg.log_path) || error("No forecast log exists at $(cfg.log_path)")
    df = CSV.read(cfg.log_path, DataFrame)
    if dst_times === nothing || dst_vals === nothing
        dst_times, dst_vals = _fetch_dst()
    end
    dst_map = _dst_lookup(dst_times, dst_vals)

    updated = 0
    changed = 0
    for row_idx in 1:nrow(df)
        String(:target_time_utc) in names(df) || continue
        target = _parse_dt(df[row_idx, :target_time_utc])
        haskey(dst_map, target) || continue
        observed = Float64(dst_map[target])
        old_observed = _optional_float(df, row_idx, :observation_dst_nt)
        if ismissing(old_observed) || Float64(old_observed) != observed
            !ismissing(old_observed) && (changed += 1)
            _score_row!(df, row_idx, observed)
            updated += 1
        end
    end

    updated > 0 && _atomic_csv(cfg.log_path, df)
    println(
        "Refreshed $updated observation row(s); " *
        "$changed previously scored row(s) changed."
    )
    return updated
end

function _set_if_missing!(df::DataFrame, row_idx::Int, col::Symbol, value)
    if !(String(col) in names(df)) || ismissing(df[row_idx, col])
        _set_value!(df, row_idx, col, value)
        return true
    end
    return false
end

"""
    backfill_baselines!(log_path)

Fill baseline forecasts/residuals for rows that lack them. This is a
**fill-if-missing** operation: `issue_forecast` advances the baselines with the
true per-hour drivers, but only the final hour's drivers are stored, so
re-deriving from the row uses a single frozen driver and would NOT reproduce the
issued multi-step baselines. To avoid silently overwriting issued values with
that frozen-driver approximation, existing baseline cells are left untouched;
only missing ones are filled (a documented approximation for legacy rows).
"""
function backfill_baselines!(log_path::String)
    isfile(log_path) || error("No forecast log exists at $log_path")
    df = CSV.read(log_path, DataFrame)
    updated = 0

    for row_idx in 1:nrow(df)
        required = (:anchor_dst_star_nt, :latest_dst_nt, :target_time_utc,
                    :latest_dst_time_utc, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa)
        all(String(col) in names(df) for col in required) || continue
        baseline = _baseline_predictions_from_row(df, row_idx)

        filled = false
        filled |= _set_if_missing!(df, row_idx, :persistence_dst_nt, baseline.persistence)
        filled |= _set_if_missing!(df, row_idx, :burton_dst_nt, baseline.burton)
        filled |= _set_if_missing!(df, row_idx, :burton_full_dst_nt, baseline.burton_full)
        filled |= _set_if_missing!(df, row_idx, :obrien_dst_nt, baseline.obrien)
        filled |= _set_if_missing!(df, row_idx, :model_step_hours, baseline.model_steps)

        observed = _optional_float(df, row_idx, :observation_dst_nt)
        if filled && !ismissing(observed)
            _score_row!(df, row_idx, observed)
        end
        filled && (updated += 1)
    end

    updated > 0 && _atomic_csv(log_path, df)
    println("Backfilled baseline forecasts for $updated row(s) (fill-if-missing).")
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
            _atomic_csv(cfg.log_path, df)
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

function _copy_config(cfg::LiveVerifyConfig; kwargs...)
    values = Dict{Symbol,Any}(
        :mode => cfg.mode,
        :model => cfg.model,
        :poll_seconds => cfg.poll_seconds,
        :timeout_hours => cfg.timeout_hours,
        :horizon_hours => cfg.horizon_hours,
        :campaign_horizons => copy(cfg.campaign_horizons),
        :log_path => cfg.log_path,
        :report_path => cfg.report_path,
        :replay_hours => cfg.replay_hours,
        :replay_horizons => copy(cfg.replay_horizons),
        :table_path => cfg.table_path,
        :table_limit => cfg.table_limit,
        :v2_calibration_path => cfg.v2_calibration_path,
        :v2_train_fraction => cfg.v2_train_fraction,
        :v2_validation_fraction => cfg.v2_validation_fraction,
        :v2_ridge => cfg.v2_ridge,
        :v2_ridge_grid => copy(cfg.v2_ridge_grid),
        :v2_interval_coverage => cfg.v2_interval_coverage,
        :v2_selector_margin_nt => cfg.v2_selector_margin_nt,
        :v2_coverage_floor => cfg.v2_coverage_floor,
    )
    for (key, value) in kwargs
        values[key] = value
    end
    return LiveVerifyConfig(; values...)
end

function _rows_verified(log_path::String, row_indices::Vector{Int})
    isfile(log_path) || return false
    df = CSV.read(log_path, DataFrame)
    String(:observation_dst_nt) in names(df) || return false
    for row_idx in row_indices
        if row_idx > nrow(df) || ismissing(df[row_idx, :observation_dst_nt])
            return false
        end
    end
    return true
end

function _pending_row_indices(log_path::String, row_indices::Vector{Int})
    isfile(log_path) || return copy(row_indices)
    df = CSV.read(log_path, DataFrame)
    String(:observation_dst_nt) in names(df) || return copy(row_indices)
    return [
        row_idx for row_idx in row_indices
        if row_idx > nrow(df) || ismissing(df[row_idx, :observation_dst_nt])
    ]
end

function run_campaign(cfg::LiveVerifyConfig;
                      issue_fn=issue_forecast,
                      verify_fn=verify_pending!,
                      report_fn=write_live_comparison_report,
                      sleep_fn=sleep,
                      clock_fn=() -> now(UTC))
    forecasts = NamedTuple[]
    row_indices = Int[]
    println(
        "Starting locked live campaign: model=$(cfg.model), " *
        "horizons=$(join(cfg.campaign_horizons, ",")) hr"
    )
    for horizon in cfg.campaign_horizons
        issue_cfg = _copy_config(cfg; mode=:issue, horizon_hours=horizon)
        forecast = issue_fn(issue_cfg)
        push!(forecasts, forecast)
        push!(row_indices, Int(forecast.row_idx))
    end

    report_fn(cfg.log_path, cfg.report_path)
    deadline = clock_fn() + Millisecond(round(Int, cfg.timeout_hours * 3600 * 1000))
    while !_rows_verified(cfg.log_path, row_indices)
        verified = verify_fn(cfg)
        report_fn(cfg.log_path, cfg.report_path)
        pending = _pending_row_indices(cfg.log_path, row_indices)
        println(
            "Campaign verification pass: newly_verified=$verified, " *
            "campaign_pending=$(length(pending))"
        )
        isempty(pending) && break
        clock_fn() >= deadline && error(
            "Timed out waiting for campaign rows $(join(pending, ",")); " *
            "report written to $(cfg.report_path)"
        )
        sleep_fn(cfg.poll_seconds)
    end
    println("Campaign complete: all $(length(row_indices)) issued forecast row(s) scored.")
    return (; forecasts, rows=row_indices, report_path=cfg.report_path)
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
        "Operational v2" => :v2_pred_dst_nt,
        "SINDy v1" => :v1_pred_dst_nt,
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

function _has_prediction(df::DataFrame, row_idx::Int, pred_col::Symbol)
    value = _prediction_value(df, row_idx, pred_col)
    return !ismissing(value) && isfinite(value)
end

function _same_row_model_indices(df::DataFrame, rows::Vector{Int},
                                 model_specs::Vector{Pair{String,Symbol}})
    required = last.(model_specs)
    return [row_idx for row_idx in rows if all(col -> _has_prediction(df, row_idx, col), required)]
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

function _interval_contains(df::DataFrame, row_idx::Int, ci05_col::Symbol, ci95_col::Symbol)
    observed = _optional_float(df, row_idx, :observation_dst_nt)
    ci05 = _optional_float(df, row_idx, ci05_col)
    ci95 = _optional_float(df, row_idx, ci95_col)
    if ismissing(observed) || ismissing(ci05) || ismissing(ci95)
        return missing
    end
    return min(ci05, ci95) <= observed <= max(ci05, ci95)
end

function _v2_residual(df::DataFrame, row_idx::Int)
    observed = _optional_float(df, row_idx, :observation_dst_nt)
    pred = _prediction_value(df, row_idx, :v2_pred_dst_nt)
    if ismissing(observed) || ismissing(pred)
        return missing
    end
    return observed - pred
end

function write_live_comparison_report(log_path::String, report_path::String)
    isfile(log_path) || error("No forecast log exists at $log_path")
    df = CSV.read(log_path, DataFrame)
    verified = _verified_indices(df)
    valid_verified = [i for i in verified if _row_is_strictly_future(df, i)]
    invalid_verified = setdiff(verified, valid_verified)
    pending = _pending_indices(df)
    model_specs = _standard_model_columns(df)
    comparison_rows = _same_row_model_indices(df, valid_verified, model_specs)
    coverage = _coverage_fraction(df, comparison_rows)

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
    push!(lines, "Same-row v2 comparison rows: $(length(comparison_rows))")
    if !ismissing(coverage)
        push!(lines, "Operational v2 90% interval coverage: $(_fmt3(coverage))")
    end

    push!(lines, "")
    push!(lines, "## Same-Row Model Comparison")
    push!(lines, "")
    push!(lines, "Operational v2 is the upgraded method. This table compares v2, v1, and baselines on the identical verified rows.")
    push!(lines, "")
    push!(lines, "| model | n | RMSE nT | MAE nT | bias nT |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: |")
    for spec in model_specs
        preds, obs = _metric_rows_for_indices(df, last(spec), comparison_rows)
        push!(lines, _metric_markdown_row(first(spec), preds, obs))
    end

    push!(lines, "")
    push!(lines, "## Verified Operational V2 Rows")
    push!(lines, "")
    push!(lines, "| issue UTC | target UTC | lead h | observed | operational v2 pred | v2 residual obs-pred | v2 abs error | v2 inside 90% CI | SINDy v1 pred | persistence | Burton | OBrien |")
    push!(lines, "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for row_idx in comparison_rows
        observed = _optional_float(df, row_idx, :observation_dst_nt)
        pred = _prediction_value(df, row_idx, :v2_pred_dst_nt)
        residual = _v2_residual(df, row_idx)
        abs_error = ismissing(residual) ? missing : abs(residual)
        lead = _optional_float(df, row_idx, :wall_clock_lead_hours)
        if ismissing(lead)
            lead = _optional_float(df, row_idx, :horizon_hours)
        end
        in_ci = _interval_contains(df, row_idx, :v2_pred_dst_ci05_nt, :v2_pred_dst_ci95_nt)
        push!(lines,
            "| $(_fmt_text(df[row_idx, :issue_time_utc])) | " *
            "$(_fmt_text(df[row_idx, :target_time_utc])) | " *
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
        push!(lines, "| issue UTC | target UTC | model | operational v2 pred | CI05 | CI95 |")
        push!(lines, "| --- | --- | --- | ---: | ---: | ---: |")
        for row_idx in pending
            model_version = _row_model_version(df, row_idx)
            pred = _prediction_value(df, row_idx, :v2_pred_dst_nt)
            ismissing(pred) && (pred = _optional_float(df, row_idx, :pred_dst_nt))
            ci05 = _optional_float(df, row_idx, :v2_pred_dst_ci05_nt)
            ismissing(ci05) && (ci05 = _optional_float(df, row_idx, :pred_dst_ci05_nt))
            ci95 = _optional_float(df, row_idx, :v2_pred_dst_ci95_nt)
            ismissing(ci95) && (ci95 = _optional_float(df, row_idx, :pred_dst_ci95_nt))
            push!(lines,
                "| $(_fmt_text(df[row_idx, :issue_time_utc])) | " *
                "$(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "$(_fmt_text(model_version)) | " *
                "$(_fmt2(pred)) | $(_fmt2(ci05)) | $(_fmt2(ci95)) |"
            )
        end
    end

    if !isempty(comparison_rows)
        worst = sort(comparison_rows; by=i -> abs(_v2_residual(df, i)), rev=true)
        push!(lines, "")
        push!(lines, "## Worst Operational V2 Misses")
        push!(lines, "")
        push!(lines, "| target UTC | observed | operational v2 pred | residual obs-pred | abs error |")
        push!(lines, "| --- | ---: | ---: | ---: | ---: |")
        for row_idx in worst[1:min(10, length(worst))]
            residual = _v2_residual(df, row_idx)
            push!(lines,
                "| $(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "$(_fmt2(_optional_float(df, row_idx, :observation_dst_nt))) | " *
                "$(_fmt2(_prediction_value(df, row_idx, :v2_pred_dst_nt))) | " *
                "$(_fmt2(residual)) | $(_fmt2(abs(residual))) |"
            )
        end
    end

    if String(:v2_selected_component) in names(df) && !isempty(comparison_rows)
        push!(lines, "")
        push!(lines, "## Operational V2 Audit")
        push!(lines, "")
        push!(lines, "The component column is internal v2 audit metadata, not a separate headline model.")
        push!(lines, "")
        push!(lines, "| target UTC | v2 component |")
        push!(lines, "| --- | --- |")
        for row_idx in comparison_rows
            component = df[row_idx, :v2_selected_component]
            component_text = ismissing(component) || isempty(String(component)) ?
                "not recorded" : String(component)
            push!(lines,
                "| $(_fmt_text(df[row_idx, :target_time_utc])) | " *
                "$component_text |"
            )
        end
    end

    push!(lines, "")
    push!(lines, "## Standard Interpretation")
    push!(lines, "")
    push!(lines, "- A row is correct for point accuracy only by its absolute error against the locked target observation.")
    push!(lines, "- A row is correct for probabilistic coverage only if the observation falls inside the locked interval.")
    push!(lines, "- Operational v2 is the upgraded method; judge it against v1 and baselines on the same verified rows.")
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
        ("Operational v2", :v2_pred_dst_nt),
        ("SINDy v1", :v1_pred_dst_nt),
        ("Persistence", :persistence_dst_nt),
        ("Burton", :burton_dst_nt),
        ("BurtonFull", :burton_full_dst_nt),
        ("OBrienMcP", :obrien_dst_nt),
    )
        preds, obs = _metric_rows(df, col)
        if isempty(preds) && col == :v1_pred_dst_nt
            preds, obs = _metric_rows(df, :pred_dst_nt)
        end
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
    elseif cfg.mode == :refresh_observations
        refresh_observations!(cfg)
    elseif cfg.mode == :backfill_baselines
        backfill_baselines!(cfg.log_path)
    elseif cfg.mode == :replay_recent
        run_replay_recent(cfg)
    elseif cfg.mode == :replay_omni
        run_replay_omni(cfg)
    elseif cfg.mode == :fit_v2_calibration
        fit_v2_calibration!(cfg)
    elseif cfg.mode == :wait
        forecast = issue_forecast(cfg)
        wait_for_observation(cfg, forecast)
    elseif cfg.mode == :campaign
        run_campaign(cfg)
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
