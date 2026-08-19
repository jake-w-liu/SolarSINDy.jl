#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Printf
using Statistics

isdefined(@__MODULE__, :_selftest_v2) || include(joinpath(@__DIR__, "v2_replay.jl"))

const BROAD_CATALOG = OPERATIONAL_STORM_CATALOG
const BROAD_OUT_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_scored.csv")
const BROAD_OUT_SUMMARY = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_summary.csv")
const BROAD_OUT_MD = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_report.md")
const BROAD_DEFAULT_THRESHOLD = -100.0
const BROAD_EXPECTED_CATALOG_COUNT = 714
const BROAD_EXPECTED_INTENSE_COUNT = 193
const BROAD_EXPECTED_SEVERE_COUNT = 30
const BROAD_EXPECTED_EXTREME_COUNT = 8

Base.@kwdef struct BroadReplayConfig
    threshold_dst_star_nt::Float64 = BROAD_DEFAULT_THRESHOLD
    split::String = "all"
    limit::Int = 0
    out_csv::String = BROAD_OUT_CSV
    out_summary::String = BROAD_OUT_SUMMARY
    out_report::String = BROAD_OUT_MD
    self_test_only::Bool = false
end

function _parse_broad_datetime(x)
    x isa DateTime && return x
    s = strip(String(x))
    s = replace(s, r"\.\d+$" => "")
    return DateTime(s)
end

function _parse_broad_args(args)::BroadReplayConfig
    cfg = BroadReplayConfig()
    threshold = cfg.threshold_dst_star_nt
    split_name = cfg.split
    limit = cfg.limit
    out_csv = cfg.out_csv
    out_summary = cfg.out_summary
    out_report = cfg.out_report
    self_test_only = cfg.self_test_only

    for arg in args
        if arg == "--self-test"
            self_test_only = true
        elseif startswith(arg, "--threshold-dst-star=")
            threshold = parse(Float64, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--threshold=")
            # Backward-compatible CLI alias; all emitted metadata is explicit Dst*.
            threshold = parse(Float64, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--split=")
            split_name = lowercase(strip(Base.split(arg, "=", limit = 2)[2]))
        elseif startswith(arg, "--limit=")
            limit = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--out-csv=")
            out_csv = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-summary=")
            out_summary = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-report=")
            out_report = Base.split(arg, "=", limit = 2)[2]
        else
            error("unknown argument: $arg")
        end
    end

    limit >= 0 || throw(ArgumentError("--limit must be nonnegative"))
    split_name in ("all", "train", "val", "test") ||
        throw(ArgumentError("--split must be all/train/val/test"))
    return BroadReplayConfig(;
        threshold_dst_star_nt = threshold,
        split = split_name,
        limit = limit,
        out_csv = out_csv,
        out_summary = out_summary,
        out_report = out_report,
        self_test_only = self_test_only,
    )
end

function load_storm_catalog(path::AbstractString = BROAD_CATALOG)
    df = CSV.read(path, DataFrame)
    required = [:storm_id, :onset_time, :min_dst_star, :min_dst_star_time,
                :recovery_end_time, :duration_hr, :solar_cycle, :split,
                :onset_idx, :end_idx]
    missing_cols = [String(c) for c in required if !(String(c) in names(df))]
    isempty(missing_cols) || error("storm catalog missing columns: $(join(missing_cols, ", "))")
    return df
end

function select_broad_storms(catalog::DataFrame; threshold_dst_star_nt::Real = BROAD_DEFAULT_THRESHOLD,
                             split::AbstractString = "all", limit::Int = 0)
    threshold = Float64(threshold_dst_star_nt)
    selected = catalog[Float64.(catalog.min_dst_star) .<= threshold, :]
    if split != "all"
        selected = selected[String.(selected.split) .== split, :]
    end
    sort!(selected, [:onset_time, :storm_id])
    if limit > 0
        selected = first(selected, min(limit, nrow(selected)))
    end
    return selected
end

function _storm_from_row(r)
    sid = Int(r.storm_id)
    t0 = _parse_broad_datetime(r.onset_time)
    tmin = _parse_broad_datetime(r.min_dst_star_time)
    t1 = _parse_broad_datetime(r.recovery_end_time)
    min_dst_star = Float64(r.min_dst_star)
    return (
        name = @sprintf("Catalog storm %d (%s, min Dst* %.1f nT)", sid, Dates.format(t0, dateformat"yyyy-mm-dd"), min_dst_star),
        t0 = t0,
        t1 = t1,
        storm_id = sid,
        min_dst_star = min_dst_star,
        min_dst_star_time = tmin,
        duration_hr = Float64(r.duration_hr),
        solar_cycle = Int(r.solar_cycle),
        split = String(r.split),
    )
end

function _with_broad_metadata(rows::DataFrame, storm)
    out = copy(rows)
    out[!, :storm_id] = fill(storm.storm_id, nrow(out))
    out[!, :storm_min_dst_star_nt] = fill(storm.min_dst_star, nrow(out))
    out[!, :storm_split] = fill(storm.split, nrow(out))
    out[!, :storm_solar_cycle] = fill(storm.solar_cycle, nrow(out))
    out[!, :storm_onset_utc] = fill(storm.t0, nrow(out))
    out[!, :storm_min_dst_star_utc] = fill(storm.min_dst_star_time, nrow(out))
    out[!, :storm_recovery_end_utc] = fill(storm.t1, nrow(out))
    out[!, :target_utc] = out.issue_utc .+ Hour.(Int.(out.lead))
    return select(out, :storm_id, :storm, :storm_split, :storm_solar_cycle,
                  :storm_min_dst_star_nt, :storm_onset_utc, :storm_min_dst_star_utc,
                  :storm_recovery_end_utc, :issue_utc, :target_utc, :lead,
                  :obs, :v2_1, :v2_1_pre_rate_guard,
                  :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                  :v2_0, :v2_1_frozen,
                  :persistence, :rate)
end

function run_broad_replay(selected::DataFrame)
    current_core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    historical_core = load_operational_core(OPERATIONAL_V2_0_MODEL_VERSION)
    current_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
    historical_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_0_MODEL_VERSION).point_csv,
    )
    min_year = minimum(year(_parse_broad_datetime(t)) for t in selected.onset_time) - 1
    max_year = maximum(year(_parse_broad_datetime(t)) for t in selected.recovery_end_time)
    archive = _load_replay_archive(min_year, max_year)
    scored = DataFrame()
    skipped = DataFrame(storm_id = Int[], reason = String[])

    for r in eachrow(selected)
        storm = _storm_from_row(r)
        try
            rows = replay_v2_storm(
                storm, current_core, current_cal, historical_core, historical_cal,
                archive.lookup; replay_inputs=archive.inputs,
            )
            if nrow(rows) == 0
                push!(skipped, (storm.storm_id, "no finite scored rows"))
            else
                append!(scored, _with_broad_metadata(rows, storm); cols = :union)
            end
        catch err
            push!(skipped, (storm.storm_id, sprint(showerror, err)))
        end
    end
    return scored, skipped
end

_broad_rmse(residuals) = sqrt(mean(abs2, Float64.(residuals)))

function _metric_row(rows::DataFrame, cohort::AbstractString, lead::Int,
                     threshold_dst_star_nt::Real)
    sub = rows[rows.lead .== lead, :]
    nrow(sub) == 0 && return nothing
    rv20 = _broad_rmse(sub.obs .- sub.v2_0)
    rv21 = _broad_rmse(sub.obs .- sub.v2_1)
    rpers = _broad_rmse(sub.obs .- sub.persistence)
    storm_count = length(unique(Int.(sub.storm_id)))
    return (
        cohort = String(cohort),
        threshold_dst_star_nt = Float64(threshold_dst_star_nt),
        lead_h = lead,
        n_rows = nrow(sub),
        n_storms = storm_count,
        rmse_v2_0_nt = rv20,
        rmse_v2_1_nt = rv21,
        rmse_persistence_nt = rpers,
        improvement_vs_best_nt = min(rv20, rpers) - rv21,
        max_tail_effect_nt = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen))),
        max_core_change_nt = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0))),
    )
end

function broad_summary(rows::DataFrame;
                       threshold_dst_star_nt::Real = BROAD_DEFAULT_THRESHOLD)
    out = DataFrame(cohort = String[], threshold_dst_star_nt = Float64[], lead_h = Int[],
                    n_rows = Int[], n_storms = Int[],
                    rmse_v2_0_nt = Float64[], rmse_v2_1_nt = Float64[],
                    rmse_persistence_nt = Float64[],
                    improvement_vs_best_nt = Float64[], max_tail_effect_nt = Float64[],
                    max_core_change_nt = Float64[])
    isempty(rows) && return out
    for (cohort, sub) in (("all", rows),)
        for lead in LEADS
            m = _metric_row(sub, cohort, lead, threshold_dst_star_nt)
            m === nothing || push!(out, m)
        end
    end
    for split_name in sort(unique(String.(rows.storm_split)))
        sub = rows[String.(rows.storm_split) .== split_name, :]
        for lead in LEADS
            m = _metric_row(sub, "split=$(split_name)", lead, threshold_dst_star_nt)
            m === nothing || push!(out, m)
        end
    end
    for sev in (-100.0, -200.0, -300.0)
        sub = rows[Float64.(rows.storm_min_dst_star_nt) .<= sev, :]
        isempty(sub) && continue
        for lead in LEADS
            m = _metric_row(sub, @sprintf("storm_min_dst_star<=%.0f", sev), lead, sev)
            m === nothing || push!(out, m)
        end
    end
    return out
end

function _validate_broad_rows(rows::DataFrame)
    isempty(rows) && error("broad replay produced no scored rows")
    required = [:storm_id, :storm, :storm_split, :storm_min_dst_star_nt, :issue_utc,
                :target_utc, :lead, :obs, :v2_1, :v2_1_pre_rate_guard,
                :v2_1_pre_one_hour_inertia, :v2_1_pre_state_inertia,
                :v2_0, :v2_1_frozen,
                :persistence, :rate]
    missing_cols = [String(c) for c in required if !(String(c) in names(rows))]
    isempty(missing_cols) || error("broad replay missing columns: $(join(missing_cols, ", "))")
    numeric = [:obs, :v2_1, :v2_1_pre_rate_guard, :v2_1_pre_one_hour_inertia,
               :v2_1_pre_state_inertia,
               :v2_0, :v2_1_frozen, :persistence]
    for col in numeric
        all(isfinite, Float64.(rows[!, col])) || error("non-finite values in $col")
    end
    all(rows.target_utc .== rows.issue_utc .+ Hour.(Int.(rows.lead))) ||
        error("target_utc does not match issue_utc + lead")
    return true
end

function _deep_broad_bias(rows::DataFrame)
    sub = rows[(rows.lead .== 6) .&
               isfinite.(Float64.(rows.rate)) .&
               (Float64.(rows.rate) .< MAIN_RATE) .&
               (Float64.(rows.obs) .< -100.0), :]
    nrow(sub) == 0 && return (n=0, v2_1=NaN, v2_0=NaN)
    return (
        n=nrow(sub),
        v2_1=mean(Float64.(sub.obs) .- Float64.(sub.v2_1)),
        v2_0=mean(Float64.(sub.obs) .- Float64.(sub.v2_0)),
    )
end

function write_broad_report(path::AbstractString, cfg::BroadReplayConfig,
                            selected::DataFrame, rows::DataFrame,
                            summary::DataFrame, skipped::DataFrame)
    db = _deep_broad_bias(rows)
    open(path, "w") do io
        println(io, "# Broad historical Operational V2.1 replay\n")
        @printf(io, "Selection: storm-catalog rows with minimum pressure-corrected Dst* <= %.1f nT, split=%s. Selected storms=%d; scored storms=%d; scored rows=%d; skipped storms=%d.\n\n",
                cfg.threshold_dst_star_nt, cfg.split, nrow(selected), length(unique(Int.(rows.storm_id))), nrow(rows), nrow(skipped))
        println(io, "This replay uses the same V2.1 point-forecast function and compares it on identical rows with the archived V2.0 product and persistence. It is a Dst*-threshold replay, not an exact NOAA Kp/G-scale replay.\n")
        println(io, "## Lead-wise RMSE\n")
        println(io, "| cohort | lead [h] | n rows | n storms | historical V2.0 [nT] | V2.1 [nT] | persistence [nT] | improvement vs stronger comparator [nT] | max tail effect [nT] | max core change [nT] |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(summary)
            startswith(r.cohort, "split=") && continue
            @printf(io, "| %s | %d | %d | %d | %.2f | %.2f | %.2f | %+.2f | %.2f | %.2f |\n",
                    r.cohort, r.lead_h, r.n_rows, r.n_storms, r.rmse_v2_0_nt,
                    r.rmse_v2_1_nt, r.rmse_persistence_nt, r.improvement_vs_best_nt,
                    r.max_tail_effect_nt, r.max_core_change_nt)
        end
        @printf(io, "\nDeep-deepening 6 h subset: n=%d, mean(obs-V2.1)=%+.2f nT, mean(obs-V2.0)=%+.2f nT.\n",
                db.n, db.v2_1, db.v2_0)
        if nrow(skipped) > 0
            println(io, "\n## Skipped storms\n")
            println(io, "| storm id | reason |")
            println(io, "|---:|---|")
            for r in eachrow(first(skipped, min(20, nrow(skipped))))
                println(io, "| ", r.storm_id, " | ", replace(r.reason, "\n" => " "), " |")
            end
            nrow(skipped) > 20 && println(io, "\nOnly the first 20 skipped storms are shown.")
        end
        println(io, "\n## CRC reflection\n")
        println(io, "Catalog selection counts are checked before replay; scored rows are checked for finite values and issue/target ordering, while the V2.1 frozen path is independently checked against the primary replay implementation. The result applies only to the Dst*-threshold event set represented by the catalog.")
    end
end

function _selftest_broad()
    cat = load_storm_catalog()
    @assert nrow(cat) == BROAD_EXPECTED_CATALOG_COUNT "unexpected storm catalog row count"
    @assert nrow(select_broad_storms(cat; threshold_dst_star_nt = -100.0)) == BROAD_EXPECTED_INTENSE_COUNT
    @assert nrow(select_broad_storms(cat; threshold_dst_star_nt = -200.0)) == BROAD_EXPECTED_SEVERE_COUNT
    @assert nrow(select_broad_storms(cat; threshold_dst_star_nt = -300.0)) == BROAD_EXPECTED_EXTREME_COUNT
    @assert nrow(select_broad_storms(cat; threshold_dst_star_nt = -100.0, split = "test")) == 23
    row684 = cat[Int.(cat.storm_id) .== 684, :][1, :]
    s684 = _storm_from_row(row684)
    @assert s684.storm_id == 684
    @assert s684.t0 == DateTime(2024, 5, 10, 18)
    @assert s684.min_dst_star <= -400.0

    toy = DataFrame(storm_id = [1, 1, 1, 1], storm = fill("toy", 4),
                    storm_split = fill("test", 4), storm_solar_cycle = fill(25, 4),
                    storm_min_dst_star_nt = fill(-120.0, 4),
                    storm_onset_utc = fill(DateTime(2024, 1, 1), 4),
                    storm_min_dst_star_utc = fill(DateTime(2024, 1, 1, 3), 4),
                    storm_recovery_end_utc = fill(DateTime(2024, 1, 2), 4),
                    issue_utc = fill(DateTime(2024, 1, 1), 4),
                    target_utc = DateTime(2024, 1, 1) .+ Hour.([1, 2, 3, 6]),
                    lead = [1, 2, 3, 6],
                    obs = [-10.0, -20.0, -30.0, -40.0],
                    v2_1 = [-11.0, -21.0, -29.0, -39.0],
                    v2_1_pre_rate_guard = [-11.0, -21.0, -29.0, -39.0],
                    v2_1_pre_one_hour_inertia = [-11.0, -21.0, -29.0, -39.0],
                    v2_1_pre_state_inertia = [-11.0, -21.0, -29.0, -39.0],
                    v2_0 = [-12.0, -22.0, -32.0, -42.0],
                    v2_1_frozen = [-11.5, -21.5, -31.5, -41.5],
                    persistence = [-8.0, -18.0, -33.0, -45.0],
                    rate = [NaN, -1.0, -2.0, -3.0])
    @assert _validate_broad_rows(toy)
    sm = broad_summary(toy; threshold_dst_star_nt = -100.0)
    one = sm[(sm.cohort .== "all") .& (sm.lead_h .== 1), :][1, :]
    @assert one.n_rows == 1 && one.n_storms == 1
    @assert isapprox(one.rmse_v2_1_nt, 1.0; atol = 1e-12)
    @assert isapprox(one.rmse_v2_0_nt, 2.0; atol = 1e-12)
    println("  ✓ broad replay self-test: catalog counts, datetime parsing, row CRC, summary RMSE")
    return true
end

function main_broad(cfg::BroadReplayConfig = _parse_broad_args(ARGS))
    _selftest_v2()          # forecast-layer oracles (continuity, regime, relaxation, inertia)
    _selftest_broad()
    cfg.self_test_only && return nothing
    catalog = load_storm_catalog()
    selected = select_broad_storms(catalog; threshold_dst_star_nt = cfg.threshold_dst_star_nt,
                                    split = cfg.split, limit = cfg.limit)
    nrow(selected) > 0 || error("no storms selected")
    println("Broad V2.1 replay: selected ", nrow(selected), " storms at minimum pressure-corrected Dst* <= ",
            cfg.threshold_dst_star_nt, " nT, split=", cfg.split)
    rows, skipped = run_broad_replay(selected)
    _validate_broad_rows(rows)
    summary = broad_summary(rows; threshold_dst_star_nt = cfg.threshold_dst_star_nt)
    CSV.write(cfg.out_csv, rows)
    CSV.write(cfg.out_summary, summary)
    write_broad_report(cfg.out_report, cfg, selected, rows, summary, skipped)
    println("  wrote ", cfg.out_csv)
    println("  wrote ", cfg.out_summary)
    println("  wrote ", cfg.out_report)
    println("  scored rows=", nrow(rows), ", storms=", length(unique(Int.(rows.storm_id))),
            ", skipped=", nrow(skipped))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_broad()
end
