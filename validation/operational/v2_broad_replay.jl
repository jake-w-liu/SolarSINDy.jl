#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "v2_replay.jl"))

const BROAD_CATALOG = OPERATIONAL_STORM_CATALOG
const BROAD_OUT_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_scored.csv")
const BROAD_OUT_SUMMARY = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_summary.csv")
const BROAD_OUT_MD = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_report.md")
const BROAD_DEFAULT_THRESHOLD = -100.0
const BROAD_EXPECTED_INTENSE_COUNT = 204
const BROAD_EXPECTED_SEVERE_COUNT = 32
const BROAD_EXPECTED_EXTREME_COUNT = 8
const BROAD_RECENT_SEVERE_IDS = Set([699, 710, 717, 719, 739])

Base.@kwdef struct BroadReplayConfig
    threshold_nt::Float64 = BROAD_DEFAULT_THRESHOLD
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
    threshold = cfg.threshold_nt
    split_name = cfg.split
    limit = cfg.limit
    out_csv = cfg.out_csv
    out_summary = cfg.out_summary
    out_report = cfg.out_report
    self_test_only = cfg.self_test_only

    for arg in args
        if arg == "--self-test"
            self_test_only = true
        elseif startswith(arg, "--threshold=")
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
        threshold_nt = threshold,
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
    required = [:storm_id, :onset_time, :min_dst, :min_dst_time, :recovery_end_time,
                :duration_hr, :solar_cycle, :split]
    missing_cols = [String(c) for c in required if !(String(c) in names(df))]
    isempty(missing_cols) || error("storm catalog missing columns: $(join(missing_cols, ", "))")
    return df
end

function select_broad_storms(catalog::DataFrame; threshold_nt::Real = BROAD_DEFAULT_THRESHOLD,
                             split::AbstractString = "all", limit::Int = 0)
    threshold = Float64(threshold_nt)
    selected = catalog[Float64.(catalog.min_dst) .<= threshold, :]
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
    tmin = _parse_broad_datetime(r.min_dst_time)
    t1 = _parse_broad_datetime(r.recovery_end_time)
    min_dst = Float64(r.min_dst)
    return (
        name = @sprintf("Catalog storm %d (%s, min Dst %.1f nT)", sid, Dates.format(t0, dateformat"yyyy-mm-dd"), min_dst),
        t0 = t0,
        t1 = t1,
        storm_id = sid,
        min_dst = min_dst,
        min_dst_time = tmin,
        duration_hr = Float64(r.duration_hr),
        solar_cycle = Int(r.solar_cycle),
        split = String(r.split),
    )
end

function _with_broad_metadata(rows::DataFrame, storm)
    out = copy(rows)
    out[!, :storm_id] = fill(storm.storm_id, nrow(out))
    out[!, :storm_min_dst_nt] = fill(storm.min_dst, nrow(out))
    out[!, :storm_split] = fill(storm.split, nrow(out))
    out[!, :storm_solar_cycle] = fill(storm.solar_cycle, nrow(out))
    out[!, :storm_onset_utc] = fill(storm.t0, nrow(out))
    out[!, :storm_min_dst_utc] = fill(storm.min_dst_time, nrow(out))
    out[!, :storm_recovery_end_utc] = fill(storm.t1, nrow(out))
    out[!, :target_utc] = out.issue_utc .+ Hour.(Int.(out.lead))
    return select(out, :storm_id, :storm, :storm_split, :storm_solar_cycle,
                  :storm_min_dst_nt, :storm_onset_utc, :storm_min_dst_utc,
                  :storm_recovery_end_utc, :issue_utc, :target_utc, :lead,
                  :obs, :audit_baseline, :v2, :v2_frozen, :persistence, :rate)
end

function run_broad_replay(selected::DataFrame)
    lib, ξ0, _ = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    lookups = Dict{Int,Any}()
    scored = DataFrame()
    skipped = DataFrame(storm_id = Int[], reason = String[])

    for r in eachrow(selected)
        storm = _storm_from_row(r)
        yr = year(storm.t1)
        if !haskey(lookups, yr)
            lookups[yr] = _driver_lookup(yr)
        end
        try
            rows = replay_v2_storm(storm, lib, ξ0, cal, lookups[yr])
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

function _metric_row(rows::DataFrame, cohort::AbstractString, lead::Int, threshold_nt::Real)
    sub = rows[rows.lead .== lead, :]
    nrow(sub) == 0 && return nothing
    rbase = _broad_rmse(sub.obs .- sub.audit_baseline)
    rv2 = _broad_rmse(sub.obs .- sub.v2)
    rpers = _broad_rmse(sub.obs .- sub.persistence)
    fair = maximum(abs.(Float64.(sub.v2_frozen) .- Float64.(sub.audit_baseline)))
    storm_count = length(unique(Int.(sub.storm_id)))
    return (
        cohort = String(cohort),
        threshold_nt = Float64(threshold_nt),
        lead_h = lead,
        n_rows = nrow(sub),
        n_storms = storm_count,
        rmse_preupgrade_nt = rbase,
        rmse_v2_nt = rv2,
        rmse_persistence_nt = rpers,
        improvement_vs_best_nt = min(rbase, rpers) - rv2,
        fair_max_abs_nt = fair,
    )
end

function broad_summary(rows::DataFrame; threshold_nt::Real = BROAD_DEFAULT_THRESHOLD)
    out = DataFrame(cohort = String[], threshold_nt = Float64[], lead_h = Int[],
                    n_rows = Int[], n_storms = Int[],
                    rmse_preupgrade_nt = Float64[], rmse_v2_nt = Float64[],
                    rmse_persistence_nt = Float64[],
                    improvement_vs_best_nt = Float64[], fair_max_abs_nt = Float64[])
    isempty(rows) && return out
    for (cohort, sub) in (("all", rows),)
        for lead in LEADS
            m = _metric_row(sub, cohort, lead, threshold_nt)
            m === nothing || push!(out, m)
        end
    end
    for split_name in sort(unique(String.(rows.storm_split)))
        sub = rows[String.(rows.storm_split) .== split_name, :]
        for lead in LEADS
            m = _metric_row(sub, "split=$(split_name)", lead, threshold_nt)
            m === nothing || push!(out, m)
        end
    end
    for sev in (-100.0, -200.0, -300.0)
        sub = rows[Float64.(rows.storm_min_dst_nt) .<= sev, :]
        isempty(sub) && continue
        for lead in LEADS
            m = _metric_row(sub, @sprintf("storm_min_dst<=%.0f", sev), lead, sev)
            m === nothing || push!(out, m)
        end
    end
    return out
end

function _validate_broad_rows(rows::DataFrame)
    isempty(rows) && error("broad replay produced no scored rows")
    required = [:storm_id, :storm, :storm_split, :storm_min_dst_nt, :issue_utc,
                :target_utc, :lead, :obs, :audit_baseline, :v2, :v2_frozen,
                :persistence, :rate]
    missing_cols = [String(c) for c in required if !(String(c) in names(rows))]
    isempty(missing_cols) || error("broad replay missing columns: $(join(missing_cols, ", "))")
    numeric = [:obs, :audit_baseline, :v2, :v2_frozen, :persistence]
    for col in numeric
        all(isfinite, Float64.(rows[!, col])) || error("non-finite values in $col")
    end
    all(rows.target_utc .== rows.issue_utc .+ Hour.(Int.(rows.lead))) ||
        error("target_utc does not match issue_utc + lead")
    fair = maximum(abs.(Float64.(rows.v2_frozen) .- Float64.(rows.audit_baseline)))
    fair <= 1e-9 || error("frozen-tail continuity failed: max gap $fair")
    return true
end

function _deep_broad_bias(rows::DataFrame)
    sub = rows[(rows.lead .== 6) .&
               isfinite.(Float64.(rows.rate)) .&
               (Float64.(rows.rate) .< MAIN_RATE) .&
               (Float64.(rows.obs) .< -100.0), :]
    nrow(sub) == 0 && return (n = 0, v2 = NaN, baseline = NaN)
    return (n = nrow(sub),
            v2 = mean(Float64.(sub.obs) .- Float64.(sub.v2)),
            baseline = mean(Float64.(sub.obs) .- Float64.(sub.audit_baseline)))
end

function write_broad_report(path::AbstractString, cfg::BroadReplayConfig,
                            selected::DataFrame, rows::DataFrame,
                            summary::DataFrame, skipped::DataFrame)
    db = _deep_broad_bias(rows)
    open(path, "w") do io
        println(io, "# Broad historical V2 replay\n")
        @printf(io, "Selection: storm-catalog rows with min Dst <= %.1f nT, split=%s. Selected storms=%d; scored storms=%d; scored rows=%d; skipped storms=%d.\n\n",
                cfg.threshold_nt, cfg.split, nrow(selected), length(unique(Int.(rows.storm_id))), nrow(rows), nrow(skipped))
        println(io, "This replay uses the same V2 point-forecast function and same-row pre-upgrade baseline/persistence scoring as the severe-storm replay. It is Dst-threshold replay, not an exact NOAA Kp/G-scale replay.\n")
        println(io, "## Lead-wise RMSE\n")
        println(io, "| cohort | lead [h] | n rows | n storms | pre-upgrade baseline [nT] | V2 [nT] | persistence [nT] | improvement vs stronger baseline [nT] | fair max [nT] |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(summary)
            startswith(r.cohort, "split=") && continue
            @printf(io, "| %s | %d | %d | %d | %.2f | %.2f | %.2f | %+.2f | %.3g |\n",
                    r.cohort, r.lead_h, r.n_rows, r.n_storms, r.rmse_preupgrade_nt,
                    r.rmse_v2_nt, r.rmse_persistence_nt, r.improvement_vs_best_nt,
                    r.fair_max_abs_nt)
        end
        @printf(io, "\nDeep-deepening 6 h subset: n=%d, mean(obs-V2)=%+.2f nT, mean(obs-pre-upgrade baseline)=%+.2f nT.\n",
                db.n, db.v2, db.baseline)
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
        println(io, "Catalog selection counts are checked before replay; scored rows are checked for finite values, issue/target ordering, and frozen-tail equality. The result strengthens the historical replay evidence only to the Dst-threshold event set represented by the catalog.")
    end
end

function _selftest_broad()
    cat = load_storm_catalog()
    @assert nrow(cat) == 741 "unexpected storm catalog row count"
    @assert nrow(select_broad_storms(cat; threshold_nt = -100.0)) == BROAD_EXPECTED_INTENSE_COUNT
    @assert nrow(select_broad_storms(cat; threshold_nt = -200.0)) == BROAD_EXPECTED_SEVERE_COUNT
    @assert nrow(select_broad_storms(cat; threshold_nt = -300.0)) == BROAD_EXPECTED_EXTREME_COUNT
    @assert nrow(select_broad_storms(cat; threshold_nt = -100.0, split = "test")) == 23
    row710 = cat[Int.(cat.storm_id) .== 710, :][1, :]
    s710 = _storm_from_row(row710)
    @assert s710.storm_id == 710
    @assert s710.t0 == DateTime(2024, 5, 10, 18)
    @assert s710.min_dst <= -400.0

    toy = DataFrame(storm_id = [1, 1, 1, 1], storm = fill("toy", 4),
                    storm_split = fill("test", 4), storm_solar_cycle = fill(25, 4),
                    storm_min_dst_nt = fill(-120.0, 4),
                    storm_onset_utc = fill(DateTime(2024, 1, 1), 4),
                    storm_min_dst_utc = fill(DateTime(2024, 1, 1, 3), 4),
                    storm_recovery_end_utc = fill(DateTime(2024, 1, 2), 4),
                    issue_utc = fill(DateTime(2024, 1, 1), 4),
                    target_utc = DateTime(2024, 1, 1) .+ Hour.([1, 2, 3, 6]),
                    lead = [1, 2, 3, 6],
                    obs = [-10.0, -20.0, -30.0, -40.0],
                    audit_baseline = [-12.0, -22.0, -32.0, -42.0],
                    v2 = [-11.0, -21.0, -29.0, -39.0],
                    v2_frozen = [-12.0, -22.0, -32.0, -42.0],
                    persistence = [-8.0, -18.0, -33.0, -45.0],
                    rate = [NaN, -1.0, -2.0, -3.0])
    @assert _validate_broad_rows(toy)
    sm = broad_summary(toy; threshold_nt = -100.0)
    one = sm[(sm.cohort .== "all") .& (sm.lead_h .== 1), :][1, :]
    @assert one.n_rows == 1 && one.n_storms == 1
    @assert isapprox(one.rmse_v2_nt, 1.0; atol = 1e-12)
    @assert isapprox(one.fair_max_abs_nt, 0.0; atol = 1e-12)
    println("  ✓ broad replay self-test: catalog counts, datetime parsing, row CRC, summary RMSE")
    return true
end

function main_broad(cfg::BroadReplayConfig = _parse_broad_args(ARGS))
    _selftest_v2()          # forecast-layer oracles (continuity, regime, relaxation, inertia)
    _selftest_broad()
    cfg.self_test_only && return nothing
    catalog = load_storm_catalog()
    selected = select_broad_storms(catalog; threshold_nt = cfg.threshold_nt,
                                   split = cfg.split, limit = cfg.limit)
    nrow(selected) > 0 || error("no storms selected")
    println("Broad V2 replay: selected ", nrow(selected), " storms at min Dst <= ",
            cfg.threshold_nt, " nT, split=", cfg.split)
    rows, skipped = run_broad_replay(selected)
    _validate_broad_rows(rows)
    summary = broad_summary(rows; threshold_nt = cfg.threshold_nt)
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
