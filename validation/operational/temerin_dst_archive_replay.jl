#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Downloads
using Printf
using Statistics

include(joinpath(@__DIR__, "paths.jl"))

const TEMERIN_BASE =
    "https://lasp.colorado.edu/space_weather/dsttemerin"
const TEMERIN_CACHE = OPERATIONAL_TEMERIN_CACHE
const TEMERIN_BROAD_REPLAY = joinpath(OPERATIONAL_EVIDENCE_DIR, "v2_broad_replay_scored.csv")
const TEMERIN_OUT_ROWS = joinpath(OPERATIONAL_OUTPUT_DIR, "temerin_dst_archive_scored.csv")
const TEMERIN_OUT_SUMMARY = joinpath(OPERATIONAL_OUTPUT_DIR, "temerin_dst_archive_summary.csv")
const TEMERIN_OUT_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "temerin_dst_archive_report.md")
const TEMERIN_DEFAULT_START = DateTime(2013, 5, 1)
const TEMERIN_DEFAULT_END = DateTime(2025, 12, 31, 23)
const TEMERIN_MATCH_TOL_MIN = 20.0
const TEMERIN_DSCOVR_START = DateTime(2016, 12, 1)

Base.@kwdef struct TemerinDstArchiveConfig
    start_utc::DateTime = TEMERIN_DEFAULT_START
    end_utc::DateTime = TEMERIN_DEFAULT_END
    cache_dir::String = TEMERIN_CACHE
    source_base::String = TEMERIN_BASE
    broad_replay::String = TEMERIN_BROAD_REPLAY
    out_rows::String = TEMERIN_OUT_ROWS
    out_summary::String = TEMERIN_OUT_SUMMARY
    out_report::String = TEMERIN_OUT_REPORT
    max_match_gap_min::Float64 = TEMERIN_MATCH_TOL_MIN
    refresh::Bool = false
    self_test_only::Bool = false
end

function _parse_temerin_args(args)::TemerinDstArchiveConfig
    cfg = TemerinDstArchiveConfig()
    vals = Dict(
        :start_utc => cfg.start_utc,
        :end_utc => cfg.end_utc,
        :cache_dir => cfg.cache_dir,
        :source_base => cfg.source_base,
        :broad_replay => cfg.broad_replay,
        :out_rows => cfg.out_rows,
        :out_summary => cfg.out_summary,
        :out_report => cfg.out_report,
        :max_match_gap_min => cfg.max_match_gap_min,
        :refresh => cfg.refresh,
        :self_test_only => cfg.self_test_only,
    )
    for arg in args
        if arg == "--refresh"
            vals[:refresh] = true
        elseif arg == "--self-test"
            vals[:self_test_only] = true
        elseif startswith(arg, "--start=")
            vals[:start_utc] = DateTime(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--end=")
            vals[:end_utc] = DateTime(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--cache-dir=")
            vals[:cache_dir] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--source-base=")
            vals[:source_base] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--broad-replay=")
            vals[:broad_replay] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-rows=")
            vals[:out_rows] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-summary=")
            vals[:out_summary] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-report=")
            vals[:out_report] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--max-match-gap-min=")
            vals[:max_match_gap_min] = parse(Float64, split(arg, "=", limit = 2)[2])
        else
            error("unknown argument: $arg")
        end
    end
    vals[:start_utc] <= vals[:end_utc] || throw(ArgumentError("--start must not exceed --end"))
    vals[:max_match_gap_min] > 0 || throw(ArgumentError("--max-match-gap-min must be positive"))
    return TemerinDstArchiveConfig(; vals...)
end

function _month_grid(start_utc::DateTime, end_utc::DateTime)
    first_month = Date(year(start_utc), month(start_utc), 1)
    last_month = Date(year(end_utc), month(end_utc), 1)
    months = Tuple{Int, Int}[]
    d = first_month
    while d <= last_month
        push!(months, (year(d), month(d)))
        d += Month(1)
    end
    return months
end

function _needed_months_for_targets(targets::AbstractVector, start_utc::DateTime,
                                    end_utc::DateTime)
    months = Set{Tuple{Int, Int}}()
    for t in targets
        dt = _parse_archive_datetime(t)
        start_utc <= dt <= end_utc || continue
        push!(months, (year(dt), month(dt)))
    end
    return sort(collect(months))
end

function _temerin_url(base::AbstractString, yr::Int, mo::Int)
    return @sprintf("%s/archive/dst_%04d_%02d.txt", base, yr, mo)
end

function _temerin_cache_path(cache_dir::AbstractString, yr::Int, mo::Int)
    return joinpath(cache_dir, @sprintf("%04d", yr), @sprintf("dst_%04d_%02d.txt", yr, mo))
end

function ensure_temerin_dst_cache(cfg::TemerinDstArchiveConfig, months::Vector{Tuple{Int, Int}})
    files = String[]
    for (yr, mo) in months
        path = _temerin_cache_path(cfg.cache_dir, yr, mo)
        if cfg.refresh || !isfile(path)
            mkpath(dirname(path))
            url = _temerin_url(cfg.source_base, yr, mo)
            try
                Downloads.download(url, path)
            catch
                isfile(path) && rm(path; force = true)
                continue
            end
        end
        if isfile(path) && filesize(path) > 0
            push!(files, path)
        end
    end
    isempty(files) && error("no Temerin-Li Dst archive files available")
    return sort(files)
end

function _parse_archive_datetime(x)
    x isa DateTime && return x
    s = replace(strip(String(x)), r"Z$" => "", r"\.\d+$" => "")
    return DateTime(s)
end

function _parse_temerin_time(yr::Int, doy::Int, hh::Int, mm::Int, ss::Int)
    doy >= 1 || throw(ArgumentError("day-of-year must be >= 1"))
    return DateTime(Date(yr, 1, 1) + Day(doy - 1)) + Hour(hh) + Minute(mm) + Second(ss)
end

function parse_temerin_dst_text(text::AbstractString; source::AbstractString = "")
    rows = DataFrame(valid_utc = DateTime[], temerin_li_dst_nt = Float64[],
                     source_file = String[])
    for line in split(String(text), '\n')
        m = match(r"^\s*(\d{4})/(\d{3})-(\d{2}):(\d{2}):(\d{2})\s+(-?\d+(?:\.\d+)?)\s*$",
                  line)
        m === nothing && continue
        yr = parse(Int, m.captures[1])
        doy = parse(Int, m.captures[2])
        hh = parse(Int, m.captures[3])
        mm = parse(Int, m.captures[4])
        ss = parse(Int, m.captures[5])
        val = parse(Float64, m.captures[6])
        push!(rows, (_parse_temerin_time(yr, doy, hh, mm, ss), val, String(source)))
    end
    isempty(rows) && error("Temerin-Li Dst text produced no data rows")
    sort!(rows, :valid_utc)
    all(isfinite, Float64.(rows.temerin_li_dst_nt)) ||
        error("Temerin-Li Dst text contains non-finite Dst values")
    return rows
end

function load_temerin_dst(files::Vector{String}; start_utc::DateTime, end_utc::DateTime)
    out = DataFrame()
    for path in files
        try
            rows = parse_temerin_dst_text(read(path, String); source = basename(path))
            append!(out, rows; cols = :union)
        catch err
            @warn "skipping unparsable Temerin-Li Dst archive file" path exception = sprint(showerror, err)
        end
    end
    isempty(out) && error("no parsable Temerin-Li Dst rows")
    margin = Minute(ceil(Int, TEMERIN_MATCH_TOL_MIN))
    out = out[(out.valid_utc .>= start_utc - margin) .& (out.valid_utc .<= end_utc + margin), :]
    sort!(out, :valid_utc)
    unique!(out, :valid_utc)
    return out
end

function _nearest_temerin_row(valids::Vector{DateTime}, target::DateTime)
    idx = searchsortedfirst(valids, target)
    best_idx = 0
    best_gap_ms = typemax(Int64)
    for j in (idx - 1):(idx + 1)
        1 <= j <= length(valids) || continue
        gap = abs(Dates.value(valids[j] - target))
        if gap < best_gap_ms
            best_gap_ms = gap
            best_idx = j
        end
    end
    return best_idx, best_gap_ms / 60_000.0
end

function _source_epoch(target::DateTime)
    target >= TEMERIN_DSCOVR_START && return "DSCOVR real-time input era"
    target >= TEMERIN_DEFAULT_START && return "ACE real-time input era"
    return "pre-real-time-input archive era"
end

_csv_audit_safe_label(x) = replace(String(x), "," => ";")

function score_temerin_dst_archive(broad::DataFrame, temerin::DataFrame;
                                   start_utc::DateTime = TEMERIN_DEFAULT_START,
                                   end_utc::DateTime = TEMERIN_DEFAULT_END,
                                   max_match_gap_min::Real = TEMERIN_MATCH_TOL_MIN)
    isempty(broad) && error("broad replay has no rows")
    isempty(temerin) && error("Temerin-Li archive has no rows")
    required = [:storm_id, :storm, :storm_min_dst_nt, :issue_utc, :target_utc, :lead,
                :obs, :audit_baseline, :v2, :persistence]
    missing_cols = [String(c) for c in required if !(String(c) in names(broad))]
    isempty(missing_cols) || error("broad replay missing columns: $(join(missing_cols, ", "))")

    sort!(temerin, :valid_utc)
    valids = DateTime.(temerin.valid_utc)
    out = DataFrame(storm_id = Int[], storm = String[], storm_min_dst_nt = Float64[],
                    issue_utc = DateTime[], target_utc = DateTime[], lead = Int[],
                    obs = Float64[], audit_baseline = Float64[], v2 = Float64[],
                    persistence = Float64[], temerin_li_dst = Float64[],
                    temerin_valid_utc = DateTime[], match_abs_gap_min = Float64[],
                    temerin_source_file = String[], source_epoch = String[])
    for r in eachrow(broad)
        target = _parse_archive_datetime(r.target_utc)
        start_utc <= target <= end_utc || continue
        idx, gap_min = _nearest_temerin_row(valids, target)
        idx > 0 || continue
        gap_min <= Float64(max_match_gap_min) || continue
        issue = _parse_archive_datetime(r.issue_utc)
        push!(out, (
            Int(r.storm_id),
            _csv_audit_safe_label(r.storm),
            Float64(r.storm_min_dst_nt),
            issue,
            target,
            Int(r.lead),
            Float64(r.obs),
            Float64(r.audit_baseline),
            Float64(r.v2),
            Float64(r.persistence),
            Float64(temerin.temerin_li_dst_nt[idx]),
            valids[idx],
            gap_min,
            String(temerin.source_file[idx]),
            _source_epoch(target),
        ))
    end
    _validate_temerin_rows(out; max_match_gap_min = Float64(max_match_gap_min))
    return sort(out, [:issue_utc, :target_utc, :lead])
end

function _validate_temerin_rows(rows::DataFrame; max_match_gap_min::Float64 = TEMERIN_MATCH_TOL_MIN)
    isempty(rows) && error("Temerin-Li Dst archive scoring produced no rows")
    required = [:storm_id, :storm, :storm_min_dst_nt, :issue_utc, :target_utc, :lead,
                :obs, :audit_baseline, :v2, :persistence, :temerin_li_dst,
                :temerin_valid_utc, :match_abs_gap_min, :temerin_source_file,
                :source_epoch]
    missing_cols = [String(c) for c in required if !(String(c) in names(rows))]
    isempty(missing_cols) || error("Temerin-Li scoring missing columns: $(join(missing_cols, ", "))")
    for col in (:obs, :audit_baseline, :v2, :persistence, :temerin_li_dst,
                :match_abs_gap_min)
        all(isfinite, Float64.(rows[!, col])) || error("non-finite values in $col")
    end
    all(Int.(rows.lead) .> 0) || error("non-positive lead in Temerin-Li scoring")
    all(rows.target_utc .== rows.issue_utc .+ Hour.(Int.(rows.lead))) ||
        error("target_utc does not match issue_utc + lead")
    maximum(Float64.(rows.match_abs_gap_min)) <= max_match_gap_min + 1e-9 ||
        error("Temerin-Li nearest-valid-time gap exceeds tolerance")
    return true
end

_dst_rmse(residuals) = sqrt(mean(abs2, Float64.(residuals)))

function _temerin_metric_row(rows::DataFrame, scope::AbstractString, lead::Int)
    sub = rows[Int.(rows.lead) .== lead, :]
    nrow(sub) == 0 && return nothing
    rtem = _dst_rmse(sub.obs .- sub.temerin_li_dst)
    rv2 = _dst_rmse(sub.obs .- sub.v2)
    rbase = _dst_rmse(sub.obs .- sub.audit_baseline)
    rpers = _dst_rmse(sub.obs .- sub.persistence)
    return (
        scope = String(scope),
        lead_h = lead,
        n_rows = nrow(sub),
        n_storms = length(unique(Int.(sub.storm_id))),
        rmse_temerin_valid_nt = rtem,
        rmse_v2_nt = rv2,
        rmse_preupgrade_nt = rbase,
        rmse_persistence_nt = rpers,
        v2_minus_temerin_valid_rmse_nt = rv2 - rtem,
        median_match_gap_min = median(Float64.(sub.match_abs_gap_min)),
        max_match_gap_min = maximum(Float64.(sub.match_abs_gap_min)),
    )
end

function temerin_dst_summary(rows::DataFrame)
    out = DataFrame(scope = String[], lead_h = Int[], n_rows = Int[], n_storms = Int[],
                    rmse_temerin_valid_nt = Float64[], rmse_v2_nt = Float64[],
                    rmse_preupgrade_nt = Float64[], rmse_persistence_nt = Float64[],
                    v2_minus_temerin_valid_rmse_nt = Float64[],
                    median_match_gap_min = Float64[], max_match_gap_min = Float64[])
    for (scope, sub) in (
        ("all_operational_input_era", rows),
        ("dscovr_real_time_input_era",
         rows[rows.target_utc .>= TEMERIN_DSCOVR_START, :]),
    )
        isempty(sub) && continue
        for lead in sort(unique(Int.(sub.lead)))
            m = _temerin_metric_row(sub, scope, lead)
            m === nothing || push!(out, m)
        end
    end
    return out
end

function _temerin_summary_consistent(rows::DataFrame, summary::DataFrame)
    max_error = 0.0
    for r in eachrow(summary)
        sub = String(r.scope) == "dscovr_real_time_input_era" ?
              rows[rows.target_utc .>= TEMERIN_DSCOVR_START, :] : rows
        sub = sub[Int.(sub.lead) .== Int(r.lead_h), :]
        nrow(sub) == Int(r.n_rows) || return (ok = false,
            detail = "row-count mismatch for $(r.scope) $(r.lead_h)h")
        rtem = _dst_rmse(sub.obs .- sub.temerin_li_dst)
        rv2 = _dst_rmse(sub.obs .- sub.v2)
        rbase = _dst_rmse(sub.obs .- sub.audit_baseline)
        rpers = _dst_rmse(sub.obs .- sub.persistence)
        vals = [
            abs(rtem - Float64(r.rmse_temerin_valid_nt)),
            abs(rv2 - Float64(r.rmse_v2_nt)),
            abs(rbase - Float64(r.rmse_preupgrade_nt)),
            abs(rpers - Float64(r.rmse_persistence_nt)),
            abs((rv2 - rtem) - Float64(r.v2_minus_temerin_valid_rmse_nt)),
        ]
        max_error = max(max_error, maximum(vals))
    end
    return (ok = max_error <= 1e-9, detail = @sprintf("max summary recompute error %.3g", max_error))
end

function _write_temerin_report(path::AbstractString, scored::DataFrame, summary::DataFrame,
                               cfg::TemerinDstArchiveConfig)
    all_rows = summary[summary.scope .== "all_operational_input_era", :]
    open(path, "w") do io
        println(io, "# Temerin-Li Dst external archive comparison\n")
        println(io, "Source: LASP/Colorado Temerin-Li Dst archive monthly ASCII predicted Dst.")
        println(io, "Scoring range: ", cfg.start_utc, " to ", cfg.end_utc,
                "; scored rows=", nrow(scored),
                "; scored storms=", length(unique(Int.(scored.storm_id))), ".")
        println(io, "Alignment: each V2 target hour is matched to the nearest archived predicted-Dst valid time within ",
                cfg.max_match_gap_min, " minutes.")
        println(io, "Interpretation boundary: the archive exposes valid-time Dst predictions, not issue-time/lead rows. This is a same-unit external operational-valid-time context check, not a matched 1--6 h promotion baseline.\n")
        println(io, "| Scope | Lead [h] | Rows | Storms | Temerin-Li valid-time RMSE [nT] | V2 RMSE [nT] | Pre-upgrade [nT] | Persistence [nT] | V2 minus Temerin [nT] | Max gap [min] |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(summary)
            @printf(io, "| %s | %d | %d | %d | %.2f | %.2f | %.2f | %.2f | %+.2f | %.2f |\n",
                    r.scope, r.lead_h, r.n_rows, r.n_storms,
                    r.rmse_temerin_valid_nt, r.rmse_v2_nt,
                    r.rmse_preupgrade_nt, r.rmse_persistence_nt,
                    r.v2_minus_temerin_valid_rmse_nt, r.max_match_gap_min)
        end
        if nrow(all_rows) > 0
            worst = all_rows[argmax(Float64.(all_rows.v2_minus_temerin_valid_rmse_nt)), :]
            @printf(io, "\nLargest V2-minus-valid-time-context gap in the operational-input era is %.2f nT at %d h.\n",
                    worst.v2_minus_temerin_valid_rmse_nt, worst.lead_h)
        end
    end
end

function _sample_temerin_text()
    return """      Time          Predicted Dst
2024/122-00:01:20      -41.1
2024/122-00:11:26      -43.0
2024/122-01:02:06      -49.1
2024/123-00:01:20      -50.0
"""
end

function _selftest_temerin_dst()
    rows = parse_temerin_dst_text(_sample_temerin_text(); source = "sample")
    @assert nrow(rows) == 4
    @assert rows.valid_utc[1] == DateTime(2024, 5, 1, 0, 1, 20)
    @assert rows.valid_utc[end] == DateTime(2024, 5, 2, 0, 1, 20)
    broad = DataFrame(
        storm_id = [1, 1],
        storm = ["synthetic", "synthetic"],
        storm_min_dst_nt = [-120.0, -120.0],
        issue_utc = [DateTime(2024, 4, 30, 23), DateTime(2024, 5, 1, 0)],
        target_utc = [DateTime(2024, 5, 1, 0), DateTime(2024, 5, 1, 1)],
        lead = [1, 1],
        obs = [-40.0, -50.0],
        audit_baseline = [-39.0, -48.0],
        v2 = [-40.5, -49.0],
        persistence = [-38.0, -45.0],
    )
    scored = score_temerin_dst_archive(broad, rows; start_utc = DateTime(2024, 5, 1),
                                       end_utc = DateTime(2024, 5, 2),
                                       max_match_gap_min = 5.0)
    @assert nrow(scored) == 2
    @assert maximum(scored.match_abs_gap_min) < 3.0
    summary = temerin_dst_summary(scored)
    @assert nrow(summary) == 2
    crc = _temerin_summary_consistent(scored, summary)
    @assert crc.ok
    broken = copy(scored)
    broken.match_abs_gap_min[1] = 99.0
    try
        _validate_temerin_rows(broken; max_match_gap_min = 5.0)
        error("expected broken Temerin-Li match gap to fail")
    catch err
        @assert err isa ErrorException
    end
    println("  ✓ Temerin-Li Dst archive self-test: parser, nearest-time match, summary CRC")
    return true
end

function main_temerin_dst(args = ARGS)
    cfg = _parse_temerin_args(args)
    _selftest_temerin_dst()
    cfg.self_test_only && return nothing
    broad = CSV.read(cfg.broad_replay, DataFrame)
    targets = _parse_archive_datetime.(broad.target_utc)
    months = _needed_months_for_targets(targets, cfg.start_utc, cfg.end_utc)
    isempty(months) && (months = _month_grid(cfg.start_utc, cfg.end_utc))
    files = ensure_temerin_dst_cache(cfg, months)
    temerin = load_temerin_dst(files; start_utc = cfg.start_utc, end_utc = cfg.end_utc)
    scored = score_temerin_dst_archive(broad, temerin;
                                       start_utc = cfg.start_utc,
                                       end_utc = cfg.end_utc,
                                       max_match_gap_min = cfg.max_match_gap_min)
    summary = temerin_dst_summary(scored)
    crc = _temerin_summary_consistent(scored, summary)
    crc.ok || error("Temerin-Li summary CRC failed: $(crc.detail)")
    mkpath(dirname(cfg.out_rows))
    mkpath(dirname(cfg.out_summary))
    mkpath(dirname(cfg.out_report))
    CSV.write(cfg.out_rows, scored)
    CSV.write(cfg.out_summary, summary)
    _write_temerin_report(cfg.out_report, scored, summary, cfg)
    println("Temerin-Li Dst archive comparison: scored rows=", nrow(scored),
            ", storms=", length(unique(scored.storm_id)))
    println("  wrote ", cfg.out_rows)
    println("  wrote ", cfg.out_summary)
    println("  wrote ", cfg.out_report)
    return (; scored, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_temerin_dst()
end
