#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Downloads
using Printf
using Statistics

include(joinpath(@__DIR__, "v2_gscale_replay.jl"))

const NOAA_3DAY_BASE =
    "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast"
const NOAA_KP_CACHE = OPERATIONAL_NOAA_KP_CACHE
const NOAA_KP_OUT_ROWS = joinpath(OPERATIONAL_OUTPUT_DIR, "noaa_kp_forecast_scored.csv")
const NOAA_KP_OUT_SUMMARY = joinpath(OPERATIONAL_OUTPUT_DIR, "noaa_kp_forecast_summary.csv")
const NOAA_KP_OUT_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "noaa_kp_forecast_report.md")
const NOAA_KP_START_UTC = DateTime(2023, 1, 1)
const NOAA_KP_END_UTC = DateTime(2025, 12, 31, 21)

const NOAA_MONTHS = Dict(
    "Jan" => 1, "Feb" => 2, "Mar" => 3, "Apr" => 4,
    "May" => 5, "Jun" => 6, "Jul" => 7, "Aug" => 8,
    "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dec" => 12,
)

Base.@kwdef struct NoaaKpReplayConfig
    start_utc::DateTime = NOAA_KP_START_UTC
    end_utc::DateTime = NOAA_KP_END_UTC
    cache_dir::String = NOAA_KP_CACHE
    source_base::String = NOAA_3DAY_BASE
    kp_source::String = GFZ_KP_SOURCE
    out_rows::String = NOAA_KP_OUT_ROWS
    out_summary::String = NOAA_KP_OUT_SUMMARY
    out_report::String = NOAA_KP_OUT_REPORT
    refresh::Bool = false
    self_test_only::Bool = false
end

function _parse_noaa_args(args)::NoaaKpReplayConfig
    cfg = NoaaKpReplayConfig()
    vals = Dict(
        :start_utc => cfg.start_utc,
        :end_utc => cfg.end_utc,
        :cache_dir => cfg.cache_dir,
        :source_base => cfg.source_base,
        :kp_source => cfg.kp_source,
        :out_rows => cfg.out_rows,
        :out_summary => cfg.out_summary,
        :out_report => cfg.out_report,
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
        elseif startswith(arg, "--kp-source=")
            vals[:kp_source] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-rows=")
            vals[:out_rows] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-summary=")
            vals[:out_summary] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-report=")
            vals[:out_report] = split(arg, "=", limit = 2)[2]
        else
            error("unknown argument: $arg")
        end
    end
    vals[:start_utc] <= vals[:end_utc] || throw(ArgumentError("--start must not exceed --end"))
    return NoaaKpReplayConfig(; vals...)
end

function _ceil_next_kp_bin(t::DateTime)
    minutes = hour(t) * 60 + minute(t) + second(t) / 60
    next_h = 3 * (floor(Int, minutes / 180) + 1)
    return DateTime(Date(t)) + Hour(Int(next_h))
end

function _noaa_forecast_url(base::AbstractString, issue::DateTime)
    return @sprintf("%s/%04d/%02d/%04d%02d%02d%02d%02dthree_day_forecast.txt",
                    base, year(issue), month(issue), year(issue), month(issue), day(issue),
                    hour(issue), minute(issue))
end

function _noaa_cache_path(cache_dir::AbstractString, issue::DateTime)
    return joinpath(cache_dir, @sprintf("%04d", year(issue)), @sprintf("%02d", month(issue)),
                    @sprintf("%04d%02d%02d%02d%02dthree_day_forecast.txt",
                             year(issue), month(issue), day(issue), hour(issue), minute(issue)))
end

function _issue_grid(start_utc::DateTime, end_utc::DateTime)
    start_date = Date(start_utc)
    end_date = Date(end_utc)
    out = DateTime[]
    for d in start_date:Day(1):end_date
        push!(out, DateTime(d) + Minute(30))
        push!(out, DateTime(d) + Hour(12) + Minute(30))
    end
    return out
end

function ensure_noaa_3day_cache(cfg::NoaaKpReplayConfig)
    win_start = cfg.start_utc - Day(3)
    grid = _issue_grid(win_start, cfg.end_utc)
    n_downloaded = 0
    n_404 = 0
    n_fail = 0
    # Always complete the cache: attempt a download for every grid slot not already
    # present, instead of freezing a non-empty partial cache. Failures are logged
    # (404 = upstream archive gap vs transport error), never silently swallowed.
    for issue in grid
        path = _noaa_cache_path(cfg.cache_dir, issue)
        if !cfg.refresh && isfile(path) && filesize(path) > 0
            continue
        end
        mkpath(dirname(path))
        url = _noaa_forecast_url(cfg.source_base, issue)
        try
            Downloads.download(url, path)
            n_downloaded += 1
        catch err
            isfile(path) && rm(path; force = true)
            status = 0
            if err isa Downloads.RequestError
                try
                    status = err.response.status
                catch
                end
            end
            if status == 404
                n_404 += 1                       # upstream archive gap (issue never published/archived)
            else
                n_fail += 1
                @warn "NOAA 3-day forecast download failed" url exception = err
            end
            continue
        end
    end
    # Merge every file present after completion, including off-grid archive names
    # (e.g. a 1203 UTC issue replacing the 1230 slot) recovered from the directory.
    files = _cached_noaa_files(cfg.cache_dir; start_utc = win_start, end_utc = cfg.end_utc)
    @info "NOAA 3-day cache coverage" grid_slots = length(grid) files_found = length(files) downloaded = n_downloaded http_404 = n_404 failures = n_fail
    isempty(files) && error("no NOAA 3-day forecast files available in cache")
    return sort(files)
end

function _issue_from_noaa_filename(path::AbstractString)
    m = match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})three_day_forecast\.txt$", basename(path))
    m === nothing && return nothing
    return DateTime(parse(Int, m.captures[1]), parse(Int, m.captures[2]),
                    parse(Int, m.captures[3]), parse(Int, m.captures[4]),
                    parse(Int, m.captures[5]))
end

function _cached_noaa_files(cache_dir::AbstractString; start_utc::DateTime, end_utc::DateTime)
    out = String[]
    for (root, _, files) in walkdir(cache_dir)
        for f in files
            endswith(f, "three_day_forecast.txt") || continue
            path = joinpath(root, f)
            issue = _issue_from_noaa_filename(path)
            issue === nothing && continue
            start_utc <= issue <= end_utc || continue
            filesize(path) > 0 || continue
            push!(out, path)
        end
    end
    return sort(out)
end

function _parse_issue(line::AbstractString)
    m = match(r"^:?Issued:\s+(\d{4})\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})\s+UTC", line)
    m === nothing && return nothing
    yr = parse(Int, m.captures[1])
    mo = NOAA_MONTHS[m.captures[2]]
    dy = parse(Int, m.captures[3])
    hm = m.captures[4]
    return DateTime(yr, mo, dy, parse(Int, hm[1:2]), parse(Int, hm[3:4]))
end

function _forecast_dates_from_header(line::AbstractString, issue::DateTime)
    dates = Date[]
    for m in eachmatch(r"([A-Z][a-z]{2})\s+(\d{1,2})", line)
        mo = NOAA_MONTHS[m.captures[1]]
        dy = parse(Int, m.captures[2])
        yr = year(issue)
        candidate = Date(yr, mo, dy)
        if candidate < Date(issue) - Day(7)
            candidate = Date(yr + 1, mo, dy)
        end
        push!(dates, candidate)
    end
    return dates
end

function parse_noaa_3day_kp_text(text::AbstractString; source::AbstractString = "")
    lines = split(String(text), '\n')
    issue = nothing
    for line in lines
        parsed = _parse_issue(strip(line))
        if parsed !== nothing
            issue = parsed
            break
        end
    end
    issue === nothing && error("NOAA 3-day forecast has no :Issued: line")

    block_idx = findfirst(i -> occursin("NOAA Kp index breakdown", lines[i]), eachindex(lines))
    block_idx === nothing && error("NOAA 3-day forecast has no Kp breakdown block")
    header_range = block_idx:min(block_idx + 8, lastindex(lines))
    header_pos = findfirst(i -> length(_forecast_dates_from_header(lines[i], issue)) >= 3,
                           header_range)
    header_pos === nothing && error("NOAA 3-day forecast has no Kp date header")
    header_idx = header_range[header_pos]
    forecast_dates = _forecast_dates_from_header(lines[header_idx], issue)

    rows = DataFrame(issue_utc = DateTime[], target_bin_start_utc = DateTime[],
                     target_bin_end_utc = DateTime[], lead_h = Float64[],
                     forecast_day = Int[], forecast_kp = Float64[],
                     forecast_g_level = Int[], source_file = String[])
    earliest_forecast_bin = _ceil_next_kp_bin(issue)
    for i in (header_idx + 1):min(header_idx + 12, lastindex(lines))
        line = lines[i]
        m = match(r"^\s*(\d{2})-(\d{2})UT\s+(.+)$", line)
        m === nothing && continue
        start_hour = parse(Int, m.captures[1])
        cell_text = replace(m.captures[3], r"\([^)]*\)" => "")
        vals = [parse(Float64, mm.match) for mm in eachmatch(r"\d+(?:\.\d+)?", cell_text)]
        length(vals) >= length(forecast_dates) || continue
        for (j, date) in enumerate(forecast_dates)
            target_start = DateTime(date) + Hour(start_hour)
            target_end = target_start + Hour(3)
            target_start < earliest_forecast_bin && continue
            push!(rows, (issue, target_start, target_end,
                         Dates.value(target_start - issue) / 3_600_000,
                         j, vals[j], noaa_g_level(vals[j]), String(source)))
        end
    end
    isempty(rows) && error("NOAA 3-day forecast produced no future Kp rows")
    return rows
end

function load_noaa_3day_kp(files::Vector{String}; start_utc::DateTime, end_utc::DateTime)
    out = DataFrame()
    for path in files
        try
            rows = parse_noaa_3day_kp_text(read(path, String); source = basename(path))
            append!(out, rows; cols = :union)
        catch err
            @warn "skipping unparsable NOAA 3-day forecast" path exception = sprint(showerror, err)
        end
    end
    isempty(out) && error("no parsable NOAA 3-day Kp forecast rows")
    out = out[(out.target_bin_start_utc .>= start_utc) .&
              (out.target_bin_start_utc .<= end_utc), :]
    sort!(out, [:issue_utc, :target_bin_start_utc])
    return out
end

function score_noaa_kp_forecasts(forecasts::DataFrame, kp_rows::DataFrame)
    obs = select(kp_rows, :utc => :target_bin_start_utc, :kp => :observed_kp)
    joined = innerjoin(forecasts, obs; on = :target_bin_start_utc)
    joined[!, :observed_g_level] = noaa_g_level.(Float64.(joined.observed_kp))
    joined[!, :kp_error] = Float64.(joined.forecast_kp) .- Float64.(joined.observed_kp)
    joined[!, :lead_band] = [_lead_band(x) for x in Float64.(joined.lead_h)]
    sort!(joined, [:issue_utc, :target_bin_start_utc])
    _validate_noaa_rows(joined)
    return joined
end

function _lead_band(lead_h::Real)
    x = Float64(lead_h)
    x < 12 && return "00-12h"
    x < 24 && return "12-24h"
    x < 48 && return "24-48h"
    return "48-72h"
end

function _validate_noaa_rows(rows::DataFrame)
    isempty(rows) && error("NOAA Kp forecast scoring produced no rows")
    required = [:issue_utc, :target_bin_start_utc, :target_bin_end_utc, :lead_h,
                :forecast_day, :forecast_kp, :observed_kp, :forecast_g_level,
                :observed_g_level, :kp_error, :lead_band]
    missing_cols = [String(c) for c in required if !(String(c) in names(rows))]
    isempty(missing_cols) || error("NOAA Kp scoring missing columns: $(join(missing_cols, ", "))")
    all(rows.target_bin_start_utc .< rows.target_bin_end_utc) || error("invalid Kp bin geometry")
    all(rows.target_bin_start_utc .>= _ceil_next_kp_bin.(rows.issue_utc)) ||
        error("NOAA scoring includes a target bin already underway at issue time")
    for col in (:lead_h, :forecast_kp, :observed_kp, :kp_error)
        all(isfinite, Float64.(rows[!, col])) || error("non-finite values in $col")
    end
    all((Float64.(rows.forecast_kp) .>= 0.0) .& (Float64.(rows.forecast_kp) .<= 9.0)) ||
        error("forecast Kp outside [0, 9]")
    all((Float64.(rows.observed_kp) .>= 0.0) .& (Float64.(rows.observed_kp) .<= 9.0)) ||
        error("observed Kp outside [0, 9]")
    return true
end

_safe_div(a, b) = b == 0 ? NaN : Float64(a) / Float64(b)
_kp_rmse(x) = sqrt(mean(abs2, Float64.(x)))

function _event_metric_row(rows::DataFrame, scope::AbstractString, lead_band::AbstractString,
                           threshold_g::Int)
    obs = Int.(rows.observed_g_level) .>= threshold_g
    pred = Int.(rows.forecast_g_level) .>= threshold_g
    hits = count(obs .& pred)
    misses = count(obs .& .!pred)
    false_alarms = count(.!obs .& pred)
    correct_negatives = count(.!obs .& .!pred)
    n = length(obs)
    return (
        scope = String(scope),
        lead_band = String(lead_band),
        threshold_g = threshold_g,
        n_rows = n,
        hits = hits,
        misses = misses,
        false_alarms = false_alarms,
        correct_negatives = correct_negatives,
        observed_events = hits + misses,
        forecast_events = hits + false_alarms,
        pod = _safe_div(hits, hits + misses),
        precision = _safe_div(hits, hits + false_alarms),
        far = _safe_div(false_alarms, hits + false_alarms),
        csi = _safe_div(hits, hits + misses + false_alarms),
        bias = _safe_div(hits + false_alarms, hits + misses),
        rmse_kp = _kp_rmse(rows.kp_error),
        mae_kp = mean(abs.(Float64.(rows.kp_error))),
    )
end

function noaa_kp_summary(rows::DataFrame)
    out = DataFrame(scope = String[], lead_band = String[], threshold_g = Int[],
                    n_rows = Int[], hits = Int[], misses = Int[],
                    false_alarms = Int[], correct_negatives = Int[],
                    observed_events = Int[], forecast_events = Int[],
                    pod = Float64[], precision = Float64[], far = Float64[],
                    csi = Float64[], bias = Float64[], rmse_kp = Float64[],
                    mae_kp = Float64[])
    for threshold_g in 1:5
        push!(out, _event_metric_row(rows, "all", "all", threshold_g))
        for band in ["00-12h", "12-24h", "24-48h", "48-72h"]
            sub = rows[rows.lead_band .== band, :]
            isempty(sub) || push!(out, _event_metric_row(sub, "all", band, threshold_g))
        end
    end
    return out
end

function _write_noaa_report(path::AbstractString, scored::DataFrame, summary::DataFrame,
                            cfg::NoaaKpReplayConfig)
    g3 = summary[(summary.scope .== "all") .& (summary.lead_band .== "all") .&
                 (summary.threshold_g .== 3), :][1, :]
    open(path, "w") do io
        println(io, "# NOAA 3-day Kp operational archive replay\n")
        println(io, "Source: NOAA SWPC 3-day forecast archive via NCEI; forecast rows are official Kp/G-scale forecasts, not Dst forecasts.")
        println(io, "Scoring range: ", cfg.start_utc, " to ", cfg.end_utc,
                "; scored forecast rows=", nrow(scored), ".")
        println(io, "Causality rule: target 3-hour Kp bin must start strictly after the issue time.")
        println(io, "Observed label: GFZ three-hour Kp at the target-bin start.\n")
        @printf(io, "Overall G3+ event skill: hits=%d, misses=%d, false alarms=%d, POD=%.3f, FAR=%.3f, CSI=%.3f, Kp RMSE=%.3f.\n\n",
                g3.hits, g3.misses, g3.false_alarms, g3.pod, g3.far, g3.csi, g3.rmse_kp)
        println(io, "| scope | lead band | threshold | rows | hits | misses | false alarms | POD | FAR | CSI | bias | Kp RMSE |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in eachrow(summary)
            r.threshold_g in (3, 4, 5) || continue
            @printf(io, "| %s | %s | G%d+ | %d | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
                    r.scope, r.lead_band, r.threshold_g, r.n_rows, r.hits, r.misses,
                    r.false_alarms, r.pod, r.far, r.csi, r.bias, r.rmse_kp)
        end
        println(io, "\nInterpretation boundary: this is an external operational Kp/G-scale forecast audit. It is not a same-unit Dst RMSE baseline for V2.")
    end
end

function _sample_noaa_3day_text()
    return """:Product: 05101230three_day_forecast.txt
:Issued: 2024 May 10 1230 UTC
# Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center
#
A. NOAA Geomagnetic Activity Observation and Forecast

NOAA Kp index breakdown May 10-May 12 2024

             May 10       May 11       May 12
00-03UT       3.00         5.33 (G1)    4.67 (G1)
03-06UT       3.00         7.00 (G3)    5.67 (G2)
06-09UT       2.67         8.33 (G4)    4.67 (G1)
09-12UT       2.33         6.67 (G3)    4.00
12-15UT       2.67         5.67 (G2)    3.67
15-18UT       3.67         5.00 (G1)    2.67
18-21UT       4.33         4.67 (G1)    2.67
21-00UT       5.33 (G1)    4.33         3.67
"""
end

function _selftest_noaa_kp()
    rows = parse_noaa_3day_kp_text(_sample_noaa_3day_text(); source = "sample")
    @assert nrow(rows) == 19
    @assert minimum(rows.target_bin_start_utc) == DateTime(2024, 5, 10, 15)
    @assert maximum(rows.target_bin_start_utc) == DateTime(2024, 5, 12, 21)
    @assert rows.forecast_kp[rows.target_bin_start_utc .== DateTime(2024, 5, 11, 6)][1] == 8.33
    @assert rows.forecast_g_level[rows.target_bin_start_utc .== DateTime(2024, 5, 11, 6)][1] == 4
    observed = DataFrame(utc = rows.target_bin_start_utc,
                         kp = [i <= 5 ? 7.0 : 2.0 for i in 1:nrow(rows)])
    scored = score_noaa_kp_forecasts(rows, observed)
    sm = noaa_kp_summary(scored)
    g3 = sm[(sm.scope .== "all") .& (sm.lead_band .== "all") .&
            (sm.threshold_g .== 3), :][1, :]
    @assert g3.hits == 2
    @assert g3.misses == 3
    @assert g3.false_alarms == 0
    @assert isapprox(g3.pod, 0.4; atol = 1e-12)
    @assert _validate_noaa_rows(scored)
    println("  ✓ NOAA Kp archive self-test: parser, causality filter, G-threshold CRC")
    return true
end

function main_noaa_kp(args = ARGS)
    cfg = _parse_noaa_args(args)
    _selftest_noaa_kp()
    cfg.self_test_only && return nothing
    files = ensure_noaa_3day_cache(cfg)
    forecasts = load_noaa_3day_kp(files; start_utc = cfg.start_utc, end_utc = cfg.end_utc)
    kp = load_gfz_kp(cfg.kp_source)
    scored = score_noaa_kp_forecasts(forecasts, kp)
    summary = noaa_kp_summary(scored)
    mkpath(dirname(cfg.out_rows))
    mkpath(dirname(cfg.out_summary))
    mkpath(dirname(cfg.out_report))
    CSV.write(cfg.out_rows, scored)
    CSV.write(cfg.out_summary, summary)
    _write_noaa_report(cfg.out_report, scored, summary, cfg)
    println("NOAA Kp archive replay: scored rows=", nrow(scored),
            ", issues=", length(unique(scored.issue_utc)))
    println("  wrote ", cfg.out_rows)
    println("  wrote ", cfg.out_summary)
    println("  wrote ", cfg.out_report)
    return (; scored, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_noaa_kp()
end
