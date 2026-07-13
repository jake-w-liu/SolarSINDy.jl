#!/usr/bin/env julia
# Prospective external Dst forecast snapshot collector.
#
# Captures public same-unit Dst forecast/nowcast products as issue-time snapshots, stores raw
# response hashes, and scores rows later against the SWPC-served Kyoto Dst product. Driven by the
# live monitor, and runnable standalone.
#
# Output locations are parameterized so the collector writes into the monitor directory. The
# defaults resolve against the working directory (default "live_forecasts"), which reproduces the
# deployed layout when run from the project root.
#
# Env:
#   SOLARSINDY_EXTERNAL_DST_DIR  output directory for the external Dst log/report/raw snapshots
#                                (default SOLARSINDY_MONITOR_DIR, else "live_forecasts")

using CSV
using DataFrames
using Dates
using HTTP
using JSON3
using Printf
using SHA
using Statistics

const EXTERNAL_DST_DIR = get(ENV, "SOLARSINDY_EXTERNAL_DST_DIR",
                             get(ENV, "SOLARSINDY_MONITOR_DIR", "live_forecasts"))
const EXTERNAL_DST_LOG = joinpath(EXTERNAL_DST_DIR, "external_dst_forecast_log.csv")
const EXTERNAL_DST_REPORT = joinpath(EXTERNAL_DST_DIR, "external_dst_forecast_report.md")
const EXTERNAL_DST_RAW_DIR = joinpath(EXTERNAL_DST_DIR, "source_cache", "external_dst_snapshots")
const EXTERNAL_DST_REPO_ROOT = normpath(joinpath(EXTERNAL_DST_DIR, ".."))
const EXTERNAL_DST_MAX_OBS_GAP_MIN = 30.0
const EXTERNAL_DST_SOURCES = [
    (; name = "swpc_geospace_dst_1_hour",
       url = "https://services.swpc.noaa.gov/json/geospace/geospace_dst_1_hour.json",
       kind = "swpc_geospace_json"),
    (; name = "temerin_li_dst_last_96h",
       url = "https://lasp.colorado.edu/space_weather/dsttemerin/dst_last_96_hrs.txt",
       kind = "temerin_li_ascii",
       run_url = "https://lasp.colorado.edu/space_weather/dsttemerin/dsttemerin.html"),
]
const EXTERNAL_DST_OBS_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
const EXTERNAL_DST_SCHEMA = [
    :source, :issue_utc, :fetched_utc, :target_utc, :lead_h, :forecast_dst_nt,
    :forecast_cadence_min, :issue_basis, :source_url, :raw_sha256, :raw_path,
    :source_run_utc, :source_last_modified_utc, :source_max_target_utc, :row_role,
    :observed_dst_nt, :observed_time_utc, :observed_gap_min, :abs_error_nt, :scored_utc,
]

Base.@kwdef struct ExternalDstCollectorConfig
    log_path::String = EXTERNAL_DST_LOG
    report_path::String = EXTERNAL_DST_REPORT
    raw_dir::String = EXTERNAL_DST_RAW_DIR
    repo_root::String = EXTERNAL_DST_REPO_ROOT
    sources::Vector = EXTERNAL_DST_SOURCES
    obs_url::String = EXTERNAL_DST_OBS_URL
    max_obs_gap_min::Float64 = EXTERNAL_DST_MAX_OBS_GAP_MIN
end

function _external_empty_log()
    return DataFrame(
        source = String[],
        issue_utc = String[],
        fetched_utc = String[],
        target_utc = String[],
        lead_h = Float64[],
        forecast_dst_nt = Float64[],
        forecast_cadence_min = Float64[],
        issue_basis = String[],
        source_url = String[],
        raw_sha256 = String[],
        raw_path = String[],
        source_run_utc = Union{Missing, String}[],
        source_last_modified_utc = Union{Missing, String}[],
        source_max_target_utc = Union{Missing, String}[],
        row_role = String[],
        observed_dst_nt = Union{Missing, Float64}[],
        observed_time_utc = Union{Missing, String}[],
        observed_gap_min = Union{Missing, Float64}[],
        abs_error_nt = Union{Missing, Float64}[],
        scored_utc = Union{Missing, String}[],
    )
end

_fmt_utc(t::DateTime) = Dates.format(t, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
_slug(s::AbstractString) = replace(lowercase(String(s)), r"[^a-z0-9]+" => "_")

function _parse_external_time(x)
    (x === missing || x === nothing) && return missing
    s = strip(String(string(x)))
    isempty(s) && return missing
    s = replace(s, " " => "T")
    s = replace(s, r"Z$" => "")
    s = split(s, '.')[1]
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS", dateformat"yyyy/mm/dd-HH:MM:SS")
        t = tryparse(DateTime, s, fmt)
        t !== nothing && return t
    end
    return missing
end

function _parse_http_last_modified(headers)
    month_map = Dict("Jan" => 1, "Feb" => 2, "Mar" => 3, "Apr" => 4,
                     "May" => 5, "Jun" => 6, "Jul" => 7, "Aug" => 8,
                     "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dec" => 12)
    for (k, v) in headers
        lowercase(String(k)) == "last-modified" || continue
        raw = String(v)
        m = match(r"^\w{3},\s+(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+GMT$", raw)
        if m !== nothing && haskey(month_map, m.captures[2])
            return DateTime(parse(Int, m.captures[3]), month_map[m.captures[2]],
                            parse(Int, m.captures[1]), parse(Int, m.captures[4]),
                            parse(Int, m.captures[5]), parse(Int, m.captures[6]))
        end
    end
    return missing
end

_sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(String(text))))

function _http_text(url::AbstractString; http_get::Function = HTTP.get)
    resp = http_get(String(url); connect_timeout = 15, readtimeout = 30,
                    retries = 1, status_exception = true)
    body = String(getproperty(resp, :body))
    last_modified = _parse_http_last_modified(getproperty(resp, :headers))
    return body, last_modified
end

function _write_raw_snapshot(raw_dir::AbstractString, source::AbstractString,
                             fetched_utc::DateTime, sha::AbstractString,
                             body::AbstractString, repo_root::AbstractString)
    mkpath(raw_dir)
    path = joinpath(raw_dir, string(Dates.format(fetched_utc, dateformat"yyyymmddTHHMMSS"),
                                   "Z_", _slug(source), "_", first(String(sha), 12), ".raw"))
    isfile(path) || write(path, body)
    return relpath(path, repo_root)
end

function _parse_temerin_model_run(html::AbstractString)
    m = match(r"Time of model run:\s+(\d{4})/(\d{1,3})-(\d{2}):(\d{2}):(\d{2})", html)
    m === nothing && return missing
    yr = parse(Int, m.captures[1])
    doy = parse(Int, m.captures[2])
    hh = parse(Int, m.captures[3])
    mm = parse(Int, m.captures[4])
    ss = parse(Int, m.captures[5])
    return DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hh) + Minute(mm) + Second(ss)
end

function _parse_temerin_ascii(text::AbstractString)
    rows = DataFrame(target_utc = DateTime[], forecast_dst_nt = Float64[])
    pat = r"^\s*(\d{4})/(\d{1,3})-(\d{2}):(\d{2}):(\d{2})\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    for line in split(text, '\n')
        m = match(pat, line)
        m === nothing && continue
        yr = parse(Int, m.captures[1])
        doy = parse(Int, m.captures[2])
        hh = parse(Int, m.captures[3])
        mm = parse(Int, m.captures[4])
        ss = parse(Int, m.captures[5])
        val = parse(Float64, m.captures[6])
        push!(rows, (DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hh) + Minute(mm) + Second(ss), val))
    end
    sort!(rows, :target_utc)
    return rows
end

function _parse_swpc_geospace_json(text::AbstractString)
    raw = JSON3.read(text)
    rows = DataFrame(target_utc = DateTime[], forecast_dst_nt = Float64[])
    for r in raw
        tt = get(r, :time_tag, get(r, "time_tag", nothing))
        dv = get(r, :dst, get(r, "dst", nothing))
        t = _parse_external_time(tt)
        t === missing && continue
        val = try
            Float64(dv)
        catch
            NaN
        end
        isfinite(val) && push!(rows, (t, val))
    end
    sort!(rows, :target_utc)
    return rows
end

function _median_cadence_min(times::Vector{DateTime})
    length(times) < 2 && return NaN
    gaps = [Dates.value(times[i] - times[i - 1]) / 60000 for i in 2:length(times)]
    return Float64(median(gaps))
end

function _future_rows_for_source(source; fetched_utc::DateTime = now(UTC),
                                 http_get::Function = HTTP.get,
                                 raw_dir::AbstractString = EXTERNAL_DST_RAW_DIR,
                                 repo_root::AbstractString = EXTERNAL_DST_REPO_ROOT)
    body, last_modified = _http_text(source.url; http_get = http_get)
    sha = _sha256_hex(body)
    raw_path = _write_raw_snapshot(raw_dir, source.name, fetched_utc, sha, body, repo_root)
    source_run = missing
    issue_basis = "http_last_modified"
    if source.kind == "temerin_li_ascii"
        run_html, _ = _http_text(source.run_url; http_get = http_get)
        source_run = _parse_temerin_model_run(run_html)
        source_run === missing || (issue_basis = "source_model_run")
        forecast = _parse_temerin_ascii(body)
    elseif source.kind == "swpc_geospace_json"
        forecast = _parse_swpc_geospace_json(body)
    else
        error("unknown external Dst source kind: $(source.kind)")
    end
    isempty(forecast) && error("$(source.name) produced no parseable Dst rows")
    issue = source_run !== missing ? source_run :
            last_modified !== missing ? last_modified : fetched_utc
    issue_basis = source_run !== missing ? issue_basis :
                  last_modified !== missing ? "http_last_modified" : "fetch_time"
    source_max = maximum(forecast.target_utc)
    cadence = _median_cadence_min(DateTime.(forecast.target_utc))

    out = _external_empty_log()
    for r in eachrow(forecast)
        target = DateTime(r.target_utc)
        target > issue || continue
        lead_h = Dates.value(target - issue) / 3_600_000
        push!(out, (
            String(source.name), _fmt_utc(issue), _fmt_utc(fetched_utc),
            _fmt_utc(target), Float64(lead_h), Float64(r.forecast_dst_nt),
            cadence, issue_basis, String(source.url), String(sha), String(raw_path),
            source_run === missing ? missing : _fmt_utc(source_run),
            last_modified === missing ? missing : _fmt_utc(last_modified),
            _fmt_utc(source_max), "future_forecast",
            missing, missing, missing, missing, missing,
        ))
    end
    return out
end

function _load_external_log(path::AbstractString)
    isfile(path) || return _external_empty_log()
    df = CSV.read(path, DataFrame; missingstring = "")
    for col in EXTERNAL_DST_SCHEMA
        col in propertynames(df) || error("external Dst log missing column $col")
    end
    return df
end

function _dedupe_external_log(df::DataFrame)
    seen = Set{String}()
    keep = trues(nrow(df))
    for i in 1:nrow(df)
        issue = _parse_external_time(df.issue_utc[i])
        target = _parse_external_time(df.target_utc[i])
        issue_key = issue === missing ? string(df.issue_utc[i]) : _fmt_utc(issue)
        target_key = target === missing ? string(df.target_utc[i]) : _fmt_utc(target)
        key = join((string(df.source[i]), issue_key, target_key, string(df.raw_sha256[i])), "|")
        if key in seen
            keep[i] = false
        else
            push!(seen, key)
        end
    end
    return df[keep, :]
end

function _parse_observed_dst_json(text::AbstractString)
    raw = JSON3.read(text)
    rows = DataFrame(observed_time_utc = DateTime[], observed_dst_nt = Float64[])
    for r in raw
        tt = get(r, :time_tag, get(r, "time_tag", nothing))
        dv = get(r, :dst, get(r, "dst", nothing))
        t = _parse_external_time(tt)
        t === missing && continue
        val = try
            Float64(dv)
        catch
            NaN
        end
        isfinite(val) && push!(rows, (t, val))
    end
    sort!(rows, :observed_time_utc)
    return rows
end

function _nearest_observation(obs::DataFrame, target::DateTime)
    isempty(obs) && return nothing, Inf
    times = DateTime.(obs.observed_time_utc)
    idx = searchsortedfirst(times, target)
    best_i = 0
    best_gap = Inf
    for j in (idx - 1):(idx + 1)
        1 <= j <= length(times) || continue
        gap = abs(Dates.value(times[j] - target)) / 60000
        if gap < best_gap
            best_i = j
            best_gap = gap
        end
    end
    best_i == 0 && return nothing, Inf
    return best_i, best_gap
end

function score_external_dst_rows!(df::DataFrame, obs::DataFrame;
                                  max_obs_gap_min::Real = EXTERNAL_DST_MAX_OBS_GAP_MIN,
                                  scored_utc::DateTime = now(UTC))
    isempty(df) && return 0
    sort!(obs, :observed_time_utc)
    scored = 0
    for i in 1:nrow(df)
        ismissing(df.observed_dst_nt[i]) || continue
        target = _parse_external_time(df.target_utc[i])
        target === missing && continue
        idx, gap = _nearest_observation(obs, target)
        idx === nothing && continue
        gap <= max_obs_gap_min || continue
        obs_val = Float64(obs.observed_dst_nt[idx])
        df.observed_dst_nt[i] = obs_val
        df.observed_time_utc[i] = _fmt_utc(DateTime(obs.observed_time_utc[idx]))
        df.observed_gap_min[i] = Float64(gap)
        df.abs_error_nt[i] = abs(Float64(df.forecast_dst_nt[i]) - obs_val)
        df.scored_utc[i] = _fmt_utc(scored_utc)
        scored += 1
    end
    return scored
end

function _fetch_observations(url::AbstractString; http_get::Function = HTTP.get)
    body, _ = _http_text(url; http_get = http_get)
    return _parse_observed_dst_json(body)
end

function _validate_external_dst_log(df::DataFrame)
    for col in EXTERNAL_DST_SCHEMA
        col in propertynames(df) || error("external Dst log missing column $col")
    end
    for i in 1:nrow(df)
        issue = _parse_external_time(df.issue_utc[i])
        target = _parse_external_time(df.target_utc[i])
        issue === missing && error("external Dst row $i has unparsable issue_utc")
        target === missing && error("external Dst row $i has unparsable target_utc")
        target > issue || error("external Dst row $i is not a future forecast row")
        isfinite(Float64(df.lead_h[i])) && Float64(df.lead_h[i]) > 0 ||
            error("external Dst row $i has invalid lead_h")
        isfinite(Float64(df.forecast_dst_nt[i])) ||
            error("external Dst row $i has invalid forecast_dst_nt")
        length(String(df.raw_sha256[i])) == 64 ||
            error("external Dst row $i has invalid raw_sha256")
        if !ismissing(df.observed_dst_nt[i])
            isfinite(Float64(df.observed_dst_nt[i])) ||
                error("external Dst row $i has invalid observed_dst_nt")
            isfinite(Float64(df.abs_error_nt[i])) ||
                error("external Dst row $i has invalid abs_error_nt")
            isfinite(Float64(df.observed_gap_min[i])) &&
                Float64(df.observed_gap_min[i]) <= EXTERNAL_DST_MAX_OBS_GAP_MIN + 1e-9 ||
                error("external Dst row $i has invalid observed_gap_min")
        end
    end
    return true
end

function external_dst_summary(df::DataFrame)
    out = DataFrame(source = String[], n_rows = Int[], n_scored = Int[],
                    n_issues = Int[], max_lead_h = Float64[],
                    rmse_nt = Union{Missing, Float64}[], mae_nt = Union{Missing, Float64}[])
    for source in sort(unique(String.(df.source)))
        sub = df[String.(df.source) .== source, :]
        scored_mask = .!ismissing.(sub.observed_dst_nt)
        if any(scored_mask)
            err = Float64.(sub.forecast_dst_nt[scored_mask]) .- Float64.(sub.observed_dst_nt[scored_mask])
            rmse_val = sqrt(mean(err .^ 2))
            mae_val = mean(abs.(err))
        else
            rmse_val = missing
            mae_val = missing
        end
        push!(out, (source, nrow(sub), count(scored_mask), length(unique(String.(sub.issue_utc))),
                    maximum(Float64.(sub.lead_h)), rmse_val, mae_val))
    end
    return out
end

function _write_external_dst_report(path::AbstractString, df::DataFrame)
    summary = external_dst_summary(df)
    open(path, "w") do io
        println(io, "# Prospective external Dst forecast snapshots\n")
        println(io, "This log captures public same-unit Dst forecast/nowcast products as issue-time snapshots. Rows are written only when the product target time is after the inferred issue time. The collector stores raw-response SHA-256 hashes and scores rows later against the SWPC-served Kyoto Dst product within $(EXTERNAL_DST_MAX_OBS_GAP_MIN) min.\n")
        println(io, "| Source | Rows | Scored | Issues | Max lead [h] | RMSE [nT] | MAE [nT] |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(summary)
            rmse_s = ismissing(r.rmse_nt) ? "pending" : @sprintf("%.2f", r.rmse_nt)
            mae_s = ismissing(r.mae_nt) ? "pending" : @sprintf("%.2f", r.mae_nt)
            @printf(io, "| %s | %d | %d | %d | %.3f | %s | %s |\n",
                    r.source, r.n_rows, r.n_scored, r.n_issues, r.max_lead_h,
                    rmse_s, mae_s)
        end
        println(io, "\nBoundary: this starts a prospective issue-time-resolved external Dst archive. It does not backfill missing historical issue snapshots, and current public products may provide sub-hour future Dst rows rather than the full 1--6 h V2 lead set.")
    end
end

function capture_and_score_external_dst_snapshot!(cfg::ExternalDstCollectorConfig = ExternalDstCollectorConfig();
                                                  fetched_utc::DateTime = now(UTC),
                                                  http_get::Function = HTTP.get)
    current = _load_external_log(cfg.log_path)
    new_rows = _external_empty_log()
    for source in cfg.sources
        rows = _future_rows_for_source(source; fetched_utc = fetched_utc,
                                       http_get = http_get, raw_dir = cfg.raw_dir,
                                       repo_root = cfg.repo_root)
        append!(new_rows, rows; cols = :union)
    end
    combined = vcat(current, new_rows; cols = :union)
    combined = _dedupe_external_log(combined)
    obs = _fetch_observations(cfg.obs_url; http_get = http_get)
    n_scored = score_external_dst_rows!(combined, obs; max_obs_gap_min = cfg.max_obs_gap_min,
                                        scored_utc = fetched_utc)
    _validate_external_dst_log(combined)
    mkpath(dirname(cfg.log_path))
    CSV.write(cfg.log_path, combined)
    _write_external_dst_report(cfg.report_path, combined)
    return (; rows_added = nrow(combined) - nrow(current),
            rows_total = nrow(combined),
            rows_scored_now = n_scored,
            summary = external_dst_summary(combined))
end

function _mock_response(body::AbstractString; last_modified::Union{Nothing, String} = nothing)
    headers = last_modified === nothing ? Pair{String, String}[] : ["Last-Modified" => last_modified]
    return (; status = 200, body = Vector{UInt8}(codeunits(String(body))), headers)
end

function _selftest_external_dst_collector()
    swpc = """[
      {"time_tag":"2026-06-27T05:00:00","dst":-20.0},
      {"time_tag":"2026-06-27T05:20:00","dst":-25.0},
      {"time_tag":"2026-06-27T05:40:00","dst":-30.0}
    ]"""
    temerin = """
          Time          Predicted Dst
    2026/178-05:01:00      -19.0
    2026/178-05:11:00      -21.0
    2026/178-05:31:00      -27.0
    """
    html = """
    <pre>
    Time of model run:     2026/178-05:05:44
    </pre>
    """
    obs = """[
      {"time_tag":"2026-06-27T05:00:00","dst":-18},
      {"time_tag":"2026-06-27T06:00:00","dst":-29}
    ]"""
    function fake_get(url; kwargs...)
        u = String(url)
        if occursin("geospace_dst_1_hour", u)
            return _mock_response(swpc; last_modified = "Sat, 27 Jun 2026 05:10:00 GMT")
        elseif occursin("dst_last_96_hrs", u)
            return _mock_response(temerin)
        elseif occursin("dsttemerin.html", u)
            return _mock_response(html)
        elseif occursin("kyoto-dst", u)
            return _mock_response(obs)
        end
        error("unexpected URL $u")
    end
    mktempdir() do dir
        cfg = ExternalDstCollectorConfig(;
            log_path = joinpath(dir, "external_dst_forecast_log.csv"),
            report_path = joinpath(dir, "external_dst_forecast_report.md"),
            raw_dir = joinpath(dir, "raw"),
            repo_root = dir,
        )
        result = capture_and_score_external_dst_snapshot!(cfg;
            fetched_utc = DateTime(2026, 6, 27, 5, 12),
            http_get = fake_get,
        )
        @assert result.rows_total == 4 "future-row filtering should keep 4 rows"
        df = CSV.read(cfg.log_path, DataFrame)
        @assert _validate_external_dst_log(df)
        @assert all(DateTime.(replace.(df.target_utc, "Z" => "")) .>
                    DateTime.(replace.(df.issue_utc, "Z" => "")))
        @assert all(length.(String.(df.raw_sha256)) .== 64)
        @assert all(!isabspath(String(p)) for p in df.raw_path)
        @assert count(.!ismissing.(df.observed_dst_nt)) == 4
        @assert isfile(cfg.report_path)
        result2 = capture_and_score_external_dst_snapshot!(cfg;
            fetched_utc = DateTime(2026, 6, 27, 5, 13),
            http_get = fake_get,
        )
        @assert result2.rows_total == 4 "dedupe should not append duplicate source issue/target/hash rows"
    end
    println("  ✓ external Dst snapshot collector self-test: future rows, raw hashes, scoring CRC")
    return true
end

function main_external_dst_collector(args = ARGS)
    if "--self-test" in args
        return _selftest_external_dst_collector()
    end
    cfg = ExternalDstCollectorConfig()
    result = capture_and_score_external_dst_snapshot!(cfg)
    println("External Dst snapshot collector: rows_total=", result.rows_total,
            ", rows_added=", result.rows_added,
            ", rows_scored_now=", result.rows_scored_now)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_external_dst_collector()
end
