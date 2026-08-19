#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using Downloads
using Printf
using Statistics

isdefined(@__MODULE__, :_selftest_v2) || include(joinpath(@__DIR__, "v2_replay.jl"))

const GFZ_KP_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
const GFZ_KP_SOURCE = OPERATIONAL_GFZ_KP_SOURCE
const GSCALE_OUT_EVENTS = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_event_catalog.csv")
const GSCALE_OUT_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_replay_scored.csv")
const GSCALE_OUT_SUMMARY = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_replay_summary.csv")
const GSCALE_OUT_MD = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_replay_report.md")
const GSCALE_MIN_KP = 7.0
const GSCALE_MIN_LEVEL = 3
const GSCALE_GAP_HOURS = 6
const GSCALE_PRE_HOURS = 6
const GSCALE_POST_HOURS = 24
const GSCALE_START_UTC = DateTime(1964, 1, 1)
const GSCALE_END_UTC = DateTime(2025, 12, 31, 23)

Base.@kwdef struct GScaleReplayConfig
    kp_source::String = GFZ_KP_SOURCE
    min_level::Int = GSCALE_MIN_LEVEL
    min_kp::Float64 = GSCALE_MIN_KP
    gap_hours::Int = GSCALE_GAP_HOURS
    pre_hours::Int = GSCALE_PRE_HOURS
    post_hours::Int = GSCALE_POST_HOURS
    start_utc::DateTime = GSCALE_START_UTC
    end_utc::DateTime = GSCALE_END_UTC
    limit::Int = 0
    out_events::String = GSCALE_OUT_EVENTS
    out_csv::String = GSCALE_OUT_CSV
    out_summary::String = GSCALE_OUT_SUMMARY
    out_report::String = GSCALE_OUT_MD
    self_test_only::Bool = false
end

function _parse_gscale_args(args)::GScaleReplayConfig
    cfg = GScaleReplayConfig()
    vals = Dict(
        :kp_source => cfg.kp_source,
        :min_level => cfg.min_level,
        :min_kp => cfg.min_kp,
        :gap_hours => cfg.gap_hours,
        :pre_hours => cfg.pre_hours,
        :post_hours => cfg.post_hours,
        :start_utc => cfg.start_utc,
        :end_utc => cfg.end_utc,
        :limit => cfg.limit,
        :out_events => cfg.out_events,
        :out_csv => cfg.out_csv,
        :out_summary => cfg.out_summary,
        :out_report => cfg.out_report,
        :self_test_only => cfg.self_test_only,
    )
    for arg in args
        if arg == "--self-test"
            vals[:self_test_only] = true
        elseif startswith(arg, "--kp-source=")
            vals[:kp_source] = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--min-level=")
            vals[:min_level] = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--min-kp=")
            vals[:min_kp] = parse(Float64, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--gap-hours=")
            vals[:gap_hours] = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--pre-hours=")
            vals[:pre_hours] = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--post-hours=")
            vals[:post_hours] = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--start=")
            vals[:start_utc] = DateTime(Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--end=")
            vals[:end_utc] = DateTime(Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--limit=")
            vals[:limit] = parse(Int, Base.split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--out-events=")
            vals[:out_events] = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-csv=")
            vals[:out_csv] = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-summary=")
            vals[:out_summary] = Base.split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-report=")
            vals[:out_report] = Base.split(arg, "=", limit = 2)[2]
        else
            error("unknown argument: $arg")
        end
    end
    vals[:min_level] in 1:5 || throw(ArgumentError("--min-level must be in 1:5"))
    vals[:min_kp] >= 0.0 || throw(ArgumentError("--min-kp must be nonnegative"))
    vals[:gap_hours] >= 0 || throw(ArgumentError("--gap-hours must be nonnegative"))
    vals[:pre_hours] >= 0 || throw(ArgumentError("--pre-hours must be nonnegative"))
    vals[:post_hours] >= 0 || throw(ArgumentError("--post-hours must be nonnegative"))
    vals[:limit] >= 0 || throw(ArgumentError("--limit must be nonnegative"))
    vals[:start_utc] <= vals[:end_utc] || throw(ArgumentError("--start must not exceed --end"))
    return GScaleReplayConfig(; vals...)
end

function ensure_gfz_kp_source(path::AbstractString = GFZ_KP_SOURCE; url::AbstractString = GFZ_KP_URL)
    if isfile(path) && filesize(path) > 0
        return path
    end
    mkpath(dirname(path))
    Downloads.download(url, path)
    filesize(path) > 0 || error("downloaded Kp source is empty: $path")
    return path
end

function noaa_g_level(kp::Real)
    x = Float64(kp)
    x >= 9.0 && return 5
    x >= 8.0 && return 4
    x >= 7.0 && return 3
    x >= 6.0 && return 2
    x >= 5.0 && return 1
    return 0
end

function load_gfz_kp(path::AbstractString = GFZ_KP_SOURCE)
    ensure_gfz_kp_source(path)
    out = DataFrame(utc = DateTime[], kp = Float64[], ap = Int[], definitive = Int[])
    open(path, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            isempty(strip(line)) && continue
            parts = split(strip(line))
            length(parts) >= 28 || continue
            y = parse(Int, parts[1])
            m = parse(Int, parts[2])
            d = parse(Int, parts[3])
            kps = parse.(Float64, parts[8:15])
            aps = parse.(Int, parts[16:23])
            definitive = parse(Int, parts[28])
            day0 = DateTime(y, m, d)
            for i in 1:8
                kps[i] < 0 && continue
                push!(out, (day0 + Hour(3 * (i - 1)), kps[i], aps[i], definitive))
            end
        end
    end
    sort!(out, :utc)
    return out
end

function build_gscale_events(kp_rows::DataFrame; min_kp::Real = GSCALE_MIN_KP,
                             gap_hours::Int = GSCALE_GAP_HOURS,
                             pre_hours::Int = GSCALE_PRE_HOURS,
                             post_hours::Int = GSCALE_POST_HOURS,
                             start_utc::DateTime = GSCALE_START_UTC,
                             end_utc::DateTime = GSCALE_END_UTC,
                             limit::Int = 0)
    rows = kp_rows[(kp_rows.utc .>= start_utc) .& (kp_rows.utc .<= end_utc) .&
                   (Float64.(kp_rows.kp) .>= Float64(min_kp)), :]
    sort!(rows, :utc)
    events = DataFrame(g_event_id = Int[], event_start_utc = DateTime[],
                       event_end_utc = DateTime[], replay_start_utc = DateTime[],
                       replay_end_utc = DateTime[], peak_kp = Float64[],
                       peak_g_level = Int[], peak_utc = DateTime[],
                       n_kp_bins = Int[], source = String[])
    isempty(rows) && return events
    event_id = 0
    cur_start = rows.utc[1]
    cur_end = rows.utc[1] + Hour(3)
    cur_peak = Float64(rows.kp[1])
    cur_peak_utc = rows.utc[1]
    n_bins = 1

    function flush_event!()
        event_id += 1
        push!(events, (
            event_id,
            cur_start,
            cur_end,
            cur_start - Hour(pre_hours),
            cur_end + Hour(post_hours),
            cur_peak,
            noaa_g_level(cur_peak),
            cur_peak_utc,
            n_bins,
            "GFZ Kp/ap since 1932; NOAA G scale from Kp",
        ))
    end

    for i in 2:nrow(rows)
        t = rows.utc[i]
        kp = Float64(rows.kp[i])
        if t - cur_end <= Hour(gap_hours)
            cur_end = t + Hour(3)
            n_bins += 1
            if kp > cur_peak
                cur_peak = kp
                cur_peak_utc = t
            end
        else
            flush_event!()
            cur_start = t
            cur_end = t + Hour(3)
            cur_peak = kp
            cur_peak_utc = t
            n_bins = 1
        end
    end
    flush_event!()
    events = _merge_overlapping_gscale_windows(events)
    if limit > 0
        events = first(events, min(limit, nrow(events)))
    end
    return events
end

function _merge_overlapping_gscale_windows(events::DataFrame)
    nrow(events) <= 1 && return events
    sort!(events, :replay_start_utc)
    merged = DataFrame(g_event_id = Int[], event_start_utc = DateTime[],
                       event_end_utc = DateTime[], replay_start_utc = DateTime[],
                       replay_end_utc = DateTime[], peak_kp = Float64[],
                       peak_g_level = Int[], peak_utc = DateTime[],
                       n_kp_bins = Int[], source = String[])

    cur_event_start = DateTime(events.event_start_utc[1])
    cur_event_end = DateTime(events.event_end_utc[1])
    cur_replay_start = DateTime(events.replay_start_utc[1])
    cur_replay_end = DateTime(events.replay_end_utc[1])
    cur_peak_kp = Float64(events.peak_kp[1])
    cur_peak_g = Int(events.peak_g_level[1])
    cur_peak_utc = DateTime(events.peak_utc[1])
    cur_bins = Int(events.n_kp_bins[1])
    cur_source = String(events.source[1])
    for i in 2:nrow(events)
        r = events[i, :]
        if DateTime(r.replay_start_utc) <= cur_replay_end
            cur_event_end = max(cur_event_end, DateTime(r.event_end_utc))
            cur_replay_end = max(cur_replay_end, DateTime(r.replay_end_utc))
            cur_bins += Int(r.n_kp_bins)
            if Float64(r.peak_kp) > cur_peak_kp
                cur_peak_kp = Float64(r.peak_kp)
                cur_peak_g = Int(r.peak_g_level)
                cur_peak_utc = DateTime(r.peak_utc)
            end
        else
            push!(merged, (nrow(merged) + 1, cur_event_start, cur_event_end,
                           cur_replay_start, cur_replay_end, cur_peak_kp,
                           cur_peak_g, cur_peak_utc, cur_bins, cur_source))
            cur_event_start = DateTime(r.event_start_utc)
            cur_event_end = DateTime(r.event_end_utc)
            cur_replay_start = DateTime(r.replay_start_utc)
            cur_replay_end = DateTime(r.replay_end_utc)
            cur_peak_kp = Float64(r.peak_kp)
            cur_peak_g = Int(r.peak_g_level)
            cur_peak_utc = DateTime(r.peak_utc)
            cur_bins = Int(r.n_kp_bins)
            cur_source = String(r.source)
        end
    end
    push!(merged, (nrow(merged) + 1, cur_event_start, cur_event_end,
                   cur_replay_start, cur_replay_end, cur_peak_kp,
                   cur_peak_g, cur_peak_utc, cur_bins, cur_source))
    return merged
end

function _gscale_storm_from_row(r)
    return (
        name = @sprintf("G%d Kp event %d (%s, peak Kp %.3g)",
                        Int(r.peak_g_level), Int(r.g_event_id),
                        Dates.format(r.event_start_utc, dateformat"yyyy-mm-dd"), Float64(r.peak_kp)),
        t0 = DateTime(r.replay_start_utc),
        t1 = DateTime(r.replay_end_utc),
        event_start = DateTime(r.event_start_utc),
        event_end = DateTime(r.event_end_utc),
        event_id = Int(r.g_event_id),
        peak_kp = Float64(r.peak_kp),
        peak_g_level = Int(r.peak_g_level),
        peak_utc = DateTime(r.peak_utc),
        n_kp_bins = Int(r.n_kp_bins),
    )
end

function _with_gscale_metadata(rows::DataFrame, storm)
    out = copy(rows)
    out[!, :g_event_id] = fill(storm.event_id, nrow(out))
    out[!, :g_level] = fill(storm.peak_g_level, nrow(out))
    out[!, :peak_kp] = fill(storm.peak_kp, nrow(out))
    out[!, :peak_kp_utc] = fill(storm.peak_utc, nrow(out))
    out[!, :n_kp_bins] = fill(storm.n_kp_bins, nrow(out))
    out[!, :event_start_utc] = fill(storm.event_start, nrow(out))
    out[!, :event_end_utc] = fill(storm.event_end, nrow(out))
    out[!, :replay_start_utc] = fill(storm.t0, nrow(out))
    out[!, :replay_end_utc] = fill(storm.t1, nrow(out))
    out[!, :target_utc] = out.issue_utc .+ Hour.(Int.(out.lead))
    return select(out, :g_event_id, :storm, :g_level, :peak_kp, :peak_kp_utc,
                  :n_kp_bins, :event_start_utc, :event_end_utc, :replay_start_utc,
                  :replay_end_utc, :issue_utc, :target_utc, :lead, :obs,
                  :v2_1, :v2_1_pre_rate_guard, :v2_1_pre_one_hour_inertia,
                  :v2_1_pre_state_inertia,
                  :v2_0, :v2_1_frozen,
                  :persistence, :rate)
end

function run_gscale_replay(events::DataFrame)
    current_core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    historical_core = load_operational_core(OPERATIONAL_V2_0_MODEL_VERSION)
    current_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
    historical_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_0_MODEL_VERSION).point_csv,
    )
    min_year = minimum(year(DateTime(t)) for t in events.replay_start_utc) - 1
    max_year = maximum(year(DateTime(t)) for t in events.replay_end_utc)
    archive = _load_replay_archive(min_year, max_year)
    scored = DataFrame()
    skipped = DataFrame(g_event_id = Int[], reason = String[])
    for r in eachrow(events)
        storm = _gscale_storm_from_row(r)
        try
            rows = replay_v2_storm(
                storm, current_core, current_cal, historical_core, historical_cal,
                archive.lookup; replay_inputs=archive.inputs,
            )
            if nrow(rows) == 0
                push!(skipped, (storm.event_id, "no finite scored rows"))
            else
                append!(scored, _with_gscale_metadata(rows, storm); cols = :union)
            end
        catch err
            push!(skipped, (storm.event_id, sprint(showerror, err)))
        end
    end
    return scored, skipped
end

_gscale_rmse(residuals) = sqrt(mean(abs2, Float64.(residuals)))

function _gscale_metric_row(rows::DataFrame, cohort::AbstractString, lead::Int)
    sub = rows[Int.(rows.lead) .== lead, :]
    nrow(sub) == 0 && return nothing
    rv20 = _gscale_rmse(sub.obs .- sub.v2_0)
    rv21 = _gscale_rmse(sub.obs .- sub.v2_1)
    rpers = _gscale_rmse(sub.obs .- sub.persistence)
    return (
        cohort = String(cohort),
        lead_h = lead,
        n_rows = nrow(sub),
        n_events = length(unique(Int.(sub.g_event_id))),
        min_g_level = minimum(Int.(sub.g_level)),
        max_g_level = maximum(Int.(sub.g_level)),
        rmse_v2_0_nt = rv20,
        rmse_v2_1_nt = rv21,
        rmse_persistence_nt = rpers,
        improvement_vs_best_nt = min(rv20, rpers) - rv21,
        max_tail_effect_nt = maximum(abs.(Float64.(sub.v2_1) .- Float64.(sub.v2_1_frozen))),
        max_core_change_nt = maximum(abs.(Float64.(sub.v2_1_frozen) .- Float64.(sub.v2_0))),
    )
end

function gscale_summary(rows::DataFrame)
    out = DataFrame(cohort = String[], lead_h = Int[], n_rows = Int[], n_events = Int[],
                    min_g_level = Int[], max_g_level = Int[],
                    rmse_v2_0_nt = Float64[], rmse_v2_1_nt = Float64[],
                    rmse_persistence_nt = Float64[],
                    improvement_vs_best_nt = Float64[], max_tail_effect_nt = Float64[],
                    max_core_change_nt = Float64[])
    isempty(rows) && return out
    for lead in LEADS
        m = _gscale_metric_row(rows, "all_G3plus", lead)
        m === nothing || push!(out, m)
    end
    for g in sort(unique(Int.(rows.g_level)))
        sub = rows[Int.(rows.g_level) .== g, :]
        for lead in LEADS
            m = _gscale_metric_row(sub, "G$(g)", lead)
            m === nothing || push!(out, m)
        end
    end
    return out
end

function _validate_gscale_rows(rows::DataFrame)
    isempty(rows) && error("G-scale replay produced no scored rows")
    required = [:g_event_id, :storm, :g_level, :peak_kp, :event_start_utc, :event_end_utc,
                :issue_utc, :target_utc, :lead, :obs, :v2_1,
                :v2_1_pre_rate_guard, :v2_1_pre_one_hour_inertia,
                :v2_1_pre_state_inertia,
                :v2_0, :v2_1_frozen,
                :persistence, :rate]
    missing_cols = [String(c) for c in required if !(String(c) in names(rows))]
    isempty(missing_cols) || error("G-scale replay missing columns: $(join(missing_cols, ", "))")
    all(rows.target_utc .== rows.issue_utc .+ Hour.(Int.(rows.lead))) ||
        error("target_utc does not match issue_utc + lead")
    for col in (:obs, :v2_1, :v2_1_pre_rate_guard, :v2_1_pre_one_hour_inertia,
                :v2_1_pre_state_inertia,
                :v2_0, :v2_1_frozen,
                :persistence, :peak_kp)
        all(isfinite, Float64.(rows[!, col])) || error("non-finite values in $col")
    end
    all(Int.(rows.g_level) .>= GSCALE_MIN_LEVEL) || error("found rows below G3")
    return true
end

function _write_gscale_report(path::AbstractString, events::DataFrame, scored::DataFrame,
                              skipped::DataFrame, summary::DataFrame, cfg::GScaleReplayConfig)
    open(path, "w") do io
        println(io, "# Operational V2.1 exact Kp/G-scale replay\n")
        println(io, "Selection: GFZ three-hour Kp bins with Kp >= ", cfg.min_kp,
                " (NOAA G", cfg.min_level, "+), clustered with gap <= ", cfg.gap_hours,
                " h; replay window = event start - ", cfg.pre_hours,
                " h through event end + ", cfg.post_hours, " h.")
        println(io, "Event range: ", cfg.start_utc, " to ", cfg.end_utc,
                ". Events=", nrow(events), "; scored events=",
                isempty(scored) ? 0 : length(unique(Int.(scored.g_event_id))),
                "; scored rows=", nrow(scored), "; skipped events=", nrow(skipped), ".")
        println(io, "Source: GFZ Kp/ap since 1932; NOAA G-scale thresholds are Kp 5/6/7/8/9 for G1/G2/G3/G4/G5.\n")
        println(io, "| cohort | lead h | rows | events | G range | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | improve vs best | max tail effect | max core change |")
        println(io, "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in eachrow(summary)
            @printf(io, "| %s | %d | %d | %d | G%d-G%d | %.2f | %.2f | %.2f | %+.2f | %.2f | %.2f |\n",
                    r.cohort, r.lead_h, r.n_rows, r.n_events, r.min_g_level, r.max_g_level,
                    r.rmse_v2_0_nt, r.rmse_v2_1_nt, r.rmse_persistence_nt,
                    r.improvement_vs_best_nt, r.max_tail_effect_nt, r.max_core_change_nt)
        end
        if nrow(skipped) > 0
            println(io, "\n## Skipped events")
            println(io, "| event id | reason |")
            println(io, "| ---: | --- |")
            for r in eachrow(skipped)
                println(io, "| ", r.g_event_id, " | ", replace(String(r.reason), "\n" => " "), " |")
            end
        end
    end
end

function _selftest_gscale()
    @assert noaa_g_level(4.999) == 0
    @assert noaa_g_level(5.0) == 1
    @assert noaa_g_level(6.0) == 2
    @assert noaa_g_level(7.0) == 3
    @assert noaa_g_level(8.0) == 4
    @assert noaa_g_level(9.0) == 5
    kp = load_gfz_kp()
    @assert nrow(kp) > 250_000
    @assert minimum(kp.utc) == DateTime(1932, 1, 1)
    may2024 = kp[(kp.utc .>= DateTime(2024, 5, 10)) .& (kp.utc .<= DateTime(2024, 5, 12, 21)), :]
    @assert maximum(Float64.(may2024.kp)) == 9.0
    events = build_gscale_events(kp; start_utc = DateTime(2024, 5, 1),
                                 end_utc = DateTime(2024, 5, 31, 23))
    @assert nrow(events) >= 1
    @assert maximum(Float64.(events.peak_kp)) == 9.0
    @assert all(events.replay_start_utc[2:end] .> events.replay_end_utc[1:end-1])
    toy = DataFrame(g_event_id = [1, 1, 1, 1],
                    storm = fill("toy", 4),
                    g_level = fill(3, 4),
                    peak_kp = fill(7.0, 4),
                    peak_kp_utc = fill(DateTime(2024, 1, 1, 3), 4),
                    n_kp_bins = fill(1, 4),
                    event_start_utc = fill(DateTime(2024, 1, 1, 3), 4),
                    event_end_utc = fill(DateTime(2024, 1, 1, 6), 4),
                    replay_start_utc = fill(DateTime(2024, 1, 1), 4),
                    replay_end_utc = fill(DateTime(2024, 1, 2), 4),
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
    @assert _validate_gscale_rows(toy)
    sm = gscale_summary(toy)
    one = sm[(sm.cohort .== "all_G3plus") .& (sm.lead_h .== 1), :][1, :]
    @assert one.n_rows == 1
    @assert isapprox(one.rmse_v2_1_nt, 1.0; atol = 1e-12)
    @assert isapprox(one.rmse_v2_0_nt, 2.0; atol = 1e-12)
    bad = copy(toy)
    bad.target_utc[1] = bad.issue_utc[1]
    try
        _validate_gscale_rows(bad)
        error("bad timing accepted")
    catch err
        err isa ErrorException || rethrow()
    end
    println("  ✓ G-scale replay self-test: GFZ parser, NOAA thresholds, May 2024 G5, event CRC")
    return true
end

function main_gscale(args = ARGS)
    cfg = _parse_gscale_args(args)
    _selftest_v2()          # forecast-layer oracles (continuity, regime, relaxation, inertia)
    _selftest_gscale()
    cfg.self_test_only && return nothing
    kp = load_gfz_kp(cfg.kp_source)
    events = build_gscale_events(kp; min_kp = cfg.min_kp, gap_hours = cfg.gap_hours,
                                 pre_hours = cfg.pre_hours, post_hours = cfg.post_hours,
                                 start_utc = cfg.start_utc, end_utc = cfg.end_utc,
                                 limit = cfg.limit)
    isempty(events) && error("no G-scale events selected")
    println("G-scale V2.1 replay: selected ", nrow(events), " exact Kp/G events at Kp >= ", cfg.min_kp)
    scored, skipped = run_gscale_replay(events)
    _validate_gscale_rows(scored)
    summary = gscale_summary(scored)
    mkpath(dirname(cfg.out_events))
    mkpath(dirname(cfg.out_csv))
    mkpath(dirname(cfg.out_summary))
    mkpath(dirname(cfg.out_report))
    CSV.write(cfg.out_events, events)
    CSV.write(cfg.out_csv, scored)
    CSV.write(cfg.out_summary, summary)
    _write_gscale_report(cfg.out_report, events, scored, skipped, summary, cfg)
    println("  wrote ", cfg.out_events)
    println("  wrote ", cfg.out_csv)
    println("  wrote ", cfg.out_summary)
    println("  wrote ", cfg.out_report)
    println("  scored rows=", nrow(scored), ", events=", length(unique(Int.(scored.g_event_id))),
            ", skipped=", nrow(skipped))
    return (; events, scored, skipped, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_gscale()
end
