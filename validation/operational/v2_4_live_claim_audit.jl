#!/usr/bin/env julia

module V24LiveClaimAudit

using CSV
using DataFrames
using Dates
using JSON3
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "examples", "live_log_lock.jl"))

export SERVED_MANIFEST_SHA256, SHADOW_IDENTITY, SHADOW_CONFIG_SHA256,
       SUPPORTED_MODEL_STEPS,
       canonical_shadow_rows, group_storm_events, prospective_status,
       run_claim_audit

const SERVED_IDENTITY = "v2.4+sindy20x11+superlearner10floor+conformal"
const SERVED_MANIFEST_SHA256 =
    "057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede"
const SHADOW_IDENTITY =
    "v2.4e-cal-shadow-A3+median24+width1h1.50+widthOther1.20+warm30"
const SHADOW_CONFIG_SHA256 =
    "31d67e5077ae6fe69cee133fa07dceec3aa639903a17861d179d3877ed0c21af"
const SHADOW_START_UTC = DateTime(2026, 8, 24, 23, 15, 51, 419)
const SUPPORTED_MODEL_STEPS = (1, 2, 3, 4, 6, 7)
const PUBLIC_HORIZONS = (1, 2, 3, 6)
const STORM_THRESHOLD_NT = -50.0
const QUIET_SEPARATION_THRESHOLD_NT = -30.0
const QUIET_SEPARATION_HOURS = 72
const DEFAULT_BOOTSTRAP_REPS = 20_000
const DEFAULT_BOOTSTRAP_SEED = 2_408_250

const MARGINAL_MINIMUMS = (
    consecutive_days=30,
    complete_issue_cycles=500,
    rows=2_000,
    rows_per_supported_step=400,
    pooled_coverage_lo=0.88,
    pooled_coverage_hi=0.92,
    bootstrap_lower_floor=0.85,
    step_coverage_floor=0.85,
    seven_day_coverage_floor=0.80,
    width_ratio_ceiling=1.25,
)

const STORM_MINIMUMS = (
    independent_events=5,
    matched_rows=200,
    rows_per_supported_step=30,
    strongest_comparator_loss_ceiling_nt=0.5,
    absolute_bias_ceiling_nt=10.0,
    interval_coverage_floor=0.80,
)

const COMPARATOR_COLUMNS = (
    v22=:v2_2_stack_pred_dst_nt,
    v23=:v23_shadow_pred_dst_nt,
    direct_gbm=:direct_gbm_pred_dst_nt,
    v21=:v2_1_served_pred_dst_nt,
    persistence=:persistence_dst_nt,
    burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,
    obrien=:obrien_dst_nt,
)

_shadow_width_scale(step::Int) = step == 1 ? 1.50 : 1.20

_text(value) = ismissing(value) ? "" : string(value)

function _time(value)
    ismissing(value) && return nothing
    value isa DateTime && return value
    value isa AbstractString || return nothing
    text = strip(String(value))
    endswith(text, "Z") && (text = text[1:end-1])
    try
        return DateTime(text)
    catch
        return nothing
    end
end

function _number(value)
    ismissing(value) && return nothing
    value isa Bool && return nothing
    converted = value isa Real ? Float64(value) :
                value isa AbstractString ? tryparse(Float64, strip(String(value))) : nothing
    converted === nothing || isfinite(converted) || return nothing
    return converted
end

_rowvalue(row, column::Symbol) = hasproperty(row, column) ? row[column] : missing

function _interval_score(lo::Real, hi::Real, observed::Real; alpha::Float64=0.10)
    l, h, y = Float64(lo), Float64(hi), Float64(observed)
    all(isfinite, (l, h, y)) && l <= h || throw(ArgumentError("invalid interval score row"))
    return (h - l) + (y < l ? (2 / alpha) * (l - y) : 0.0) +
           (y > h ? (2 / alpha) * (y - h) : 0.0)
end

function _empty_shadow_rows()
    return DataFrame(
        issue_time_utc=DateTime[], issue_hour_utc=DateTime[],
        latest_dst_time_utc=DateTime[], target_time_utc=DateTime[],
        horizon_hours=Int[], model_step_hours=Int[], status=String[],
        observation_dst_nt=Union{Missing,Float64}[], point_dst_nt=Float64[],
        static_lo_dst_nt=Float64[], static_hi_dst_nt=Float64[],
        shadow_lo_dst_nt=Union{Missing,Float64}[],
        shadow_hi_dst_nt=Union{Missing,Float64}[],
        shadow_history_n=Union{Missing,Int}[],
        shadow_location_shift_nt=Union{Missing,Float64}[],
        v22_pred_dst_nt=Union{Missing,Float64}[],
        v23_pred_dst_nt=Union{Missing,Float64}[],
        direct_gbm_pred_dst_nt=Union{Missing,Float64}[],
        v21_pred_dst_nt=Union{Missing,Float64}[],
        persistence_pred_dst_nt=Union{Missing,Float64}[],
        burton_pred_dst_nt=Union{Missing,Float64}[],
        burton_full_pred_dst_nt=Union{Missing,Float64}[],
        obrien_pred_dst_nt=Union{Missing,Float64}[],
    )
end

"Canonical post-freeze shadow rows plus fail-closed integrity findings."
function canonical_shadow_rows(df::DataFrame)
    identity_column = :v24_cal_shadow_model_version

    records = NamedTuple[]
    violations = String[]
    deployment_rows = 0
    warmup_rows = 0
    unexpected_status_rows = 0

    for (row_index, row) in enumerate(eachrow(df))
        shadow_identity = _text(_rowvalue(row, identity_column))
        if isempty(shadow_identity)
            issue = _time(_rowvalue(row, :issue_time_utc))
            issue !== nothing && issue < SHADOW_START_UTC && continue
            deployment_rows += 1
            push!(violations, "row $row_index lacks a shadow identity and cannot establish a pre-deployment issue")
            continue
        end
        deployment_rows += 1
        shadow_identity == SHADOW_IDENTITY || begin
            push!(violations, "row $row_index has an unexpected calibration-shadow identity")
            continue
        end
        _text(_rowvalue(row, :v24_cal_shadow_config_sha256)) == SHADOW_CONFIG_SHA256 || begin
            push!(violations, "row $row_index has an unexpected calibration-shadow digest")
            continue
        end
        _text(_rowvalue(row, :v24_manifest_sha256)) == SERVED_MANIFEST_SHA256 || begin
            push!(violations, "row $row_index has an unexpected served-manifest digest")
            continue
        end
        _text(_rowvalue(row, :sub_hourly_model_version)) == SERVED_IDENTITY || begin
            push!(violations, "row $row_index attaches the shadow to a non-V2.4e served row")
            continue
        end
        _text(_rowvalue(row, :v24_status)) == "ok" || begin
            push!(violations, "row $row_index attaches the shadow to a non-ok V2.4e stage")
            continue
        end

        issue = _time(_rowvalue(row, :issue_time_utc))
        latest = _time(_rowvalue(row, :latest_dst_time_utc))
        target = _time(_rowvalue(row, :target_time_utc))
        wall_horizon = _number(_rowvalue(row, :horizon_hours))
        step_value = _number(_rowvalue(row, :model_step_hours))
        point = _number(_rowvalue(row, :served_pred_dst_nt))
        static_lo = _number(_rowvalue(row, :served_pred_dst_ci05_nt))
        static_hi = _number(_rowvalue(row, :served_pred_dst_ci95_nt))
        subhourly_point = _number(_rowvalue(row, :sub_hourly_pred_dst_nt))
        subhourly_lo = _number(_rowvalue(row, :sub_hourly_pred_dst_ci05_nt))
        subhourly_hi = _number(_rowvalue(row, :sub_hourly_pred_dst_ci95_nt))
        v24_point = _number(_rowvalue(row, :v24_pred_dst_nt))
        v24_lo = _number(_rowvalue(row, :v24_ci05_nt))
        v24_hi = _number(_rowvalue(row, :v24_ci95_nt))
        if any(isnothing, (issue, latest, target, wall_horizon, step_value,
                           point, static_lo, static_hi, subhourly_point,
                           subhourly_lo, subhourly_hi, v24_point, v24_lo, v24_hi))
            push!(violations, "row $row_index has malformed required shadow fields")
            continue
        end
        horizon_value = (target - floor(issue, Hour)) / Hour(1)
        wall_horizon_value = (target - issue) / Hour(1)
        horizon = round(Int, horizon_value)
        latest <= issue < target || begin
            push!(violations, "row $row_index violates the future-target chronology")
            continue
        end
        issue >= SHADOW_START_UTC || begin
            push!(violations, "row $row_index predates the prospective shadow deployment")
            continue
        end
        (isapprox(horizon_value, horizon; atol=1e-9, rtol=0) &&
         horizon in PUBLIC_HORIZONS && wall_horizon > 0 &&
         isapprox(wall_horizon, wall_horizon_value; atol=1e-9, rtol=0)) || begin
            push!(violations, "row $row_index has an unsupported public horizon")
            continue
        end
        step_value in SUPPORTED_MODEL_STEPS || begin
            push!(violations, "row $row_index has an unsupported internal model step")
            continue
        end
        step = Int(step_value)
        ((target - latest) == Hour(step) &&
         (floor(issue, Hour) - latest) in (Hour(0), Hour(1))) || begin
            push!(violations, "row $row_index has inconsistent model-step and Dst-anchor geometry")
            continue
        end
        static_lo <= point <= static_hi && static_lo < static_hi || begin
            push!(violations, "row $row_index has invalid served interval geometry")
            continue
        end
        (point == subhourly_point == v24_point &&
         static_lo == subhourly_lo == v24_lo &&
         static_hi == subhourly_hi == v24_hi) || begin
            push!(violations, "row $row_index does not preserve the served V2.4e values")
            continue
        end

        status = _text(_rowvalue(row, :v24_cal_shadow_status))
        shadow_lo = _number(_rowvalue(row, :v24_cal_shadow_ci05_nt))
        shadow_hi = _number(_rowvalue(row, :v24_cal_shadow_ci95_nt))
        history_value = _number(_rowvalue(row, :v24_cal_shadow_history_n))
        location = _number(_rowvalue(row, :v24_cal_shadow_location_shift_nt))
        history_n = history_value !== nothing && 0 <= history_value < typemax(Int) &&
                    isinteger(history_value) ?
                    round(Int, history_value) : nothing
        if status == "ok"
            (shadow_lo !== nothing && shadow_hi !== nothing && shadow_lo < shadow_hi &&
             history_n !== nothing && history_n >= 30 && location !== nothing) || begin
                push!(violations, "row $row_index has invalid available shadow endpoints")
                continue
            end
            scale = _shadow_width_scale(step)
            expected_lo = point + location - scale * (point - static_lo)
            expected_hi = point + location + scale * (static_hi - point)
            (isapprox(shadow_lo, expected_lo; atol=1e-9, rtol=1e-12) &&
             isapprox(shadow_hi, expected_hi; atol=1e-9, rtol=1e-12)) || begin
                push!(violations, "row $row_index does not match the frozen A3 interval rule")
                continue
            end
        elseif startswith(status, "warmup:")
            warmup_rows += 1
            parsed = match(r"^warmup:(\d+)/30$", status)
            expected_history = parsed === nothing ? nothing : tryparse(Int, parsed.captures[1])
            (expected_history !== nothing && expected_history < 30 &&
             history_n == expected_history && shadow_lo === nothing &&
             shadow_hi === nothing && location === nothing) ||
                push!(violations, "row $row_index has invalid shadow warm-up state")
        else
            unexpected_status_rows += 1
        end

        raw_observation = _rowvalue(row, :observation_dst_nt)
        observation = _number(raw_observation)
        ismissing(raw_observation) || observation !== nothing || begin
            push!(violations, "row $row_index has a non-finite or malformed observation")
            continue
        end
        comparator_values = map(column -> _number(_rowvalue(row, column)),
                                values(COMPARATOR_COLUMNS))
        record = (
            issue_time_utc=issue, issue_hour_utc=floor(issue, Hour),
            latest_dst_time_utc=latest, target_time_utc=target,
            horizon_hours=horizon, model_step_hours=step, status,
            observation_dst_nt=something(observation, missing), point_dst_nt=point,
            static_lo_dst_nt=static_lo, static_hi_dst_nt=static_hi,
            shadow_lo_dst_nt=something(shadow_lo, missing),
            shadow_hi_dst_nt=something(shadow_hi, missing),
            shadow_history_n=something(history_n, missing),
            shadow_location_shift_nt=something(location, missing),
            v22_pred_dst_nt=something(comparator_values[1], missing),
            v23_pred_dst_nt=something(comparator_values[2], missing),
            direct_gbm_pred_dst_nt=something(comparator_values[3], missing),
            v21_pred_dst_nt=something(comparator_values[4], missing),
            persistence_pred_dst_nt=something(comparator_values[5], missing),
            burton_pred_dst_nt=something(comparator_values[6], missing),
            burton_full_pred_dst_nt=something(comparator_values[7], missing),
            obrien_pred_dst_nt=something(comparator_values[8], missing),
        )
        push!(records, record)
    end

    chosen = Dict{Tuple{DateTime,DateTime,String},NamedTuple}()
    for record in records
        key = (record.issue_hour_utc, record.target_time_utc, SHADOW_IDENTITY)
        previous = get(chosen, key, nothing)
        if previous === nothing || record.issue_time_utc > previous.issue_time_utc
            chosen[key] = record
        elseif record.issue_time_utc == previous.issue_time_utc && !isequal(record, previous)
            push!(violations, "conflicting exact-time shadow duplicate for $key")
        end
    end
    rows = isempty(chosen) ? _empty_shadow_rows() :
        DataFrame(sort!(collect(values(chosen));
                        by=row -> (row.issue_time_utc, row.target_time_utc)))
    if !isempty(rows)
        target_observation = Dict{DateTime,Float64}()
        conflicting_targets = Set{DateTime}()
        for row in eachrow(rows)
            ismissing(row.observation_dst_nt) && continue
            observation = Float64(row.observation_dst_nt)
            previous = get(target_observation, row.target_time_utc, nothing)
            if previous !== nothing && previous != observation
                push!(violations, "one prospective target has conflicting observations")
                push!(conflicting_targets, row.target_time_utc)
            end
            target_observation[row.target_time_utc] = observation
        end
        filter!(row -> !(row.target_time_utc in conflicting_targets), rows)
    end
    return (; rows, violations=unique(violations), deployment_rows,
              warmup_rows, unexpected_status_rows)
end

function _claim_source_rows(path::String)
    isfile(path) && !islink(path) || error("live log is missing or not a regular file: $path")
    for suffix in (".append.json", ".retention.json")
        (ispath(path * suffix) || islink(path * suffix)) &&
            error("live log transaction requires recovery: $(path * suffix)")
    end
    archive_dir = joinpath(dirname(path), "archive")
    islink(archive_dir) && error("cold archive directory is a symlink: $archive_dir")
    ispath(archive_dir) && !isdir(archive_dir) &&
        error("cold archive path is not a directory: $archive_dir")
    segments = Dict{Int,String}()
    for name in (isdir(archive_dir) ? readdir(archive_dir) : String[])
        matched = match(r"^live_forecast_log_archive(?:\.(\d+))?\.csv(?:\.manifest\.json)?$", name)
        matched === nothing && continue
        index_text = matched.captures[1]
        index = index_text === nothing ? 0 : tryparse(Int, index_text)
        (index !== nothing && 0 <= index <= 1000 &&
         (index_text === nothing || (index > 0 && string(index) == index_text))) ||
            error("invalid cold archive segment name: $name")
        filename = index == 0 ? "live_forecast_log_archive.csv" :
                               "live_forecast_log_archive.$index.csv"
        segments[index] = joinpath(archive_dir, filename)
    end
    indices = sort!(collect(keys(segments)))
    isempty(indices) || indices == collect(0:(length(indices) - 1)) ||
        error("cold archive segment sequence is incomplete: $archive_dir")

    frames = DataFrame[]
    for index in indices
        segment = segments[index]
        receipt = segment * ".manifest.json"
        all(file -> isfile(file) && !islink(file), (segment, receipt)) ||
            error("cold archive segment or manifest is missing or not a regular file: $segment")
        manifest = JSON3.read(read(receipt, String), Dict{String,Any})
        recorded_rows = get(manifest, "archived_rows", nothing)
        recorded_bytes = get(manifest, "archive_bytes", nothing)
        recorded_index = get(manifest, "segment_index", nothing)
        last_rows = get(manifest, "last_segment_rows", nothing)
        all(value -> value isa Integer && !(value isa Bool),
            (recorded_rows, recorded_bytes, recorded_index, last_rows)) ||
            error("cold archive manifest has malformed counts: $receipt")
        recorded_index == index && recorded_bytes == filesize(segment) &&
            0 < last_rows <= recorded_rows || error("cold archive manifest disagrees with segment: $receipt")
        # Existing receipts hash only the last append, not the whole file. Row/byte counts
        # validate completeness here; they do not authenticate historical CSV contents.
        digest = get(manifest, "last_segment_sha256", nothing)
        digest isa AbstractString && occursin(r"^[0-9a-f]{64}$", digest) ||
            error("cold archive manifest has a malformed append digest: $receipt")
        frame = CSV.read(segment, DataFrame; strict=true)
        nrow(frame) == recorded_rows || error("cold archive row count disagrees with manifest: $receipt")
        push!(frames, frame)
    end
    push!(frames, CSV.read(path, DataFrame; strict=true))
    return length(frames) == 1 ? only(frames) : vcat(frames...; cols=:union)
end

function canonical_shadow_rows(path::AbstractString; lock_timeout_sec::Real=30.0)
    try
        rows = _with_forecast_log_lock(abspath(path); timeout_sec=Float64(lock_timeout_sec)) do
            _claim_source_rows(abspath(path))
        end
        return canonical_shadow_rows(rows)
    catch error
        error isa InterruptException && rethrow()
        empty = canonical_shadow_rows(DataFrame())
        return merge(empty, (violations=["prospective source snapshot unavailable: " *
                                         sprint(showerror, error)],))
    end
end

function _max_consecutive_days(days::Vector{Date})
    ordered = sort!(unique(days))
    isempty(ordered) && return 0
    best = 1
    current = 1
    for index in 2:length(ordered)
        current = ordered[index] == ordered[index - 1] + Day(1) ? current + 1 : 1
        best = max(best, current)
    end
    return best
end

function _bootstrap_mean(values::Vector{Float64}, days::Vector{Date}, reps::Int,
                         seed::Int)
    unique_days = sort!(unique(days))
    length(unique_days) >= 2 || return nothing
    blocks = [values[days .== day] for day in unique_days]
    rng = MersenneTwister(seed)
    draws = Vector{Float64}(undef, reps)
    for rep in 1:reps
        total = 0.0
        count_rows = 0
        for _ in eachindex(blocks)
            block = blocks[rand(rng, eachindex(blocks))]
            total += sum(block)
            count_rows += length(block)
        end
        draws[rep] = total / count_rows
    end
    return (lower=quantile(draws, 0.025), upper=quantile(draws, 0.975))
end

function _minimum_seven_day_coverage(covered::Vector{Float64}, days::Vector{Date})
    observed_days = Set(days)
    values = Float64[]
    for first_day in sort!(unique(days))
        window = [first_day + Day(offset) for offset in 0:6]
        all(day -> day in observed_days, window) || continue
        mask = [first_day <= day <= first_day + Day(6) for day in days]
        push!(values, mean(covered[mask]))
    end
    return isempty(values) ? nothing : minimum(values)
end

"Assign storm targets to conservative independent events. Gaps cannot prove quiet separation."
function group_storm_events(targets::Vector{DateTime}, observations::Vector{Float64})
    length(targets) == length(observations) || throw(ArgumentError("target/observation length mismatch"))
    target_observation = Dict{DateTime,Float64}()
    for (target, observation) in zip(targets, observations)
        isfinite(observation) || throw(ArgumentError("storm grouping needs finite observations"))
        previous = get(target_observation, target, nothing)
        previous === nothing || previous == observation ||
            throw(ArgumentError("one target has conflicting observations"))
        target_observation[target] = observation
    end

    event_by_target = Dict{DateTime,Int}()
    event = 0
    quiet_run = 0
    previous_target = nothing
    for target in sort!(collect(keys(target_observation)))
        observation = target_observation[target]
        contiguous = previous_target !== nothing && target == previous_target + Hour(1)
        contiguous || (quiet_run = 0)
        if observation > QUIET_SEPARATION_THRESHOLD_NT
            quiet_run += 1
        else
            if observation <= STORM_THRESHOLD_NT
                (event == 0 || quiet_run >= QUIET_SEPARATION_HOURS) && (event += 1)
                event_by_target[target] = event
            end
            quiet_run = 0
        end
        previous_target = target
    end
    return event_by_target
end

_rmse(predicted::Vector{Float64}, observed::Vector{Float64}) =
    sqrt(mean((predicted .- observed) .^ 2))

function _event_gain_lower(v24::Vector{Float64}, v22::Vector{Float64},
                           observed::Vector{Float64}, events::Vector{Int},
                           reps::Int, seed::Int)
    unique_events = sort!(unique(events))
    length(unique_events) >= 2 || return nothing
    blocks = [findall(==(event), events) for event in unique_events]
    rng = MersenneTwister(seed)
    draws = Vector{Float64}(undef, reps)
    for rep in 1:reps
        indices = Int[]
        for _ in eachindex(blocks)
            append!(indices, blocks[rand(rng, eachindex(blocks))])
        end
        draws[rep] = _rmse(v22[indices], observed[indices]) -
                     _rmse(v24[indices], observed[indices])
    end
    return quantile(draws, 0.05)
end

function _prospective_status(canonical; bootstrap_reps::Int=DEFAULT_BOOTSTRAP_REPS,
                             bootstrap_seed::Int=DEFAULT_BOOTSTRAP_SEED)
    bootstrap_reps >= 100 || throw(ArgumentError("bootstrap_reps must be at least 100"))
    rows = canonical.rows
    eligible = isempty(rows) ? rows : rows[
        (rows.status .== "ok") .& .!ismissing.(rows.observation_dst_nt) .&
        .!ismissing.(rows.shadow_lo_dst_nt) .& .!ismissing.(rows.shadow_hi_dst_nt), :]

    n = nrow(eligible)
    observed = n == 0 ? Float64[] : Float64.(eligible.observation_dst_nt)
    shadow_lo = n == 0 ? Float64[] : Float64.(eligible.shadow_lo_dst_nt)
    shadow_hi = n == 0 ? Float64[] : Float64.(eligible.shadow_hi_dst_nt)
    covered = n == 0 ? Float64[] :
        Float64.((shadow_lo .<= observed) .& (observed .<= shadow_hi))
    issue_days = n == 0 ? Date[] : Date.(eligible.issue_time_utc)

    cycles = Dict{DateTime,Set{Int}}()
    for row in eachrow(eligible)
        push!(get!(cycles, row.issue_hour_utc, Set{Int}()), row.horizon_hours)
    end
    required_horizons = Set(PUBLIC_HORIZONS)
    complete_cycle_hours = sort!([hour for (hour, horizons) in cycles
                                  if horizons == required_horizons])
    complete_days = Date.(complete_cycle_hours)
    consecutive_days = _max_consecutive_days(complete_days)

    by_step = NamedTuple[]
    step_counts = Dict{Int,Int}()
    step_coverage_gate = true
    for step in SUPPORTED_MODEL_STEPS
        mask = n == 0 ? Bool[] : eligible.model_step_hours .== step
        step_n = count(mask)
        step_counts[step] = step_n
        coverage = step_n == 0 ? nothing : mean(covered[mask])
        step_pass = step_n >= MARGINAL_MINIMUMS.rows_per_supported_step &&
                    coverage !== nothing && coverage >= MARGINAL_MINIMUMS.step_coverage_floor
        step_coverage_gate &= step_pass
        push!(by_step, (model_step_hours=step, n=step_n, coverage_90=coverage,
                        gate_pass=step_pass))
    end

    coverage = n == 0 ? nothing : mean(covered)
    static_width = n == 0 ? Float64[] :
        Float64.(eligible.static_hi_dst_nt .- eligible.static_lo_dst_nt)
    shadow_width = n == 0 ? Float64[] : shadow_hi .- shadow_lo
    width_ratio = n == 0 ? nothing : mean(shadow_width) / mean(static_width)
    score_difference = if n == 0
        Float64[]
    else
        [_interval_score(shadow_lo[index], shadow_hi[index], observed[index]) -
         _interval_score(eligible.static_lo_dst_nt[index],
                         eligible.static_hi_dst_nt[index], observed[index])
         for index in 1:n]
    end
    mean_score_difference = n == 0 ? nothing : mean(score_difference)
    coverage_ci = n == 0 ? nothing :
        _bootstrap_mean(covered, issue_days, bootstrap_reps, bootstrap_seed)
    score_ci = n == 0 ? nothing :
        _bootstrap_mean(score_difference, issue_days, bootstrap_reps, bootstrap_seed + 1)
    min_seven_day_coverage = n == 0 ? nothing :
        _minimum_seven_day_coverage(covered, issue_days)

    sample_gate = consecutive_days >= MARGINAL_MINIMUMS.consecutive_days &&
                  length(complete_cycle_hours) >= MARGINAL_MINIMUMS.complete_issue_cycles &&
                  n >= MARGINAL_MINIMUMS.rows &&
                  minimum(values(step_counts)) >= MARGINAL_MINIMUMS.rows_per_supported_step
    coverage_gate = coverage !== nothing &&
                    MARGINAL_MINIMUMS.pooled_coverage_lo <= coverage <=
                        MARGINAL_MINIMUMS.pooled_coverage_hi &&
                    coverage_ci !== nothing && coverage_ci.lower <= 0.90 <= coverage_ci.upper &&
                    coverage_ci.lower >= MARGINAL_MINIMUMS.bootstrap_lower_floor
    seven_day_gate = min_seven_day_coverage !== nothing &&
                     min_seven_day_coverage >= MARGINAL_MINIMUMS.seven_day_coverage_floor
    width_gate = width_ratio !== nothing &&
                 (width_ratio <= MARGINAL_MINIMUMS.width_ratio_ceiling ||
                  isapprox(width_ratio, MARGINAL_MINIMUMS.width_ratio_ceiling;
                           atol=1e-12, rtol=0))
    score_gate = mean_score_difference !== nothing && mean_score_difference <= 0 &&
                 score_ci !== nothing && score_ci.upper <= 0
    integrity_gate = isempty(canonical.violations) && canonical.unexpected_status_rows == 0
    marginal_ready = sample_gate && coverage_gate && step_coverage_gate &&
                     seven_day_gate && width_gate && score_gate && integrity_gate

    target_observation = Dict{DateTime,Float64}()
    for row in eachrow(rows)
        ismissing(row.observation_dst_nt) && continue
        observation = Float64(row.observation_dst_nt)
        previous = get(target_observation, row.target_time_utc, nothing)
        previous === nothing || previous == observation ||
            error("canonical shadow rows retained conflicting target observations")
        target_observation[row.target_time_utc] = observation
    end
    event_by_target = group_storm_events(collect(keys(target_observation)),
                                         collect(values(target_observation)))
    storm_candidate = n == 0 ? eligible : eligible[observed .<= STORM_THRESHOLD_NT, :]
    comparator_columns = (
        :v22_pred_dst_nt, :v23_pred_dst_nt, :direct_gbm_pred_dst_nt,
        :v21_pred_dst_nt, :persistence_pred_dst_nt, :burton_pred_dst_nt,
        :burton_full_pred_dst_nt, :obrien_pred_dst_nt,
    )
    matched_mask = [all(column -> !ismissing(row[column]), comparator_columns) &&
                    haskey(event_by_target, row.target_time_utc)
                    for row in eachrow(storm_candidate)]
    storm = storm_candidate[matched_mask, :]
    storm_n = nrow(storm)
    storm_observed = storm_n == 0 ? Float64[] : Float64.(storm.observation_dst_nt)
    storm_v24 = storm_n == 0 ? Float64[] : Float64.(storm.point_dst_nt)
    storm_events = storm_n == 0 ? Int[] :
        Int[event_by_target[target] for target in storm.target_time_utc]
    storm_by_step = NamedTuple[]
    storm_step_gate = true
    for step in SUPPORTED_MODEL_STEPS
        mask = storm_n == 0 ? Bool[] : storm.model_step_hours .== step
        step_n = count(mask)
        if step_n == 0
            push!(storm_by_step, (model_step_hours=step, n=0, n_events=0,
                                  rmse_v24_nt=nothing, rmse_v22_nt=nothing,
                                  gain_vs_v22_nt=nothing, gain_lower_95_nt=nothing,
                                  strongest_comparator=nothing,
                                  loss_to_strongest_nt=nothing, gate_pass=false))
            storm_step_gate = false
            continue
        end
        step_obs = storm_observed[mask]
        step_v24 = storm_v24[mask]
        step_v22 = Float64.(storm.v22_pred_dst_nt[mask])
        step_events = storm_events[mask]
        comparator_rmse = Dict(
            "V2.2" => _rmse(step_v22, step_obs),
            "V2.3 shadow" => _rmse(Float64.(storm.v23_pred_dst_nt[mask]), step_obs),
            "direct GBM" => _rmse(Float64.(storm.direct_gbm_pred_dst_nt[mask]), step_obs),
            "V2.1" => _rmse(Float64.(storm.v21_pred_dst_nt[mask]), step_obs),
            "persistence" => _rmse(Float64.(storm.persistence_pred_dst_nt[mask]), step_obs),
            "Burton" => _rmse(Float64.(storm.burton_pred_dst_nt[mask]), step_obs),
            "BurtonFull" => _rmse(Float64.(storm.burton_full_pred_dst_nt[mask]), step_obs),
            "OBrien-McPherron" => _rmse(Float64.(storm.obrien_pred_dst_nt[mask]), step_obs),
        )
        rmse_v24 = _rmse(step_v24, step_obs)
        rmse_v22 = comparator_rmse["V2.2"]
        gain = rmse_v22 - rmse_v24
        lower = _event_gain_lower(step_v24, step_v22, step_obs, step_events,
                                  bootstrap_reps, bootstrap_seed + 100 + step)
        strongest_name = first(sort!(collect(keys(comparator_rmse));
                                     by=name -> (comparator_rmse[name], name)))
        loss = rmse_v24 - comparator_rmse[strongest_name]
        step_event_n = length(unique(step_events))
        step_pass = step_n >= STORM_MINIMUMS.rows_per_supported_step &&
                    step_event_n >= STORM_MINIMUMS.independent_events &&
                    rmse_v24 < rmse_v22 && lower !== nothing && lower > 0 &&
                    loss <= STORM_MINIMUMS.strongest_comparator_loss_ceiling_nt
        storm_step_gate &= step_pass
        push!(storm_by_step, (
            model_step_hours=step, n=step_n, n_events=step_event_n,
            rmse_v24_nt=rmse_v24, rmse_v22_nt=rmse_v22,
            gain_vs_v22_nt=gain, gain_lower_95_nt=lower,
            strongest_comparator=strongest_name,
            loss_to_strongest_nt=loss, gate_pass=step_pass,
        ))
    end
    storm_bias = storm_n == 0 ? nothing : mean(storm_v24 .- storm_observed)
    storm_coverage = storm_n == 0 ? nothing : mean(Float64.(
        (Float64.(storm.shadow_lo_dst_nt) .<= storm_observed) .&
        (storm_observed .<= Float64.(storm.shadow_hi_dst_nt)),
    ))
    storm_sample_gate = length(unique(storm_events)) >= STORM_MINIMUMS.independent_events &&
                        storm_n >= STORM_MINIMUMS.matched_rows
    storm_global_gate = storm_bias !== nothing &&
                        abs(storm_bias) <= STORM_MINIMUMS.absolute_bias_ceiling_nt &&
                        storm_coverage !== nothing &&
                        storm_coverage >= STORM_MINIMUMS.interval_coverage_floor
    storm_ready = storm_sample_gate && storm_step_gate && storm_global_gate &&
                  integrity_gate && marginal_ready

    return (
        generated_utc=string(now(UTC)) * "Z",
        scope="prospective_shadow_only_not_served",
        served_manifest_sha256=SERVED_MANIFEST_SHA256,
        shadow_identity=SHADOW_IDENTITY,
        shadow_config_sha256=SHADOW_CONFIG_SHA256,
        integrity=(
            deployment_rows=canonical.deployment_rows,
            canonical_rows=nrow(rows),
            warmup_rows=canonical.warmup_rows,
            unexpected_status_rows=canonical.unexpected_status_rows,
            violations=unique(canonical.violations),
            gate_pass=integrity_gate,
        ),
        marginal=(
            claim_ready=marginal_ready,
            n_rows=n,
            complete_issue_cycles=length(complete_cycle_hours),
            consecutive_days,
            coverage_90=coverage,
            coverage_bootstrap_95=coverage_ci,
            min_seven_day_coverage_90=min_seven_day_coverage,
            width_ratio,
            mean_interval_score_difference=mean_score_difference,
            score_difference_bootstrap_95=score_ci,
            by_step,
            gates=(sample=sample_gate, coverage=coverage_gate,
                   per_step_coverage=step_coverage_gate,
                   seven_day_coverage=seven_day_gate, width=width_gate,
                   interval_score=score_gate, integrity=integrity_gate),
            minimums=MARGINAL_MINIMUMS,
        ),
        storm=(
            claim_ready=storm_ready,
            n_candidate_rows=nrow(storm_candidate),
            n_matched_rows=storm_n,
            n_independent_events=length(unique(storm_events)),
            absolute_bias_nt=storm_bias === nothing ? nothing : abs(storm_bias),
            interval_coverage_90=storm_coverage,
            by_step=storm_by_step,
            gates=(sample=storm_sample_gate, per_step_skill=storm_step_gate,
                   bias_and_coverage=storm_global_gate,
                   marginal_calibration=marginal_ready, integrity=integrity_gate),
            minimums=STORM_MINIMUMS,
        ),
    )
end

prospective_status(df::DataFrame; kwargs...) =
    _prospective_status(canonical_shadow_rows(df); kwargs...)

prospective_status(path::AbstractString; lock_timeout_sec::Real=30.0, kwargs...) =
    _prospective_status(canonical_shadow_rows(path; lock_timeout_sec); kwargs...)

function _atomic_text(path::AbstractString, content::AbstractString)
    mkpath(dirname(abspath(path)))
    temporary, io = mktemp(dirname(abspath(path)))
    try
        write(io, content)
        close(io)
        mv(temporary, abspath(path); force=true)
    catch
        isopen(io) && close(io)
        rm(temporary; force=true)
        rethrow()
    end
    return abspath(path)
end

_display(value; digits::Int=3) = value === nothing ? "unavailable" :
    value isa AbstractFloat ? string(round(value; digits)) : string(value)
_display(value::NamedTuple; digits::Int=3) =
    hasproperty(value, :lower) && hasproperty(value, :upper) ?
    "[$(_display(value.lower; digits)), $(_display(value.upper; digits))]" : string(value)
_gate(value::Bool) = value ? "PASS" : "NOT MET"

function _report(status)
    marginal = status.marginal
    storm = status.storm
    lines = String[
        "# V2.4e Prospective Live-Claim Audit",
        "",
        "Shadow identity: `$(status.shadow_identity)`  ",
        "Configuration digest: `$(status.shadow_config_sha256)`  ",
        "Served V2.4e manifest: `$(status.served_manifest_sha256)`  ",
        "Generated: $(status.generated_utc)",
        "",
        "Operational V2.4e remains the served product. This audit evaluates a separately identified interval shadow and does not promote it.",
        "",
        "## Marginal calibration",
        "",
        "Decision: **$(marginal.claim_ready ? "CLAIM READY" : "NOT YET JUSTIFIED")**.",
        "",
        "- Matured shadow rows: $(marginal.n_rows) / $(marginal.minimums.rows).",
        "- Complete issue cycles: $(marginal.complete_issue_cycles) / $(marginal.minimums.complete_issue_cycles).",
        "- Consecutive issuance days: $(marginal.consecutive_days) / $(marginal.minimums.consecutive_days).",
        "- Empirical coverage: $(_display(marginal.coverage_90)).",
        "- Day-block 95% coverage interval: $(_display(marginal.coverage_bootstrap_95)).",
        "- Lowest complete seven-day coverage: $(_display(marginal.min_seven_day_coverage_90)).",
        "- Mean width ratio to the static band: $(_display(marginal.width_ratio)).",
        "- Paired mean interval-score difference: $(_display(marginal.mean_interval_score_difference)).",
        "- Day-block 95% interval-score difference: $(_display(marginal.score_difference_bootstrap_95)).",
        "",
        "| Marginal gate | Status |",
        "|---|---:|",
        "| Sample | $(_gate(marginal.gates.sample)) |",
        "| Pooled coverage and day-block interval | $(_gate(marginal.gates.coverage)) |",
        "| Per-step coverage and support | $(_gate(marginal.gates.per_step_coverage)) |",
        "| Seven-day stability | $(_gate(marginal.gates.seven_day_coverage)) |",
        "| Width | $(_gate(marginal.gates.width)) |",
        "| Paired interval score | $(_gate(marginal.gates.interval_score)) |",
        "| Integrity | $(_gate(marginal.gates.integrity)) |",
        "",
        "| Internal step | Matured rows | Coverage | Gate |",
        "|---:|---:|---:|---:|",
    ]
    append!(lines, [
        "| $(row.model_step_hours) | $(row.n) | $(_display(row.coverage_90)) | $(_gate(row.gate_pass)) |"
        for row in marginal.by_step
    ])
    append!(lines, [
        "",
        "## Storm skill",
        "",
        "Decision: **$(storm.claim_ready ? "CLAIM READY" : "BLOCKED")**.",
        "",
        "- Same-row matched storm forecasts: $(storm.n_matched_rows) / $(storm.minimums.matched_rows).",
        "- Independent events: $(storm.n_independent_events) / $(storm.minimums.independent_events).",
        "- Absolute bias (nT): $(_display(storm.absolute_bias_nt)).",
        "- Shadow interval coverage: $(_display(storm.interval_coverage_90)).",
        "",
        "| Storm gate | Status |",
        "|---|---:|",
        "| Event and row sample | $(_gate(storm.gates.sample)) |",
        "| Per-step skill and comparator bound | $(_gate(storm.gates.per_step_skill)) |",
        "| Bias and interval coverage | $(_gate(storm.gates.bias_and_coverage)) |",
        "| Marginal calibration | $(_gate(storm.gates.marginal_calibration)) |",
        "| Integrity | $(_gate(storm.gates.integrity)) |",
        "",
        "| Internal step | Rows | Events | V2.4e RMSE | V2.2 RMSE | Gain lower 95% | Strongest comparator | Loss to strongest | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    append!(lines, [
        "| $(row.model_step_hours) | $(row.n) | $(row.n_events) | " *
        "$(_display(row.rmse_v24_nt)) | $(_display(row.rmse_v22_nt)) | " *
        "$(_display(row.gain_lower_95_nt)) | $(something(row.strongest_comparator, "unavailable")) | " *
        "$(_display(row.loss_to_strongest_nt)) | $(_gate(row.gate_pass)) |"
        for row in storm.by_step
    ])
    append!(lines, [
        "",
        "Retrospective storm replay is not counted as prospective live evidence.",
        "",
        "## Integrity",
        "",
        "Decision: **$(_gate(status.integrity.gate_pass))**.",
        "",
        "- Deployment rows: $(status.integrity.deployment_rows).",
        "- Canonical rows: $(status.integrity.canonical_rows).",
        "- Warm-up rows: $(status.integrity.warmup_rows).",
        "- Unexpected-status rows: $(status.integrity.unexpected_status_rows).",
    ])
    if !isempty(status.integrity.violations)
        push!(lines, "", "### Failures", "")
        append!(lines, ["- $violation" for violation in status.integrity.violations])
    end
    return join(lines, "\n") * "\n"
end

function run_claim_audit(log_path::AbstractString, json_path::AbstractString,
                         report_path::AbstractString; kwargs...)
    status = prospective_status(log_path; kwargs...)
    _atomic_text(json_path, JSON3.write(status))
    _atomic_text(report_path, _report(status))
    return status
end

function _cli(args)
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    monitor_dir = joinpath(package_root, "var", "monitor")
    options = Dict{String,String}()
    for arg in args
        pieces = split(startswith(arg, "--") ? arg[3:end] : arg, "="; limit=2)
        length(pieces) == 2 || throw(ArgumentError("expected --name=value, got $arg"))
        options[pieces[1]] = pieces[2]
    end
    log_path = get(options, "log", joinpath(monitor_dir, "live_forecast_log.csv"))
    json_path = get(options, "json", joinpath(monitor_dir, "v2_4_live_claim_status.json"))
    report_path = get(options, "report", joinpath(monitor_dir, "v2_4_live_claim_status.md"))
    reps = parse(Int, get(options, "bootstrap-reps", string(DEFAULT_BOOTSTRAP_REPS)))
    status = run_claim_audit(log_path, json_path, report_path; bootstrap_reps=reps)
    println("V2.4e prospective live-claim audit")
    println("  marginal claim ready: $(status.marginal.claim_ready)")
    println("  storm-skill claim ready: $(status.storm.claim_ready)")
    println("  report: $report_path")
    return isempty(status.integrity.violations)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(V24LiveClaimAudit._cli(ARGS) ? 0 : 1)
end
