#!/usr/bin/env julia

module V24IntervalUpgrade

using CSV, DataFrames, Dates, JSON3, Random, SHA, Statistics
import SolarSINDy
include("v2_4_live_calibration.jl")
using .V24LiveCalibration: canonical_live_rows, conformal_upper, empirical_lower,
                           empirical_upper, interval_score
import .V24LiveCalibration: _parse_time, _float

export CANDIDATES, anchor_witnesses, candidate_interval, replay_upgrade,
       bootstrap_days, common_rows, summarize_upgrade, select_candidate, run_upgrade

const CANDIDATES = ("L24", "L48", "S48", "S96", "T48", "T96")
const CONTROLS = ("static", "issued_A3")
const SPLIT_UTC = DateTime(2026, 9, 1)
const EXPECTED_INPUT_SHA = "582df81bbf4b2b4aafbe4397b90035e7f1843d3f9019bd246325c28298a9c82b"
const A3_ID = "v2.4e-cal-shadow-A3+median24+width1h1.50+widthOther1.20+warm30"
const A3_SHA = "31d67e5077ae6fe69cee133fa07dceec3aa639903a17861d179d3877ed0c21af"
const STEPS = (1, 2, 3, 4, 6, 7)

_tail(values, n) = @view values[max(1, length(values) - n + 1):end]
_optional(row, name) = hasproperty(row, name) ? getproperty(row, name) : missing
_text(value) = ismissing(value) ? "" : string(value)
_finite(value, label) = _float(value, label)
_diagnostic_number(value) = ismissing(value) ? NaN : Float64(value)

"Unique observed anchors, timestamped by the issue that demonstrates their availability."
function anchor_witnesses(input::DataFrame)
    seen = Dict{DateTime,Tuple{DateTime,Float64}}()
    for row in eachrow(input)
        issue = _parse_time(row.issue_time_utc)
        target = _parse_time(row.latest_dst_time_utc)
        target <= issue || throw(ArgumentError("anchor witness is later than its issue"))
        value = _finite(row.latest_dst_nt, "anchor witness Dst")
        witness = (target, value)
        haskey(seen, issue) && seen[issue] != witness &&
            throw(ArgumentError("one issue carries conflicting anchor witnesses"))
        seen[issue] = witness
    end
    return [(available_utc=issue, target_utc=seen[issue][1], dst_nt=seen[issue][2])
            for issue in sort!(collect(keys(seen)))]
end

"One fixed-grid interval; unavailable history returns NaN endpoints, never a static substitute."
function candidate_interval(name::AbstractString, point::Real, lo::Real, hi::Real,
                            residuals::AbstractVector, scores::AbstractVector;
                            warmup::Int=30)
    name in CANDIDATES || throw(ArgumentError("unknown interval candidate: $name"))
    warmup >= 1 || throw(ArgumentError("warmup must be positive"))
    p, l, h = _finite.((point, lo, hi), ("point", "lower endpoint", "upper endpoint"))
    l < p < h || throw(ArgumentError("static half-widths must both be positive"))
    left, right = p - l, h - p
    all(isfinite, (left, right)) || throw(ArgumentError("static half-width overflow"))
    history = Float64[_finite(value, "residual") for value in residuals]
    standardized = Float64[_finite(value, "standardized score") for value in scores]
    all(>=(0), standardized) || throw(ArgumentError("standardized scores must be nonnegative"))
    available = length(history) >= warmup &&
                (!startswith(name, "S") || length(standardized) >= warmup)
    location = length(history) >= warmup ? median(_tail(history, 24)) : NaN
    available || return (; available=false, lo=NaN, hi=NaN, location)
    window = parse(Int, name[2:end])
    lower, upper = if startswith(name, "L")
        translated = SolarSINDy._v24_calibration_shadow_interval(
            p, l, h, history; window, warmup, width_scale=1.25)
        location = translated.location
        (translated.lo, translated.hi)
    elseif startswith(name, "S")
        scale = conformal_upper(_tail(standardized, window), 0.90)
        (p + location - scale * left, p + location + scale * right)
    else
        sample = _tail(history, window)
        (p + empirical_lower(sample, 0.05), p + empirical_upper(sample, 0.95))
    end
    all(isfinite, (lower, upper, location)) && lower <= upper ||
        throw(ArgumentError("candidate interval is non-finite or reversed"))
    return (; available=true, lo=Float64(lower), hi=Float64(upper), location)
end

function _context(row)
    return (
        a3_id=_text(_optional(row, :v24_cal_shadow_model_version)),
        a3_sha=_text(_optional(row, :v24_cal_shadow_config_sha256)),
        a3_status=_text(_optional(row, :v24_cal_shadow_status)),
        a3_lo=_optional(row, :v24_cal_shadow_ci05_nt),
        a3_hi=_optional(row, :v24_cal_shadow_ci95_nt),
        rate=_optional(row, :dst_delta_1h_nt),
        coupling=_optional(row, :VBsouth_mvm),
        driver_gap=_optional(row, :driver_data_gap),
    )
end

function _context_lookup(input)
    result = Dict{Tuple{DateTime,DateTime},NamedTuple}()
    for row in eachrow(input)
        _text(row.sub_hourly_model_version) == V24LiveCalibration.EXACT_V24_IDENTITY || continue
        key = (_parse_time(row.issue_time_utc), _parse_time(row.target_time_utc))
        context = _context(row)
        haskey(result, key) && !isequal(result[key], context) &&
            throw(ArgumentError("duplicate forecast has conflicting A3 or driver context"))
        result[key] = context
    end
    return result
end

function _issued_a3(context)
    context.a3_status == "ok" || return (available=false, lo=NaN, hi=NaN, location=NaN)
    context.a3_id == A3_ID && context.a3_sha == A3_SHA ||
        throw(ArgumentError("issued A3 control identity or configuration differs"))
    lower = _finite(context.a3_lo, "issued A3 lower endpoint")
    upper = _finite(context.a3_hi, "issued A3 upper endpoint")
    lower < upper || throw(ArgumentError("issued A3 control interval is reversed or empty"))
    return (available=true, lo=lower, hi=upper, location=NaN)
end

"Replay each issue from logged anchor witnesses; final scoring outcomes never enter histories."
function replay_upgrade(input::DataFrame; warmup::Int=30)
    warmup >= 1 || throw(ArgumentError("warmup must be positive"))
    rows = canonical_live_rows(input; require_matured=false)
    witnesses = anchor_witnesses(input)
    contexts = _context_lookup(input)
    targets = Dict{DateTime,Float64}()
    for row in eachrow(rows)
        ismissing(row.observation_dst_nt) && continue
        haskey(targets, row.target_time_utc) && targets[row.target_time_utc] != row.observation_dst_nt &&
            throw(ArgumentError("one target carries conflicting final scoring observations"))
        targets[row.target_time_utc] = row.observation_dst_nt
    end
    known = Dict{DateTime,Tuple{Float64,DateTime}}()
    prior = Dict(step => Int[] for step in STEPS)
    issued_location = fill(NaN, nrow(rows))
    output = NamedTuple[]
    history_rows = NamedTuple[]
    cursor = 1
    for cycle in groupby(rows, :issue_time_utc; sort=true)
        issue = first(cycle.issue_time_utc)
        cutoffs = unique(cycle.latest_dst_time_utc)
        length(cutoffs) == 1 || throw(ArgumentError("one cycle carries inconsistent Dst anchors"))
        cutoff = only(cutoffs)
        while cursor <= length(witnesses) && witnesses[cursor].available_utc <= issue
            witness = witnesses[cursor]
            known[witness.target_utc] = (witness.dst_nt, witness.available_utc)
            cursor += 1
        end
        current = Int[]
        for index in parentindices(cycle)[1]
            row = rows[index, :]
            step = row.model_step_hours
            step in STEPS || throw(ArgumentError("unsupported model step: $step"))
            matured = filter(prior[step]) do earlier
                target = rows.target_time_utc[earlier]
                target <= cutoff && haskey(known, target)
            end
            sort!(matured; by=i -> (rows.target_time_utc[i], rows.issue_time_utc[i], rows.row_key[i]))
            residuals = Float64[]
            scores = Float64[]
            receipt_times = DateTime[]
            for earlier in matured
                old = rows[earlier, :]
                value, seen = known[old.target_time_utc]
                residual = _finite(value - old.point_dst_nt, "witness residual")
                push!(residuals, residual)
                push!(receipt_times, seen)
                if isfinite(issued_location[earlier])
                    error = _finite(residual - issued_location[earlier], "translated residual")
                    half = error >= 0 ? old.static_hi_dst_nt - old.point_dst_nt :
                                       old.point_dst_nt - old.static_lo_dst_nt
                    isfinite(half) && half > 0 ||
                        throw(ArgumentError("historical standardization half-width is invalid"))
                    push!(scores, _finite(abs(error) / half, "standardized error"))
                end
            end
            issued_location[index] = length(residuals) >= warmup ? median(_tail(residuals, 24)) : NaN
            context = contexts[(row.issue_time_utc, row.target_time_utc)]
            a3 = _issued_a3(context)
            observation = ismissing(row.observation_dst_nt) ? NaN : row.observation_dst_nt
            static_score = isfinite(observation) ? interval_score(
                row.static_lo_dst_nt, row.static_hi_dst_nt, observation) : NaN
            push!(history_rows, (
                row_key=row.row_key, issue_time_utc=issue, model_step_hours=step,
                anchor_utc=cutoff, prior_rows=length(prior[step]),
                residual_n=length(residuals), score_n=length(scores),
                newest_witness_utc=isempty(receipt_times) ? "" : string(maximum(receipt_times)) * "Z",
                newest_history_target_utc=isempty(matured) ? "" : string(maximum(rows.target_time_utc[matured])) * "Z",
            ))
            for name in (CANDIDATES..., CONTROLS...)
                band = name == "static" ?
                    (available=true, lo=row.static_lo_dst_nt, hi=row.static_hi_dst_nt, location=0.0) :
                    name == "issued_A3" ? a3 : candidate_interval(name, row.point_dst_nt,
                        row.static_lo_dst_nt, row.static_hi_dst_nt, residuals, scores; warmup)
                scored = band.available && isfinite(observation)
                score = scored ? _finite(interval_score(band.lo, band.hi, observation), "interval score") : NaN
                band.available && !isfinite(band.hi - band.lo) &&
                    throw(ArgumentError("candidate width overflow"))
                push!(output, (
                    candidate=name, row_key=row.row_key, issue_time_utc=issue,
                    target_time_utc=row.target_time_utc, anchor_utc=cutoff,
                    model_step_hours=step, latest_dst_nt=row.latest_dst_nt,
                    anchor_lag_h=Float64((floor(issue, Hour) - cutoff) / Hour(1)),
                    rate_nt_h=_diagnostic_number(context.rate), coupling_mvm=_diagnostic_number(context.coupling),
                    driver_gap=ismissing(context.driver_gap) ? "unknown" : string(context.driver_gap), point_dst_nt=row.point_dst_nt,
                    observation_dst_nt=observation, available=band.available,
                    scored, lower_dst_nt=band.lo, upper_dst_nt=band.hi,
                    width_nt=band.hi - band.lo, location_nt=band.location,
                    covered=scored && band.lo <= observation <= band.hi,
                    lower_miss=scored && observation < band.lo,
                    upper_miss=scored && observation > band.hi,
                    interval_score_nt=score, static_score_nt=static_score,
                    static_width_nt=row.static_hi_dst_nt - row.static_lo_dst_nt,
                ))
            end
            push!(current, index)
        end
        # None of this issue's target outcomes can affect another lead in the same issue.
        for index in current
            push!(prior[rows.model_step_hours[index]], index)
        end
    end
    return (; rows, witnesses, replay=DataFrame(output), histories=DataFrame(history_rows))
end

"Paired UTC-day resampling, weighted by the number of forecast rows in each sampled day."
function bootstrap_days(values::AbstractVector, days::AbstractVector{Date};
                        reps::Int=2000, seed::Int=20260908)
    length(values) == length(days) || throw(DimensionMismatch("values and days differ in length"))
    reps >= 100 || throw(ArgumentError("at least 100 bootstrap resamples are required"))
    seed >= 0 || throw(ArgumentError("bootstrap seed must be nonnegative"))
    v = Float64[_finite(value, "bootstrap value") for value in values]
    labels = sort!(unique(days))
    length(labels) >= 2 || return (lower=NaN, upper=NaN, draws=Float64[])
    blocks = [v[days .== day] for day in labels]
    totals = [_finite(sum(block), "bootstrap block total") for block in blocks]
    counts = length.(blocks)
    rng = MersenneTwister(seed)
    draws = Vector{Float64}(undef, reps)
    for rep in 1:reps
        total = 0.0
        n = 0
        for _ in eachindex(labels)
            index = rand(rng, eachindex(labels))
            total += totals[index]
            n += counts[index]
        end
        draws[rep] = _finite(total / n, "bootstrap mean")
    end
    return (lower=quantile(draws, 0.025), upper=quantile(draws, 0.975), draws)
end

"Return identical scored row keys for the entire fixed grid and both controls."
function common_rows(replay::DataFrame; start::DateTime, stop::DateTime)
    start < stop || throw(ArgumentError("evaluation start must precede stop"))
    isempty(replay) && return copy(replay)
    selected = replay[(start .<= replay.issue_time_utc) .& (replay.issue_time_utc .< stop), :]
    expected = Set((CANDIDATES..., CONTROLS...))
    keep = String[]
    for group in groupby(selected, :row_key)
        nrow(group) == length(expected) && Set(group.candidate) == expected ||
            throw(ArgumentError("forecast row lacks the fixed candidate/control set or has duplicates"))
        for column in (:issue_time_utc, :target_time_utc, :anchor_utc, :model_step_hours,
                       :point_dst_nt, :observation_dst_nt, :static_width_nt, :static_score_nt)
            all(value -> isequal(value, first(group[!, column])), group[!, column]) ||
                throw(ArgumentError("candidate rows disagree on paired field $column"))
        end
        all(group.scored) && push!(keep, first(group.row_key))
    end
    return sort!(selected[in.(selected.row_key, Ref(Set(keep))), :], [:candidate, :issue_time_utc, :row_key])
end

function _metrics(group)
    nrow(group) > 0 || return (n=0, coverage=NaN, lower_misses=0, upper_misses=0,
        mean_width_nt=NaN, mean_interval_score_nt=NaN, width_ratio=NaN,
        score_difference_nt=NaN, point_rmse_nt=NaN)
    lo = Float64.(group.lower_dst_nt)
    hi = Float64.(group.upper_dst_nt)
    y = Float64.(group.observation_dst_nt)
    all(isfinite, [lo; hi; y; group.static_width_nt; group.static_score_nt]) ||
        throw(ArgumentError("scored metric inputs are non-finite"))
    all(lo .<= hi) && all(>(0), group.static_width_nt) ||
        throw(ArgumentError("scored metric widths are invalid"))
    scores = interval_score.(lo, hi, y)
    width = _finite(mean(hi .- lo), "mean interval width")
    score = _finite(mean(scores), "mean interval score")
    delta = _finite(mean(scores .- group.static_score_nt), "paired score difference")
    return (n=nrow(group), coverage=mean(lo .<= y .<= hi),
        lower_misses=count(y .< lo), upper_misses=count(y .> hi),
        mean_width_nt=width, mean_interval_score_nt=score,
        width_ratio=width / mean(group.static_width_nt), score_difference_nt=delta,
        point_rmse_nt=SolarSINDy.rmse(group.point_dst_nt, y))
end

function _complete_days(group, start, stop)
    first_day = Date(ceil(start, Day))
    end_day = Date(floor(stop, Day))
    present = Set(Date.(group.issue_time_utc))
    return [day for day in first_day:Day(1):(end_day - Day(1)) if day in present]
end

function _seven_day_minimum(group, full_days)
    days = Date.(group.issue_time_utc)
    full = Set(full_days)
    values = Float64[]
    for start in full_days
        window = [start + Day(k) for k in 0:6]
        all(in(full), window) || continue
        selected = group[in.(days, Ref(Set(window))), :]
        push!(values, _metrics(selected).coverage)
    end
    return isempty(values) ? NaN : minimum(values)
end

function _advancement_reasons(metrics, steps, complete_days, minimum_week, coverage_ci, score_ci)
    reasons = String[]
    metrics.n > 0 || push!(reasons, "no common scored rows")
    complete_days >= 7 || push!(reasons, "fewer than seven complete UTC days")
    length(steps) == length(STEPS) && all(item -> item.n >= 40 && item.coverage >= 0.85, steps) ||
        push!(reasons, "step support or coverage below threshold")
    0.88 <= metrics.coverage <= 0.92 || push!(reasons, "pooled coverage outside 0.88–0.92")
    minimum_week >= 0.80 || push!(reasons, "seven-day coverage below threshold or unavailable")
    # Mean-of-width arithmetic can differ by a few ulps for a uniform 1.25 scaling.
    metrics.width_ratio <= 1.25 + 32eps(1.25) || push!(reasons, "width ratio exceeds 1.25")
    metrics.score_difference_nt <= 0 || push!(reasons, "mean interval score exceeds static")
    coverage_ci.lower >= 0.85 && coverage_ci.lower <= 0.90 <= coverage_ci.upper ||
        push!(reasons, "coverage bootstrap criterion not met")
    score_ci.upper <= 0 || push!(reasons, "paired score bootstrap upper bound is positive or unavailable")
    return reasons
end

"Evaluate the frozen advancement rules; this never authorizes deployment."
function summarize_upgrade(replay::DataFrame; start::DateTime, stop::DateTime,
                           reps::Int=2000, seed::Int=20260908)
    common = common_rows(replay; start, stop)
    summary = NamedTuple[]
    by_step = NamedTuple[]
    draws = NamedTuple[]
    for name in (CANDIDATES..., CONTROLS...)
        group = isempty(common) ? common : common[common.candidate .== name, :]
        metrics = _metrics(group)
        step_metrics = NamedTuple[]
        for step in STEPS
            subset = isempty(group) ? group : group[group.model_step_hours .== step, :]
            item = (candidate=name, model_step_hours=step, _metrics(subset)...)
            push!(step_metrics, item)
            push!(by_step, item)
        end
        days = isempty(group) ? Date[] : Date.(group.issue_time_utc)
        hits = isempty(group) ? Float64[] : Float64.(group.lower_dst_nt .<= group.observation_dst_nt .<= group.upper_dst_nt)
        difference = isempty(group) ? Float64[] : interval_score.(group.lower_dst_nt,
            group.upper_dst_nt, group.observation_dst_nt) .- group.static_score_nt
        coverage_ci = bootstrap_days(hits, days; reps, seed)
        score_ci = bootstrap_days(difference, days; reps, seed)
        full_days = isempty(group) ? Date[] : _complete_days(group, start, stop)
        minimum_week = isempty(group) ? NaN : _seven_day_minimum(group, full_days)
        reasons = _advancement_reasons(metrics, step_metrics, length(full_days),
                                       minimum_week, coverage_ci, score_ci)
        for index in eachindex(coverage_ci.draws)
            push!(draws, (candidate=name, draw=index, coverage=coverage_ci.draws[index],
                          score_difference_nt=score_ci.draws[index]))
        end
        available = isempty(replay) ? 0 : count((replay.candidate .== name) .& replay.scored .&
            (start .<= replay.issue_time_utc) .& (replay.issue_time_utc .< stop))
        push!(summary, (candidate=name, own_scored_rows=available, metrics...,
            complete_utc_days=length(full_days), minimum_seven_day_coverage=minimum_week,
            coverage_ci_lower=coverage_ci.lower, coverage_ci_upper=coverage_ci.upper,
            score_ci_lower_nt=score_ci.lower, score_ci_upper_nt=score_ci.upper,
            advances=name in CANDIDATES && isempty(reasons),
            reason=isempty(reasons) ? "development criteria met" : join(reasons, "; ")))
    end
    return (; common, summary=DataFrame(summary), by_step=DataFrame(by_step), bootstrap=DataFrame(draws))
end

function select_candidate(summary::DataFrame)
    isempty(summary) && return nothing
    passing = [row for row in eachrow(summary) if row.candidate in CANDIDATES && row.advances]
    isempty(passing) && return nothing
    sort!(passing; by=row -> (row.mean_interval_score_nt, row.mean_width_nt,
                             findfirst(==(row.candidate), CANDIDATES)))
    return String(first(passing).candidate)
end

function _diagnostic_bin(row, feature)
    feature == :model_step_hours && return string(row.model_step_hours)
    feature == :anchor_lag_h && return string(row.anchor_lag_h)
    feature == :issue_day && return string(Date(row.issue_time_utc))
    feature == :driver_gap && return row.driver_gap
    value = row[feature]
    isfinite(value) || return "unknown"
    if feature == :latest_dst_nt
        return value <= -50 ? "Dst<=-50" : value <= -30 ? "-50<Dst<=-30" : "Dst>-30"
    elseif feature == :rate_nt_h
        return value < -5 ? "rate<-5" : value > 5 ? "rate>5" : "-5<=rate<=5"
    end
    return value <= 0 ? "coupling<=0" : value < 2 ? "0<coupling<2" : "coupling>=2"
end

"Paired issued-control diagnostics; no candidate is tuned from these tables."
function development_diagnostics(replay; start, stop)
    controls = replay[in.(replay.candidate, Ref(Set(CONTROLS))) .& replay.scored .&
        (start .<= replay.issue_time_utc) .& (replay.issue_time_utc .< stop), :]
    paired_keys = Set(first(group.row_key) for group in groupby(controls, :row_key)
                      if nrow(group) == 2 && Set(group.candidate) == Set(CONTROLS))
    paired = controls[in.(controls.row_key, Ref(paired_keys)), :]
    records = NamedTuple[]
    for feature in (:model_step_hours, :anchor_lag_h, :issue_day, :latest_dst_nt,
                    :rate_nt_h, :coupling_mvm, :driver_gap)
        bins = [_diagnostic_bin(row, feature) for row in eachrow(paired)]
        for label in sort!(unique(bins)), candidate in CONTROLS
            group = paired[(bins .== label) .& (paired.candidate .== candidate), :]
            errors = group.observation_dst_nt .- group.point_dst_nt
            push!(records, (feature=string(feature), stratum=label, candidate,
                _metrics(group)..., bias_nt=mean(errors), mae_nt=mean(abs.(errors))))
        end
    end
    return DataFrame(records)
end

function witness_support(result)
    records = NamedTuple[]
    by_target = Dict{DateTime,Vector{NamedTuple}}()
    for witness in result.witnesses
        push!(get!(by_target, witness.target_utc, NamedTuple[]), witness)
    end
    for row in eachrow(result.rows)
        available = get(by_target, row.target_time_utc, NamedTuple[])
        value = isempty(available) ? NaN : last(available).dst_nt
        observed = ismissing(row.observation_dst_nt) ? NaN : row.observation_dst_nt
        push!(records, (row_key=row.row_key, target_utc=row.target_time_utc,
            witnessed=!isempty(available), witness_n=length(available),
            distinct_values=length(unique(w.dst_nt for w in available)),
            first_available_utc=isempty(available) ? "" : string(first(available).available_utc) * "Z",
            last_available_utc=isempty(available) ? "" : string(last(available).available_utc) * "Z",
            last_witness_dst_nt=value, final_scoring_dst_nt=observed,
            last_witness_minus_final_nt=value - observed))
    end
    return DataFrame(records)
end

"Run exactly one frozen partition evaluation into a new directory. Existing results are never overwritten."
function run_upgrade(input_path::AbstractString, output_dir::AbstractString;
                     stage::Symbol, expected_sha::AbstractString=EXPECTED_INPUT_SHA)
    stage in (:development, :validation) || throw(ArgumentError("stage must be development or validation"))
    ispath(output_dir) && throw(ArgumentError("output already exists: $output_dir"))
    digest = bytes2hex(sha256(read(input_path)))
    digest == expected_sha || throw(ArgumentError("frozen development input digest differs"))
    raw = CSV.read(input_path, DataFrame; strict=true)
    times = _parse_time.(raw.issue_time_utc)
    if stage == :development
        raw = raw[times .< SPLIT_UTC, :]
    end
    result = replay_upgrade(raw)
    isempty(result.rows) && throw(ArgumentError("no exact-identity forecast rows"))
    start = stage == :development ? minimum(result.rows.issue_time_utc) : SPLIT_UTC
    stop = stage == :development ? SPLIT_UTC : maximum(result.rows.issue_time_utc) + Millisecond(1)
    tables = summarize_upgrade(result.replay; start, stop)
    selected = stage == :validation ? select_candidate(tables.summary) : nothing
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "replay.csv"), result.replay)
    CSV.write(joinpath(output_dir, "histories.csv"), result.histories)
    CSV.write(joinpath(output_dir, "witnesses.csv"), DataFrame(result.witnesses))
    CSV.write(joinpath(output_dir, "summary.csv"), tables.summary)
    CSV.write(joinpath(output_dir, "by_step.csv"), tables.by_step)
    CSV.write(joinpath(output_dir, "bootstrap.csv"), tables.bootstrap)
    CSV.write(joinpath(output_dir, "witness_support.csv"), witness_support(result))
    if stage == :development
        CSV.write(joinpath(output_dir, "diagnostics.csv"), development_diagnostics(result.replay; start, stop))
    end
    files = sort(readdir(output_dir))
    hashes = Dict(file => bytes2hex(sha256(read(joinpath(output_dir, file)))) for file in files)
    receipt = (stage=string(stage), input_sha256=digest, split_utc=string(SPLIT_UTC) * "Z",
        candidates=CANDIDATES, controls=CONTROLS, selected, prospective=false,
        deployment_authorized=false, source_sha256=bytes2hex(sha256(read(@__FILE__))),
        calibration_source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__, "v2_4_live_calibration.jl")))),
        files=hashes, generated_utc=string(now(UTC)) * "Z")
    write(joinpath(output_dir, "receipt.json"), JSON3.write(receipt))
    return (; result, tables, selected, receipt)
end

function main(args)
    length(args) == 3 || throw(ArgumentError("Usage: v2_4_interval_upgrade.jl development|validation INPUT.csv NEW_OUTPUT_DIR"))
    result = run_upgrade(args[2], args[3]; stage=Symbol(args[1]))
    println("Interval study ", args[1], ": ", nrow(result.result.rows), " exact-identity forecasts")
    println("Selected candidate: ", something(result.selected, "none"), "; deployment is not authorized")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    V24IntervalUpgrade.main(ARGS)
end
