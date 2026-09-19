#!/usr/bin/env julia

module V24LiveCalibration

using CSV
using DataFrames
using Dates
using SHA
using Statistics
import SolarSINDy

export CandidateSpec, EXACT_V24_IDENTITY, EXACT_V24_MANIFEST_SHA256,
       candidate_specs, canonical_live_rows,
       conformal_upper, empirical_lower, empirical_upper, interval_score,
       replay_candidates, summarize_candidates, select_shadow_candidate, run_study

const EXACT_V24_IDENTITY = "v2.4+sindy20x11+superlearner10floor+conformal"
const EXACT_V24_MANIFEST_SHA256 =
    "057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede"
const TARGET_COVERAGE = 0.90
const DEFAULT_WARMUP = 30

struct CandidateSpec
    family::String
    name::String
    method::Symbol
    window::Int
    gamma::Float64
    simplicity::Int
    width_scale::Float64
    width_scale_by_step::Dict{Int,Float64}
end

CandidateSpec(family::String, name::String, method::Symbol, window::Int,
              gamma::Float64, simplicity::Int) =
    CandidateSpec(family, name, method, window, gamma, simplicity, 1.0,
                  Dict{Int,Float64}())

CandidateSpec(family::String, name::String, method::Symbol, window::Int,
              gamma::Float64, simplicity::Int, width_scale::Real) =
    CandidateSpec(family, name, method, window, gamma, simplicity,
                  Float64(width_scale), Dict{Int,Float64}())

candidate_specs() = CandidateSpec[
    CandidateSpec("C1", "C1-aci-g0.01-w500", :aci, 500, 0.01, 1),
    CandidateSpec("C1", "C1-aci-g0.03-w500", :aci, 500, 0.03, 1),
    CandidateSpec("C2", "C2-symmetric-w48", :symmetric, 48, 0.0, 2),
    CandidateSpec("C2", "C2-symmetric-w96", :symmetric, 96, 0.0, 2),
    CandidateSpec("C3", "C3-location-scale-w48", :location_scale, 48, 0.0, 3),
    CandidateSpec("C3", "C3-location-scale-w96", :location_scale, 96, 0.0, 3),
    CandidateSpec("C4", "C4-equal-tail-w48", :equal_tail, 48, 0.0, 4),
    CandidateSpec("C4", "C4-equal-tail-w96", :equal_tail, 96, 0.0, 4),
    CandidateSpec("C5", "C5-static-transport-w24-s1.50", :translated_static,
                  24, 0.0, 5, 1.50),
    CandidateSpec("C6", "C6-static-transport-w24-s1.25", :translated_static,
                  24, 0.0, 6, 1.25),
    CandidateSpec("C7", "C7-static-transport-w24-s1.50-others1.20", :translated_static,
                  24, 0.0, 7, 1.20, Dict(1 => 1.50)),
]

_candidate_width_scale(spec::CandidateSpec, step::Int) =
    get(spec.width_scale_by_step, step, spec.width_scale)

function _finite(values::AbstractVector{<:Real})
    out = Float64[]
    for value in values
        converted = Float64(value)
        isfinite(converted) && push!(out, converted)
    end
    isempty(out) && throw(ArgumentError("order statistic needs at least one finite value"))
    return sort!(out)
end

"Finite-sample upper order statistic at probability `p`, with endpoint clipping."
function conformal_upper(values::AbstractVector{<:Real}, p::Real)
    isfinite(p) && 0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))
    sorted = _finite(values)
    k = clamp(ceil(Int, (length(sorted) + 1) * Float64(p)), 1, length(sorted))
    return sorted[k]
end

"Lower empirical order statistic used by the two-sided signed-residual interval."
function empirical_lower(values::AbstractVector{<:Real}, p::Real)
    isfinite(p) && 0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))
    sorted = _finite(values)
    k = clamp(floor(Int, (length(sorted) + 1) * Float64(p)), 1, length(sorted))
    return sorted[k]
end

"Upper empirical order statistic used by the two-sided signed-residual interval."
function empirical_upper(values::AbstractVector{<:Real}, p::Real)
    isfinite(p) && 0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))
    sorted = _finite(values)
    k = clamp(ceil(Int, (length(sorted) + 1) * Float64(p)), 1, length(sorted))
    return sorted[k]
end

"Negatively oriented central prediction-interval score. Lower is better."
function interval_score(lo::Real, hi::Real, observed::Real; alpha::Real=0.10)
    values = Float64.((lo, hi, observed, alpha))
    all(isfinite, values) || throw(ArgumentError("interval score inputs must be finite"))
    l, h, y, a = values
    l <= h || throw(ArgumentError("interval lower endpoint exceeds upper endpoint"))
    0 < a < 1 || throw(ArgumentError("alpha must lie in (0, 1)"))
    return (h - l) + (y < l ? (2 / a) * (l - y) : 0.0) +
           (y > h ? (2 / a) * (y - h) : 0.0)
end

_parse_time(value::DateTime) = value
function _parse_time(value)
    ismissing(value) && throw(ArgumentError("required UTC timestamp is missing"))
    text = strip(String(value))
    endswith(text, "Z") && (text = text[1:end-1])
    try
        return DateTime(text)
    catch
        throw(ArgumentError("invalid UTC timestamp: $(repr(value))"))
    end
end

function _float(value, label::AbstractString)
    ismissing(value) && throw(ArgumentError("$label is missing"))
    value isa Bool && throw(ArgumentError("$label is not numeric"))
    converted = value isa Real ? Float64(value) : tryparse(Float64, strip(String(value)))
    converted === nothing && throw(ArgumentError("$label is not numeric"))
    isfinite(converted) || throw(ArgumentError("$label must be finite"))
    return converted
end

_cell_text(value) = ismissing(value) ? "" : String(value)

const REQUIRED_COLUMNS = (
    :issue_time_utc, :latest_dst_time_utc, :latest_dst_nt, :target_time_utc,
    :model_step_hours, :observation_dst_nt, :served_pred_dst_nt,
    :served_pred_dst_ci05_nt, :served_pred_dst_ci95_nt,
    :sub_hourly_model_version, :v24_status, :v24_manifest_sha256,
)

"Load exact V2.4e future rows and canonicalize retries; retain pending outcomes only when requested."
function canonical_live_rows(df::DataFrame; identity::AbstractString=EXACT_V24_IDENTITY,
                             require_matured::Bool=true)
    missing_columns = [String(column) for column in REQUIRED_COLUMNS if !(String(column) in names(df))]
    isempty(missing_columns) || throw(ArgumentError(
        "live log lacks required columns: $(join(missing_columns, ", "))",
    ))

    records = NamedTuple[]
    for row in eachrow(df)
        _cell_text(row.sub_hourly_model_version) == identity || continue
        _cell_text(row.v24_status) == "ok" || throw(ArgumentError(
            "exact V2.4e row has non-ok v24_status: $(_cell_text(row.v24_status))",
        ))
        _cell_text(row.v24_manifest_sha256) == EXACT_V24_MANIFEST_SHA256 ||
            throw(ArgumentError("exact V2.4e row has an unexpected served-manifest digest"))
        require_matured && ismissing(row.observation_dst_nt) && continue

        issue = _parse_time(row.issue_time_utc)
        latest = _parse_time(row.latest_dst_time_utc)
        target = _parse_time(row.target_time_utc)
        latest <= issue < target || throw(ArgumentError(
            "exact V2.4e row violates the future-target chronology",
        ))
        step_value = _float(row.model_step_hours, "model_step_hours")
        step_value >= 1 && isinteger(step_value) &&
            step_value == (target - latest) / Hour(1) ||
            throw(ArgumentError("model_step_hours must equal the positive integer target-to-anchor lead"))
        step = Int(step_value)
        point = _float(row.served_pred_dst_nt, "served_pred_dst_nt")
        lo = _float(row.served_pred_dst_ci05_nt, "served_pred_dst_ci05_nt")
        hi = _float(row.served_pred_dst_ci95_nt, "served_pred_dst_ci95_nt")
        lo < hi && lo <= point <= hi || throw(ArgumentError(
            "served interval must have positive width and contain its point forecast",
        ))
        observation = ismissing(row.observation_dst_nt) ? missing :
                      _float(row.observation_dst_nt, "observation_dst_nt")
        latest_dst = _float(row.latest_dst_nt, "latest_dst_nt")
        issue_hour = floor(issue, Hour)
        key = (issue_hour, target, String(identity))
        signature = (issue, latest, step, point, lo, hi, observation, latest_dst)
        push!(records, (; key, signature, issue_time_utc=issue, issue_hour_utc=issue_hour,
                        latest_dst_time_utc=latest, latest_dst_nt=latest_dst,
                        target_time_utc=target, model_step_hours=step,
                        point_dst_nt=point, observation_dst_nt=observation,
                        static_lo_dst_nt=lo, static_hi_dst_nt=hi,
                        served_identity=String(identity)))
    end

    isempty(records) && return DataFrame(
        row_key=String[], issue_time_utc=DateTime[], issue_hour_utc=DateTime[],
        latest_dst_time_utc=DateTime[], latest_dst_nt=Float64[], target_time_utc=DateTime[],
        model_step_hours=Int[], point_dst_nt=Float64[], observation_dst_nt=Float64[],
        static_lo_dst_nt=Float64[], static_hi_dst_nt=Float64[], served_identity=String[],
    )

    # Keep the latest retry in an issue hour. Exact-time duplicates must agree scientifically.
    chosen = Dict{Tuple{DateTime,DateTime,String},NamedTuple}()
    for record in records
        previous = get(chosen, record.key, nothing)
        if previous === nothing || record.issue_time_utc > previous.issue_time_utc
            chosen[record.key] = record
        elseif record.issue_time_utc == previous.issue_time_utc &&
               !isequal(record.signature, previous.signature)
            throw(ArgumentError("conflicting exact-time duplicate for $(record.key)"))
        end
    end

    canonical = sort!(collect(values(chosen));
                      by=r -> (r.issue_time_utc, r.target_time_utc, r.model_step_hours))
    rows = NamedTuple[]
    for record in canonical
        row_key = string(record.issue_hour_utc, "|", record.target_time_utc, "|",
                         record.served_identity)
        push!(rows, (; row_key, issue_time_utc=record.issue_time_utc,
                      issue_hour_utc=record.issue_hour_utc,
                      latest_dst_time_utc=record.latest_dst_time_utc,
                      latest_dst_nt=record.latest_dst_nt,
                      target_time_utc=record.target_time_utc,
                      model_step_hours=record.model_step_hours,
                      point_dst_nt=record.point_dst_nt,
                      observation_dst_nt=record.observation_dst_nt,
                      static_lo_dst_nt=record.static_lo_dst_nt,
                      static_hi_dst_nt=record.static_hi_dst_nt,
                      served_identity=record.served_identity))
    end
    return DataFrame(rows)
end

canonical_live_rows(path::AbstractString; kwargs...) =
    canonical_live_rows(CSV.read(path, DataFrame); kwargs...)

function _tail(history::Vector{Float64}, window::Int)
    first_index = max(1, length(history) - window + 1)
    return @view history[first_index:end]
end

function _candidate_interval(spec::CandidateSpec, point::Float64,
                             history::Vector{Float64}, alpha_t::Float64,
                             warmup::Int, coverage::Float64,
                             static_lo::Float64, static_hi::Float64, step::Int)
    n = length(history)
    sample = _tail(history, spec.window)
    if spec.method == :aci
        absolute = abs.(sample)
        hw = isempty(absolute) ? Inf :
             (n < warmup ? maximum(absolute) : conformal_upper(absolute, 1 - alpha_t))
        available = n >= warmup && isfinite(hw)
        return (; lo=point - hw, hi=point + hw, feedback_lo=point - hw,
                  feedback_hi=point + hw, available, location=0.0, alpha_t)
    end
    n >= warmup || return (; lo=NaN, hi=NaN, feedback_lo=NaN, feedback_hi=NaN,
                             available=false, location=NaN, alpha_t=NaN)
    if spec.method == :symmetric
        hw = conformal_upper(abs.(sample), coverage)
        return (; lo=point - hw, hi=point + hw, feedback_lo=NaN, feedback_hi=NaN,
                  available=true, location=0.0, alpha_t=NaN)
    elseif spec.method == :location_scale
        location = median(sample)
        hw = conformal_upper(abs.(sample .- location), coverage)
        return (; lo=point + location - hw, hi=point + location + hw,
                  feedback_lo=NaN, feedback_hi=NaN, available=true,
                  location, alpha_t=NaN)
    elseif spec.method == :equal_tail
        alpha = 1 - coverage
        lo_offset = empirical_lower(sample, alpha / 2)
        hi_offset = empirical_upper(sample, 1 - alpha / 2)
        return (; lo=point + lo_offset, hi=point + hi_offset,
                  feedback_lo=NaN, feedback_hi=NaN, available=true,
                  location=median(sample), alpha_t=NaN)
    elseif spec.method == :translated_static
        width_scale = _candidate_width_scale(spec, step)
        result = SolarSINDy._v24_calibration_shadow_interval(
            point, static_lo, static_hi, history;
            window=spec.window, warmup, width_scale,
        )
        result.available || return (; lo=NaN, hi=NaN, feedback_lo=NaN,
                                      feedback_hi=NaN, available=false,
                                      location=NaN, alpha_t=NaN)
        return (; lo=Float64(result.lo), hi=Float64(result.hi), feedback_lo=NaN,
                  feedback_hi=NaN, available=true, location=Float64(result.location),
                  alpha_t=NaN)
    end
    throw(ArgumentError("unknown candidate method: $(spec.method)"))
end

"Availability-causal replay. Observations enter a step history only after their target matures."
function replay_candidates(input::DataFrame; specs::Vector{CandidateSpec}=candidate_specs(),
                           identity::AbstractString=EXACT_V24_IDENTITY,
                           warmup::Int=DEFAULT_WARMUP,
                           coverage::Float64=TARGET_COVERAGE)
    0 < coverage < 1 || throw(ArgumentError("coverage must lie in (0, 1)"))
    warmup >= 1 || throw(ArgumentError("warmup must be positive"))
    rows = canonical_live_rows(input; identity)
    isempty(rows) && return DataFrame()

    histories = Dict{Int,Vector{Float64}}()
    alpha_state = Dict{Tuple{String,Int},Float64}()
    aci_feedback = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    activated = Set{String}()
    prior = Int[]
    output = NamedTuple[]

    issue_times = unique(rows.issue_time_utc)
    for issue_time in issue_times
        current = findall(==(issue_time), rows.issue_time_utc)
        cutoffs = unique(rows.latest_dst_time_utc[current])
        length(cutoffs) == 1 || throw(ArgumentError(
            "one issue cycle carries inconsistent latest_dst_time_utc values",
        ))
        cutoff = only(cutoffs)

        matured = [index for index in prior
                   if !(rows.row_key[index] in activated) && rows.target_time_utc[index] <= cutoff]
        sort!(matured; by=index -> (rows.target_time_utc[index], rows.issue_time_utc[index],
                                    rows.model_step_hours[index], rows.row_key[index]))
        for index in matured
            step = rows.model_step_hours[index]
            residual = rows.observation_dst_nt[index] - rows.point_dst_nt[index]
            for spec in specs
                spec.method == :aci || continue
                feedback = get(aci_feedback, (spec.name, rows.row_key[index]), nothing)
                feedback === nothing && error("missing ACI issuance feedback for $(rows.row_key[index])")
                covered = feedback[1] <= rows.observation_dst_nt[index] <= feedback[2]
                key = (spec.name, step)
                alpha = get(alpha_state, key, 1 - coverage)
                miss = covered ? 0.0 : 1.0
                alpha_state[key] = clamp(alpha + spec.gamma * ((1 - coverage) - miss), 0.0, 1.0)
            end
            push!(get!(histories, step, Float64[]), residual)
            push!(activated, rows.row_key[index])
        end

        for index in current
            step = rows.model_step_hours[index]
            history = get!(histories, step, Float64[])
            point = rows.point_dst_nt[index]
            observation = rows.observation_dst_nt[index]
            static_lo = rows.static_lo_dst_nt[index]
            static_hi = rows.static_hi_dst_nt[index]
            static_covered = static_lo <= observation <= static_hi
            static_score = interval_score(static_lo, static_hi, observation;
                                          alpha=1 - coverage)
            for spec in specs
                key = (spec.name, step)
                alpha = get(alpha_state, key, 1 - coverage)
                interval = _candidate_interval(spec, point, history, alpha, warmup, coverage,
                                               static_lo, static_hi, step)
                if spec.method == :aci
                    aci_feedback[(spec.name, rows.row_key[index])] =
                        (interval.feedback_lo, interval.feedback_hi)
                end
                available = interval.available && isfinite(interval.lo) && isfinite(interval.hi) &&
                            interval.lo <= interval.hi
                covered = available && interval.lo <= observation <= interval.hi
                score = available ? interval_score(interval.lo, interval.hi, observation;
                                                     alpha=1 - coverage) : NaN
                push!(output, (
                    candidate_family=spec.family, candidate=spec.name,
                    method=String(spec.method), window=spec.window, gamma=spec.gamma,
                    simplicity=spec.simplicity,
                    width_scale=_candidate_width_scale(spec, step),
                    row_key=rows.row_key[index],
                    issue_time_utc=issue_time, issue_hour_utc=rows.issue_hour_utc[index],
                    latest_dst_time_utc=rows.latest_dst_time_utc[index],
                    target_time_utc=rows.target_time_utc[index], model_step_hours=step,
                    served_identity=rows.served_identity[index], history_n=length(history),
                    point_dst_nt=point, observation_dst_nt=observation,
                    residual_dst_nt=observation - point,
                    static_lo_dst_nt=static_lo, static_hi_dst_nt=static_hi,
                    static_covered, static_width_nt=static_hi - static_lo,
                    static_interval_score=static_score,
                    shadow_available=available, shadow_lo_dst_nt=interval.lo,
                    shadow_hi_dst_nt=interval.hi, shadow_covered=covered,
                    shadow_width_nt=available ? interval.hi - interval.lo : NaN,
                    shadow_interval_score=score, location_shift_nt=interval.location,
                    alpha_t=interval.alpha_t,
                ))
            end
        end
        append!(prior, current)
    end
    return DataFrame(output)
end

replay_candidates(path::AbstractString; kwargs...) =
    replay_candidates(CSV.read(path, DataFrame); kwargs...)

function _metrics(df::DataFrame)
    n = nrow(df)
    n > 0 || return (; n=0, coverage=NaN, mean_width_nt=NaN, mean_interval_score=NaN,
                       static_coverage=NaN, static_mean_width_nt=NaN,
                       static_mean_interval_score=NaN, width_ratio=NaN,
                       score_difference=NaN)
    coverage = mean(df.shadow_covered)
    mean_width = mean(df.shadow_width_nt)
    mean_score = mean(df.shadow_interval_score)
    static_coverage = mean(df.static_covered)
    static_width = mean(df.static_width_nt)
    static_score = mean(df.static_interval_score)
    return (; n, coverage, mean_width_nt=mean_width, mean_interval_score=mean_score,
              static_coverage, static_mean_width_nt=static_width,
              static_mean_interval_score=static_score,
              width_ratio=mean_width / static_width,
              score_difference=mean_score - static_score)
end

"Summarize the common-row control and apply the frozen development gates."
function summarize_candidates(replay::DataFrame; min_step_n::Int=40)
    isempty(replay) && return (summary=DataFrame(), by_step=DataFrame())
    summary_rows = NamedTuple[]
    step_rows = NamedTuple[]
    for candidate in unique(replay.candidate)
        all_rows = replay[replay.candidate .== candidate, :]
        eligible = all_rows[all_rows.shadow_available, :]
        metrics = _metrics(eligible)
        step_gate = true
        for step in sort(unique(eligible.model_step_hours))
            group = eligible[eligible.model_step_hours .== step, :]
            step_metrics = _metrics(group)
            step_pass = step_metrics.n < min_step_n || step_metrics.coverage >= 0.85
            step_gate &= step_pass
            push!(step_rows, (candidate, model_step_hours=step, step_metrics...,
                              coverage_gate_applicable=step_metrics.n >= min_step_n,
                              coverage_gate_pass=step_pass))
        end
        finite_ordered = all(row -> !row.shadow_available ||
                             (isfinite(row.shadow_lo_dst_nt) &&
                              isfinite(row.shadow_hi_dst_nt) &&
                              row.shadow_lo_dst_nt <= row.shadow_hi_dst_nt), eachrow(all_rows))
        reasons = String[]
        metrics.n > 0 || push!(reasons, "no eligible rows")
        0.88 <= metrics.coverage <= 0.95 || push!(reasons, "pooled coverage outside 0.88-0.95")
        step_gate || push!(reasons, "a populated step is below 0.85 coverage")
        (metrics.width_ratio <= 1.50 ||
         isapprox(metrics.width_ratio, 1.50; atol=1e-12, rtol=0)) ||
            push!(reasons, "width ratio exceeds 1.50")
        metrics.score_difference <= 0 || push!(reasons, "interval score is worse than static")
        finite_ordered || push!(reasons, "non-finite or reversed endpoint")
        gate_pass = isempty(reasons)
        first_row = first(eachrow(all_rows))
        push!(summary_rows, (
            candidate, candidate_family=first_row.candidate_family,
            method=first_row.method, window=first_row.window, gamma=first_row.gamma,
            width_scale=maximum(all_rows.width_scale), simplicity=first_row.simplicity,
            metrics..., finite_ordered,
            step_coverage_gate_pass=step_gate, gate_pass,
            gate_reason=gate_pass ? "all frozen development gates pass" : join(reasons, "; "),
        ))
    end
    return (summary=DataFrame(summary_rows), by_step=DataFrame(step_rows))
end

function select_shadow_candidate(summary::DataFrame)
    isempty(summary) && return nothing
    passing = summary[summary.gate_pass, :]
    isempty(passing) && return nothing
    # Amendments A2/A3 are the only families built to satisfy the already-frozen
    # prospective 1.25 mean-width gate. Prefer the latest passing gate-aligned
    # family before any prospective row is collected; retain A1 only as a
    # diagnostic fallback when neither clears the unchanged development gates.
    for family in ("C7", "C6")
        gate_aligned = passing[passing.candidate_family .== family, :]
        isempty(gate_aligned) || (passing = gate_aligned; break)
    end
    sort!(passing, [:mean_interval_score, :mean_width_nt, :simplicity, :candidate])
    return String(passing.candidate[1])
end

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function _candidate_digest(spec::CandidateSpec, warmup::Int, coverage::Float64)
    overrides = join(("$(step):$(repr(scale))" for (step, scale) in
                      sort(collect(spec.width_scale_by_step); by=first)), ",")
    payload = join((spec.family, spec.name, String(spec.method), string(spec.window),
                    repr(spec.gamma), repr(spec.width_scale), overrides, string(warmup),
                    repr(coverage), EXACT_V24_IDENTITY), "|")
    return bytes2hex(sha256(codeunits(payload)))
end

function _write_report(path::AbstractString, input_path::AbstractString, input_sha::String,
                       rows::DataFrame, summary::DataFrame, by_step::DataFrame,
                       winner, digest)
    storm_rows = count(value -> value <= -50.0, rows.observation_dst_nt)
    open(path, "w") do io
        println(io, "# V2.4e Live-Calibration Development Decision")
        println(io)
        println(io, "Input: `$(abspath(input_path))`  ")
        println(io, "SHA-256: `$input_sha`  ")
        println(io, "Exact identity: `$(EXACT_V24_IDENTITY)`  ")
        println(io, "Canonical matured rows: $(nrow(rows))  ")
        println(io, "Observed Dst <= -50 nT rows: $storm_rows")
        println(io)
        if winner === nothing
            println(io, "Decision: **NO SHADOW WINNER**. No candidate cleared every frozen development gate.")
        else
            selected = summary[summary.candidate .== winner, :][1, :]
            println(io, "Decision: **SHADOW ONLY** — `$winner` cleared the frozen development gates.")
            println(io, "Configuration digest: `$digest`")
            println(io)
            println(io, "This result authorizes prospective shadow collection only. It does not change the served V2.4e interval and does not establish nominal live calibration or storm skill.")
            println(io)
            println(io, "Selected development metrics: coverage $(round(selected.coverage; digits=3)), mean width $(round(selected.mean_width_nt; digits=2)) nT, width ratio $(round(selected.width_ratio; digits=3)), paired interval-score difference $(round(selected.score_difference; digits=2)).")
        end
        println(io)
        println(io, "## Candidate gate table")
        println(io)
        println(io, "| Candidate | n | Coverage | Width ratio | Score difference | Gate |")
        println(io, "|---|---:|---:|---:|---:|---|")
        for row in eachrow(sort(summary, :candidate))
            println(io, "| $(row.candidate) | $(row.n) | $(round(row.coverage; digits=3)) | $(round(row.width_ratio; digits=3)) | $(round(row.score_difference; digits=2)) | $(row.gate_pass ? "PASS" : "FAIL") |")
        end
        println(io)
        println(io, "Step-level metrics are in `candidate_by_step.csv`; row-level causal endpoints are in `replay.csv`.")
        storm_rows == 0 && println(io, "No storm row is present, so the prospective storm-skill gate remains blocked.")
    end
end

"Run the frozen development study and persist its auditable tables."
function run_study(input_path::AbstractString, output_dir::AbstractString;
                   expected_sha::Union{Nothing,AbstractString}=nothing,
                   specs::Vector{CandidateSpec}=candidate_specs(),
                   warmup::Int=DEFAULT_WARMUP,
                   coverage::Float64=TARGET_COVERAGE)
    isfile(input_path) || throw(ArgumentError("live-log snapshot is missing: $input_path"))
    input_sha = _sha256_file(input_path)
    expected_sha === nothing || lowercase(String(expected_sha)) == input_sha ||
        error("live-log snapshot SHA-256 mismatch: expected $(lowercase(String(expected_sha))), got $input_sha")
    source = CSV.read(input_path, DataFrame)
    rows = canonical_live_rows(source)
    replay = replay_candidates(source; specs, warmup, coverage)
    tables = summarize_candidates(replay)
    winner = select_shadow_candidate(tables.summary)
    spec = winner === nothing ? nothing : only(filter(item -> item.name == winner, specs))
    digest = spec === nothing ? "" : _candidate_digest(spec, warmup, coverage)

    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "canonical_rows.csv"), rows)
    CSV.write(joinpath(output_dir, "replay.csv"), replay)
    CSV.write(joinpath(output_dir, "candidate_summary.csv"), tables.summary)
    CSV.write(joinpath(output_dir, "candidate_by_step.csv"), tables.by_step)
    receipt = DataFrame(
        input_path=[abspath(input_path)], input_sha256=[input_sha],
        exact_identity=[EXACT_V24_IDENTITY], canonical_rows=[nrow(rows)],
        target_coverage=[coverage], warmup=[warmup], selected_candidate=[something(winner, "")],
        selected_config_sha256=[digest], decision=[winner === nothing ? "NO_SHADOW_WINNER" : "SHADOW_ONLY"],
    )
    CSV.write(joinpath(output_dir, "receipt.csv"), receipt)
    _write_report(joinpath(output_dir, "DECISION.md"), input_path, input_sha, rows,
                  tables.summary, tables.by_step, winner, digest)
    return (; rows, replay, summary=tables.summary, by_step=tables.by_step,
              winner, digest, input_sha)
end

function _cli(args)
    options = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || throw(ArgumentError("unexpected argument: $arg"))
        pieces = split(arg[3:end], "="; limit=2)
        length(pieces) == 2 || throw(ArgumentError("expected --name=value, got $arg"))
        options[pieces[1]] = pieces[2]
    end
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    input = get(options, "input", joinpath(package_root, "validation", "input", "operational",
                                             "v2_4_live_calibration_dev_20260824T213044Z.csv"))
    output = get(options, "output", joinpath(package_root, "validation", "output", "operational",
                                               "v2_4_live_calibration"))
    expected = get(options, "expected-sha", nothing)
    result = run_study(input, output; expected_sha=expected)
    println("V2.4e live-calibration development replay")
    println("  canonical rows: $(nrow(result.rows))")
    println("  input SHA-256: $(result.input_sha)")
    println("  decision: $(result.winner === nothing ? "NO_SHADOW_WINNER" : "SHADOW_ONLY $(result.winner)")")
    println("  report: $(joinpath(output, "DECISION.md"))")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    V24LiveCalibration._cli(ARGS)
end
