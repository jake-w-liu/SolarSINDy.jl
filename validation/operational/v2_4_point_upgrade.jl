#!/usr/bin/env julia

module V24PointUpgrade

using CSV, DataFrames, Dates, Random, SHA, Statistics
include("v2_4_interval_upgrade.jl")
using .V24IntervalUpgrade: anchor_witnesses
import .V24IntervalUpgrade.V24LiveCalibration: _float, _parse_time

export POINT_CANDIDATES, POINT_MODELS, corrected_point, checked_panel,
       replay_points, historical_panel, storm_events, point_metrics, paired_rmse_gain

const POINT_CANDIDATES = ("Mean24", "Mean48", "Mean96", "Median24", "Median48", "Median96")
const POINT_MODELS = (:v2_4e, :static_v2_2, :served_v2_1, :frozen_v2_1, :v2_3_shadow,
    :direct_gbm, :t1r_analog, :persistence, :burton, :burton_full, :obrien, :climatology)
const STEPS = (1, 2, 3, 4, 6, 7)

"Bounded signed-residual location correction; sparse history preserves the original product."
function corrected_point(name::AbstractString, point::Real, lo::Real, hi::Real,
                         step::Integer, history::AbstractVector)
    name in POINT_CANDIDATES || throw(ArgumentError("unknown point candidate: $name"))
    step in STEPS || throw(ArgumentError("unsupported model step: $step"))
    p, l, u = _float.((point, lo, hi), ("point", "lower endpoint", "upper endpoint"))
    l <= p <= u && l < u && isfinite(u-l) || throw(ArgumentError("static interval must contain the point and have finite positive width"))
    h = [_float(value, "historical residual") for value in history]
    length(h) < 30 && return (point=p, lower=l, upper=u, corrected=false, used_n=0,
        estimate=0.0, capped=0.0, actual_shift=0.0, cap_applied=false, projection_applied=false)
    window = parse(Int, match(r"\d+$", name).match)
    sample = @view h[max(1, length(h) - window + 1):end]
    estimate = startswith(name, "Mean") ? mean(sample) : median(sample)
    isfinite(estimate) || throw(ArgumentError("residual estimate overflow"))
    cap = 10.0 + 5.0 * step
    capped = clamp(estimate, -cap, cap)
    unprojected = p + capped
    center = clamp(unprojected, -2000.0, 50.0)
    lower = center - 1.25 * (p - l)
    upper = center + 1.25 * (u - p)
    all(isfinite, (lower, upper, upper - lower)) || throw(ArgumentError("translated interval overflow"))
    return (point=center, lower, upper, corrected=true, used_n=length(sample), estimate,
        capped, actual_shift=center-p, cap_applied=capped != estimate,
        projection_applied=center != unprojected)
end

"Normalize timestamps and reject incomplete or noncausal comparison panels. Pending targets may be NaN."
function checked_panel(input::DataFrame)
    required = (:issue_time_utc, :latest_dst_time_utc, :target_time_utc, :model_step_hours,
        :latest_dst_nt, :observation_dst_nt, :static_lo_nt, :static_hi_nt, :model_epoch,
        POINT_MODELS...)
    absent = [String(name) for name in required if !hasproperty(input, name)]
    isempty(absent) || throw(ArgumentError("point panel lacks columns: " * join(absent, ", ")))
    panel = copy(input)
    for column in (:issue_time_utc, :latest_dst_time_utc, :target_time_utc)
        panel[!, column] = _parse_time.(panel[!, column])
    end
    for column in (POINT_MODELS..., :latest_dst_nt, :static_lo_nt, :static_hi_nt)
        panel[!, column] = [_float(value, String(column)) for value in panel[!, column]]
    end
    panel[!, :observation_dst_nt] = [ismissing(value) || (value isa Real && isnan(value)) ? NaN :
        _float(value, "scoring observation") for value in panel.observation_dst_nt]
    steps = [_float(value, "model step") for value in panel.model_step_hours]
    all(value -> isinteger(value) && value in STEPS, steps) || throw(ArgumentError("invalid model steps"))
    panel[!, :model_step_hours] = Int.(steps)
    all(value -> !ismissing(value) && !isempty(strip(String(value))), panel.model_epoch) ||
        throw(ArgumentError("model epoch must be an explicit nonempty identity"))
    panel[!, :model_epoch] = String.(panel.model_epoch)
    keys = String[]
    observations = Dict{DateTime,Float64}()
    for row in eachrow(panel)
        row.latest_dst_time_utc <= row.issue_time_utc < row.target_time_utc ||
            throw(ArgumentError("point panel violates anchor/issue/future-target chronology"))
        row.target_time_utc == row.latest_dst_time_utc + Hour(row.model_step_hours) ||
            throw(ArgumentError("point panel target does not match the model step"))
        row.static_lo_nt <= row.v2_4e <= row.static_hi_nt && row.static_lo_nt < row.static_hi_nt &&
            isfinite(row.static_hi_nt-row.static_lo_nt) ||
            throw(ArgumentError("invalid base interval"))
        if isfinite(row.observation_dst_nt)
            old = get(observations, row.target_time_utc, row.observation_dst_nt)
            old == row.observation_dst_nt || throw(ArgumentError("conflicting target scoring observations"))
            observations[row.target_time_utc] = row.observation_dst_nt
        end
        push!(keys, string(row.issue_time_utc, "|", row.target_time_utc, "|", row.model_epoch))
    end
    length(unique(keys)) == length(keys) || throw(ArgumentError("duplicate point-panel forecast key"))
    panel[!, :row_key] = keys
    return sort!(panel, [:issue_time_utc, :target_time_utc, :row_key])
end

"Replay the fixed grid from issued anchor witnesses, with separate histories per model epoch and step."
function replay_points(input::DataFrame, witness_input::DataFrame; delay_hours::Integer=0)
    delay_hours in (0, 1) || throw(ArgumentError("residual feedback delay must be zero or one hour"))
    panel = checked_panel(input)
    witnesses = anchor_witnesses(witness_input)
    known = Dict{DateTime,Float64}()
    ordered = Dict{Tuple{String,Int},Vector{Int}}()
    for index in 1:nrow(panel)
        push!(get!(ordered, (panel.model_epoch[index], panel.model_step_hours[index]), Int[]), index)
    end
    for indices in values(ordered)
        sort!(indices; by=i -> (panel.target_time_utc[i], panel.issue_time_utc[i], panel.row_key[i]))
    end
    targets = Dict(key => panel.target_time_utc[indices] for (key, indices) in ordered)
    history_n = zeros(Int, nrow(panel))
    newest = fill("", nrow(panel))
    output = Dict(name => NamedTuple[] for name in POINT_CANDIDATES)
    cursor = 1
    for (index, row) in enumerate(eachrow(panel))
        while cursor <= length(witnesses) && witnesses[cursor].available_utc + Hour(delay_hours) <= row.issue_time_utc
            witness = witnesses[cursor]
            known[witness.target_utc] = witness.dst_nt
            cursor += 1
        end
        key = (row.model_epoch, row.model_step_hours)
        indices = ordered[key]
        last = searchsortedlast(targets[key], row.latest_dst_time_utc)
        history = Float64[]
        for position in last:-1:1
            earlier = indices[position]
            target = panel.target_time_utc[earlier]
            panel.issue_time_utc[earlier] < row.issue_time_utc && haskey(known, target) || continue
            isempty(history) && (newest[index] = string(target) * "Z")
            push!(history, _float(known[target] - panel.v2_4e[earlier], "witnessed residual"))
            length(history) == 96 && break
        end
        reverse!(history)
        history_n[index] = length(history) # retained history count, capped at the largest fixed window
        for name in POINT_CANDIDATES
            push!(output[name], corrected_point(name, row.v2_4e, row.static_lo_nt,
                row.static_hi_nt, row.model_step_hours, history))
        end
    end
    panel[!, :history_retained_n] = history_n
    panel[!, :newest_history_target_utc] = newest
    panel[!, :feedback_delay_hours] = fill(Int(delay_hours), nrow(panel))
    for name in POINT_CANDIDATES
        fields = (:point, :lower, :upper, :corrected, :used_n, :estimate, :capped,
            :actual_shift, :cap_applied, :projection_applied)
        for field in fields
            panel[!, Symbol(name, "_", field)] = [getproperty(row, field) for row in output[name]]
        end
    end
    return panel
end

"Map a hash-checked rolling fold to the point-study schema without altering any predictions."
function historical_panel(path::AbstractString, expected_sha::AbstractString, year::Integer)
    bytes = read(path)
    bytes2hex(sha256(bytes)) == expected_sha || throw(ArgumentError("rolling source digest differs: $path"))
    panel = CSV.read(IOBuffer(bytes), DataFrame; strict=true)
    times = _parse_time.(panel.issue_time_utc)
    all(t -> Dates.year(t) == year, times) || throw(ArgumentError("rolling source contains a different model year"))
    panel[!, :latest_dst_time_utc] = times
    panel[!, :target_time_utc] = times .+ Hour.(panel.model_step_hours)
    panel[!, :static_lo_nt] = copy(panel.v2_4e_lo_nt)
    panel[!, :static_hi_nt] = copy(panel.v2_4e_hi_nt)
    panel[!, :model_epoch] = fill("fold$(year)", nrow(panel))
    return checked_panel(panel)
end

"Storm IDs require 72 consecutive known hourly values above -30 nT before separating events."
function storm_events(times::AbstractVector, values::AbstractVector)
    length(times) == length(values) || throw(DimensionMismatch("storm times and observations differ"))
    observed = Dict{DateTime,Float64}()
    for (time, value) in zip(times, values)
        t = _parse_time(time)
        t == floor(t, Hour) || throw(ArgumentError("storm observations must lie on the hourly grid"))
        y = _float(value, "storm observation")
        haskey(observed, t) && observed[t] != y && throw(ArgumentError("conflicting storm observations"))
        observed[t] = y
    end
    event = 0
    separated = true
    quiet_hours = 0
    previous = nothing
    result = Dict{DateTime,Int}()
    for t in sort!(collect(keys(observed)))
        previous === nothing || t == previous + Hour(1) || (quiet_hours = 0)
        y = observed[t]
        quiet_hours = y > -30 ? quiet_hours + 1 : 0
        quiet_hours >= 72 && (separated = true)
        if y <= -50
            if event == 0 || separated
                event += 1
            end
            separated = false
            result[t] = event
        end
        previous = t
    end
    return result
end

"Point metrics use observed-minus-predicted signed error and require identical finite rows."
function point_metrics(prediction::AbstractVector, observation::AbstractVector)
    length(prediction) == length(observation) || throw(DimensionMismatch("point metric lengths differ"))
    p = [_float(value,"prediction") for value in prediction]
    y = [_float(value,"observation") for value in observation]
    isempty(p) && return (n=0,rmse_nt=NaN,mae_nt=NaN,bias_nt=NaN)
    error = y .- p
    squared = error.^2
    all(isfinite,squared) || throw(ArgumentError("point error or squared error overflow"))
    values = (sqrt(mean(squared)),mean(abs.(error)),mean(error))
    all(isfinite,values) || throw(ArgumentError("point metric reduction overflow"))
    return (n=length(p),rmse_nt=values[1],mae_nt=values[2],bias_nt=values[3])
end

"Paired whole-block bootstrap of reference RMSE minus candidate RMSE, retaining row weights."
function paired_rmse_gain(prediction::AbstractVector, reference::AbstractVector,
                          observation::AbstractVector, blocks::AbstractVector;
                          reps::Integer=2000, seed::Integer=20260908)
    length(prediction) == length(reference) == length(observation) == length(blocks) ||
        throw(DimensionMismatch("paired bootstrap lengths differ"))
    reps >= 100 || throw(ArgumentError("at least 100 bootstrap resamples are required"))
    seed >= 0 || throw(ArgumentError("bootstrap seed must be nonnegative"))
    candidate = point_metrics(prediction,observation)
    control = point_metrics(reference,observation)
    gain = control.rmse_nt - candidate.rmse_nt
    labels = sort!(unique(blocks))
    length(labels) >= 2 || return (gain_nt=gain,lower_nt=NaN,upper_nt=NaN,draws=Float64[])
    p,r,y = Float64.(prediction),Float64.(reference),Float64.(observation)
    index = Dict(label=>i for (i,label) in enumerate(labels))
    candidate_sse,reference_sse = zeros(length(labels)),zeros(length(labels))
    counts = zeros(Int,length(labels))
    for i in eachindex(p)
        j = index[blocks[i]]
        candidate_sse[j] += (y[i]-p[i])^2
        reference_sse[j] += (y[i]-r[i])^2
        counts[j] += 1
    end
    rng = MersenneTwister(seed)
    draws = Float64[]
    for _ in 1:reps
        a,b,n = 0.0,0.0,0
        for _ in eachindex(labels)
            j = rand(rng,eachindex(labels))
            a += candidate_sse[j]
            b += reference_sse[j]
            n += counts[j]
        end
        value = sqrt(b/n)-sqrt(a/n)
        isfinite(value) || throw(ArgumentError("paired bootstrap reduction overflow"))
        push!(draws,value)
    end
    return (gain_nt=gain,lower_nt=quantile(draws,0.05),upper_nt=quantile(draws,0.95),draws)
end

end # module
