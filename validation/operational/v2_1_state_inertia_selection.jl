#!/usr/bin/env julia

# Select the causal state-inertia safeguard from development evidence only.
# Broad-replay test storms and the exact G3+ archive remain evaluation cohorts.

using CSV
using DataFrames
using Printf
using Statistics

isdefined(@__MODULE__, :_selftest_v2) || include(joinpath(@__DIR__, "v2_replay.jl"))

const STATE_BROAD_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_scored.csv")
const STATE_SEVERE_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_scored.csv")
const STATE_GSCALE_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_replay_scored.csv")
const STATE_CANDIDATES_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_state_inertia_candidates.csv")
const STATE_EVALUATION_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_state_inertia_evaluation.csv")
const STATE_REGIMES_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_state_inertia_regimes.csv")
const STATE_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_state_inertia_report.md")
const STATE_WEIGHT_GRID = collect(0.0:0.125:1.0)
const STATE_MIN_REGIME_ROWS = 40
const STATE_VALIDATION_TIE_TOL_NT = 0.01

_state_rmse(pred, obs) = sqrt(mean((Float64.(pred) .- Float64.(obs)) .^ 2))

function _state_required(df::DataFrame)
    required = [:lead, :obs, :persistence, :rate, :v2_0,
                :v2_1_pre_state_inertia]
    missing_cols = [String(c) for c in required if String(c) ∉ names(df)]
    isempty(missing_cols) || error("state-inertia input omits: $(join(missing_cols, ", "))")
    return df
end

function _state_candidate(
    df::DataFrame,
    h1_quiet_weight::Real,
    h1_deepening_weight::Real,
    h2_quiet_weight::Real,
    h3_quiet_weight::Real,
)
    _state_required(df)
    weights = Float64[
        h1_quiet_weight, h1_deepening_weight,
        h2_quiet_weight, h3_quiet_weight,
    ]
    all(w -> 0.0 <= w <= 1.0, weights) || error("candidate weights must lie in [0, 1]")
    point = copy(Float64.(df.v2_1_pre_state_inertia))
    latest = Float64.(df.persistence)
    rate = Float64.(df.rate)
    for i in eachindex(point)
        h = Int(df.lead[i])
        weight = if h == 1 &&
                    V2_STATE_INERTIA_DEEPENING_LO_NT_PER_H <= rate[i] <
                    V2_STATE_INERTIA_DEEPENING_HI_NT_PER_H
            weights[2]
        elseif h == 1 && latest[i] > V2_STATE_INERTIA_QUIET_DST_NT &&
               point[i] > V2_STATE_INERTIA_QUIET_DST_NT
            weights[1]
        elseif h == 2 && latest[i] > V2_STATE_INERTIA_QUIET_DST_NT &&
               point[i] > V2_STATE_INERTIA_QUIET_DST_NT
            weights[3]
        elseif h == 3 && latest[i] > V2_STATE_INERTIA_QUIET_DST_NT &&
               point[i] > V2_STATE_INERTIA_QUIET_DST_NT
            weights[4]
        else
            1.0
        end
        point[i] = clamp(latest[i] + weight * (point[i] - latest[i]), -2000.0, 50.0)
        _extreme_inertia_guard(latest[i], h) && (point[i] = latest[i])
    end
    return point
end

_state_target_regime(x::Real) = x > -30 ? "quiet" : x > -50 ? "minor" :
                                 x > -100 ? "moderate" : x > -200 ? "intense" : "extreme"
_state_rate_regime(x::Real) = x < -15 ? "rapid_deepening" : x < -5 ? "deepening" :
                               x <= 5 ? "steady" : "recovering"

function _state_regime_rows(df::DataFrame, pred::AbstractVector;
                            min_n::Int=STATE_MIN_REGIME_ROWS)
    out = DataFrame(
        axis=String[], lead=Int[], regime=String[], n=Int[],
        rmse_v2_0_nt=Float64[], rmse_v2_1_nt=Float64[],
        rmse_persistence_nt=Float64[], delta_vs_v2_0_nt=Float64[],
        delta_vs_best_nt=Float64[],
    )
    labels = (
        ("target_dst", _state_target_regime.(Float64.(df.obs))),
        ("issue_rate", _state_rate_regime.(Float64.(df.rate))),
    )
    for (axis, lab) in labels, lead in LEADS, regime in sort(unique(lab))
        mask = (Int.(df.lead) .== lead) .& (lab .== regime)
        n = count(mask)
        n >= min_n || continue
        rv20 = _state_rmse(df.v2_0[mask], df.obs[mask])
        rv21 = _state_rmse(pred[mask], df.obs[mask])
        rpers = _state_rmse(df.persistence[mask], df.obs[mask])
        push!(out, (axis, lead, regime, n, rv20, rv21, rpers,
                    rv21 - rv20, rv21 - min(rv20, rpers)))
    end
    return out
end

function _state_lead_metrics(df::DataFrame, pred::AbstractVector, cohort::AbstractString)
    out = DataFrame(
        cohort=String[], lead=Int[], n=Int[], n_groups=Int[],
        rmse_v2_0_nt=Float64[], rmse_v2_1_nt=Float64[],
        rmse_persistence_nt=Float64[], improvement_vs_best_nt=Float64[],
    )
    for lead in LEADS
        mask = Int.(df.lead) .== lead
        count(mask) > 0 || continue
        groups = if :storm_id in propertynames(df)
            length(unique(Int.(df.storm_id[mask])))
        elseif :g_event_id in propertynames(df)
            length(unique(Int.(df.g_event_id[mask])))
        elseif :storm in propertynames(df)
            length(unique(String.(df.storm[mask])))
        else
            0
        end
        rv20 = _state_rmse(df.v2_0[mask], df.obs[mask])
        rv21 = _state_rmse(pred[mask], df.obs[mask])
        rpers = _state_rmse(df.persistence[mask], df.obs[mask])
        push!(out, (String(cohort), lead, count(mask), groups, rv20, rv21,
                    rpers, min(rv20, rpers) - rv21))
    end
    return out
end

function _state_development_gate(broad::DataFrame, severe::DataFrame,
                                 wq1::Real, wd1::Real, wq2::Real, wq3::Real)
    severe_pred = _state_candidate(severe, wq1, wd1, wq2, wq3)
    regimes = _state_regime_rows(severe, severe_pred)
    expected_cells = length(LEADS) * 9
    regime_ok = nrow(regimes) == expected_cells &&
                all(regimes.delta_vs_v2_0_nt .<= 1e-9)
    split_ok = true
    validation_sum = 0.0
    for split in ("train", "val")
        rows = broad[String.(broad.storm_split) .== split, :]
        pred = _state_candidate(rows, wq1, wd1, wq2, wq3)
        metrics = _state_lead_metrics(rows, pred, split)
        split_ok &= nrow(metrics) == length(LEADS) &&
                    all(metrics.improvement_vs_best_nt .> 0.0)
        split == "val" && (validation_sum = sum(metrics.rmse_v2_1_nt))
    end
    return regime_ok && split_ok, validation_sum, regimes
end

function select_state_inertia(broad::DataFrame, severe::DataFrame)
    :storm_split in propertynames(broad) || error("broad replay omits storm_split")
    out = DataFrame(
        h1_quiet_weight=Float64[], h1_deepening_weight=Float64[],
        h2_quiet_weight=Float64[], h3_quiet_weight=Float64[],
        validation_rmse_sum_nt=Float64[], passes_development_gate=Bool[],
    )
    for wq1 in STATE_WEIGHT_GRID, wd1 in STATE_WEIGHT_GRID,
        wq2 in STATE_WEIGHT_GRID, wq3 in STATE_WEIGHT_GRID
        passes, score, _ = _state_development_gate(
            broad, severe, wq1, wd1, wq2, wq3,
        )
        push!(out, (wq1, wd1, wq2, wq3, score, passes))
    end
    passing = findall(out.passes_development_gate)
    isempty(passing) && error("no state-inertia candidate passed the development gates")
    minimum_score = minimum(out.validation_rmse_sum_nt[passing])
    tied = [i for i in passing if
            out.validation_rmse_sum_nt[i] <= minimum_score + STATE_VALIDATION_TIE_TOL_NT]
    # Dst is published at 1 nT resolution. Validation-score differences below
    # 0.01 nT across all four leads are therefore treated as practically tied.
    # Within that tie, preserve the most one-hour model signal because the
    # general H1 inertia operator has already performed one persistence blend;
    # then minimize total intervention across the remaining branches.
    best_index = first(sort(tied; by=i -> (
        -out.h1_quiet_weight[i],
        -(out.h1_quiet_weight[i] + out.h1_deepening_weight[i] +
          out.h2_quiet_weight[i] + out.h3_quiet_weight[i]),
        -out.h2_quiet_weight[i], -out.h3_quiet_weight[i],
        -out.h1_deepening_weight[i], out.validation_rmse_sum_nt[i],
    )))
    out.within_validation_tie = [i in tied for i in 1:nrow(out)]
    out.selected = [i == best_index for i in 1:nrow(out)]
    best = (
        wq1=out.h1_quiet_weight[best_index],
        wd1=out.h1_deepening_weight[best_index],
        wq2=out.h2_quiet_weight[best_index],
        wq3=out.h3_quiet_weight[best_index],
        minimum_score=minimum_score,
        selected_score=out.validation_rmse_sum_nt[best_index],
    )
    selected_weights = (best.wq1, best.wd1, best.wq2, best.wq3)
    deployed_weights = (
        V2_STATE_INERTIA_H1_QUIET_WEIGHT,
        V2_STATE_INERTIA_H1_DEEPENING_WEIGHT,
        V2_STATE_INERTIA_H2_QUIET_WEIGHT,
        V2_STATE_INERTIA_H3_QUIET_WEIGHT,
    )
    selected_weights == deployed_weights || error(
        "selected state-inertia weights $selected_weights differ from deployed $deployed_weights",
    )
    return out, best
end

function _state_gscale_gate(df::DataFrame, pred::AbstractVector)
    for level in (3, 4, 5), lead in LEADS
        mask = (Int.(df.g_level) .== level) .& (Int.(df.lead) .== lead)
        count(mask) > 0 || continue
        rv21 = _state_rmse(pred[mask], df.obs[mask])
        rv20 = _state_rmse(df.v2_0[mask], df.obs[mask])
        rpers = _state_rmse(df.persistence[mask], df.obs[mask])
        rv21 < min(rv20, rpers) || error(
            "state-inertia V2.1 does not beat both comparators for G$level at $(lead) h",
        )
    end
    return true
end

function main_state_inertia()
    broad = _state_required(CSV.read(STATE_BROAD_CSV, DataFrame))
    severe = _state_required(CSV.read(STATE_SEVERE_CSV, DataFrame))
    gscale = _state_required(CSV.read(STATE_GSCALE_CSV, DataFrame))
    candidates, selected = select_state_inertia(broad, severe)
    weights = (selected.wq1, selected.wd1, selected.wq2, selected.wq3)

    evaluation = DataFrame()
    cohorts = (
        ("broad_train", broad[String.(broad.storm_split) .== "train", :]),
        ("broad_validation", broad[String.(broad.storm_split) .== "val", :]),
        ("broad_test", broad[String.(broad.storm_split) .== "test", :]),
        ("broad_all", broad),
        ("severe_development_stress", severe),
        ("exact_G3plus", gscale),
    )
    for (label, rows) in cohorts
        pred = _state_candidate(rows, weights...)
        append!(evaluation, _state_lead_metrics(rows, pred, label))
    end
    all(evaluation.improvement_vs_best_nt .> 0.0) || error(
        "selected state-inertia candidate fails an evaluation cohort",
    )
    gpred = _state_candidate(gscale, weights...)
    _state_gscale_gate(gscale, gpred)
    regimes = _state_regime_rows(severe, _state_candidate(severe, weights...))
    all(regimes.delta_vs_v2_0_nt .<= 1e-9) || error(
        "selected state-inertia candidate fails the historical V2.0 regime guard",
    )

    CSV.write(STATE_CANDIDATES_CSV, candidates)
    CSV.write(STATE_EVALUATION_CSV, evaluation)
    CSV.write(STATE_REGIMES_CSV, regimes)
    open(STATE_REPORT, "w") do io
        println(io, "# Operational V2.1 state-inertia selection\n")
        @printf(io, "The development selector evaluated %d causal weight tuples. It required every broad-replay training/validation lead to beat historical V2.0 and persistence and every populated severe-development target-Dst/rate cell to be non-inferior to historical V2.0. Validation RMSE sums within %.3f nT of the minimum were treated as tied at observational precision; the tie-break retained the most one-hour model signal before minimizing intervention in the remaining branches. The broad test partition and exact G3+ archive were then used as post-selection safety evaluations.\n\n", nrow(candidates), STATE_VALIDATION_TIE_TOL_NT)
        @printf(io, "Minimum passing validation RMSE sum: %.6f nT; selected tied-candidate sum: %.6f nT.\n\n", selected.minimum_score, selected.selected_score)
        @printf(io, "Selected weights: one-hour near-quiet %.3f, one-hour moderate deepening %.3f, two-hour near-quiet %.3f, and three-hour near-quiet %.3f. The live operator uses only issue-time Dst, issue-time Dst rate, lead, and the V2.1 center.\n\n", weights...)
        println(io, "| cohort | lead h | rows | groups | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | improvement vs stronger comparator |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(evaluation)
            @printf(io, "| %s | %d | %d | %d | %.3f | %.3f | %.3f | %+.3f |\n",
                    r.cohort, r.lead, r.n, r.n_groups, r.rmse_v2_0_nt,
                    r.rmse_v2_1_nt, r.rmse_persistence_nt,
                    r.improvement_vs_best_nt)
        end
    end
    println("Selected Operational V2.1 state-inertia weights: ", weights)
    show(stdout, MIME("text/plain"), evaluation)
    println()
    return candidates, evaluation, regimes, selected
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_state_inertia()
end
