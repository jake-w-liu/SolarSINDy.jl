#!/usr/bin/env julia

# Select and audit the causal rapid-deepening projection guard for Operational
# V2.1. Candidate selection uses the broad-replay train and validation splits
# plus the explicit seven-storm development stress cohort. Broad test metrics
# are computed only after the candidate is locked.

using CSV
using DataFrames
using Printf
using Statistics

include(joinpath(@__DIR__, "v2_replay.jl"))

const RATE_GUARD_BROAD_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_scored.csv")
const RATE_GUARD_SEVERE_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_scored.csv")
const RATE_GUARD_CANDIDATES_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_rate_guard_candidates.csv")
const RATE_GUARD_SPLITS_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_rate_guard_split_metrics.csv")
const RATE_GUARD_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_rate_guard_report.md")
const RATE_GUARD_BIAS_LIMIT_NT = 10.0
const RATE_GUARD_RECENT_SEVERE_IDS = Set([699, 710, 717, 719, 739])

_guard_raw_column(df::DataFrame) =
    "v2_1_pre_rate_guard" in names(df) ? :v2_1_pre_rate_guard : :v2_1

function _guard_predictions(df::DataFrame, factor::Real, max_drop::Real,
                            extreme_max_drop::Real, extreme_rate::Real)
    raw = Float64.(df[!, _guard_raw_column(df)])
    out = copy(raw)
    for i in eachindex(out)
        h = Int(df.lead[i])
        latest = Float64(df.persistence[i])
        rate = Float64(df.rate[i])
        if _extreme_inertia_guard(latest, h)
            out[i] = latest
        else
            out[i] = _rapid_deepening_projection_guard(
                raw[i], latest, h, rate;
                factor=factor,
                max_drop=max_drop,
                extreme_max_drop=extreme_max_drop,
                extreme_rate=extreme_rate,
            )
        end
    end
    return out
end

_guard_rmse(errors) = sqrt(mean(abs2, Float64.(errors)))

function _guard_metrics(df::DataFrame, pred::AbstractVector, mask::AbstractVector{Bool})
    length(pred) == nrow(df) == length(mask) || throw(DimensionMismatch(
        "prediction, replay, and mask lengths differ",
    ))
    lead_rmse = Float64[]
    for h in LEADS
        rows = mask .& (Int.(df.lead) .== h)
        any(rows) || error("cohort has no rows at lead $h")
        push!(lead_rmse, _guard_rmse(Float64.(df.obs[rows]) .- pred[rows]))
    end
    deep = mask .& (Int.(df.lead) .== 6) .& isfinite.(Float64.(df.rate)) .&
           (Float64.(df.rate) .< MAIN_RATE) .& (Float64.(df.obs) .< -100.0)
    any(deep) || error("cohort has no deep-deepening 6 h rows")
    residual = Float64.(df.obs[deep]) .- pred[deep]
    return (
        n_rows=count(mask),
        n_storms=length(unique(String.(df.storm[mask]))),
        deep_n=count(deep),
        mean_lead_rmse_nt=mean(lead_rmse),
        deep_bias_nt=mean(residual),
        deep_rmse_nt=_guard_rmse(residual),
    )
end

function _candidate_passes(candidate, train_base, validation_base, severe_base)
    return abs(candidate.train.deep_bias_nt) <= RATE_GUARD_BIAS_LIMIT_NT &&
           candidate.train.mean_lead_rmse_nt <= train_base.mean_lead_rmse_nt &&
           candidate.train.deep_rmse_nt <= train_base.deep_rmse_nt &&
           abs(candidate.validation.deep_bias_nt) <= RATE_GUARD_BIAS_LIMIT_NT &&
           candidate.validation.mean_lead_rmse_nt <= validation_base.mean_lead_rmse_nt &&
           candidate.validation.deep_rmse_nt <= validation_base.deep_rmse_nt &&
           abs(candidate.severe.deep_bias_nt) <= RATE_GUARD_BIAS_LIMIT_NT &&
           candidate.severe.mean_lead_rmse_nt <= severe_base.mean_lead_rmse_nt &&
           candidate.severe.deep_rmse_nt <= severe_base.deep_rmse_nt
end

function select_v2_1_rate_guard(broad::DataFrame, severe::DataFrame)
    required_broad = [:storm, :storm_id, :storm_split, :lead, :obs, :persistence, :rate]
    required_severe = [:storm, :lead, :obs, :persistence, :rate]
    all(String(c) in names(broad) for c in required_broad) || error("broad replay schema drift")
    all(String(c) in names(severe) for c in required_severe) || error("severe replay schema drift")
    String(_guard_raw_column(broad)) in names(broad) || error("broad replay omits raw V2.1 center")
    String(_guard_raw_column(severe)) in names(severe) || error("severe replay omits raw V2.1 center")

    train_mask = String.(broad.storm_split) .== "train"
    validation_mask = String.(broad.storm_split) .== "val"
    test_mask = String.(broad.storm_split) .== "test"
    all_mask = trues(nrow(broad))
    severe_mask = trues(nrow(severe))
    raw_broad = Float64.(broad[!, _guard_raw_column(broad)])
    raw_severe = Float64.(severe[!, _guard_raw_column(severe)])
    baselines = (
        train=_guard_metrics(broad, raw_broad, train_mask),
        validation=_guard_metrics(broad, raw_broad, validation_mask),
        test=_guard_metrics(broad, raw_broad, test_mask),
        all=_guard_metrics(broad, raw_broad, all_mask),
        severe=_guard_metrics(severe, raw_severe, severe_mask),
    )

    candidates = NamedTuple[]
    for factor in 0.2:0.025:0.4,
        max_drop in (30.0, 40.0, 50.0),
        extreme_max_drop in (70.0, 80.0, 100.0, 120.0),
        extreme_rate in (-40.0, -50.0, -60.0)
        broad_pred = _guard_predictions(
            broad, factor, max_drop, extreme_max_drop, extreme_rate,
        )
        severe_pred = _guard_predictions(
            severe, factor, max_drop, extreme_max_drop, extreme_rate,
        )
        metrics = (
            train=_guard_metrics(broad, broad_pred, train_mask),
            validation=_guard_metrics(broad, broad_pred, validation_mask),
            severe=_guard_metrics(severe, severe_pred, severe_mask),
        )
        passed = _candidate_passes(
            metrics, baselines.train, baselines.validation, baselines.severe,
        )
        push!(candidates, (
            factor=Float64(factor), max_drop_nt=max_drop,
            extreme_max_drop_nt=extreme_max_drop,
            extreme_rate_nt_per_h=extreme_rate,
            train_mean_lead_rmse_nt=metrics.train.mean_lead_rmse_nt,
            train_deep_bias_nt=metrics.train.deep_bias_nt,
            train_deep_rmse_nt=metrics.train.deep_rmse_nt,
            validation_mean_lead_rmse_nt=metrics.validation.mean_lead_rmse_nt,
            validation_deep_bias_nt=metrics.validation.deep_bias_nt,
            validation_deep_rmse_nt=metrics.validation.deep_rmse_nt,
            severe_mean_lead_rmse_nt=metrics.severe.mean_lead_rmse_nt,
            severe_deep_bias_nt=metrics.severe.deep_bias_nt,
            severe_deep_rmse_nt=metrics.severe.deep_rmse_nt,
            passes_development_gates=passed,
        ))
    end
    table = DataFrame(candidates)
    passing = table[table.passes_development_gates, :]
    nrow(passing) > 0 || error("no rate-projection candidate passes development gates")
    sort!(passing, [:validation_mean_lead_rmse_nt, :validation_deep_rmse_nt,
                    :factor, :max_drop_nt, :extreme_max_drop_nt,
                    :extreme_rate_nt_per_h])
    selected = passing[1, :]
    exact = (
        Float64(selected.factor) == V2_RATE_PROJECTION_FACTOR &&
        Float64(selected.max_drop_nt) == V2_RATE_PROJECTION_MAX_DROP_NT &&
        Float64(selected.extreme_max_drop_nt) == V2_RATE_PROJECTION_EXTREME_MAX_DROP_NT &&
        Float64(selected.extreme_rate_nt_per_h) == V2_RATE_PROJECTION_EXTREME_RATE_NT_PER_H
    )
    exact || error(
        "selected rate guard factor=$(selected.factor), max_drop=$(selected.max_drop_nt), " *
        "extreme_max_drop=$(selected.extreme_max_drop_nt), " *
        "extreme_rate=$(selected.extreme_rate_nt_per_h) does not match deployed constants",
    )
    table[!, :selected_by_validation] = falses(nrow(table))
    idx = findfirst(
        (table.factor .== selected.factor) .&
        (table.max_drop_nt .== selected.max_drop_nt) .&
        (table.extreme_max_drop_nt .== selected.extreme_max_drop_nt) .&
        (table.extreme_rate_nt_per_h .== selected.extreme_rate_nt_per_h),
    )
    idx === nothing && error("selected candidate disappeared from audit table")
    table.selected_by_validation[idx] = true

    selected_broad = _guard_predictions(
        broad, selected.factor, selected.max_drop_nt,
        selected.extreme_max_drop_nt, selected.extreme_rate_nt_per_h,
    )
    selected_severe = _guard_predictions(
        severe, selected.factor, selected.max_drop_nt,
        selected.extreme_max_drop_nt, selected.extreme_rate_nt_per_h,
    )
    nonoverlap = test_mask .& .!in.(Int.(broad.storm_id), Ref(RATE_GUARD_RECENT_SEVERE_IDS))
    split_rows = NamedTuple[]
    for (cohort, df, raw, guarded, mask) in (
        ("broad_train", broad, raw_broad, selected_broad, train_mask),
        ("broad_validation", broad, raw_broad, selected_broad, validation_mask),
        ("broad_test", broad, raw_broad, selected_broad, test_mask),
        ("broad_test_nonoverlap", broad, raw_broad, selected_broad, nonoverlap),
        ("broad_all", broad, raw_broad, selected_broad, all_mask),
        ("severe_development_stress", severe, raw_severe, selected_severe, severe_mask),
    )
        before = _guard_metrics(df, raw, mask)
        after = _guard_metrics(df, guarded, mask)
        push!(split_rows, (
            cohort=cohort, n_rows=after.n_rows, n_storms=after.n_storms,
            deep_n=after.deep_n,
            raw_mean_lead_rmse_nt=before.mean_lead_rmse_nt,
            guarded_mean_lead_rmse_nt=after.mean_lead_rmse_nt,
            raw_deep_bias_nt=before.deep_bias_nt,
            guarded_deep_bias_nt=after.deep_bias_nt,
            raw_deep_rmse_nt=before.deep_rmse_nt,
            guarded_deep_rmse_nt=after.deep_rmse_nt,
        ))
    end
    return table, DataFrame(split_rows), selected
end

function write_rate_guard_report(path::AbstractString, selected, splits::DataFrame,
                                 candidates::DataFrame)
    open(path, "w") do io
        println(io, "# Operational V2.1 rapid-deepening projection selection\n")
        @printf(io, "Selected by minimum validation mean lead-wise RMSE among %d/%d candidates that passed the unchanged development gates: factor %.3f, moderate cap %.0f nT, extreme cap %.0f nT, extreme-rate threshold %.0f nT/h.\n\n",
                count(candidates.passes_development_gates), nrow(candidates),
                selected.factor, selected.max_drop_nt, selected.extreme_max_drop_nt,
                selected.extreme_rate_nt_per_h)
        println(io, "The broad test split is scored only after selection. The seven-storm stress cohort is a development gate and overlaps part of the recent broad test period; the non-overlap test row is therefore reported separately.\n")
        println(io, "| cohort | rows | storms | deep n | mean lead RMSE raw | mean lead RMSE guarded | deep bias raw | deep bias guarded | deep RMSE raw | deep RMSE guarded |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(splits)
            @printf(io, "| %s | %d | %d | %d | %.3f | %.3f | %+.2f | %+.2f | %.2f | %.2f |\n",
                    r.cohort, r.n_rows, r.n_storms, r.deep_n,
                    r.raw_mean_lead_rmse_nt, r.guarded_mean_lead_rmse_nt,
                    r.raw_deep_bias_nt, r.guarded_deep_bias_nt,
                    r.raw_deep_rmse_nt, r.guarded_deep_rmse_nt)
        end
    end
end

function main_rate_guard_selection()
    broad = CSV.read(RATE_GUARD_BROAD_CSV, DataFrame)
    severe = CSV.read(RATE_GUARD_SEVERE_CSV, DataFrame)
    candidates, splits, selected = select_v2_1_rate_guard(broad, severe)
    CSV.write(RATE_GUARD_CANDIDATES_CSV, candidates)
    CSV.write(RATE_GUARD_SPLITS_CSV, splits)
    write_rate_guard_report(RATE_GUARD_REPORT, selected, splits, candidates)
    println("Selected Operational V2.1 rate projection: factor=$(selected.factor), " *
            "caps=$(selected.max_drop_nt)/$(selected.extreme_max_drop_nt) nT, " *
            "extreme rate=$(selected.extreme_rate_nt_per_h) nT/h")
    println(splits)
    return selected
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_rate_guard_selection()
end
