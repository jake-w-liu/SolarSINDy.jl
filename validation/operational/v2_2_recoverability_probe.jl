#!/usr/bin/env julia

# Development-only recoverability probe. This intentionally combines the
# noncausal realized-driver oracle with causal, matured forecast innovations to
# test an upper bound on the proposed M2 + error-state mechanism. It reads only
# two pinned pre-2023 OOF tables and writes no artifacts. Its output is never a
# deployable forecast or promotion result.

using SolarSINDy
using CSV
using DataFrames
using Dates
using EvoTrees
using EvoTrees: fit
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "v2_2_temporal_probe.jl"))

const V22_RECOVERABILITY_HISTORY_OOF = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "v2_2_history",
    "v2_2_history_selected_oof.csv",
))
const V22_RECOVERABILITY_HISTORY_OOF_SHA256 =
    "786b0344a18eb6facd9637a5c6a4eb00870fb2aa3e2af1b30d212558d097cd0f"
const V22_RECOVERABILITY_COMPARATORS = (
    :served_v2_1_dst_nt,
    :frozen_v2_1_dst_nt,
    :raw_sindy_dst_nt,
    :persistence_dst_nt,
    :burton_dst_nt,
    :burton_full_dst_nt,
    :obrien_dst_nt,
)
const V22_RECOVERABILITY_EFFECT_GATE_NT = 0.25
const V22_RECOVERABILITY_BOOTSTRAP_REPLICATES = 10_000
const V22_RECOVERABILITY_BOOTSTRAP_SEED = 22_022_026
const V22_RECOVERABILITY_BOOTSTRAP_BLOCK_H = 168

function _v22_recoverability_history_table(
        path::AbstractString=V22_RECOVERABILITY_HISTORY_OOF)
    isfile(path) && !islink(path) || error(
        "recoverability probe requires a regular pinned history OOF table",
    )
    _v22_temporal_sha256(path) == V22_RECOVERABILITY_HISTORY_OOF_SHA256 ||
        error("recoverability history OOF hash changed")
    columns = [
        :fold, :issue_time_utc, :target_time_utc, :model_step_hours,
        :noncausal_input_oracle_dst_nt,
    ]
    table = CSV.read(path, DataFrame; select=String.(columns), types=Dict(
        :issue_time_utc => DateTime, :target_time_utc => DateTime,
    ))
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) ||
        error("recoverability history OOF contains a post-2022 target")
    all(isfinite, Float64.(table.noncausal_input_oracle_dst_nt)) ||
        error("recoverability oracle predictions must be finite")
    return table
end

function _v22_recoverability_join(primary::DataFrame, history::DataFrame)
    keys = [:issue_time_utc, :target_time_utc, :model_step_hours]
    for (label, table) in (("primary", primary), ("history", history))
        missing = setdiff(keys, Symbol.(names(table)))
        isempty(missing) || error("$label OOF table omits join keys")
        tuples = Tuple.(eachrow(select(table, keys)))
        length(unique(tuples)) == nrow(table) ||
            error("$label OOF table contains duplicate keys")
    end
    joined = innerjoin(primary, history; on=keys, validate=(true, true))
    nrow(joined) > 0 || error("recoverability OOF tables have no common rows")
    all(String.(joined.v2_2_crossfit_fold) .== String.(joined.fold)) ||
        error("recoverability OOF fold labels disagree")
    maximum(joined.target_time_utc) < DateTime(2023, 1, 1) ||
        error("recoverability join contains a post-2022 target")
    steps = sort!(unique(Int.(joined.model_step_hours)))
    steps == collect((1, 2, 3, 4, 6, 7)) ||
        error("recoverability join does not cover all six model steps")
    counts = combine(groupby(joined, :issue_time_utc), nrow => :rows)
    all(==(length(steps)), counts.rows) ||
        error("recoverability join does not retain whole six-lead anchors")
    return joined
end

function _v22_recoverability_add_oracle_lags(
        table::DataFrame; lags_h::Tuple=V22_TEMPORAL_LAGS_H)
    required = (
        :issue_time_utc, :target_time_utc, :model_step_hours,
        :observation_dst_nt, :noncausal_input_oracle_dst_nt,
    )
    all(name -> String(name) in names(table), required) ||
        error("recoverability table omits oracle-lag inputs")
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) ||
        error("oracle-lag construction refuses post-2022 targets")
    all(lag -> lag isa Integer && lag > 0, lags_h) ||
        throw(ArgumentError("oracle innovation lags must be positive integers"))
    length(unique(lags_h)) == length(lags_h) ||
        throw(ArgumentError("oracle innovation lags must be unique"))

    anchors = table[table.model_step_hours .== 1, :]
    length(unique(anchors.issue_time_utc)) == nrow(anchors) ||
        error("recoverability table has duplicate one-step anchors")
    observed = Float64.(anchors.observation_dst_nt)
    oracle = Float64.(anchors.noncausal_input_oracle_dst_nt)
    all(isfinite, observed) && all(isfinite, oracle) ||
        error("oracle innovation inputs must be finite")
    innovation_at = Dict(
        DateTime(anchors.issue_time_utc[i]) => observed[i] - oracle[i]
        for i in 1:nrow(anchors)
    )
    keep = [
        all(haskey(innovation_at, DateTime(issue) - Hour(lag)) for lag in lags_h)
        for issue in table.issue_time_utc
    ]
    enriched = copy(table[keep, :])
    for lag in lags_h
        name = Symbol(:oracle_h1_innovation_nt_lag_, lag, :h)
        enriched[!, name] = Float64[
            innovation_at[DateTime(issue) - Hour(lag)]
            for issue in enriched.issue_time_utc
        ]
    end
    return enriched
end

function _v22_recoverability_table()
    primary, primary_features = _v22_temporal_table()
    history = _v22_recoverability_history_table()
    joined = _v22_recoverability_join(primary, history)
    table = _v22_recoverability_add_oracle_lags(joined)
    oracle_lags = [
        Symbol(:oracle_h1_innovation_nt_lag_, lag, :h)
        for lag in V22_TEMPORAL_LAGS_H
    ]
    features = vcat(
        primary_features,
        [:noncausal_input_oracle_dst_nt],
        oracle_lags,
    )
    length(unique(features)) == length(features) ||
        error("recoverability feature schema contains duplicates")
    all(feature -> String(feature) in names(table), features) ||
        error("recoverability feature construction is incomplete")
    all(isfinite, Matrix{Float64}(table[:, features])) ||
        error("recoverability feature matrix is non-finite")
    return table, features
end

function _v22_recoverability_probe(table::DataFrame, features, lead::Int)
    lead_table = table[table.model_step_hours .== lead, :]
    configs = [
        (depth, rounds, rho)
        for depth in V22_TEMPORAL_DEPTHS
        for rounds in V22_TEMPORAL_ROUNDS
        for rho in V22_TEMPORAL_SHRINKAGES
    ]
    predictions = Dict(config => Float64[] for config in configs)
    evaluation_issues = DateTime[]
    observed = Float64[]
    oracle_base = Float64[]
    regimes = String[]

    for fold in V22_TEMPORAL_FOLDS
        year = parse(Int, last(split(fold, '_')))
        train_folds = Set("calendar_$y" for y in 2013:(year - 1))
        train_rows = findall(in(train_folds), String.(lead_table.v2_2_crossfit_fold))
        eval_rows = findall(==(fold), String.(lead_table.v2_2_crossfit_fold))
        isempty(train_rows) && error("empty recoverability training fold $fold")
        isempty(eval_rows) && error("empty recoverability evaluation fold $fold")
        x_train = Matrix{Float64}(lead_table[train_rows, features])
        x_eval = Matrix{Float64}(lead_table[eval_rows, features])
        fold_observed = Float64.(lead_table.observation_dst_nt[eval_rows])
        fold_base = Float64.(lead_table.noncausal_input_oracle_dst_nt[eval_rows])
        y_train = Float64.(lead_table.observation_dst_nt[train_rows]) .-
                  Float64.(lead_table.noncausal_input_oracle_dst_nt[train_rows])
        append!(observed, fold_observed)
        append!(oracle_base, fold_base)
        append!(evaluation_issues, DateTime.(lead_table.issue_time_utc[eval_rows]))
        append!(regimes, String.(lead_table.v2_2_regime[eval_rows]))

        for depth in V22_TEMPORAL_DEPTHS
            model = fit(
                _v22_temporal_config(depth); x_train, y_train, verbosity=0,
                feature_names=String.(features),
            )
            for rounds in V22_TEMPORAL_ROUNDS
                raw = Float64.(model(x_eval; ntree_limit=rounds + 1))
                cap = 5.0 + 5.0lead
                for rho in V22_TEMPORAL_SHRINKAGES
                    append!(predictions[(depth, rounds, rho)],
                            fold_base .+ rho .* clamp.(raw, -cap, cap))
                end
            end
        end
    end

    best = nothing
    for config in configs
        candidate = predictions[config]
        gains = _v22_temporal_metrics(observed, oracle_base, candidate, regimes)
        safe = all(get(gains, regime, 0.0) >= 0.0
                   for regime in ("quiet", "active_deepening", "recovery"))
        record = (; config, safe, gains,
                  rmse=_v22_temporal_rmse(observed, candidate))
        if safe && (best === nothing || record.rmse < best.rmse)
            best = record
        end
    end
    best === nothing && error("lead $lead has no regime-safe recoverability setting")

    evaluation_rows = findall(
        in(Set(V22_TEMPORAL_FOLDS)), String.(lead_table.v2_2_crossfit_fold),
    )
    DateTime.(lead_table.issue_time_utc[evaluation_rows]) == evaluation_issues ||
        error("recoverability evaluation-row order changed")
    comparator_metrics = [
        (name, _v22_temporal_rmse(
            observed, Float64.(lead_table[evaluation_rows, name]),
        ))
        for name in V22_RECOVERABILITY_COMPARATORS
    ]
    sort!(comparator_metrics; by=item -> (item[2], String(item[1])))
    best_name, best_comparator_rmse = first(comparator_metrics)
    oracle_rmse = _v22_temporal_rmse(observed, oracle_base)
    gain_vs_best = best_comparator_rmse - best.rmse
    result = (
        lead_h=lead,
        rows=length(observed),
        config=best.config,
        best_comparator=best_name,
        best_comparator_rmse_nt=best_comparator_rmse,
        realized_driver_oracle_rmse_nt=oracle_rmse,
        oracle_plus_error_state_rmse_nt=best.rmse,
        gain_vs_oracle_nt=oracle_rmse - best.rmse,
        gain_vs_best_nt=gain_vs_best,
        passes_effect_gate=gain_vs_best >= V22_RECOVERABILITY_EFFECT_GATE_NT,
        quiet_gain_vs_oracle_nt=best.gains["quiet"],
        active_gain_vs_oracle_nt=best.gains["active_deepening"],
        recovery_gain_vs_oracle_nt=best.gains["recovery"],
        bootstrap_payload=(
            issues=evaluation_issues,
            observed=copy(observed),
            candidate=copy(predictions[best.config]),
            comparator=Float64.(lead_table[evaluation_rows, best_name]),
        ),
    )
    println(join((
        "lead=$(result.lead_h)",
        "rows=$(result.rows)",
        "config=$(result.config)",
        "best=$(result.best_comparator)",
        "best_rmse=$(result.best_comparator_rmse_nt)",
        "oracle_rmse=$(result.realized_driver_oracle_rmse_nt)",
        "combined_rmse=$(result.oracle_plus_error_state_rmse_nt)",
        "gain_vs_best=$(result.gain_vs_best_nt)",
        "pass=$(result.passes_effect_gate)",
    ), " "))
    return result
end

function _v22_recoverability_bootstrap(
        results;
        replicates::Int=V22_RECOVERABILITY_BOOTSTRAP_REPLICATES,
        seed::Int=V22_RECOVERABILITY_BOOTSTRAP_SEED,
        block_hours::Int=V22_RECOVERABILITY_BOOTSTRAP_BLOCK_H)
    !isempty(results) || throw(ArgumentError(
        "recoverability bootstrap requires at least one lead result",
    ))
    replicates >= 1 || throw(ArgumentError(
        "recoverability bootstrap replicates must be positive",
    ))
    seed >= 0 || throw(ArgumentError(
        "recoverability bootstrap seed must be nonnegative",
    ))
    block_hours >= 1 || throw(ArgumentError(
        "recoverability bootstrap block_hours must be positive",
    ))

    reference_issues = results[1].bootstrap_payload.issues
    !isempty(reference_issues) || error("recoverability bootstrap has no issue rows")
    issorted(reference_issues) || error(
        "recoverability bootstrap issues must be sorted",
    )
    length(unique(reference_issues)) == length(reference_issues) || error(
        "recoverability bootstrap issues must be unique within a lead",
    )
    for result in results
        payload = result.bootstrap_payload
        payload.issues == reference_issues || error(
            "recoverability bootstrap lead issue rows are not identical",
        )
        n = length(reference_issues)
        all(length(values) == n for values in (
            payload.observed, payload.candidate, payload.comparator,
        )) || throw(DimensionMismatch(
            "recoverability bootstrap payload lengths disagree",
        ))
        all(isfinite, payload.observed) && all(isfinite, payload.candidate) &&
            all(isfinite, payload.comparator) || error(
            "recoverability bootstrap payload is non-finite",
        )
    end

    origin = first(reference_issues)
    block_ms = block_hours * 60 * 60 * 1_000
    labels = [
        div(Dates.value(issue - origin), block_ms)
        for issue in reference_issues
    ]
    unique_labels = sort!(unique(labels))
    block_rows = [findall(==(label), labels) for label in unique_labels]
    isempty(block_rows) && error("recoverability bootstrap has no blocks")

    lead_summaries = [
        (
            candidate_sse=[
                sum(abs2, result.bootstrap_payload.observed[rows] .-
                          result.bootstrap_payload.candidate[rows])
                for rows in block_rows
            ],
            comparator_sse=[
                sum(abs2, result.bootstrap_payload.observed[rows] .-
                          result.bootstrap_payload.comparator[rows])
                for rows in block_rows
            ],
            counts=length.(block_rows),
        )
        for result in results
    ]
    rng = MersenneTwister(seed)
    draws = Matrix{Float64}(undef, replicates, length(results))
    sampled_blocks = Vector{Int}(undef, length(block_rows))
    for replicate in 1:replicates
        rand!(rng, sampled_blocks, 1:length(block_rows))
        for (lead_index, summary) in enumerate(lead_summaries)
            count_rows = sum(summary.counts[index] for index in sampled_blocks)
            candidate_sse = sum(
                summary.candidate_sse[index] for index in sampled_blocks
            )
            comparator_sse = sum(
                summary.comparator_sse[index] for index in sampled_blocks
            )
            draws[replicate, lead_index] =
                sqrt(comparator_sse / count_rows) -
                sqrt(candidate_sse / count_rows)
        end
    end
    per_lead_lower_95_nt = [
        quantile(view(draws, :, lead_index), 0.05)
        for lead_index in axes(draws, 2)
    ]
    simultaneous_lower_95_nt = quantile(
        [minimum(view(draws, replicate, :)) for replicate in axes(draws, 1)],
        0.05,
    )
    return (
        per_lead_lower_95_nt=per_lead_lower_95_nt,
        simultaneous_lower_95_nt=simultaneous_lower_95_nt,
        replicates=replicates,
        blocks=length(block_rows),
        seed=seed,
        block_hours=block_hours,
    )
end

function main()
    table, features = _v22_recoverability_table()
    println(
        "diagnostic_only=true rows=$(nrow(table)) features=$(length(features)) " *
        "max_target=$(maximum(table.target_time_utc))",
    )
    results = [
        _v22_recoverability_probe(table, features, lead)
        for lead in (1, 2, 3, 4, 6, 7)
    ]
    inference = _v22_recoverability_bootstrap(results)
    println("all_effect_gates=$(all(result.passes_effect_gate for result in results))")
    println(
        "bootstrap_replicates=$(inference.replicates) " *
        "bootstrap_blocks=$(inference.blocks) " *
        "per_lead_lower_95_nt=$(join(inference.per_lead_lower_95_nt, ';')) " *
        "simultaneous_lower_95_nt=$(inference.simultaneous_lower_95_nt)",
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
