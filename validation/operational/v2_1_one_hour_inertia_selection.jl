#!/usr/bin/env julia

# Select the one-hour V2.1 inertia blend without consulting the broad-replay
# test partition or the exact G-scale/severe evaluation cohorts.

using CSV
using DataFrames
using Printf
using Statistics

isdefined(@__MODULE__, :_selftest_v2) || include(joinpath(@__DIR__, "v2_replay.jl"))

const H1_BROAD_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_broad_replay_scored.csv")
const H1_GSCALE_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_gscale_replay_scored.csv")
const H1_SEVERE_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_scored.csv")
const H1_CANDIDATES_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_one_hour_inertia_candidates.csv")
const H1_EVALUATION_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_one_hour_inertia_evaluation.csv")
const H1_REPORT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_one_hour_inertia_report.md")
const H1_WEIGHT_GRID = collect(0.0:0.125:1.0)

_h1_rmse(pred, obs) = sqrt(mean((Float64.(pred) .- Float64.(obs)) .^ 2))

function _h1_rows(df::DataFrame)
    required = [:lead, :obs, :persistence, :v2_0, :v2_1_pre_one_hour_inertia]
    missing_cols = [String(c) for c in required if String(c) ∉ names(df)]
    isempty(missing_cols) || error("one-hour selection input omits: $(join(missing_cols, ", "))")
    rows = df[Int.(df.lead) .== 1, :]
    isempty(rows) && error("one-hour selection input contains no lead=1 rows")
    return rows
end

function _h1_candidate(rows::DataFrame, weight::Real)
    pred = _one_hour_inertia_blend.(
        rows.v2_1_pre_one_hour_inertia,
        rows.persistence,
        1;
        weight=Float64(weight),
    )
    extreme = Float64.(rows.persistence) .<= EXTREME_INERTIA_DST_NT
    pred[extreme] .= Float64.(rows.persistence[extreme])
    return pred
end

function _h1_metric(rows::DataFrame, weight::Real)
    pred = _h1_candidate(rows, weight)
    rv21 = _h1_rmse(pred, rows.obs)
    rv20 = _h1_rmse(rows.v2_0, rows.obs)
    rpers = _h1_rmse(rows.persistence, rows.obs)
    return (
        n_rows=nrow(rows),
        rmse_v2_1_nt=rv21,
        rmse_v2_0_nt=rv20,
        rmse_persistence_nt=rpers,
        improvement_vs_best_nt=min(rv20, rpers) - rv21,
    )
end

function select_one_hour_inertia(broad::DataFrame)
    hasproperty(broad, :storm_split) || error("broad replay omits storm_split")
    train = _h1_rows(broad[String.(broad.storm_split) .== "train", :])
    validation = _h1_rows(broad[String.(broad.storm_split) .== "val", :])
    out = DataFrame(
        weight=Float64[], train_n=Int[], validation_n=Int[],
        train_rmse_v2_1_nt=Float64[], train_rmse_v2_0_nt=Float64[],
        train_rmse_persistence_nt=Float64[], train_improvement_vs_best_nt=Float64[],
        validation_rmse_v2_1_nt=Float64[], validation_rmse_v2_0_nt=Float64[],
        validation_rmse_persistence_nt=Float64[],
        validation_improvement_vs_best_nt=Float64[], passes_development_gate=Bool[],
    )
    for weight in H1_WEIGHT_GRID
        tr = _h1_metric(train, weight)
        va = _h1_metric(validation, weight)
        passes = tr.improvement_vs_best_nt > 0.0 && va.improvement_vs_best_nt > 0.0
        push!(out, (
            weight, tr.n_rows, va.n_rows,
            tr.rmse_v2_1_nt, tr.rmse_v2_0_nt, tr.rmse_persistence_nt,
            tr.improvement_vs_best_nt,
            va.rmse_v2_1_nt, va.rmse_v2_0_nt, va.rmse_persistence_nt,
            va.improvement_vs_best_nt, passes,
        ))
    end
    eligible = out[out.passes_development_gate, :]
    isempty(eligible) && error("no one-hour inertia candidate passed both development partitions")
    sort!(eligible, [:validation_rmse_v2_1_nt, :weight])
    selected = eligible[1, :]
    Float64(selected.weight) == V2_ONE_HOUR_INERTIA_WEIGHT || error(
        "selected weight $(selected.weight) differs from deployed $(V2_ONE_HOUR_INERTIA_WEIGHT)",
    )
    return out, selected
end

function _h1_evaluation_row(label::AbstractString, rows::DataFrame, weight::Real)
    h1 = _h1_rows(rows)
    metric = _h1_metric(h1, weight)
    n_groups = hasproperty(h1, :storm_id) ? length(unique(Int.(h1.storm_id))) :
               hasproperty(h1, :g_event_id) ? length(unique(Int.(h1.g_event_id))) :
               hasproperty(h1, :storm) ? length(unique(String.(h1.storm))) : 0
    return (
        cohort=String(label), n_rows=metric.n_rows, n_groups=n_groups,
        rmse_v2_0_nt=metric.rmse_v2_0_nt,
        rmse_v2_1_nt=metric.rmse_v2_1_nt,
        rmse_persistence_nt=metric.rmse_persistence_nt,
        improvement_vs_best_nt=metric.improvement_vs_best_nt,
    )
end

function main_one_hour_inertia()
    broad = CSV.read(H1_BROAD_CSV, DataFrame)
    candidates, selected = select_one_hour_inertia(broad)
    weight = Float64(selected.weight)

    evaluation = DataFrame()
    append!(evaluation, DataFrame([
        _h1_evaluation_row("broad_train", broad[String.(broad.storm_split) .== "train", :], weight),
        _h1_evaluation_row("broad_validation", broad[String.(broad.storm_split) .== "val", :], weight),
        _h1_evaluation_row("broad_test", broad[String.(broad.storm_split) .== "test", :], weight),
        _h1_evaluation_row("broad_all", broad, weight),
        _h1_evaluation_row("exact_G3plus", CSV.read(H1_GSCALE_CSV, DataFrame), weight),
        _h1_evaluation_row("severe_development_stress", CSV.read(H1_SEVERE_CSV, DataFrame), weight),
    ]); cols=:union)

    all(evaluation.improvement_vs_best_nt .> 0.0) || error(
        "selected one-hour inertia weight fails an evaluation cohort",
    )
    CSV.write(H1_CANDIDATES_CSV, candidates)
    CSV.write(H1_EVALUATION_CSV, evaluation)
    open(H1_REPORT, "w") do io
        println(io, "# Operational V2.1 one-hour inertia selection\n")
        @printf(io, "A fixed weight of %.3f was selected by minimum validation RMSE among %d/%d candidates that beat both historical V2.0 and persistence on the broad-replay training and validation partitions. The broad test partition, exact G3+ replay, and severe-storm cohort were scored only after selection.\n\n",
                weight, count(candidates.passes_development_gate), nrow(candidates))
        println(io, "The operator is `latest Dst + weight × (V2.1 − latest Dst)` at one hour only. It uses no V2.0 value at issue time; historical V2.0 appears only in offline scoring.\n")
        println(io, "| cohort | rows | groups | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | improvement vs stronger comparator |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(evaluation)
            @printf(io, "| %s | %d | %d | %.3f | %.3f | %.3f | %+.3f |\n",
                    r.cohort, r.n_rows, r.n_groups, r.rmse_v2_0_nt,
                    r.rmse_v2_1_nt, r.rmse_persistence_nt,
                    r.improvement_vs_best_nt)
        end
    end
    println("Selected Operational V2.1 one-hour inertia weight: ", weight)
    show(stdout, MIME("text/plain"), evaluation)
    println()
    return candidates, evaluation, selected
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_one_hour_inertia()
end
