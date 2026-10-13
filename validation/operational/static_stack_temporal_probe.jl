#!/usr/bin/env julia

# Development-only probe for the hypothesis that the failed static stack
# residual omitted causal magnetospheric history. It reads only the pinned
# pre-2023 primary cross-fit table and writes no artifacts.

using SolarSINDy
using CSV
using DataFrames
using Dates
using EvoTrees
using EvoTrees: fit
using SHA
using Statistics

const STATIC_STACK_TEMPORAL_OOF = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "static_stack_crossfit",
    "static_stack_primary_crossfit_oof.csv",
))
const STATIC_STACK_TEMPORAL_OOF_SHA256 =
    "5e78d1e9e23824cab48249e977646c2fe73ea5798065d977a0ed9b1143f77d40"
const STATIC_STACK_TEMPORAL_LAGS_H = (1, 2, 3, 4, 6, 9, 12, 18, 24)
const STATIC_STACK_TEMPORAL_VARIABLES = (
    :latest_dst_nt,
    :Bz_nt,
    :VBsouth_mvm,
    :sqrt_Pdyn_npa,
    :h1_innovation_nt,
)
const STATIC_STACK_TEMPORAL_FOLDS = (
    "calendar_2014", "calendar_2015", "calendar_2016", "calendar_2017",
)
const STATIC_STACK_TEMPORAL_DEPTHS = (3, 4)
const STATIC_STACK_TEMPORAL_ROUNDS = (64, 128, 256)
const STATIC_STACK_TEMPORAL_SHRINKAGES = (0.25, 0.50, 0.75, 1.00)
const STATIC_STACK_TEMPORAL_SEED = 22_022_026

_static_stack_temporal_sha256(path) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _static_stack_temporal_features()
    names = collect(STATIC_STACK_RESIDUAL_FEATURES)
    for variable in STATIC_STACK_TEMPORAL_VARIABLES, lag in STATIC_STACK_TEMPORAL_LAGS_H
        push!(names, Symbol(variable, "_lag_", lag, "h"))
    end
    return names
end

function _static_stack_temporal_table()
    isfile(STATIC_STACK_TEMPORAL_OOF) && !islink(STATIC_STACK_TEMPORAL_OOF) ||
        error("temporal probe requires the regular pinned OOF table")
    _static_stack_temporal_sha256(STATIC_STACK_TEMPORAL_OOF) == STATIC_STACK_TEMPORAL_OOF_SHA256 ||
        error("temporal probe OOF hash changed")
    table = CSV.read(STATIC_STACK_TEMPORAL_OOF, DataFrame; types=Dict(
        :issue_time_utc => DateTime, :target_time_utc => DateTime,
    ))
    SolarSINDy.canonicalize_static_stack_frozen_table!(table)
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) ||
        error("temporal probe refuses post-2022 targets")
    anchors = table[table.model_step_hours .== 1, :]
    anchors[!, :h1_innovation_nt] = Float64.(anchors.observation_dst_nt) .-
                                     Float64.(anchors.static_stack_pred_dst_nt)
    row_at = Dict(DateTime(anchors[i, :issue_time_utc]) => i for i in 1:nrow(anchors))
    keep_anchor = Dict(
        time => all(haskey(row_at, time - Hour(lag)) for lag in STATIC_STACK_TEMPORAL_LAGS_H)
        for time in keys(row_at)
    )
    keep = [get(keep_anchor, DateTime(t), false) for t in table.issue_time_utc]
    table = table[keep, :]
    for variable in STATIC_STACK_TEMPORAL_VARIABLES, lag in STATIC_STACK_TEMPORAL_LAGS_H
        name = Symbol(variable, "_lag_", lag, "h")
        table[!, name] = Float64[
            anchors[row_at[DateTime(t) - Hour(lag)], variable]
            for t in table.issue_time_utc
        ]
    end
    features = _static_stack_temporal_features()
    all(name -> String(name) in names(table), features) ||
        error("temporal feature construction is incomplete")
    all(isfinite, Matrix{Float64}(table[:, features])) ||
        error("temporal feature matrix is non-finite")
    return table, features
end

function _static_stack_temporal_config(depth::Int)
    return EvoTreeRegressor(
        loss=:mse, metric=:rmse, nrounds=maximum(STATIC_STACK_TEMPORAL_ROUNDS),
        eta=0.03, L2=2.0, lambda=10.0, gamma=0.0, max_depth=depth,
        min_weight=96.0, rowsample=1.0, colsample=1.0, nbins=64,
        tree_type=:binary, seed=STATIC_STACK_TEMPORAL_SEED, device=:cpu,
    )
end

_static_stack_temporal_rmse(observed, predicted) =
    sqrt(mean(abs2, observed .- predicted))

function _static_stack_temporal_metrics(observed, base, candidate, regimes)
    gains = Dict{String,Float64}(
        "overall" => _static_stack_temporal_rmse(observed, base) -
                     _static_stack_temporal_rmse(observed, candidate),
    )
    for regime in ("quiet", "active_deepening", "recovery")
        rows = findall(==(regime), regimes)
        isempty(rows) && continue
        gains[regime] = _static_stack_temporal_rmse(observed[rows], base[rows]) -
                        _static_stack_temporal_rmse(observed[rows], candidate[rows])
    end
    return gains
end

function _static_stack_temporal_probe(table::DataFrame, features, lead::Int)
    lead_table = table[table.model_step_hours .== lead, :]
    predictions = Dict(
        (depth, rounds, rho) => Float64[]
        for depth in STATIC_STACK_TEMPORAL_DEPTHS for rounds in STATIC_STACK_TEMPORAL_ROUNDS
        for rho in STATIC_STACK_TEMPORAL_SHRINKAGES
    )
    observed = Float64[]
    base = Float64[]
    regimes = String[]
    for fold in STATIC_STACK_TEMPORAL_FOLDS
        year = parse(Int, last(split(fold, '_')))
        train_folds = Set("calendar_$y" for y in 2013:(year - 1))
        train_rows = findall(in(train_folds), String.(lead_table.static_stack_crossfit_fold))
        eval_rows = findall(==(fold), String.(lead_table.static_stack_crossfit_fold))
        isempty(train_rows) && error("empty temporal training fold $fold")
        isempty(eval_rows) && error("empty temporal evaluation fold $fold")
        x_train = Matrix{Float64}(lead_table[train_rows, features])
        x_eval = Matrix{Float64}(lead_table[eval_rows, features])
        y_train = Float64.(lead_table.observation_dst_nt[train_rows]) .-
                  Float64.(lead_table.static_stack_pred_dst_nt[train_rows])
        fold_observed = Float64.(lead_table.observation_dst_nt[eval_rows])
        fold_base = Float64.(lead_table.static_stack_pred_dst_nt[eval_rows])
        append!(observed, fold_observed)
        append!(base, fold_base)
        append!(regimes, String.(lead_table.static_stack_regime[eval_rows]))
        for depth in STATIC_STACK_TEMPORAL_DEPTHS
            model = fit(
                _static_stack_temporal_config(depth); x_train, y_train, verbosity=0,
                feature_names=String.(features),
            )
            for rounds in STATIC_STACK_TEMPORAL_ROUNDS
                raw = Float64.(model(x_eval; ntree_limit=rounds + 1))
                cap = 5.0 + 5.0lead
                for rho in STATIC_STACK_TEMPORAL_SHRINKAGES
                    append!(predictions[(depth, rounds, rho)],
                            fold_base .+ rho .* clamp.(raw, -cap, cap))
                end
            end
        end
    end
    best = nothing
    for (key, candidate) in predictions
        gains = _static_stack_temporal_metrics(observed, base, candidate, regimes)
        safe = all(get(gains, regime, 0.0) >= 0.0
                   for regime in ("quiet", "active_deepening", "recovery"))
        record = (; key, safe, gains,
                  rmse=_static_stack_temporal_rmse(observed, candidate))
        if safe && (best === nothing || record.rmse < best.rmse)
            best = record
        end
    end
    best === nothing && error("lead $lead has no regime-safe temporal setting")
    overall_gain = best.gains["overall"]
    quiet_gain = best.gains["quiet"]
    active_gain = best.gains["active_deepening"]
    recovery_gain = best.gains["recovery"]
    println(
        "lead=$lead config=$(best.key) gain=$overall_gain " *
        "quiet=$quiet_gain active=$active_gain recovery=$recovery_gain",
    )
    return best
end

function main_static_stack_temporal_probe()
    table, features = _static_stack_temporal_table()
    println("rows=$(nrow(table)) features=$(length(features)) max_target=$(maximum(table.target_time_utc))")
    for lead in (1, 2, 3, 4, 6, 7)
        _static_stack_temporal_probe(table, features, lead)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_static_stack_temporal_probe()
end
