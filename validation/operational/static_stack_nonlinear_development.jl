#!/usr/bin/env julia

# Select the final bounded nonlinear static stack family from pre-2023 cross-fitted rows.
# The exposed 2023--2025 benchmark is not located or read here.

using SolarSINDy
using CSV
using DataFrames
using Dates
using EvoTrees
using EvoTrees: fit
using SHA
using Statistics

const STATIC_STACK_NL_CROSSFIT_PATH = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "static_stack_crossfit",
    "static_stack_primary_crossfit_oof.csv",
))
const STATIC_STACK_NL_DEVELOPMENT_PATH = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "static_stack_residual_development",
    "static_stack_residual_replay.csv",
))
const STATIC_STACK_NL_OUTPUT_DIR = normpath(joinpath(
    @__DIR__, "..", "output", "operational", "static_stack_nonlinear_development",
))
const STATIC_STACK_NL_GRID_AUDIT_PATH = joinpath(STATIC_STACK_NL_OUTPUT_DIR, "static_stack_nonlinear_grid_audit.csv")
const STATIC_STACK_NL_FOLD_AUDIT_PATH = joinpath(STATIC_STACK_NL_OUTPUT_DIR, "static_stack_nonlinear_fold_audit.csv")
const STATIC_STACK_NL_VALIDATION_PATH = joinpath(STATIC_STACK_NL_OUTPUT_DIR, "static_stack_nonlinear_validation.csv")
const STATIC_STACK_NL_AUDIT_PATH = joinpath(STATIC_STACK_NL_OUTPUT_DIR, "static_stack_nonlinear_audit.csv")

const STATIC_STACK_NL_CROSSFIT_SHA256 =
    "5e78d1e9e23824cab48249e977646c2fe73ea5798065d977a0ed9b1143f77d40"
const STATIC_STACK_NL_DEVELOPMENT_SHA256 =
    "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f"
const STATIC_STACK_NL_FEATURE_SCHEMA_SHA256 =
    "dbd15d62ad43ae930895768f01570a3d2ed1248a6d01202634d5be246fcf6d09"
const STATIC_STACK_NL_FEATURES = STATIC_STACK_RESIDUAL_FEATURES
const STATIC_STACK_NL_SUPPORTED_STEPS = Tuple(OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS)
const STATIC_STACK_NL_PRIMARY_LEADS = (1, 2, 3, 6)
const STATIC_STACK_NL_SELECTION_FOLDS = (
    "calendar_2014", "calendar_2015", "calendar_2016", "calendar_2017",
)
const STATIC_STACK_NL_DEPTHS = (2, 3)
const STATIC_STACK_NL_ROUNDS = (32, 64, 128)
const STATIC_STACK_NL_SHRINKAGES = (0.25, 0.50, 0.75, 1.00)
const STATIC_STACK_NL_SEED = 22_022_026

_static_stack_nl_file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

# The schema digest certifies the feature vocabulary recorded inside the sha-pinned frozen
# replays, whose columns predate the canonical rename. The live feature list is hashed under its
# recorded spellings and the recorded preimage label so the pin still verifies the declared schema.
function _static_stack_nl_schema_sha256(features=STATIC_STACK_NL_FEATURES)
    recorded = String.(get.(
        Ref(SolarSINDy._STATIC_STACK_FROZEN_FEATURE_SPELLINGS),
        collect(Symbol.(features)), collect(Symbol.(features)),
    ))
    payload = "operational_v2_2_issue_feature_schema_v1\n" *
              join(recorded, '\n') * "\n"
    return bytes2hex(sha256(codeunits(payload)))
end

function _static_stack_nl_require_file(path::AbstractString, expected_sha::AbstractString,
                              label::AbstractString)
    isfile(path) && !islink(path) || error("$label must be a regular non-symlink file")
    actual = _static_stack_nl_file_sha256(path)
    actual == expected_sha || error("$label SHA-256 changed: $actual")
    return actual
end

function _static_stack_nl_columns(; crossfit::Bool)
    columns = Symbol[
        :issue_time_utc, :target_time_utc, :model_step_hours,
        :observation_dst_nt, :static_stack_pred_dst_nt, :latest_dst_nt,
        :static_stack_regime, :served_inertial_sindy_dst_nt, :frozen_inertial_sindy_dst_nt,
        :raw_sindy_dst_nt, :persistence_dst_nt, :burton_dst_nt,
        :burton_full_dst_nt, :obrien_dst_nt,
    ]
    crossfit ? push!(columns, :static_stack_crossfit_fold) : push!(columns, :split_label)
    append!(columns, STATIC_STACK_NL_FEATURES)
    return unique(columns)
end

function _static_stack_nl_read(path::AbstractString; crossfit::Bool)
    columns = _static_stack_nl_columns(; crossfit)
    table = CSV.read(path, DataFrame; types=Dict(
        :issue_time_utc => DateTime, :target_time_utc => DateTime,
    ))
    SolarSINDy.canonicalize_static_stack_frozen_table!(table)
    missing_columns = setdiff(columns, Symbol.(names(table)))
    isempty(missing_columns) || error(
        "static stack nonlinear input omits $(join(String.(missing_columns), ','))",
    )
    all(table.target_time_utc .== table.issue_time_utc .+
        Hour.(Int.(table.model_step_hours))) || error(
        "static stack nonlinear input violates lag-zero target semantics",
    )
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) || error(
        "static stack nonlinear development input contains a post-2022 target",
    )
    keys = Tuple.(eachrow(select(
        table, :issue_time_utc, :target_time_utc, :model_step_hours,
    )))
    length(unique(keys)) == nrow(table) || error(
        "static stack nonlinear input contains duplicate keys",
    )
    for feature in STATIC_STACK_NL_FEATURES
        all(isfinite, Float64.(table[!, feature])) || error(
            "static stack nonlinear feature $feature contains a non-finite value",
        )
    end
    return table
end

function _static_stack_nl_matrix(df::DataFrame, rows)
    matrix = Matrix{Float64}(df[rows, collect(STATIC_STACK_NL_FEATURES)])
    all(isfinite, matrix) || error("static stack nonlinear matrix is non-finite")
    return matrix
end

_static_stack_nl_rmse(observed, predicted) = sqrt(mean(abs2, observed .- predicted))

function _static_stack_nl_severity(latest_dst::Real)
    value = Float64(latest_dst)
    value <= -100.0 && return :extreme
    value <= -50.0 && return :intense
    value <= -30.0 && return :moderate
    return :quiet_or_weak
end

function _static_stack_nl_group_metrics(observed, base, candidate, regimes, latest_dst)
    length(observed) == length(base) == length(candidate) == length(regimes) ==
        length(latest_dst) || error("static stack nonlinear metric vectors differ in length")
    rows = NamedTuple[]
    push!(rows, (
        group_kind="overall", group_label="all", n_rows=length(observed),
        base_rmse_nt=_static_stack_nl_rmse(observed, base),
        candidate_rmse_nt=_static_stack_nl_rmse(observed, candidate),
    ))
    regime_symbols = Symbol.(regimes)
    for regime in (:quiet, :active_deepening, :recovery)
        idx = findall(==(regime), regime_symbols)
        isempty(idx) && continue
        push!(rows, (
            group_kind="issue_regime", group_label=String(regime), n_rows=length(idx),
            base_rmse_nt=_static_stack_nl_rmse(observed[idx], base[idx]),
            candidate_rmse_nt=_static_stack_nl_rmse(observed[idx], candidate[idx]),
        ))
    end
    severities = _static_stack_nl_severity.(latest_dst)
    for severity in (:quiet_or_weak, :moderate, :intense, :extreme)
        idx = findall(==(severity), severities)
        isempty(idx) && continue
        push!(rows, (
            group_kind="current_dst", group_label=String(severity), n_rows=length(idx),
            base_rmse_nt=_static_stack_nl_rmse(observed[idx], base[idx]),
            candidate_rmse_nt=_static_stack_nl_rmse(observed[idx], candidate[idx]),
        ))
    end
    out = DataFrame(rows)
    out[!, :gain_nt] = out.base_rmse_nt .- out.candidate_rmse_nt
    return out
end

function _static_stack_nl_safe(metrics::DataFrame)
    overall = only(eachrow(metrics[metrics.group_kind .== "overall", :]))
    overall.candidate_rmse_nt < overall.base_rmse_nt || return false
    regimes = metrics[metrics.group_kind .== "issue_regime", :]
    all(regimes.candidate_rmse_nt .<= regimes.base_rmse_nt) || return false
    severities = metrics[
        (metrics.group_kind .== "current_dst") .& (metrics.n_rows .>= 40), :,
    ]
    return all(severities.candidate_rmse_nt .<= severities.base_rmse_nt .+ 0.50)
end

function _static_stack_nl_config(depth::Int, nrounds::Int)
    return EvoTreeRegressor(
        loss=:mse, metric=:rmse, nrounds=nrounds, eta=0.05,
        L2=1.0, lambda=10.0, gamma=0.0, max_depth=depth,
        min_weight=128.0, rowsample=1.0, colsample=1.0, nbins=64,
        tree_type=:binary, seed=STATIC_STACK_NL_SEED, device=:cpu,
    )
end

function _static_stack_nl_fit_maximum(x_train, y_train, depth::Int)
    model = fit(
        _static_stack_nl_config(depth, maximum(STATIC_STACK_NL_ROUNDS));
        x_train, y_train, verbosity=0,
        feature_names=collect(String.(STATIC_STACK_NL_FEATURES)),
    )
    length(model.trees) == maximum(STATIC_STACK_NL_ROUNDS) + 1 || error(
        "EvoTrees did not retain the expected bias tree and boosting rounds",
    )
    return model
end

function _static_stack_nl_correction(model, x_eval, lead::Int, rounds::Int, rho::Float64)
    raw = Float64.(model(x_eval; ntree_limit=rounds + 1))
    cap = 5.0 + 5.0lead
    correction = rho .* clamp.(raw, -cap, cap)
    all(isfinite, correction) || error("static stack nonlinear correction is non-finite")
    return raw, correction
end

function _static_stack_nl_better(candidate, incumbent)
    incumbent === nothing && return true
    delta = candidate.rmse_nt - incumbent.rmse_nt
    abs(delta) > 1e-12 && return delta < 0.0
    candidate.depth != incumbent.depth && return candidate.depth < incumbent.depth
    candidate.rounds != incumbent.rounds && return candidate.rounds < incumbent.rounds
    return candidate.rho < incumbent.rho
end

function _static_stack_nl_select_lead(oof::DataFrame, lead::Int)
    lead_table = oof[oof.model_step_hours .== lead, :]
    predictions = Dict(
        (depth, rounds, rho) => Float64[]
        for depth in STATIC_STACK_NL_DEPTHS for rounds in STATIC_STACK_NL_ROUNDS
        for rho in STATIC_STACK_NL_SHRINKAGES
    )
    observed = Float64[]
    base = Float64[]
    regimes = String[]
    latest_dst = Float64[]
    fold_rows = NamedTuple[]

    for fold in STATIC_STACK_NL_SELECTION_FOLDS
        fold_year = parse(Int, last(split(fold, '_')))
        training_folds = Set("calendar_$year" for year in 2013:(fold_year - 1))
        train_rows = findall(in(training_folds), String.(lead_table.static_stack_crossfit_fold))
        eval_rows = findall(==(fold), String.(lead_table.static_stack_crossfit_fold))
        !isempty(train_rows) && !isempty(eval_rows) || error(
            "static stack nonlinear fold $fold is empty at lead $lead",
        )
        maximum(lead_table.issue_time_utc[train_rows]) <
            minimum(lead_table.issue_time_utc[eval_rows]) || error(
            "static stack nonlinear rolling fold chronology failed",
        )
        x_train = _static_stack_nl_matrix(lead_table, train_rows)
        x_eval = _static_stack_nl_matrix(lead_table, eval_rows)
        y_train = Float64.(lead_table.observation_dst_nt[train_rows]) .-
                  Float64.(lead_table.static_stack_pred_dst_nt[train_rows])
        fold_observed = Float64.(lead_table.observation_dst_nt[eval_rows])
        fold_base = Float64.(lead_table.static_stack_pred_dst_nt[eval_rows])
        append!(observed, fold_observed)
        append!(base, fold_base)
        append!(regimes, String.(lead_table.static_stack_regime[eval_rows]))
        append!(latest_dst, Float64.(lead_table.latest_dst_nt[eval_rows]))

        models = Dict(depth => _static_stack_nl_fit_maximum(x_train, y_train, depth)
                      for depth in STATIC_STACK_NL_DEPTHS)
        for depth in STATIC_STACK_NL_DEPTHS, rounds in STATIC_STACK_NL_ROUNDS,
            rho in STATIC_STACK_NL_SHRINKAGES
            raw, correction = _static_stack_nl_correction(
                models[depth], x_eval, lead, rounds, rho,
            )
            candidate = fold_base .+ correction
            append!(predictions[(depth, rounds, rho)], candidate)
            push!(fold_rows, (
                model_step_hours=lead, fold=fold, depth=depth, rounds=rounds,
                shrinkage=rho, n_rows=length(eval_rows),
                base_rmse_nt=_static_stack_nl_rmse(fold_observed, fold_base),
                candidate_rmse_nt=_static_stack_nl_rmse(fold_observed, candidate),
                raw_correction_mean_nt=mean(raw),
                correction_mean_nt=mean(correction),
                correction_cap_fraction=mean(abs.(raw) .> (5.0 + 5.0lead)),
            ))
        end
    end

    grid_rows = NamedTuple[]
    best = nothing
    best_diagnostic = nothing
    for depth in STATIC_STACK_NL_DEPTHS, rounds in STATIC_STACK_NL_ROUNDS, rho in STATIC_STACK_NL_SHRINKAGES
        candidate = predictions[(depth, rounds, rho)]
        metrics = _static_stack_nl_group_metrics(
            observed, base, candidate, regimes, latest_dst,
        )
        safe = _static_stack_nl_safe(metrics)
        overall = only(eachrow(metrics[metrics.group_kind .== "overall", :]))
        record = (
            model_step_hours=lead, depth=depth, rounds=rounds, shrinkage=rho,
            n_rows=length(observed), base_rmse_nt=overall.base_rmse_nt,
            candidate_rmse_nt=overall.candidate_rmse_nt,
            gain_vs_base_nt=overall.gain_nt, safety_pass=safe,
            quiet_gain_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "quiet"), :gain_nt,
            ]),
            active_deepening_gain_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "active_deepening"), :gain_nt,
            ]),
            recovery_gain_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "recovery"), :gain_nt,
            ]),
        )
        push!(grid_rows, record)
        candidate_key = (
            rmse_nt=overall.candidate_rmse_nt, depth=depth,
            rounds=rounds, rho=rho,
        )
        _static_stack_nl_better(candidate_key, best_diagnostic) &&
            (best_diagnostic = candidate_key)
        safe && _static_stack_nl_better(candidate_key, best) && (best = candidate_key)
    end
    return (
        selected=best === nothing ? best_diagnostic : best,
        selection_safe=best !== nothing,
        grid=DataFrame(grid_rows), folds=DataFrame(fold_rows),
    )
end

function _static_stack_nl_final_validation(oof::DataFrame, development::DataFrame,
                                  selections)
    rows = NamedTuple[]
    models = Dict{Int,Any}()
    for lead in STATIC_STACK_NL_SUPPORTED_STEPS
        selected = selections[lead]
        train_rows = findall(==(lead), Int.(oof.model_step_hours))
        validation_rows = findall(
            (Int.(development.model_step_hours) .== lead) .&
            (String.(development.split_label) .== "validation"),
        )
        x_train = _static_stack_nl_matrix(oof, train_rows)
        y_train = Float64.(oof.observation_dst_nt[train_rows]) .-
                  Float64.(oof.static_stack_pred_dst_nt[train_rows])
        model = fit(
            _static_stack_nl_config(selected.depth, selected.rounds);
            x_train, y_train, verbosity=0,
            feature_names=collect(String.(STATIC_STACK_NL_FEATURES)),
        )
        models[lead] = model
        x_validation = _static_stack_nl_matrix(development, validation_rows)
        raw, correction = _static_stack_nl_correction(
            model, x_validation, lead, selected.rounds, selected.rho,
        )
        observed = Float64.(development.observation_dst_nt[validation_rows])
        base = Float64.(development.static_stack_pred_dst_nt[validation_rows])
        candidate = base .+ correction
        metrics = _static_stack_nl_group_metrics(
            observed, base, candidate,
            String.(development.static_stack_regime[validation_rows]),
            Float64.(development.latest_dst_nt[validation_rows]),
        )
        safety = _static_stack_nl_safe(metrics)
        comparator_names = (
            served_inertial_sindy=:served_inertial_sindy_dst_nt,
            frozen_inertial_sindy=:frozen_inertial_sindy_dst_nt,
            raw_sindy=:raw_sindy_dst_nt,
            persistence=:persistence_dst_nt,
            burton=:burton_dst_nt,
            burton_full=:burton_full_dst_nt,
            obrien=:obrien_dst_nt,
        )
        comparator_rmse = Dict(
            name => _static_stack_nl_rmse(
                observed, Float64.(development[validation_rows, column]),
            ) for (name, column) in pairs(comparator_names)
        )
        best_name = first(sort!(collect(keys(comparator_rmse));
                               by=name -> comparator_rmse[name]))
        best_rmse = comparator_rmse[best_name]
        candidate_rmse = _static_stack_nl_rmse(observed, candidate)
        gain = best_rmse - candidate_rmse
        push!(rows, (
            model_step_hours=lead, n_rows=length(validation_rows),
            depth=selected.depth, rounds=selected.rounds,
            shrinkage=selected.rho, safety_pass=safety,
            base_rmse_nt=_static_stack_nl_rmse(observed, base),
            candidate_rmse_nt=candidate_rmse,
            gain_vs_base_nt=_static_stack_nl_rmse(observed, base) - candidate_rmse,
            best_comparator=String(best_name), best_comparator_rmse_nt=best_rmse,
            gain_vs_best_nt=gain,
            beats_every_local_comparator=gain > 0.0,
            passes_general_effect_gate=lead in STATIC_STACK_NL_PRIMARY_LEADS &&
                gain >= max(0.25, 0.02best_rmse),
            raw_correction_mean_nt=mean(raw),
            correction_mean_nt=mean(correction),
            correction_cap_fraction=mean(abs.(raw) .> (5.0 + 5.0lead)),
            quiet_gain_vs_base_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "quiet"), :gain_nt,
            ]),
            active_deepening_gain_vs_base_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "active_deepening"), :gain_nt,
            ]),
            recovery_gain_vs_base_nt=only(metrics[
                (metrics.group_kind .== "issue_regime") .&
                (metrics.group_label .== "recovery"), :gain_nt,
            ]),
        ))
    end
    return DataFrame(rows), models
end

function _static_stack_nl_atomic_write(path::AbstractString, table::DataFrame)
    mkpath(dirname(path))
    temporary = tempname(dirname(path))
    try
        CSV.write(temporary, table)
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return path
end

function run_static_stack_nonlinear_development()
    _static_stack_nl_schema_sha256() == STATIC_STACK_NL_FEATURE_SCHEMA_SHA256 || error(
        "static stack nonlinear feature schema changed",
    )
    crossfit_sha = _static_stack_nl_require_file(
        STATIC_STACK_NL_CROSSFIT_PATH, STATIC_STACK_NL_CROSSFIT_SHA256, "static stack cross-fit replay",
    )
    development_sha = _static_stack_nl_require_file(
        STATIC_STACK_NL_DEVELOPMENT_PATH, STATIC_STACK_NL_DEVELOPMENT_SHA256,
        "static stack development replay",
    )
    oof = _static_stack_nl_read(STATIC_STACK_NL_CROSSFIT_PATH; crossfit=true)
    development = _static_stack_nl_read(STATIC_STACK_NL_DEVELOPMENT_PATH; crossfit=false)
    selections = Dict{Int,Any}()
    selection_safe = Dict{Int,Bool}()
    grids = DataFrame[]
    folds = DataFrame[]
    for lead in STATIC_STACK_NL_SUPPORTED_STEPS
        result = _static_stack_nl_select_lead(oof, lead)
        selections[lead] = result.selected
        selection_safe[lead] = result.selection_safe
        push!(grids, result.grid)
        push!(folds, result.folds)
    end
    grid = vcat(grids...)
    fold_audit = vcat(folds...)
    all_selection_safe = all(values(selection_safe))
    validation, models = if all_selection_safe
        _static_stack_nl_final_validation(oof, development, selections)
    else
        DataFrame(), Dict{Int,Any}()
    end
    final_pass = all_selection_safe && all(validation.safety_pass) &&
        all(validation.beats_every_local_comparator) &&
        all(validation[validation.model_step_hours .∈ Ref(STATIC_STACK_NL_PRIMARY_LEADS),
                       :passes_general_effect_gate])
    audit = DataFrame(
        crossfit_path=[abspath(STATIC_STACK_NL_CROSSFIT_PATH)],
        crossfit_sha256=[crossfit_sha],
        development_path=[abspath(STATIC_STACK_NL_DEVELOPMENT_PATH)],
        development_sha256=[development_sha],
        feature_schema_sha256=[STATIC_STACK_NL_FEATURE_SCHEMA_SHA256],
        evotrees_version=[string(Base.pkgversion(EvoTrees))],
        crossfit_rows=[nrow(oof)],
        validation_rows=[count(==("validation"), String.(development.split_label))],
        crossfit_selection_safe=[all_selection_safe],
        crossfit_rejected_leads=[join(
            string.(sort!([lead for lead in keys(selection_safe)
                           if !selection_safe[lead]])), ";",
        )],
        exposed_benchmark_rows_read=[0],
        final_validation_pass=[final_pass],
    )
    _static_stack_nl_atomic_write(STATIC_STACK_NL_GRID_AUDIT_PATH, grid)
    _static_stack_nl_atomic_write(STATIC_STACK_NL_FOLD_AUDIT_PATH, fold_audit)
    all_selection_safe && _static_stack_nl_atomic_write(STATIC_STACK_NL_VALIDATION_PATH, validation)
    _static_stack_nl_atomic_write(STATIC_STACK_NL_AUDIT_PATH, audit)
    return (;
        final_pass, selections, selection_safe, grid, fold_audit,
        validation, audit, models,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_stack_nonlinear_development()
    println("static stack bounded nonlinear final validation pass = ", result.final_pass)
end
