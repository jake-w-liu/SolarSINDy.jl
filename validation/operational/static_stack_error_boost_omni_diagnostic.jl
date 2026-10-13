isdefined(@__MODULE__, :run_static_stack_error_omni_diagnostic) ||
    include(joinpath(@__DIR__, "static_stack_error_omni_diagnostic.jl"))

using EvoTrees
using EvoTrees: fit

const STATIC_STACK_ERROR_BOOST_DEPTHS = (3, 4)
const STATIC_STACK_ERROR_BOOST_ROUNDS = (64, 128, 256)
const STATIC_STACK_ERROR_BOOST_SHRINKAGES = (0.25, 0.50, 0.75, 1.00)
const STATIC_STACK_ERROR_BOOST_SEED = 22_022_026
const STATIC_STACK_ERROR_BOOST_COMPARATORS = (
    "m3_sparse_error_state",
    STATIC_STACK_ERROR_OMNI_COMPARATORS...,
)
const STATIC_STACK_ERROR_BOOST_CANDIDATE = "m3_lag_feature_tree"
const STATIC_STACK_ERROR_BOOST_SCHEMA = "static_stack_error_lag_feature_tree_omni_diagnostic_v1"

function _err_boost_config(depth::Int;
        nrounds::Int=maximum(STATIC_STACK_ERROR_BOOST_ROUNDS))
    depth in STATIC_STACK_ERROR_BOOST_DEPTHS || throw(ArgumentError(
        "M3 boost depth is outside the frozen grid",
    ))
    nrounds in STATIC_STACK_ERROR_BOOST_ROUNDS || throw(ArgumentError(
        "M3 boost rounds are outside the frozen grid",
    ))
    return EvoTreeRegressor(
        loss=:mse,
        metric=:rmse,
        nrounds=nrounds,
        eta=0.03,
        L2=2.0,
        lambda=10.0,
        gamma=0.0,
        max_depth=depth,
        min_weight=96.0,
        rowsample=1.0,
        colsample=1.0,
        nbins=64,
        tree_type=:binary,
        seed=STATIC_STACK_ERROR_BOOST_SEED,
        device=:cpu,
    )
end

function _err_boost_read_examples(path::AbstractString,
        expected_sha256::AbstractString; expected_folds)
    _drv_omni_sha256(path) == expected_sha256 || throw(ArgumentError(
        "M3 boost example-table checksum changed",
    ))
    columns = (
        :fold, :issue_time_utc, :target_time_utc, :model_step_hours,
        :base_center_sha256, :base_prediction_dst_nt, :observation_dst_nt,
        STATIC_STACK_ERROR_EXOGENOUS_FEATURES...,
    )
    table = CSV.read(path, DataFrame)
    SolarSINDy.canonicalize_static_stack_frozen_table!(table)
    Set(columns) ⊆ Set(Symbol.(names(table))) || throw(ArgumentError(
        "M3 boost example-table schema changed",
    ))
    select!(table, collect(columns))
    table.fold = String.(table.fold)
    table.issue_time_utc = _err_omni_parse_datetime.(table.issue_time_utc)
    table.target_time_utc = _err_omni_parse_datetime.(table.target_time_utc)
    table.model_step_hours = Int.(table.model_step_hours)
    Set(table.fold) == Set(String.(expected_folds)) || throw(ArgumentError(
        "M3 boost example-table fold set changed",
    ))
    all(table.model_step_hours .∈ Ref(STATIC_STACK_DRIVER_OMNI_LEADS_H)) ||
        throw(ArgumentError("M3 boost examples contain an unsupported lead"))
    all(table.target_time_utc .== table.issue_time_utc .+
        Hour.(table.model_step_hours)) || throw(ArgumentError(
            "M3 boost example targets do not match their leads",
        ))
    maximum(table.target_time_utc) < DateTime(2023, 1, 1) ||
        throw(ArgumentError("M3 boost examples cross the pre-2023 boundary"))
    all(isfinite, Matrix{Float64}(
        table[:, collect(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)],
    )) || throw(ArgumentError("M3 boost feature matrix is non-finite"))
    all(isfinite, Float64.(table.base_prediction_dst_nt)) &&
        all(isfinite, Float64.(table.observation_dst_nt)) ||
        throw(ArgumentError("M3 boost targets or centers are non-finite"))
    all(table.m2_core_center_dst_nt .== table.base_prediction_dst_nt) ||
        throw(ArgumentError("M3 boost center feature differs from its center"))
    keys = collect(zip(table.issue_time_utc, table.model_step_hours))
    length(unique(keys)) == length(keys) || throw(ArgumentError(
        "M3 boost examples contain duplicate issue/lead keys",
    ))
    counts = combine(groupby(table, :issue_time_utc), nrow => :rows)
    all(==(length(STATIC_STACK_DRIVER_OMNI_LEADS_H)), counts.rows) ||
        throw(ArgumentError("M3 boost examples do not retain whole anchors"))
    return table
end

function _err_boost_regime_masks(table::DataFrame)
    latest = Float64.(table.latest_dst_nt)
    delta = Float64.(table.dst_delta_1h_nt)
    observed = Float64.(table.observation_dst_nt)
    return (
        overall=trues(nrow(table)),
        target_storm=observed .<= -50.0,
        active_or_deepening=(latest .<= -30.0) .& (delta .< 0.0),
        recovery=(latest .<= -30.0) .& (delta .>= 0.0),
    )
end

function _err_boost_losses(table::DataFrame, prediction::Vector{Float64})
    length(prediction) == nrow(table) || throw(DimensionMismatch(
        "M3 boost prediction length does not match its table",
    ))
    observed = Float64.(table.observation_dst_nt)
    base = Float64.(table.base_prediction_dst_nt)
    masks = _err_boost_regime_masks(table)
    values = NamedTuple[]
    for (regime, selected) in pairs(masks)
        rows = count(selected)
        base_rmse = rows == 0 ? NaN :
            sqrt(mean((base[selected] .- observed[selected]).^2))
        candidate_rmse = rows == 0 ? NaN :
            sqrt(mean((prediction[selected] .- observed[selected]).^2))
        push!(values, (
            regime=String(regime),
            rows=rows,
            base_rmse_nt=base_rmse,
            candidate_rmse_nt=candidate_rmse,
            gain_nt=rows == 0 ? NaN : base_rmse - candidate_rmse,
        ))
    end
    return DataFrame(values)
end

function _err_boost_regime_safe(losses::DataFrame)
    safe = true
    for regime in ("target_storm", "active_or_deepening", "recovery")
        row = only(eachrow(losses[losses.regime .== regime, :]))
        row.rows >= 20 && (safe &= row.gain_nt >= 0.0)
    end
    return safe
end

function _err_boost_config_less(left, right)
    left.candidate_rmse_nt != right.candidate_rmse_nt &&
        return left.candidate_rmse_nt < right.candidate_rmse_nt
    left.depth != right.depth && return left.depth < right.depth
    left.rounds != right.rounds && return left.rounds < right.rounds
    return left.shrinkage < right.shrinkage
end

function _err_boost_choose_index(audit::DataFrame)
    eligible = findall(audit.regime_safe .& (audit.gain_nt .> 0.0))
    isempty(eligible) && return nothing
    best = first(eligible)
    for index in Iterators.drop(eligible, 1)
        _err_boost_config_less(audit[index, :], audit[best, :]) &&
            (best = index)
    end
    return best
end

function static_stack_error_boost_select(table::DataFrame, lead::Int)
    lead in STATIC_STACK_DRIVER_OMNI_LEADS_H || throw(ArgumentError(
        "M3 boost selection lead is unsupported",
    ))
    lead_table = table[table.model_step_hours .== lead, :]
    configs = [
        (depth=depth, rounds=rounds, shrinkage=shrinkage)
        for depth in STATIC_STACK_ERROR_BOOST_DEPTHS
        for rounds in STATIC_STACK_ERROR_BOOST_ROUNDS
        for shrinkage in STATIC_STACK_ERROR_BOOST_SHRINKAGES
    ]
    predictions = Dict(config => Float64[] for config in configs)
    evaluation = DataFrame()
    for fold in STATIC_STACK_DRIVER_OMNI_FOLDS[2:end]
        year_value = year(fold.validation_start)
        training_folds = Set(
            "calendar_$year" for year in 2013:(year_value - 1)
        )
        train_rows = findall(in(training_folds), lead_table.fold)
        eval_rows = findall(==(fold.name), lead_table.fold)
        isempty(train_rows) && throw(ArgumentError(
            "M3 boost selection has no training rows for $(fold.name)",
        ))
        isempty(eval_rows) && throw(ArgumentError(
            "M3 boost selection has no evaluation rows for $(fold.name)",
        ))
        train = lead_table[train_rows, :]
        current = lead_table[eval_rows, :]
        maximum(train.target_time_utc) <=
            fold.validation_start - Hour(STATIC_STACK_DRIVER_OMNI_EMBARGO_HOURS) ||
            throw(ArgumentError(
                "M3 boost fold $(fold.name) violates the target embargo",
            ))
        x_train = Matrix{Float64}(
            train[:, collect(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)],
        )
        x_eval = Matrix{Float64}(
            current[:, collect(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)],
        )
        target = Float64.(train.observation_dst_nt) .-
                 Float64.(train.base_prediction_dst_nt)
        evaluation = isempty(evaluation) ? copy(current) :
            vcat(evaluation, current; cols=:setequal)
        for depth in STATIC_STACK_ERROR_BOOST_DEPTHS
            model = fit(
                _err_boost_config(depth);
                x_train=x_train,
                y_train=target,
                verbosity=0,
                feature_names=collect(String.(
                    STATIC_STACK_ERROR_EXOGENOUS_FEATURES,
                )),
            )
            for rounds in STATIC_STACK_ERROR_BOOST_ROUNDS
                raw = Float64.(model(x_eval; ntree_limit=rounds + 1))
                cap = SolarSINDy._static_stack_error_cap(lead)
                for shrinkage in STATIC_STACK_ERROR_BOOST_SHRINKAGES
                    config = (depth=depth, rounds=rounds,
                              shrinkage=shrinkage)
                    append!(predictions[config],
                            Float64.(current.base_prediction_dst_nt) .+
                            shrinkage .* clamp.(raw, -cap, cap))
                end
            end
        end
    end
    rows = NamedTuple[]
    for config in configs
        losses = _err_boost_losses(evaluation, predictions[config])
        overall = only(eachrow(losses[losses.regime .== "overall", :]))
        safe = _err_boost_regime_safe(losses)
        record = (
            model_step_hours=lead,
            depth=config.depth,
            rounds=config.rounds,
            shrinkage=config.shrinkage,
            rows=overall.rows,
            base_rmse_nt=overall.base_rmse_nt,
            candidate_rmse_nt=overall.candidate_rmse_nt,
            gain_nt=overall.gain_nt,
            target_storm_gain_nt=only(losses.gain_nt[
                losses.regime .== "target_storm",
            ]),
            active_or_deepening_gain_nt=only(losses.gain_nt[
                losses.regime .== "active_or_deepening",
            ]),
            recovery_gain_nt=only(losses.gain_nt[
                losses.regime .== "recovery",
            ]),
            regime_safe=safe,
            selected=false,
        )
        push!(rows, record)
    end
    audit = DataFrame(rows)
    selected_row = _err_boost_choose_index(audit)
    if selected_row === nothing
        return (
            audit=audit,
            selected=(model_step_hours=lead, depth=0, rounds=0,
                      shrinkage=0.0, regime_safe=true,
                      reason="no_regime_safe_positive_oof_gain"),
        )
    end
    best = audit[selected_row, :]
    audit.selected[selected_row] = true
    return (
        audit=audit,
        selected=(model_step_hours=lead, depth=best.depth,
                  rounds=best.rounds, shrinkage=best.shrinkage,
                  regime_safe=true, reason="selected_regime_safe_oof_minimum"),
    )
end

function _err_boost_final_predictions(oof::DataFrame,
        final_ready::DataFrame, selected)
    lead = selected.model_step_hours
    selected.depth == 0 && return Dict(
        (DateTime(row.issue_time_utc), lead) =>
            Float64(row.base_prediction_dst_nt)
        for row in eachrow(final_ready[final_ready.model_step_hours .== lead, :])
    )
    training = oof[
        (oof.model_step_hours .== lead) .&
        (oof.issue_time_utc .<= STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP -
                                  Hour(maximum(STATIC_STACK_DRIVER_OMNI_LEADS_H))) .&
        (oof.target_time_utc .<= STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP), :]
    evaluation = final_ready[final_ready.model_step_hours .== lead, :]
    isempty(training) && throw(ArgumentError(
        "M3 boost final training rows are empty",
    ))
    isempty(evaluation) && throw(ArgumentError(
        "M3 boost final ready rows are empty",
    ))
    x_train = Matrix{Float64}(
        training[:, collect(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)],
    )
    target = Float64.(training.observation_dst_nt) .-
             Float64.(training.base_prediction_dst_nt)
    model = fit(
        _err_boost_config(selected.depth; nrounds=selected.rounds);
        x_train=x_train,
        y_train=target,
        verbosity=0,
        feature_names=collect(String.(
            STATIC_STACK_ERROR_EXOGENOUS_FEATURES,
        )),
    )
    x_eval = Matrix{Float64}(
        evaluation[:, collect(STATIC_STACK_ERROR_EXOGENOUS_FEATURES)],
    )
    raw = Float64.(model(x_eval))
    cap = SolarSINDy._static_stack_error_cap(lead)
    prediction = Float64.(evaluation.base_prediction_dst_nt) .+
                 selected.shrinkage .* clamp.(raw, -cap, cap)
    return Dict(
        (DateTime(evaluation.issue_time_utc[index]), lead) => prediction[index]
        for index in 1:nrow(evaluation)
    )
end

function static_stack_error_boost_score(base_errors::DataFrame,
        ready_predictions::Dict{Tuple{DateTime,Int},Float64})
    base = base_errors[base_errors.method .== "sindy_sparse_driver", :]
    rows = NamedTuple[]
    for row in eachrow(base)
        key = (DateTime(row.issue_time_utc), Int(row.model_step_hours))
        ready = haskey(ready_predictions, key)
        prediction = ready ? ready_predictions[key] : Float64(row.predicted_dst_nt)
        push!(rows, merge(NamedTuple(row), (
            method=STATIC_STACK_ERROR_BOOST_CANDIDATE,
            comparison_role="strong_accuracy_comparator_candidate",
            candidate_ready=ready,
            fallback_reason=ready ? "none" :
                "missing_contiguous_24h_history",
            predicted_dst_nt=prediction,
            squared_error_nt2=(prediction - Float64(row.observed_dst_nt))^2,
            absolute_error_nt=abs(prediction - Float64(row.observed_dst_nt)),
            raw_candidate_correction_nt=prediction -
                Float64(row.predicted_dst_nt),
            applied_candidate_correction_nt=prediction -
                Float64(row.predicted_dst_nt),
        )))
    end
    comparators = base_errors[in(STATIC_STACK_ERROR_BOOST_COMPARATORS).(
        base_errors.method), :]
    errors = vcat(DataFrame(rows), comparators; cols=:setequal)
    expected = nrow(base)
    all(nrow(group) == expected for group in groupby(errors, :method)) ||
        throw(ArgumentError("M3 boost comparison rows are not common"))
    return errors
end

function run_static_stack_error_boost_omni_diagnostic(;
        m3_output_root=joinpath(@__DIR__, "..", "output", "operational",
            "static_stack_error_omni_diagnostic"),
        output_root=joinpath(@__DIR__, "..", "output", "operational",
            "static_stack_error_boost_omni_diagnostic"))
    mkpath(output_root)
    manifest = CSV.read(joinpath(
        m3_output_root, "static_stack_error_omni_source_manifest.csv",
    ), DataFrame)
    nrow(manifest) == 1 || throw(ArgumentError(
        "M3 boost source manifest must contain one row",
    ))
    oof_path = joinpath(m3_output_root, "static_stack_error_omni_oof_examples.csv")
    final_path = joinpath(
        m3_output_root, "static_stack_error_omni_final_ready_examples.csv",
    )
    errors_path = joinpath(
        m3_output_root, "static_stack_error_omni_paired_errors.csv",
    )
    oof = _err_boost_read_examples(
        oof_path, String(manifest.oof_examples_sha256[1]);
        expected_folds=Tuple(fold.name for fold in STATIC_STACK_DRIVER_OMNI_FOLDS),
    )
    final_ready = _err_boost_read_examples(
        final_path, String(manifest.final_ready_examples_sha256[1]);
        expected_folds=("exposed_point_validation",),
    )
    _drv_omni_sha256(errors_path) ==
        String(manifest.paired_errors_sha256[1]) || throw(ArgumentError(
            "M3 boost paired-error checksum changed",
        ))
    base_errors = CSV.read(errors_path, DataFrame;
        types=Dict(:issue_time_utc => DateTime, :target_time_utc => DateTime),
    )
    Set(base_errors.method) == Set(("m3_sparse_error_state",
                                   STATIC_STACK_ERROR_OMNI_COMPARATORS...)) ||
        throw(ArgumentError("M3 boost inherited comparator set changed"))

    selection_tables = DataFrame[]
    selected_rows = NamedTuple[]
    ready_predictions = Dict{Tuple{DateTime,Int},Float64}()
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        result = static_stack_error_boost_select(oof, lead)
        push!(selection_tables, result.audit)
        push!(selected_rows, result.selected)
        merge!(ready_predictions, _err_boost_final_predictions(
            oof, final_ready, result.selected,
        ))
    end
    selection = reduce((left, right) -> vcat(left, right; cols=:setequal),
                       selection_tables)
    selected = DataFrame(selected_rows)
    errors = static_stack_error_boost_score(base_errors, ready_predictions)
    metrics = static_stack_error_omni_metrics(errors)
    bootstrap = static_stack_error_omni_bootstrap(
        errors;
        candidate=STATIC_STACK_ERROR_BOOST_CANDIDATE,
        comparators=STATIC_STACK_ERROR_BOOST_COMPARATORS,
    )
    driver_decision = CSV.read(joinpath(
        m3_output_root, "..", "static_stack_driver_omni_diagnostic",
        "static_stack_driver_omni_decision.csv",
    ), DataFrame)
    by_comparison, decision = static_stack_error_omni_decision(
        metrics, bootstrap, driver_decision;
        candidate=STATIC_STACK_ERROR_BOOST_CANDIDATE,
        comparators=STATIC_STACK_ERROR_BOOST_COMPARATORS,
        scheduled_issues=Int(manifest.m2_scheduled_validation_issues[1]),
        m2_eligible_issues=Int(manifest.m2_eligible_validation_issues[1]),
    )
    decision[!, :lag_feature_tree_comparator_included] .= true
    decision[!, :diagnostic_outcome] .=
        decision.available_comparator_diagnostic_supported[1] ?
        "BOOST_AVAILABLE_COMPARATOR_SIGNAL_ONLY" :
        "BOOST_AVAILABLE_COMPARATOR_SIGNAL_NOT_SUPPORTED"
    decision[!, :promotion_eligible] .= false

    CSV.write(joinpath(output_root, "static_stack_error_boost_selection.csv"), selection)
    CSV.write(joinpath(output_root, "static_stack_error_boost_selected.csv"), selected)
    CSV.write(joinpath(output_root, "static_stack_error_boost_paired_errors.csv"), errors)
    CSV.write(joinpath(output_root, "static_stack_error_boost_metrics.csv"), metrics)
    CSV.write(joinpath(output_root, "static_stack_error_boost_bootstrap.csv"), bootstrap)
    CSV.write(joinpath(output_root,
                       "static_stack_error_boost_decision_by_comparison.csv"),
              by_comparison)
    CSV.write(joinpath(output_root, "static_stack_error_boost_decision.csv"), decision)
    CSV.write(joinpath(output_root, "static_stack_error_boost_source_manifest.csv"),
              DataFrame(
                  schema=[STATIC_STACK_ERROR_BOOST_SCHEMA],
                  inherited_oof_sha256=[manifest.oof_examples_sha256[1]],
                  inherited_final_ready_sha256=[
                      manifest.final_ready_examples_sha256[1],
                  ],
                  inherited_paired_errors_sha256=[
                      manifest.paired_errors_sha256[1],
                  ],
                  feature_schema_sha256=[_err_omni_hash_values(
                      STATIC_STACK_ERROR_EXOGENOUS_FEATURES,
                  )],
                  post_2022_rows_read=[0],
                  exposed_point_validation_window=[true],
                  receipt_causal=[false],
                  event_block_inference_available=[false],
                  portable_artifact_frozen=[false],
                  promotion_eligible=[false],
              ))
    return (
        decision=decision,
        by_comparison=by_comparison,
        metrics=metrics,
        bootstrap=bootstrap,
        selection=selection,
        selected=selected,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_stack_error_boost_omni_diagnostic()
    println("static stack-M3 lag-feature tree OMNI diagnostic: ",
            result.decision.diagnostic_outcome[1])
end
