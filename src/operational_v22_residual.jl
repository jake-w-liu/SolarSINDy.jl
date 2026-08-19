# Lead-specific, bounded causal residual layer for the Operational V2.2 stack.

"Predeclared issue-time features eligible for the V2.2 secondary candidate."
const OPERATIONAL_V22_RESIDUAL_FEATURES = (
    :latest_dst_nt,
    :dst_delta_1h_nt,
    :dst_delta_3h_nt,
    :dst_delta_6h_nt,
    :Bz_nt,
    :Bz_delta_1h_nt,
    :VBsouth_mvm,
    :VBsouth_delta_1h_mvm,
    :VBsouth_mean_3h_mvm,
    :VBsouth_mean_6h_mvm,
    :sqrt_Pdyn_npa,
    :main_phase_pressure_nt,
    :main_phase_pressure_6h_nt,
    :recovery_pressure_nt,
    :main_phase_recovery_pressure,
    :served_minus_frozen_v2_1_nt,
    :primary_minus_served_v2_1_nt,
    :primary_minus_frozen_v2_1_nt,
    :primary_minus_persistence_nt,
    :primary_minus_burton_full_nt,
    :primary_minus_obrien_nt,
    :baseline_spread_nt,
)

const OPERATIONAL_V22_RESIDUAL_RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)
const OPERATIONAL_V22_RESIDUAL_TOP_K_GRID = (2, 4, 6)
const OPERATIONAL_V22_RESIDUAL_SCHEMA_VERSION = "operational_v2_2_residual_v1"

_operational_v22_residual_cap(model_step_hours::Integer) =
    operational_v22_correction_cap_nt(model_step_hours)

"Immutable fitted residual cell for one forecast lead."
struct OperationalV22ResidualCell
    model_step_hours::Int
    feature_names::Tuple{Vararg{Symbol}}
    ranked_feature_names::Tuple{Vararg{Symbol}}
    feature_mean::Tuple{Vararg{Float64}}
    feature_scale::Tuple{Vararg{Float64}}
    coefficients::Tuple{Vararg{Float64}}
    ridge::Float64
    top_k::Int
    correction_cap_nt::Float64
    fit_rows::Int
    validation_rows::Int
    validation_base_rmse_nt::Float64
    validation_rmse_nt::Float64
    validation_active_rows::Int
    validation_active_base_rmse_nt::Float64
    validation_active_rmse_nt::Float64
    validation_recovery_rows::Int
    validation_recovery_base_rmse_nt::Float64
    validation_recovery_rmse_nt::Float64

    function OperationalV22ResidualCell(
            model_step_hours::Int,
            feature_names::Tuple{Vararg{Symbol}},
            ranked_feature_names::Tuple{Vararg{Symbol}},
            feature_mean::Tuple{Vararg{Float64}},
            feature_scale::Tuple{Vararg{Float64}},
            coefficients::Tuple{Vararg{Float64}},
            ridge::Float64,
            top_k::Int,
            correction_cap_nt::Float64,
            fit_rows::Int,
            validation_rows::Int,
            validation_base_rmse_nt::Float64,
            validation_rmse_nt::Float64,
            validation_active_rows::Int,
            validation_active_base_rmse_nt::Float64,
            validation_active_rmse_nt::Float64,
            validation_recovery_rows::Int,
            validation_recovery_base_rmse_nt::Float64,
            validation_recovery_rmse_nt::Float64)
        model_step_hours > 0 || throw(ArgumentError(
            "residual model_step_hours must be positive",
        ))
        top_k > 0 && top_k == length(feature_names) || throw(ArgumentError(
            "residual top_k must equal the selected feature count",
        ))
        length(unique(feature_names)) == top_k || throw(ArgumentError(
            "residual selected feature names must be unique",
        ))
        length(ranked_feature_names) >= top_k &&
            length(unique(ranked_feature_names)) == length(ranked_feature_names) ||
            throw(ArgumentError("residual ranked feature names must be unique and complete"))
        Set(feature_names) == Set(ranked_feature_names[1:top_k]) || throw(ArgumentError(
            "residual selected support must equal the first top_k ranked features",
        ))
        length(feature_mean) == top_k || throw(DimensionMismatch(
            "residual feature_mean length must equal top_k",
        ))
        length(feature_scale) == top_k || throw(DimensionMismatch(
            "residual feature_scale length must equal top_k",
        ))
        length(coefficients) == top_k + 1 || throw(DimensionMismatch(
            "residual coefficients require one intercept plus top_k slopes",
        ))
        all(isfinite, feature_mean) || throw(ArgumentError(
            "residual feature means must be finite",
        ))
        all(x -> isfinite(x) && x > 0.0, feature_scale) || throw(ArgumentError(
            "residual feature scales must be finite and positive",
        ))
        all(isfinite, coefficients) || throw(ArgumentError(
            "residual coefficients must be finite",
        ))
        isfinite(ridge) && ridge > 0.0 || throw(ArgumentError(
            "residual ridge must be finite and positive",
        ))
        correction_cap_nt == _operational_v22_residual_cap(model_step_hours) ||
            throw(ArgumentError("residual correction cap must equal 5 + 5h nT"))
        fit_rows >= 2 || throw(ArgumentError("residual cell requires at least two fit rows"))
        validation_rows >= 1 || throw(ArgumentError(
            "residual cell requires at least one validation row",
        ))
        all(x -> isfinite(x) && x >= 0.0,
            (validation_base_rmse_nt, validation_rmse_nt)) || throw(ArgumentError(
                "residual overall validation RMSE values must be finite and nonnegative",
            ))
        validation_rmse_nt < validation_base_rmse_nt || throw(ArgumentError(
            "residual validation RMSE must strictly improve the base",
        ))
        _operational_v22_residual_validate_group_metrics(
            validation_active_rows, validation_rows,
            validation_active_base_rmse_nt, validation_active_rmse_nt,
            "active/deepening",
        )
        _operational_v22_residual_validate_group_metrics(
            validation_recovery_rows, validation_rows,
            validation_recovery_base_rmse_nt, validation_recovery_rmse_nt,
            "recovery",
        )
        return new(
            model_step_hours, feature_names, ranked_feature_names,
            feature_mean, feature_scale, coefficients, ridge, top_k,
            correction_cap_nt, fit_rows, validation_rows,
            validation_base_rmse_nt, validation_rmse_nt,
            validation_active_rows, validation_active_base_rmse_nt,
            validation_active_rmse_nt, validation_recovery_rows,
            validation_recovery_base_rmse_nt, validation_recovery_rmse_nt,
        )
    end
end

function _operational_v22_residual_validate_group_metrics(
        rows::Int, validation_rows::Int, base_rmse::Float64,
        candidate_rmse::Float64, label::AbstractString)
    0 <= rows <= validation_rows || throw(ArgumentError(
        "residual $label row count is outside the validation range",
    ))
    if rows == 0
        isnan(base_rmse) && isnan(candidate_rmse) || throw(ArgumentError(
            "residual absent $label metrics must be NaN",
        ))
    else
        all(x -> isfinite(x) && x >= 0.0, (base_rmse, candidate_rmse)) ||
            throw(ArgumentError(
                "residual $label RMSE values must be finite and nonnegative",
            ))
        candidate_rmse <= base_rmse || throw(ArgumentError(
            "residual candidate worsens $label validation RMSE",
        ))
    end
    return nothing
end

function OperationalV22ResidualCell(
        model_step_hours::Integer,
        feature_names::AbstractVector{Symbol},
        ranked_feature_names::AbstractVector{Symbol},
        feature_mean::AbstractVector{<:Real},
        feature_scale::AbstractVector{<:Real},
        coefficients::AbstractVector{<:Real};
        ridge::Real,
        top_k::Integer=length(feature_names),
        correction_cap_nt::Real=_operational_v22_residual_cap(model_step_hours),
        fit_rows::Integer,
        validation_rows::Integer,
        validation_base_rmse_nt::Real,
        validation_rmse_nt::Real,
        validation_active_rows::Integer=0,
        validation_active_base_rmse_nt::Real=NaN,
        validation_active_rmse_nt::Real=NaN,
        validation_recovery_rows::Integer=0,
        validation_recovery_base_rmse_nt::Real=NaN,
        validation_recovery_rmse_nt::Real=NaN)
    return OperationalV22ResidualCell(
        Int(model_step_hours), tuple(feature_names...), tuple(ranked_feature_names...),
        Tuple(Float64.(feature_mean)), Tuple(Float64.(feature_scale)),
        Tuple(Float64.(coefficients)), Float64(ridge), Int(top_k),
        Float64(correction_cap_nt), Int(fit_rows), Int(validation_rows),
        Float64(validation_base_rmse_nt), Float64(validation_rmse_nt),
        Int(validation_active_rows), Float64(validation_active_base_rmse_nt),
        Float64(validation_active_rmse_nt), Int(validation_recovery_rows),
        Float64(validation_recovery_base_rmse_nt),
        Float64(validation_recovery_rmse_nt),
    )
end

"Immutable collection of accepted lead-specific V2.2 residual cells."
struct OperationalV22ResidualCore
    label::String
    candidate_feature_names::Tuple{Vararg{Symbol}}
    ridge_grid::Tuple{Vararg{Float64}}
    top_k_grid::Tuple{Vararg{Int}}
    supported_model_steps::Tuple{Vararg{Int}}
    cells::Tuple{Vararg{OperationalV22ResidualCell}}

    function OperationalV22ResidualCore(
            label::String,
            candidate_feature_names::Tuple{Vararg{Symbol}},
            ridge_grid::Tuple{Vararg{Float64}},
            top_k_grid::Tuple{Vararg{Int}},
            supported_model_steps::Tuple{Vararg{Int}},
            cells::Tuple{Vararg{OperationalV22ResidualCell}})
        isempty(strip(label)) && throw(ArgumentError("residual label must not be empty"))
        _operational_v22_residual_validate_feature_names(candidate_feature_names)
        isempty(ridge_grid) && throw(ArgumentError("residual ridge grid must not be empty"))
        collect(ridge_grid) == sort!(unique(collect(ridge_grid))) || throw(ArgumentError(
            "residual ridge grid must be sorted and unique",
        ))
        all(x -> isfinite(x) && x > 0.0, ridge_grid) || throw(ArgumentError(
            "residual ridge values must be finite and positive",
        ))
        isempty(top_k_grid) && throw(ArgumentError("residual top-k grid must not be empty"))
        collect(top_k_grid) == sort!(unique(collect(top_k_grid))) || throw(ArgumentError(
            "residual top-k grid must be sorted and unique",
        ))
        all(k -> 0 < k <= length(candidate_feature_names), top_k_grid) ||
            throw(ArgumentError("residual top-k values exceed the candidate feature count"))
        isempty(supported_model_steps) && throw(ArgumentError(
            "residual core must support at least one model step",
        ))
        collect(supported_model_steps) == sort!(unique(collect(supported_model_steps))) ||
            throw(ArgumentError("residual supported model steps must be sorted and unique"))
        all(>(0), supported_model_steps) || throw(ArgumentError(
            "residual supported model steps must be positive",
        ))
        length(cells) == length(supported_model_steps) || throw(ArgumentError(
            "residual core requires exactly one cell per supported model step",
        ))
        Tuple(cell.model_step_hours for cell in cells) == supported_model_steps ||
            throw(ArgumentError("residual cells must follow supported model-step order"))
        candidate_set = Set(candidate_feature_names)
        for cell in cells
            cell.ridge in ridge_grid || throw(ArgumentError(
                "residual cell ridge is absent from the frozen grid",
            ))
            cell.top_k in top_k_grid || throw(ArgumentError(
                "residual cell top_k is absent from the frozen grid",
            ))
            length(cell.ranked_feature_names) == length(candidate_feature_names) &&
                Set(cell.ranked_feature_names) == candidate_set || throw(ArgumentError(
                    "residual cell ranking must be a permutation of all candidate features",
                ))
            selected = Set(cell.feature_names)
            Tuple(name for name in candidate_feature_names if name in selected) ==
                cell.feature_names || throw(ArgumentError(
                    "residual selected features must retain fixed candidate order",
                ))
        end
        return new(
            label, candidate_feature_names, ridge_grid, top_k_grid,
            supported_model_steps, cells,
        )
    end
end

function _operational_v22_residual_validate_feature_names(feature_names)
    isempty(feature_names) && throw(ArgumentError(
        "residual candidate feature list must not be empty",
    ))
    length(unique(feature_names)) == length(feature_names) || throw(ArgumentError(
        "residual candidate feature names must be unique",
    ))
    allowed = Set(OPERATIONAL_V22_RESIDUAL_FEATURES)
    invalid = [name for name in feature_names if !(name in allowed)]
    isempty(invalid) || throw(ArgumentError(
        "residual feature(s) are not in the predeclared causal set: " *
        join(String.(invalid), ","),
    ))
    return nothing
end

function OperationalV22ResidualCore(
        cells::AbstractVector{OperationalV22ResidualCell};
        label::AbstractString="operational_v2_2_secondary",
        candidate_feature_names=OPERATIONAL_V22_RESIDUAL_FEATURES,
        ridge_grid=OPERATIONAL_V22_RESIDUAL_RIDGE_GRID,
        top_k_grid=OPERATIONAL_V22_RESIDUAL_TOP_K_GRID)
    ordered_cells = sort!(collect(cells); by=cell -> cell.model_step_hours)
    return OperationalV22ResidualCore(
        String(label), tuple(Symbol.(collect(candidate_feature_names))...),
        Tuple(Float64.(collect(ridge_grid))), Tuple(Int.(collect(top_k_grid))),
        Tuple(cell.model_step_hours for cell in ordered_cells), tuple(ordered_cells...),
    )
end

function _operational_v22_residual_ridge(
        design::Matrix{Float64}, target::Vector{Float64}, ridge::Float64)
    size(design, 1) == length(target) || throw(DimensionMismatch(
        "residual ridge design and target row counts differ",
    ))
    size(design, 1) >= 1 || throw(ArgumentError("residual ridge fit is empty"))
    all(isfinite, design) && all(isfinite, target) || throw(ArgumentError(
        "residual ridge inputs must be finite",
    ))
    isfinite(ridge) && ridge > 0.0 || throw(ArgumentError(
        "residual ridge must be finite and positive",
    ))
    gram = design' * design
    for j in 2:size(gram, 1)
        gram[j, j] += ridge
    end
    cross = design' * target
    all(isfinite, gram) && all(isfinite, cross) || throw(ArgumentError(
        "residual ridge normal equations became non-finite",
    ))
    coefficients = cholesky(Symmetric(gram); check=true) \ cross
    all(isfinite, coefficients) || throw(ErrorException(
        "residual ridge coefficients became non-finite",
    ))
    return collect(coefficients)
end

function _operational_v22_residual_matrix(df::DataFrame,
                                          rows::Vector{Int},
                                          feature_names,
                                          means,
                                          scales)
    design = ones(length(rows), length(feature_names) + 1)
    for (j, name) in enumerate(feature_names)
        for (i, row) in enumerate(rows)
            design[i, j + 1] =
                (_operational_v22_finite_cell(df, row, name) - means[j]) / scales[j]
        end
    end
    all(isfinite, design) || throw(ArgumentError(
        "residual standardized design became non-finite",
    ))
    return design
end

function _operational_v22_residual_fit_scaling(df::DataFrame,
                                               rows::Vector{Int},
                                               feature_names)
    length(rows) >= 2 || throw(ArgumentError(
        "each residual lead requires at least two fit rows",
    ))
    means = Vector{Float64}(undef, length(feature_names))
    scales = similar(means)
    for (j, name) in enumerate(feature_names)
        values = [_operational_v22_finite_cell(df, row, name) for row in rows]
        means[j] = mean(values)
        centered = values .- means[j]
        scales[j] = sqrt(sum(abs2, centered) / (length(values) - 1))
        iszero(scales[j]) && (scales[j] = 1.0)
    end
    all(isfinite, means) && all(x -> isfinite(x) && x > 0.0, scales) ||
        throw(ArgumentError("residual fit scaling became non-finite"))
    return means, scales
end

function _operational_v22_residual_response(df::DataFrame,
                                            rows::Vector{Int},
                                            base_column::Symbol,
                                            observation_column::Symbol)
    return [
        _operational_v22_finite_cell(df, row, observation_column) -
        _operational_v22_finite_cell(df, row, base_column)
        for row in rows
    ]
end

function _operational_v22_residual_rmse(observed::Vector{Float64},
                                        predicted::Vector{Float64})
    length(observed) == length(predicted) || throw(DimensionMismatch(
        "residual RMSE vectors have different lengths",
    ))
    isempty(observed) && return NaN
    value = sqrt(sum(abs2, observed .- predicted) / length(observed))
    isfinite(value) || throw(ErrorException("residual validation RMSE became non-finite"))
    return value
end

function _operational_v22_residual_metrics(
        observed::Vector{Float64}, base::Vector{Float64},
        predicted::Vector{Float64}, regimes::Vector{Symbol})
    length(observed) == length(base) == length(predicted) == length(regimes) ||
        throw(DimensionMismatch("residual validation vectors have different lengths"))
    active = findall(==(:active_deepening), regimes)
    recovery = findall(==(:recovery), regimes)
    return (
        base_rmse=_operational_v22_residual_rmse(observed, base),
        rmse=_operational_v22_residual_rmse(observed, predicted),
        active_rows=length(active),
        active_base_rmse=_operational_v22_residual_rmse(observed[active], base[active]),
        active_rmse=_operational_v22_residual_rmse(observed[active], predicted[active]),
        recovery_rows=length(recovery),
        recovery_base_rmse=
            _operational_v22_residual_rmse(observed[recovery], base[recovery]),
        recovery_rmse=
            _operational_v22_residual_rmse(observed[recovery], predicted[recovery]),
    )
end

function _operational_v22_residual_candidate_is_better(candidate, incumbent)
    incumbent === nothing && return true
    tied = isapprox(
        candidate.metrics.rmse, incumbent.metrics.rmse;
        rtol=1e-12, atol=1e-12,
    )
    !tied && return candidate.metrics.rmse < incumbent.metrics.rmse
    candidate.top_k < incumbent.top_k && return true
    candidate.top_k > incumbent.top_k && return false
    return candidate.ridge > incumbent.ridge
end

"""
    fit_operational_v22_residual(fit_df, validation_df; feature_names, kwargs...)

Fit the predeclared lead-specific sparse ridge residual on `fit_df`, select
ridge/support size only on `validation_df`, and fail closed unless the selected
candidate improves overall base RMSE without worsening populated active or
recovery issue-time regimes.
"""
function fit_operational_v22_residual(
        fit_df::DataFrame,
        validation_df::DataFrame;
        feature_names=OPERATIONAL_V22_RESIDUAL_FEATURES,
        ridge_grid=OPERATIONAL_V22_RESIDUAL_RIDGE_GRID,
        top_k_grid=OPERATIONAL_V22_RESIDUAL_TOP_K_GRID,
        base_column::Symbol=:v2_2_pred_dst_nt,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours,
        latest_dst_column::Symbol=:latest_dst_nt,
        dst_rate_column::Symbol=:dst_delta_1h_nt,
        causal_coupling_column::Symbol=:coupling_active_mvm,
        label::AbstractString="operational_v2_2_secondary")
    features = tuple(Symbol.(collect(feature_names))...)
    ridges = Tuple(Float64.(collect(ridge_grid)))
    top_ks = Tuple(Int.(collect(top_k_grid)))
    _operational_v22_residual_validate_feature_names(features)
    isempty(ridges) && throw(ArgumentError("residual ridge grid must not be empty"))
    collect(ridges) == sort!(unique(collect(ridges))) &&
        all(x -> isfinite(x) && x > 0.0, ridges) || throw(ArgumentError(
            "residual ridge grid must be sorted, unique, finite, and positive",
        ))
    isempty(top_ks) && throw(ArgumentError("residual top-k grid must not be empty"))
    collect(top_ks) == sort!(unique(collect(top_ks))) &&
        all(k -> 0 < k <= length(features), top_ks) || throw(ArgumentError(
            "residual top-k grid must be sorted, unique, positive, and in range",
        ))
    required = (
        features..., base_column, observation_column, model_step_column,
        latest_dst_column, dst_rate_column, causal_coupling_column,
    )
    _operational_v22_require_columns(fit_df, required)
    _operational_v22_require_columns(validation_df, required)
    nrow(fit_df) >= 2 || throw(ArgumentError("residual fit table is too small"))
    nrow(validation_df) >= 1 || throw(ArgumentError("residual validation table is empty"))

    fit_steps = [
        _operational_v22_model_step(fit_df, row, model_step_column)
        for row in 1:nrow(fit_df)
    ]
    validation_steps = [
        _operational_v22_model_step(validation_df, row, model_step_column)
        for row in 1:nrow(validation_df)
    ]
    leads = sort!(unique(fit_steps))
    leads == sort!(unique(validation_steps)) || throw(ArgumentError(
        "residual fit and validation tables must contain identical lead sets",
    ))

    cells = OperationalV22ResidualCell[]
    for lead in leads
        fit_rows = findall(==(lead), fit_steps)
        validation_rows = findall(==(lead), validation_steps)
        means, scales = _operational_v22_residual_fit_scaling(
            fit_df, fit_rows, features,
        )
        full_fit_design = _operational_v22_residual_matrix(
            fit_df, fit_rows, features, means, scales,
        )
        fit_target = _operational_v22_residual_response(
            fit_df, fit_rows, base_column, observation_column,
        )
        validation_observed = [
            _operational_v22_finite_cell(validation_df, row, observation_column)
            for row in validation_rows
        ]
        validation_base = [
            _operational_v22_finite_cell(validation_df, row, base_column)
            for row in validation_rows
        ]
        validation_regimes = [
            operational_v22_regime(
                _operational_v22_finite_cell(validation_df, row, latest_dst_column),
                _operational_v22_finite_cell(validation_df, row, dst_rate_column),
                _operational_v22_finite_cell(
                    validation_df, row, causal_coupling_column,
                ),
            ) for row in validation_rows
        ]
        best = nothing
        for ridge in ridges
            full_coefficients = _operational_v22_residual_ridge(
                full_fit_design, fit_target, ridge,
            )
            ranking = sortperm(1:length(features); by=j -> (-abs(full_coefficients[j + 1]), j))
            ranked_names = tuple((features[j] for j in ranking)...)
            for top_k in top_ks
                support_indices = sort(ranking[1:top_k])
                support_names = tuple((features[j] for j in support_indices)...)
                support_means = means[support_indices]
                support_scales = scales[support_indices]
                fit_design = _operational_v22_residual_matrix(
                    fit_df, fit_rows, support_names, support_means, support_scales,
                )
                coefficients = _operational_v22_residual_ridge(
                    fit_design, fit_target, ridge,
                )
                validation_design = _operational_v22_residual_matrix(
                    validation_df, validation_rows, support_names,
                    support_means, support_scales,
                )
                raw_correction = validation_design * coefficients
                cap = _operational_v22_residual_cap(lead)
                predicted = validation_base .+ clamp.(raw_correction, -cap, cap)
                metrics = _operational_v22_residual_metrics(
                    validation_observed, validation_base, predicted,
                    validation_regimes,
                )
                candidate = (
                    ridge=ridge, top_k=top_k, support_names=support_names,
                    ranked_names=ranked_names, support_means=support_means,
                    support_scales=support_scales, coefficients=coefficients,
                    cap=cap, metrics=metrics,
                )
                _operational_v22_residual_candidate_is_better(candidate, best) &&
                    (best = candidate)
            end
        end
        best === nothing && throw(ErrorException(
            "residual candidate search produced no candidate for lead $lead",
        ))
        metrics = best.metrics
        overall_pass = metrics.rmse < metrics.base_rmse
        active_pass = metrics.active_rows == 0 ||
            metrics.active_rmse <= metrics.active_base_rmse
        recovery_pass = metrics.recovery_rows == 0 ||
            metrics.recovery_rmse <= metrics.recovery_base_rmse
        overall_pass && active_pass && recovery_pass || throw(ErrorException(
            "residual candidate rejected at lead $lead: " *
            "overall_pass=$overall_pass, active_pass=$active_pass, " *
            "recovery_pass=$recovery_pass",
        ))
        push!(cells, OperationalV22ResidualCell(
            lead, collect(best.support_names), collect(best.ranked_names),
            best.support_means, best.support_scales, best.coefficients;
            ridge=best.ridge, top_k=best.top_k,
            correction_cap_nt=best.cap, fit_rows=length(fit_rows),
            validation_rows=length(validation_rows),
            validation_base_rmse_nt=metrics.base_rmse,
            validation_rmse_nt=metrics.rmse,
            validation_active_rows=metrics.active_rows,
            validation_active_base_rmse_nt=metrics.active_base_rmse,
            validation_active_rmse_nt=metrics.active_rmse,
            validation_recovery_rows=metrics.recovery_rows,
            validation_recovery_base_rmse_nt=metrics.recovery_base_rmse,
            validation_recovery_rmse_nt=metrics.recovery_rmse,
        ))
    end
    return OperationalV22ResidualCore(
        cells; label=label, candidate_feature_names=features,
        ridge_grid=ridges, top_k_grid=top_ks,
    )
end

function _operational_v22_residual_cell(core::OperationalV22ResidualCore,
                                        model_step_hours::Int)
    model_step_hours in core.supported_model_steps || throw(ArgumentError(
        "unsupported residual model_step_hours=$model_step_hours; supported steps are " *
        join(core.supported_model_steps, ","),
    ))
    for cell in core.cells
        cell.model_step_hours == model_step_hours && return cell
    end
    throw(ErrorException("residual core is missing supported lead $model_step_hours"))
end

function _operational_v22_residual_feature(features, name::Symbol)
    raw = if features isa NamedTuple
        haskey(features, name) || throw(ArgumentError("missing residual feature: $name"))
        getfield(features, name)
    elseif features isa AbstractDict
        haskey(features, name) || throw(ArgumentError("missing residual feature: $name"))
        features[name]
    else
        throw(ArgumentError("residual features must be a NamedTuple or dictionary"))
    end
    ismissing(raw) && throw(ArgumentError("residual feature $name is missing"))
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "residual feature $name must be real",
    ))
    value = Float64(raw)
    isfinite(value) || throw(ArgumentError("residual feature $name must be finite"))
    return value
end

"Pure bounded residual prediction with lead-specific support and contribution logging."
function operational_v22_residual_predict(
        core::OperationalV22ResidualCore,
        model_step_hours::Integer,
        base_prediction_dst_nt::Real,
        features)
    lead = Int(model_step_hours)
    lead > 0 || throw(ArgumentError("residual model_step_hours must be positive"))
    base = Float64(base_prediction_dst_nt)
    isfinite(base) || throw(ArgumentError("residual base prediction must be finite"))
    cell = _operational_v22_residual_cell(core, lead)
    standardized = ntuple(
        j -> (_operational_v22_residual_feature(features, cell.feature_names[j]) -
              cell.feature_mean[j]) / cell.feature_scale[j],
        cell.top_k,
    )
    contributions = ntuple(
        j -> cell.coefficients[j + 1] * standardized[j], cell.top_k,
    )
    all(isfinite, standardized) && all(isfinite, contributions) ||
        throw(ArgumentError("residual standardized contribution became non-finite"))
    raw_correction = cell.coefficients[1] + sum(contributions)
    isfinite(raw_correction) || throw(ArgumentError(
        "residual raw correction became non-finite",
    ))
    correction = clamp(raw_correction, -cell.correction_cap_nt, cell.correction_cap_nt)
    prediction = base + correction
    isfinite(prediction) || throw(ArgumentError(
        "residual corrected prediction became non-finite",
    ))
    names = cell.feature_names
    return (
        pred_dst=prediction,
        base_pred_dst=base,
        raw_correction_nt=raw_correction,
        correction_nt=correction,
        correction_was_capped=correction != raw_correction,
        correction_cap_nt=cell.correction_cap_nt,
        model_step_hours=lead,
        ridge=cell.ridge,
        top_k=cell.top_k,
        feature_names=names,
        standardized_features=NamedTuple{names}(standardized),
        feature_coefficients=NamedTuple{names}(cell.coefficients[2:end]),
        feature_contributions=NamedTuple{names}(contributions),
        intercept_nt=cell.coefficients[1],
        label=core.label,
    )
end

"Score a table without consulting observation or any post-issue column for prediction."
function score_operational_v22_residual(
        df::DataFrame,
        core::OperationalV22ResidualCore;
        base_column::Symbol=:v2_2_pred_dst_nt,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours)
    _operational_v22_require_columns(
        df, (core.candidate_feature_names..., base_column, model_step_column),
    )
    out = copy(df)
    n = nrow(out)
    prediction = Vector{Float64}(undef, n)
    raw_correction = similar(prediction)
    correction = similar(prediction)
    capped = Vector{Bool}(undef, n)
    cap = similar(prediction)
    ridge = similar(prediction)
    top_k = Vector{Int}(undef, n)
    support = Vector{String}(undef, n)
    coefficients = Vector{String}(undef, n)
    contributions = Vector{String}(undef, n)
    for row in 1:n
        lead = _operational_v22_model_step(out, row, model_step_column)
        cell = _operational_v22_residual_cell(core, lead)
        features = Dict(
            name => _operational_v22_finite_cell(out, row, name)
            for name in cell.feature_names
        )
        result = operational_v22_residual_predict(
            core, lead, _operational_v22_finite_cell(out, row, base_column), features,
        )
        prediction[row] = result.pred_dst
        raw_correction[row] = result.raw_correction_nt
        correction[row] = result.correction_nt
        capped[row] = result.correction_was_capped
        cap[row] = result.correction_cap_nt
        ridge[row] = result.ridge
        top_k[row] = result.top_k
        support[row] = join(String.(result.feature_names), ";")
        coefficients[row] = join(string.((result.intercept_nt,
                                          Tuple(result.feature_coefficients)...)), ";")
        contributions[row] = join(string.(Tuple(result.feature_contributions)), ";")
    end
    out[!, :v2_2_secondary_pred_dst_nt] = prediction
    out[!, :v2_2_secondary_raw_correction_nt] = raw_correction
    out[!, :v2_2_secondary_correction_nt] = correction
    out[!, :v2_2_secondary_correction_was_capped] = capped
    out[!, :v2_2_secondary_correction_cap_nt] = cap
    out[!, :v2_2_secondary_ridge] = ridge
    out[!, :v2_2_secondary_top_k] = top_k
    out[!, :v2_2_secondary_feature_names] = support
    out[!, :v2_2_secondary_coefficients] = coefficients
    out[!, :v2_2_secondary_feature_contributions_nt] = contributions
    out[!, :v2_2_secondary_label] = fill(core.label, n)
    if String(observation_column) in names(out)
        residual = Vector{Union{Missing,Float64}}(undef, n)
        for row in 1:n
            raw = out[row, observation_column]
            if ismissing(raw)
                residual[row] = missing
            else
                raw isa Real && !(raw isa Bool) || throw(ArgumentError(
                    "residual observation must be real or missing",
                ))
                observed = Float64(raw)
                isfinite(observed) || throw(ArgumentError(
                    "residual observation must be finite when present",
                ))
                residual[row] = observed - prediction[row]
            end
        end
        out[!, :v2_2_secondary_residual_dst_nt] = residual
    end
    return out
end

const _OPERATIONAL_V22_RESIDUAL_CSV_COLUMNS = (
    :schema_version, :label, :candidate_feature_names, :ridge_grid, :top_k_grid,
    :supported_model_steps, :model_step_hours, :feature_names,
    :ranked_feature_names, :feature_mean, :feature_scale, :coefficients,
    :ridge, :top_k, :correction_cap_nt, :fit_rows, :validation_rows,
    :validation_base_rmse_nt, :validation_rmse_nt, :validation_active_rows,
    :validation_active_base_rmse_nt, :validation_active_rmse_nt,
    :validation_recovery_rows, :validation_recovery_base_rmse_nt,
    :validation_recovery_rmse_nt,
)

"Atomically write the strictly versioned V2.2 residual core CSV."
function write_operational_v22_residual(path::AbstractString,
                                        core::OperationalV22ResidualCore)
    target = String(path)
    mkpath(dirname(abspath(target)))
    rows = NamedTuple[]
    for cell in core.cells
        push!(rows, (
            schema_version=OPERATIONAL_V22_RESIDUAL_SCHEMA_VERSION,
            label=core.label,
            candidate_feature_names=join(String.(core.candidate_feature_names), ";"),
            ridge_grid=join(string.(core.ridge_grid), ";"),
            top_k_grid=join(string.(core.top_k_grid), ";"),
            supported_model_steps=join(string.(core.supported_model_steps), ";"),
            model_step_hours=cell.model_step_hours,
            feature_names=join(String.(cell.feature_names), ";"),
            ranked_feature_names=join(String.(cell.ranked_feature_names), ";"),
            feature_mean=join(string.(cell.feature_mean), ";"),
            feature_scale=join(string.(cell.feature_scale), ";"),
            coefficients=join(string.(cell.coefficients), ";"),
            ridge=cell.ridge,
            top_k=cell.top_k,
            correction_cap_nt=cell.correction_cap_nt,
            fit_rows=cell.fit_rows,
            validation_rows=cell.validation_rows,
            validation_base_rmse_nt=cell.validation_base_rmse_nt,
            validation_rmse_nt=cell.validation_rmse_nt,
            validation_active_rows=cell.validation_active_rows,
            validation_active_base_rmse_nt=cell.validation_active_base_rmse_nt,
            validation_active_rmse_nt=cell.validation_active_rmse_nt,
            validation_recovery_rows=cell.validation_recovery_rows,
            validation_recovery_base_rmse_nt=cell.validation_recovery_base_rmse_nt,
            validation_recovery_rmse_nt=cell.validation_recovery_rmse_nt,
        ))
    end
    _write_selection_csv(target, rows)
    return target
end

function _operational_v22_residual_split_symbols(value, field::AbstractString)
    text = string(value)
    isempty(text) && throw(ArgumentError("residual $field must not be empty"))
    names = Symbol.(split(text, ";"))
    any(name -> isempty(String(name)), names) && throw(ArgumentError(
        "residual $field contains an empty feature name",
    ))
    return names
end

function _operational_v22_residual_split_floats(value, field::AbstractString)
    values = try
        value isa Real ? Float64[Float64(value)] : parse.(Float64, split(string(value), ";"))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("residual $field is not a valid float list"))
    end
    isempty(values) && throw(ArgumentError("residual $field must not be empty"))
    all(isfinite, values) || throw(ArgumentError("residual $field must be finite"))
    return values
end

function _operational_v22_residual_split_ints(value, field::AbstractString)
    values = try
        value isa Real ? Int[_operational_v22_csv_int(value, field)] :
            parse.(Int, split(string(value), ";"))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("residual $field is not a valid integer list"))
    end
    isempty(values) && throw(ArgumentError("residual $field must not be empty"))
    return values
end

function _operational_v22_residual_metric(value, field::AbstractString;
                                          allow_nan::Bool=false)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "residual $field must be numeric",
    ))
    converted = Float64(value)
    (isfinite(converted) || (allow_nan && isnan(converted))) || throw(ArgumentError(
        "residual $field is invalid",
    ))
    return converted
end

"Read and strictly validate a V2.2 residual core CSV."
function read_operational_v22_residual(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "residual core must be a regular non-symlink file: $source",
    ))
    # Text-valued columns are read as text; see `read_operational_v22_stack` for why.
    df = CSV.read(source, DataFrame;
                  types = Dict(:schema_version => String, :label => String,
                               :candidate_feature_names => String, :ridge_grid => String,
                               :top_k_grid => String, :supported_model_steps => String,
                               :feature_names => String, :ranked_feature_names => String,
                               :feature_mean => String, :feature_scale => String,
                               :coefficients => String),
                  validate = false)
    names(df) == collect(String.(_OPERATIONAL_V22_RESIDUAL_CSV_COLUMNS)) ||
        throw(ArgumentError(
            "residual core CSV schema does not exactly match " *
            OPERATIONAL_V22_RESIDUAL_SCHEMA_VERSION,
        ))
    nrow(df) >= 1 || throw(ArgumentError("residual core CSV is empty"))
    for row in 1:nrow(df), column in _OPERATIONAL_V22_RESIDUAL_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "residual core CSV contains missing at row $row column $column",
        ))
    end
    schema = string(_operational_v22_consistent_column(df, :schema_version))
    schema == OPERATIONAL_V22_RESIDUAL_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported residual core schema version: $schema",
    ))
    label = string(_operational_v22_consistent_column(df, :label))
    candidate_features = _operational_v22_residual_split_symbols(
        _operational_v22_consistent_column(df, :candidate_feature_names),
        "candidate_feature_names",
    )
    ridge_grid = _operational_v22_residual_split_floats(
        _operational_v22_consistent_column(df, :ridge_grid), "ridge_grid",
    )
    top_k_grid = _operational_v22_residual_split_ints(
        _operational_v22_consistent_column(df, :top_k_grid), "top_k_grid",
    )
    supported_steps = _operational_v22_residual_split_ints(
        _operational_v22_consistent_column(df, :supported_model_steps),
        "supported_model_steps",
    )
    cells = OperationalV22ResidualCell[]
    for row in 1:nrow(df)
        push!(cells, OperationalV22ResidualCell(
            _operational_v22_csv_int(df[row, :model_step_hours], "model_step_hours"),
            _operational_v22_residual_split_symbols(
                df[row, :feature_names], "feature_names",
            ),
            _operational_v22_residual_split_symbols(
                df[row, :ranked_feature_names], "ranked_feature_names",
            ),
            _operational_v22_residual_split_floats(
                df[row, :feature_mean], "feature_mean",
            ),
            _operational_v22_residual_split_floats(
                df[row, :feature_scale], "feature_scale",
            ),
            _operational_v22_residual_split_floats(
                df[row, :coefficients], "coefficients",
            );
            ridge=_operational_v22_residual_metric(df[row, :ridge], "ridge"),
            top_k=_operational_v22_csv_int(df[row, :top_k], "top_k"),
            correction_cap_nt=_operational_v22_residual_metric(
                df[row, :correction_cap_nt], "correction_cap_nt",
            ),
            fit_rows=_operational_v22_csv_int(df[row, :fit_rows], "fit_rows"),
            validation_rows=_operational_v22_csv_int(
                df[row, :validation_rows], "validation_rows",
            ),
            validation_base_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_base_rmse_nt], "validation_base_rmse_nt",
            ),
            validation_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_rmse_nt], "validation_rmse_nt",
            ),
            validation_active_rows=_operational_v22_csv_int(
                df[row, :validation_active_rows], "validation_active_rows",
            ),
            validation_active_base_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_active_base_rmse_nt],
                "validation_active_base_rmse_nt"; allow_nan=true,
            ),
            validation_active_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_active_rmse_nt],
                "validation_active_rmse_nt"; allow_nan=true,
            ),
            validation_recovery_rows=_operational_v22_csv_int(
                df[row, :validation_recovery_rows], "validation_recovery_rows",
            ),
            validation_recovery_base_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_recovery_base_rmse_nt],
                "validation_recovery_base_rmse_nt"; allow_nan=true,
            ),
            validation_recovery_rmse_nt=_operational_v22_residual_metric(
                df[row, :validation_recovery_rmse_nt],
                "validation_recovery_rmse_nt"; allow_nan=true,
            ),
        ))
    end
    # The file's own `supported_model_steps` metadata was parsed and then discarded, so a core whose
    # metadata claimed one lead set while its cells carried another loaded without complaint and was
    # then queried by the metadata's steps. The two must describe the same core.
    cell_steps = sort!(unique(cell.model_step_hours for cell in cells))
    sort!(unique!(supported_steps)) == cell_steps || throw(ArgumentError(
        "residual core CSV declares supported model steps $(supported_steps) but carries cells for " *
        "$(cell_steps)",
    ))
    return OperationalV22ResidualCore(
        cells; label=label, candidate_feature_names=candidate_features,
        ridge_grid=ridge_grid, top_k_grid=top_k_grid,
    )
end
