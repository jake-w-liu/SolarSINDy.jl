# Bounded, SINDy-dominant stacking layer for Operational V2.2 research.

"Ordered point-forecast components accepted by the V2.2 stack."
const OPERATIONAL_V22_COMPONENTS = (
    :served_v2_1,
    :frozen_v2_1,
    :persistence,
    :burton,
    :burton_full,
    :obrien,
)

"Default replay-table columns corresponding to `OPERATIONAL_V22_COMPONENTS`."
const DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS = (
    served_v2_1=:served_v2_1_dst_nt,
    frozen_v2_1=:frozen_v2_1_dst_nt,
    persistence=:persistence_dst_nt,
    burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,
    obrien=:obrien_dst_nt,
)

const OPERATIONAL_V22_REGIMES = (:quiet, :active_deepening, :recovery)
const OPERATIONAL_V22_POOLED_REGIME = :pooled
const OPERATIONAL_V22_DEFAULT_SINDY_MASS_FLOOR = 0.60
const OPERATIONAL_V22_DEFAULT_MINIMUM_CELL_ROWS = 48
const OPERATIONAL_V22_DEFAULT_DISTURBED_DST_NT = -30.0
const OPERATIONAL_V22_DEFAULT_DEEPENING_RATE_NT_PER_H = -5.0
const OPERATIONAL_V22_DEFAULT_COUPLING_THRESHOLD_MVM = 0.0
const OPERATIONAL_V22_STACK_SCHEMA_VERSION = "operational_v2_2_stack_v1"

const _OPERATIONAL_V22_WEIGHT_TOLERANCE = 1e-10

"One immutable lead/regime weight cell in an [`OperationalV22Stack`](@ref)."
struct OperationalV22Cell
    model_step_hours::Int
    regime::Symbol
    n_rows::Int
    weights::NTuple{6,Float64}
    objective_mse::Float64
    iterations::Int

    function OperationalV22Cell(model_step_hours::Int,
                                regime::Symbol,
                                n_rows::Int,
                                weights::NTuple{6,Float64},
                                objective_mse::Float64,
                                iterations::Int)
        model_step_hours > 0 || throw(ArgumentError(
            "V2.2 model_step_hours must be positive",
        ))
        regime in (OPERATIONAL_V22_POOLED_REGIME, OPERATIONAL_V22_REGIMES...) ||
            throw(ArgumentError("unknown V2.2 regime: $regime"))
        n_rows >= 1 || throw(ArgumentError("V2.2 cell n_rows must be positive"))
        all(isfinite, weights) || throw(ArgumentError("V2.2 weights must be finite"))
        all(w -> w >= 0.0, weights) || throw(ArgumentError(
            "V2.2 weights must be nonnegative",
        ))
        isapprox(sum(weights), 1.0; rtol=0.0, atol=_OPERATIONAL_V22_WEIGHT_TOLERANCE) ||
            throw(ArgumentError("V2.2 weights must sum to one"))
        isfinite(objective_mse) && objective_mse >= 0.0 || throw(ArgumentError(
            "V2.2 objective_mse must be finite and nonnegative",
        ))
        iterations >= 0 || throw(ArgumentError("V2.2 iterations must be nonnegative"))
        return new(model_step_hours, regime, n_rows, weights, objective_mse, iterations)
    end
end

function OperationalV22Cell(model_step_hours::Integer,
                            regime::Symbol,
                            n_rows::Integer,
                            weights::AbstractVector{<:Real};
                            objective_mse::Real=0.0,
                            iterations::Integer=0)
    length(weights) == length(OPERATIONAL_V22_COMPONENTS) || throw(DimensionMismatch(
        "V2.2 requires $(length(OPERATIONAL_V22_COMPONENTS)) component weights",
    ))
    converted = ntuple(i -> Float64(weights[i]), length(OPERATIONAL_V22_COMPONENTS))
    return OperationalV22Cell(
        Int(model_step_hours), regime, Int(n_rows), converted,
        Float64(objective_mse), Int(iterations),
    )
end

"""
    OperationalV22Stack

Immutable collection of constrained lead/regime stacking cells. Every supported
lead has a pooled fallback. A regime-specific cell is used only when it met the
stored minimum-row threshold during fitting. The first two components are the
served and frozen-tail 20/11 SINDy forecasts; their combined mass is at least
`sindy_mass_floor` in every cell.
"""
struct OperationalV22Stack
    label::String
    sindy_mass_floor::Float64
    minimum_cell_rows::Int
    disturbed_dst_threshold_nt::Float64
    deepening_rate_threshold_nt_per_h::Float64
    coupling_threshold_mvm::Float64
    supported_model_steps::Tuple{Vararg{Int}}
    cells::Tuple{Vararg{OperationalV22Cell}}

    function OperationalV22Stack(label::String,
                                 sindy_mass_floor::Float64,
                                 minimum_cell_rows::Int,
                                 disturbed_dst_threshold_nt::Float64,
                                 deepening_rate_threshold_nt_per_h::Float64,
                                 coupling_threshold_mvm::Float64,
                                 supported_model_steps::Tuple{Vararg{Int}},
                                 cells::Tuple{Vararg{OperationalV22Cell}})
        isempty(strip(label)) && throw(ArgumentError("V2.2 stack label must not be empty"))
        isfinite(sindy_mass_floor) && 0.0 < sindy_mass_floor <= 1.0 ||
            throw(ArgumentError("sindy_mass_floor must lie in (0, 1]"))
        minimum_cell_rows >= 1 || throw(ArgumentError(
            "minimum_cell_rows must be positive",
        ))
        isfinite(disturbed_dst_threshold_nt) || throw(ArgumentError(
            "disturbed Dst threshold must be finite",
        ))
        isfinite(deepening_rate_threshold_nt_per_h) &&
            deepening_rate_threshold_nt_per_h < 0.0 || throw(ArgumentError(
                "deepening-rate threshold must be finite and negative",
            ))
        isfinite(coupling_threshold_mvm) && coupling_threshold_mvm >= 0.0 ||
            throw(ArgumentError("coupling threshold must be finite and nonnegative"))
        isempty(supported_model_steps) && throw(ArgumentError(
            "V2.2 stack must support at least one model step",
        ))
        collect(supported_model_steps) == sort!(unique(collect(supported_model_steps))) ||
            throw(ArgumentError("supported_model_steps must be sorted and unique"))
        all(>(0), supported_model_steps) || throw(ArgumentError(
            "supported_model_steps must be positive",
        ))
        isempty(cells) && throw(ArgumentError("V2.2 stack must contain cells"))

        seen = Set{Tuple{Int,Symbol}}()
        pooled = Set{Int}()
        for cell in cells
            cell.model_step_hours in supported_model_steps || throw(ArgumentError(
                "V2.2 cell has unsupported model step $(cell.model_step_hours)",
            ))
            cell.n_rows >= minimum_cell_rows || throw(ArgumentError(
                "V2.2 cell $(cell.model_step_hours)/$(cell.regime) is below minimum_cell_rows",
            ))
            key = (cell.model_step_hours, cell.regime)
            key in seen && throw(ArgumentError("duplicate V2.2 cell: $key"))
            push!(seen, key)
            cell.regime == OPERATIONAL_V22_POOLED_REGIME &&
                push!(pooled, cell.model_step_hours)
            sindy_mass = cell.weights[1] + cell.weights[2]
            sindy_mass + _OPERATIONAL_V22_WEIGHT_TOLERANCE >= sindy_mass_floor ||
                throw(ArgumentError(
                    "V2.2 cell $key violates the SINDy-family mass floor",
                ))
        end
        pooled == Set(supported_model_steps) || throw(ArgumentError(
            "every supported V2.2 model step requires exactly one pooled fallback",
        ))
        return new(
            label, sindy_mass_floor, minimum_cell_rows,
            disturbed_dst_threshold_nt, deepening_rate_threshold_nt_per_h,
            coupling_threshold_mvm, supported_model_steps, cells,
        )
    end
end

function OperationalV22Stack(cells::AbstractVector{OperationalV22Cell};
        label::AbstractString="operational_v2_2",
        sindy_mass_floor::Real=OPERATIONAL_V22_DEFAULT_SINDY_MASS_FLOOR,
        minimum_cell_rows::Integer=OPERATIONAL_V22_DEFAULT_MINIMUM_CELL_ROWS,
        disturbed_dst_threshold_nt::Real=OPERATIONAL_V22_DEFAULT_DISTURBED_DST_NT,
        deepening_rate_threshold_nt_per_h::Real=
            OPERATIONAL_V22_DEFAULT_DEEPENING_RATE_NT_PER_H,
        coupling_threshold_mvm::Real=OPERATIONAL_V22_DEFAULT_COUPLING_THRESHOLD_MVM,
        supported_model_steps=nothing)
    ordered_cells = sort!(collect(cells); by=c -> (c.model_step_hours, String(c.regime)))
    steps = if supported_model_steps === nothing
        sort!(unique([
            c.model_step_hours for c in ordered_cells
            if c.regime == OPERATIONAL_V22_POOLED_REGIME
        ]))
    else
        Int.(collect(supported_model_steps))
    end
    return OperationalV22Stack(
        String(label), Float64(sindy_mass_floor), Int(minimum_cell_rows),
        Float64(disturbed_dst_threshold_nt),
        Float64(deepening_rate_threshold_nt_per_h),
        Float64(coupling_threshold_mvm), tuple(steps...), tuple(ordered_cells...),
    )
end

"""
    operational_v22_regime(latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm; kwargs...)

Classify an issue using only causal issue-time state. Strong coupled deepening or
any still-deepening disturbed state is `:active_deepening`; a disturbed state
that is no longer falling is `:recovery`; all other states are `:quiet`.
"""
function operational_v22_regime(latest_dst_nt::Real,
                                dst_rate_nt_per_h::Real,
                                causal_coupling_mvm::Real;
                                disturbed_dst_threshold_nt::Real=
                                    OPERATIONAL_V22_DEFAULT_DISTURBED_DST_NT,
                                deepening_rate_threshold_nt_per_h::Real=
                                    OPERATIONAL_V22_DEFAULT_DEEPENING_RATE_NT_PER_H,
                                coupling_threshold_mvm::Real=
                                    OPERATIONAL_V22_DEFAULT_COUPLING_THRESHOLD_MVM)
    latest = Float64(latest_dst_nt)
    rate = Float64(dst_rate_nt_per_h)
    coupling = Float64(causal_coupling_mvm)
    disturbed_threshold = Float64(disturbed_dst_threshold_nt)
    rate_threshold = Float64(deepening_rate_threshold_nt_per_h)
    coupling_threshold = Float64(coupling_threshold_mvm)
    all(isfinite, (latest, rate, coupling, disturbed_threshold, rate_threshold,
                   coupling_threshold)) || throw(ArgumentError(
        "V2.2 regime inputs and thresholds must be finite",
    ))
    rate_threshold < 0.0 || throw(ArgumentError(
        "V2.2 deepening-rate threshold must be negative",
    ))
    coupling >= 0.0 && coupling_threshold >= 0.0 || throw(ArgumentError(
        "V2.2 causal coupling and its threshold must be nonnegative",
    ))

    strongly_driven = rate <= rate_threshold && coupling > coupling_threshold
    if strongly_driven || (latest <= disturbed_threshold && rate < 0.0)
        return :active_deepening
    elseif latest <= disturbed_threshold
        return :recovery
    end
    return :quiet
end

function operational_v22_regime(stack::OperationalV22Stack,
                                latest_dst_nt::Real,
                                dst_rate_nt_per_h::Real,
                                causal_coupling_mvm::Real)
    return operational_v22_regime(
        latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm;
        disturbed_dst_threshold_nt=stack.disturbed_dst_threshold_nt,
        deepening_rate_threshold_nt_per_h=stack.deepening_rate_threshold_nt_per_h,
        coupling_threshold_mvm=stack.coupling_threshold_mvm,
    )
end

# Exact Euclidean projection onto {x >= 0, sum(x) = mass}.
function _operational_v22_project_simplex(values::AbstractVector{<:Real}, mass::Real)
    m = Float64(mass)
    isfinite(m) && m >= 0.0 || throw(ArgumentError(
        "simplex mass must be finite and nonnegative",
    ))
    v = Float64.(values)
    all(isfinite, v) || throw(ArgumentError("simplex values must be finite"))
    iszero(m) && return zeros(length(v))
    isempty(v) && throw(ArgumentError("cannot project an empty vector to positive mass"))

    sorted = sort(v; rev=true)
    cumulative = 0.0
    rho = 0
    cumulative_at_rho = 0.0
    for j in eachindex(sorted)
        cumulative += sorted[j]
        if sorted[j] - (cumulative - m) / j > 0.0
            rho = j
            cumulative_at_rho = cumulative
        end
    end
    rho > 0 || throw(ErrorException("simplex projection failed to identify an active set"))
    theta = (cumulative_at_rho - m) / rho
    projected = max.(v .- theta, 0.0)
    # Remove the final floating summation residual without changing the active set.
    idx = argmax(projected)
    projected[idx] += m - sum(projected)
    projected[idx] >= 0.0 || throw(ErrorException(
        "simplex projection produced a negative correction",
    ))
    return projected
end

# Exact projection onto the unit simplex intersected with the physical-mass cap.
# If the ordinary simplex projection violates the cap, the cap is active and the
# problem separates into a SINDy simplex of mass=floor and a physical simplex of
# mass=1-floor.
function _operational_v22_project_weights(values::AbstractVector{<:Real},
                                          sindy_mass_floor::Real)
    length(values) == length(OPERATIONAL_V22_COMPONENTS) || throw(DimensionMismatch(
        "V2.2 projection requires six component values",
    ))
    floor64 = Float64(sindy_mass_floor)
    isfinite(floor64) && 0.0 < floor64 <= 1.0 || throw(ArgumentError(
        "sindy_mass_floor must lie in (0, 1]",
    ))
    cap = 1.0 - floor64
    ordinary = _operational_v22_project_simplex(values, 1.0)
    sum(@view ordinary[3:6]) <= cap + 8eps(Float64) && return ordinary

    out = zeros(length(OPERATIONAL_V22_COMPONENTS))
    out[1:2] = _operational_v22_project_simplex(@view(values[1:2]), floor64)
    out[3:6] = _operational_v22_project_simplex(@view(values[3:6]), cap)
    return out
end

function _operational_v22_fit_weights(centers::Matrix{Float64},
                                      observations::Vector{Float64},
                                      sindy_mass_floor::Float64;
                                      tolerance::Float64,
                                      max_iterations::Int)
    n, p = size(centers)
    p == length(OPERATIONAL_V22_COMPONENTS) || throw(DimensionMismatch(
        "V2.2 center matrix must have six columns",
    ))
    length(observations) == n || throw(DimensionMismatch(
        "V2.2 observations and centers must have equal rows",
    ))
    n >= 1 || throw(ArgumentError("V2.2 fit requires at least one row"))
    all(isfinite, centers) && all(isfinite, observations) || throw(ArgumentError(
        "V2.2 fit inputs must be finite",
    ))
    isfinite(tolerance) && tolerance > 0.0 || throw(ArgumentError(
        "V2.2 PGD tolerance must be finite and positive",
    ))
    max_iterations >= 1 || throw(ArgumentError(
        "V2.2 PGD max_iterations must be positive",
    ))

    # The simplex fixes sum(w)=1. Subtracting each row's component mean therefore
    # leaves the objective exactly unchanged while removing the large common Dst
    # offset that would otherwise make the six-by-six Gram matrix ill-conditioned.
    row_reference = vec(mean(centers; dims=2))
    design = centers .- reshape(row_reference, :, 1)
    target = observations .- row_reference
    gram = Matrix(Symmetric((design' * design) / n))
    cross = (design' * target) / n
    all(isfinite, gram) && all(isfinite, cross) || throw(ArgumentError(
        "V2.2 Gram system became non-finite",
    ))

    weights = _operational_v22_project_weights(fill(1.0 / p, p), sindy_mass_floor)
    lipschitz = eigmax(Symmetric(gram))
    isfinite(lipschitz) && lipschitz >= 0.0 || throw(ErrorException(
        "V2.2 Gram matrix has an invalid Lipschitz constant",
    ))
    if lipschitz <= eps(Float64)
        residual = centers * weights .- observations
        mse = sum(abs2, residual) / n
        isfinite(mse) || throw(ErrorException("V2.2 degenerate fit produced non-finite MSE"))
        return (weights=weights, objective_mse=mse, iterations=0)
    end

    step = 1.0 / lipschitz
    for iteration in 1:max_iterations
        gradient = gram * weights .- cross
        candidate = _operational_v22_project_weights(
            weights .- step .* gradient,
            sindy_mass_floor,
        )
        mapping_norm = maximum(abs.((weights .- candidate) ./ step))
        scale = max(1.0, maximum(abs.(gram * candidate)), maximum(abs.(cross)))
        weights = candidate
        if mapping_norm <= tolerance * scale
            residual = centers * weights .- observations
            mse = sum(abs2, residual) / n
            isfinite(mse) || throw(ErrorException("V2.2 fit produced non-finite MSE"))
            return (weights=weights, objective_mse=mse, iterations=iteration)
        end
    end
    throw(ErrorException(
        "V2.2 constrained least-squares PGD did not converge in $max_iterations iterations",
    ))
end

function _operational_v22_require_columns(df::DataFrame, columns)
    missing_columns = [String(c) for c in columns if !(String(c) in names(df))]
    isempty(missing_columns) || throw(ArgumentError(
        "missing required V2.2 column(s): $(join(missing_columns, ", "))",
    ))
    return nothing
end

function _operational_v22_finite_cell(df::DataFrame, row::Int, column::Symbol)
    raw = df[row, column]
    ismissing(raw) && throw(ArgumentError(
        "V2.2 column $column contains missing at row $row",
    ))
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "V2.2 column $column must contain real values",
    ))
    value = Float64(raw)
    isfinite(value) || throw(ArgumentError(
        "V2.2 column $column contains a non-finite value at row $row",
    ))
    return value
end

function _operational_v22_model_step(df::DataFrame, row::Int, column::Symbol)
    value = _operational_v22_finite_cell(df, row, column)
    value > 0.0 && isinteger(value) || throw(ArgumentError(
        "V2.2 model steps must be positive integers",
    ))
    value <= typemax(Int) || throw(ArgumentError("V2.2 model step exceeds Int range"))
    return Int(value)
end

function _operational_v22_validate_component_columns(component_columns::NamedTuple)
    propertynames(component_columns) == OPERATIONAL_V22_COMPONENTS || throw(ArgumentError(
        "component_columns must have the exact ordered keys " *
        join(String.(OPERATIONAL_V22_COMPONENTS), ","),
    ))
    columns = Tuple(Symbol(getproperty(component_columns, c))
                    for c in OPERATIONAL_V22_COMPONENTS)
    length(unique(columns)) == length(columns) || throw(ArgumentError(
        "V2.2 component columns must be unique",
    ))
    return columns
end

"""
    fit_operational_v22_stack(df; kwargs...)

Fit nonnegative constrained least-squares weights for each supported lead and
each sufficiently populated causal regime. Every lead also receives a pooled
fallback. The optimizer uses a precomputed 6×6 Gram system and projected-gradient
descent with exact projection onto the simplex and the physical-mass cap.
"""
function fit_operational_v22_stack(df::DataFrame;
        component_columns::NamedTuple=DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours,
        latest_dst_column::Symbol=:latest_dst_nt,
        dst_rate_column::Symbol=:dst_delta_1h_nt,
        causal_coupling_column::Symbol=:coupling_active_mvm,
        sindy_mass_floor::Real=OPERATIONAL_V22_DEFAULT_SINDY_MASS_FLOOR,
        minimum_cell_rows::Integer=OPERATIONAL_V22_DEFAULT_MINIMUM_CELL_ROWS,
        disturbed_dst_threshold_nt::Real=OPERATIONAL_V22_DEFAULT_DISTURBED_DST_NT,
        deepening_rate_threshold_nt_per_h::Real=
            OPERATIONAL_V22_DEFAULT_DEEPENING_RATE_NT_PER_H,
        coupling_threshold_mvm::Real=OPERATIONAL_V22_DEFAULT_COUPLING_THRESHOLD_MVM,
        tolerance::Real=1e-9,
        max_iterations::Integer=200_000,
        label::AbstractString="operational_v2_2")
    component_cols = _operational_v22_validate_component_columns(component_columns)
    required = (
        component_cols..., observation_column, model_step_column, latest_dst_column,
        dst_rate_column, causal_coupling_column,
    )
    _operational_v22_require_columns(df, required)
    minimum_rows = Int(minimum_cell_rows)
    minimum_rows >= 1 || throw(ArgumentError("minimum_cell_rows must be positive"))
    nrow(df) >= minimum_rows || throw(ArgumentError(
        "V2.2 fit has fewer rows than minimum_cell_rows",
    ))

    floor64 = Float64(sindy_mass_floor)
    disturbed_threshold = Float64(disturbed_dst_threshold_nt)
    rate_threshold = Float64(deepening_rate_threshold_nt_per_h)
    coupling_threshold = Float64(coupling_threshold_mvm)
    tol64 = Float64(tolerance)
    maxiter = Int(max_iterations)

    centers = Matrix{Float64}(undef, nrow(df), length(component_cols))
    observations = Vector{Float64}(undef, nrow(df))
    model_steps = Vector{Int}(undef, nrow(df))
    regimes = Vector{Symbol}(undef, nrow(df))
    for row in 1:nrow(df)
        for (j, column) in enumerate(component_cols)
            centers[row, j] = _operational_v22_finite_cell(df, row, column)
        end
        observations[row] = _operational_v22_finite_cell(df, row, observation_column)
        model_steps[row] = _operational_v22_model_step(df, row, model_step_column)
        regimes[row] = operational_v22_regime(
            _operational_v22_finite_cell(df, row, latest_dst_column),
            _operational_v22_finite_cell(df, row, dst_rate_column),
            _operational_v22_finite_cell(df, row, causal_coupling_column);
            disturbed_dst_threshold_nt=disturbed_threshold,
            deepening_rate_threshold_nt_per_h=rate_threshold,
            coupling_threshold_mvm=coupling_threshold,
        )
    end

    cells = OperationalV22Cell[]
    for lead in sort!(unique(model_steps))
        pooled_indices = findall(==(lead), model_steps)
        length(pooled_indices) >= minimum_rows || throw(ArgumentError(
            "V2.2 pooled lead $lead has fewer than minimum_cell_rows=$minimum_rows",
        ))
        pooled_fit = _operational_v22_fit_weights(
            centers[pooled_indices, :], observations[pooled_indices], floor64;
            tolerance=tol64, max_iterations=maxiter,
        )
        push!(cells, OperationalV22Cell(
            lead, OPERATIONAL_V22_POOLED_REGIME, length(pooled_indices),
            pooled_fit.weights;
            objective_mse=pooled_fit.objective_mse,
            iterations=pooled_fit.iterations,
        ))
        for regime in OPERATIONAL_V22_REGIMES
            indices = [i for i in pooled_indices if regimes[i] == regime]
            length(indices) >= minimum_rows || continue
            fitted = _operational_v22_fit_weights(
                centers[indices, :], observations[indices], floor64;
                tolerance=tol64, max_iterations=maxiter,
            )
            push!(cells, OperationalV22Cell(
                lead, regime, length(indices), fitted.weights;
                objective_mse=fitted.objective_mse,
                iterations=fitted.iterations,
            ))
        end
    end
    return OperationalV22Stack(
        cells;
        label=label,
        sindy_mass_floor=floor64,
        minimum_cell_rows=minimum_rows,
        disturbed_dst_threshold_nt=disturbed_threshold,
        deepening_rate_threshold_nt_per_h=rate_threshold,
        coupling_threshold_mvm=coupling_threshold,
    )
end

function _operational_v22_cell(stack::OperationalV22Stack,
                               model_step_hours::Int,
                               regime::Symbol)
    model_step_hours in stack.supported_model_steps || throw(ArgumentError(
        "unsupported V2.2 model_step_hours=$model_step_hours; supported steps are " *
        join(stack.supported_model_steps, ","),
    ))
    pooled = nothing
    for cell in stack.cells
        cell.model_step_hours == model_step_hours || continue
        cell.regime == regime && return (cell=cell, used_fallback=false)
        cell.regime == OPERATIONAL_V22_POOLED_REGIME && (pooled = cell)
    end
    pooled === nothing && throw(ErrorException(
        "V2.2 stack lacks its required pooled fallback for lead $model_step_hours",
    ))
    return (cell=pooled, used_fallback=true)
end

function _operational_v22_center(centers, component::Symbol)
    raw = if centers isa NamedTuple
        haskey(centers, component) || throw(ArgumentError(
            "missing V2.2 center: $component",
        ))
        getfield(centers, component)
    elseif centers isa AbstractDict
        haskey(centers, component) || throw(ArgumentError(
            "missing V2.2 center: $component",
        ))
        centers[component]
    else
        throw(ArgumentError("V2.2 centers must be a NamedTuple or dictionary"))
    end
    ismissing(raw) && throw(ArgumentError("V2.2 center $component is missing"))
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "V2.2 center $component must be real",
    ))
    value = Float64(raw)
    isfinite(value) || throw(ArgumentError("V2.2 center $component must be finite"))
    return value
end

"""
    operational_v22_predict(stack, model_step_hours, latest_dst_nt,
                            dst_rate_nt_per_h, causal_coupling_mvm, centers)

Pure V2.2 point prediction. Returns the total, chosen causal regime, fallback
status, weights, and each component's additive contribution for immutable logging.
"""
function operational_v22_predict(stack::OperationalV22Stack,
                                 model_step_hours::Integer,
                                 latest_dst_nt::Real,
                                 dst_rate_nt_per_h::Real,
                                 causal_coupling_mvm::Real,
                                 centers)
    lead = Int(model_step_hours)
    lead > 0 || throw(ArgumentError("V2.2 model_step_hours must be positive"))
    regime = operational_v22_regime(
        stack, latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm,
    )
    selected = _operational_v22_cell(stack, lead, regime)
    values = ntuple(
        i -> _operational_v22_center(centers, OPERATIONAL_V22_COMPONENTS[i]),
        length(OPERATIONAL_V22_COMPONENTS),
    )
    contributions = ntuple(
        i -> selected.cell.weights[i] * values[i],
        length(OPERATIONAL_V22_COMPONENTS),
    )
    total = sum(contributions)
    isfinite(total) || throw(ArgumentError("V2.2 prediction became non-finite"))
    sindy_mass = selected.cell.weights[1] + selected.cell.weights[2]
    sindy_center = (contributions[1] + contributions[2]) / sindy_mass
    correction = total - sindy_center
    all(isfinite, (sindy_center, correction)) || throw(ArgumentError(
        "V2.2 SINDy-family center or correction became non-finite",
    ))
    return (
        pred_dst=total,
        sindy_family_center_dst=sindy_center,
        correction_dst=correction,
        model_step_hours=lead,
        regime=regime,
        cell_regime=selected.cell.regime,
        used_pooled_fallback=selected.used_fallback,
        weights=NamedTuple{OPERATIONAL_V22_COMPONENTS}(selected.cell.weights),
        component_contributions=NamedTuple{OPERATIONAL_V22_COMPONENTS}(contributions),
        sindy_mass=sindy_mass,
        label=stack.label,
    )
end

"Apply a fitted V2.2 stack row by row without consulting any post-issue column."
function score_operational_v22(df::DataFrame,
                               stack::OperationalV22Stack;
        component_columns::NamedTuple=DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours,
        latest_dst_column::Symbol=:latest_dst_nt,
        dst_rate_column::Symbol=:dst_delta_1h_nt,
        causal_coupling_column::Symbol=:coupling_active_mvm)
    component_cols = _operational_v22_validate_component_columns(component_columns)
    _operational_v22_require_columns(
        df,
        (component_cols..., model_step_column, latest_dst_column, dst_rate_column,
         causal_coupling_column),
    )
    out = copy(df)
    n = nrow(out)
    predicted = Vector{Float64}(undef, n)
    regimes = Vector{String}(undef, n)
    cell_regimes = Vector{String}(undef, n)
    fallback = Vector{Bool}(undef, n)
    sindy_mass = Vector{Float64}(undef, n)
    sindy_center = Vector{Float64}(undef, n)
    correction = Vector{Float64}(undef, n)
    weights = zeros(n, length(OPERATIONAL_V22_COMPONENTS))
    contributions = similar(weights)

    for row in 1:n
        centers = NamedTuple{OPERATIONAL_V22_COMPONENTS}(
            ntuple(j -> _operational_v22_finite_cell(out, row, component_cols[j]),
                   length(component_cols)),
        )
        result = operational_v22_predict(
            stack,
            _operational_v22_model_step(out, row, model_step_column),
            _operational_v22_finite_cell(out, row, latest_dst_column),
            _operational_v22_finite_cell(out, row, dst_rate_column),
            _operational_v22_finite_cell(out, row, causal_coupling_column),
            centers,
        )
        predicted[row] = result.pred_dst
        regimes[row] = String(result.regime)
        cell_regimes[row] = String(result.cell_regime)
        fallback[row] = result.used_pooled_fallback
        sindy_mass[row] = result.sindy_mass
        sindy_center[row] = result.sindy_family_center_dst
        correction[row] = result.correction_dst
        for (j, component) in enumerate(OPERATIONAL_V22_COMPONENTS)
            weights[row, j] = getfield(result.weights, component)
            contributions[row, j] = getfield(result.component_contributions, component)
        end
    end

    out[!, :v2_2_pred_dst_nt] = predicted
    out[!, :v2_2_regime] = regimes
    out[!, :v2_2_cell_regime] = cell_regimes
    out[!, :v2_2_used_pooled_fallback] = fallback
    out[!, :v2_2_sindy_mass] = sindy_mass
    out[!, :v2_2_sindy_family_center_dst_nt] = sindy_center
    out[!, :v2_2_correction_dst_nt] = correction
    out[!, :v2_2_stack_label] = fill(stack.label, n)
    for (j, component) in enumerate(OPERATIONAL_V22_COMPONENTS)
        out[!, Symbol("v2_2_weight_", component)] = weights[:, j]
        out[!, Symbol("v2_2_contribution_", component, "_nt")] = contributions[:, j]
    end
    if String(observation_column) in names(out)
        residuals = Vector{Union{Missing,Float64}}(undef, n)
        for row in 1:n
            raw = out[row, observation_column]
            if ismissing(raw)
                residuals[row] = missing
            else
                raw isa Real && !(raw isa Bool) || throw(ArgumentError(
                    "V2.2 observation must be real or missing",
                ))
                observed = Float64(raw)
                isfinite(observed) || throw(ArgumentError(
                    "V2.2 observation must be finite when present",
                ))
                residuals[row] = observed - predicted[row]
            end
        end
        out[!, :v2_2_residual_dst_nt] = residuals
    end
    return out
end

const _OPERATIONAL_V22_CSV_COLUMNS = (
    :schema_version, :label, :sindy_mass_floor, :minimum_cell_rows,
    :disturbed_dst_threshold_nt, :deepening_rate_threshold_nt_per_h,
    :coupling_threshold_mvm, :supported_model_steps, :model_step_hours,
    :regime, :n_rows, :weight_served_v2_1, :weight_frozen_v2_1,
    :weight_persistence, :weight_burton, :weight_burton_full, :weight_obrien,
    :objective_mse, :iterations,
)

"Atomically write a strictly versioned Operational V2.2 stack CSV."
function write_operational_v22_stack(path::AbstractString,
                                     stack::OperationalV22Stack)
    target = String(path)
    mkpath(dirname(abspath(target)))
    supported = join(stack.supported_model_steps, ";")
    rows = NamedTuple[]
    for cell in stack.cells
        push!(rows, (
            schema_version=OPERATIONAL_V22_STACK_SCHEMA_VERSION,
            label=stack.label,
            sindy_mass_floor=stack.sindy_mass_floor,
            minimum_cell_rows=stack.minimum_cell_rows,
            disturbed_dst_threshold_nt=stack.disturbed_dst_threshold_nt,
            deepening_rate_threshold_nt_per_h=stack.deepening_rate_threshold_nt_per_h,
            coupling_threshold_mvm=stack.coupling_threshold_mvm,
            supported_model_steps=supported,
            model_step_hours=cell.model_step_hours,
            regime=String(cell.regime),
            n_rows=cell.n_rows,
            weight_served_v2_1=cell.weights[1],
            weight_frozen_v2_1=cell.weights[2],
            weight_persistence=cell.weights[3],
            weight_burton=cell.weights[4],
            weight_burton_full=cell.weights[5],
            weight_obrien=cell.weights[6],
            objective_mse=cell.objective_mse,
            iterations=cell.iterations,
        ))
    end
    _write_selection_csv(target, rows)
    return target
end

function _operational_v22_consistent_column(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "V2.2 stack metadata column $column contains missing",
    ))
    first_value = first(values)
    all(isequal(first_value), values) || throw(ArgumentError(
        "V2.2 stack metadata column $column is inconsistent",
    ))
    return first_value
end

function _operational_v22_csv_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$field must be an integer",
    ))
    converted = Float64(value)
    isfinite(converted) && isinteger(converted) && converted <= typemax(Int) &&
        converted >= typemin(Int) || throw(ArgumentError("$field must be an integer"))
    return Int(converted)
end

"Read and strictly validate an Operational V2.2 stack CSV."
function read_operational_v22_stack(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2 stack must be a regular non-symlink file: $source",
    ))
    df = CSV.read(source, DataFrame)
    names(df) == collect(String.(_OPERATIONAL_V22_CSV_COLUMNS)) || throw(ArgumentError(
        "V2.2 stack CSV schema does not exactly match " *
        OPERATIONAL_V22_STACK_SCHEMA_VERSION,
    ))
    nrow(df) >= 1 || throw(ArgumentError("V2.2 stack CSV is empty"))
    schema = String(_operational_v22_consistent_column(df, :schema_version))
    schema == OPERATIONAL_V22_STACK_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported V2.2 stack schema version: $schema",
    ))
    label = String(_operational_v22_consistent_column(df, :label))
    floor64 = Float64(_operational_v22_consistent_column(df, :sindy_mass_floor))
    minimum_rows = _operational_v22_csv_int(
        _operational_v22_consistent_column(df, :minimum_cell_rows),
        "minimum_cell_rows",
    )
    disturbed_threshold = Float64(_operational_v22_consistent_column(
        df, :disturbed_dst_threshold_nt,
    ))
    rate_threshold = Float64(_operational_v22_consistent_column(
        df, :deepening_rate_threshold_nt_per_h,
    ))
    coupling_threshold = Float64(_operational_v22_consistent_column(
        df, :coupling_threshold_mvm,
    ))
    supported_text = string(_operational_v22_consistent_column(
        df, :supported_model_steps,
    ))
    isempty(supported_text) && throw(ArgumentError(
        "V2.2 supported_model_steps metadata is empty",
    ))
    supported = try
        parse.(Int, split(supported_text, ";"))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("V2.2 supported_model_steps metadata is invalid"))
    end

    weight_columns = (
        :weight_served_v2_1, :weight_frozen_v2_1, :weight_persistence,
        :weight_burton, :weight_burton_full, :weight_obrien,
    )
    cells = OperationalV22Cell[]
    for row in 1:nrow(df)
        any(ismissing(df[row, column]) for column in _OPERATIONAL_V22_CSV_COLUMNS) &&
            throw(ArgumentError("V2.2 stack CSV contains missing at row $row"))
        weights = Float64[df[row, column] for column in weight_columns]
        push!(cells, OperationalV22Cell(
            _operational_v22_csv_int(df[row, :model_step_hours], "model_step_hours"),
            Symbol(String(df[row, :regime])),
            _operational_v22_csv_int(df[row, :n_rows], "n_rows"),
            weights;
            objective_mse=Float64(df[row, :objective_mse]),
            iterations=_operational_v22_csv_int(df[row, :iterations], "iterations"),
        ))
    end
    return OperationalV22Stack(
        cells;
        label=label,
        sindy_mass_floor=floor64,
        minimum_cell_rows=minimum_rows,
        disturbed_dst_threshold_nt=disturbed_threshold,
        deepening_rate_threshold_nt_per_h=rate_threshold,
        coupling_threshold_mvm=coupling_threshold,
        supported_model_steps=supported,
    )
end
