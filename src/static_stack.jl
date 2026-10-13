# Bounded, SINDy-dominant stacking layer for Operational static stack research.

"Ordered point-forecast components accepted by the static stack."
const STATIC_STACK_COMPONENTS = (
    :served_inertial_sindy,
    :frozen_inertial_sindy,
    :persistence,
    :burton,
    :burton_full,
    :obrien,
)

"Default replay-table columns corresponding to `STATIC_STACK_COMPONENTS`."
const DEFAULT_STATIC_STACK_COMPONENT_COLUMNS = (
    served_inertial_sindy=:served_inertial_sindy_dst_nt,
    frozen_inertial_sindy=:frozen_inertial_sindy_dst_nt,
    persistence=:persistence_dst_nt,
    burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,
    obrien=:obrien_dst_nt,
)

# Column names recorded by the release-era writers inside the sha-pinned frozen artifacts
# (development and residual replays, serialized stacks, serialized residual cores). Readers
# map them onto the canonical names after the byte-level digest check, so the recorded bytes
# are never modified and a frozen artifact loads identically under either serialization.
const _STATIC_STACK_FROZEN_COLUMN_NAMES = Dict{Symbol,Symbol}(
    :served_v2_1_dst_nt => :served_inertial_sindy_dst_nt,
    :frozen_v2_1_dst_nt => :frozen_inertial_sindy_dst_nt,
    :v2_2_pred_dst_nt => :static_stack_pred_dst_nt,
    :v2_2_stack_pred_dst_nt => :static_stack_pred_dst_nt,
    :v2_2_stack_ci05_dst_nt => :static_stack_ci05_dst_nt,
    :v2_2_regime => :static_stack_regime,
    :v2_2_cell_regime => :static_stack_cell_regime,
    :v2_2_used_pooled_fallback => :static_stack_used_pooled_fallback,
    :v2_2_pooled_fallback => :static_stack_pooled_fallback,
    :v2_2_coupling_active_mvm => :static_stack_coupling_active_mvm,
    :v2_2_status => :static_stack_status,
    :v2_2_sindy_mass => :static_stack_sindy_mass,
    :v2_2_sindy_family_center_dst_nt => :static_stack_sindy_family_center_dst_nt,
    :v2_2_correction_dst_nt => :static_stack_correction_dst_nt,
    :v2_2_stack_label => :static_stack_label,
    :v2_2_residual_dst_nt => :static_stack_residual_dst_nt,
    :v2_2_weight_served_v2_1 => :static_stack_weight_served_inertial_sindy,
    :v2_2_weight_frozen_v2_1 => :static_stack_weight_frozen_inertial_sindy,
    :v2_2_weight_persistence => :static_stack_weight_persistence,
    :v2_2_weight_burton => :static_stack_weight_burton,
    :v2_2_weight_burton_full => :static_stack_weight_burton_full,
    :v2_2_weight_obrien => :static_stack_weight_obrien,
    :v2_2_contribution_served_v2_1_nt => :static_stack_contribution_served_inertial_sindy_nt,
    :v2_2_contribution_frozen_v2_1_nt => :static_stack_contribution_frozen_inertial_sindy_nt,
    :v2_2_contribution_persistence_nt => :static_stack_contribution_persistence_nt,
    :v2_2_contribution_burton_nt => :static_stack_contribution_burton_nt,
    :v2_2_contribution_burton_full_nt => :static_stack_contribution_burton_full_nt,
    :v2_2_contribution_obrien_nt => :static_stack_contribution_obrien_nt,
    :v2_2_secondary_pred_dst_nt => :static_stack_secondary_pred_dst_nt,
    :v2_2_secondary_raw_correction_nt => :static_stack_secondary_raw_correction_nt,
    :v2_2_secondary_correction_nt => :static_stack_secondary_correction_nt,
    :v2_2_secondary_correction_was_capped => :static_stack_secondary_correction_was_capped,
    :v2_2_secondary_correction_cap_nt => :static_stack_secondary_correction_cap_nt,
    :v2_2_secondary_ridge => :static_stack_secondary_ridge,
    :v2_2_secondary_top_k => :static_stack_secondary_top_k,
    :v2_2_secondary_feature_names => :static_stack_secondary_feature_names,
    :v2_2_secondary_coefficients => :static_stack_secondary_coefficients,
    :v2_2_secondary_feature_contributions_nt => :static_stack_secondary_feature_contributions_nt,
    :v2_2_secondary_label => :static_stack_secondary_label,
    :v2_2_secondary_residual_dst_nt => :static_stack_secondary_residual_dst_nt,
    :served_minus_frozen_v2_1_nt => :served_minus_frozen_inertial_sindy_nt,
    :primary_minus_served_v2_1_nt => :primary_minus_served_inertial_sindy_nt,
    :primary_minus_frozen_v2_1_nt => :primary_minus_frozen_inertial_sindy_nt,
    :weight_served_v2_1 => :weight_served_inertial_sindy,
    :weight_frozen_v2_1 => :weight_frozen_inertial_sindy,
    :v2_2_crossfit_fold => :static_stack_crossfit_fold,
    :v2_2_crossfit_block_start_utc => :static_stack_crossfit_block_start_utc,
    :v2_2_crossfit_block_stop_utc => :static_stack_crossfit_block_stop_utc,
    :v2_2_crossfit_fit_issue_max_utc => :static_stack_crossfit_fit_issue_max_utc,
    :v2_2_crossfit_fit_target_max_utc => :static_stack_crossfit_fit_target_max_utc,
    :v2_2_crossfit_fit_target_gap_hours => :static_stack_crossfit_fit_target_gap_hours,
    :v2_2_crossfit_fit_rows => :static_stack_crossfit_fit_rows,
)

# String view of the same map, used to canonicalize the semicolon-joined feature-name cells a
# frozen serialized core or scored replay records.
const _STATIC_STACK_FROZEN_NAME_TOKENS = Dict{String,String}(
    String(legacy) => String(canonical) for (legacy, canonical) in _STATIC_STACK_FROZEN_COLUMN_NAMES
)

# Schema markers the release-era serialized artifacts declare. The canonical writer emits only
# the canonical marker; the release-era marker still identifies the same format.
const _STATIC_STACK_FROZEN_SCHEMA_VERSIONS = Dict{String,String}(
    "operational_v2_2_stack_v1" => "static_stack_v1",
    "operational_v2_2_residual_v1" => "static_stack_residual_v1",
)

# Canonical feature name → the spelling recorded inside the sha-pinned frozen artifacts. Schema
# digests pinned against the recorded vocabulary hash these spellings.
const _STATIC_STACK_FROZEN_FEATURE_SPELLINGS = Dict{Symbol,Symbol}(
    :served_minus_frozen_inertial_sindy_nt => :served_minus_frozen_v2_1_nt,
    :primary_minus_served_inertial_sindy_nt => :primary_minus_served_v2_1_nt,
    :primary_minus_frozen_inertial_sindy_nt => :primary_minus_frozen_v2_1_nt,
)

# Artifact labels recorded under the release-era product name.
const _STATIC_STACK_FROZEN_LABEL_PATTERN = "operational_v2_2"

function _static_stack_frozen_cell_names(value)
    return join(
        (get(_STATIC_STACK_FROZEN_NAME_TOKENS, token, token)
         for token in split(String(value), ";")),
        ";",
    )
end

"""
    canonicalize_static_stack_frozen_table!(df)

Map the release-era column spellings recorded inside a sha-pinned frozen static-stack artifact
onto the canonical names, in place. Called after the artifact's byte-level digest check: the
recorded bytes are never modified, and a table that already uses canonical names passes through
unchanged. Throws when a table carries both spellings of one column, which would make the
recorded serialization ambiguous.
"""
function canonicalize_static_stack_frozen_table!(df::DataFrame)
    present = Set(Symbol.(names(df)))
    for (legacy, canonical) in _STATIC_STACK_FROZEN_COLUMN_NAMES
        legacy in present || continue
        canonical in present && throw(ArgumentError(
            "static stack table carries both $legacy and $canonical",
        ))
        rename!(df, String(legacy) => String(canonical))
    end
    for column in (:label, :static_stack_label, :static_stack_secondary_label)
        column in propertynames(df) || continue
        df[!, column] = map(df[!, column]) do cell
            text = String(cell)
            startswith(text, _STATIC_STACK_FROZEN_LABEL_PATTERN) ?
                "static_stack" * text[length(_STATIC_STACK_FROZEN_LABEL_PATTERN)+1:end] : text
        end
    end
    if :schema_version in propertynames(df)
        df[!, :schema_version] = get.(
            Ref(_STATIC_STACK_FROZEN_SCHEMA_VERSIONS),
            String.(df[!, :schema_version]), String.(df[!, :schema_version]),
        )
    end
    for column in (:candidate_feature_names, :feature_names, :ranked_feature_names,
                   :static_stack_secondary_feature_names)
        column in propertynames(df) || continue
        df[!, column] = _static_stack_frozen_cell_names.(df[!, column])
    end
    return df
end

const STATIC_STACK_REGIMES = (:quiet, :active_deepening, :recovery)
const STATIC_STACK_POOLED_REGIME = :pooled
const STATIC_STACK_DEFAULT_SINDY_MASS_FLOOR = 0.60
const STATIC_STACK_DEFAULT_MINIMUM_CELL_ROWS = 48
const STATIC_STACK_DEFAULT_DISTURBED_DST_NT = -30.0
const STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H = -5.0
const STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM = 0.0
const STATIC_STACK_SCHEMA_VERSION = "static_stack_v1"

const _STATIC_STACK_WEIGHT_TOLERANCE = 1e-10

"""
Spectral-radius reproducibility budget for the offline static stack companion artifacts (M2 driver, M3
error-state, M3 exogenous).

The radius is `maximum(abs, eigvals(companion))`. LAPACK reaches the same eigenvalues by different
orderings and blockings depending on the matrix layout and the BLAS thread count, so two runs on the
same coefficients can differ in the last few bits — `0.944846470372869` versus `0.9448464703728752`
was measured on a 125x125 companion here. An exact `==` between a stored radius and the one the
constructor recomputes therefore refuses a perfectly good artifact on a different host, and an exact
radius folded into a "portable" identity makes that identity host-dependent.

The reproducibility budget is the fix on both counts: the stored and recomputed radii must agree to
`rtol`, and the identity hashes the radius rounded to `digits`. The *stability* gate is untouched —
a candidate is still refused unless its recomputed radius satisfies the module's own
`radius <= 1 + stability tolerance`.
"""
const STATIC_STACK_SPECTRAL_RADIUS_RTOL = 1e-9
const STATIC_STACK_SPECTRAL_RADIUS_ATOL = 1e-12
const STATIC_STACK_SPECTRAL_RADIUS_DIGITS = 9

"Radius as a portable identity token: rounded so an eigen-solver path cannot move the hash."
_static_stack_hashable_spectral_radius(radius::Real) =
    round(Float64(radius); digits = STATIC_STACK_SPECTRAL_RADIUS_DIGITS)

"Whether a stored radius and a recomputed one describe the same companion matrix."
_static_stack_spectral_radius_agrees(stored::Real, recomputed::Real) =
    isapprox(Float64(stored), Float64(recomputed);
             rtol = STATIC_STACK_SPECTRAL_RADIUS_RTOL,
             atol = STATIC_STACK_SPECTRAL_RADIUS_ATOL)

"""
    STATIC_STACK_MODEL_STEPS

Forecast leads every Operational static stack layer supports. The residual core, the boosted layer, the M3
error state, the M3 exogenous correction, the hourly core path and the shadow-chain contract all
declare this same set; each used to spell it out separately, so a lead added to one layer and not the
others would have gone unnoticed until a cell lookup failed at issuance.
"""
const STATIC_STACK_MODEL_STEPS = (1, 2, 3, 4, 6, 7)

"""
    static_stack_correction_cap_nt(model_step_hours)

Bounded magnitude a static stack correction layer may add at a given lead: 5 nT plus 5 nT per lead hour. The
residual core, the boosted layer and the M3 error state share this bound by construction rather than
by three identical expressions.
"""
const STATIC_STACK_CORRECTION_CAP_BASE_NT = 5.0
const STATIC_STACK_CORRECTION_CAP_PER_HOUR_NT = 5.0
static_stack_correction_cap_nt(model_step_hours::Integer) =
    STATIC_STACK_CORRECTION_CAP_BASE_NT +
    STATIC_STACK_CORRECTION_CAP_PER_HOUR_NT * Int(model_step_hours)

"One immutable lead/regime weight cell in an [`StaticStack`](@ref)."
struct StaticStackCell
    model_step_hours::Int
    regime::Symbol
    n_rows::Int
    weights::NTuple{6,Float64}
    objective_mse::Float64
    iterations::Int

    function StaticStackCell(model_step_hours::Int,
                                regime::Symbol,
                                n_rows::Int,
                                weights::NTuple{6,Float64},
                                objective_mse::Float64,
                                iterations::Int)
        model_step_hours > 0 || throw(ArgumentError(
            "static stack model_step_hours must be positive",
        ))
        regime in (STATIC_STACK_POOLED_REGIME, STATIC_STACK_REGIMES...) ||
            throw(ArgumentError("unknown static stack regime: $regime"))
        n_rows >= 1 || throw(ArgumentError("static stack cell n_rows must be positive"))
        all(isfinite, weights) || throw(ArgumentError("static stack weights must be finite"))
        all(w -> w >= 0.0, weights) || throw(ArgumentError(
            "static stack weights must be nonnegative",
        ))
        isapprox(sum(weights), 1.0; rtol=0.0, atol=_STATIC_STACK_WEIGHT_TOLERANCE) ||
            throw(ArgumentError("static stack weights must sum to one"))
        isfinite(objective_mse) && objective_mse >= 0.0 || throw(ArgumentError(
            "static stack objective_mse must be finite and nonnegative",
        ))
        iterations >= 0 || throw(ArgumentError("static stack iterations must be nonnegative"))
        return new(model_step_hours, regime, n_rows, weights, objective_mse, iterations)
    end
end

function StaticStackCell(model_step_hours::Integer,
                            regime::Symbol,
                            n_rows::Integer,
                            weights::AbstractVector{<:Real};
                            objective_mse::Real=0.0,
                            iterations::Integer=0)
    length(weights) == length(STATIC_STACK_COMPONENTS) || throw(DimensionMismatch(
        "static stack requires $(length(STATIC_STACK_COMPONENTS)) component weights",
    ))
    converted = ntuple(i -> Float64(weights[i]), length(STATIC_STACK_COMPONENTS))
    return StaticStackCell(
        Int(model_step_hours), regime, Int(n_rows), converted,
        Float64(objective_mse), Int(iterations),
    )
end

"""
    StaticStack

Immutable collection of constrained lead/regime stacking cells. Every supported
lead has a pooled fallback. A regime-specific cell is used only when it met the
stored minimum-row threshold during fitting. The first two components are the
served and frozen-tail 20/11 SINDy forecasts; their combined mass is at least
`sindy_mass_floor` in every cell.
"""
struct StaticStack
    label::String
    sindy_mass_floor::Float64
    minimum_cell_rows::Int
    disturbed_dst_threshold_nt::Float64
    deepening_rate_threshold_nt_per_h::Float64
    coupling_threshold_mvm::Float64
    supported_model_steps::Tuple{Vararg{Int}}
    cells::Tuple{Vararg{StaticStackCell}}

    function StaticStack(label::String,
                                 sindy_mass_floor::Float64,
                                 minimum_cell_rows::Int,
                                 disturbed_dst_threshold_nt::Float64,
                                 deepening_rate_threshold_nt_per_h::Float64,
                                 coupling_threshold_mvm::Float64,
                                 supported_model_steps::Tuple{Vararg{Int}},
                                 cells::Tuple{Vararg{StaticStackCell}})
        isempty(strip(label)) && throw(ArgumentError("static stack label must not be empty"))
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
            "static stack must support at least one model step",
        ))
        collect(supported_model_steps) == sort!(unique(collect(supported_model_steps))) ||
            throw(ArgumentError("supported_model_steps must be sorted and unique"))
        all(>(0), supported_model_steps) || throw(ArgumentError(
            "supported_model_steps must be positive",
        ))
        isempty(cells) && throw(ArgumentError("static stack must contain cells"))

        seen = Set{Tuple{Int,Symbol}}()
        pooled = Set{Int}()
        for cell in cells
            cell.model_step_hours in supported_model_steps || throw(ArgumentError(
                "static stack cell has unsupported model step $(cell.model_step_hours)",
            ))
            cell.n_rows >= minimum_cell_rows || throw(ArgumentError(
                "static stack cell $(cell.model_step_hours)/$(cell.regime) is below minimum_cell_rows",
            ))
            key = (cell.model_step_hours, cell.regime)
            key in seen && throw(ArgumentError("duplicate static stack cell: $key"))
            push!(seen, key)
            cell.regime == STATIC_STACK_POOLED_REGIME &&
                push!(pooled, cell.model_step_hours)
            sindy_mass = cell.weights[1] + cell.weights[2]
            sindy_mass + _STATIC_STACK_WEIGHT_TOLERANCE >= sindy_mass_floor ||
                throw(ArgumentError(
                    "static stack cell $key violates the SINDy-family mass floor",
                ))
        end
        pooled == Set(supported_model_steps) || throw(ArgumentError(
            "every supported static stack model step requires exactly one pooled fallback",
        ))
        return new(
            label, sindy_mass_floor, minimum_cell_rows,
            disturbed_dst_threshold_nt, deepening_rate_threshold_nt_per_h,
            coupling_threshold_mvm, supported_model_steps, cells,
        )
    end
end

function StaticStack(cells::AbstractVector{StaticStackCell};
        label::AbstractString="static_stack",
        sindy_mass_floor::Real=STATIC_STACK_DEFAULT_SINDY_MASS_FLOOR,
        minimum_cell_rows::Integer=STATIC_STACK_DEFAULT_MINIMUM_CELL_ROWS,
        disturbed_dst_threshold_nt::Real=STATIC_STACK_DEFAULT_DISTURBED_DST_NT,
        deepening_rate_threshold_nt_per_h::Real=
            STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H,
        coupling_threshold_mvm::Real=STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM,
        supported_model_steps=nothing)
    ordered_cells = sort!(collect(cells); by=c -> (c.model_step_hours, String(c.regime)))
    steps = if supported_model_steps === nothing
        sort!(unique([
            c.model_step_hours for c in ordered_cells
            if c.regime == STATIC_STACK_POOLED_REGIME
        ]))
    else
        Int.(collect(supported_model_steps))
    end
    return StaticStack(
        String(label), Float64(sindy_mass_floor), Int(minimum_cell_rows),
        Float64(disturbed_dst_threshold_nt),
        Float64(deepening_rate_threshold_nt_per_h),
        Float64(coupling_threshold_mvm), tuple(steps...), tuple(ordered_cells...),
    )
end

"""
    static_stack_regime(latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm; kwargs...)

Classify an issue using only causal issue-time state. Strong coupled deepening or
any still-deepening disturbed state is `:active_deepening`; a disturbed state
that is no longer falling is `:recovery`; all other states are `:quiet`.
"""
function static_stack_regime(latest_dst_nt::Real,
                                dst_rate_nt_per_h::Real,
                                causal_coupling_mvm::Real;
                                disturbed_dst_threshold_nt::Real=
                                    STATIC_STACK_DEFAULT_DISTURBED_DST_NT,
                                deepening_rate_threshold_nt_per_h::Real=
                                    STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H,
                                coupling_threshold_mvm::Real=
                                    STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM)
    latest = Float64(latest_dst_nt)
    rate = Float64(dst_rate_nt_per_h)
    coupling = Float64(causal_coupling_mvm)
    disturbed_threshold = Float64(disturbed_dst_threshold_nt)
    rate_threshold = Float64(deepening_rate_threshold_nt_per_h)
    coupling_threshold = Float64(coupling_threshold_mvm)
    all(isfinite, (latest, rate, coupling, disturbed_threshold, rate_threshold,
                   coupling_threshold)) || throw(ArgumentError(
        "static stack regime inputs and thresholds must be finite",
    ))
    rate_threshold < 0.0 || throw(ArgumentError(
        "static stack deepening-rate threshold must be negative",
    ))
    coupling >= 0.0 && coupling_threshold >= 0.0 || throw(ArgumentError(
        "static stack causal coupling and its threshold must be nonnegative",
    ))

    strongly_driven = rate <= rate_threshold && coupling > coupling_threshold
    if strongly_driven || (latest <= disturbed_threshold && rate < 0.0)
        return :active_deepening
    elseif latest <= disturbed_threshold
        return :recovery
    end
    return :quiet
end

function static_stack_regime(stack::StaticStack,
                                latest_dst_nt::Real,
                                dst_rate_nt_per_h::Real,
                                causal_coupling_mvm::Real)
    return static_stack_regime(
        latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm;
        disturbed_dst_threshold_nt=stack.disturbed_dst_threshold_nt,
        deepening_rate_threshold_nt_per_h=stack.deepening_rate_threshold_nt_per_h,
        coupling_threshold_mvm=stack.coupling_threshold_mvm,
    )
end

# Exact Euclidean projection onto {x >= 0, sum(x) = mass}.
function _static_stack_project_simplex(values::AbstractVector{<:Real}, mass::Real)
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
function _static_stack_project_weights(values::AbstractVector{<:Real},
                                          sindy_mass_floor::Real)
    length(values) == length(STATIC_STACK_COMPONENTS) || throw(DimensionMismatch(
        "static stack projection requires six component values",
    ))
    floor64 = Float64(sindy_mass_floor)
    isfinite(floor64) && 0.0 < floor64 <= 1.0 || throw(ArgumentError(
        "sindy_mass_floor must lie in (0, 1]",
    ))
    cap = 1.0 - floor64
    ordinary = _static_stack_project_simplex(values, 1.0)
    sum(@view ordinary[3:6]) <= cap + 8eps(Float64) && return ordinary

    out = zeros(length(STATIC_STACK_COMPONENTS))
    out[1:2] = _static_stack_project_simplex(@view(values[1:2]), floor64)
    out[3:6] = _static_stack_project_simplex(@view(values[3:6]), cap)
    return out
end

function _static_stack_fit_weights(centers::Matrix{Float64},
                                      observations::Vector{Float64},
                                      sindy_mass_floor::Float64;
                                      tolerance::Float64,
                                      max_iterations::Int)
    n, p = size(centers)
    p == length(STATIC_STACK_COMPONENTS) || throw(DimensionMismatch(
        "static stack center matrix must have six columns",
    ))
    length(observations) == n || throw(DimensionMismatch(
        "static stack observations and centers must have equal rows",
    ))
    n >= 1 || throw(ArgumentError("static stack fit requires at least one row"))
    all(isfinite, centers) && all(isfinite, observations) || throw(ArgumentError(
        "static stack fit inputs must be finite",
    ))
    isfinite(tolerance) && tolerance > 0.0 || throw(ArgumentError(
        "static stack PGD tolerance must be finite and positive",
    ))
    max_iterations >= 1 || throw(ArgumentError(
        "static stack PGD max_iterations must be positive",
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
        "static stack Gram system became non-finite",
    ))

    weights = _static_stack_project_weights(fill(1.0 / p, p), sindy_mass_floor)
    lipschitz = eigmax(Symmetric(gram))
    isfinite(lipschitz) && lipschitz >= 0.0 || throw(ErrorException(
        "static stack Gram matrix has an invalid Lipschitz constant",
    ))
    if lipschitz <= eps(Float64)
        residual = centers * weights .- observations
        mse = sum(abs2, residual) / n
        isfinite(mse) || throw(ErrorException("static stack degenerate fit produced non-finite MSE"))
        return (weights=weights, objective_mse=mse, iterations=0)
    end

    step = 1.0 / lipschitz
    for iteration in 1:max_iterations
        gradient = gram * weights .- cross
        candidate = _static_stack_project_weights(
            weights .- step .* gradient,
            sindy_mass_floor,
        )
        mapping_norm = maximum(abs.((weights .- candidate) ./ step))
        scale = max(1.0, maximum(abs.(gram * candidate)), maximum(abs.(cross)))
        weights = candidate
        if mapping_norm <= tolerance * scale
            residual = centers * weights .- observations
            mse = sum(abs2, residual) / n
            isfinite(mse) || throw(ErrorException("static stack fit produced non-finite MSE"))
            return (weights=weights, objective_mse=mse, iterations=iteration)
        end
    end
    throw(ErrorException(
        "static stack constrained least-squares PGD did not converge in $max_iterations iterations",
    ))
end

function _static_stack_require_columns(df::DataFrame, columns)
    missing_columns = [String(c) for c in columns if !(String(c) in names(df))]
    isempty(missing_columns) || throw(ArgumentError(
        "missing required static stack column(s): $(join(missing_columns, ", "))",
    ))
    return nothing
end

function _static_stack_finite_cell(df::DataFrame, row::Int, column::Symbol)
    raw = df[row, column]
    ismissing(raw) && throw(ArgumentError(
        "static stack column $column contains missing at row $row",
    ))
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "static stack column $column must contain real values",
    ))
    value = Float64(raw)
    isfinite(value) || throw(ArgumentError(
        "static stack column $column contains a non-finite value at row $row",
    ))
    return value
end

function _static_stack_model_step(df::DataFrame, row::Int, column::Symbol)
    value = _static_stack_finite_cell(df, row, column)
    value > 0.0 && isinteger(value) || throw(ArgumentError(
        "static stack model steps must be positive integers",
    ))
    value <= typemax(Int) || throw(ArgumentError("static stack model step exceeds Int range"))
    return Int(value)
end

function _static_stack_validate_component_columns(component_columns::NamedTuple)
    propertynames(component_columns) == STATIC_STACK_COMPONENTS || throw(ArgumentError(
        "component_columns must have the exact ordered keys " *
        join(String.(STATIC_STACK_COMPONENTS), ","),
    ))
    columns = Tuple(Symbol(getproperty(component_columns, c))
                    for c in STATIC_STACK_COMPONENTS)
    length(unique(columns)) == length(columns) || throw(ArgumentError(
        "static stack component columns must be unique",
    ))
    return columns
end

"""
    fit_static_stack(df; kwargs...)

Fit nonnegative constrained least-squares weights for each supported lead and
each sufficiently populated causal regime. Every lead also receives a pooled
fallback. The optimizer uses a precomputed 6×6 Gram system and projected-gradient
descent with exact projection onto the simplex and the physical-mass cap.
"""
function fit_static_stack(df::DataFrame;
        component_columns::NamedTuple=DEFAULT_STATIC_STACK_COMPONENT_COLUMNS,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours,
        latest_dst_column::Symbol=:latest_dst_nt,
        dst_rate_column::Symbol=:dst_delta_1h_nt,
        causal_coupling_column::Symbol=:coupling_active_mvm,
        sindy_mass_floor::Real=STATIC_STACK_DEFAULT_SINDY_MASS_FLOOR,
        minimum_cell_rows::Integer=STATIC_STACK_DEFAULT_MINIMUM_CELL_ROWS,
        disturbed_dst_threshold_nt::Real=STATIC_STACK_DEFAULT_DISTURBED_DST_NT,
        deepening_rate_threshold_nt_per_h::Real=
            STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H,
        coupling_threshold_mvm::Real=STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM,
        tolerance::Real=1e-9,
        max_iterations::Integer=200_000,
        label::AbstractString="static_stack")
    component_cols = _static_stack_validate_component_columns(component_columns)
    required = (
        component_cols..., observation_column, model_step_column, latest_dst_column,
        dst_rate_column, causal_coupling_column,
    )
    _static_stack_require_columns(df, required)
    minimum_rows = Int(minimum_cell_rows)
    minimum_rows >= 1 || throw(ArgumentError("minimum_cell_rows must be positive"))
    nrow(df) >= minimum_rows || throw(ArgumentError(
        "static stack fit has fewer rows than minimum_cell_rows",
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
            centers[row, j] = _static_stack_finite_cell(df, row, column)
        end
        observations[row] = _static_stack_finite_cell(df, row, observation_column)
        model_steps[row] = _static_stack_model_step(df, row, model_step_column)
        regimes[row] = static_stack_regime(
            _static_stack_finite_cell(df, row, latest_dst_column),
            _static_stack_finite_cell(df, row, dst_rate_column),
            _static_stack_finite_cell(df, row, causal_coupling_column);
            disturbed_dst_threshold_nt=disturbed_threshold,
            deepening_rate_threshold_nt_per_h=rate_threshold,
            coupling_threshold_mvm=coupling_threshold,
        )
    end

    cells = StaticStackCell[]
    for lead in sort!(unique(model_steps))
        pooled_indices = findall(==(lead), model_steps)
        length(pooled_indices) >= minimum_rows || throw(ArgumentError(
            "static stack pooled lead $lead has fewer than minimum_cell_rows=$minimum_rows",
        ))
        pooled_fit = _static_stack_fit_weights(
            centers[pooled_indices, :], observations[pooled_indices], floor64;
            tolerance=tol64, max_iterations=maxiter,
        )
        push!(cells, StaticStackCell(
            lead, STATIC_STACK_POOLED_REGIME, length(pooled_indices),
            pooled_fit.weights;
            objective_mse=pooled_fit.objective_mse,
            iterations=pooled_fit.iterations,
        ))
        for regime in STATIC_STACK_REGIMES
            indices = [i for i in pooled_indices if regimes[i] == regime]
            length(indices) >= minimum_rows || continue
            fitted = _static_stack_fit_weights(
                centers[indices, :], observations[indices], floor64;
                tolerance=tol64, max_iterations=maxiter,
            )
            push!(cells, StaticStackCell(
                lead, regime, length(indices), fitted.weights;
                objective_mse=fitted.objective_mse,
                iterations=fitted.iterations,
            ))
        end
    end
    return StaticStack(
        cells;
        label=label,
        sindy_mass_floor=floor64,
        minimum_cell_rows=minimum_rows,
        disturbed_dst_threshold_nt=disturbed_threshold,
        deepening_rate_threshold_nt_per_h=rate_threshold,
        coupling_threshold_mvm=coupling_threshold,
    )
end

function _static_stack_cell(stack::StaticStack,
                               model_step_hours::Int,
                               regime::Symbol)
    model_step_hours in stack.supported_model_steps || throw(ArgumentError(
        "unsupported static stack model_step_hours=$model_step_hours; supported steps are " *
        join(stack.supported_model_steps, ","),
    ))
    pooled = nothing
    for cell in stack.cells
        cell.model_step_hours == model_step_hours || continue
        cell.regime == regime && return (cell=cell, used_fallback=false)
        cell.regime == STATIC_STACK_POOLED_REGIME && (pooled = cell)
    end
    pooled === nothing && throw(ErrorException(
        "static stack lacks its required pooled fallback for lead $model_step_hours",
    ))
    return (cell=pooled, used_fallback=true)
end

function _static_stack_center(centers, component::Symbol)
    raw = if centers isa NamedTuple
        haskey(centers, component) || throw(ArgumentError(
            "missing static stack center: $component",
        ))
        getfield(centers, component)
    elseif centers isa AbstractDict
        haskey(centers, component) || throw(ArgumentError(
            "missing static stack center: $component",
        ))
        centers[component]
    else
        throw(ArgumentError("static stack centers must be a NamedTuple or dictionary"))
    end
    ismissing(raw) && throw(ArgumentError("static stack center $component is missing"))
    raw isa Real && !(raw isa Bool) || throw(ArgumentError(
        "static stack center $component must be real",
    ))
    value = Float64(raw)
    isfinite(value) || throw(ArgumentError("static stack center $component must be finite"))
    return value
end

"""
    static_stack_predict(stack, model_step_hours, latest_dst_nt,
                            dst_rate_nt_per_h, causal_coupling_mvm, centers)

Pure static stack point prediction. Returns the total, chosen causal regime, fallback
status, weights, and each component's additive contribution for immutable logging.
"""
function static_stack_predict(stack::StaticStack,
                                 model_step_hours::Integer,
                                 latest_dst_nt::Real,
                                 dst_rate_nt_per_h::Real,
                                 causal_coupling_mvm::Real,
                                 centers)
    lead = Int(model_step_hours)
    lead > 0 || throw(ArgumentError("static stack model_step_hours must be positive"))
    regime = static_stack_regime(
        stack, latest_dst_nt, dst_rate_nt_per_h, causal_coupling_mvm,
    )
    selected = _static_stack_cell(stack, lead, regime)
    values = ntuple(
        i -> _static_stack_center(centers, STATIC_STACK_COMPONENTS[i]),
        length(STATIC_STACK_COMPONENTS),
    )
    contributions = ntuple(
        i -> selected.cell.weights[i] * values[i],
        length(STATIC_STACK_COMPONENTS),
    )
    total = sum(contributions)
    isfinite(total) || throw(ArgumentError("static stack prediction became non-finite"))
    sindy_mass = selected.cell.weights[1] + selected.cell.weights[2]
    sindy_center = (contributions[1] + contributions[2]) / sindy_mass
    correction = total - sindy_center
    all(isfinite, (sindy_center, correction)) || throw(ArgumentError(
        "static stack SINDy-family center or correction became non-finite",
    ))
    return (
        pred_dst=total,
        sindy_family_center_dst=sindy_center,
        correction_dst=correction,
        model_step_hours=lead,
        regime=regime,
        cell_regime=selected.cell.regime,
        used_pooled_fallback=selected.used_fallback,
        weights=NamedTuple{STATIC_STACK_COMPONENTS}(selected.cell.weights),
        component_contributions=NamedTuple{STATIC_STACK_COMPONENTS}(contributions),
        sindy_mass=sindy_mass,
        label=stack.label,
    )
end

"Apply a fitted static stack row by row without consulting any post-issue column."
function score_static_stack(df::DataFrame,
                               stack::StaticStack;
        component_columns::NamedTuple=DEFAULT_STATIC_STACK_COMPONENT_COLUMNS,
        observation_column::Symbol=:observation_dst_nt,
        model_step_column::Symbol=:model_step_hours,
        latest_dst_column::Symbol=:latest_dst_nt,
        dst_rate_column::Symbol=:dst_delta_1h_nt,
        causal_coupling_column::Symbol=:coupling_active_mvm)
    component_cols = _static_stack_validate_component_columns(component_columns)
    _static_stack_require_columns(
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
    weights = zeros(n, length(STATIC_STACK_COMPONENTS))
    contributions = similar(weights)

    for row in 1:n
        centers = NamedTuple{STATIC_STACK_COMPONENTS}(
            ntuple(j -> _static_stack_finite_cell(out, row, component_cols[j]),
                   length(component_cols)),
        )
        result = static_stack_predict(
            stack,
            _static_stack_model_step(out, row, model_step_column),
            _static_stack_finite_cell(out, row, latest_dst_column),
            _static_stack_finite_cell(out, row, dst_rate_column),
            _static_stack_finite_cell(out, row, causal_coupling_column),
            centers,
        )
        predicted[row] = result.pred_dst
        regimes[row] = String(result.regime)
        cell_regimes[row] = String(result.cell_regime)
        fallback[row] = result.used_pooled_fallback
        sindy_mass[row] = result.sindy_mass
        sindy_center[row] = result.sindy_family_center_dst
        correction[row] = result.correction_dst
        for (j, component) in enumerate(STATIC_STACK_COMPONENTS)
            weights[row, j] = getfield(result.weights, component)
            contributions[row, j] = getfield(result.component_contributions, component)
        end
    end

    out[!, :static_stack_pred_dst_nt] = predicted
    out[!, :static_stack_regime] = regimes
    out[!, :static_stack_cell_regime] = cell_regimes
    out[!, :static_stack_used_pooled_fallback] = fallback
    out[!, :static_stack_sindy_mass] = sindy_mass
    out[!, :static_stack_sindy_family_center_dst_nt] = sindy_center
    out[!, :static_stack_correction_dst_nt] = correction
    out[!, :static_stack_label] = fill(stack.label, n)
    for (j, component) in enumerate(STATIC_STACK_COMPONENTS)
        out[!, Symbol("static_stack_weight_", component)] = weights[:, j]
        out[!, Symbol("static_stack_contribution_", component, "_nt")] = contributions[:, j]
    end
    if String(observation_column) in names(out)
        residuals = Vector{Union{Missing,Float64}}(undef, n)
        for row in 1:n
            raw = out[row, observation_column]
            if ismissing(raw)
                residuals[row] = missing
            else
                raw isa Real && !(raw isa Bool) || throw(ArgumentError(
                    "static stack observation must be real or missing",
                ))
                observed = Float64(raw)
                isfinite(observed) || throw(ArgumentError(
                    "static stack observation must be finite when present",
                ))
                residuals[row] = observed - predicted[row]
            end
        end
        out[!, :static_stack_residual_dst_nt] = residuals
    end
    return out
end

const _STATIC_STACK_CSV_COLUMNS = (
    :schema_version, :label, :sindy_mass_floor, :minimum_cell_rows,
    :disturbed_dst_threshold_nt, :deepening_rate_threshold_nt_per_h,
    :coupling_threshold_mvm, :supported_model_steps, :model_step_hours,
    :regime, :n_rows, :weight_served_inertial_sindy, :weight_frozen_inertial_sindy,
    :weight_persistence, :weight_burton, :weight_burton_full, :weight_obrien,
    :objective_mse, :iterations,
)

"Atomically write a strictly versioned Operational static stack CSV."
function write_static_stack(path::AbstractString,
                                     stack::StaticStack)
    target = String(path)
    mkpath(dirname(abspath(target)))
    supported = join(stack.supported_model_steps, ";")
    rows = NamedTuple[]
    for cell in stack.cells
        push!(rows, (
            schema_version=STATIC_STACK_SCHEMA_VERSION,
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
            weight_served_inertial_sindy=cell.weights[1],
            weight_frozen_inertial_sindy=cell.weights[2],
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

function _static_stack_consistent_column(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "static stack metadata column $column contains missing",
    ))
    first_value = first(values)
    all(isequal(first_value), values) || throw(ArgumentError(
        "static stack metadata column $column is inconsistent",
    ))
    return first_value
end

function _static_stack_csv_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$field must be an integer",
    ))
    converted = Float64(value)
    isfinite(converted) && isinteger(converted) && converted <= typemax(Int) &&
        converted >= typemin(Int) || throw(ArgumentError("$field must be an integer"))
    return Int(converted)
end

"Read and strictly validate an Operational static stack CSV."
function read_static_stack(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "static stack must be a regular non-symlink file: $source",
    ))
    # The text-valued columns are read as text. Without this a numeric-looking label ("007",
    # "2026") parses as a number, and `String(cell)` then throws a bare `MethodError` — or, for a
    # zero-padded label, silently round-trips to a different string. `validate = false` keeps a
    # missing column reporting through the schema check below rather than through the parser.
    df = CSV.read(IOBuffer(read(source)), DataFrame;
                  types = Dict(:schema_version => String, :label => String, :regime => String,
                               :supported_model_steps => String),
                  validate = false)
    canonicalize_static_stack_frozen_table!(df)
    names(df) == collect(String.(_STATIC_STACK_CSV_COLUMNS)) || throw(ArgumentError(
        "static stack CSV schema does not exactly match " *
        STATIC_STACK_SCHEMA_VERSION,
    ))
    nrow(df) >= 1 || throw(ArgumentError("static stack CSV is empty"))
    schema = String(_static_stack_consistent_column(df, :schema_version))
    schema == STATIC_STACK_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported static stack schema version: $schema",
    ))
    label = String(_static_stack_consistent_column(df, :label))
    floor64 = Float64(_static_stack_consistent_column(df, :sindy_mass_floor))
    minimum_rows = _static_stack_csv_int(
        _static_stack_consistent_column(df, :minimum_cell_rows),
        "minimum_cell_rows",
    )
    disturbed_threshold = Float64(_static_stack_consistent_column(
        df, :disturbed_dst_threshold_nt,
    ))
    rate_threshold = Float64(_static_stack_consistent_column(
        df, :deepening_rate_threshold_nt_per_h,
    ))
    coupling_threshold = Float64(_static_stack_consistent_column(
        df, :coupling_threshold_mvm,
    ))
    supported_text = string(_static_stack_consistent_column(
        df, :supported_model_steps,
    ))
    isempty(supported_text) && throw(ArgumentError(
        "static stack supported_model_steps metadata is empty",
    ))
    supported = try
        parse.(Int, split(supported_text, ";"))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("static stack supported_model_steps metadata is invalid"))
    end

    weight_columns = (
        :weight_served_inertial_sindy, :weight_frozen_inertial_sindy, :weight_persistence,
        :weight_burton, :weight_burton_full, :weight_obrien,
    )
    cells = StaticStackCell[]
    for row in 1:nrow(df)
        any(ismissing(df[row, column]) for column in _STATIC_STACK_CSV_COLUMNS) &&
            throw(ArgumentError("static stack CSV contains missing at row $row"))
        weights = Float64[df[row, column] for column in weight_columns]
        push!(cells, StaticStackCell(
            _static_stack_csv_int(df[row, :model_step_hours], "model_step_hours"),
            Symbol(String(df[row, :regime])),
            _static_stack_csv_int(df[row, :n_rows], "n_rows"),
            weights;
            objective_mse=Float64(df[row, :objective_mse]),
            iterations=_static_stack_csv_int(df[row, :iterations], "iterations"),
        ))
    end
    return StaticStack(
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
