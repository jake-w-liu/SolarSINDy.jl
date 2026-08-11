# Causal one-memory sparse augmentation of the frozen Operational V2.1 core.

import SHA

"Ordered sparse augmentation terms for the V2.2-M1 history model."
const OPERATIONAL_V22_HISTORY_TERMS = (
    :memory_coupling_mvm,
    :coupling_innovation_mvm,
    :dst_star_memory_nt_mvm,
)

"Dst-anchor lags admitted by the hourly V2.2-M1 trajectory kernel."
const OPERATIONAL_V22_HISTORY_SUPPORTED_ANCHOR_LAGS = (0, 1)

const OPERATIONAL_V22_HISTORY_SCHEMA_VERSION = "operational_v2_2_m1_v1"
const OPERATIONAL_V22_HISTORY_PACKAGE_VERSION = "SolarSINDy-0.2.1"
const OPERATIONAL_V22_HISTORY_DEFAULT_COUPLING_BOUND_MVM = 50.0
const _OPERATIONAL_V22_HISTORY_STATE_MIN_NT = -2000.0
const _OPERATIONAL_V22_HISTORY_STATE_MAX_NT = 50.0

function _operational_v22_history_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M1 $field must be a real number",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "V2.2-M1 $field must be finite",
    ))
    return converted
end

"One hourly solar-wind driver row with explicit V2.1-core units."
struct OperationalV22HistoryDriver
    speed_km_s::Float64
    bz_nt::Float64
    by_nt::Float64
    density_cm3::Float64
    pdyn_npa::Float64

    function OperationalV22HistoryDriver(
            speed_km_s::Float64,
            bz_nt::Float64,
            by_nt::Float64,
            density_cm3::Float64,
            pdyn_npa::Float64)
        all(isfinite, (speed_km_s, bz_nt, by_nt, density_cm3, pdyn_npa)) ||
            throw(ArgumentError("V2.2-M1 driver values must be finite"))
        speed_km_s >= 0.0 || throw(ArgumentError(
            "V2.2-M1 speed_km_s must be nonnegative",
        ))
        density_cm3 >= 0.0 || throw(ArgumentError(
            "V2.2-M1 density_cm3 must be nonnegative",
        ))
        pdyn_npa >= 0.0 || throw(ArgumentError(
            "V2.2-M1 pdyn_npa must be nonnegative",
        ))
        return new(speed_km_s, bz_nt, by_nt, density_cm3, pdyn_npa)
    end
end

function OperationalV22HistoryDriver(
        speed_km_s::Real,
        bz_nt::Real,
        by_nt::Real,
        density_cm3::Real,
        pdyn_npa::Real)
    return OperationalV22HistoryDriver(
        _operational_v22_history_float(speed_km_s, "speed_km_s"),
        _operational_v22_history_float(bz_nt, "bz_nt"),
        _operational_v22_history_float(by_nt, "by_nt"),
        _operational_v22_history_float(density_cm3, "density_cm3"),
        _operational_v22_history_float(pdyn_npa, "pdyn_npa"),
    )
end

"Hourly V2.2-M1 Markov state immediately before applying the current driver."
struct OperationalV22HistoryState
    t_current::DateTime
    dst_star_nt::Float64
    memory_mvm::Float64

    function OperationalV22HistoryState(
            t_current::DateTime,
            dst_star_nt::Float64,
            memory_mvm::Float64)
        all(isfinite, (dst_star_nt, memory_mvm)) || throw(ArgumentError(
            "V2.2-M1 state values must be finite",
        ))
        _OPERATIONAL_V22_HISTORY_STATE_MIN_NT <= dst_star_nt <=
            _OPERATIONAL_V22_HISTORY_STATE_MAX_NT || throw(ArgumentError(
                "V2.2-M1 Dst* state must lie in " *
                "[$(_OPERATIONAL_V22_HISTORY_STATE_MIN_NT), " *
                "$(_OPERATIONAL_V22_HISTORY_STATE_MAX_NT)] nT",
            ))
        memory_mvm >= 0.0 || throw(ArgumentError(
            "V2.2-M1 memory state must be nonnegative",
        ))
        return new(t_current, dst_star_nt, memory_mvm)
    end
end

function OperationalV22HistoryState(
        t_current::DateTime,
        dst_star_nt::Real,
        memory_mvm::Real)
    return OperationalV22HistoryState(
        t_current,
        _operational_v22_history_float(dst_star_nt, "dst_star_nt"),
        _operational_v22_history_float(memory_mvm, "memory_mvm"),
    )
end

"Immutable, checksummed V2.2-M1 sparse-history artifact."
struct OperationalV22HistoryArtifact
    label::String
    core_version::String
    core_sha256::String
    term_names::NTuple{3,Symbol}
    support_mask::NTuple{3,Bool}
    tau_memory_hours::Float64
    coefficients::NTuple{3,Float64}
    coupling_bound_mvm::Float64
    frozen_dst_slope_per_h::Float64
    fit_rows::Int
    supported_anchor_lags::NTuple{2,Int}

    function OperationalV22HistoryArtifact(
            label::String,
            core_version::String,
            core_sha256::String,
            term_names::NTuple{3,Symbol},
            support_mask::NTuple{3,Bool},
            tau_memory_hours::Float64,
            coefficients::NTuple{3,Float64},
            coupling_bound_mvm::Float64,
            frozen_dst_slope_per_h::Float64,
            fit_rows::Int,
            supported_anchor_lags::NTuple{2,Int},
            ::Val{:validated})
        isempty(strip(label)) && throw(ArgumentError(
            "V2.2-M1 artifact label must not be empty",
        ))
        core_version == OPERATIONAL_V2_1_MODEL_VERSION || throw(ArgumentError(
            "V2.2-M1 must augment the frozen V2.1 core",
        ))
        occursin(r"^[0-9a-f]{64}$", core_sha256) || throw(ArgumentError(
            "V2.2-M1 core checksum must be lowercase SHA-256",
        ))
        term_names == OPERATIONAL_V22_HISTORY_TERMS || throw(ArgumentError(
            "V2.2-M1 term order does not match the frozen three-term schema",
        ))
        isfinite(tau_memory_hours) && tau_memory_hours > 0.0 ||
            throw(ArgumentError("V2.2-M1 memory time constant must be positive and finite"))
        all(isfinite, coefficients) || throw(ArgumentError(
            "V2.2-M1 coefficients must be finite",
        ))
        all(index -> support_mask[index] || coefficients[index] == 0.0, 1:3) ||
            throw(ArgumentError(
                "V2.2-M1 excluded support terms must have exact zero coefficients",
            ))
        coefficients[1] <= 0.0 || throw(ArgumentError(
            "V2.2-M1 memory-coupling coefficient must be nonpositive",
        ))
        coefficients[3] <= 0.0 || throw(ArgumentError(
            "V2.2-M1 state-memory coefficient must be nonpositive",
        ))
        isfinite(coupling_bound_mvm) && coupling_bound_mvm > 0.0 ||
            throw(ArgumentError("V2.2-M1 coupling bound must be positive and finite"))
        isfinite(frozen_dst_slope_per_h) || throw(ArgumentError(
            "V2.2-M1 frozen Dst* slope must be finite",
        ))
        fit_rows >= 3 || throw(ArgumentError(
            "V2.2-M1 artifact requires at least three fit rows",
        ))
        supported_anchor_lags == OPERATIONAL_V22_HISTORY_SUPPORTED_ANCHOR_LAGS ||
            throw(ArgumentError("V2.2-M1 anchor-lag contract must be exactly (0, 1)"))

        multiplier_at_zero = 1.0 + frozen_dst_slope_per_h
        multiplier_at_bound = multiplier_at_zero +
                              coefficients[3] * coupling_bound_mvm
        0.0 <= multiplier_at_zero <= 1.0 || throw(ArgumentError(
            "V2.2-M1 frozen state multiplier must lie in [0, 1]",
        ))
        0.0 <= multiplier_at_bound <= 1.0 || throw(ArgumentError(
            "V2.2-M1 state-memory coefficient violates the nonoscillatory " *
            "stability bound",
        ))

        return new(
            label, core_version, core_sha256, term_names, support_mask,
            tau_memory_hours, coefficients, coupling_bound_mvm,
            frozen_dst_slope_per_h, fit_rows, supported_anchor_lags,
        )
    end
end

function _operational_v22_history_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    type_text = string(typeof(value))
    print(io, ncodeunits(type_text), ':', type_text, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

function _operational_v22_history_validate_core(core::OperationalCore)
    core.artifacts.version == OPERATIONAL_V2_1_MODEL_VERSION || throw(ArgumentError(
        "V2.2-M1 requires the canonical V2.1 core",
    ))
    terms = get_term_names(core.library)
    canonical_library = _operational_library(OPERATIONAL_V2_1_MODEL_VERSION)
    canonical_terms = get_term_names(canonical_library)
    terms == canonical_terms || throw(ArgumentError(
        "V2.2-M1 core library does not match the canonical V2.1 term order",
    ))
    core.library._contract_term_codes == canonical_library._contract_term_codes ||
        throw(ArgumentError(
            "V2.2-M1 core library does not match the canonical V2.1 term semantics",
        ))
    length(terms) == core.artifacts.candidate_count == length(core.coefficients) ||
        throw(DimensionMismatch("V2.2-M1 core terms and coefficients differ in length"))
    all(isfinite, core.coefficients) || throw(ArgumentError(
        "V2.2-M1 core coefficients must be finite",
    ))
    count(!=(0.0), core.coefficients) == core.artifacts.active_count ||
        throw(ArgumentError("V2.2-M1 core active support does not match its contract"))
    count(==("Dst_star"), terms) == 1 || throw(ArgumentError(
        "V2.2-M1 core must contain exactly one Dst_star term",
    ))
    return terms
end

function _operational_v22_history_core_sha256(core::OperationalCore)
    terms = _operational_v22_history_validate_core(core)
    io = IOBuffer()
    for value in (
            core.artifacts.version,
            core.artifacts.candidate_count,
            core.artifacts.active_count,
            length(terms),
        )
        _operational_v22_history_hash_token(io, value)
    end
    for (term, coefficient) in zip(terms, core.coefficients)
        _operational_v22_history_hash_token(io, term)
        _operational_v22_history_hash_token(io, coefficient)
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _operational_v22_history_dst_slope(core::OperationalCore)
    terms = _operational_v22_history_validate_core(core)
    return core.coefficients[only(findall(==("Dst_star"), terms))]
end

function _operational_v22_history_artifact(
        label::AbstractString,
        core_version::AbstractString,
        core_sha256::AbstractString,
        tau_memory_hours::Real,
        coefficients,
        support_mask::NTuple{3,Bool},
        coupling_bound_mvm::Real,
        frozen_dst_slope_per_h::Real,
        fit_rows::Integer)
    length(coefficients) == 3 || throw(DimensionMismatch(
        "V2.2-M1 requires exactly three augmentation coefficients",
    ))
    converted_coefficients = ntuple(
        index -> _operational_v22_history_float(
            coefficients[index], "coefficient $(OPERATIONAL_V22_HISTORY_TERMS[index])",
        ),
        3,
    )
    return OperationalV22HistoryArtifact(
        String(label), String(core_version), String(core_sha256),
        OPERATIONAL_V22_HISTORY_TERMS, support_mask,
        _operational_v22_history_float(tau_memory_hours, "tau_memory_hours"),
        converted_coefficients,
        _operational_v22_history_float(coupling_bound_mvm, "coupling_bound_mvm"),
        _operational_v22_history_float(
            frozen_dst_slope_per_h, "frozen_dst_slope_per_h",
        ),
        Int(fit_rows), OPERATIONAL_V22_HISTORY_SUPPORTED_ANCHOR_LAGS,
        Val(:validated),
    )
end

"Construct a V2.2-M1 artifact bound byte-for-byte to a frozen V2.1 core."
function OperationalV22HistoryArtifact(
        core::OperationalCore,
        coefficients;
        tau_memory_hours::Real,
        support_mask::NTuple{3,Bool}=(true, true, true),
        coupling_bound_mvm::Real=OPERATIONAL_V22_HISTORY_DEFAULT_COUPLING_BOUND_MVM,
        fit_rows::Integer,
        label::AbstractString="operational_v2_2_m1")
    return _operational_v22_history_artifact(
        label,
        core.artifacts.version,
        _operational_v22_history_core_sha256(core),
        tau_memory_hours,
        coefficients,
        support_mask,
        coupling_bound_mvm,
        _operational_v22_history_dst_slope(core),
        fit_rows,
    )
end

"Return the exact hourly exponential-memory factor of a V2.2-M1 artifact."
operational_v22_history_rho(artifact::OperationalV22HistoryArtifact) =
    exp(-1.0 / artifact.tau_memory_hours)

"Compute the causal southward-electric-field proxy `1e-3 V max(-Bz, 0)` in mV/m."
function operational_v22_history_coupling(driver::OperationalV22HistoryDriver)
    coupling = 1.0e-3 * driver.speed_km_s * max(-driver.bz_nt, 0.0)
    isfinite(coupling) || throw(ArgumentError(
        "V2.2-M1 coupling exceeds the supported Float64 range",
    ))
    return coupling
end

function operational_v22_history_coupling(speed_km_s::Real, bz_nt::Real)
    speed = _operational_v22_history_float(speed_km_s, "speed_km_s")
    bz = _operational_v22_history_float(bz_nt, "bz_nt")
    speed >= 0.0 || throw(ArgumentError("V2.2-M1 speed_km_s must be nonnegative"))
    coupling = 1.0e-3 * speed * max(-bz, 0.0)
    isfinite(coupling) || throw(ArgumentError(
        "V2.2-M1 coupling exceeds the supported Float64 range",
    ))
    return coupling
end

"Advance the causal exponential memory by one hour."
function operational_v22_history_memory(
        memory_mvm::Real,
        coupling_mvm::Real,
        tau_memory_hours::Real)
    memory = _operational_v22_history_float(memory_mvm, "memory_mvm")
    coupling = _operational_v22_history_float(coupling_mvm, "coupling_mvm")
    tau = _operational_v22_history_float(tau_memory_hours, "tau_memory_hours")
    memory >= 0.0 && coupling >= 0.0 || throw(ArgumentError(
        "V2.2-M1 memory and coupling must be nonnegative",
    ))
    tau > 0.0 || throw(ArgumentError(
        "V2.2-M1 memory time constant must be positive",
    ))
    rho = exp(-1.0 / tau)
    updated = rho * memory + (1.0 - rho) * coupling
    isfinite(updated) || throw(ArgumentError("V2.2-M1 memory update is non-finite"))
    return updated
end

"Evaluate the frozen three-term V2.2-M1 feature vector `(m, E-m, x*m)`."
function operational_v22_history_features(
        dst_star_nt::Real,
        memory_mvm::Real,
        coupling_mvm::Real)
    dst = _operational_v22_history_float(dst_star_nt, "dst_star_nt")
    memory = _operational_v22_history_float(memory_mvm, "memory_mvm")
    coupling = _operational_v22_history_float(coupling_mvm, "coupling_mvm")
    memory >= 0.0 && coupling >= 0.0 || throw(ArgumentError(
        "V2.2-M1 memory and coupling must be nonnegative",
    ))
    interaction = dst * memory
    isfinite(interaction) || throw(ArgumentError(
        "V2.2-M1 state-memory feature is non-finite",
    ))
    return (memory, coupling - memory, interaction)
end

function _operational_v22_history_validate_identity(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact)
    core.artifacts.version == artifact.core_version || throw(ArgumentError(
        "V2.2-M1 artifact core version does not match the supplied core",
    ))
    _operational_v22_history_core_sha256(core) == artifact.core_sha256 ||
        throw(ArgumentError("V2.2-M1 artifact/core checksum mismatch"))
    _operational_v22_history_dst_slope(core) == artifact.frozen_dst_slope_per_h ||
        throw(ArgumentError("V2.2-M1 frozen Dst* slope mismatch"))
    return nothing
end

function _operational_v22_history_validate_domain(
        artifact::OperationalV22HistoryArtifact,
        state::OperationalV22HistoryState,
        coupling_mvm::Float64)
    state.memory_mvm <= artifact.coupling_bound_mvm || throw(ArgumentError(
        "V2.2-M1 memory state exceeds the artifact coupling bound",
    ))
    coupling_mvm <= artifact.coupling_bound_mvm || throw(ArgumentError(
        "V2.2-M1 current coupling exceeds the artifact coupling bound",
    ))
    multiplier = 1.0 + artifact.frozen_dst_slope_per_h +
                 artifact.coefficients[3] * state.memory_mvm
    0.0 <= multiplier <= 1.0 || throw(ArgumentError(
        "V2.2-M1 local state multiplier violates the stability contract",
    ))
    return multiplier
end

function _operational_v22_history_derivative_unchecked(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact,
        state::OperationalV22HistoryState,
        driver::OperationalV22HistoryDriver,
        theta::Vector{Float64})
    coupling = operational_v22_history_coupling(driver)
    multiplier = _operational_v22_history_validate_domain(artifact, state, coupling)
    features = operational_v22_history_features(
        state.dst_star_nt, state.memory_mvm, coupling,
    )
    augmentation = artifact.coefficients[1] * features[1] +
                   artifact.coefficients[2] * features[2] +
                   artifact.coefficients[3] * features[3]
    isfinite(augmentation) || throw(ArgumentError(
        "V2.2-M1 sparse augmentation is non-finite",
    ))

    _evaluate_point_vector_unchecked!(
        theta, core.library, state.dst_star_nt, driver.speed_km_s,
        driver.bz_nt, driver.by_nt, driver.density_cm3, driver.pdyn_npa,
    )
    raw_base = dot(theta, core.coefficients)
    base = _clamped_finite_derivative(raw_base, "operational V2.1 base")
    raw_total = raw_base + augmentation
    total = _clamped_finite_derivative(raw_total, "operational V2.2-M1")
    return (
        coupling_mvm=coupling,
        memory_mvm=state.memory_mvm,
        features=features,
        raw_base_derivative_nt_per_h=raw_base,
        base_derivative_nt_per_h=base,
        augmentation_derivative_nt_per_h=augmentation,
        raw_total_derivative_nt_per_h=raw_total,
        derivative_nt_per_h=total,
        derivative_was_capped=(total != raw_total),
        state_multiplier=multiplier,
    )
end

"Pure one-step derivative diagnostics for a frozen-core V2.2-M1 state."
function operational_v22_history_derivative(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact,
        state::OperationalV22HistoryState,
        driver::OperationalV22HistoryDriver)
    _operational_v22_history_validate_identity(core, artifact)
    theta = Vector{Float64}(undef, length(core.library))
    return _operational_v22_history_derivative_unchecked(
        core, artifact, state, driver, theta,
    )
end

function _operational_v22_history_step_unchecked(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact,
        state::OperationalV22HistoryState,
        driver::OperationalV22HistoryDriver,
        theta::Vector{Float64})
    diagnostic = _operational_v22_history_derivative_unchecked(
        core, artifact, state, driver, theta,
    )
    raw_dst = state.dst_star_nt + diagnostic.derivative_nt_per_h
    dst = clamp(
        raw_dst,
        _OPERATIONAL_V22_HISTORY_STATE_MIN_NT,
        _OPERATIONAL_V22_HISTORY_STATE_MAX_NT,
    )
    memory = operational_v22_history_memory(
        state.memory_mvm, diagnostic.coupling_mvm, artifact.tau_memory_hours,
    )
    next_state = OperationalV22HistoryState(
        state.t_current + Hour(1), dst, memory,
    )
    return merge(diagnostic, (
        raw_dst_star_nt=raw_dst,
        dst_star_nt=dst,
        state_was_projected=(dst != raw_dst),
        next_state=next_state,
    ))
end

"Pure one-hour state step using the same kernel as varying-driver rollouts."
function operational_v22_history_step(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact,
        state::OperationalV22HistoryState,
        driver::OperationalV22HistoryDriver)
    _operational_v22_history_validate_identity(core, artifact)
    theta = Vector{Float64}(undef, length(core.library))
    return _operational_v22_history_step_unchecked(
        core, artifact, state, driver, theta,
    )
end

"Reconstruct the pre-driver memory state from a contiguous hourly causal history."
function init_operational_v22_history_state(
        artifact::OperationalV22HistoryArtifact,
        timestamps::AbstractVector{<:DateTime},
        drivers::AbstractVector{<:OperationalV22HistoryDriver},
        dst_star_nt::Real)
    length(timestamps) == length(drivers) || throw(DimensionMismatch(
        "V2.2-M1 history timestamps and drivers must have equal lengths",
    ))
    length(timestamps) >= 2 || throw(ArgumentError(
        "V2.2-M1 history reconstruction requires at least two hourly rows",
    ))
    for index in 1:(length(timestamps) - 1)
        timestamps[index + 1] == timestamps[index] + Hour(1) ||
            throw(ArgumentError(
                "V2.2-M1 history must be strictly contiguous on the hourly grid",
            ))
    end

    couplings = operational_v22_history_coupling.(drivers)
    all(<=(artifact.coupling_bound_mvm), couplings) || throw(ArgumentError(
        "V2.2-M1 history contains coupling beyond the artifact bound",
    ))
    memory = first(couplings)
    for index in 1:(length(couplings) - 1)
        memory = operational_v22_history_memory(
            memory, couplings[index], artifact.tau_memory_hours,
        )
    end
    return OperationalV22HistoryState(last(timestamps), dst_star_nt, memory)
end

"Run one prefix-consistent hourly trajectory with varying driver rows."
function operational_v22_history_rollout(
        core::OperationalCore,
        artifact::OperationalV22HistoryArtifact,
        initial_state::OperationalV22HistoryState,
        drivers::AbstractVector{<:OperationalV22HistoryDriver};
        anchor_lag_hours::Integer=0)
    lag = Int(anchor_lag_hours)
    anchor_lag_hours isa Bool && throw(ArgumentError(
        "V2.2-M1 anchor_lag_hours must be an integer count, not Boolean",
    ))
    lag in artifact.supported_anchor_lags || throw(ArgumentError(
        "V2.2-M1 anchor_lag_hours must be 0 or 1",
    ))
    length(drivers) > lag || throw(ArgumentError(
        "V2.2-M1 rollout requires at least one forecast step after anchor catch-up",
    ))
    _operational_v22_history_validate_identity(core, artifact)

    n_forecasts = length(drivers) - lag
    times = Vector{DateTime}(undef, n_forecasts)
    dst = Vector{Float64}(undef, n_forecasts)
    memory = Vector{Float64}(undef, n_forecasts)
    coupling = Vector{Float64}(undef, n_forecasts)
    base = Vector{Float64}(undef, n_forecasts)
    augmentation = Vector{Float64}(undef, n_forecasts)
    derivative = Vector{Float64}(undef, n_forecasts)
    projected = Vector{Bool}(undef, n_forecasts)

    theta = Vector{Float64}(undef, length(core.library))
    state = initial_state
    output_index = 0
    for (step_index, driver) in enumerate(drivers)
        result = _operational_v22_history_step_unchecked(
            core, artifact, state, driver, theta,
        )
        state = result.next_state
        if step_index > lag
            output_index += 1
            times[output_index] = state.t_current
            dst[output_index] = state.dst_star_nt
            memory[output_index] = state.memory_mvm
            coupling[output_index] = result.coupling_mvm
            base[output_index] = result.base_derivative_nt_per_h
            augmentation[output_index] = result.augmentation_derivative_nt_per_h
            derivative[output_index] = result.derivative_nt_per_h
            projected[output_index] = result.state_was_projected
        end
    end
    return (
        anchor_lag_hours=lag,
        forecast_times=times,
        dst_star_nt=dst,
        memory_mvm=memory,
        coupling_mvm=coupling,
        base_derivative_nt_per_h=base,
        augmentation_derivative_nt_per_h=augmentation,
        derivative_nt_per_h=derivative,
        state_was_projected=projected,
        final_state=state,
    )
end

function _operational_v22_history_box_fit(
        features::Matrix{Float64},
        target::Vector{Float64},
        eta3_lower::Float64,
        support_mask::NTuple{3,Bool})
    best = nothing
    best_sse = Inf
    eta1_options = support_mask[1] ? (nothing, 0.0) : (0.0,)
    eta3_options = support_mask[3] ? (nothing, eta3_lower, 0.0) : (0.0,)
    for eta1_fixed in eta1_options, eta3_fixed in eta3_options
        fixed = zeros(3)
        free = collect(support_mask)
        if eta1_fixed !== nothing
            fixed[1] = eta1_fixed
            free[1] = false
        end
        if eta3_fixed !== nothing
            fixed[3] = eta3_fixed
            free[3] = false
        end
        indices = findall(free)
        residual_target = target - features * fixed
        candidate = copy(fixed)
        isempty(indices) ||
            (candidate[indices] = features[:, indices] \ residual_target)
        candidate[1] <= 0.0 && eta3_lower <= candidate[3] <= 0.0 || continue
        all(index -> support_mask[index] || candidate[index] == 0.0, 1:3) || continue
        residual = features * candidate - target
        sse = sum(abs2, residual)
        if sse < best_sse
            best_sse = sse
            best = candidate
        end
    end
    best === nothing && throw(ErrorException(
        "V2.2-M1 constrained coefficient fit found no feasible solution",
    ))
    return best
end

"Fit the three sparse derivative-residual terms under sign and stability constraints."
function fit_operational_v22_history(
        core::OperationalCore,
        dst_star_nt::AbstractVector{<:Real},
        memory_mvm::AbstractVector{<:Real},
        coupling_mvm::AbstractVector{<:Real},
        residual_derivative_nt_per_h::AbstractVector{<:Real};
        tau_memory_hours::Real,
        support_mask::NTuple{3,Bool}=(true, true, true),
        coupling_bound_mvm::Real=OPERATIONAL_V22_HISTORY_DEFAULT_COUPLING_BOUND_MVM,
        label::AbstractString="operational_v2_2_m1")
    n_rows = length(dst_star_nt)
    n_rows >= 3 || throw(ArgumentError("V2.2-M1 fitting requires at least three rows"))
    all(==(n_rows), (
        length(memory_mvm), length(coupling_mvm),
        length(residual_derivative_nt_per_h),
    )) || throw(DimensionMismatch("V2.2-M1 fit vectors must have equal lengths"))

    dst = [_operational_v22_history_float(value, "fit dst_star_nt")
           for value in dst_star_nt]
    memory = [_operational_v22_history_float(value, "fit memory_mvm")
              for value in memory_mvm]
    coupling = [_operational_v22_history_float(value, "fit coupling_mvm")
                for value in coupling_mvm]
    target = [_operational_v22_history_float(
                  value, "fit residual_derivative_nt_per_h",
              ) for value in residual_derivative_nt_per_h]
    all(>=(0.0), memory) && all(>=(0.0), coupling) || throw(ArgumentError(
        "V2.2-M1 fit memory and coupling must be nonnegative",
    ))
    bound = _operational_v22_history_float(
        coupling_bound_mvm, "coupling_bound_mvm",
    )
    bound > 0.0 || throw(ArgumentError("V2.2-M1 coupling bound must be positive"))
    maximum(memory) <= bound && maximum(coupling) <= bound || throw(ArgumentError(
        "V2.2-M1 fit rows exceed the declared coupling bound",
    ))

    features = Matrix{Float64}(undef, n_rows, 3)
    for row in 1:n_rows
        features[row, :] .= operational_v22_history_features(
            dst[row], memory[row], coupling[row],
        )
    end
    selected = findall(support_mask)
    isempty(selected) || rank(features[:, selected]) == length(selected) ||
        throw(ArgumentError(
            "V2.2-M1 selected fit features must have full column rank",
        ))
    slope = _operational_v22_history_dst_slope(core)
    eta3_lower = (-1.0 - slope) / bound
    coefficients = _operational_v22_history_box_fit(
        features, target, eta3_lower, support_mask,
    )
    return OperationalV22HistoryArtifact(
        core, coefficients;
        tau_memory_hours=tau_memory_hours,
        support_mask=support_mask,
        coupling_bound_mvm=bound,
        fit_rows=n_rows,
        label=label,
    )
end

"Return the portable SHA-256 identity of a V2.2-M1 artifact."
function operational_v22_history_sha256(artifact::OperationalV22HistoryArtifact)
    io = IOBuffer()
    for value in (
            OPERATIONAL_V22_HISTORY_SCHEMA_VERSION,
            OPERATIONAL_V22_HISTORY_PACKAGE_VERSION,
            artifact.label,
            artifact.core_version,
            artifact.core_sha256,
            artifact.tau_memory_hours,
            artifact.coupling_bound_mvm,
            artifact.frozen_dst_slope_per_h,
            artifact.fit_rows,
        )
        _operational_v22_history_hash_token(io, value)
    end
    for values in (
            artifact.term_names,
            artifact.support_mask,
            artifact.coefficients,
            artifact.supported_anchor_lags,
        )
        _operational_v22_history_hash_token(io, length(values))
        for value in values
            _operational_v22_history_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _OPERATIONAL_V22_HISTORY_CSV_COLUMNS = (
    :schema_version,
    :package_version,
    :artifact_sha256,
    :label,
    :core_version,
    :core_sha256,
    :tau_memory_hours,
    :rho_hourly,
    :coupling_bound_mvm,
    :frozen_dst_slope_per_h,
    :fit_rows,
    :supported_anchor_lags,
    :term_index,
    :term,
    :selected,
    :coefficient,
)

"Atomically write a strictly versioned, checksummed V2.2-M1 artifact."
function write_operational_v22_history(
        path::AbstractString,
        artifact::OperationalV22HistoryArtifact)
    target = String(path)
    mkpath(dirname(abspath(target)))
    checksum = operational_v22_history_sha256(artifact)
    rows = [(
        schema_version=OPERATIONAL_V22_HISTORY_SCHEMA_VERSION,
        package_version=OPERATIONAL_V22_HISTORY_PACKAGE_VERSION,
        artifact_sha256=checksum,
        label=artifact.label,
        core_version=artifact.core_version,
        core_sha256=artifact.core_sha256,
        tau_memory_hours=artifact.tau_memory_hours,
        rho_hourly=operational_v22_history_rho(artifact),
        coupling_bound_mvm=artifact.coupling_bound_mvm,
        frozen_dst_slope_per_h=artifact.frozen_dst_slope_per_h,
        fit_rows=artifact.fit_rows,
        supported_anchor_lags=join(artifact.supported_anchor_lags, ";"),
        term_index=index,
        term=String(artifact.term_names[index]),
        selected=artifact.support_mask[index],
        coefficient=artifact.coefficients[index],
    ) for index in eachindex(artifact.coefficients)]
    _write_selection_csv(target, rows)
    return target
end

function _operational_v22_history_consistent_column(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "V2.2-M1 artifact metadata $column contains missing values",
    ))
    first_value = first(values)
    all(isequal(first_value), values) || throw(ArgumentError(
        "V2.2-M1 artifact metadata $column is inconsistent",
    ))
    return first_value
end

function _operational_v22_history_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M1 artifact $field must be an integer",
    ))
    converted = Float64(value)
    isfinite(converted) && isinteger(converted) &&
        typemin(Int) <= converted <= typemax(Int) || throw(ArgumentError(
            "V2.2-M1 artifact $field must be an integer",
        ))
    return Int(converted)
end

function _operational_v22_history_bool(value, field::AbstractString)
    value isa Bool || throw(ArgumentError(
        "V2.2-M1 artifact $field must be Boolean",
    ))
    return value
end

"Read a checksummed V2.2-M1 artifact and verify its frozen-core identity."
function read_operational_v22_history(
        path::AbstractString,
        core::OperationalCore)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2-M1 artifact must be a regular non-symlink file: $source",
    ))
    df = CSV.read(source, DataFrame)
    names(df) == collect(String.(_OPERATIONAL_V22_HISTORY_CSV_COLUMNS)) ||
        throw(ArgumentError("V2.2-M1 artifact CSV schema is invalid"))
    nrow(df) == length(OPERATIONAL_V22_HISTORY_TERMS) || throw(ArgumentError(
        "V2.2-M1 artifact must contain exactly three term rows",
    ))
    for row in 1:nrow(df), column in _OPERATIONAL_V22_HISTORY_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "V2.2-M1 artifact contains missing at row $row column $column",
        ))
    end

    schema = string(_operational_v22_history_consistent_column(df, :schema_version))
    schema == OPERATIONAL_V22_HISTORY_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported V2.2-M1 artifact schema: $schema",
    ))
    package_version = string(_operational_v22_history_consistent_column(
        df, :package_version,
    ))
    package_version == OPERATIONAL_V22_HISTORY_PACKAGE_VERSION ||
        throw(ArgumentError("unsupported V2.2-M1 package version: $package_version"))
    checksum = string(_operational_v22_history_consistent_column(
        df, :artifact_sha256,
    ))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "V2.2-M1 artifact checksum is malformed",
    ))
    [_operational_v22_history_int(df[row, :term_index], "term_index")
     for row in 1:nrow(df)] == collect(1:length(OPERATIONAL_V22_HISTORY_TERMS)) ||
        throw(ArgumentError("V2.2-M1 artifact term indices are not sequential"))
    Tuple(Symbol.(string.(df.term))) == OPERATIONAL_V22_HISTORY_TERMS ||
        throw(ArgumentError("V2.2-M1 artifact term order is invalid"))
    support_mask = Tuple(
        _operational_v22_history_bool(df[row, :selected], "selected")
        for row in 1:nrow(df)
    )
    supported_text = string(_operational_v22_history_consistent_column(
        df, :supported_anchor_lags,
    ))
    supported_text == join(OPERATIONAL_V22_HISTORY_SUPPORTED_ANCHOR_LAGS, ";") ||
        throw(ArgumentError("V2.2-M1 artifact anchor-lag schema is invalid"))

    tau = _operational_v22_history_float(
        _operational_v22_history_consistent_column(df, :tau_memory_hours),
        "artifact tau_memory_hours",
    )
    stored_rho = _operational_v22_history_float(
        _operational_v22_history_consistent_column(df, :rho_hourly),
        "artifact rho_hourly",
    )
    stored_rho == exp(-1.0 / tau) || throw(ArgumentError(
        "V2.2-M1 artifact rho is inconsistent with its time constant",
    ))
    artifact = _operational_v22_history_artifact(
        string(_operational_v22_history_consistent_column(df, :label)),
        string(_operational_v22_history_consistent_column(df, :core_version)),
        string(_operational_v22_history_consistent_column(df, :core_sha256)),
        tau,
        [_operational_v22_history_float(
             df[row, :coefficient], "artifact coefficient",
         ) for row in 1:nrow(df)],
        support_mask,
        _operational_v22_history_consistent_column(df, :coupling_bound_mvm),
        _operational_v22_history_consistent_column(df, :frozen_dst_slope_per_h),
        _operational_v22_history_int(
            _operational_v22_history_consistent_column(df, :fit_rows),
            "fit_rows",
        ),
    )
    operational_v22_history_sha256(artifact) == checksum || throw(ArgumentError(
        "V2.2-M1 artifact checksum mismatch",
    ))
    _operational_v22_history_validate_identity(core, artifact)
    return artifact
end
