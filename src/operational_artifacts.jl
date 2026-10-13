# Versioned operational SINDy artifacts.

"Canonical identifier for the current 20-candidate/11-active-term operational core."
const OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION = "inertial_sindy"

"Canonical identifier for the archived 21-candidate/10-active-term operational core."
const OPERATIONAL_SINDY_CORE_MODEL_VERSION = "sindy_core"

"Model-step horizons admitted when the Dst anchor lags the issue hour by at most one step."
const OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS = Int[1, 2, 3, 4, 6, 7]

"""
    OperationalCoreArtifacts

Paths and exact structural contract for one operational SINDy core. The fields
identify its canonical version, coefficient/ensemble/joint-draw CSVs, candidate
count, and required active-support count.
"""
struct OperationalCoreArtifacts
    version::String
    coefficients_csv::String
    ensemble_csv::String
    draws_csv::String
    candidate_count::Int
    active_count::Int
end

"""
    OperationalCalibrationArtifacts

Paths for the point and conformal calibration paired with one canonical core
version.
"""
struct OperationalCalibrationArtifacts
    version::String
    point_csv::String
    conformal_csv::String
end

"""
    OperationalCore

Validated operational point core containing its artifact contract, ordered
candidate library, and finite coefficient vector.
"""
struct OperationalCore
    artifacts::OperationalCoreArtifacts
    library::CandidateLibrary
    coefficients::Vector{Float64}
end

"""
    canonical_operational_version(version="inertial_sindy")

Normalize a public operational-model identifier. Bare `operational` is a compatibility
alias for the current inertial-SINDy model. The historical SINDy core requires an explicit
`sindy-core`/`sindy_core` identifier.
"""
function canonical_operational_version(version::Union{AbstractString,Symbol}=OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)
    token = lowercase(strip(String(version)))
    token in ("inertial_sindy", "inertial-sindy", "inertialsindy", "current", "operational") &&
        return OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION
    token in ("sindy_core", "sindy-core", "sindycore") &&
        return OPERATIONAL_SINDY_CORE_MODEL_VERSION
    throw(ArgumentError(
        "operational model version must be inertial-sindy (current; operational alias accepted) " *
        "or explicit sindy-core (historical), got $(repr(version))",
    ))
end

# Recorded base model_version values written by earlier builds, mapped to the canonical name of the
# same operator. `v1`/`v2`/`v2.0` rows are the historical SINDy core; `v2.1`/`v21`/`inertial-sindy`
# rows are the inertial-SINDy operator.
const _LEGACY_MODEL_VERSION_TOKENS = Dict(
    "v1" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "v2" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "v2.0" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "v2_0" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "v20" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "v2.1" => OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    "v2_1" => OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    "v21" => OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    "inertial-sindy" => OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    "inertial_sindy" => OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    "sindy-core" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
    "sindy_core" => OPERATIONAL_SINDY_CORE_MODEL_VERSION,
)

"""
    canonical_operational_model_version(version) -> String

Canonical name of a recorded `model_version` value. Logs written by earlier builds record the
operators under their release labels (`v1`, `v2`, `v2.1`); those rows still identify the same
operators, so readers map them onto the canonical names rather than rejecting the history.
An unrecognised value passes through unchanged, which keeps unknown labels fail-closed.
"""
canonical_operational_model_version(version::AbstractString) =
    get(_LEGACY_MODEL_VERSION_TOKENS, lowercase(strip(version)), String(version))
canonical_operational_model_version(version) = version

# Column spellings recorded by release-era builds inside frozen operational tables (the audited
# calibration-scored CSV and accumulated monitor logs). Frozen artifacts keep their release-era
# bytes for provenance; readers verify the recorded digest first and then canonicalize the loaded
# table in memory so downstream code only sees canonical names. File bytes are never rewritten.
# Cell values are left as recorded: identity cells are consumed only from already-migrated logs.
const _LEGACY_OPERATIONAL_FROZEN_COLUMNS = Dict{Symbol,Symbol}(
    :v1_pred_dst_nt => :sindy_core_pred_dst_nt,
    :v1_pred_dst_ci05_nt => :sindy_core_pred_dst_ci05_nt,
    :v1_pred_dst_ci95_nt => :sindy_core_pred_dst_ci95_nt,
    :v2_pred_dst_nt => :inertial_sindy_pred_dst_nt,
    :v2_pred_dst_ci05_nt => :inertial_sindy_pred_dst_ci05_nt,
    :v2_pred_dst_ci95_nt => :inertial_sindy_pred_dst_ci95_nt,
    :v2_correction_dst_nt => :inertial_sindy_correction_dst_nt,
    :v2_interval_scale => :inertial_sindy_interval_scale,
    :v2_calibration_label => :inertial_sindy_calibration_label,
    :v2_selected_component => :inertial_sindy_selected_component,
    :v2_selected_component_pred_nt => :inertial_sindy_selected_component_pred_nt,
    :v2_split => :operational_split,
    :v2_residual_dst_nt => :operational_residual_dst_nt,
    # An earlier in-place migration emitted this spelling; readers repair it on load.
    :inertial_sindy_residual_dst_nt => :operational_residual_dst_nt,
    :v2_observed_in_90ci => :operational_observed_in_90ci,
    :v2_1_served_pred_dst_nt => :inertial_sindy_served_pred_dst_nt,
    :v2_1_served_ci05_dst_nt => :inertial_sindy_served_ci05_dst_nt,
    :v2_2_regime => :static_stack_regime,
    :v2_2_coupling_active_mvm => :static_stack_coupling_active_mvm,
    :v2_2_pooled_fallback => :static_stack_pooled_fallback,
    :v2_2_status => :static_stack_status,
    :v2_2_stack_pred_dst_nt => :static_stack_pred_dst_nt,
    :v2_2_stack_ci05_dst_nt => :static_stack_ci05_dst_nt,
    :v23_shadow_model_version => :analog_shadow_model_version,
    :v23_manifest_sha256 => :analog_manifest_sha256,
    :v23_status => :analog_status,
    :v23_analog_k => :analog_k,
    :v23_history_hours => :analog_history_hours,
    :v23_raw_dst_nt => :analog_raw_dst_nt,
    :v23_center_dst_nt => :analog_center_dst_nt,
    :v23_step1_center_dst_nt => :analog_step1_center_dst_nt,
    :v23_shadow_pred_dst_nt => :analog_shadow_pred_dst_nt,
    :v23_e_layer_applied => :analog_e_layer_applied,
    :v24_model_version => :superlearner_model_version,
    :v24_status => :superlearner_status,
    :v24_manifest_sha256 => :superlearner_manifest_sha256,
    :v24_l1_center_dst_nt => :superlearner_l1_center_dst_nt,
    :v24_guard_applied => :superlearner_guard_applied,
    :v24_projection_applied => :superlearner_projection_applied,
    :v24_pred_dst_nt => :superlearner_pred_dst_nt,
    :v24_ci05_nt => :superlearner_ci05_nt,
    :v24_ci95_nt => :superlearner_ci95_nt,
    :v24_regime_cell => :superlearner_regime_cell,
    :v24_deepening_cell => :superlearner_deepening_cell,
    :v24_pooled_fallback => :superlearner_pooled_fallback,
    :v24_history_hours => :superlearner_history_hours,
    :v24_t1r_pred_dst_nt => :superlearner_t1r_pred_dst_nt,
    :v24_cal_shadow_model_version => :superlearner_cal_shadow_model_version,
    :v24_cal_shadow_status => :superlearner_cal_shadow_status,
    :v24_cal_shadow_config_sha256 => :superlearner_cal_shadow_config_sha256,
    :v24_cal_shadow_ci05_nt => :superlearner_cal_shadow_ci05_nt,
    :v24_cal_shadow_ci95_nt => :superlearner_cal_shadow_ci95_nt,
    :v2_4e => :superlearner_e,
    :v2_4e_lo_nt => :superlearner_e_lo_nt,
    :v2_4e_hi_nt => :superlearner_e_hi_nt,
    :static_v2_2 => :static_stack,
    :served_v2_1 => :served_inertial_sindy,
    :frozen_v2_1 => :frozen_inertial_sindy,
    :v2_3_shadow => :analog_shadow,
)

# Served-identity and shadow-candidate spellings recorded inside sha-pinned frozen tables. The
# rewrite is column-scoped — generic release-era cell values elsewhere (a `v2` component selector,
# for instance) are context-dependent and are deliberately left as recorded.
const _LEGACY_OPERATIONAL_FROZEN_SERVED_IDENTITY = "v2.4+sindy20x11+superlearner10floor+conformal"
const _CANONICAL_SERVED_IDENTITY = "dst-superlearner+sindyfloor+conformal"
const _LEGACY_OPERATIONAL_FROZEN_CAL_SHADOW_PREFIX = "v2.4e-cal-shadow-"
const _CANONICAL_CAL_SHADOW_PREFIX = "superlearner_e-cal-shadow-"

"""
    canonicalize_operational_frozen_table!(df::DataFrame) -> DataFrame

Rename release-era column spellings inside a verified frozen operational table to their canonical
names, in place, and return `df`. A table carrying both spellings of the same column is ambiguous
and rejected. Idempotent on canonical tables. Call only after the source bytes have passed their
digest check.

Served-identity cells in `sub_hourly_model_version` and calibration-shadow candidate ids in
`superlearner_cal_shadow_model_version` are unambiguous referents of the one deployment, so they
are rewritten to their canonical spellings the same way a serialized static stack rewrites its
recorded labels. Other cell values stay as recorded — a release-era `v2` component selector
denotes the inertial-SINDy component and must not be remapped blindly.
"""
function canonicalize_operational_frozen_table!(df::DataFrame)
    for (legacy, canonical) in _LEGACY_OPERATIONAL_FROZEN_COLUMNS
        hasproperty(df, legacy) || continue
        hasproperty(df, canonical) && throw(ArgumentError(
            "frozen operational table carries both $(legacy) and $(canonical)",
        ))
        rename!(df, legacy => canonical)
    end
    if :sub_hourly_model_version in propertynames(df)
        df[!, :sub_hourly_model_version] = map(df[!, :sub_hourly_model_version]) do cell
            ismissing(cell) && return missing
            String(cell) == _LEGACY_OPERATIONAL_FROZEN_SERVED_IDENTITY ?
                _CANONICAL_SERVED_IDENTITY : String(cell)
        end
    end
    if :superlearner_cal_shadow_model_version in propertynames(df)
        df[!, :superlearner_cal_shadow_model_version] =
            map(df[!, :superlearner_cal_shadow_model_version]) do cell
                ismissing(cell) && return missing
                text = String(cell)
                startswith(text, _LEGACY_OPERATIONAL_FROZEN_CAL_SHADOW_PREFIX) ?
                    _CANONICAL_CAL_SHADOW_PREFIX *
                        text[length(_LEGACY_OPERATIONAL_FROZEN_CAL_SHADOW_PREFIX)+1:end] : text
            end
    end
    return df
end

function _operational_version_dir(version::String, root::AbstractString)
    version == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION && return abspath(root)
    return abspath(joinpath(root, "historical", "sindy_core"))
end

"""
    operational_core_artifacts(version="inertial_sindy"; data_dir=get_data_dir())

Resolve the coefficient, ensemble-summary, and joint-draw artifacts for an
operational core. inertial-SINDy uses the canonical unqualified paths. SINDy core resolves only
to the labeled historical directory.
"""
function operational_core_artifacts(
    version::Union{AbstractString,Symbol}=OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION;
    data_dir::AbstractString=get_data_dir(),
)
    canonical = canonical_operational_version(version)
    root = _operational_version_dir(canonical, data_dir)
    expected = canonical == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION ? (20, 11) : (21, 10)
    return OperationalCoreArtifacts(
        canonical,
        joinpath(root, "real_sindy_discovery_coefficients.csv"),
        joinpath(root, "real_ensemble_inclusion.csv"),
        joinpath(root, "real_sindy_ensemble_draws.csv"),
        expected[1],
        expected[2],
    )
end

"""
    operational_calibration_artifacts(version="inertial_sindy"; deploy_dir)

Resolve point and conformal calibration paths. The default deploy directory is
the package-level `deploy/` directory adjacent to `data/`.
"""
function operational_calibration_artifacts(
    version::Union{AbstractString,Symbol}=OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION;
    deploy_dir::AbstractString=normpath(joinpath(@__DIR__, "..", "deploy")),
)
    canonical = canonical_operational_version(version)
    root = _operational_version_dir(canonical, deploy_dir)
    return OperationalCalibrationArtifacts(
        canonical,
        joinpath(root, "inertial_sindy_calibration.csv"),
        joinpath(root, "inertial_sindy_calibration_conformal.csv"),
    )
end

function _require_regular_artifact(path::AbstractString, label::AbstractString)
    isfile(path) && !islink(path) || throw(ArgumentError(
        "$label must be a regular non-symlink file: $path",
    ))
    return abspath(path)
end

function _operational_library(version::String)
    return build_solar_wind_library(
        include_redundant_n_operational=(version == OPERATIONAL_SINDY_CORE_MODEL_VERSION),
    )
end

"""
    validate_operational_core_artifacts(artifacts)

Fail closed unless all three artifacts match the exact library order, candidate
count, active-support count, and current/historical pressure-term identity.
Returns the validated `OperationalCore`.
"""
function validate_operational_core_artifacts(artifacts::OperationalCoreArtifacts)
    canonical = canonical_operational_version(artifacts.version)
    canonical == artifacts.version || throw(ArgumentError(
        "OperationalCoreArtifacts.version must be canonical, got $(artifacts.version)",
    ))
    expected_counts = canonical == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION ? (20, 11) : (21, 10)
    (artifacts.candidate_count, artifacts.active_count) == expected_counts ||
        throw(ArgumentError(
            "$(canonical) structural contract must be $(expected_counts[1]) candidates/" *
            "$(expected_counts[2]) active terms",
        ))

    _require_regular_artifact(artifacts.coefficients_csv, "coefficient artifact")
    _require_regular_artifact(artifacts.ensemble_csv, "ensemble-summary artifact")
    _require_regular_artifact(artifacts.draws_csv, "joint-draw artifact")

    lib = _operational_library(canonical)
    expected_terms = get_term_names(lib)
    length(expected_terms) == artifacts.candidate_count || throw(ErrorException(
        "internal $(canonical) library has $(length(expected_terms)) terms; " *
        "expected $(artifacts.candidate_count)",
    ))

    coefficient_df = CSV.read(artifacts.coefficients_csv, DataFrame; buffer_in_memory = true)
    all(in(names(coefficient_df)), ("term", "coefficient")) || throw(ArgumentError(
        "coefficient artifact must contain term and coefficient columns",
    ))
    coefficient_terms = string.(coefficient_df.term)
    coefficient_terms == expected_terms || throw(ArgumentError(
        "$(canonical) coefficient terms do not exactly match the expected library order",
    ))
    coefficients = Float64.(coefficient_df.coefficient)
    all(isfinite, coefficients) || throw(ArgumentError(
        "$(canonical) coefficients must be finite",
    ))
    count(!=(0.0), coefficients) == artifacts.active_count || throw(ArgumentError(
        "$(canonical) coefficient support has $(count(!=(0.0), coefficients)) active terms; " *
        "expected $(artifacts.active_count)",
    ))

    ensemble_df = CSV.read(artifacts.ensemble_csv, DataFrame; buffer_in_memory = true)
    "term" in names(ensemble_df) || throw(ArgumentError(
        "ensemble-summary artifact must contain a term column",
    ))
    string.(ensemble_df.term) == expected_terms || throw(ArgumentError(
        "$(canonical) ensemble-summary terms do not exactly match the expected library order",
    ))

    draws_df = CSV.read(artifacts.draws_csv, DataFrame; buffer_in_memory = true)
    names(draws_df) == expected_terms || throw(ArgumentError(
        "$(canonical) joint-draw columns do not exactly match the expected library order",
    ))
    nrow(draws_df) == 500 || throw(ArgumentError(
        "$(canonical) joint-draw artifact must contain exactly 500 draws, got $(nrow(draws_df))",
    ))
    for col in eachcol(draws_df)
        all(x -> !ismissing(x) && isfinite(Float64(x)), col) || throw(ArgumentError(
            "$(canonical) joint-draw artifact contains a missing or non-finite value",
        ))
    end

    if canonical == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION
        "n*V^2" in expected_terms && throw(ErrorException(
            "current inertial-SINDy library contains the redundant n*V^2 term",
        ))
        for required in ("Pdyn", "Pdyn*Bs")
            idx = findfirst(==(required), expected_terms)
            idx === nothing && throw(ErrorException("inertial-SINDy library omits required term $required"))
            coefficients[idx] != 0.0 || throw(ArgumentError(
                "inertial-SINDy required pressure term $required is inactive",
            ))
        end
    else
        idx = findfirst(==("n*V^2"), expected_terms)
        idx === nothing && throw(ErrorException("historical SINDy core library omits n*V^2"))
        coefficients[idx] != 0.0 || throw(ArgumentError(
            "historical SINDy core n*V^2 term must remain active",
        ))
    end

    return OperationalCore(artifacts, lib, coefficients)
end

"""
    load_operational_core(version="inertial_sindy"; data_dir=get_data_dir())

Resolve and validate one versioned operational SINDy core. Bare `operational` selects
inertial-SINDy; loading the archived SINDy core requires an explicit historical version.
"""
load_operational_core(
    version::Union{AbstractString,Symbol}=OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION;
    data_dir::AbstractString=get_data_dir(),
) = validate_operational_core_artifacts(
    operational_core_artifacts(version; data_dir=data_dir),
)

"""
    init_operational_forecast(; version="inertial_sindy", t0, dst0, dt=1.0, data_dir=get_data_dir())

Initialize the ensemble forecast state through the versioned artifact boundary.
The artifact validation runs before any state is returned.
"""
function init_operational_forecast(;
    version::Union{AbstractString,Symbol}=OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    t0::DateTime,
    dst0::Float64,
    dt::Float64=1.0,
    data_dir::AbstractString=get_data_dir(),
)
    core = load_operational_core(version; data_dir=data_dir)
    art = core.artifacts
    return init_forecast(
        coefficients_csv=art.coefficients_csv,
        ensemble_csv=art.ensemble_csv,
        draws_csv=art.draws_csv,
        t0=t0,
        dst0=dst0,
        dt=dt,
    )
end

"""
    operational_core_forecast(core, dst0, drivers, n_steps; dt=1.0)

Efficient deterministic point rollout of a validated operational core under a
fixed driver tuple `(V, Bz, By, n, Pdyn)`. Returns the Dst* state after each
step. This is the point-forecast oracle used by archive-scale calibration; it
must remain numerically identical to the primary path of `forecast_ahead`.
"""
function operational_core_forecast(
    core::OperationalCore,
    dst0::Real,
    drivers,
    n_steps::Integer;
    dt::Real=1.0,
)
    n_steps >= 0 || throw(ArgumentError("n_steps must be nonnegative"))
    dt64 = Float64(dt)
    isfinite(dt64) && dt64 > 0 || throw(ArgumentError("dt must be finite and positive"))
    values = Float64[
        Float64(dst0), Float64(drivers.V), Float64(drivers.Bz),
        Float64(drivers.By), Float64(drivers.n), Float64(drivers.Pdyn),
    ]
    all(isfinite, values) || throw(ArgumentError("state and drivers must be finite"))
    values[2] >= 0 && values[5] >= 0 && values[6] >= 0 || throw(ArgumentError(
        "V, density, and dynamic pressure must be nonnegative",
    ))

    dst = values[1]
    out = Vector{Float64}(undef, n_steps)
    theta = Vector{Float64}(undef, length(core.library))
    for step in 1:n_steps
        _evaluate_point_vector_unchecked!(
            theta, core.library, dst, values[2], values[3], values[4],
            values[5], values[6],
        )
        derivative = _clamped_finite_derivative(
            dot(theta, core.coefficients), "operational $(core.artifacts.version)",
        )
        dst = clamp(dst + dt64 * derivative, -2000.0, 50.0)
        out[step] = dst
    end
    return out
end
