# Versioned operational SINDy artifacts.

"Canonical identifier for the current 20-candidate/11-active-term operational core."
const OPERATIONAL_V2_1_MODEL_VERSION = "v2.1"

"Canonical identifier for the archived 21-candidate/10-active-term operational core."
const OPERATIONAL_V2_0_MODEL_VERSION = "v2.0"

"Model-step horizons admitted when the Dst anchor lags the issue hour by at most one step."
const OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS = Int[1, 2, 3, 4, 6, 7]

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
    canonical_operational_version(version="v2.1")

Normalize a public operational-model identifier. Bare `v2` is a compatibility
alias for the current V2.1 model. The historical V2.0 core requires an explicit
`v2.0`/`v2_0` identifier.
"""
function canonical_operational_version(version::Union{AbstractString,Symbol}=OPERATIONAL_V2_1_MODEL_VERSION)
    token = lowercase(strip(String(version)))
    token in ("v2.1", "v2_1", "v21", "2.1", "current", "v2") &&
        return OPERATIONAL_V2_1_MODEL_VERSION
    token in ("v2.0", "v2_0", "v20", "2.0") &&
        return OPERATIONAL_V2_0_MODEL_VERSION
    throw(ArgumentError(
        "operational model version must be v2.1 (current; v2 alias accepted) " *
        "or explicit v2.0 (historical), got $(repr(version))",
    ))
end

function _operational_version_dir(version::String, root::AbstractString)
    version == OPERATIONAL_V2_1_MODEL_VERSION && return abspath(root)
    return abspath(joinpath(root, "historical", "v2_0"))
end

"""
    operational_core_artifacts(version="v2.1"; data_dir=get_data_dir())

Resolve the coefficient, ensemble-summary, and joint-draw artifacts for an
operational core. V2.1 uses the canonical unqualified paths. V2.0 resolves only
to the labeled historical directory.
"""
function operational_core_artifacts(
    version::Union{AbstractString,Symbol}=OPERATIONAL_V2_1_MODEL_VERSION;
    data_dir::AbstractString=get_data_dir(),
)
    canonical = canonical_operational_version(version)
    root = _operational_version_dir(canonical, data_dir)
    expected = canonical == OPERATIONAL_V2_1_MODEL_VERSION ? (20, 11) : (21, 10)
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
    operational_calibration_artifacts(version="v2.1"; deploy_dir)

Resolve point and conformal calibration paths. The default deploy directory is
the package-level `deploy/` directory adjacent to `data/`.
"""
function operational_calibration_artifacts(
    version::Union{AbstractString,Symbol}=OPERATIONAL_V2_1_MODEL_VERSION;
    deploy_dir::AbstractString=normpath(joinpath(@__DIR__, "..", "deploy")),
)
    canonical = canonical_operational_version(version)
    root = _operational_version_dir(canonical, deploy_dir)
    return OperationalCalibrationArtifacts(
        canonical,
        joinpath(root, "operational_v2_calibration.csv"),
        joinpath(root, "operational_v2_calibration_conformal.csv"),
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
        include_redundant_n_v2=(version == OPERATIONAL_V2_0_MODEL_VERSION),
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
    expected_counts = canonical == OPERATIONAL_V2_1_MODEL_VERSION ? (20, 11) : (21, 10)
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

    coefficient_df = CSV.read(artifacts.coefficients_csv, DataFrame)
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

    ensemble_df = CSV.read(artifacts.ensemble_csv, DataFrame)
    "term" in names(ensemble_df) || throw(ArgumentError(
        "ensemble-summary artifact must contain a term column",
    ))
    string.(ensemble_df.term) == expected_terms || throw(ArgumentError(
        "$(canonical) ensemble-summary terms do not exactly match the expected library order",
    ))

    draws_df = CSV.read(artifacts.draws_csv, DataFrame)
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

    if canonical == OPERATIONAL_V2_1_MODEL_VERSION
        "n*V^2" in expected_terms && throw(ErrorException(
            "current V2.1 library contains the redundant n*V^2 term",
        ))
        for required in ("Pdyn", "Pdyn*Bs")
            idx = findfirst(==(required), expected_terms)
            idx === nothing && throw(ErrorException("V2.1 library omits required term $required"))
            coefficients[idx] != 0.0 || throw(ArgumentError(
                "V2.1 required pressure term $required is inactive",
            ))
        end
    else
        idx = findfirst(==("n*V^2"), expected_terms)
        idx === nothing && throw(ErrorException("historical V2.0 library omits n*V^2"))
        coefficients[idx] != 0.0 || throw(ArgumentError(
            "historical V2.0 n*V^2 term must remain active",
        ))
    end

    return OperationalCore(artifacts, lib, coefficients)
end

"""
    load_operational_core(version="v2.1"; data_dir=get_data_dir())

Resolve and validate one versioned operational SINDy core. Bare `v2` selects
V2.1; loading the archived V2.0 core requires an explicit historical version.
"""
load_operational_core(
    version::Union{AbstractString,Symbol}=OPERATIONAL_V2_1_MODEL_VERSION;
    data_dir::AbstractString=get_data_dir(),
) = validate_operational_core_artifacts(
    operational_core_artifacts(version; data_dir=data_dir),
)

"""
    init_operational_forecast(; version="v2.1", t0, dst0, dt=1.0, data_dir=get_data_dir())

Initialize the ensemble forecast state through the versioned artifact boundary.
The artifact validation runs before any state is returned.
"""
function init_operational_forecast(;
    version::Union{AbstractString,Symbol}=OPERATIONAL_V2_1_MODEL_VERSION,
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
