# Hourly frozen-core rollout driven by the 30-minute V2.2-M2 state path.

const OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION = "operational_v2_2_core_path_v1"
const OPERATIONAL_V22_CORE_PATH_HOURS = 7
const OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR = 2
const OPERATIONAL_V22_CORE_PATH_SUPPORTED_MODEL_STEPS = OPERATIONAL_V22_MODEL_STEPS
const _OPERATIONAL_V22_CORE_PATH_ROWS =
    OPERATIONAL_V22_CORE_PATH_HOURS * OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR

function _operational_v22_core_path_matrix(states::AbstractMatrix)
    size(states) == (_OPERATIONAL_V22_CORE_PATH_ROWS,
                     length(OPERATIONAL_V22_DRIVER_STATES)) ||
        throw(DimensionMismatch(
            "V2.2 core path requires a 14×5 half-hour M2 state matrix",
        ))
    all(value -> value isa Real && !(value isa Bool), states) ||
        throw(ArgumentError("V2.2 core path states must be real numbers"))
    converted = Matrix{Float64}(states)
    all(isfinite, converted) || throw(ArgumentError(
        "V2.2 core path states must be finite",
    ))
    return converted
end

function _operational_v22_core_path_validate_core(core::OperationalCore)
    core.artifacts.version == OPERATIONAL_V2_1_MODEL_VERSION ||
        throw(ArgumentError("V2.2 core path requires the frozen V2.1 core"))
    core.artifacts.candidate_count == 20 && core.artifacts.active_count == 11 ||
        throw(ArgumentError("V2.2 core path requires the V2.1 20/11 contract"))
    expected_terms = get_term_names(_operational_library(OPERATIONAL_V2_1_MODEL_VERSION))
    get_term_names(core.library) == expected_terms || throw(ArgumentError(
        "V2.2 core path library does not match the frozen V2.1 order",
    ))
    length(core.coefficients) == length(expected_terms) &&
        all(isfinite, core.coefficients) &&
        count(!=(0.0), core.coefficients) == core.artifacts.active_count ||
        throw(ArgumentError(
            "V2.2 core path coefficients do not satisfy the V2.1 20/11 contract",
        ))
    return core
end

"Convert fourteen half-hour `(Bx,By,Bz,logV,logn)` states to seven hourly drivers."
function operational_v22_hourly_drivers(states::AbstractMatrix)
    path = _operational_v22_core_path_matrix(states)
    drivers = Vector{NamedTuple}(undef, OPERATIONAL_V22_CORE_PATH_HOURS)
    for hour in 1:OPERATIONAL_V22_CORE_PATH_HOURS
        first_row = (hour - 1) * OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR + 1
        last_row = hour * OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR
        rows = first_row:last_row
        bx = mean(@view path[rows, 1])
        by = mean(@view path[rows, 2])
        bz = mean(@view path[rows, 3])
        speeds = exp.(@view path[rows, 4])
        densities = exp.(@view path[rows, 5])
        all(isfinite, speeds) && all(>(0.0), speeds) || throw(ArgumentError(
            "V2.2 core path speed leaves the supported physical domain",
        ))
        all(isfinite, densities) && all(>(0.0), densities) ||
            throw(ArgumentError(
                "V2.2 core path density leaves the supported physical domain",
            ))
        speed = mean(speeds)
        density = mean(densities)
        pdyn = dynamic_pressure(density, speed)
        all(isfinite, (bx, by, bz, speed, density, pdyn)) && pdyn >= 0.0 ||
            throw(ArgumentError(
                "V2.2 hourly driver leaves the supported physical domain",
            ))
        drivers[hour] = (
            Bx=bx,
            V=speed,
            Bz=bz,
            By=by,
            n=density,
            Pdyn=pdyn,
        )
    end
    return Tuple(drivers)
end

"Low-level research rollout of the unchanged V2.1 core over a raw M2 matrix."
function operational_v22_core_path_forecast(
        core::OperationalCore,
        anchor_dst_star_nt::Real,
        half_hour_states::AbstractMatrix)
    _operational_v22_core_path_validate_core(core)
    anchor_dst_star_nt isa Bool && throw(ArgumentError(
        "V2.2 core-path anchor must be a real number",
    ))
    anchor = Float64(anchor_dst_star_nt)
    isfinite(anchor) || throw(ArgumentError(
        "V2.2 core-path anchor must be finite",
    ))
    drivers = operational_v22_hourly_drivers(half_hour_states)
    dst_star = Vector{Float64}(undef, OPERATIONAL_V22_CORE_PATH_HOURS)
    dst = Vector{Float64}(undef, OPERATIONAL_V22_CORE_PATH_HOURS)
    state = anchor
    for hour in 1:OPERATIONAL_V22_CORE_PATH_HOURS
        driver = drivers[hour]
        state = only(operational_core_forecast(core, state, driver, 1))
        value = dst_star_to_dst(state, driver.Pdyn)
        isfinite(value) || throw(ArgumentError(
            "V2.2 core-path Dst inversion became non-finite",
        ))
        dst_star[hour] = state
        dst[hour] = value
    end
    return (
        schema_version=OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION,
        execution_scope=:low_level_research_only,
        internal_step_hours=ntuple(identity, OPERATIONAL_V22_CORE_PATH_HOURS),
        supported_model_steps=OPERATIONAL_V22_CORE_PATH_SUPPORTED_MODEL_STEPS,
        hourly_drivers=drivers,
        pred_dst_star_nt=Tuple(dst_star),
        pred_dst_nt=Tuple(dst),
    )
end

"Reject an unbound candidate-path call at the operational core boundary."
function operational_v22_core_path_forecast(
        core::OperationalCore,
        anchor_dst_star_nt::Real,
        path::OperationalV22ArrivalPath)
    throw(ArgumentError(
        "V2.2 operational core composition requires the bound queue, " *
        "driver artifact, pinned frozen-core SHA, and a frozen support/gate artifact",
    ))
end

"Reverify all available bindings, then fail closed while M2 remains ungated."
function operational_v22_core_path_forecast(
        core::OperationalCore,
        anchor_dst_star_nt::Real,
        path::OperationalV22ArrivalPath,
        queue::OperationalV22ArrivalQueue,
        artifact::OperationalV22DriverArtifact,
        pinned_core_sha256::AbstractString)
    verify_operational_v22_arrival_path(path, queue, artifact)
    occursin(r"^[0-9a-f]{64}$", pinned_core_sha256) || throw(ArgumentError(
        "V2.2 pinned frozen-core SHA must be a lowercase SHA-256 digest",
    ))
    operational_v22_core_sha256(core) == String(pinned_core_sha256) ||
        throw(ArgumentError(
            "V2.2 frozen core does not match the pinned semantic SHA",
        ))
    path.gate_status == OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS ||
        throw(ArgumentError("V2.2 arrival-path gate status changed"))
    throw(ArgumentError(
        "V2.2 M2 is an ungated candidate; no frozen support/gate artifact is available",
    ))
end
