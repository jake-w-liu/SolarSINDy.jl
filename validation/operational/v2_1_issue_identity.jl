#!/usr/bin/env julia

using CSV
using DataFrames
using SHA
using SolarSINDy

include(joinpath(@__DIR__, "paths.jl"))
include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const V21_ISSUE_IDENTITY = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_issue_identity.csv")

_identity_sha(path::AbstractString) = bytes2hex(sha256(read(path)))

function write_v2_1_issue_identity(path::AbstractString=V21_ISSUE_IDENTITY)
    core = load_operational_core(:v2)
    calibration_path = operational_calibration_artifacts(
        OPERATIONAL_V2_1_MODEL_VERSION,
    ).point_csv
    calibration = read_operational_v2_calibration(calibration_path)
    terms = get_term_names(core.library)
    canonical_operational_version(:v2) == OPERATIONAL_V2_1_MODEL_VERSION || error(
        "bare v2 alias does not resolve to V2.1",
    )
    length(terms) == 20 || error("current issue core does not have 20 candidates")
    count(!=(0.0), core.coefficients) == 11 || error(
        "current issue core does not have 11 active terms",
    )
    "n*V^2" ∉ terms || error("current issue core contains redundant n*V^2")
    _one_hour_inertia_blend(-120.0, -100.0, 1) == -115.0 || error(
        "one-hour inertia operator drifted",
    )
    _rapid_deepening_projection_guard(-100.0, -80.0, 6, -20.0) == -125.0 || error(
        "rate-projection operator drifted",
    )
    _state_inertia_blend(-120.0, -100.0, 1, -10.0) == -100.0 || error(
        "state-inertia operator drifted",
    )

    # The served identity is the whole served pipeline, which now ends in the fitted static regime
    # stack. Recording the V2.1 tail label here would under-report the product that produces the
    # published center, so the identity row carries the served label together with the stack label and
    # digest it is pinned to, and the shadow identity beside it.
    stack_path = V2_2_DEFAULT_STACK_PATH
    stack = load_v22_serving_stack(stack_path)
    stack_sha256 = v22_serving_stack_sha256(stack_path)
    stack_sha256 == V22_SERVED_STACK_SHA256 || error(
        "deployed served stack digest $(stack_sha256) is not the published " *
        "$(V22_SERVED_STACK_SHA256)",
    )
    shadow_dir = V2_3_DEFAULT_SHADOW_DIR
    shadow_manifest_sha256 = isdir(shadow_dir) ?
        (v23_serving_verify_manifest(shadow_dir);
         v23_serving_file_sha256(joinpath(shadow_dir, "manifest.csv"))) : ""

    artifacts = core.artifacts
    out = DataFrame(
        model_version=[OPERATIONAL_V2_1_MODEL_VERSION],
        served_model_version=[V2_2_SERVED_TAIL_VERSION],
        served_fallback_model_version=[V2_SERVED_TAIL_VERSION],
        served_stack_label=[stack.label],
        served_stack_sha256=[stack_sha256],
        shadow_model_version=[V2_3_SHADOW_TAIL_VERSION],
        shadow_manifest_sha256=[shadow_manifest_sha256],
        candidate_count=[length(terms)],
        active_count=[count(!=(0.0), core.coefficients)],
        redundant_n_v2_present=["n*V^2" in terms],
        pressure_term_active=[core.coefficients[findfirst(==("Pdyn"), terms)] != 0.0],
        pressure_coupling_active=[core.coefficients[findfirst(==("Pdyn*Bs"), terms)] != 0.0],
        one_hour_inertia_weight=[V2_ONE_HOUR_INERTIA_WEIGHT],
        state_inertia_h1_quiet_weight=[V2_STATE_INERTIA_H1_QUIET_WEIGHT],
        state_inertia_h1_deepening_weight=[V2_STATE_INERTIA_H1_DEEPENING_WEIGHT],
        state_inertia_h2_quiet_weight=[V2_STATE_INERTIA_H2_QUIET_WEIGHT],
        state_inertia_h3_quiet_weight=[V2_STATE_INERTIA_H3_QUIET_WEIGHT],
        state_inertia_quiet_dst_nt=[V2_STATE_INERTIA_QUIET_DST_NT],
        state_inertia_deepening_lo_nt_per_h=[V2_STATE_INERTIA_DEEPENING_LO_NT_PER_H],
        state_inertia_deepening_hi_nt_per_h=[V2_STATE_INERTIA_DEEPENING_HI_NT_PER_H],
        rapid_deepening_activation_rate_nt_per_h=[V2_RAPID_DEEPENING_RATE_NT_PER_H],
        rapid_deepening_projection_factor=[V2_RATE_PROJECTION_FACTOR],
        rapid_deepening_extreme_rate_nt_per_h=[V2_RATE_PROJECTION_EXTREME_RATE_NT_PER_H],
        rapid_deepening_max_drop_nt=[V2_RATE_PROJECTION_MAX_DROP_NT],
        rapid_deepening_extreme_max_drop_nt=[V2_RATE_PROJECTION_EXTREME_MAX_DROP_NT],
        calibration_label=[calibration.label],
        coefficient_sha256=[_identity_sha(artifacts.coefficients_csv)],
        ensemble_sha256=[_identity_sha(artifacts.ensemble_csv)],
        draws_sha256=[_identity_sha(artifacts.draws_csv)],
        calibration_sha256=[_identity_sha(calibration_path)],
        historical_v2_0_requires_explicit_version=[
            canonical_operational_version(:v2_0) == OPERATIONAL_V2_0_MODEL_VERSION,
        ],
    )
    mkpath(dirname(path))
    CSV.write(path, out)
    println("Wrote ", path)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_v2_1_issue_identity()
end
