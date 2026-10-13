#!/usr/bin/env julia

using CSV
using DataFrames
using SHA
using SolarSINDy

include(joinpath(@__DIR__, "paths.jl"))
isdefined(@__MODULE__, :LiveVerifyConfig) || include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const INERTIAL_SINDY_ISSUE_IDENTITY = joinpath(OPERATIONAL_OUTPUT_DIR, "inertial_sindy_issue_identity.csv")

_identity_sha(path::AbstractString) = bytes2hex(sha256(read(path)))

function write_inertial_sindy_issue_identity(path::AbstractString=INERTIAL_SINDY_ISSUE_IDENTITY)
    core = load_operational_core(:inertial_sindy)
    calibration_path = operational_calibration_artifacts(
        OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION,
    ).point_csv
    calibration = read_inertial_sindy_calibration(calibration_path)
    terms = get_term_names(core.library)
    canonical_operational_version(:inertial_sindy) == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION || error(
        "bare operational alias does not resolve to inertial-SINDy",
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

    # The served identity is the whole served pipeline, which now ends in the super-learner over
    # the ten experts, the static regime stack among them. Recording an earlier stage's label
    # here would under-report the product that produces the published center, so the identity row
    # carries the served label together with the artifacts every stage of the fallback chain is pinned
    # to: the super-learner bundle manifest digest and the static stack label and digest.
    stack_path = STATIC_STACK_DEFAULT_STACK_PATH
    stack = load_static_stack(stack_path)
    stack_sha256 = static_stack_sha256(stack_path)
    stack_sha256 == STATIC_STACK_SERVED_SHA256 || error(
        "deployed served stack digest $(stack_sha256) is not the published " *
        "$(STATIC_STACK_SERVED_SHA256)",
    )
    # The super-learner bundle is loaded the way the engine loads it, so an identity row is written only for a
    # bundle that would actually serve: the digest recorded here is the manifest the loader verified.
    served_bundle_dir = SUPERLEARNER_DEFAULT_DEPLOY_DIR
    served_bundle = load_superlearner_serving_artifacts(served_bundle_dir)
    served_bundle.identity == SUPERLEARNER_SERVED_TAIL_VERSION || error(
        "deployed super-learner bundle identity $(served_bundle.identity) is not the published " *
        "$(SUPERLEARNER_SERVED_TAIL_VERSION)",
    )

    artifacts = core.artifacts
    out = DataFrame(
        model_version=[OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION],
        served_model_version=[SUPERLEARNER_SERVED_TAIL_VERSION],
        served_fallback_model_version=[OPERATIONAL_SERVED_TAIL_VERSION],
        served_stack_model_version=[STATIC_STACK_SERVED_TAIL_VERSION],
        served_bundle_manifest_sha256=[served_bundle.manifest_sha256],
        served_bundle_fold_year=[served_bundle.fold_year],
        served_bundle_training_max_target_utc=[string(served_bundle.training_max_target_utc)],
        served_stack_label=[stack.label],
        served_stack_sha256=[stack_sha256],
        candidate_count=[length(terms)],
        active_count=[count(!=(0.0), core.coefficients)],
        redundant_n_operational_present=["n*V^2" in terms],
        pressure_term_active=[core.coefficients[findfirst(==("Pdyn"), terms)] != 0.0],
        pressure_coupling_active=[core.coefficients[findfirst(==("Pdyn*Bs"), terms)] != 0.0],
        one_hour_inertia_weight=[OPERATIONAL_ONE_HOUR_INERTIA_WEIGHT],
        state_inertia_h1_quiet_weight=[OPERATIONAL_STATE_INERTIA_H1_QUIET_WEIGHT],
        state_inertia_h1_deepening_weight=[OPERATIONAL_STATE_INERTIA_H1_DEEPENING_WEIGHT],
        state_inertia_h2_quiet_weight=[OPERATIONAL_STATE_INERTIA_H2_QUIET_WEIGHT],
        state_inertia_h3_quiet_weight=[OPERATIONAL_STATE_INERTIA_H3_QUIET_WEIGHT],
        state_inertia_quiet_dst_nt=[OPERATIONAL_STATE_INERTIA_QUIET_DST_NT],
        state_inertia_deepening_lo_nt_per_h=[OPERATIONAL_STATE_INERTIA_DEEPENING_LO_NT_PER_H],
        state_inertia_deepening_hi_nt_per_h=[OPERATIONAL_STATE_INERTIA_DEEPENING_HI_NT_PER_H],
        rapid_deepening_activation_rate_nt_per_h=[OPERATIONAL_RAPID_DEEPENING_RATE_NT_PER_H],
        rapid_deepening_projection_factor=[OPERATIONAL_RATE_PROJECTION_FACTOR],
        rapid_deepening_extreme_rate_nt_per_h=[OPERATIONAL_RATE_PROJECTION_EXTREME_RATE_NT_PER_H],
        rapid_deepening_max_drop_nt=[OPERATIONAL_RATE_PROJECTION_MAX_DROP_NT],
        rapid_deepening_extreme_max_drop_nt=[OPERATIONAL_RATE_PROJECTION_EXTREME_MAX_DROP_NT],
        calibration_label=[calibration.label],
        coefficient_sha256=[_identity_sha(artifacts.coefficients_csv)],
        ensemble_sha256=[_identity_sha(artifacts.ensemble_csv)],
        draws_sha256=[_identity_sha(artifacts.draws_csv)],
        calibration_sha256=[_identity_sha(calibration_path)],
        historical_sindy_core_requires_explicit_version=[
            canonical_operational_version(:sindy_core) == OPERATIONAL_SINDY_CORE_MODEL_VERSION,
        ],
    )
    mkpath(dirname(path))
    CSV.write(path, out)
    println("Wrote ", path)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_inertial_sindy_issue_identity()
end
