using Test
using Dates
using CSV
using DataFrames
using SolarSINDy

const _SHADOW_TOKEN_A = repeat("a", 64)
const _SHADOW_TOKEN_B = repeat("b", 64)
const _SHADOW_TOKEN_C = repeat("c", 64)
const _SHADOW_TOKEN_D = repeat("d", 64)
const _SHADOW_TOKEN_E = repeat("e", 64)

function _shadow_test_core(; coefficient_delta=0.0)
    library = build_solar_wind_library()
    terms = get_term_names(library)
    required = [findfirst(==("Pdyn"), terms), findfirst(==("Pdyn*Bs"), terms)]
    all(!isnothing, required) || error("synthetic V2.1 library is incomplete")
    active = Int[only for only in required]
    for index in eachindex(terms)
        index in active && continue
        push!(active, index)
        length(active) == 11 && break
    end
    coefficients = zeros(Float64, length(terms))
    for index in active
        coefficients[index] = 0.001 * index + 0.01
    end
    coefficients[first(active)] += coefficient_delta
    artifacts = OperationalCoreArtifacts(
        OPERATIONAL_V2_1_MODEL_VERSION,
        "synthetic-coefficients.csv",
        "synthetic-ensemble.csv",
        "synthetic-draws.csv",
        20,
        11,
    )
    return OperationalCore(artifacts, library, coefficients)
end

function _shadow_test_driver(; label="synthetic-m2")
    return OperationalV22DriverArtifact(
        zeros(Float64, 5, 5, 6);
        support_mask=falses(5, 6),
        ridge=1.0e-6,
        threshold=0.0,
        fit_rows=64,
        threshold_iterations=1,
        label=label,
    )
end

function _shadow_test_conformal(; half_width=4.0, max_horizon=7.0)
    return ConformalCalibration(
        0.9,
        [0.0, 1.5, 2.5, 4.5, Inf],
        -30.0,
        20,
        max_horizon,
        Dict(
            :h1_quiet =>
                ConformalStratum(:h1_quiet, 100, half_width, 91 / 101),
        ),
        ConformalStratum(:global, 100, half_width, 91 / 101),
    )
end

function _shadow_test_bindings(;
        receipt=_SHADOW_TOKEN_A,
        transport=_SHADOW_TOKEN_B,
        anchor_pressure=_SHADOW_TOKEN_E,
        conformal_sidecar=_SHADOW_TOKEN_C,
        point_calibration=_SHADOW_TOKEN_D,
        feature_schema=OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA,
        product_version=OPERATIONAL_V22_SHADOW_PRODUCT_VERSION)
    return OperationalV22ShadowBindings(
        receipt_pair_contract_sha256=receipt,
        transport_support_sha256=transport,
        anchor_pressure_contract_sha256=anchor_pressure,
        conformal_sidecar_sha256=conformal_sidecar,
        point_calibration_sha256=point_calibration,
        feature_schema=feature_schema,
        product_version=product_version,
    )
end

function _shadow_test_chain(; anchor_lag_hours=0)
    driver = _shadow_test_driver()
    core = _shadow_test_core()
    bindings = _shadow_test_bindings()
    base_hash = operational_v22_base_center_sha256(
        bindings, driver, core; anchor_lag_hours=anchor_lag_hours,
    )
    error_state = OperationalV22ErrorStateArtifact(
        base_hash,
        2.0,
        zeros(9);
        support_mask=ntuple(_ -> false, 9),
        ridge=1.0e-6,
        fit_rows=64,
        selection_score=1.25,
        label="synthetic-m3",
    )
    conformal = _shadow_test_conformal()
    artifact = OperationalV22ShadowChainArtifact(
        bindings,
        driver,
        core,
        error_state,
        conformal;
        anchor_lag_hours=anchor_lag_hours,
    )
    return (; artifact, bindings, driver, core, error_state, conformal, base_hash)
end

function _shadow_test_innovations(issue_time, base_hash)
    records = OperationalV22H1Innovation[]
    for offset in 24:-1:1
        issued_at = issue_time - Hour(offset)
        target_at = issued_at + Hour(1)
        push!(records, OperationalV22H1Innovation(
            issued_at,
            target_at,
            target_at,
            base_hash,
            -10.0,
            -9.0,
        ))
    end
    return records
end

function _shadow_test_exogenous_issue(
        issue_time::DateTime,
        base_hash::String;
        scale::Real=1.0,
        model_step_hours::Integer=1)
    features = [Float64(scale) * index
                for index in eachindex(OPERATIONAL_V22_RESIDUAL_FEATURES)]
    trajectory = [Float64(scale) * (10 * row + column)
                  for row in 1:14, column in 1:5]
    return OperationalV22ErrorExogenousIssue(
        issue_time,
        issue_time,
        base_hash,
        features,
        trajectory,
        model_step_hours=model_step_hours,
    )
end

function _shadow_test_exogenous_chain()
    driver = _shadow_test_driver(label="synthetic-m2-exogenous")
    core = _shadow_test_core()
    bindings = _shadow_test_bindings(
        feature_schema=Tuple(String.(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES)),
    )
    base_hash = operational_v22_base_center_sha256(bindings, driver, core)
    error_states = [OperationalV22ErrorExogenousArtifact(
        base_hash,
        horizon,
        Float64(horizon),
        zeros(length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES));
        fit_rows=64,
        label="synthetic-m3-exogenous-h$(horizon)",
    ) for horizon in OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H]
    conformal = _shadow_test_conformal()
    artifact = OperationalV22ShadowChainArtifact(
        bindings, driver, core, error_states, conformal,
    )
    return (; artifact, bindings, driver, core, error_states, conformal, base_hash)
end

function _shadow_test_exogenous_history(issue_time, base_hash)
    issues = OperationalV22ErrorExogenousIssue[]
    innovations = OperationalV22H1Innovation[]
    for lag in reverse(OPERATIONAL_V22_ERROR_LAGS_H)
        historical_issue = issue_time - Hour(lag)
        push!(issues, _shadow_test_exogenous_issue(
            historical_issue, base_hash; scale=lag,
        ))
        push!(innovations, OperationalV22H1Innovation(
            historical_issue,
            historical_issue + Hour(1),
            historical_issue + Hour(1),
            base_hash,
            -10.0,
            -9.0,
        ))
    end
    return issues, innovations
end

@testset "V2.2 shadow-chain semantic identities" begin
    chain = _shadow_test_chain()

    @test operational_v22_driver_sha256(chain.driver) == chain.artifact.driver_sha256
    @test operational_v22_core_sha256(chain.core) == chain.artifact.core_sha256
    @test operational_v22_base_center_sha256(
        chain.bindings, chain.driver, chain.core,
    ) ==
          chain.artifact.base_center_sha256
    @test operational_v22_error_state_sha256(chain.error_state) ==
          first(chain.artifact.m3_sha256_by_horizon)
    @test all(==(first(chain.artifact.m3_sha256_by_horizon)),
              chain.artifact.m3_sha256_by_horizon)
    @test chain.artifact.m3_kind == :ar_control
    @test operational_v22_conformal_sha256(chain.conformal) ==
          chain.artifact.conformal_sha256
    @test occursin(r"^[0-9a-f]{64}$",
                   operational_v22_shadow_chain_sha256(chain.artifact))
    @test operational_v22_shadow_chain_sha256(chain.artifact) ==
          operational_v22_shadow_chain_sha256(chain.artifact)
    @test validate_operational_v22_shadow_chain(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal,
    ) === chain.artifact

    @test_throws ArgumentError _shadow_test_bindings(receipt="")
    @test_throws ArgumentError _shadow_test_bindings(anchor_pressure="")
    @test_throws ArgumentError _shadow_test_bindings(receipt=uppercase(_SHADOW_TOKEN_A))
    @test_throws ArgumentError _shadow_test_bindings(feature_schema=("x", "x"))
    @test_throws ArgumentError _shadow_test_bindings(feature_schema=("x;y",))
    @test_throws ArgumentError _shadow_test_bindings(product_version="v2.2")

    wrong_core = _shadow_test_core(coefficient_delta=0.125)
    @test operational_v22_core_sha256(wrong_core) != chain.artifact.core_sha256
    @test all(!iszero, chain.core.library._contract_term_codes)
    same_names = get_term_names(chain.core.library)
    zero_functions = Function[
        data -> zeros(length(first(values(data)))) for _ in same_names
    ]
    same_named_zero_library = CandidateLibrary(same_names, zero_functions)
    @test all(iszero, same_named_zero_library._contract_term_codes)
    same_named_zero_core = OperationalCore(
        chain.core.artifacts,
        same_named_zero_library,
        copy(chain.core.coefficients),
    )
    @test_throws ArgumentError operational_v22_core_sha256(
        same_named_zero_core,
    )
    @test_throws ArgumentError OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        same_named_zero_core,
        chain.error_state,
        chain.conformal,
    )
    wrong_driver = _shadow_test_driver(label="different-m2")
    @test operational_v22_driver_sha256(wrong_driver) != chain.artifact.driver_sha256
    wrong_conformal = _shadow_test_conformal(half_width=5.0)
    @test operational_v22_conformal_sha256(wrong_conformal) !=
          chain.artifact.conformal_sha256
    invalid_key_conformal = ConformalCalibration(
        0.9,
        [0.0, 1.5, 2.5, 4.5, Inf],
        -30.0,
        20,
        7.0,
        Dict(:unknown => ConformalStratum(:unknown, 100, 4.0, 91 / 101)),
        ConformalStratum(:global, 100, 4.0, 91 / 101),
    )
    @test_throws ArgumentError operational_v22_conformal_sha256(
        invalid_key_conformal,
    )
    inconsistent_count_conformal = ConformalCalibration(
        0.9,
        [0.0, 1.5, 2.5, 4.5, Inf],
        -30.0,
        20,
        7.0,
        Dict(:h1_quiet => ConformalStratum(:h1_quiet, 99, 4.0, 0.9)),
        ConformalStratum(:global, 100, 4.0, 91 / 101),
    )
    @test_throws ArgumentError operational_v22_conformal_sha256(
        inconsistent_count_conformal,
    )
    inconsistent_floor_conformal = ConformalCalibration(
        0.9,
        [0.0, 1.5, 2.5, 4.5, Inf],
        -30.0,
        20,
        7.0,
        Dict(:h1_quiet => ConformalStratum(:h1_quiet, 100, 4.0, 0.9)),
        ConformalStratum(:global, 100, 4.0, 91 / 101),
    )
    @test_throws ArgumentError operational_v22_conformal_sha256(
        inconsistent_floor_conformal,
    )

    bad_m3 = OperationalV22ErrorStateArtifact(
        repeat("e", 64),
        2.0,
        zeros(9);
        support_mask=ntuple(_ -> false, 9),
        fit_rows=64,
    )
    @test_throws ArgumentError OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        chain.core,
        bad_m3,
        chain.conformal,
    )
    @test_throws ArgumentError OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        _shadow_test_conformal(max_horizon=6.0),
    )
    @test_throws ArgumentError OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal;
        anchor_lag_hours=1,
    )
end

@testset "V2.2 shadow-chain hand prediction and causal fallback" begin
    chain = _shadow_test_chain()
    issue_time = DateTime(2030, 1, 3, 12)
    base = OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        chain.base_hash,
        -20.0;
        execution_scope=:synthetic_research_only,
    )
    records = _shadow_test_innovations(issue_time, chain.base_hash)
    result = operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal,
        base,
        -10.0,
        records,
    )

    # Independent hand calculation: zero AR slopes and intercept 2 nT give a
    # 2 nT terminal correction; the synthetic global conformal width is 4 nT.
    @test result.raw_correction_nt == 2.0
    @test result.correction_nt == 2.0
    @test result.correction_cap_nt == 20.0
    @test result.pred_dst_nt == -18.0
    @test result.lo_dst_nt == -22.0
    @test result.hi_dst_nt == -14.0
    @test result.correction_applied
    @test result.fallback_reason == :none
    @test result.issue_time == issue_time
    @test result.anchor_time == issue_time
    @test result.target_time == issue_time + Hour(3)
    @test result.issue_relative_horizon_hours == 3
    @test result.base_center_sha256 == chain.base_hash
    @test result.chain_sha256 ==
          operational_v22_shadow_chain_sha256(chain.artifact)

    future = OperationalV22H1Innovation(
        issue_time + Hour(1),
        issue_time + Hour(2),
        issue_time + Hour(2),
        chain.base_hash,
        1.0e6,
        -1.0e6,
    )
    result_with_future = operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal,
        base,
        -10.0,
        [records; future],
    )
    @test result_with_future.pred_dst_nt == result.pred_dst_nt
    @test result_with_future.raw_correction_nt == result.raw_correction_nt
    @test result_with_future.lo_dst_nt == result.lo_dst_nt
    @test result_with_future.hi_dst_nt == result.hi_dst_nt

    fallback = operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal,
        base,
        -10.0,
        OperationalV22H1Innovation[],
    )
    @test fallback.pred_dst_nt == base.prediction_dst_nt
    @test fallback.raw_correction_nt == 0.0
    @test fallback.correction_nt == 0.0
    @test !fallback.correction_applied
    @test fallback.fallback_reason == :missing_history
    @test fallback.lo_dst_nt == -24.0
    @test fallback.hi_dst_nt == -16.0

    half_hour_states = zeros(Float64, 14, 5)
    half_hour_states[:, 4] .= log(400.0)
    half_hour_states[:, 5] .= log(5.0)
    core_path_result = operational_v22_core_path_forecast(
        chain.core, -20.0, half_hour_states,
    )
    path_base = OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        chain.base_hash,
        core_path_result,
    )
    @test path_base.prediction_dst_nt == core_path_result.pred_dst_nt[3]
    @test isfinite(path_base.prediction_dst_nt)
    @test path_base.target_time == issue_time + Hour(3)
    @test_throws ArgumentError OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        5,
        chain.base_hash,
        core_path_result,
    )
    @test_throws ArgumentError OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        chain.base_hash,
        merge(core_path_result, (schema_version="wrong",)),
    )
    four_field_core_path_result = (
        schema_version=core_path_result.schema_version,
        internal_step_hours=core_path_result.internal_step_hours,
        supported_model_steps=core_path_result.supported_model_steps,
        pred_dst_nt=core_path_result.pred_dst_nt,
    )
    @test_throws ArgumentError OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        chain.base_hash,
        four_field_core_path_result,
    )
    @test_throws ArgumentError OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        chain.base_hash,
        merge(core_path_result, (execution_scope=:issued,)),
    )
    @test_throws UndefKeywordError OperationalV22BaseCenterForecast(
        issue_time, issue_time, 3, chain.base_hash, -20.0,
    )
    @test_throws ArgumentError OperationalV22BaseCenterForecast(
        issue_time, issue_time, 3, chain.base_hash, -20.0;
        execution_scope=:issued,
    )
end

@testset "V2.2 operational prediction rejects every ungated center" begin
    ar_chain = _shadow_test_chain()
    issue_time = DateTime(2030, 1, 4, 12)
    numeric_base = OperationalV22BaseCenterForecast(
        issue_time, issue_time, 3, ar_chain.base_hash, -20.0;
        execution_scope=:synthetic_research_only,
    )
    half_hour_states = zeros(Float64, 14, 5)
    half_hour_states[:, 4] .= log(400.0)
    half_hour_states[:, 5] .= log(5.0)
    core_path_result = operational_v22_core_path_forecast(
        ar_chain.core, -20.0, half_hour_states,
    )
    low_level_base = OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        ar_chain.base_hash,
        merge(core_path_result, (execution_scope=:low_level_research_only,)),
    )
    records = _shadow_test_innovations(issue_time, ar_chain.base_hash)

    exogenous_chain = _shadow_test_exogenous_chain()
    exogenous_numeric_base = OperationalV22BaseCenterForecast(
        issue_time, issue_time, 3, exogenous_chain.base_hash, -20.0;
        execution_scope=:synthetic_research_only,
    )
    exogenous_core_path = operational_v22_core_path_forecast(
        exogenous_chain.core, -20.0, half_hour_states,
    )
    exogenous_low_level_base = OperationalV22BaseCenterForecast(
        issue_time,
        issue_time,
        3,
        exogenous_chain.base_hash,
        merge(
            exogenous_core_path,
            (execution_scope=:low_level_research_only,),
        ),
    )
    current = _shadow_test_exogenous_issue(
        issue_time, exogenous_chain.base_hash; model_step_hours=3,
    )
    issues, exogenous_innovations = _shadow_test_exogenous_history(
        issue_time, exogenous_chain.base_hash,
    )

    function caught_error(f)
        try
            f()
        catch error
            return error
        end
        return nothing
    end

    gate_errors = (
        caught_error(() -> operational_v22_shadow_predict(
            ar_chain.artifact, ar_chain.bindings, ar_chain.driver,
            ar_chain.core, ar_chain.error_state, ar_chain.conformal,
            -20.0, -10.0, records,
        )),
        caught_error(() -> operational_v22_shadow_predict(
            ar_chain.artifact, ar_chain.bindings, ar_chain.driver,
            ar_chain.core, ar_chain.error_state, ar_chain.conformal,
            numeric_base, -10.0, records,
        )),
        caught_error(() -> operational_v22_shadow_predict(
            ar_chain.artifact, ar_chain.bindings, ar_chain.driver,
            ar_chain.core, ar_chain.error_state, ar_chain.conformal,
            low_level_base, -10.0, records,
        )),
        caught_error(() -> operational_v22_shadow_predict(
            exogenous_chain.artifact, exogenous_chain.bindings,
            exogenous_chain.driver, exogenous_chain.core,
            exogenous_chain.error_states, exogenous_chain.conformal,
            exogenous_numeric_base, -10.0, current, issues,
            exogenous_innovations,
        )),
        caught_error(() -> operational_v22_shadow_predict(
            exogenous_chain.artifact, exogenous_chain.bindings,
            exogenous_chain.driver, exogenous_chain.core,
            exogenous_chain.error_states, exogenous_chain.conformal,
            exogenous_low_level_base, -10.0, current, issues,
            exogenous_innovations,
        )),
    )
    for error in gate_errors
        @test error isa ArgumentError
        @test occursin("issued-path gate artifact/proof", sprint(showerror, error))
    end
end

@testset "V2.2 shadow-chain fails closed on every binding" begin
    chain = _shadow_test_chain()
    issue_time = DateTime(2030, 2, 1, 0)
    base = OperationalV22BaseCenterForecast(
        issue_time, issue_time, 1, chain.base_hash, -15.0;
        execution_scope=:synthetic_research_only,
    )
    records = _shadow_test_innovations(issue_time, chain.base_hash)

    function predict_with(;
            bindings=chain.bindings,
            driver=chain.driver,
            core=chain.core,
            error_state=chain.error_state,
            conformal=chain.conformal,
            base_center=base)
        return operational_v22_shadow_research_predict(
            chain.artifact,
            bindings,
            driver,
            core,
            error_state,
            conformal,
            base_center,
            -10.0,
            records,
        )
    end

    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(receipt=repeat("1", 64)),
    )
    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(transport=repeat("2", 64)),
    )
    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(anchor_pressure=repeat("5", 64)),
    )
    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(conformal_sidecar=repeat("3", 64)),
    )
    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(point_calibration=repeat("4", 64)),
    )
    @test_throws ArgumentError predict_with(
        bindings=_shadow_test_bindings(feature_schema=("different",)),
    )
    @test_throws ArgumentError predict_with(driver=_shadow_test_driver(label="mutated"))
    @test_throws ArgumentError predict_with(core=_shadow_test_core(coefficient_delta=0.5))

    changed_error = OperationalV22ErrorStateArtifact(
        chain.base_hash,
        2.0,
        zeros(9);
        support_mask=ntuple(_ -> false, 9),
        ridge=1.0e-6,
        fit_rows=64,
        selection_score=1.25,
        label="changed-m3",
    )
    @test_throws ArgumentError predict_with(error_state=changed_error)
    @test_throws ArgumentError predict_with(
        conformal=_shadow_test_conformal(half_width=5.0),
    )
    @test_throws ArgumentError predict_with(
        base_center=OperationalV22BaseCenterForecast(
            issue_time, issue_time, 1, repeat("e", 64), -15.0;
            execution_scope=:synthetic_research_only,
        ),
    )
    @test_throws ArgumentError predict_with(
        base_center=OperationalV22BaseCenterForecast(
            issue_time, issue_time - Hour(1), 1, chain.base_hash, -15.0;
            execution_scope=:synthetic_research_only,
        ),
    )
    @test_throws ArgumentError predict_with(
        base_center=OperationalV22BaseCenterForecast(
            issue_time, issue_time, 5, chain.base_hash, -15.0;
            execution_scope=:synthetic_research_only,
        ),
    )

    mismatched_history = copy(records)
    first_record = first(mismatched_history)
    mismatched_history[1] = OperationalV22H1Innovation(
        first_record.issued_at,
        first_record.target_at,
        first_record.observation_available_at,
        repeat("e", 64),
        first_record.base_prediction_dst_nt,
        first_record.observation_dst_nt,
    )
    @test_throws ArgumentError operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        chain.conformal,
        base,
        -10.0,
        mismatched_history,
    )

    mutated_core = _shadow_test_core()
    mutable_artifact = OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        mutated_core,
        chain.error_state,
        chain.conformal,
    )
    mutated_core.coefficients[1] += 0.25
    @test_throws ArgumentError validate_operational_v22_shadow_chain(
        mutable_artifact,
        chain.bindings,
        chain.driver,
        mutated_core,
        chain.error_state,
        chain.conformal,
    )

    mutated_conformal = _shadow_test_conformal()
    conformal_artifact = OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        mutated_conformal,
    )
    mutated_conformal.strata[:h1_quiet] =
        ConformalStratum(:h1_quiet, 100, 3.0, 91 / 101)
    @test_throws ArgumentError validate_operational_v22_shadow_chain(
        conformal_artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_state,
        mutated_conformal,
    )
end

@testset "V2.2 full-M3 exogenous shadow chain" begin
    chain = _shadow_test_exogenous_chain()
    issue_time = DateTime(2030, 3, 2, 12)
    current = _shadow_test_exogenous_issue(
        issue_time, chain.base_hash; model_step_hours=3,
    )
    issues, innovations = _shadow_test_exogenous_history(
        issue_time, chain.base_hash,
    )
    base = OperationalV22BaseCenterForecast(
        issue_time, issue_time, 3, chain.base_hash, -20.0;
        execution_scope=:synthetic_research_only,
    )

    @test chain.artifact.m3_kind == :exogenous
    @test length(unique(chain.artifact.m3_sha256_by_horizon)) == 6
    @test chain.artifact.bindings.feature_schema ==
          Tuple(String.(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES))
    @test validate_operational_v22_shadow_chain(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_states,
        chain.conformal,
    ) === chain.artifact

    result = operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_states,
        chain.conformal,
        base,
        -10.0,
        current,
        issues,
        innovations,
    )
    @test result.raw_correction_nt == 3.0
    @test result.correction_nt == 3.0
    @test result.correction_cap_nt == 20.0
    @test result.pred_dst_nt == -17.0
    @test result.lo_dst_nt == -21.0
    @test result.hi_dst_nt == -13.0
    @test result.fallback_reason == :none
    @test result.artifact_sha256 == chain.artifact.m3_sha256_by_horizon[3]

    fallback = operational_v22_shadow_research_predict(
        chain.artifact,
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_states,
        chain.conformal,
        base,
        -10.0,
        current,
        OperationalV22ErrorExogenousIssue[],
        innovations,
    )
    @test fallback.pred_dst_nt == base.prediction_dst_nt
    @test fallback.fallback_reason == :missing_issue_history

    future_a = _shadow_test_exogenous_issue(
        issue_time + Hour(10), chain.base_hash; scale=1.0,
    )
    future_b = _shadow_test_exogenous_issue(
        issue_time + Hour(10), chain.base_hash; scale=-1000.0,
    )
    result_a = operational_v22_shadow_research_predict(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        chain.error_states, chain.conformal, base, -10.0, current,
        [issues; future_a], innovations,
    )
    result_b = operational_v22_shadow_research_predict(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        chain.error_states, chain.conformal, base, -10.0, current,
        [issues; future_b], innovations,
    )
    @test result_a.pred_dst_nt == result.pred_dst_nt
    @test result_b.pred_dst_nt == result.pred_dst_nt

    changed = copy(chain.error_states)
    changed[3] = OperationalV22ErrorExogenousArtifact(
        chain.base_hash,
        3,
        3.0,
        zeros(length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES));
        fit_rows=64,
        label="mutated-exogenous-h3",
    )
    @test_throws ArgumentError validate_operational_v22_shadow_chain(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        changed, chain.conformal,
    )
    duplicated = copy(chain.error_states)
    duplicated[end] = first(duplicated)
    @test_throws ArgumentError validate_operational_v22_shadow_chain(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        duplicated, chain.conformal,
    )
    @test_throws DimensionMismatch OperationalV22ShadowChainArtifact(
        chain.bindings,
        chain.driver,
        chain.core,
        chain.error_states[1:5],
        chain.conformal,
    )
    @test_throws ArgumentError OperationalV22ShadowChainArtifact(
        _shadow_test_bindings(),
        chain.driver,
        chain.core,
        chain.error_states,
        chain.conformal,
    )

    wrong_history = copy(issues)
    first_issue = first(wrong_history)
    wrong_history[1] = _shadow_test_exogenous_issue(
        first_issue.issue_time, repeat("f", 64),
    )
    @test_throws ArgumentError operational_v22_shadow_research_predict(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        chain.error_states, chain.conformal, base, -10.0, current,
        wrong_history, innovations,
    )
    wrong_current = _shadow_test_exogenous_issue(
        issue_time, repeat("f", 64); model_step_hours=3,
    )
    @test_throws ArgumentError operational_v22_shadow_research_predict(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        chain.error_states, chain.conformal, base, -10.0, wrong_current,
        issues, innovations,
    )
    wrong_lead_current = _shadow_test_exogenous_issue(
        issue_time, chain.base_hash; model_step_hours=1,
    )
    @test_throws ArgumentError operational_v22_shadow_research_predict(
        chain.artifact, chain.bindings, chain.driver, chain.core,
        chain.error_states, chain.conformal, base, -10.0,
        wrong_lead_current, issues, innovations,
    )

    mktempdir() do directory
        path = joinpath(directory, "exogenous-shadow-chain.csv")
        write_operational_v22_shadow_chain(path, chain.artifact)
        loaded = read_operational_v22_shadow_chain(path)
        @test loaded.m3_kind == :exogenous
        @test loaded.m3_sha256_by_horizon ==
              chain.artifact.m3_sha256_by_horizon
        @test validate_operational_v22_shadow_chain(
            loaded, chain.bindings, chain.driver, chain.core,
            chain.error_states, chain.conformal,
        ) === loaded
    end
end

@testset "V2.2 shadow-chain checksummed I/O" begin
    chain = _shadow_test_chain()
    mktempdir() do directory
        path = joinpath(directory, "shadow-chain.csv")
        @test write_operational_v22_shadow_chain(path, chain.artifact) == path
        loaded = read_operational_v22_shadow_chain(path)
        @test operational_v22_shadow_chain_sha256(loaded) ==
              operational_v22_shadow_chain_sha256(chain.artifact)
        @test loaded.bindings.feature_schema == chain.bindings.feature_schema
        @test loaded.supported_horizons_hours ==
              OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H
        @test validate_operational_v22_shadow_chain(
            loaded,
            chain.bindings,
            chain.driver,
            chain.core,
            chain.error_state,
            chain.conformal,
        ) === loaded

        numeric_bindings = OperationalV22ShadowBindings(
            receipt_pair_contract_sha256=repeat("1", 64),
            transport_support_sha256=repeat("2", 64),
            anchor_pressure_contract_sha256=repeat("3", 64),
            conformal_sidecar_sha256=repeat("4", 64),
            point_calibration_sha256=repeat("5", 64),
            feature_schema=OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA,
        )
        numeric_label = "00012345678901234567890"
        numeric_manifest = OperationalV22ShadowChainArtifact(
            numeric_label,
            numeric_bindings,
            repeat("6", 64),
            repeat("7", 64),
            repeat("8", 64),
            :ar_control,
            ntuple(_ -> repeat("9", 64), 6),
            repeat("0", 64),
            OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H,
            0,
            Val(:validated),
        )
        numeric_path = joinpath(directory, "numeric-manifest.csv")
        write_operational_v22_shadow_chain(numeric_path, numeric_manifest)
        numeric_loaded = read_operational_v22_shadow_chain(numeric_path)
        @test numeric_loaded.label == numeric_label
        @test numeric_loaded.bindings.receipt_pair_contract_sha256 ==
              repeat("1", 64)
        @test numeric_loaded.bindings.transport_support_sha256 ==
              repeat("2", 64)
        @test numeric_loaded.bindings.anchor_pressure_contract_sha256 ==
              repeat("3", 64)
        @test numeric_loaded.bindings.conformal_sidecar_sha256 ==
              repeat("4", 64)
        @test numeric_loaded.bindings.point_calibration_sha256 ==
              repeat("5", 64)
        @test numeric_loaded.driver_sha256 == repeat("6", 64)
        @test numeric_loaded.core_sha256 == repeat("7", 64)
        @test numeric_loaded.base_center_sha256 == repeat("8", 64)
        @test all(==(repeat("9", 64)), numeric_loaded.m3_sha256_by_horizon)
        @test numeric_loaded.conformal_sha256 == repeat("0", 64)
        @test operational_v22_shadow_chain_sha256(numeric_loaded) ==
              operational_v22_shadow_chain_sha256(numeric_manifest)

        nested_path = joinpath(directory, "nested", "artifacts", "shadow.csv")
        @test write_operational_v22_shadow_chain(nested_path, chain.artifact) ==
              nested_path
        @test isfile(nested_path)

        source = joinpath(directory, "source.txt")
        open(source, "w") do io
            write(io, "synthetic provenance\n")
        end
        source_hash = operational_v22_regular_file_sha256(source)
        @test occursin(r"^[0-9a-f]{64}$", source_hash)
        @test source_hash == operational_v22_regular_file_sha256(source)

        corrupted = CSV.read(path, DataFrame)
        corrupted.driver_sha256[1] = repeat("f", 64)
        corrupt_path = joinpath(directory, "corrupt.csv")
        CSV.write(corrupt_path, corrupted)
        @test_throws ArgumentError read_operational_v22_shadow_chain(corrupt_path)

        malformed = CSV.read(path, DataFrame)
        malformed.artifact_sha256[1] = uppercase(malformed.artifact_sha256[1])
        malformed_path = joinpath(directory, "malformed.csv")
        CSV.write(malformed_path, malformed)
        @test_throws ArgumentError read_operational_v22_shadow_chain(malformed_path)

        missing = select(CSV.read(path, DataFrame), Not(:point_calibration_sha256))
        missing_path = joinpath(directory, "missing-column.csv")
        CSV.write(missing_path, missing)
        @test_throws ArgumentError read_operational_v22_shadow_chain(missing_path)

        target_directory = joinpath(directory, "existing-directory")
        mkdir(target_directory)
        @test_throws ArgumentError write_operational_v22_shadow_chain(
            target_directory, chain.artifact,
        )
        @test isdir(target_directory)

        staged = joinpath(directory, "staged-shadow.csv")
        open(staged, "w") do io
            write(io, "staged artifact\n")
        end
        raced_target = joinpath(directory, "raced-target")
        mkdir(raced_target)
        marker = joinpath(raced_target, "marker.txt")
        open(marker, "w") do io
            write(io, "preserve me\n")
        end
        @test_throws ArgumentError SolarSINDy._atomic_replace_regular(
            staged, raced_target,
        )
        @test isfile(staged)
        @test isdir(raced_target)
        @test read(marker, String) == "preserve me\n"

        if Sys.isunix()
            link = joinpath(directory, "artifact-link")
            symlink(path, link)
            @test_throws ArgumentError write_operational_v22_shadow_chain(
                link, chain.artifact,
            )
            @test_throws ArgumentError read_operational_v22_shadow_chain(link)

            fifo = joinpath(directory, "artifact-fifo")
            run(`mkfifo $fifo`)
            @test_throws ArgumentError write_operational_v22_shadow_chain(
                fifo, chain.artifact,
            )
            @test ispath(fifo)
        end
    end
end
