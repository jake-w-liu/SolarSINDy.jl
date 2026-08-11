using Test
using CSV
using DataFrames
using Dates
using Random
using SolarSINDy

const V22X_HASH = repeat("c", 64)
const V22X_OTHER_HASH = repeat("d", 64)
const V22X_START = DateTime(2021, 1, 1)

function _v22x_issue(
        hour::Int;
        center_hash::String=V22X_HASH,
        feature_scale::Real=1.0,
        model_step_hours::Int=1)
    issue_time = V22X_START + Hour(hour)
    features = Float64[
        feature_scale * (100 * hour + index)
        for index in 1:length(OPERATIONAL_V22_RESIDUAL_FEATURES)
    ]
    trajectory = Float64[
        feature_scale * (1_000 * hour + 10 * row + column)
        for row in 1:14, column in 1:5
    ]
    return OperationalV22ErrorExogenousIssue(
        issue_time,
        issue_time,
        center_hash,
        features,
        trajectory;
        model_step_hours=model_step_hours,
    )
end

function _v22x_innovation(
        issued_hour::Int,
        innovation::Real;
        center_hash::String=V22X_HASH,
        available_at::DateTime=V22X_START + Hour(issued_hour + 1))
    issued_at = V22X_START + Hour(issued_hour)
    base = -20.0
    return OperationalV22H1Innovation(
        issued_at,
        issued_at + Hour(1),
        available_at,
        center_hash,
        base,
        base + innovation,
    )
end

function _v22x_complete_inputs(; issue_hour::Int=30)
    current = _v22x_issue(issue_hour; model_step_hours=3)
    issues = [_v22x_issue(issue_hour - lag)
              for lag in reverse(OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)]
    innovations = [_v22x_innovation(issue_hour - lag, -Float64(lag))
                   for lag in reverse(OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)]
    return current, issues, innovations
end

function _v22x_artifact(; lead::Int=3, intercept::Real=1.5)
    coefficients = zeros(length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES))
    coefficients[1] = 0.001
    coefficients[23] = 0.001
    coefficients[59] = 0.2
    return OperationalV22ErrorExogenousArtifact(
        V22X_HASH,
        lead,
        intercept,
        coefficients;
        fit_rows=80,
        label="v2.2-m3-exogenous-hand",
    )
end

@testset verbose=true "Operational V2.2-M3 full exogenous error model" begin
    @testset "Frozen feature and group contract" begin
        @test OPERATIONAL_V22_RESIDUAL_FEATURES == (
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
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H ==
              (1, 2, 3, 4, 6, 9, 12, 18, 24)
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES == (
            :latest_dst_nt,
            :Bz_nt,
            :VBsouth_mvm,
            :sqrt_Pdyn_npa,
            :h1_innovation_nt,
        )
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES == (
            :m2_endpoint_Bx_nt,
            :m2_endpoint_By_nt,
            :m2_endpoint_Bz_nt,
            :m2_endpoint_logV,
            :m2_endpoint_logn,
        )
        @test length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES) == 73
        @test length(unique(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES)) == 73
        @test length(OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS) == 29
        @test length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS) == 73
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[1:22] ==
              OPERATIONAL_V22_RESIDUAL_FEATURES
        expected_lag_names = Tuple(
            Symbol(variable, "_lag_", lag, "h")
            for variable in OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES
            for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H
        )
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[23:67] ==
              expected_lag_names
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[59:67] == Tuple(
            Symbol(:h1_innovation_nt, "_lag_", lag, "h")
            for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H
        )
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[68:72] ==
              OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[end] ==
              :m2_core_center_dst_nt
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[23:31] ==
              ntuple(_ -> 23, 9)
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[59:67] ==
              ntuple(_ -> 27, 9)
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[68:72] ==
              ntuple(_ -> 28, 5)
        @test OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[end] == 29

        current = _v22x_issue(30)
        exact = OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            current.model_step_hours,
            current.feature_names,
            current.issue_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        @test exact == current
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.issue_time + Hour(1),
            current.base_center_sha256,
            current.model_step_hours,
            current.feature_names,
            current.issue_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            "not-a-sha256",
            current.model_step_hours,
            current.feature_names,
            current.issue_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            5,
            current.feature_names,
            current.issue_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        bad_exact_feature_names = Base.setindex(
            current.feature_names, :wrong_feature, 1,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            current.model_step_hours,
            bad_exact_feature_names,
            current.issue_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        bad_exact_features = Base.setindex(current.issue_features, NaN, 1)
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            current.model_step_hours,
            current.feature_names,
            bad_exact_features,
            current.m2_state_names,
            current.m2_trajectory,
        )
        bad_exact_state_names = Base.setindex(
            current.m2_state_names, :wrong_state, 1,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            current.model_step_hours,
            current.feature_names,
            current.issue_features,
            bad_exact_state_names,
            current.m2_trajectory,
        )
        bad_exact_trajectory = Base.setindex(
            current.m2_trajectory,
            Base.setindex(current.m2_trajectory[1], NaN, 1),
            1,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.available_at,
            current.base_center_sha256,
            current.model_step_hours,
            current.feature_names,
            current.issue_features,
            current.m2_state_names,
            bad_exact_trajectory,
        )
        swapped = collect(OPERATIONAL_V22_RESIDUAL_FEATURES)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        @test_throws ArgumentError OperationalV22ErrorExogenousIssue(
            current.issue_time,
            current.issue_time,
            V22X_HASH,
            collect(current.issue_features),
            reduce(vcat, permutedims(collect(row))
                   for row in current.m2_trajectory);
            feature_names=swapped,
        )
        coefficients = zeros(73)
        artifact_names = collect(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES)
        artifact_names[22], artifact_names[23] =
            artifact_names[23], artifact_names[22]
        @test_throws ArgumentError OperationalV22ErrorExogenousArtifact(
            V22X_HASH,
            1,
            0.0,
            coefficients;
            feature_names=artifact_names,
            fit_rows=2,
        )
        boolean_groups = Any[
            OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS...,
        ]
        boolean_groups[1] = true
        @test_throws ArgumentError OperationalV22ErrorExogenousArtifact(
            V22X_HASH,
            1,
            0.0,
            zeros(73);
            feature_groups=boolean_groups,
            fit_rows=2,
        )
        @test_throws ArgumentError OperationalV22ErrorExogenousArtifact(
            V22X_HASH,
            1,
            0.0,
            zeros(73);
            fit_rows=2,
            threshold_iterations=true,
        )
    end

    @testset "Independent causal feature oracle and post-issue invariance" begin
        current, issues, innovations = _v22x_complete_inputs()
        base = -12.5
        row = operational_v22_error_exogenous_features(
            current, issues, innovations, 3, V22X_HASH, base,
        )
        @test row.ready
        @test row.fallback_reason == :none
        @test row.feature_names == OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES
        @test row.values[1:22] == current.issue_features
        required_hours = Tuple(30 - lag
                               for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)
        @test row.values[23:31] == Tuple(100.0 * hour + 1
                                        for hour in required_hours)
        @test row.values[32:40] == Tuple(100.0 * hour + 5
                                        for hour in required_hours)
        @test row.values[41:49] == Tuple(100.0 * hour + 7
                                        for hour in required_hours)
        @test row.values[50:58] == Tuple(100.0 * hour + 11
                                        for hour in required_hours)
        @test row.values[59:67] == Tuple(-Float64(lag)
            for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)
        @test row.values[68:72] == current.m2_trajectory[6]
        @test row.values[73] == base

        future_issue_a = _v22x_issue(40; feature_scale=1.0)
        future_issue_b = _v22x_issue(40; feature_scale=-99.0)
        future_innovation_a = _v22x_innovation(40, 1.0)
        future_innovation_b = _v22x_innovation(40, -100_000.0)
        row_a = operational_v22_error_exogenous_features(
            current,
            vcat(issues, [future_issue_a]),
            vcat(innovations, [future_innovation_a]),
            3,
            V22X_HASH,
            base,
        )
        row_b = operational_v22_error_exogenous_features(
            current,
            vcat(issues, [future_issue_b]),
            vcat(innovations, [future_innovation_b]),
            3,
            V22X_HASH,
            base,
        )
        @test row_a == row
        @test row_b == row

        late_same_key = _v22x_innovation(
            29,
            -100_000.0;
            center_hash=V22X_OTHER_HASH,
            available_at=current.issue_time + Hour(1),
        )
        late_append = operational_v22_error_exogenous_features(
            current,
            issues,
            vcat(innovations, [late_same_key]),
            3,
            V22X_HASH,
            base,
        )
        late_prepend = operational_v22_error_exogenous_features(
            current,
            issues,
            vcat([late_same_key], innovations),
            3,
            V22X_HASH,
            base,
        )
        @test late_append == row
        @test late_prepend == row
        artifact = _v22x_artifact()
        original_prediction = operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_HASH, base, issues, innovations,
        )
        revised_prediction = operational_v22_error_exogenous_predict(
            artifact,
            current,
            3,
            V22X_HASH,
            base,
            issues,
            vcat(innovations, [late_same_key]),
        )
        @test revised_prediction == original_prediction
        @test_throws ArgumentError operational_v22_error_exogenous_features(
            current, issues, innovations, 3, V22X_OTHER_HASH, base,
        )
        @test_throws ArgumentError operational_v22_error_exogenous_features(
            current, issues, innovations, 1, V22X_HASH, base,
        )
        lead_seven_issue = _v22x_issue(30; model_step_hours=7)
        lead_seven = operational_v22_error_exogenous_features(
            lead_seven_issue, issues, innovations, 7, V22X_HASH, base,
        )
        @test lead_seven.values[68:72] == lead_seven_issue.m2_trajectory[14]
        @test_throws ArgumentError operational_v22_error_exogenous_features(
            current, issues, innovations, 5, V22X_HASH, base,
        )
    end

    @testset "Missing-feature fallbacks are explicit" begin
        current, issues, innovations = _v22x_complete_inputs()
        args = (3, V22X_HASH, -12.5)
        @test operational_v22_error_exogenous_features(
            current, issues[2:end], innovations, args...,
        ).fallback_reason == :missing_issue_history
        wrong_lead_history = copy(issues)
        wrong_lead_hour = Dates.value(
            wrong_lead_history[1].issue_time - V22X_START,
        ) ÷ 3_600_000
        wrong_lead_history[1] = _v22x_issue(
            wrong_lead_hour; model_step_hours=3,
        )
        @test operational_v22_error_exogenous_features(
            current, wrong_lead_history, innovations, args...,
        ).fallback_reason == :missing_issue_history
        @test operational_v22_error_exogenous_features(
            current, vcat(issues, issues[1:1]), innovations, args...,
        ).fallback_reason == :duplicate_issue_record
        @test operational_v22_error_exogenous_features(
            current, issues, innovations[2:end], args...,
        ).fallback_reason == :missing_innovation_history
        @test operational_v22_error_exogenous_features(
            current, issues, vcat(innovations, innovations[1:1]), args...,
        ).fallback_reason == :duplicate_innovation_record

        delayed = copy(innovations)
        lag_one_hour = 30 - OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H[1]
        lag_one_index = findfirst(
            record -> record.issued_at == V22X_START + Hour(lag_one_hour),
            delayed,
        )
        delayed[lag_one_index] = _v22x_innovation(
            lag_one_hour,
            -1.0;
            available_at=current.issue_time + Hour(1),
        )
        @test operational_v22_error_exogenous_features(
            current, issues, delayed, args...,
        ).fallback_reason == :missing_innovation_history

        wrong_issue = copy(issues)
        wrong_issue[1] = _v22x_issue(
            Dates.value(wrong_issue[1].issue_time - V22X_START) ÷ 3_600_000;
            center_hash=V22X_OTHER_HASH,
        )
        @test operational_v22_error_exogenous_features(
            current, wrong_issue, innovations, args...,
        ).fallback_reason == :base_center_mismatch
        wrong_innovation = copy(innovations)
        first_record = wrong_innovation[1]
        issue_hour = Dates.value(first_record.issued_at - V22X_START) ÷ 3_600_000
        wrong_innovation[1] = _v22x_innovation(
            issue_hour,
            operational_v22_h1_innovation(first_record);
            center_hash=V22X_OTHER_HASH,
        )
        @test operational_v22_error_exogenous_features(
            current, issues, wrong_innovation, args...,
        ).fallback_reason == :base_center_mismatch
    end

    @testset "Independent prediction, stability, and cap oracles" begin
        current, issues, innovations = _v22x_complete_inputs()
        artifact = _v22x_artifact()
        result = operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_HASH, -12.5, issues, innovations,
        )
        expected_raw = 1.5 + 0.001 * 3_001.0 +
                       0.001 * 2_901.0 + 0.2 * -1.0
        @test result.raw_correction_nt == expected_raw
        @test result.correction_nt == expected_raw
        @test result.pred_dst_nt == -12.5 + expected_raw
        @test result.correction_cap_nt == 20.0
        @test result.correction_applied
        @test !result.correction_was_capped
        @test result.fallback_reason == :none
        @test result.artifact_sha256 ==
              operational_v22_error_exogenous_sha256(artifact)

        future_a = operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_HASH, -12.5,
            vcat(issues, [_v22x_issue(40; feature_scale=1.0)]),
            vcat(innovations, [_v22x_innovation(40, 1.0)]),
        )
        future_b = operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_HASH, -12.5,
            vcat(issues, [_v22x_issue(40; feature_scale=-99.0)]),
            vcat(innovations, [_v22x_innovation(40, -100_000.0)]),
        )
        @test future_a == future_b == result

        missing = operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_HASH, -12.5,
            issues[2:end], innovations,
        )
        @test missing.pred_dst_nt == -12.5
        @test missing.correction_nt == 0.0
        @test !missing.correction_applied
        @test missing.fallback_reason == :missing_issue_history

        capped = OperationalV22ErrorExogenousArtifact(
            V22X_HASH,
            3,
            100.0,
            zeros(73);
            fit_rows=10,
        )
        capped_result = operational_v22_error_exogenous_predict(
            capped, current, 3, V22X_HASH, -12.5, issues, innovations,
        )
        @test capped_result.raw_correction_nt == 100.0
        @test capped_result.correction_nt == 20.0
        @test capped_result.pred_dst_nt == 7.5
        @test capped_result.correction_was_capped

        unstable = zeros(73)
        unstable[59] = 1.0
        @test_throws ArgumentError OperationalV22ErrorExogenousArtifact(
            V22X_HASH, 1, 0.0, unstable; fit_rows=10,
        )
        exogenous_only = zeros(73)
        exogenous_only[23] = 10_000.0
        @test OperationalV22ErrorExogenousArtifact(
            V22X_HASH, 1, 0.0, exogenous_only; fit_rows=10,
        ).spectral_radius == 0.0
        scaled_ar = zeros(73)
        scaled_ar[59] = 0.4
        scales = ones(73)
        scales[59] = 2.0
        @test OperationalV22ErrorExogenousArtifact(
            V22X_HASH,
            1,
            0.0,
            scaled_ar;
            feature_scale=scales,
            fit_rows=10,
        ).spectral_radius ≈ 0.2
        @test_throws ArgumentError operational_v22_error_exogenous_predict(
            artifact, current, 1, V22X_HASH, -12.5, issues, innovations,
        )
        @test_throws ArgumentError operational_v22_error_exogenous_predict(
            artifact, current, 3, V22X_OTHER_HASH, -12.5, issues, innovations,
        )
        @test_throws MethodError operational_v22_error_exogenous_predict(
            artifact, current, 3, -12.5, issues, innovations,
        )
    end

    @testset "Known group support and deterministic fitting" begin
        rng = MersenneTwister(22_003)
        nrows = 240
        design = randn(rng, nrows, 73)
        rows = OperationalV22ErrorExogenousFitRow[]
        for row_index in 1:nrows
            issue_time = V22X_START + Hour(100 + row_index)
            base = design[row_index, 73]
            values = Tuple(design[row_index, :])
            features = OperationalV22ErrorExogenousFeatures(
                issue_time,
                1,
                V22X_HASH,
                base,
                OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
                values,
                true,
                :none,
            )
            residual = 1.2 * design[row_index, 1] +
                       0.7 * design[row_index, 23] -
                       0.5 * design[row_index, 24]
            target_time = issue_time + Hour(1)
            push!(rows, OperationalV22ErrorExogenousFitRow(
                features,
                target_time,
                target_time,
                base + residual,
            ))
        end
        kwargs = (
            base_center_sha256=V22X_HASH,
            model_step_hours=1,
            fit_as_of=V22X_START + Hour(500),
            ridge=1.0e-6,
            threshold=0.1,
            minimum_rows=200,
            label="v2.2-m3-exogenous-support",
        )
        fit_a = fit_operational_v22_error_exogenous(rows; kwargs...)
        fit_b = fit_operational_v22_error_exogenous(rows; kwargs...)
        expected_support = falses(29)
        expected_support[[1, 23]] .= true
        @test collect(fit_a.support_mask) == expected_support
        @test all(iszero, fit_a.coefficients[32:end])
        @test fit_a == fit_b
        @test operational_v22_error_exogenous_sha256(fit_a) ==
              operational_v22_error_exogenous_sha256(fit_b)
        @test fit_a.fit_rows == nrows
        @test fit_a.spectral_radius == 0.0

        delayed_target = OperationalV22ErrorExogenousFitRow(
            rows[1].features,
            rows[1].target_time,
            V22X_START + Hour(501),
            rows[1].observation_dst_nt + 1.0e6,
        )
        fit_with_unavailable_target = fit_operational_v22_error_exogenous(
            vcat(rows, [delayed_target]); kwargs...,
        )
        @test operational_v22_error_exogenous_sha256(
            fit_with_unavailable_target,
        ) == operational_v22_error_exogenous_sha256(fit_a)

        mutated_rows = copy(rows)
        first_row = mutated_rows[1]
        mutated_rows[1] = OperationalV22ErrorExogenousFitRow(
            first_row.features,
            first_row.target_time,
            first_row.observation_available_at,
            first_row.observation_dst_nt + 1.0,
        )
        mutated_fit = fit_operational_v22_error_exogenous(
            mutated_rows; kwargs...,
        )
        @test operational_v22_error_exogenous_sha256(mutated_fit) !=
              operational_v22_error_exogenous_sha256(fit_a)
    end

    @testset "Checksummed artifact I/O and corruption rejection" begin
        artifact = _v22x_artifact()
        @test length(operational_v22_error_exogenous_sha256(artifact)) == 64
        mktempdir() do directory
            path = joinpath(directory, "m3_exogenous.csv")
            @test write_operational_v22_error_exogenous(path, artifact) == path
            loaded = read_operational_v22_error_exogenous(path)
            @test loaded == artifact
            @test operational_v22_error_exogenous_sha256(loaded) ==
                  operational_v22_error_exogenous_sha256(artifact)
            replacement = _v22x_artifact(intercept=2.5)
            @test begin
                write_operational_v22_error_exogenous(path, replacement)
                read_operational_v22_error_exogenous(path) == replacement
            end

            corrupted = CSV.read(path, DataFrame)
            corrupted[1, :coefficient] += 0.25
            corrupted_path = joinpath(directory, "corrupted.csv")
            CSV.write(corrupted_path, corrupted)
            @test_throws ArgumentError read_operational_v22_error_exogenous(
                corrupted_path,
            )

            reordered = CSV.read(path, DataFrame)
            reordered[1, :feature_name], reordered[2, :feature_name] =
                reordered[2, :feature_name], reordered[1, :feature_name]
            reordered_path = joinpath(directory, "reordered.csv")
            CSV.write(reordered_path, reordered)
            @test_throws ArgumentError read_operational_v22_error_exogenous(
                reordered_path,
            )

            wrong_radius = CSV.read(path, DataFrame)
            wrong_radius[!, :spectral_radius] .= artifact.spectral_radius + 0.1
            radius_path = joinpath(directory, "wrong_radius.csv")
            CSV.write(radius_path, wrong_radius)
            @test_throws ArgumentError read_operational_v22_error_exogenous(
                radius_path,
            )

            directory_target = joinpath(directory, "existing_directory")
            mkpath(directory_target)
            @test_throws ArgumentError write_operational_v22_error_exogenous(
                directory_target, artifact,
            )
            @test isdir(directory_target)

            staged = joinpath(directory, "staged_exogenous.csv")
            CSV.write(staged, DataFrame(value=[23]))
            raced_target = joinpath(directory, "raced_target")
            mkpath(raced_target)
            marker = joinpath(raced_target, "marker.csv")
            CSV.write(marker, DataFrame(value=[29]))
            @test_throws ArgumentError SolarSINDy._atomic_replace_regular(
                staged, raced_target,
            )
            @test isfile(staged)
            @test isdir(raced_target)
            @test CSV.read(marker, DataFrame).value == [29]

            sentinel = joinpath(directory, "sentinel.csv")
            CSV.write(sentinel, DataFrame(value=[17]))
            symlink_target = joinpath(directory, "artifact_symlink.csv")
            symlink(sentinel, symlink_target)
            @test_throws ArgumentError write_operational_v22_error_exogenous(
                symlink_target, artifact,
            )
            @test islink(symlink_target)
            @test CSV.read(sentinel, DataFrame).value == [17]
            @test_throws ArgumentError read_operational_v22_error_exogenous(
                symlink_target,
            )
        end
    end
end
