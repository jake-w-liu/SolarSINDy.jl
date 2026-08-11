using Test
using CSV
using DataFrames
using Dates
using SolarSINDy

const V22E_HASH = repeat("a", 64)
const V22E_OTHER_HASH = repeat("b", 64)
const V22E_START = DateTime(2020, 1, 1)

function _v22e_record(
        issue_hour::Int,
        innovation_nt::Real;
        available_at::DateTime=V22E_START + Hour(issue_hour + 1),
        center_hash::String=V22E_HASH,
        base_prediction_dst_nt::Real=-20.0)
    issued_at = V22E_START + Hour(issue_hour)
    return OperationalV22H1Innovation(
        issued_at,
        issued_at + Hour(1),
        available_at,
        center_hash,
        base_prediction_dst_nt,
        base_prediction_dst_nt + innovation_nt,
    )
end

function _v22e_records(innovations::AbstractVector{<:Real})
    return [_v22e_record(index - 1, value)
            for (index, value) in enumerate(innovations)]
end

function _v22e_artifact(; intercept=1.0, first=0.5, second=0.25)
    coefficients = zeros(9)
    coefficients[1] = first
    coefficients[2] = second
    support = ntuple(index -> index <= 2, 9)
    return OperationalV22ErrorStateArtifact(
        V22E_HASH,
        intercept,
        coefficients;
        support_mask=support,
        fit_rows=100,
        selection_score=2.5,
        label="v2.2-m3-error-hand",
    )
end

@testset verbose=true "Operational V2.2-M3 causal error state" begin
    @testset "Matured innovation construction and fail-closed history" begin
        records = _v22e_records(collect(0.0:40.0))
        issue_time = V22E_START + Hour(30)
        history = operational_v22_matured_h1_history(
            records, issue_time, V22E_HASH,
        )
        @test history.ready
        @test history.fallback_reason == :none
        @test history.innovation_buffer_nt[1] == 29.0
        @test history.innovation_buffer_nt[24] == 6.0
        @test history.lagged_innovations_nt ==
              Tuple(30.0 - lag for lag in OPERATIONAL_V22_ERROR_LAGS_H)
        @test operational_v22_h1_innovation(records[8]) == 7.0

        post_issue_mutation = copy(records)
        post_issue_mutation[36] = _v22e_record(35, -90_000.0)
        @test operational_v22_matured_h1_history(
            post_issue_mutation, issue_time, V22E_HASH,
        ) == history

        late_same_key = _v22e_record(
            29,
            -90_000.0;
            available_at=issue_time + Hour(1),
            center_hash=V22E_OTHER_HASH,
        )
        @test operational_v22_matured_h1_history(
            vcat(records, [late_same_key]), issue_time, V22E_HASH,
        ) == history
        @test operational_v22_matured_h1_history(
            vcat([late_same_key], records), issue_time, V22E_HASH,
        ) == history

        missing = [record for record in records
                   if record.issued_at != issue_time - Hour(5)]
        missing_history = operational_v22_matured_h1_history(
            missing, issue_time, V22E_HASH,
        )
        @test !missing_history.ready
        @test missing_history.fallback_reason == :missing_history

        delayed = copy(records)
        delayed[30] = _v22e_record(
            29,
            29.0;
            available_at=issue_time + Hour(1),
        )
        @test operational_v22_matured_h1_history(
            delayed, issue_time, V22E_HASH,
        ).fallback_reason == :missing_history

        wrong_center = copy(records)
        wrong_center[30] = _v22e_record(29, 29.0; center_hash=V22E_OTHER_HASH)
        @test operational_v22_matured_h1_history(
            wrong_center, issue_time, V22E_HASH,
        ).fallback_reason == :base_center_mismatch

        duplicate = vcat(records, records[30:30])
        @test operational_v22_matured_h1_history(
            duplicate, issue_time, V22E_HASH,
        ).fallback_reason == :duplicate_issue_record

        @test_throws ArgumentError OperationalV22H1Innovation(
            issue_time,
            issue_time + Hour(2),
            issue_time + Hour(2),
            V22E_HASH,
            -20.0,
            -21.0,
        )
        @test_throws ArgumentError OperationalV22H1Innovation(
            issue_time,
            issue_time + Hour(1),
            issue_time,
            V22E_HASH,
            -20.0,
            -21.0,
        )
        @test_throws ArgumentError _v22e_record(1, NaN)
    end

    @testset "Independent recurrence, cap, and post-issue mutation oracles" begin
        artifact = _v22e_artifact()
        innovations = zeros(41)
        innovations[30] = 4.0  # issued at hour 29: lag one at issue hour 30
        innovations[29] = 2.0  # issued at hour 28: lag two
        records = _v22e_records(innovations)
        issue_time = V22E_START + Hour(30)

        lead_one = operational_v22_error_state_predict(
            artifact, issue_time, 1, V22E_HASH, -30.0, records,
        )
        @test lead_one.raw_correction_nt == 3.5
        @test lead_one.correction_nt == 3.5
        @test lead_one.pred_dst_nt == -26.5
        @test lead_one.correction_applied
        @test !lead_one.correction_was_capped

        lead_two = operational_v22_error_state_predict(
            artifact, issue_time, 2, V22E_HASH, -30.0, records,
        )
        # q[t] = 1 + 0.5*4 + 0.25*2 = 3.5;
        # q[t+1] = 1 + 0.5*3.5 + 0.25*4 = 3.75.
        @test lead_two.raw_correction_nt == 3.75
        @test lead_two.pred_dst_nt == -26.25

        late_same_key = _v22e_record(
            29,
            -90_000.0;
            available_at=issue_time + Hour(1),
            center_hash=V22E_OTHER_HASH,
        )
        @test operational_v22_error_state_predict(
            artifact,
            issue_time,
            2,
            V22E_HASH,
            -30.0,
            vcat(records, [late_same_key]),
        ) == lead_two

        future_a = vcat(records, [_v22e_record(50, 1.0)])
        future_b = vcat(records, [_v22e_record(50, -100_000.0)])
        @test operational_v22_error_state_predict(
            artifact, issue_time, 2, V22E_HASH, -30.0, future_a,
        ) == operational_v22_error_state_predict(
            artifact, issue_time, 2, V22E_HASH, -30.0, future_b,
        )

        capped = OperationalV22ErrorStateArtifact(
            V22E_HASH,
            100.0,
            zeros(9);
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
            selection_score=0.0,
        )
        cap_result = operational_v22_error_state_predict(
            capped, issue_time, 1, V22E_HASH, -30.0, records,
        )
        @test cap_result.raw_correction_nt == 100.0
        @test cap_result.correction_nt == 10.0
        @test cap_result.pred_dst_nt == -20.0
        @test cap_result.correction_was_capped

        missing = records[2:end]
        fallback = operational_v22_error_state_predict(
            artifact, V22E_START + Hour(24), 1, V22E_HASH, -30.0, missing,
        )
        @test fallback.pred_dst_nt == -30.0
        @test fallback.correction_nt == 0.0
        @test !fallback.correction_applied
        @test fallback.fallback_reason == :missing_history
        @test_throws ArgumentError operational_v22_error_state_predict(
            artifact, issue_time, 5, V22E_HASH, -30.0, records,
        )
        @test_throws ArgumentError operational_v22_error_state_predict(
            artifact, issue_time, true, V22E_HASH, -30.0, records,
        )
        @test_throws ArgumentError operational_v22_error_state_predict(
            artifact, issue_time, 1, V22E_HASH, NaN, records,
        )
        @test_throws ArgumentError operational_v22_error_state_predict(
            artifact, issue_time, 1, V22E_OTHER_HASH, -30.0, records,
        )
        @test_throws MethodError operational_v22_error_state_predict(
            artifact, issue_time, 1, -30.0, records,
        )
    end

    @testset "Stability and artifact contract rejection" begin
        artifact = _v22e_artifact()
        @test artifact.issue_lags_hours == OPERATIONAL_V22_ERROR_LAGS_H
        @test artifact.spectral_radius < OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS
        @test !ismutabletype(OperationalV22ErrorStateArtifact)

        unstable = zeros(9)
        unstable[1] = 1.0
        @test_throws ArgumentError OperationalV22ErrorStateArtifact(
            V22E_HASH,
            0.0,
            unstable;
            support_mask=ntuple(index -> index == 1, 9),
            fit_rows=100,
        )
        excluded = zeros(9)
        excluded[1] = 0.1
        @test_throws ArgumentError OperationalV22ErrorStateArtifact(
            V22E_HASH,
            0.0,
            excluded;
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
        )
        @test_throws ArgumentError OperationalV22ErrorStateArtifact(
            uppercase(V22E_HASH),
            0.0,
            zeros(9);
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
        )
        @test_throws DimensionMismatch OperationalV22ErrorStateArtifact(
            V22E_HASH,
            0.0,
            zeros(8);
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
        )
    end

    @testset "Sparse stable fit and repeat-fit determinism" begin
        innovations = zeros(180)
        for index in 1:24
            innovations[index] = sin(0.31 * index) + 0.4 * cos(0.17 * index)
        end
        for index in 25:length(innovations)
            innovations[index] = 0.35 +
                                 1.2 * innovations[index - 1] -
                                 0.81 * innovations[index - 2]
        end
        records = _v22e_records(innovations)
        fit_as_of = V22E_START + Hour(length(innovations))
        fitted_a = fit_operational_v22_error_state(
            records;
            base_center_sha256=V22E_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="v2.2-m3-error-synthetic",
        )
        fitted_b = fit_operational_v22_error_state(
            records;
            base_center_sha256=V22E_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="v2.2-m3-error-synthetic",
        )
        @test fitted_a.support_mask == ntuple(index -> index <= 2, 9)
        @test isapprox(fitted_a.intercept_nt, 0.35; rtol=0.0, atol=2e-12)
        @test isapprox(fitted_a.coefficients[1], 1.2; rtol=0.0, atol=2e-12)
        @test isapprox(fitted_a.coefficients[2], -0.81; rtol=0.0, atol=2e-12)
        @test fitted_a.spectral_radius < OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS
        @test fitted_a.fit_rows == length(innovations) - 24
        @test operational_v22_error_state_sha256(fitted_a) ==
              operational_v22_error_state_sha256(fitted_b)

        future = _v22e_record(length(innovations) + 5, 90_000.0)
        fitted_with_future = fit_operational_v22_error_state(
            vcat(records, [future]);
            base_center_sha256=V22E_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="v2.2-m3-error-synthetic",
        )
        @test operational_v22_error_state_sha256(fitted_with_future) ==
              operational_v22_error_state_sha256(fitted_a)

        @test_throws ArgumentError fit_operational_v22_error_state(
            records[1:40];
            base_center_sha256=V22E_HASH,
            fit_as_of=V22E_START + Hour(40),
            ridge=0.0,
            max_terms=2,
            minimum_rows=32,
        )
        @test_throws ArgumentError fit_operational_v22_error_state(
            vcat(records, records[50:50]);
            base_center_sha256=V22E_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
        )
    end

    @testset "Checksummed artifact round trip and corruption rejection" begin
        artifact = _v22e_artifact()
        innovations = zeros(41)
        innovations[30] = 4.0
        innovations[29] = 2.0
        records = _v22e_records(innovations)
        issue_time = V22E_START + Hour(30)
        mktempdir() do temporary
            path = joinpath(temporary, "nested", "error-state.csv")
            @test write_operational_v22_error_state(path, artifact) == path
            restored = read_operational_v22_error_state(path)
            @test operational_v22_error_state_sha256(restored) ==
                  operational_v22_error_state_sha256(artifact)
            @test operational_v22_error_state_predict(
                restored, issue_time, 2, V22E_HASH, -30.0, records,
            ) == operational_v22_error_state_predict(
                artifact, issue_time, 2, V22E_HASH, -30.0, records,
            )
            replacement = _v22e_artifact(intercept=2.0)
            @test begin
                write_operational_v22_error_state(path, replacement)
                read_operational_v22_error_state(path) == replacement
            end

            valid = CSV.read(path, DataFrame)
            corrupted = copy(valid)
            corrupted.coefficient[1] += 0.01
            corrupted_path = joinpath(temporary, "corrupted.csv")
            CSV.write(corrupted_path, corrupted)
            @test_throws ArgumentError read_operational_v22_error_state(
                corrupted_path,
            )

            rehashed_needed = copy(valid)
            rehashed_needed.intercept_nt .+= 0.01
            rehashed_needed_path = joinpath(temporary, "checksum-mismatch.csv")
            CSV.write(rehashed_needed_path, rehashed_needed)
            @test_throws ArgumentError read_operational_v22_error_state(
                rehashed_needed_path,
            )

            wrong_schema_path = joinpath(temporary, "wrong-schema.csv")
            CSV.write(wrong_schema_path, select(valid, Not(:selection_score)))
            @test_throws ArgumentError read_operational_v22_error_state(
                wrong_schema_path,
            )

            link = joinpath(temporary, "error-state-link.csv")
            symlink(path, link)
            @test_throws ArgumentError read_operational_v22_error_state(link)
            artifact_bytes = read(path)
            @test_throws ArgumentError write_operational_v22_error_state(
                link, artifact,
            )
            @test islink(link)
            @test read(path) == artifact_bytes

            directory_target = joinpath(temporary, "directory-target.csv")
            mkdir(directory_target)
            sentinel = joinpath(directory_target, "preserve.txt")
            write(sentinel, "preserve")
            @test_throws ArgumentError write_operational_v22_error_state(
                directory_target, artifact,
            )
            @test isdir(directory_target)
            @test read(sentinel, String) == "preserve"
        end
    end
end
