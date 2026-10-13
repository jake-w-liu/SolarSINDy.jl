using Test
using CSV
using DataFrames
using Dates
using SolarSINDy

const StaticStackE_HASH = repeat("a", 64)
const StaticStackE_OTHER_HASH = repeat("b", 64)
const StaticStackE_START = DateTime(2020, 1, 1)

function _errstate_record(
        issue_hour::Int,
        innovation_nt::Real;
        available_at::DateTime=StaticStackE_START + Hour(issue_hour + 1),
        center_hash::String=StaticStackE_HASH,
        base_prediction_dst_nt::Real=-20.0)
    issued_at = StaticStackE_START + Hour(issue_hour)
    return StaticStackH1Innovation(
        issued_at,
        issued_at + Hour(1),
        available_at,
        center_hash,
        base_prediction_dst_nt,
        base_prediction_dst_nt + innovation_nt,
    )
end

function _errstate_records(innovations::AbstractVector{<:Real})
    return [_errstate_record(index - 1, value)
            for (index, value) in enumerate(innovations)]
end

function _errstate_artifact(; intercept=1.0, first=0.5, second=0.25)
    coefficients = zeros(9)
    coefficients[1] = first
    coefficients[2] = second
    support = ntuple(index -> index <= 2, 9)
    return StaticStackErrorStateArtifact(
        StaticStackE_HASH,
        intercept,
        coefficients;
        support_mask=support,
        fit_rows=100,
        selection_score=2.5,
        label="static-stack-m3-error-hand",
    )
end

@testset verbose=true "Operational static stack-M3 causal error state" begin
    @testset "Matured innovation construction and fail-closed history" begin
        records = _errstate_records(collect(0.0:40.0))
        issue_time = StaticStackE_START + Hour(30)
        history = static_stack_matured_h1_history(
            records, issue_time, StaticStackE_HASH,
        )
        @test history.ready
        @test history.fallback_reason == :none
        @test history.innovation_buffer_nt[1] == 29.0
        @test history.innovation_buffer_nt[24] == 6.0
        @test history.lagged_innovations_nt ==
              Tuple(30.0 - lag for lag in STATIC_STACK_ERROR_LAGS_H)
        @test static_stack_h1_innovation(records[8]) == 7.0

        post_issue_mutation = copy(records)
        post_issue_mutation[36] = _errstate_record(35, -90_000.0)
        @test static_stack_matured_h1_history(
            post_issue_mutation, issue_time, StaticStackE_HASH,
        ) == history

        late_same_key = _errstate_record(
            29,
            -90_000.0;
            available_at=issue_time + Hour(1),
            center_hash=StaticStackE_OTHER_HASH,
        )
        @test static_stack_matured_h1_history(
            vcat(records, [late_same_key]), issue_time, StaticStackE_HASH,
        ) == history
        @test static_stack_matured_h1_history(
            vcat([late_same_key], records), issue_time, StaticStackE_HASH,
        ) == history

        missing = [record for record in records
                   if record.issued_at != issue_time - Hour(5)]
        missing_history = static_stack_matured_h1_history(
            missing, issue_time, StaticStackE_HASH,
        )
        @test !missing_history.ready
        @test missing_history.fallback_reason == :missing_history

        # A record that exists but whose observation has not matured is a different operational
        # state from a record that was never produced: only the first resolves by waiting. Both are
        # not-ready, and the fallback reason has to tell them apart.
        delayed = copy(records)
        delayed[30] = _errstate_record(
            29,
            29.0;
            available_at=issue_time + Hour(1),
        )
        delayed_history = static_stack_matured_h1_history(
            delayed, issue_time, StaticStackE_HASH,
        )
        @test !delayed_history.ready
        @test delayed_history.fallback_reason == :observation_not_mature
        @test delayed_history.fallback_reason != :missing_history
        # An immature record at the very edge of the 24 h buffer is still classified as immature.
        edge_delayed = copy(records)
        edge_delayed[7] = _errstate_record(6, 6.0; available_at=issue_time + Hour(1))
        @test static_stack_matured_h1_history(
            edge_delayed, issue_time, StaticStackE_HASH,
        ).fallback_reason == :observation_not_mature
        # An immature record OUTSIDE the buffer must not relabel a genuinely complete history.
        outside_delayed = copy(records)
        outside_delayed[6] = _errstate_record(5, 5.0; available_at=issue_time + Hour(1))
        @test static_stack_matured_h1_history(
            outside_delayed, issue_time, StaticStackE_HASH,
        ) == history
        # And a record that is genuinely absent still reports the absence, not immaturity.
        absent = [record for record in records
                  if record.issued_at != issue_time - Hour(3)]
        absent_history = static_stack_matured_h1_history(absent, issue_time, StaticStackE_HASH)
        @test !absent_history.ready
        @test absent_history.fallback_reason == :missing_history

        wrong_center = copy(records)
        wrong_center[30] = _errstate_record(29, 29.0; center_hash=StaticStackE_OTHER_HASH)
        @test static_stack_matured_h1_history(
            wrong_center, issue_time, StaticStackE_HASH,
        ).fallback_reason == :base_center_mismatch

        duplicate = vcat(records, records[30:30])
        @test static_stack_matured_h1_history(
            duplicate, issue_time, StaticStackE_HASH,
        ).fallback_reason == :duplicate_issue_record

        @test_throws ArgumentError StaticStackH1Innovation(
            issue_time,
            issue_time + Hour(2),
            issue_time + Hour(2),
            StaticStackE_HASH,
            -20.0,
            -21.0,
        )
        @test_throws ArgumentError StaticStackH1Innovation(
            issue_time,
            issue_time + Hour(1),
            issue_time,
            StaticStackE_HASH,
            -20.0,
            -21.0,
        )
        @test_throws ArgumentError _errstate_record(1, NaN)
    end

    @testset "Independent recurrence, cap, and post-issue mutation oracles" begin
        artifact = _errstate_artifact()
        innovations = zeros(41)
        innovations[30] = 4.0  # issued at hour 29: lag one at issue hour 30
        innovations[29] = 2.0  # issued at hour 28: lag two
        records = _errstate_records(innovations)
        issue_time = StaticStackE_START + Hour(30)

        lead_one = static_stack_error_state_predict(
            artifact, issue_time, 1, StaticStackE_HASH, -30.0, records,
        )
        @test lead_one.raw_correction_nt == 3.5
        @test lead_one.correction_nt == 3.5
        @test lead_one.pred_dst_nt == -26.5
        @test lead_one.correction_applied
        @test !lead_one.correction_was_capped

        lead_two = static_stack_error_state_predict(
            artifact, issue_time, 2, StaticStackE_HASH, -30.0, records,
        )
        # q[t] = 1 + 0.5*4 + 0.25*2 = 3.5;
        # q[t+1] = 1 + 0.5*3.5 + 0.25*4 = 3.75.
        @test lead_two.raw_correction_nt == 3.75
        @test lead_two.pred_dst_nt == -26.25

        late_same_key = _errstate_record(
            29,
            -90_000.0;
            available_at=issue_time + Hour(1),
            center_hash=StaticStackE_OTHER_HASH,
        )
        @test static_stack_error_state_predict(
            artifact,
            issue_time,
            2,
            StaticStackE_HASH,
            -30.0,
            vcat(records, [late_same_key]),
        ) == lead_two

        future_a = vcat(records, [_errstate_record(50, 1.0)])
        future_b = vcat(records, [_errstate_record(50, -100_000.0)])
        @test static_stack_error_state_predict(
            artifact, issue_time, 2, StaticStackE_HASH, -30.0, future_a,
        ) == static_stack_error_state_predict(
            artifact, issue_time, 2, StaticStackE_HASH, -30.0, future_b,
        )

        capped = StaticStackErrorStateArtifact(
            StaticStackE_HASH,
            100.0,
            zeros(9);
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
            selection_score=0.0,
        )
        cap_result = static_stack_error_state_predict(
            capped, issue_time, 1, StaticStackE_HASH, -30.0, records,
        )
        @test cap_result.raw_correction_nt == 100.0
        @test cap_result.correction_nt == 10.0
        @test cap_result.pred_dst_nt == -20.0
        @test cap_result.correction_was_capped

        missing = records[2:end]
        fallback = static_stack_error_state_predict(
            artifact, StaticStackE_START + Hour(24), 1, StaticStackE_HASH, -30.0, missing,
        )
        @test fallback.pred_dst_nt == -30.0
        @test fallback.correction_nt == 0.0
        @test !fallback.correction_applied
        @test fallback.fallback_reason == :missing_history
        @test_throws ArgumentError static_stack_error_state_predict(
            artifact, issue_time, 5, StaticStackE_HASH, -30.0, records,
        )
        @test_throws ArgumentError static_stack_error_state_predict(
            artifact, issue_time, true, StaticStackE_HASH, -30.0, records,
        )
        @test_throws ArgumentError static_stack_error_state_predict(
            artifact, issue_time, 1, StaticStackE_HASH, NaN, records,
        )
        @test_throws ArgumentError static_stack_error_state_predict(
            artifact, issue_time, 1, StaticStackE_OTHER_HASH, -30.0, records,
        )
        @test_throws MethodError static_stack_error_state_predict(
            artifact, issue_time, 1, -30.0, records,
        )
    end

    @testset "Stability and artifact contract rejection" begin
        artifact = _errstate_artifact()
        @test artifact.issue_lags_hours == STATIC_STACK_ERROR_LAGS_H
        @test artifact.spectral_radius < STATIC_STACK_ERROR_MAX_SPECTRAL_RADIUS
        @test !ismutabletype(StaticStackErrorStateArtifact)

        unstable = zeros(9)
        unstable[1] = 1.0
        @test_throws ArgumentError StaticStackErrorStateArtifact(
            StaticStackE_HASH,
            0.0,
            unstable;
            support_mask=ntuple(index -> index == 1, 9),
            fit_rows=100,
        )
        excluded = zeros(9)
        excluded[1] = 0.1
        @test_throws ArgumentError StaticStackErrorStateArtifact(
            StaticStackE_HASH,
            0.0,
            excluded;
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
        )
        @test_throws ArgumentError StaticStackErrorStateArtifact(
            uppercase(StaticStackE_HASH),
            0.0,
            zeros(9);
            support_mask=ntuple(_ -> false, 9),
            fit_rows=100,
        )
        @test_throws DimensionMismatch StaticStackErrorStateArtifact(
            StaticStackE_HASH,
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
        records = _errstate_records(innovations)
        fit_as_of = StaticStackE_START + Hour(length(innovations))
        fitted_a = fit_static_stack_error_state(
            records;
            base_center_sha256=StaticStackE_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="static-stack-m3-error-synthetic",
        )
        fitted_b = fit_static_stack_error_state(
            records;
            base_center_sha256=StaticStackE_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="static-stack-m3-error-synthetic",
        )
        @test fitted_a.support_mask == ntuple(index -> index <= 2, 9)
        @test isapprox(fitted_a.intercept_nt, 0.35; rtol=0.0, atol=2e-12)
        @test isapprox(fitted_a.coefficients[1], 1.2; rtol=0.0, atol=2e-12)
        @test isapprox(fitted_a.coefficients[2], -0.81; rtol=0.0, atol=2e-12)
        @test fitted_a.spectral_radius < STATIC_STACK_ERROR_MAX_SPECTRAL_RADIUS
        @test fitted_a.fit_rows == length(innovations) - 24
        @test static_stack_error_state_sha256(fitted_a) ==
              static_stack_error_state_sha256(fitted_b)

        future = _errstate_record(length(innovations) + 5, 90_000.0)
        fitted_with_future = fit_static_stack_error_state(
            vcat(records, [future]);
            base_center_sha256=StaticStackE_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
            label="static-stack-m3-error-synthetic",
        )
        @test static_stack_error_state_sha256(fitted_with_future) ==
              static_stack_error_state_sha256(fitted_a)

        @test_throws ArgumentError fit_static_stack_error_state(
            records[1:40];
            base_center_sha256=StaticStackE_HASH,
            fit_as_of=StaticStackE_START + Hour(40),
            ridge=0.0,
            max_terms=2,
            minimum_rows=32,
        )
        @test_throws ArgumentError fit_static_stack_error_state(
            vcat(records, records[50:50]);
            base_center_sha256=StaticStackE_HASH,
            fit_as_of,
            ridge=0.0,
            max_terms=2,
            minimum_rows=100,
        )
    end

    @testset "Checksummed artifact round trip and corruption rejection" begin
        artifact = _errstate_artifact()
        innovations = zeros(41)
        innovations[30] = 4.0
        innovations[29] = 2.0
        records = _errstate_records(innovations)
        issue_time = StaticStackE_START + Hour(30)
        mktempdir() do temporary
            path = joinpath(temporary, "nested", "error-state.csv")
            @test write_static_stack_error_state(path, artifact) == path
            restored = read_static_stack_error_state(path)
            @test static_stack_error_state_sha256(restored) ==
                  static_stack_error_state_sha256(artifact)
            @test static_stack_error_state_predict(
                restored, issue_time, 2, StaticStackE_HASH, -30.0, records,
            ) == static_stack_error_state_predict(
                artifact, issue_time, 2, StaticStackE_HASH, -30.0, records,
            )
            replacement = _errstate_artifact(intercept=2.0)
            @test begin
                write_static_stack_error_state(path, replacement)
                read_static_stack_error_state(path) == replacement
            end

            valid = CSV.read(path, DataFrame)
            corrupted = copy(valid)
            corrupted.coefficient[1] += 0.01
            corrupted_path = joinpath(temporary, "corrupted.csv")
            CSV.write(corrupted_path, corrupted)
            @test_throws ArgumentError read_static_stack_error_state(
                corrupted_path,
            )

            rehashed_needed = copy(valid)
            rehashed_needed.intercept_nt .+= 0.01
            rehashed_needed_path = joinpath(temporary, "checksum-mismatch.csv")
            CSV.write(rehashed_needed_path, rehashed_needed)
            @test_throws ArgumentError read_static_stack_error_state(
                rehashed_needed_path,
            )

            wrong_schema_path = joinpath(temporary, "wrong-schema.csv")
            CSV.write(wrong_schema_path, select(valid, Not(:selection_score)))
            @test_throws ArgumentError read_static_stack_error_state(
                wrong_schema_path,
            )

            link = joinpath(temporary, "error-state-link.csv")
            # Windows requires Administrator/Developer Mode to create symlinks; when the OS
            # denies creation the symlink-rejection assertions cannot run here and are skipped.
            can_symlink = try
                symlink(path, link)
                true
            catch e
                e isa Base.IOError || rethrow()
                false
            end
            if can_symlink
                @test_throws ArgumentError read_static_stack_error_state(link)
                artifact_bytes = read(path)
                @test_throws ArgumentError write_static_stack_error_state(
                    link, artifact,
                )
                @test islink(link)
                @test read(path) == artifact_bytes
            else
                @warn "symlink creation denied by the OS; skipping symlink evidence checks"
            end

            directory_target = joinpath(temporary, "directory-target.csv")
            mkdir(directory_target)
            sentinel = joinpath(directory_target, "preserve.txt")
            write(sentinel, "preserve")
            @test_throws ArgumentError write_static_stack_error_state(
                directory_target, artifact,
            )
            @test isdir(directory_target)
            @test read(sentinel, String) == "preserve"
        end
    end

    @testset "Numeric-looking labels round-trip as text (G05-02)" begin
        mktempdir() do temporary
            for label in ("007", "1e3")
                artifact = StaticStackErrorStateArtifact(
                    StaticStackE_HASH, 0.1, [0.5, 0, 0, 0, 0, 0, 0, 0, 0];
                    fit_rows=10, label=label,
                )
                path = joinpath(temporary, "error-$label.csv")
                write_static_stack_error_state(path, artifact)
                restored = read_static_stack_error_state(path)
                @test restored.label == label
                @test static_stack_error_state_sha256(restored) ==
                      static_stack_error_state_sha256(artifact)
            end
        end
    end
end
