using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "examples", "live_forecast_verify.jl"))

@testset "Live Forecast Verification Workflow" begin
    @testset "A/D: target time is strictly future relative to issue time" begin
        issue_time = DateTime(2026, 6, 6, 4, 0, 34)
        latest_dst_time = DateTime(2026, 6, 6, 3)
        target = _next_hourly_target(issue_time, 1, latest_dst_time)

        @test target == DateTime(2026, 6, 6, 5)
        @test target > issue_time
        @test target > latest_dst_time

        exact_hour_issue = DateTime(2026, 6, 6, 4)
        @test _next_hourly_target(exact_hour_issue, 1, latest_dst_time) ==
              DateTime(2026, 6, 6, 5)
    end

    @testset "A/D: append preserves old log rows while adding baseline columns" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            old_row = DataFrame(
                issue_time_utc = ["2026-06-06T02:17:54.992"],
                target_time_utc = ["2026-06-06T03:00:00"],
                pred_dst_nt = [-49.56],
                pred_dst_ci05_nt = [-54.52],
                pred_dst_ci95_nt = [-44.48],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
            )
            CSV.write(log_path, old_row)

            new_row = DataFrame(
                issue_time_utc = ["2026-06-06T04:15:00"],
                target_time_utc = ["2026-06-06T05:00:00"],
                pred_dst_nt = [-50.0],
                pred_dst_ci05_nt = [-55.0],
                pred_dst_ci95_nt = [-45.0],
                persistence_dst_nt = [-49.0],
                burton_dst_nt = [-48.0],
                burton_full_dst_nt = [-47.0],
                obrien_dst_nt = [-46.0],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
            )
            row_idx = _append_forecast!(log_path, new_row)
            df = CSV.read(log_path, DataFrame)

            @test row_idx == 2
            @test nrow(df) == 2
            @test :obrien_dst_nt in propertynames(df)
            @test ismissing(df.obrien_dst_nt[1])
            @test df.obrien_dst_nt[2] == -46.0
        end
    end

    @testset "A/D: verify_pending! scores SINDy and baseline residuals" begin
        mktempdir() do tmp
            log_path = joinpath(tmp, "live_forecast_log.csv")
            target = DateTime(2026, 6, 6, 5)
            row = DataFrame(
                issue_time_utc = ["2026-06-06T04:15:00"],
                target_time_utc = [string(target)],
                pred_dst_nt = [-50.0],
                pred_dst_ci05_nt = [-55.0],
                pred_dst_ci95_nt = [-45.0],
                persistence_dst_nt = [-49.0],
                burton_dst_nt = [-48.0],
                burton_full_dst_nt = [-47.0],
                obrien_dst_nt = [-46.0],
                observation_dst_nt = [missing],
                residual_dst_nt = [missing],
                observed_in_90ci = [missing],
                persistence_residual_dst_nt = [missing],
                burton_residual_dst_nt = [missing],
                burton_full_residual_dst_nt = [missing],
                obrien_residual_dst_nt = [missing],
            )
            CSV.write(log_path, row)

            cfg = LiveVerifyConfig(mode=:verify_pending, log_path=log_path)
            n_verified = verify_pending!(cfg;
                dst_times=[target],
                dst_vals=[-63.0],
            )
            df = CSV.read(log_path, DataFrame)

            @test n_verified == 1
            @test df.observation_dst_nt[1] == -63.0
            @test df.residual_dst_nt[1] == -13.0
            @test df.observed_in_90ci[1] == false
            @test df.persistence_residual_dst_nt[1] == -14.0
            @test df.burton_residual_dst_nt[1] == -15.0
            @test df.burton_full_residual_dst_nt[1] == -16.0
            @test df.obrien_residual_dst_nt[1] == -17.0
        end
    end
end
