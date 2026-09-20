module CalibrationDiagnosisTests
using Test, DataFrames, Dates
include(joinpath(@__DIR__, "../validation/operational/calibration_diagnosis.jl"))
const D = CalibrationDiagnosis

@testset "Independent interval diagnosis counts and score" begin
    y = [-2., 0., 4., 5.]; p = [0., 0., 3., 3.]
    lo = [-1., -1., 2., 2.]; hi = [1., 1., 4., 4.]
    result = D.interval_metrics(y, p, lo, hi, lo.-1, hi.+1)
    @test result.n == 4 && result.covered == 2 && result.below == result.above == 1
    @test result.coverage == .5 && result.static_coverage == 1
    @test result.width_ratio == .5 && result.mean_width_nt == 2
    @test result.mean_point_residual_nt == .25
    @test result.mean_interval_center_residual_nt == .25
    @test result.point_rmse_nt == 1.5
    @test result.mean_score_difference_nt == 8
    @test_throws ArgumentError D.interval_metrics([], [], [], [], [], [])
    @test_throws ArgumentError D.interval_metrics(y[1:3], p, lo, hi, lo, hi)
    @test_throws ArgumentError D.interval_metrics([NaN; y[2:4]], p, lo, hi, lo, hi)
    @test_throws ArgumentError D.interval_metrics(y, p, hi, lo, lo, hi)
    @test_throws ArgumentError D.interval_metrics(fill(true,4), p, lo, hi, lo, hi)
    frame = DataFrame(;observation_dst_nt=y, point_dst_nt=p, shadow_lo_dst_nt=lo,
        shadow_hi_dst_nt=hi, static_lo_dst_nt=lo.-1, static_hi_dst_nt=hi.+1,
        model_step_hours=[1,1,2,2])
    grouped = D.summarize_intervals(frame, [:model_step_hours])
    @test grouped.n == [2,2] && grouped.covered == [1,1]
    @test grouped.below == [1,0] && grouped.above == [0,1]
    @test only(D.summarize_intervals(frame, Symbol[]).covered) == 2
    moved = copy(frame); moved.observation_dst_nt[1] = 0
    @test only(D.summarize_intervals(moved, Symbol[]).covered) == 3
end

@testset "Ground diagnosis uses identical rows and units" begin
    frame = DataFrame(prediction=[1.,2.,3.], target=[0.,2.,6.], past=[0.,1.,2.], upper=[1.,3.,5.])
    result = D.ground_metrics(frame)
    @test result.n == 3 && result.covered == 2 && result.missed == 1
    @test result.coverage == 2/3
    @test result.mean_upper_nt_min == 3
    @test result.mean_target_nt_min == 8/3
    @test result.point_rmse_nt_min == sqrt(10/3)
    @test result.past_rmse_nt_min == sqrt(17/3)
    @test result.mean_exceedance_nt_min == 1/3
    @test_throws ArgumentError D.ground_metrics(frame[1:0,:])
    frame.target[1] = -1
    @test_throws ArgumentError D.ground_metrics(frame)
end

@testset "Calibration seed maturity respects receipt chronology and gaps" begin
    start = DateTime(2020)
    times = vcat(collect(start:Minute(1):start+Minute(1550)),
        collect(start+Day(5):Minute(1):start+Day(5)+Minute(1490)))
    for delay in (0,5,10)
        ends = times .+ Minute(delay+30)
        result = D.calibration_phases(times, ends, delay)
        independent = String[]
        epoch = 1
        for k in eachindex(times)
            k > 1 && times[k]-times[k-1] > Day(1) && (epoch=k)
            count_received = count(j -> ends[j]+Minute(delay) < times[k]+Minute(delay), epoch:k-1)
            push!(independent, count_received < 1440 ? "seed_present" : "live_residuals_only")
        end
        @test result == independent
        @test result[1470+delay] == "seed_present"
        @test result[1471+delay] == "live_residuals_only"
        @test result[1552] == "seed_present"
    end
    @test_throws ArgumentError D.calibration_phases(reverse(times), reverse(times).+Minute(30), 0)
    @test_throws ArgumentError D.calibration_phases(times, times.+Minute(31), 0)
    @test_throws ArgumentError D.calibration_phases(times, times.+Minute(33), 3)
end
end
