module NOAAKpForecastReplayTests

using Test
using DataFrames
using Dates

const NOAA_KP_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "..", "live_forecasts",
                                                "noaa_kp_forecast_replay.jl"))

@testset "NOAA 3-day Kp forecast archive replay helpers" begin
    if !isfile(NOAA_KP_REPLAY_SCRIPT)
        @test_skip "research-workspace NOAA Kp replay script is not present"
    else
        include(NOAA_KP_REPLAY_SCRIPT)

        rows = parse_noaa_3day_kp_text(_sample_noaa_3day_text(); source = "sample")
        @test nrow(rows) == 19
        @test minimum(rows.target_bin_start_utc) == DateTime(2024, 5, 10, 15)
        @test maximum(rows.target_bin_start_utc) == DateTime(2024, 5, 12, 21)
        @test all(rows.target_bin_start_utc .>= _ceil_next_kp_bin.(rows.issue_utc))

        may11_06 = rows[rows.target_bin_start_utc .== DateTime(2024, 5, 11, 6), :][1, :]
        @test may11_06.forecast_kp == 8.33
        @test may11_06.forecast_g_level == 4

        old_text = replace(_sample_noaa_3day_text(), ":Issued:" => "Issued:")
        old_rows = parse_noaa_3day_kp_text(old_text; source = "old-format")
        @test nrow(old_rows) == nrow(rows)
        @test old_rows.issue_utc == rows.issue_utc

        boundary_text = replace(_sample_noaa_3day_text(), "1230 UTC" => "0000 UTC")
        boundary_rows = parse_noaa_3day_kp_text(boundary_text; source = "boundary")
        @test minimum(boundary_rows.target_bin_start_utc) == DateTime(2024, 5, 10, 3)
        @test all(boundary_rows.target_bin_start_utc .> boundary_rows.issue_utc)

        observed = DataFrame(utc = rows.target_bin_start_utc,
                             kp = [i <= 5 ? 7.0 : 2.0 for i in 1:nrow(rows)])
        scored = score_noaa_kp_forecasts(rows, observed)
        @test _validate_noaa_rows(scored)
        summary = noaa_kp_summary(scored)
        g3 = summary[(summary.scope .== "all") .& (summary.lead_band .== "all") .&
                     (summary.threshold_g .== 3), :][1, :]
        @test g3.n_rows == 19
        @test g3.hits == 2
        @test g3.misses == 3
        @test g3.false_alarms == 0
        @test isapprox(g3.pod, 0.4; atol = 1e-12)

        broken = copy(scored)
        broken.target_bin_start_utc[1] = broken.issue_utc[1] - Hour(1)
        @test_throws ErrorException _validate_noaa_rows(broken)
    end
end

end # module NOAAKpForecastReplayTests
