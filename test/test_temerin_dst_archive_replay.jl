using Test
using DataFrames
using Dates

const TEMERIN_DST_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "..", "live_forecasts",
                                                    "temerin_dst_archive_replay.jl"))

@testset "Temerin-Li Dst archive replay helpers" begin
    if !isfile(TEMERIN_DST_REPLAY_SCRIPT)
        @test_skip "research-workspace Temerin-Li Dst replay script is not present"
    else
        include(TEMERIN_DST_REPLAY_SCRIPT)

        rows = parse_temerin_dst_text(_sample_temerin_text(); source = "sample")
        @test nrow(rows) == 4
        @test rows.valid_utc[1] == DateTime(2024, 5, 1, 0, 1, 20)
        @test rows.valid_utc[end] == DateTime(2024, 5, 2, 0, 1, 20)
        @test rows.temerin_li_dst_nt[1] == -41.1

        broad = DataFrame(
            storm_id = [1, 1, 1],
            storm = ["synthetic", "synthetic", "synthetic"],
            storm_min_dst_nt = [-120.0, -120.0, -120.0],
            issue_utc = [DateTime(2024, 4, 30, 23), DateTime(2024, 5, 1, 0),
                         DateTime(2024, 5, 1, 4)],
            target_utc = [DateTime(2024, 5, 1, 0), DateTime(2024, 5, 1, 1),
                          DateTime(2024, 5, 1, 5)],
            lead = [1, 1, 1],
            obs = [-40.0, -50.0, -60.0],
            audit_baseline = [-39.0, -48.0, -61.0],
            v2 = [-40.5, -49.0, -59.0],
            persistence = [-38.0, -45.0, -55.0],
        )
        scored = score_temerin_dst_archive(broad, rows;
                                           start_utc = DateTime(2024, 5, 1),
                                           end_utc = DateTime(2024, 5, 2),
                                           max_match_gap_min = 5.0)
        @test nrow(scored) == 2
        @test all(scored.target_utc .== scored.issue_utc .+ Hour.(scored.lead))
        @test maximum(scored.match_abs_gap_min) < 3.0
        @test _validate_temerin_rows(scored; max_match_gap_min = 5.0)

        summary = temerin_dst_summary(scored)
        @test nrow(summary) == 2
        @test Set(summary.scope) == Set(["all_operational_input_era", "dscovr_real_time_input_era"])
        crc = _temerin_summary_consistent(scored, summary)
        @test crc.ok

        broken = copy(scored)
        broken.target_utc[1] = broken.target_utc[1] + Hour(1)
        @test_throws ErrorException _validate_temerin_rows(broken; max_match_gap_min = 5.0)

        broken_gap = copy(scored)
        broken_gap.match_abs_gap_min[1] = 99.0
        @test_throws ErrorException _validate_temerin_rows(broken_gap; max_match_gap_min = 5.0)
    end
end
