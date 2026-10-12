module NOAAKpForecastReplayTests

using Test
using DataFrames
using Dates

const NOAA_KP_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "validation", "operational",
                                                "noaa_kp_forecast_replay.jl"))
include(NOAA_KP_REPLAY_SCRIPT)

@testset "NOAA 3-day Kp forecast archive replay helpers" begin
        rows = parse_noaa_3day_kp_text(_sample_noaa_3day_text(); source = "sample")
        @test nrow(rows) == 19
        @test minimum(rows.target_bin_start_utc) == DateTime(2024, 5, 10, 15)
        @test maximum(rows.target_bin_start_utc) == DateTime(2024, 5, 12, 21)
        @test all(rows.target_bin_start_utc .>= _ceil_next_kp_bin.(rows.issue_utc))

        may11_06 = rows[rows.target_bin_start_utc .== DateTime(2024, 5, 11, 6), :][1, :]
        @test may11_06.forecast_kp == 8.33
        @test may11_06.forecast_g_level == 4
        @test may11_06.forecast_g_label_printed == 4
        printed_rows = rows[rows.forecast_g_label_printed .>= 0, :]
        @test nrow(printed_rows) == 11   # all 11 printed (Gn) cells are future bins
        @test printed_rows.forecast_g_label_printed == printed_rows.forecast_g_level
        @test all(rows[rows.forecast_g_label_printed .== -1, :].forecast_g_level .== 0)

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
        # Was 0 under the integer mapping. SWPC prints "6.67 (G3)" (May 11 09-12UT) and the
        # observed Kp there is 2.0, so that G3 forecast is a false alarm (V11-02).
        @test g3.false_alarms == 1
        @test isapprox(g3.pod, 0.4; atol = 1e-12)
        @test isapprox(g3.far, 1 / 3; atol = 1e-12)

        broken = copy(scored)
        broken.target_bin_start_utc[1] = broken.issue_utc[1] - Hour(1)
        @test_throws ErrorException _validate_noaa_rows(broken)

        @testset "SWPC printed (Gn) labels equal the recomputed G level" begin
            # Independent regex over the raw sample: every number in the Kp table, with its
            # printed label (absent label = below G1).
            text = _sample_noaa_3day_text()
            table = [l for l in split(text, '\n') if occursin(r"^\s*\d{2}-\d{2}UT", l)]
            @test length(table) == 8
            n_labelled = 0
            n_cells = 0
            for l in table
                for mm in eachmatch(r"(\d+\.\d+)(?:\s*\(G(\d)\))?", l)
                    kp = parse(Float64, mm.captures[1])
                    printed = mm.captures[2] === nothing ? 0 : parse(Int, mm.captures[2])
                    n_cells += 1
                    n_labelled += printed > 0
                    @test noaa_g_level(kp) == printed
                end
            end
            @test n_cells == 24
            @test n_labelled == 11
            # The parsed forecast level for each printed label (future bins only).
            by_t(t) = rows[rows.target_bin_start_utc .== t, :][1, :]
            @test by_t(DateTime(2024, 5, 11, 9)).forecast_g_level == 3   # 6.67 (G3)
            @test by_t(DateTime(2024, 5, 12, 0)).forecast_g_level == 1   # 4.67 (G1)
            @test by_t(DateTime(2024, 5, 12, 3)).forecast_g_level == 2   # 5.67 (G2)
            @test by_t(DateTime(2024, 5, 11, 12)).forecast_g_level == 2  # 5.67 (G2)
            @test by_t(DateTime(2024, 5, 11, 18)).forecast_g_level == 1  # 4.67 (G1)
            @test by_t(DateTime(2024, 5, 11, 21)).forecast_g_level == 0  # 4.33

            # GFZ observed thirds are mapped the same way.
            obs = DataFrame(utc = rows.target_bin_start_utc[1:5],
                            kp = [4.667, 6.667, 8.667, 9.0, 4.333])
            sc = score_noaa_kp_forecasts(rows[1:5, :], obs)
            @test nrow(sc) == 5   # score_noaa_kp_forecasts re-sorts rows by target time
            @test Dict(zip(Float64.(sc.observed_kp), Int.(sc.observed_g_level))) ==
                  Dict(4.667 => 1, 6.667 => 3, 8.667 => 4, 9.0 => 5, 4.333 => 0)
        end

        @testset "label/level disagreement fails closed" begin
            bad = replace(_sample_noaa_3day_text(), "6.67 (G3)" => "6.67 (G2)")
            @test_throws NoaaGLabelMismatch parse_noaa_3day_kp_text(bad; source = "bad")
            unlabelled = replace(_sample_noaa_3day_text(), "7.00 (G3)    5.67 (G2)" =>
                                                            "7.00 (G3)    5.67     ")
            @test unlabelled != _sample_noaa_3day_text()
            # A missing label is not a printed disagreement (real SWPC omissions exist, e.g.
            # 202503280030): the row is scored at the SWPC level and flagged, not dropped.
            ul = @test_logs (:warn, r"without a printed G label") match_mode = :any parse_noaa_3day_kp_text(unlabelled; source = "omitted")
            r = ul[ul.target_bin_start_utc .== DateTime(2024, 5, 12, 3), :][1, :]
            @test r.forecast_kp == 5.67
            @test r.forecast_g_level == 2
            @test r.forecast_g_label_printed == -1
            @test nrow(_unlabelled_storm_rows(ul)) == 1
            @test nrow(_unlabelled_storm_rows(rows)) == 0
            # Scored-row validation rejects a printed label that disagrees with the level.
            tampered = copy(scored)
            k = findfirst(>(0), tampered.forecast_g_label_printed)
            tampered.forecast_g_label_printed[k] += 1
            @test_throws NoaaGLabelMismatch _validate_noaa_rows(tampered)
            regraded = copy(scored)
            regraded.forecast_g_level[k] += 1
            @test_throws ErrorException _validate_noaa_rows(regraded)
            spurious = replace(_sample_noaa_3day_text(), "4.33         3.67" =>
                                                          "4.33 (G1)    3.67")
            @test spurious != _sample_noaa_3day_text()
            @test_throws NoaaGLabelMismatch parse_noaa_3day_kp_text(spurious; source = "bad")
            junk = replace(_sample_noaa_3day_text(), "8.33 (G4)" => "8.33 (x4)")
            @test_throws NoaaGLabelMismatch parse_noaa_3day_kp_text(junk; source = "bad")

            # load_noaa_3day_kp skips unparsable files but must not skip a label mismatch.
            mktempdir() do dir
                good = joinpath(dir, "202405101230three_day_forecast.txt")
                badf = joinpath(dir, "202405110030three_day_forecast.txt")
                write(good, _sample_noaa_3day_text())
                write(badf, bad)
                @test nrow(load_noaa_3day_kp([good]; start_utc = DateTime(2024, 5, 1),
                                             end_utc = DateTime(2024, 5, 31))) == 19
                @test_throws NoaaGLabelMismatch load_noaa_3day_kp([good, badf];
                                                                  start_utc = DateTime(2024, 5, 1),
                                                                  end_utc = DateTime(2024, 5, 31))
            end
        end
end

end # module NOAAKpForecastReplayTests
