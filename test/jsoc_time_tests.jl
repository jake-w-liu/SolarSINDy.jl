module JSOCTimeIndependentTests

using Test, Dates
include(joinpath(@__DIR__, "..", "tools", "source_clocks", "JSOCTime.jl"))
using .JSOCTime: parse_trec_utc

@testset "JSOC TAI labels convert to the UTC flare clock" begin
    # Literal UTC expectations from IERS offsets, independent of helper tables.
    for (label, expected) in (
        ("2011.01.01_12:00:00_TAI", DateTime(2011, 1, 1, 11, 59, 26)),
        ("2012.07.12_12:00:00_TAI", DateTime(2012, 7, 12, 11, 59, 25)),
        ("2016.01.01_12:00:00_TAI", DateTime(2016, 1, 1, 11, 59, 24)),
        ("2017.01.01_12:00:00_TAI", DateTime(2017, 1, 1, 11, 59, 23)),
        ("2026.11.10_12:00:00_TAI", DateTime(2026, 11, 10, 11, 59, 23)),
        ("2009.01.01_00:00:34_TAI", DateTime(2009, 1, 1)),
        ("2027.06.28_00:00:36_TAI", DateTime(2027, 6, 27, 23, 59, 59)),
        ("2012.07.01_00:00:33_TAI", DateTime(2012, 6, 30, 23, 59, 59)),
        ("2012.07.01_00:00:35_TAI", DateTime(2012, 7, 1)),
        ("2015.07.01_00:00:34_TAI", DateTime(2015, 6, 30, 23, 59, 59)),
        ("2015.07.01_00:00:36_TAI", DateTime(2015, 7, 1)),
        ("2017.01.01_00:00:35_TAI", DateTime(2016, 12, 31, 23, 59, 59)),
        ("2017.01.01_00:00:37_TAI", DateTime(2017, 1, 1)),
    )
        @test parse_trec_utc(label) == expected
    end
    # DateTime has no UTC second60; silently mapping it to midnight is wrong.
    for label in ("2012.07.01_00:00:34_TAI", "2015.07.01_00:00:35_TAI",
                  "2017.01.01_00:00:36_TAI", "2009.01.01_00:00:33_TAI",
                  "2027.06.28_00:00:37_TAI", "2030.01.01_12:00:00_TAI",
                  "2012.07.12_12:00:00_UTC", "2012.07.12_12:00:00",
                  "2012.07.12_12:00:00_tai", "2012.13.12_12:00:00_TAI",
                  "2012.07.12_12:00:60_TAI", " 2012.07.12_12:00:00_TAI")
        @test_throws Exception parse_trec_utc(label)
    end

    origin = parse_trec_utc("2012.07.12_12:00:00_TAI")
    onsets = [DateTime(2012, 7, 12, 11, 59, 24), DateTime(2012, 7, 12, 11, 59, 25),
        DateTime(2012, 7, 12, 11, 59, 26), DateTime(2012, 7, 12, 11, 59, 50),
        DateTime(2012, 7, 13, 11, 59, 24), DateTime(2012, 7, 13, 11, 59, 25),
        DateTime(2012, 7, 13, 11, 59, 26), DateTime(2012, 7, 13, 11, 59, 50)]
    @test [origin < onset <= origin + Hour(24) for onset in onsets] ==
        [false, false, true, true, true, true, false, false]
end

end # module
