using HTTP
using Dates

@testset "Realtime And Monitor" begin

    @testset "Dst refresh retains the last successful feed" begin
        refreshed = ([DateTime(2026, 1, 1)], [-40.0])
        current = SolarSINDy._refresh_dst_feed(nothing, () -> refreshed)
        @test current === refreshed
        current = SolarSINDy._refresh_dst_feed(current, () -> error("transient"))
        @test current === refreshed
        @test_throws InterruptException SolarSINDy._refresh_dst_feed(
            current, () -> throw(InterruptException()),
        )
    end

    @testset "A/B: fetch_realtime_solar_wind performs hourly averaging and interpolation" begin
        plasma = DataFrame(
            time_tag = [
                DateTime(2026, 1, 1, 0, 0, 0),
                DateTime(2026, 1, 1, 0, 30, 0),
                DateTime(2026, 1, 1, 2, 15, 0),
                DateTime(2026, 1, 1, 2, 45, 0),
                DateTime(2026, 1, 1, 3, 0, 0),
            ],
            density = [5.0, 7.0, 9.0, 11.0, 11.0],
            speed = [400.0, 420.0, 440.0, 460.0, 460.0],
            temperature = [1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.3e5],
        )
        mag = DataFrame(
            time_tag = [
                DateTime(2026, 1, 1, 0, 0, 0),
                DateTime(2026, 1, 1, 0, 30, 0),
                DateTime(2026, 1, 1, 2, 15, 0),
                DateTime(2026, 1, 1, 2, 45, 0),
                DateTime(2026, 1, 1, 3, 0, 0),
            ],
            bx_gsm = [1.0, 1.0, 1.0, 1.0, 1.0],
            by_gsm = [2.0, 4.0, 8.0, 10.0, 10.0],
            bz_gsm = [-3.0, -5.0, -7.0, -9.0, -9.0],
            bt = [3.7, 6.4, 10.7, 13.5, 13.5],
        )

        # Coarse synthetic fixture (2 samples/hour): opt out of the 1-min-cadence
        # coverage gate so this test exercises the averaging/interpolation path.
        swd, tags, t_fresh = fetch_realtime_solar_wind(hours=3; plasma=plasma, mag=mag,
                                                       min_hourly_samples=1)

        @test length(tags) == 3
        # Freshness anchor is the newest actual common sample (03:00), not the last hour-floored
        # bin start (02:00) — the monitor uses this so a live feed is not falsely flagged STALE.
        @test t_fresh == DateTime(2026, 1, 1, 3, 0, 0)
        @test tags[end] == DateTime(2026, 1, 1, 2, 0, 0)
        @test swd.t == [0.0, 1.0, 2.0]
        @test swd.V[1] ≈ 410.0 atol=1e-12
        @test swd.n[1] ≈ 6.0 atol=1e-12
        @test swd.Bz[1] ≈ -4.0 atol=1e-12
        @test swd.By[1] ≈ 3.0 atol=1e-12
        @test swd.V[2] ≈ 430.0 atol=1e-12  # One missing bin linearly interpolated between 410 and 450.
        @test swd.n[2] ≈ 8.0 atol=1e-12
        @test swd.Bz[2] ≈ -6.0 atol=1e-12
        @test swd.V[3] ≈ 450.0 atol=1e-12
        @test swd.Pdyn[1] ≈ 1.6726e-6 * 6.0 * 410.0^2 atol=1e-12
        @test all(isnan, swd.Dst_star)
    end

    @testset "INGEST-01: one-pass hourly aggregation matches the mask oracle" begin
        t0 = DateTime(2026, 1, 1)
        times = [
            t0 + Hour(1) + Minute(45), # deliberately unsorted
            t0 - Minute(1),            # before the retained window
            t0 + Minute(10),
            t0 + Hour(3),              # right edge is excluded
            t0 + Minute(50),
            t0 + Hour(1),
            t0 + Hour(2) + Minute(10),
            t0 + Hour(1) + Minute(20),
        ]
        x = [9.0, 999.0, 1.0, 111.0, 3.0, 5.0, Inf, 7.0]
        y = [8.0, 999.0, NaN, 111.0, 4.0, 6.0, 10.0, 2.0]
        min_samples = 2

        function mask_oracle(values)
            return [begin
                left = t0 + Hour(i - 1)
                right = left + Hour(1)
                selected = [Float64(values[j]) for j in eachindex(times)
                            if left <= times[j] < right && isfinite(values[j])]
                length(selected) >= min_samples ? mean(selected) : NaN
            end for i in 1:3]
        end

        actual_x = fill(NaN, 3)
        actual_y = fill(NaN, 3)
        SolarSINDy._hourly_means!(actual_x, actual_y, times, x, y, t0, min_samples)
        expected_x = mask_oracle(x)
        expected_y = mask_oracle(y)
        @test all(isequal.(actual_x, expected_x))
        @test all(isequal.(actual_y, expected_y))
    end

    @testset "INGEST-01: hourly ingestion allocation scales linearly" begin
        function dense_feed(n_hours)
            t0 = DateTime(2026, 1, 1)
            times = collect(t0:Minute(1):t0 + Hour(n_hours))
            n = length(times)
            plasma = DataFrame(time_tag=times, density=fill(5.0, n),
                               speed=fill(400.0, n), temperature=fill(1.0e5, n))
            mag = DataFrame(time_tag=times, bx_gsm=fill(1.0, n),
                            by_gsm=fill(2.0, n), bz_gsm=fill(-5.0, n),
                            bt=fill(5.5, n))
            return plasma, mag
        end

        plasma_24, mag_24 = dense_feed(24)
        plasma_168, mag_168 = dense_feed(168)
        fetch_realtime_solar_wind(hours=24; plasma=plasma_24, mag=mag_24)
        fetch_realtime_solar_wind(hours=168; plasma=plasma_168, mag=mag_168)
        bytes_24 = @allocated fetch_realtime_solar_wind(
            hours=24; plasma=plasma_24, mag=mag_24)
        bytes_168 = @allocated fetch_realtime_solar_wind(
            hours=168; plasma=plasma_168, mag=mag_168)

        # Seven times the data/window should remain near-linear. The old
        # full-feed mask per bin allocated 1,368,160 bytes for this 168 h case.
        @test bytes_168 <= 8 * bytes_24
        @test bytes_168 < 300_000
    end

    @testset "A/D: RTSW named-key parsers retry truncated JSON, keep active source, guard schema" begin
        # Captured-sample payloads matching the live NOAA SWPC RTSW products (arrays of OBJECTS
        # with named keys). Each time_tag carries an active primary (SOLAR1) and an inactive
        # secondary (ACE); the parser must keep only the active, physically-valid rows.
        plasma_calls = Ref(0)
        function flaky_plasma_get(url; kwargs...)
            plasma_calls[] += 1
            if plasma_calls[] == 1
                return (; status=200, body="""[{"time_tag":"2026-01-01T00:00:00",""")  # truncated
            end
            return (; status=200, body="""
                [{"time_tag":"2026-01-01T00:01:00","active":true,"source":"SOLAR1",
                  "proton_speed":610.0,"proton_density":5.5,"proton_temperature":500000},
                 {"time_tag":"2026-01-01T00:01:00","active":false,"source":"ACE",
                  "proton_speed":123.0,"proton_density":99.0,"proton_temperature":1.0},
                 {"time_tag":"2026-01-01T00:00:00","active":true,"source":"SOLAR1",
                  "proton_speed":600.0,"proton_density":5.0,"proton_temperature":490000},
                 {"time_tag":"2026-01-01T00:02:00","active":true,"source":"SOLAR1",
                  "proton_speed":null,"proton_density":6.0,"proton_temperature":null}]
                """)
        end

        plasma_df = fetch_swpc_plasma(;
            http_get=flaky_plasma_get,
            max_retries=2,
            retry_delay_sec=0,
        )
        @test plasma_calls[] == 2
        # Only the two active rows with finite speed AND density survive (the null-speed row and the
        # inactive ACE row are dropped), and the frame is sorted ascending by time_tag.
        @test plasma_df.speed == [600.0, 610.0]
        @test plasma_df.density == [5.0, 5.5]
        @test 123.0 ∉ plasma_df.speed        # the inactive ACE source never masquerades as primary

        # NOAA's live RTSW endpoint occasionally serializes missing measurements as bare NaN
        # rather than standard-JSON null. The transport must accept the feed, while the existing
        # field guards drop a row whose required speed is non-finite and retain NaN only for the
        # optional temperature field.
        nonfinite_get(url; kwargs...) = (; status=200, body="""
            [{"time_tag":"2026-01-01T00:00:00","active":true,"source":"SOLAR1",
              "proton_speed":400.0,"proton_density":5.0,"proton_temperature":NaN},
             {"time_tag":"2026-01-01T00:01:00","active":true,"source":"SOLAR1",
              "proton_speed":NaN,"proton_density":6.0,"proton_temperature":500000},
             {"time_tag":"2026-01-01T00:02:00","active":true,"source":"SOLAR1",
              "proton_speed":410.0,"proton_density":5.5,"proton_temperature":510000}]
            """)
        nonfinite_df = fetch_swpc_plasma(;
            http_get=nonfinite_get,
            max_retries=1,
            retry_delay_sec=0,
        )
        @test nonfinite_df.speed == [400.0, 410.0]
        @test nonfinite_df.density == [5.0, 5.5]
        @test isnan(nonfinite_df.temperature[1])

        function mag_get(url; kwargs...)
            return (; status=200, body="""
                [{"time_tag":"2026-01-01T00:00:00","active":true,"source":"SOLAR1",
                  "bt":8.31,"bx_gsm":1.0,"by_gsm":2.0,"bz_gsm":-8.0},
                 {"time_tag":"2026-01-01T00:00:00","active":false,"source":"ACE",
                  "bt":77.0,"bx_gsm":9.0,"by_gsm":9.0,"bz_gsm":-1.0}]
                """)
        end

        mag_df = fetch_swpc_mag(; http_get=mag_get)
        @test mag_df.bz_gsm == [-8.0]
        @test mag_df.bt == [8.31]
        @test mag_df.by_gsm == [2.0]
        @test mag_df.bt != [77.0]      # the inactive ACE record is not selected
    end

    @testset "C2: fetch_swpc_dst parses feed and Dst anchoring populates Dst*" begin
        # The live Kyoto Dst product is an array of OBJECTS with ISO-8601 (`T`) timestamps
        # and numeric dst — NOT the header + array-of-arrays format of the plasma/mag feeds.
        # This mock matches the real feed so the parser is actually guarded against it.
        function dst_get(url; kwargs...)
            return (; status=200, body="""
                [{"time_tag":"2026-01-01T00:00:00","dst":-40},
                 {"time_tag":"2026-01-01T01:00:00","dst":-55}]
                """)
        end
        times, dst = fetch_swpc_dst(; http_get=dst_get)
        @test times == [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 1)]
        @test dst == [-40.0, -55.0]

        # A legacy array-of-arrays form (with header, space-separated time) is still tolerated.
        legacy_get(url; kwargs...) = (; status=200, body="""
            [["time_tag","dst"], ["2026-01-01 00:00:00","-40"], ["2026-01-01 01:00:00","-55"]]
            """)
        times2, dst2 = fetch_swpc_dst(; http_get=legacy_get)
        @test times2 == [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 1)]
        @test dst2 == [-40.0, -55.0]

        plasma = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0, 0), DateTime(2026, 1, 1, 0, 30, 0),
                        DateTime(2026, 1, 1, 1, 15, 0), DateTime(2026, 1, 1, 2, 0, 0)],
            density = [5.0, 7.0, 9.0, 9.0],
            speed = [400.0, 420.0, 440.0, 440.0],
            temperature = [1.0e5, 1.1e5, 1.2e5, 1.2e5],
        )
        mag = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0, 0), DateTime(2026, 1, 1, 0, 30, 0),
                        DateTime(2026, 1, 1, 1, 15, 0), DateTime(2026, 1, 1, 2, 0, 0)],
            bx_gsm = [1.0, 1.0, 1.0, 1.0],
            by_gsm = [2.0, 4.0, 8.0, 8.0],
            bz_gsm = [-3.0, -5.0, -7.0, -7.0],
            bt = [3.7, 6.4, 10.7, 10.7],
        )

        # Coarse synthetic fixture (≤2 samples/hour): opt out of the coverage gate.
        # Without Dst the forecaster is unanchored (regression: must stay NaN).
        swd0, _ = fetch_realtime_solar_wind(hours=2; plasma=plasma, mag=mag,
                                            min_hourly_samples=1)
        @test all(isnan, swd0.Dst_star)

        # With observed Dst, the matching hour bins are anchored.
        swd, tags = fetch_realtime_solar_wind(hours=2; plasma=plasma, mag=mag,
                                              dst=(times, dst), min_hourly_samples=1)
        @test swd.Dst[1] == -40.0
        @test swd.Dst[2] == -55.0
        @test swd.Dst_star[1] ≈ -40.0 - 7.26 * sqrt(swd.Pdyn[1]) + 11.0 atol=1e-9
        # A monitor warm-up would now seed dst0 from a real observation, not 0.
        dst0 = NaN
        for i in length(tags):-1:1
            if !isnan(swd.Dst_star[i]); dst0 = swd.Dst_star[i]; break; end
        end
        @test !isnan(dst0)
        @test dst0 != 0.0
    end

    @testset "NEW-1/NEW-2: non-hour-aligned feed start still anchors observed Dst*" begin
        # Live SWPC feeds are 1-min cadence, so the earliest time_tag is
        # generically NOT on the hour (here :17:30). Before flooring the binning
        # grid, the bin starts straddled hour boundaries and never matched the
        # hour-floored Kyoto Dst keys, leaving Dst_star all-NaN. With the floor,
        # at least one bin coincides with a top-of-hour Dst observation.
        plasma = DataFrame(
            time_tag = [
                DateTime(2026, 1, 1, 0, 17, 30),
                DateTime(2026, 1, 1, 0, 47, 30),
                DateTime(2026, 1, 1, 1, 17, 30),
                DateTime(2026, 1, 1, 1, 47, 30),
                DateTime(2026, 1, 1, 2, 17, 30),
            ],
            density = [5.0, 7.0, 9.0, 11.0, 12.0],
            speed = [400.0, 420.0, 440.0, 460.0, 470.0],
            temperature = [1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.35e5],
        )
        mag = DataFrame(
            time_tag = [
                DateTime(2026, 1, 1, 0, 17, 30),
                DateTime(2026, 1, 1, 0, 47, 30),
                DateTime(2026, 1, 1, 1, 17, 30),
                DateTime(2026, 1, 1, 1, 47, 30),
                DateTime(2026, 1, 1, 2, 17, 30),
            ],
            bx_gsm = [1.0, 1.0, 1.0, 1.0, 1.0],
            by_gsm = [2.0, 4.0, 8.0, 10.0, 11.0],
            bz_gsm = [-3.0, -5.0, -7.0, -9.0, -10.0],
            bt = [3.7, 6.4, 10.7, 13.5, 14.9],
        )
        # Top-of-hour Kyoto Dst (the published cadence).
        dst_times = [DateTime(2026, 1, 1, 1), DateTime(2026, 1, 1, 2)]
        dst_vals = [-40.0, -55.0]

        swd, _ = fetch_realtime_solar_wind(hours=3; plasma=plasma, mag=mag,
                                           dst=(dst_times, dst_vals), min_hourly_samples=1)

        # Mutation guard for NEW-1: without flooring the grid, the hour-floored
        # Dst keys never match any bin start and Dst_star is all-NaN.
        @test count(!isnan, swd.Dst_star) >= 1
    end

    @testset "NF-DATA-02: NaN-Pdyn Dst* fallback uses the quiet-time pressure" begin
        # Density is missing for the whole window, so Pdyn is NaN. Rather than the
        # physically impossible Pdyn=0 (flat Dst+11) fallback, the anchor uses the
        # climatological quiet-time pressure, matching the train-time data_cleaning
        # Dst* definition and the canonical resolve_pdyn/dst_to_dst_star helpers.
        plasma = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0, 0), DateTime(2026, 1, 1, 0, 30, 0),
                        DateTime(2026, 1, 1, 1, 0, 0)],
            density = [NaN, NaN, NaN],
            speed = [400.0, 420.0, 440.0],
            temperature = [1.0e5, 1.1e5, 1.2e5],
        )
        mag = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0, 0), DateTime(2026, 1, 1, 0, 30, 0),
                        DateTime(2026, 1, 1, 1, 0, 0)],
            bx_gsm = [1.0, 1.0, 1.0],
            by_gsm = [2.0, 4.0, 6.0],
            bz_gsm = [-3.0, -5.0, -7.0],
            bt = [3.7, 6.4, 9.2],
        )
        dst_times = [DateTime(2026, 1, 1, 0)]
        dst_vals = [-40.0]

        swd, _ = fetch_realtime_solar_wind(hours=1; plasma=plasma, mag=mag,
                                           dst=(dst_times, dst_vals), min_hourly_samples=1)
        @test isnan(swd.Pdyn[1])           # confirm we exercise the NaN-Pdyn branch
        @test swd.Dst[1] == -40.0
        # With no observed or prior Pdyn, the fallback uses the climatological quiet-time
        # pressure (Dst* = Dst - 7.26√Pdyn0 + 11), not the physically-impossible +11-only
        # (Pdyn=0) baseline that left outage-hour anchors ~10 nT too shallow.
        @test swd.Dst_star[1] ≈ -40.0 - 7.26 * sqrt(SolarSINDy.QUIET_PDYN_NPA) + 11.0 atol=1e-9
    end

    @testset "COV: a sparse hour bin below the coverage floor is not a measured average" begin
        # A feed-brownout hour with 2 of ~60 finite 1-min samples must NOT be
        # served as a measured hourly average. With the coverage gate it is left
        # as a gap (interpolated from full-coverage neighbours), so a 2-minute
        # spike is rejected; with the gate disabled the spike becomes the "average".
        base = DateTime(2026, 1, 1, 0, 0, 0)
        h0 = [base + Minute(m) for m in 0:5:55]                 # 12 samples, hour 0
        h1 = [base + Hour(1) + Minute(1), base + Hour(1) + Minute(2)]  # 2 samples, hour 1 (brownout)
        h2 = [base + Hour(2) + Minute(m) for m in 0:5:55]       # 12 samples, hour 2
        closer = [base + Hour(3)]                               # closes the hour-2 bin
        ptimes = vcat(h0, h1, h2, closer)
        spd = vcat(fill(400.0, length(h0)), fill(900.0, length(h1)),
                   fill(400.0, length(h2)), [400.0])            # hour-1 samples are a 900 km/s spike
        nden = fill(5.0, length(ptimes))
        plasma = DataFrame(time_tag = ptimes, density = nden, speed = spd,
                           temperature = fill(1.0e5, length(ptimes)))
        mag = DataFrame(time_tag = ptimes, bx_gsm = fill(1.0, length(ptimes)),
                        by_gsm = fill(2.0, length(ptimes)), bz_gsm = fill(-5.0, length(ptimes)),
                        bt = fill(5.5, length(ptimes)))

        swd, _ = fetch_realtime_solar_wind(hours=3; plasma=plasma, mag=mag,
                                           min_hourly_samples=10)
        @test !isnan(swd.V[1])                     # hour 0 measured
        @test !isnan(swd.V[3])                     # hour 2 measured
        # Hour 1 (2 samples) is gated out and interpolated from its 400/400 neighbours,
        # so the 900 spike never reaches the served average.
        @test swd.V[2] ≈ 400.0 atol=1e-9

        # Mutation guard: with the gate disabled the 2-sample spike IS served.
        swd1, _ = fetch_realtime_solar_wind(hours=3; plasma=plasma, mag=mag,
                                            min_hourly_samples=1)
        @test swd1.V[2] ≈ 900.0 atol=1e-9
    end

    @testset "freshness requires a coverage-qualified recent hour" begin
        base = DateTime(2026, 1, 1)
        covered = [base + Minute(5m) for m in 0:11]
        brownout = base + Hour(5) + Minute(59)
        times = vcat(reverse(covered), [brownout])
        plasma = DataFrame(
            time_tag=times, density=fill(5.0, length(times)),
            speed=fill(400.0, length(times)),
            temperature=fill(1.0e5, length(times)),
        )
        mag = DataFrame(
            time_tag=times, bx_gsm=fill(1.0, length(times)),
            by_gsm=fill(2.0, length(times)), bz_gsm=fill(-5.0, length(times)),
            bt=fill(5.5, length(times)),
        )
        swd, _, fresh = fetch_realtime_solar_wind(
            hours=6, plasma=plasma, mag=mag, min_hourly_samples=10,
        )
        @test fresh == base + Minute(55)
        @test findlast(isfinite.(swd.V) .& isfinite.(swd.Bz)) == 1
        @test DateTime(2026, 1, 1, 6) - fresh > Hour(5)
    end

    @testset "SENT-1: fetch_swpc_dst rejects a numeric fill sentinel" begin
        # A 9999-type fill value in the Kyoto feed must not survive as a real Dst.
        sentinel_get(url; kwargs...) = (; status=200, body="""
            [{"time_tag":"2026-01-01T00:00:00","dst":-40},
             {"time_tag":"2026-01-01T01:00:00","dst":9999},
             {"time_tag":"2026-01-01T02:00:00","dst":-55}]
            """)
        times, dst = fetch_swpc_dst(; http_get=sentinel_get)
        @test times == [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 2)]  # the 01:00 fill is dropped
        @test dst == [-40.0, -55.0]
        @test !any(v -> abs(v) > 9000, dst)
    end

    @testset "SENT-2: sentinel Dst is never anchored into the forecast bins" begin
        # _hourly_dst_lookup drops out-of-range fill values, and end-to-end a
        # sentinel Dst tuple leaves the forecaster unanchored (NaN), not pinned high.
        lookup = SolarSINDy._hourly_dst_lookup(
            [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 1)], [-40.0, 9999.0])
        @test lookup[DateTime(2026, 1, 1, 0)] == -40.0
        @test !haskey(lookup, DateTime(2026, 1, 1, 1))     # sentinel not anchored
        @test_throws DimensionMismatch SolarSINDy._hourly_dst_lookup(
            [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 1)], [-40.0],
        )
        @test_throws DimensionMismatch SolarSINDy._hourly_dst_lookup(
            [DateTime(2026, 1, 1, 0)], [-40.0, -50.0],
        )

        plasma = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0), DateTime(2026, 1, 1, 0, 30),
                        DateTime(2026, 1, 1, 1, 0)],
            density = [5.0, 6.0, 7.0], speed = [400.0, 410.0, 420.0],
            temperature = [1.0e5, 1.0e5, 1.0e5])
        mag = DataFrame(
            time_tag = [DateTime(2026, 1, 1, 0, 0), DateTime(2026, 1, 1, 0, 30),
                        DateTime(2026, 1, 1, 1, 0)],
            bx_gsm = [1.0, 1.0, 1.0], by_gsm = [2.0, 3.0, 4.0],
            bz_gsm = [-3.0, -4.0, -5.0], bt = [3.7, 5.0, 6.4])
        swd, _ = fetch_realtime_solar_wind(hours=1; plasma=plasma, mag=mag,
                                           dst=([DateTime(2026, 1, 1, 0)], [9999.0]),
                                           min_hourly_samples=1)
        @test all(isnan, swd.Dst)          # the 9999 was rejected, not anchored
        @test all(isnan, swd.Dst_star)
    end

    @testset "SCHEMA: a renamed RTSW field fails closed (no silent wrong physics)" begin
        # Named-key parsing means a reordered column is harmless, but a renamed/removed
        # field yields no valid rows and must raise rather than ingest garbage.
        bad_get(url; kwargs...) = (; status=200, body="""
            [{"time_tag":"2026-01-01T00:00:00","active":true,"speed":610.0,"density":5.5},
             {"time_tag":"2026-01-01T00:01:00","active":true,"speed":611.0,"density":5.6}]
            """)   # proton_speed/proton_density renamed to speed/density
        @test_throws ErrorException fetch_swpc_plasma(; http_get=bad_get,
                                                        max_retries=1, retry_delay_sec=0.0)
    end

    @testset "M1: forecast stepping rejects duplicate or skipped model time" begin
        # The state transition itself now enforces the hourly clock contract, so
        # callers cannot accidentally compound a repeated bin or compress a gap.
        mktempdir() do tmp
            coef_path = joinpath(tmp, "coefficients.csv")
            ens_path = joinpath(tmp, "ensemble.csv")
            # Strong southward driver pushes Dst* down each step.
            CSV.write(coef_path, DataFrame(term = ["Bs", "Dst_star"],
                                           coefficient = [-2.0, -0.05]))
            CSV.write(ens_path, DataFrame(term = ["Bs", "Dst_star"],
                                          inclusion_prob = [0.95, 0.99],
                                          ci_025 = [-2.1, -0.06], ci_975 = [-1.9, -0.04]))
            t = DateTime(2026, 1, 1, 0)
            Bz = -15.0; V = 600.0; n = 10.0; Pd = 1.6726e-6 * n * V^2

            # One legitimate hourly step from the anchor.
            s1 = init_forecast(coefficients_csv = coef_path, ensemble_csv = ens_path,
                               t0 = t, dst0 = -50.0)
            r1 = step_forecast!(s1, t + Hour(1), V, Bz, 0.0, n, Pd)
            @test isfinite(r1.dst_predicted)

            @test_throws ArgumentError step_forecast!(s1, t + Hour(1), V, Bz, 0.0, n, Pd)
            @test_throws ArgumentError step_forecast!(s1, t + Hour(3), V, Bz, 0.0, n, Pd)
        end
    end

    @testset "M1b: warm-up uses a past anchor inside the newest contiguous driver block" begin
        times = [DateTime(2026, 1, 1) + Hour(i) for i in 0:5]
        swd = SolarWindData(collect(0.0:5.0),
            [400.0, 410.0, NaN, 430.0, 440.0, 450.0],
            [-2.0, -3.0, NaN, -5.0, -6.0, -7.0], zeros(6), fill(5.0, 6),
            fill(2.0, 6), fill(NaN, 6), [NaN, -90.0, NaN, NaN, -60.0, -55.0])
        start, last, anchor = SolarSINDy._monitor_warmup_window(swd, times)
        @test (start, last, anchor) == (4, 6, 5)
        # The older observation at index 2 is separated by a driver gap and may
        # not initialise the newer block.
        @test anchor != 2
    end

    @testset "NEW-3: live-loop index selection requires finite V AND Bz" begin
        # A window with finite V in the trailing bin but NaN Bz there: selecting
        # on V alone would land on the trailing NaN-Bz bin (forcing Bz=0, a
        # suppressed/under-alarmed storm). The fix selects the last bin where
        # BOTH are finite.
        V = [400.0, 450.0, 500.0]
        Bz = [-15.0, -18.0, NaN]   # strong southward driving, then a mag gap
        idx = SolarSINDy._latest_finite_VBz_idx(V, Bz)
        @test idx == 2                     # not 3 (Bz is NaN there)
        @test !isnan(Bz[idx])

        # All-Bz-NaN window must be skipped (return nothing), as warm-up does.
        @test SolarSINDy._latest_finite_VBz_idx([400.0, 450.0], [NaN, NaN]) === nothing
        @test SolarSINDy._latest_finite_VBz_idx([400.0, Inf], [-5.0, -6.0]) == 1
        @test SolarSINDy._latest_finite_VBz_idx([400.0, 450.0], [-5.0, Inf]) == 1
        @test_throws DimensionMismatch SolarSINDy._latest_finite_VBz_idx([400.0], [-5.0, -6.0])

        # Behavioural consequence: stepping from the last finite-Bz bin keeps the
        # southward-driving signal, whereas the old Bz=0 substitution (what the
        # V-only selection would feed via _safe_val) produces a less-alarmed
        # forecast. Verify the two differ in the alarming direction.
        mktempdir() do tmp
            coef_path = joinpath(tmp, "coefficients.csv")
            ens_path = joinpath(tmp, "ensemble.csv")
            # Active Bs (southward) driver pushes Dst* down.
            CSV.write(coef_path, DataFrame(
                term = ["Bs", "Dst_star"],
                coefficient = [-2.0, -0.05],
            ))
            CSV.write(ens_path, DataFrame(
                term = ["Bs", "Dst_star"],
                inclusion_prob = [0.95, 0.99],
                ci_025 = [-2.2, -0.06],
                ci_975 = [-1.8, -0.04],
            ))
            state_true = init_forecast(coefficients_csv=coef_path,
                                       ensemble_csv=ens_path,
                                       t0=DateTime(2026, 1, 1, 0), dst0=-50.0)
            state_bz0 = init_forecast(coefficients_csv=coef_path,
                                      ensemble_csv=ens_path,
                                      t0=DateTime(2026, 1, 1, 0), dst0=-50.0)
            Pd = 1.6726e-6 * 5.0 * 500.0^2
            r_true = step_forecast!(state_true, DateTime(2026, 1, 1, 1),
                                    500.0, Bz[idx], 0.0, 5.0, Pd)
            r_bz0 = step_forecast!(state_bz0, DateTime(2026, 1, 1, 1),
                                   500.0, 0.0, 0.0, 5.0, Pd)
            # True (southward) forecast must be more storm-like (lower Dst*) than
            # the spurious Bz=0 forecast the buggy selection would have produced.
            @test r_true.dst_predicted < r_bz0.dst_predicted
        end
    end

    @testset "NEW-3b: delayed Dst anchor replays every intervening hour" begin
        lib = build_minimal_library() # [1, Dst_star, V*Bs]
        ξ = [0.0, -0.1, 0.0]
        state = ForecastState(DateTime(2025, 12, 31, 23), -5.0, lib,
                              ξ, repeat(ξ', 5), 1.0, ForecastResult[])
        times = [DateTime(2026, 1, 1) + Hour(h) for h in 0:3]
        swd = SolarWindData(collect(0.0:3.0), fill(400.0, 4), fill(5.0, 4),
                            zeros(4), fill(5.0, 4), fill(2.0, 4),
                            [-100.0, NaN, NaN, NaN], [-100.0, NaN, NaN, NaN])
        result = SolarSINDy._replay_monitor_from_anchor!(state, swd, times, 1, 4)
        # Three hourly Euler updates from -100 with dDst/dt=-0.1*Dst:
        # -100 -> -90 -> -81 -> -72.9. The old one-step path returned -90.
        @test result.dst_predicted ≈ -72.9 atol=1e-12
        @test state.dst_current ≈ -72.9 atol=1e-12
        @test state.t_current == times[4]
        @test length(state.history) == 3

        # Without any Dst observation, a three-bin fetch gap must also take
        # three steps from the existing state instead of only the newest bin.
        free = ForecastState(times[1], -100.0, lib, ξ, repeat(ξ', 5), 1.0,
                             ForecastResult[])
        free_swd = SolarWindData(swd.t, swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn,
                                 fill(NaN, 4), fill(NaN, 4))
        free_result = SolarSINDy._replay_monitor_from_anchor!(free, free_swd,
                                                              times, nothing, 4)
        @test free_result.dst_predicted ≈ -72.9 atol=1e-12
        @test length(free.history) == 3

        # A same-bin observation is an anchor, not another elapsed model hour.
        same = ForecastState(times[1], -10.0, lib, ξ, repeat(ξ', 5), 1.0,
                             ForecastResult[])
        same_result = SolarSINDy._replay_monitor_from_anchor!(same, swd, times, 1, 1)
        @test same_result.dst_predicted == -100.0
        @test same.dst_current == -100.0
        @test length(same.history) == 1

        # Hourly row j contains the driver average for [t[j], t[j+1]). A
        # transition to t[2] must therefore use row 1, never future row 2.
        drive_ξ = [0.0, 0.0, -0.01]
        varying = SolarWindData(
            [0.0, 1.0], [100.0, 200.0], [-1.0, -1.0], zeros(2),
            fill(5.0, 2), fill(2.0, 2), fill(NaN, 2), fill(NaN, 2),
        )
        causal = ForecastState(times[1], 0.0, lib, drive_ξ,
                               repeat(drive_ξ', 5), 1.0, ForecastResult[])
        causal_result = SolarSINDy._replay_monitor_from_anchor!(
            causal, varying, times[1:2], nothing, 2,
        )
        oracle = simulate_sindy(drive_ξ, lib, varying, 1.0; Dst0=0.0)
        @test causal_result.dst_predicted == oracle[2] == -1.0

        unavailable_predecessor = ForecastState(
            times[1] - Hour(1), 0.0, lib, drive_ξ,
            repeat(drive_ξ', 5), 1.0, ForecastResult[],
        )
        @test_throws ArgumentError SolarSINDy._replay_monitor_from_anchor!(
            unavailable_predecessor, varying, times[1:2], nothing, 2,
        )
    end

    @testset "NEW-4: print_status flags a stale feed and not a fresh one" begin
        fr = ForecastResult(DateTime(2026, 1, 1, 0), -30.0, -30.0, -45.0, -15.0, NaN)
        forecast = ForecastResult[]

        # Capture terminal output by redirecting stdout through a temp file.
        capture_status(; data_age, stale) = mktemp() do path, io
            redirect_stdout(io) do
                SolarSINDy.print_status(fr, forecast, nothing, 450.0, -5.0, 5.0;
                                        data_age=data_age, stale=stale)
            end
            flush(io)
            read(path, String)
        end

        # >= 6 h old newest hour: staleness banner present.
        stale_out = capture_status(data_age=Hour(6), stale=true)
        @test occursin("STALE", stale_out)

        # Fresh window: no staleness banner.
        fresh_out = capture_status(data_age=Minute(20), stale=false)
        @test !occursin("STALE", fresh_out)
    end

    @testset "A/D: init_forecast maps coefficient CSVs into deterministic state" begin
        mktempdir() do tmp
            coef_path = joinpath(tmp, "coefficients.csv")
            ens_path = joinpath(tmp, "ensemble.csv")

            CSV.write(coef_path, DataFrame(
                term = ["Bs", "Dst_star"],
                coefficient = [-2.0, -0.125],
            ))
            CSV.write(ens_path, DataFrame(
                term = ["Bs", "Dst_star", "V"],
                inclusion_prob = [0.95, 0.99, 0.50],
                ci_025 = [-2.0, -0.125, 99.0],
                ci_975 = [-2.0, -0.125, 101.0],
            ))

            state = init_forecast(
                coefficients_csv = coef_path,
                ensemble_csv = ens_path,
                t0 = DateTime(2026, 1, 1, 0),
                dst0 = -75.0,
            )

            terms = get_term_names(state.lib)
            bs_idx = findfirst(==("Bs"), terms)
            dst_idx = findfirst(==("Dst_star"), terms)
            v_idx = findfirst(==("V"), terms)

            @test state.dst_current == -75.0
            @test state.ξ_primary[bs_idx] == -2.0
            @test state.ξ_primary[dst_idx] == -0.125
            @test state.ξ_primary[v_idx] == 0.0
            @test all(state.ξ_ensemble[:, bs_idx] .== -2.0)
            @test all(state.ξ_ensemble[:, dst_idx] .== -0.125)
            @test all(state.ξ_ensemble[:, v_idx] .== 0.0)
        end
    end

    @testset "M5b/M5g: ensemble is centered on the deployed point model" begin
        mktempdir() do tmp
            coef_path = joinpath(tmp, "coefficients.csv")
            ens_path = joinpath(tmp, "ensemble.csv")
            # Point model: Bs and an active n*Bs term.
            CSV.write(coef_path, DataFrame(
                term = ["Bs", "n*Bs"],
                coefficient = [-0.7, 0.016],
            ))
            # Ensemble CSV deliberately disagrees: Bs CI is centered on -0.8 (not
            # -0.7), and n*Bs has inclusion 0.002 (< 0.9) — the old code would
            # have centered on -0.8 and dropped n*Bs entirely.
            CSV.write(ens_path, DataFrame(
                term = ["Bs", "n*Bs"],
                inclusion_prob = [0.95, 0.002],
                ci_025 = [-1.0, 0.011],
                ci_975 = [-0.6, 0.021],
            ))

            state = init_forecast(
                coefficients_csv = coef_path,
                ensemble_csv = ens_path,
                t0 = DateTime(2026, 1, 1, 0),
                dst0 = -50.0,
            )
            terms = get_term_names(state.lib)
            bs_idx = findfirst(==("Bs"), terms)
            nbs_idx = findfirst(==("n*Bs"), terms)

            # M5b: ensemble mean tracks ξ_primary (-0.7), NOT the CSV median (-0.8).
            @test mean(state.ξ_ensemble[:, bs_idx]) ≈ -0.7 atol = 0.02
            @test abs(mean(state.ξ_ensemble[:, bs_idx]) - (-0.8)) > 0.05
            @test std(state.ξ_ensemble[:, bs_idx]) > 0.0   # has spread

            # M5g: the active n*Bs term is seeded despite inclusion < 0.9.
            @test all(state.ξ_ensemble[:, nbs_idx] .!= 0.0)
            @test mean(state.ξ_ensemble[:, nbs_idx]) ≈ 0.016 atol = 0.003
        end
    end

    @testset "Phase D: SWPC secondary-vendor fallback" begin
        # Hand-written RTSW-shaped JSON (array of named-key objects, 2 active rows), no encoder dep.
        good_body = Vector{UInt8}(
            "[{\"time_tag\":\"2026-01-01T00:00:00\",\"active\":true,\"source\":\"SOLAR1\"," *
            "\"proton_density\":5.0,\"proton_speed\":400.0,\"proton_temperature\":1.0e5}," *
            "{\"time_tag\":\"2026-01-01T01:00:00\",\"active\":true,\"source\":\"SOLAR1\"," *
            "\"proton_density\":6.0,\"proton_speed\":410.0,\"proton_temperature\":1.1e5}]")
        primary = "https://primary.example/plasma.json"
        secondary = "https://secondary.example/plasma.json"

        # Mock transport: primary always errors, secondary returns good rows.
        function mock_get(u; kwargs...)
            u == secondary || error("primary down")
            return (; status = 200, body = good_body)
        end

        # With a fallback supplied, the secondary vendor rescues the fetch.
        df = fetch_swpc_plasma(; url = primary, fallback_url = secondary,
                                 max_retries = 2, retry_delay_sec = 0.0, http_get = mock_get)
        @test nrow(df) == 2
        @test df.speed[1] ≈ 400.0 atol = 1e-9

        # Without a fallback (default nothing), primary failure still throws — non-breaking.
        @test_throws ErrorException fetch_swpc_plasma(; url = primary, max_retries = 2,
                                                        retry_delay_sec = 0.0, http_get = mock_get)
    end

    @testset "Phase D: shadow-state recovery + feed dead-man" begin
        sentinel = (:loaded,)
        boot = (:bootstrapped,)
        # load succeeds -> use it
        @test recover_shadow_state(() -> sentinel, () -> boot) === sentinel
        # load returns nothing (missing state) -> bootstrap
        @test recover_shadow_state(() -> nothing, () -> boot) === boot
        # load throws (torn/corrupt state) -> bootstrap
        @test recover_shadow_state(() -> error("torn file"), () -> boot) === boot
        @test_throws InterruptException recover_shadow_state(
            () -> throw(InterruptException()), () -> boot,
        )
        @test_throws InterruptException SolarSINDy._fetch_swpc_json(
            "https://example.invalid";
            max_retries=2,
            retry_delay_sec=0.0,
            http_get=(args...; kwargs...) -> throw(InterruptException()),
        )

        # dead-man predicate
        @test feed_deadman_tripped(0) == false
        @test feed_deadman_tripped(DEFAULT_FEED_DEADMAN_THRESHOLD - 1) == false
        @test feed_deadman_tripped(DEFAULT_FEED_DEADMAN_THRESHOLD) == true
        @test feed_deadman_tripped(2; threshold = 2) == true
        @test feed_deadman_tripped(1; threshold = 2) == false
        @test_throws ArgumentError feed_deadman_tripped(1; threshold = 0)
        @test_throws ArgumentError feed_deadman_tripped(-1)
    end

    @testset "M4b: print_status banner uses the caller's alarm_config" begin
        # Custom MODERATE=-40; a ci05=-45 fires a MODERATE alarm, so the status
        # banner must read MODERATE, not the QUIET the default -50 threshold gives.
        config = AlarmConfig(
            Dict(MODERATE => -40.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            true, x -> nothing, 6,
        )
        fr = ForecastResult(DateTime(2026, 1, 1, 0), -45.0, -45.0, -45.0, -30.0, NaN)

        capture(cfg) = mktemp() do path, io
            redirect_stdout(io) do
                SolarSINDy.print_status(fr, ForecastResult[], nothing, 450.0, -5.0, 5.0;
                                        alarm_config = cfg)
            end
            flush(io)
            read(path, String)
        end

        out = capture(config)
        # The banner severity matches what check_alarm would classify for this config.
        fired = classify_severity(config.use_worst_case ? fr.dst_ci_05 : fr.dst_predicted,
                                  config.thresholds)
        @test fired == MODERATE
        @test occursin("Status: MODERATE STORM", out)
        # Mutation guard: the old code hardcoded default thresholds and read QUIET here.
        @test !occursin("Status: QUIET", out)

        # use_worst_case=false must classify on the median/predicted, not ci05.
        config_pred = AlarmConfig(config.thresholds, false, x -> nothing, 6)
        out_pred = capture(config_pred)   # predicted -45 <= -40 -> still MODERATE
        @test occursin("Status: MODERATE STORM", out_pred)
    end

    @testset "M3: _cap_history! bounds the retained forecast history" begin
        mktempdir() do tmp
            coef_path = joinpath(tmp, "coefficients.csv")
            ens_path = joinpath(tmp, "ensemble.csv")
            CSV.write(coef_path, DataFrame(term = ["Bs"], coefficient = [-1.0]))
            CSV.write(ens_path, DataFrame(term = ["Bs"], inclusion_prob = [0.95],
                                          ci_025 = [-1.1], ci_975 = [-0.9]))
            state = init_forecast(coefficients_csv = coef_path, ensemble_csv = ens_path,
                                  t0 = DateTime(2026, 1, 1, 0), dst0 = -20.0)
            for h in 1:50
                push!(state.history,
                      ForecastResult(DateTime(2026, 1, 1, 0) + Hour(h),
                                     Float64(-h), Float64(-h), Float64(-h - 1),
                                     Float64(-h + 1), NaN))
            end
            SolarSINDy._cap_history!(state, 10)
            @test length(state.history) == 10
            # FIFO: oldest dropped, newest retained.
            @test state.history[end].t == DateTime(2026, 1, 1, 0) + Hour(50)
            @test state.history[1].t == DateTime(2026, 1, 1, 0) + Hour(41)
            # Non-positive cap disables trimming.
            SolarSINDy._cap_history!(state, 0)
            @test length(state.history) == 10
        end
    end

    @testset "M3: _rotate_log! caps an append-only log by size" begin
        mktempdir() do tmp
            path = joinpath(tmp, "monitor.log")
            write(path, "x"^2000)

            # Under threshold: no rotation.
            SolarSINDy._rotate_log!(path, 10_000)
            @test isfile(path) && !isfile(path * ".1")

            # Over threshold: current file rotated to .1, active path cleared for reuse.
            SolarSINDy._rotate_log!(path, 1000)
            @test !isfile(path)
            @test isfile(path * ".1") && filesize(path * ".1") == 2000

            # A subsequent append starts a fresh, bounded file.
            open(path, "a") do io; println(io, "new row"); end
            @test isfile(path) && filesize(path) < 2000

            # Disabled rotation (max_bytes <= 0) never rotates.
            write(path, "y"^5000)
            SolarSINDy._rotate_log!(path, 0)
            @test isfile(path) && filesize(path) == 5000
        end
    end

    @testset "RTSW active flag honors explicit non-Bool encodings" begin
        act = SolarSINDy._rtsw_active
        @test act(Dict(:active => true)) == true
        @test act(Dict(:active => false)) == false
        @test act(Dict(:active => "true")) == true
        @test act(Dict(:active => "false")) == false      # explicit NOT active
        @test act(Dict(:active => "FALSE")) == false       # case-insensitive
        @test act(Dict(:active => 1)) == true
        @test act(Dict(:active => 0)) == false             # integer 0 -> inactive
        @test act(Dict(:source => "ACE")) == true          # key absent -> keep (schema safety)
        @test act(Dict(:active => nothing)) == true        # null -> keep
        @test act("not-a-dict") == true                    # non-object -> keep
    end

    @testset "Ballistic L1->Earth propagation shifts driver bins by transit lag" begin
        # Raw L1-time binning pairs a driver bin with the Earth-UT Dst hour of the
        # SAME clock label; the OMNI training convention (and the deployed issuance
        # path) instead pairs Earth hour H with the L1 wind measured ~one transit
        # lag earlier. With V=400 km/s the lag is a full hour, so the Earth 01:00
        # bin must carry the L1 wind measured near 00:00 (strong Bz here), not the
        # 01:00 L1 wind (weak Bz).
        function feed(bz_by_hour)
            ptime = DateTime[]; bz = Float64[]
            for (h, b) in bz_by_hour, m in (10, 30, 50)
                push!(ptime, DateTime(2026, 1, 1, h, m, 0)); push!(bz, b)
            end
            n = length(ptime)
            plasma = DataFrame(time_tag = ptime, density = fill(5.0, n),
                               speed = fill(400.0, n), temperature = fill(1e5, n))
            mag = DataFrame(time_tag = ptime, bx_gsm = fill(1.0, n),
                            by_gsm = fill(2.0, n), bz_gsm = bz, bt = fill(6.0, n))
            return plasma, mag
        end
        plasma, mag = feed([0 => -10.0, 1 => -2.0, 2 => -8.0, 3 => -1.0])
        raw, raw_tags, raw_fresh = fetch_realtime_solar_wind(
            hours = 8; plasma = plasma, mag = mag, min_hourly_samples = 1)
        prop, prop_tags, prop_fresh = fetch_realtime_solar_wind(
            hours = 8; plasma = plasma, mag = mag, min_hourly_samples = 1,
            propagate_l1_to_earth = true)

        raw_at1 = findfirst(==(DateTime(2026, 1, 1, 1, 0, 0)), raw_tags)
        prop_at1 = findfirst(==(DateTime(2026, 1, 1, 1, 0, 0)), prop_tags)
        @test raw_at1 !== nothing && prop_at1 !== nothing
        @test raw.Bz[raw_at1] ≈ -2.0 atol = 1e-12          # raw: 01:00 L1 wind
        @test prop.Bz[prop_at1] ≈ -10.0 atol = 1e-12       # propagated: ~00:00 L1 wind
        # Uniform speed -> every sample shifts by exactly the transit lag.
        @test prop_fresh - raw_fresh == Millisecond(SolarSINDy._l1_transit_ms(400.0))
        @test prop_fresh - raw_fresh == Millisecond(round(Int, (1.5e6 / 400 / 3600) * 3_600_000))
    end

    @testset "M5: monitor poll-cycle glue is testable in isolation" begin
        # --- staleness computation ---
        stale = SolarSINDy._monitor_data_stale
        @test stale(Minute(179), 3.0) == false
        @test stale(Minute(180), 3.0) == true
        @test stale(Minute(200), 3.0) == true
        @test stale(-Minute(200), 3.0) == true             # future-dated -> |age|
        # 2.5 h threshold is 150 min (not the 2 h that Hour(round(Int,2.5)) would give).
        @test stale(Minute(149), 2.5) == false
        @test stale(Minute(150), 2.5) == true

        # --- horizon_seen pruning: keep the current hour, drop the past ---
        t = DateTime(2026, 1, 1, 12)
        seen = Dict(t - Hour(1) => MODERATE, t => INTENSE, t + Hour(1) => MODERATE)
        SolarSINDy._prune_horizon_seen!(seen, t)
        @test !haskey(seen, t - Hour(1))
        @test haskey(seen, t) && haskey(seen, t + Hour(1))

        # --- anti-compounding new-bin gate + one-step advance ---
        lib = build_minimal_library() # [1, Dst_star, V*Bs]
        ξ = [0.0, -0.1, 0.0]
        times = [DateTime(2026, 1, 1) + Hour(h) for h in 0:3]
        swd = SolarWindData(collect(0.0:3.0), fill(400.0, 4), fill(-5.0, 4),
                            zeros(4), fill(5.0, 4), fill(2.0, 4),
                            fill(NaN, 4), fill(NaN, 4))
        cfg = default_alarm_config()
        prior = ForecastResult(times[4], -50.0, -50.0, -55.0, -45.0, NaN)
        seen2 = Dict{DateTime,SolarSINDy.StormSeverity}()

        # Same bin (t_new[latest] == state.t_current): NO advance, forecast reused.
        state_same = ForecastState(times[4], -50.0, lib, ξ, repeat(ξ', 5), 1.0,
                                   ForecastResult[])
        out_same = SolarSINDy._monitor_cycle!(
            state_same, swd, times, 4, 400.0, -5.0, 0.0, 5.0, 2.0;
            forecast_horizon_hr = 3, alarm_config = cfg, history_cap = 2000,
            last_result = prior, last_forecast = ForecastResult[prior],
            last_alarm = nothing, last_alarm_time = DateTime(1970),
            last_horizon_alarm = nothing, last_obs_time = times[4],
            horizon_seen = seen2)
        @test out_same.new_bin == false
        @test state_same.t_current == times[4]             # model time unchanged
        @test out_same.result === prior

        # New bin: exactly one replay step from the preceding driver bin.
        state_new = ForecastState(times[3], -50.0, lib, ξ, repeat(ξ', 5), 1.0,
                                  ForecastResult[])
        out_new = SolarSINDy._monitor_cycle!(
            state_new, swd, times, 4, 400.0, -5.0, 0.0, 5.0, 2.0;
            forecast_horizon_hr = 3, alarm_config = cfg, history_cap = 2000,
            last_result = prior, last_forecast = ForecastResult[prior],
            last_alarm = nothing, last_alarm_time = DateTime(1970),
            last_horizon_alarm = nothing, last_obs_time = times[3],
            horizon_seen = seen2)
        @test out_new.new_bin == true
        @test state_new.t_current == times[4]              # advanced exactly one hour
        @test length(state_new.history) == 1               # one Euler step, not four
        @test out_new.result.dst_predicted ≈ -45.0 atol = 1e-12   # -50 + 0.1*50
        @test length(out_new.forecast) == 3

        # Thread the returned last_alarm_time back into a SECOND new-bin cycle, exactly
        # as run_monitor's loop does (store cyc.last_alarm_time, pass it back next poll).
        # check_alarm returns an AlarmCooldownState, so after the first new-bin advance
        # this field is no longer a bare DateTime; under the old ::DateTime keyword the
        # second call threw TypeError and run_monitor wedged (adv-pkg-code CRITICAL).
        @test out_new.last_alarm_time isa SolarSINDy.AlarmCooldownState
        times5 = [DateTime(2026, 1, 1) + Hour(h) for h in 0:4]
        swd5 = SolarWindData(collect(0.0:4.0), fill(400.0, 5), fill(-5.0, 5),
                             zeros(5), fill(5.0, 5), fill(2.0, 5),
                             fill(NaN, 5), fill(NaN, 5))
        out_new2 = SolarSINDy._monitor_cycle!(
            state_new, swd5, times5, 5, 400.0, -5.0, 0.0, 5.0, 2.0;
            forecast_horizon_hr = 3, alarm_config = cfg, history_cap = 2000,
            last_result = out_new.result, last_forecast = out_new.forecast,
            last_alarm = out_new.last_alarm, last_alarm_time = out_new.last_alarm_time,
            last_horizon_alarm = out_new.last_horizon_alarm,
            last_obs_time = out_new.last_obs_time, horizon_seen = seen2)
        @test out_new2.new_bin == true
        @test state_new.t_current == times5[5]             # advanced a second hour, no wedge
        @test out_new2.last_alarm_time isa SolarSINDy.AlarmCooldownState
    end

    @testset "M5r: run_monitor threads the cooldown state across new-bin cycles" begin
        # Regression for the run_monitor wedge (adv-pkg-code CRITICAL): check_alarm
        # returns an AlarmCooldownState, which run_monitor stores as last_alarm_time and
        # threads straight back into the next _monitor_cycle!. Under the old ::DateTime
        # keyword the second new-bin cycle threw TypeError, so run_monitor's recovery
        # guard warn/re-warmed forever and never issued another forecast. This drives the
        # exact store-and-pass-back discipline over several genuinely new bins.
        lib = build_minimal_library() # [1, Dst_star, V*Bs]
        ξ = [0.0, -0.1, 0.0]
        nbin = 6
        times = [DateTime(2026, 1, 1) + Hour(h) for h in 0:(nbin - 1)]
        # Deep, sustained southward driving so check_alarm crosses a storm tier and the
        # fired-alarm AlarmCooldownState (not only the QUIET-path conversion) is exercised.
        swd = SolarWindData(collect(0.0:(nbin - 1)), fill(600.0, nbin), fill(-20.0, nbin),
                            zeros(nbin), fill(8.0, nbin), fill(4.0, nbin),
                            fill(NaN, nbin), fill(NaN, nbin))
        fired = SolarSINDy.Alarm[]
        cfg = default_alarm_config(callback = a -> push!(fired, a))
        state = ForecastState(times[1], -120.0, lib, ξ, repeat(ξ', 5), 1.0, ForecastResult[])

        # Bootstrap value is a bare DateTime, exactly as run_monitor initialises it.
        last_alarm_time = DateTime(1970)
        last_result = ForecastResult(times[1], -120.0, -120.0, -125.0, -115.0, -120.0)
        last_forecast = ForecastResult[]
        last_alarm = nothing
        last_horizon_alarm = nothing
        last_obs_time = times[1]
        horizon_seen = Dict{DateTime,SolarSINDy.StormSeverity}()

        saw_cooldown_state = false
        for k in 2:nbin
            cyc = SolarSINDy._monitor_cycle!(
                state, swd, times, k, 600.0, -20.0, 0.0, 8.0, 4.0;
                forecast_horizon_hr = 3, alarm_config = cfg, history_cap = 2000,
                last_result = last_result, last_forecast = last_forecast,
                last_alarm = last_alarm, last_alarm_time = last_alarm_time,
                last_horizon_alarm = last_horizon_alarm, last_obs_time = last_obs_time,
                horizon_seen = horizon_seen)
            @test cyc.new_bin == true
            # run_monitor's store step: thread every returned field back into the loop.
            last_result = cyc.result
            last_forecast = cyc.forecast
            last_alarm = cyc.last_alarm
            last_alarm_time = cyc.last_alarm_time
            last_horizon_alarm = cyc.last_horizon_alarm
            last_obs_time = cyc.last_obs_time
            cyc.last_alarm_time isa SolarSINDy.AlarmCooldownState && (saw_cooldown_state = true)
        end
        @test saw_cooldown_state                       # the threaded value is the cooldown state
        @test last_alarm_time isa SolarSINDy.AlarmCooldownState
        @test state.t_current == times[nbin]           # advanced through every bin, never wedged
        @test !isempty(fired)                          # deep driving armed a real alarm
    end

    @testset "M6: unguarded replay bridge is reachable and recoverable" begin
        # The daemon-killing exception (realtime-monitor-01): a state older than the
        # fetch window with no Dst anchor makes the replay bridge raise ArgumentError.
        # _monitor_cycle! propagates it (so run_monitor's guard must catch it), and
        # re-warming rebuilds a valid state from the same window.
        lib = build_minimal_library()
        ξ = [0.0, -0.1, 0.0]
        times = [DateTime(2026, 1, 1) + Hour(h) for h in 0:3]
        # Dst feed outage (all-NaN anchor) + a state that has fallen behind the
        # window => the bridge cannot anchor and raises.
        swd_gap = SolarWindData(collect(0.0:3.0), fill(400.0, 4), fill(-5.0, 4),
                                zeros(4), fill(5.0, 4), fill(2.0, 4),
                                fill(NaN, 4), fill(NaN, 4))
        stranded = ForecastState(times[1] - Hour(1), -20.0, lib, ξ,
                                 repeat(ξ', 5), 1.0, ForecastResult[])
        cfg = default_alarm_config()
        @test_throws ArgumentError SolarSINDy._monitor_cycle!(
            stranded, swd_gap, times, 2, 400.0, -5.0, 0.0, 5.0, 2.0;
            forecast_horizon_hr = 6, alarm_config = cfg, history_cap = 2000,
            last_result = ForecastResult(times[1], -20.0, -20.0, -25.0, -15.0, NaN),
            last_forecast = ForecastResult[], last_alarm = nothing,
            last_alarm_time = DateTime(1970), last_horizon_alarm = nothing,
            last_obs_time = nothing, horizon_seen = Dict{DateTime,SolarSINDy.StormSeverity}())

        # Recovery re-warms from a window that carries an observed Dst* anchor.
        swd_good = SolarWindData(collect(0.0:3.0), fill(400.0, 4), fill(-5.0, 4),
                                 zeros(4), fill(5.0, 4), fill(2.0, 4),
                                 [-30.0, NaN, NaN, NaN], [-30.0, NaN, NaN, NaN])
        mktempdir() do tmp
            coef = joinpath(tmp, "c.csv"); ens = joinpath(tmp, "e.csv")
            CSV.write(coef, DataFrame(term = ["Bs", "Dst_star"], coefficient = [-2.0, -0.05]))
            CSV.write(ens, DataFrame(term = ["Bs", "Dst_star"], inclusion_prob = [0.95, 0.99],
                                     ci_025 = [-2.2, -0.06], ci_975 = [-1.8, -0.04]))
            state, last_result, last_obs_time = SolarSINDy._warmup_and_init_monitor(
                swd_good, times; coefficients_csv = coef, ensemble_csv = ens, history_cap = 2000)
            @test state.t_current == times[4]               # recovered to newest bin
            @test last_obs_time == times[1]                 # anchored on observed Dst*
            @test last_result !== nothing
        end
    end

end
