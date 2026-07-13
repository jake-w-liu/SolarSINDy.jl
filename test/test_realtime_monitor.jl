using HTTP

@testset "Realtime And Monitor" begin

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

    @testset "M1: unanchored re-stepping of one bin compounds (why run_monitor gates on new bins)" begin
        # Root-cause guard for the poll-cadence bug: step_forecast! integrates a
        # fixed hour of dynamics per call, so re-stepping the SAME hourly bin every
        # poll cycle (12/hour at the 5-min default) free-runs the modeled Dst* ~12x
        # faster than wall clock when no observation re-anchors it. run_monitor now
        # steps only when a new hourly bin appears; this test pins the hazard that
        # gating prevents.
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
            r1 = step_forecast!(s1, t, V, Bz, 0.0, n, Pd)  # unanchored (no dst_observed)

            # Twelve poll cycles re-stepping the SAME bin timestamp, unanchored.
            s12 = init_forecast(coefficients_csv = coef_path, ensemble_csv = ens_path,
                                t0 = t, dst0 = -50.0)
            local r12
            for _ in 1:12
                r12 = step_forecast!(s12, t, V, Bz, 0.0, n, Pd)
            end
            # Compounding: 12 same-bin steps drive Dst* far below one honest step.
            @test r12.dst_predicted < r1.dst_predicted - 20.0
            # The bin timestamp is unchanged across the 12 steps, so the gate
            # `t_new[latest_idx] > state.t_current` is false after the first — the
            # monitor reuses the prior result instead of taking any of these steps.
            @test s12.t_current == t
        end
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

end
