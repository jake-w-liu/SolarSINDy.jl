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

        swd, tags = fetch_realtime_solar_wind(hours=3; plasma=plasma, mag=mag)

        @test length(tags) == 3
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

    @testset "A/D: SWPC parsers retry truncated JSON and preserve magnetic Bt" begin
        plasma_calls = Ref(0)
        function flaky_plasma_get(url; kwargs...)
            plasma_calls[] += 1
            if plasma_calls[] == 1
                return (; status=200, body="""[["time_tag",""")
            end
            return (; status=200, body="""
                [["time_tag","density","speed","temperature"],
                 ["2026-01-01 00:00:00.000","5.5","610.0","500000"]]
                """)
        end

        plasma_df = fetch_swpc_plasma(;
            http_get=flaky_plasma_get,
            max_retries=2,
            retry_delay_sec=0,
        )
        @test plasma_calls[] == 2
        @test plasma_df.speed == [610.0]
        @test plasma_df.density == [5.5]

        function mag_get(url; kwargs...)
            return (; status=200, body="""
                [["time_tag","bx_gsm","by_gsm","bz_gsm","lon_gsm","lat_gsm","bt"],
                 ["2026-01-01 00:00:00.000","1.0","2.0","-8.0","123.0","77.0","8.31"]]
                """)
        end

        mag_df = fetch_swpc_mag(; http_get=mag_get)
        @test mag_df.bz_gsm == [-8.0]
        @test mag_df.bt == [8.31]
        @test mag_df.bt != [77.0]  # Regression guard: column 6 is lat_gsm, not bt.
    end

    @testset "C2: fetch_swpc_dst parses feed and Dst anchoring populates Dst*" begin
        function dst_get(url; kwargs...)
            return (; status=200, body="""
                [["time_tag","dst"],
                 ["2026-01-01 00:00:00","-40"],
                 ["2026-01-01 01:00:00","-55"]]
                """)
        end
        times, dst = fetch_swpc_dst(; http_get=dst_get)
        @test times == [DateTime(2026, 1, 1, 0), DateTime(2026, 1, 1, 1)]
        @test dst == [-40.0, -55.0]

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

        # Without Dst the forecaster is unanchored (regression: must stay NaN).
        swd0, _ = fetch_realtime_solar_wind(hours=2; plasma=plasma, mag=mag)
        @test all(isnan, swd0.Dst_star)

        # With observed Dst, the matching hour bins are anchored.
        swd, tags = fetch_realtime_solar_wind(hours=2; plasma=plasma, mag=mag,
                                              dst=(times, dst))
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

end
