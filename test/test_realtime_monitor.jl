using HTTP

@testset "Realtime And Monitor" begin

    @testset "A/B: fetch_realtime_solar_wind performs hourly averaging and interpolation" begin
        @eval SolarSINDy begin
            function fetch_swpc_plasma(; url::String=SWPC_PLASMA_URL)
                DataFrame(
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
            end

            function fetch_swpc_mag(; url::String=SWPC_MAG_URL)
                DataFrame(
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
                    bt = [0.0, 0.0, 0.0, 0.0, 0.0],
                )
            end
        end

        swd, tags = fetch_realtime_solar_wind(hours=3)

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

end
