using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "ensemble forecasts match independent linear-decay recurrences" begin
        library = build_minimal_library()
        primary = [0.0, -0.1, 0.0]
        ensemble = hcat(zeros(5), -collect(0.1:0.1:0.5), zeros(5))
        state = ForecastState(DateTime(2020), -50.0, library, primary, ensemble, 1.0)
        result = forecast_ahead(state, 400.0, -5.0, 0.0, 5.0, 2.0, 6)
        for (h, row) in enumerate(result)
            # Tolerance accounts for six floating-point recurrence multiplications.
            @test row.dst_predicted ≈ -50 * 0.9^h rtol=16eps(Float64)
            @test row.dst_median ≈ -50 * 0.7^h rtol=16eps(Float64)
            @test row.dst_ci_05 ≈ -50 * 0.9^h rtol=16eps(Float64)
            @test row.dst_ci_95 ≈ -50 * 0.5^h rtol=16eps(Float64)
            @test row.t == DateTime(2020) + Hour(h)
        end
        @test state.dst_current == -50.0 && isempty(state.history)
        @test_throws ArgumentError SolarSINDy._clamped_finite_derivative(
            Inf, "ensemble member", 3,
        )
    end
