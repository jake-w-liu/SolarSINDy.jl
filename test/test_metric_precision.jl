using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "optimized metrics retain independent expectations and bounded storage" begin
        predicted, observed = [1.0, 2.0, 4.0], [0.0, 2.0, 2.0]
        summary = metrics_summary(predicted, observed)
        @test summary.rmse ≈ sqrt(5 / 3) rtol=4eps(Float64)
        @test summary.mae == 1.0
        @test summary.corr ≈ 2 / sqrt(7) rtol=4eps(Float64)
        @test summary.pe ≈ -0.875 rtol=8eps(Float64)
        @test skill_score(predicted, observed, observed .+ 2) ≈ 7 / 12 rtol=4eps(Float64)
        base = Int64(2)^60
        @test rmse([base], [base + 1]) == 1.0
        @test mae([base], [base + 1]) == 1.0
        for T in (Float16, Float32)
            high = floatmax(T)
            exact_span = 2 * Float64(high)
            @test rmse([high], [-high]) == exact_span
            @test mae([high], [-high]) == exact_span
            @test fit_conformal([high], [-high], [1.0], [0.0]).global_stratum.half_width ==
                  exact_span
        end
        tiny = BigFloat("1e-1000")
        @test_throws ArgumentError rmse(BigFloat[tiny], BigFloat[0])
        @test_throws ArgumentError mae(BigFloat[tiny], BigFloat[0])
        @test_throws ArgumentError fit_conformal(BigFloat[tiny], BigFloat[0], [1.0], [0.0])
        large = zeros(100_000)
        rmse(large, large) # Compile before counting allocations.
        # The residual vector is necessary; two additional full-size conversion
        # copies exceed this cap and would reproduce the measured regression.
        @test (@allocated rmse(large, large)) < 2 * sizeof(large)
    end

