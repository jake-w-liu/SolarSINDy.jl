using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "conformal residuals survive exact wide endpoints" begin
        base = Int64(2)^60
        for endpoints in (([base], [base + 1]),
                          (BigFloat[base], BigFloat[base + 1]))
            points, observations = endpoints
            cal = fit_conformal(points, observations, [1.0], [0.0];
                                min_stratum_n=1)
            @test cal.global_stratum.half_width == 1.0
            @test conformal_halfwidth(cal, 1.0, 0.0) == 1.0
            ac = init_adaptive_conformal(warmup=0)
            # Float64 has spacing 256 at 2^60, so neither +1 observation nor
            # +/-1 interval bounds can be faithfully served by the typed ACI API.
            # Split calibration supports the exact residual; ACI must fail closed.
            @test_throws ArgumentError adaptive_conformal_step!(ac, points[1], observations[1])
            @test isempty(ac.history)
            @test_throws ArgumentError run_adaptive_conformal(
                repeat(points, 3), [points[1] + 1, points[1] + 2, points[1] + 1];
                warmup=1,
            )
        end
        # Overflowing fixed-width subtraction must still have the exact span,
        # while ordinary Float64 residuals retain their previous evaluation.
        span = Float64(BigInt(typemax(Int)) - BigInt(typemin(Int)))
        cal = fit_conformal([typemin(Int)], [typemax(Int)], [1.0], [0.0])
        @test cal.global_stratum.half_width == span
        @test fit_conformal([1.25], [3.75], [1.0], [0.0]).global_stratum.half_width == 2.5
    end

