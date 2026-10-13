using Test, SolarSINDy

@testset "Static-stack memory retains a small positive forcing response" begin
    for tau in (1e8, 1e12, 1e16, 1e20, 1e100, floatmax(Float64))
        actual = static_stack_history_memory(0.0, 3.0, tau)
        expected = setprecision(BigFloat, 2048) do
            Float64(3 * (1 - exp(-1 / BigFloat(tau))))
        end
        @test actual > 0.0
        # The final subnormal response loses at most a few Float64 subnormal ULPs.
        @test actual ≈ expected rtol=8eps(Float64) atol=8nextfloat(0.0)
    end
    # Validated artifact timescales retain the old arithmetic exactly.
    for tau in (1.0, 2.0, 4.0, 6.0, 12.0, 24.0, 48.0), memory in (0.0, 2.0, 10.0)
        rho = exp(-1 / tau)
        @test static_stack_history_memory(memory, 3.0, tau) ==
              rho * memory + (1 - rho) * 3.0
    end
end
