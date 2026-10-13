using Test, SolarSINDy, LinearAlgebra

@testset "Static-stack simplex precision and translation invariance" begin
    @testset "large common offset retains constrained symmetry" begin
        expected = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
        for offset in (1e20, -1e20, floatmax(Float64), -floatmax(Float64))
            actual = SolarSINDy._static_stack_project_weights(fill(offset, 6), 0.6)
            @test actual ≈ expected rtol=8eps(Float64) atol=0
            @test all(>=(0.0), actual)
            @test sum(actual) ≈ 1.0 rtol=2eps(Float64)
            @test sum(actual[1:2]) >= 0.6 - 2eps(Float64)
        end
    end
    @testset "finite extreme spread and translation with a known active set" begin
        # The unique largest coordinate receives all mass when its gap exceeds
        # that mass; this remains true when the Float64 spread itself overflows.
        extreme = [floatmax(Float64), -floatmax(Float64), 0.0]
        @test SolarSINDy._static_stack_project_simplex(extreme, 1.0) == [1.0, 0.0, 0.0]
        for offset in (0.0, 2.0^52, -2.0^52)
            values = offset .+ [2.0, 0.0, -2.0]
            @test SolarSINDy._static_stack_project_simplex(values, 1.0) == [1.0, 0.0, 0.0]
        end
        # Independent active-set hand calculation: theta=(2+1-2)/2=0.5.
        @test SolarSINDy._static_stack_project_simplex([2.0, 1.0, -2.0], 2.0) ==
              [1.5, 0.5, 0.0]
        @test SolarSINDy._static_stack_project_simplex(fill(1e20, 4), 2.0) == fill(0.5, 4)
    end
end
