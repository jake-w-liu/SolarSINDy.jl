using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "finite gap interpolation uses a finite convex combination" begin
        for gap in 1:3, bounds in ((-1e308, 1e308), (1e308, -1e308),
                                  (-floatmax(Float64), floatmax(Float64)))
            before, after = bounds
            samples = vcat(before, fill(NaN, gap), after)
            SolarSINDy._interp_short_gaps!(samples, 3)
            expected = [Float64((1 - BigFloat(j) / (gap + 1)) * BigFloat(before) +
                                BigFloat(j) / (gap + 1) * BigFloat(after))
                        for j in 1:gap]
            @test all(isfinite, samples)
            @test samples[1] == before && samples[end] == after
            @test samples[2:end-1] ≈ expected rtol=8eps(Float64) atol=0
        end
        ordinary = [1.0, NaN, NaN, 4.0]
        SolarSINDy._interp_short_gaps!(ordinary, 3)
        @test ordinary == [1.0, 2.0, 3.0, 4.0]
        longer = [1.0, NaN, NaN, NaN, NaN, 4.0]
        SolarSINDy._interp_short_gaps!(longer, 3)
        @test all(isnan, longer[2:end-1])
    end

