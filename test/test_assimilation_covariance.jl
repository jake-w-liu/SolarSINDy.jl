using Test, SolarSINDy, LinearAlgebra, Dates

    @testset "restored EKF uncertainty must be a covariance" begin
        lib = build_minimal_library()
        good = (V=400.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0)
        for field in (:cov, :Q), damaged in ([-1.0 0.0; 0.0 1.0],
                                             [1.0 2.0; 2.0 1.0],
                                             [1.0 0.0; 0.25 1.0])
            for mutation in (f -> assimilation_predict!(f, good),
                             f -> assimilation_update!(f, 1.0),
                             f -> assimilation_update!(f, NaN))
                f = init_assimilation(lib, zeros(3), [2], 0.0)
                setfield!(f, field, copy(damaged))
                old_mean, old_cov = copy(f.mean), copy(f.cov)
                @test_throws ArgumentError mutation(f)
                @test f.mean == old_mean && f.cov == old_cov
            end
        end
        # Rank-deficient PSD matrices are legitimate covariances.
        f = init_assimilation(lib, zeros(3), [2], 0.0)
        f.cov = [1.0 1.0; 1.0 1.0]
        f.Q .= 0.0
        assimilation_update!(f, 1.0)
        @test f.mean ≈ [0.2, 0.2] rtol=2eps(Float64)
        @test f.cov ≈ fill(0.8, 2, 2) rtol=2eps(Float64)
    end

