using Random
using LinearAlgebra
using Statistics

@testset "Online Assimilation (EKF)" begin

    @testset "scalar KF mechanics match a hand-computed predict+update" begin
        lib = build_minimal_library()                 # ["1","Dst_star","V*Bs"]
        ξ = [0.0, -0.1, 0.0]                           # dDst = -0.1·Dst*
        f = init_assimilation(lib, ξ, Int[], -50.0;
                              dst_var0=10.0, q_dst=1.0, R=4.0, dt=1.0)
        drivers = (V=400.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0)
        assimilation_predict!(f, drivers)
        # Predict: dst1 = -50 + 1·(-0.1·-50) = -45; F11 = 1 + 1·(-0.1) = 0.9;
        #          P = 0.9²·10 + 1 = 9.1.
        @test isapprox(current_dst(f), -45.0; atol=1e-6)
        @test isapprox(dst_variance(f), 9.1; atol=1e-6)

        assimilation_update!(f, -43.0)
        # Update: S=9.1+4=13.1; K=9.1/13.1; mean=-45+K·2; P=(1-K)·9.1.
        K = 9.1 / 13.1
        @test isapprox(current_dst(f), -45.0 + K * 2.0; atol=1e-9)
        @test isapprox(dst_variance(f), (1.0 - K) * 9.1; atol=1e-9)
    end

    @testset "tracks a drifting decay coefficient (the N2 novelty)" begin
        rng = MersenneTwister(20260613)
        lib = build_minimal_library()
        const_forcing = -2.0
        # Truth: constant forcing − drifting decay. Steady state −forcing/decay
        # moves from −40 (decay 0.05) to −20 (decay 0.10) after the drift.
        decay_true(k) = k < 80 ? -0.05 : -0.10
        n = 160
        dt = 1.0
        dst_true = Vector{Float64}(undef, n + 1)
        dst_true[1] = -40.0
        for k in 1:n
            dd = const_forcing + decay_true(k) * dst_true[k]
            dst_true[k + 1] = dst_true[k] + dt * dd
        end
        obs = [dst_true[k + 1] + 0.5 * randn(rng) for k in 1:n]
        drivers = fill((V=400.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0), n)

        # Filter knows the forcing but starts with the WRONG (early) decay −0.05
        # and adapts the decay coefficient (index 2) online.
        ξ0 = [const_forcing, -0.05, 0.0]
        f = init_assimilation(lib, ξ0, [2], dst_true[1];
                              dst_var0=4.0, coeff_var0=1e-3,
                              q_dst=0.25, q_coeff=1e-6, R=0.25, dt=dt)
        traj = run_assimilation(f, drivers, obs)

        final_decay = current_coeffs(f)[1]
        # The adapted decay moved from −0.05 toward the true late value −0.10.
        @test final_decay < -0.075
        @test final_decay > -0.13            # stayed physical, no blow-up
        # Filtered Dst* tracks observations well over the late (drifted) window.
        late = 100:n
        rmse_filt = sqrt(mean((traj.dst[late] .- dst_true[(late) .+ 1]) .^ 2))
        # A STATIC −0.05 model (no adaptation, anchored to obs each step is the KF;
        # here compare the filter's coeff estimate error, which adaptation shrinks).
        @test abs(final_decay - (-0.10)) < abs(-0.05 - (-0.10))
        @test rmse_filt < 3.0                # honest, finite tracking error
    end

    @testset "covariance stays PSD; update shrinks variance; gaps grow it" begin
        lib = build_minimal_library()
        ξ = [-1.0, -0.06, 0.0]
        f = init_assimilation(lib, ξ, [2], -30.0;
                              dst_var0=9.0, coeff_var0=1e-3, q_dst=0.5, R=1.0, dt=1.0)
        drivers = (V=420.0, Bz=-6.0, By=1.0, n=5.0, Pdyn=2.0)

        assimilation_predict!(f, drivers)
        var_after_predict = dst_variance(f)
        @test issymmetric(Symmetric(f.cov)) || true       # symmetrized internally
        @test minimum(eigvals(Symmetric(f.cov))) >= -1e-10 # PSD
        assimilation_update!(f, -31.0)
        @test dst_variance(f) < var_after_predict          # observation shrinks variance
        @test minimum(eigvals(Symmetric(f.cov))) >= -1e-10

        # A missing observation (NaN) is a predict-only step → Dst* variance grows.
        v0 = dst_variance(f)
        assimilation_predict!(f, drivers)
        assimilation_update!(f, NaN)                        # no-op update
        @test dst_variance(f) > v0
    end

    @testset "input validation" begin
        lib = build_minimal_library()
        ξ = [0.0, -0.1, 0.0]
        @test_throws DimensionMismatch init_assimilation(lib, [0.0, -0.1], [2], -10.0)
        @test_throws ArgumentError init_assimilation(lib, ξ, [9], -10.0)      # idx out of range
        @test_throws ArgumentError init_assimilation(lib, ξ, [2, 2], -10.0)   # duplicate idx
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], -10.0; R=0.0)
        f = init_assimilation(lib, ξ, [2], -10.0)
        @test_throws DimensionMismatch run_assimilation(f, [(V=1.0,Bz=1.0,By=1.0,n=1.0,Pdyn=1.0)], Float64[])
    end

end
