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
        # Symmetry is enforced on the RAW stored matrix by _symmetrize!, so assert
        # it directly (not on a Symmetric() view, which would ignore the stored
        # lower triangle and pass even if symmetrization were a no-op).
        @test issymmetric(f.cov)
        @test minimum(eigvals(Symmetric(f.cov))) >= -1e-10 # PSD (after symmetry holds)
        assimilation_update!(f, -31.0)
        @test issymmetric(f.cov)
        @test dst_variance(f) < var_after_predict          # observation shrinks variance
        @test minimum(eigvals(Symmetric(f.cov))) >= -1e-10

        # A missing observation (NaN) is a predict-only step → Dst* variance grows.
        v0 = dst_variance(f)
        assimilation_predict!(f, drivers)
        assimilation_update!(f, NaN)                        # no-op update
        @test dst_variance(f) > v0
        @test_throws ArgumentError assimilation_update!(f, Inf)
    end

    @testset "raw stored covariance stays symmetric over a multi-coefficient path" begin
        extreme_covariance = fill(floatmax(Float64), 2, 2)
        SolarSINDy._symmetrize!(extreme_covariance)
        @test all(==(floatmax(Float64)), extreme_covariance)
        @test_throws ArgumentError SolarSINDy._symmetrize!(
            [1.0 Inf; 0.0 1.0],
        )
        @test_throws DimensionMismatch SolarSINDy._symmetrize!(zeros(2, 1))

        # Mutation-sensitivity guard for _symmetrize!: with two adapted
        # coefficients the F·P·F' and K·row updates accumulate off-diagonal
        # asymmetry that the symmetrizer must scrub on the RAW matrix every step.
        # If _symmetrize! regressed to a no-op, issymmetric(f.cov) fails here.
        rng = MersenneTwister(20260619)
        lib = build_minimal_library()
        ξ = [-1.5, -0.07, 0.02]
        f = init_assimilation(lib, ξ, [2, 3], -25.0;
                              dst_var0=9.0, coeff_var0=1e-3,
                              q_dst=0.5, q_coeff=1e-6, R=1.0, dt=1.0)
        drivers = (V=430.0, Bz=-7.0, By=2.0, n=6.0, Pdyn=2.5)
        for k in 1:50
            assimilation_predict!(f, drivers)
            obs = current_dst(f) + 0.3 * randn(rng)
            assimilation_update!(f, obs)
            @test issymmetric(f.cov)                         # raw matrix exactly symmetric
            @test minimum(eigvals(Symmetric(f.cov))) >= -1e-10
        end
    end

    @testset "non-finite driver coasts without corrupting the filter (N2 robustness)" begin
        lib = build_minimal_library()
        ξ = [-1.0, -0.06, 0.0]
        f = init_assimilation(lib, ξ, [2], -30.0;
                              dst_var0=9.0, coeff_var0=1e-3, q_dst=0.5, q_coeff=1e-6,
                              R=1.0, dt=1.0)
        good = (V=420.0, Bz=-6.0, By=1.0, n=5.0, Pdyn=2.0)
        assimilation_predict!(f, good)
        assimilation_update!(f, -31.0)
        mean_before = copy(f.mean)
        var_before = dst_variance(f)

        # A NaN in any driver triggers a predict-only coast: mean held, cov grows
        # by Q, and nothing in the state goes non-finite. Without the guard the
        # NaN propagates through ddst and the Jacobian into mean and cov.
        bad = (V=NaN, Bz=-6.0, By=1.0, n=5.0, Pdyn=2.0)
        assimilation_predict!(f, bad)
        @test !any(isnan, f.cov)
        @test isfinite(current_dst(f))
        @test f.mean == mean_before                          # coast: mean unchanged
        @test dst_variance(f) > var_before                   # cov advanced by Q only
        # An Inf driver coasts the same way.
        bad2 = (V=420.0, Bz=-6.0, By=1.0, n=5.0, Pdyn=Inf)
        assimilation_predict!(f, bad2)
        @test !any(isnan, f.cov) && all(isfinite, f.cov)
        @test isfinite(current_dst(f))
    end

    @testset "coefficient stability constraint (opt-in) clamps the adapted decay coefficient" begin
        rng = MersenneTwister(20260622)
        lib = build_minimal_library()                 # ["1","Dst_star","V*Bs"]
        ξ0 = [0.0, -0.1, 0.0]                          # start at strong decay −0.1
        drivers = [(V=400.0, Bz=-3.0, By=0.0, n=5.0, Pdyn=2.0) for _ in 1:200]
        # Truth relaxes ~−0.005·Dst* (effective decay ABOVE the −0.01 cap), so the filter wants to push
        # the adapted decay up past −0.01; the constraint must hold it at −0.01.
        obs = Float64[]; d = -40.0
        for _ in 1:200; d = 0.995 * d + 0.5 * randn(rng); push!(obs, d); end
        cons = init_assimilation(lib, ξ0, [2], obs[1]; q_coeff=1e-3, coeff_bounds=[(-Inf, -0.01)])
        unc  = init_assimilation(lib, ξ0, [2], obs[1]; q_coeff=1e-3)
        cmax = -Inf; umax = -Inf
        for k in 1:199
            assimilation_predict!(cons, drivers[k]); assimilation_update!(cons, obs[k+1])
            assimilation_predict!(unc,  drivers[k]); assimilation_update!(unc,  obs[k+1])
            cmax = max(cmax, current_coeffs(cons)[1]); umax = max(umax, current_coeffs(unc)[1])
        end
        @test cmax <= -0.01 + 1e-9                     # constrained coefficient never leaves the box
        @test umax > -0.01                             # unconstrained DOES exceed it → test is non-vacuous
        @test isfinite(current_dst(cons))
        # initial coefficient outside the box is projected in at construction
        fc = init_assimilation(lib, [0.0, 0.5, 0.0], [2], -10.0; coeff_bounds=[(-Inf, -0.01)])
        @test current_coeffs(fc)[1] <= -0.01 + 1e-12
        # validation: wrong-length bounds and lo>hi are rejected
        @test_throws ArgumentError init_assimilation(lib, ξ0, [2], -10.0; coeff_bounds=[(-1.0, 0.0), (0.0, 1.0)])
        @test_throws ArgumentError init_assimilation(lib, ξ0, [2], -10.0; coeff_bounds=[(0.0, -1.0)])
        @test_throws ArgumentError init_assimilation(
            lib, ξ0, [2], -10.0; coeff_bounds=[(Inf, Inf)],
        )
        @test_throws ArgumentError init_assimilation(
            lib, ξ0, [2], -10.0; coeff_bounds=[(-Inf, -Inf)],
        )
    end

    @testset "input validation" begin
        lib = build_minimal_library()
        ξ = [0.0, -0.1, 0.0]
        @test_throws DimensionMismatch init_assimilation(lib, [0.0, -0.1], [2], -10.0)
        @test_throws ArgumentError init_assimilation(lib, ξ, [9], -10.0)      # idx out of range
        @test_throws ArgumentError init_assimilation(lib, ξ, [2, 2], -10.0)   # duplicate idx
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], -10.0; R=0.0)
        @test_throws ArgumentError init_assimilation(lib, [0.0, Inf, 0.0], [2], -10.0)
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], Inf)
        @test_throws ArgumentError init_assimilation(
            lib, ξ, [2], big"1e400",
        )
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], -10.0; R=Inf)
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], -10.0; q_dst=Inf)
        @test_throws ArgumentError init_assimilation(
            lib, BigFloat[big"1e10000", -0.1, 0.0], [2], -10.0)
        @test_throws ArgumentError init_assimilation(lib, ξ, [2], -10.0;
                                                     coeff_bounds=[(NaN, 1.0)])
        f = init_assimilation(lib, ξ, [2], -10.0)
        @test_throws DimensionMismatch run_assimilation(f, [(V=1.0,Bz=1.0,By=1.0,n=1.0,Pdyn=1.0)], Float64[])

        # The public contract accepts arbitrary finite `Real` values, including
        # integer-valued drivers; they must dispatch through the Float64 kernel.
        fint = init_assimilation(lib, ξ, [2], -10.0)
        assimilation_predict!(fint, (V=400, Bz=-5, By=0, n=5, Pdyn=2))
        @test isfinite(current_dst(fint))

        for bad in ((V=-1.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0),
                    (V=400.0, Bz=-5.0, By=0.0, n=-1.0, Pdyn=2.0),
                    (V=400.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=-1.0))
            fbad = init_assimilation(lib, ξ, [2], -10.0)
            @test_throws ArgumentError assimilation_predict!(fbad, bad)
        end

        foverflow = init_assimilation(lib, ξ, [2], -10.0)
        @test_throws ArgumentError assimilation_predict!(foverflow,
            (V=big"1e10000", Bz=-5, By=0, n=5, Pdyn=2))

        # The filter is public and mutable.  Every public mutation boundary must
        # reject a damaged checkpoint before an `@inbounds` EKF loop can observe
        # inconsistent dimensions; even a NaN no-op update validates first.
        fbad_bounds = init_assimilation(lib, ξ, [2], -10.0)
        empty!(fbad_bounds.bounds)
        @test_throws DimensionMismatch assimilation_update!(fbad_bounds, NaN)

        fbad_cov = init_assimilation(lib, ξ, [2], -10.0)
        fbad_cov.cov = zeros(1, 1)
        @test_throws DimensionMismatch assimilation_predict!(fbad_cov,
            (V=400.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0))

        fbad_q = init_assimilation(lib, ξ, [2], -10.0)
        fbad_q.Q = zeros(1, 1)
        @test_throws DimensionMismatch assimilation_predict!(fbad_q,
            (V=NaN, Bz=-5.0, By=0.0, n=5.0, Pdyn=2.0))

        fbad_mean = init_assimilation(lib, ξ, [2], -10.0;
                                      coeff_bounds=[(-Inf, -0.01)])
        fbad_mean.mean[2] = 0.0
        @test_throws ArgumentError assimilation_update!(fbad_mean, -11.0)
    end

end
