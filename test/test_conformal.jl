using Random
using Statistics

@testset "Conformal UQ" begin

    @testset "finite-sample quantile index and coverage floor" begin
        # 99 residuals 1..99; coverage 0.90 → k = ceil(100*0.9)=90 → 90th smallest = 90.
        res = collect(1.0:99.0)
        hw, cf = SolarSINDy._conformal_quantile(res, 0.90)
        @test hw == 90.0
        @test cf ≈ 90 / 100 atol = 1e-12
        # Tiny sample: n=5, coverage 0.9 → k=ceil(6*0.9)=6 > n → clamp to 5 (the max),
        # honest coverage floor 5/6, never the overstated 0.90.
        hw2, cf2 = SolarSINDy._conformal_quantile([2.0, 4.0, 6.0, 8.0, 10.0], 0.90)
        @test hw2 == 10.0
        @test cf2 ≈ 5 / 6 atol = 1e-12
        # Uses absolute value of signed residuals.
        hw3, _ = SolarSINDy._conformal_quantile([-5.0, 3.0, -1.0], 0.5)
        @test hw3 == 3.0   # |.|=[5,3,1] sorted [1,3,5], k=ceil(4*0.5)=2 → 3.0
    end

    @testset "marginal coverage guarantee holds on held-out data" begin
        rng = MersenneTwister(20260613)
        # Exchangeable residuals: point=0, obs ~ N(0,σ). Calibrate on 2000, test on 4000.
        σ = 8.0
        n_cal, n_test = 2000, 4000
        cal_pts = zeros(n_cal); cal_obs = σ .* randn(rng, n_cal)
        cal_h = fill(1.0, n_cal); cal_d = fill(0.0, n_cal)
        cov_target = 0.90
        cc = fit_conformal(cal_pts, cal_obs, cal_h, cal_d;
                           coverage=cov_target, min_stratum_n=10)
        test_pts = zeros(n_test); test_obs = σ .* randn(rng, n_test)
        test_h = fill(1.0, n_test); test_d = fill(0.0, n_test)
        emp = conformal_coverage(cc, test_pts, test_obs, test_h, test_d)
        # Split-conformal marginal coverage ≈ nominal (within sampling error).
        @test emp >= cov_target - 0.03
        @test emp <= cov_target + 0.03
        # Half-width should be near the analytic 90% abs-normal quantile σ·1.645.
        hw = conformal_halfwidth(cc, 1.0, 0.0)
        @test isapprox(hw, σ * 1.6449, rtol = 0.08)
    end

    @testset "stratification: disturbed wider than quiet, long-lead wider than short" begin
        rng = MersenneTwister(7)
        n = 1500
        pts = zeros(4n); obs = zeros(4n); hor = zeros(4n); dst = zeros(4n)
        # 4 strata: (quiet|disturbed) × (short|long) with increasing residual scale.
        specs = [(:q_s, 0.0, 1.0, 4.0), (:q_l, 0.0, 5.0, 8.0),
                 (:d_s, -80.0, 1.0, 10.0), (:d_l, -80.0, 5.0, 20.0)]
        for (j, (_, d, h, σ)) in enumerate(specs)
            idx = ((j - 1) * n + 1):(j * n)
            obs[idx] = σ .* randn(rng, n)
            hor[idx] .= h
            dst[idx] .= d
        end
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=50,
                           horizon_edges=[0.0, 3.5, Inf], activity_threshold_nt=-30.0)
        hw_quiet_short = conformal_halfwidth(cc, 1.0, 0.0)
        hw_quiet_long  = conformal_halfwidth(cc, 5.0, 0.0)
        hw_dist_short  = conformal_halfwidth(cc, 1.0, -80.0)
        hw_dist_long   = conformal_halfwidth(cc, 5.0, -80.0)
        @test hw_quiet_long > hw_quiet_short      # longer lead → wider
        @test hw_dist_short > hw_quiet_short      # disturbed → wider
        @test hw_dist_long == maximum([hw_quiet_short, hw_quiet_long,
                                       hw_dist_short, hw_dist_long])
    end

    @testset "robustness: sparse stratum falls back to global pool" begin
        rng = MersenneTwister(99)
        # Big quiet pool + a 3-row disturbed stratum (< min_stratum_n).
        n = 400
        pts = zeros(n + 3); obs = vcat(5.0 .* randn(rng, n), [100.0, -90.0, 95.0])
        hor = fill(1.0, n + 3)
        dst = vcat(fill(0.0, n), fill(-90.0, 3))
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=20)
        s_quiet = conformal_stratum(cc, 1.0, 0.0)
        s_dist = conformal_stratum(cc, 1.0, -90.0)
        @test s_quiet.key != :global          # quiet stratum is populated
        @test s_dist.key == :global           # sparse disturbed → global fallback
        @test conformal_halfwidth(cc, 1.0, -90.0) == cc.global_stratum.half_width
    end

    @testset "persistence round-trip preserves all strata" begin
        rng = MersenneTwister(2024)
        n = 600
        pts = zeros(n)
        obs = vcat(4.0 .* randn(rng, 300), 15.0 .* randn(rng, 300))
        hor = vcat(fill(1.0, 300), fill(5.0, 300))
        dst = vcat(fill(0.0, 300), fill(-80.0, 300))
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=20,
                           horizon_edges=[0.0, 3.5, Inf], activity_threshold_nt=-30.0)
        mktempdir() do tmp
            path = joinpath(tmp, "conformal.csv")
            write_conformal_calibration(path, cc)
            rc = read_conformal_calibration(path)
            @test rc.coverage == cc.coverage
            @test rc.horizon_edges == cc.horizon_edges
            @test rc.activity_threshold_nt == cc.activity_threshold_nt
            @test rc.min_stratum_n == cc.min_stratum_n
            @test rc.global_stratum.half_width == cc.global_stratum.half_width
            @test Set(keys(rc.strata)) == Set(keys(cc.strata))
            for k in keys(cc.strata)
                @test rc.strata[k].half_width == cc.strata[k].half_width
                @test rc.strata[k].n == cc.strata[k].n
                @test rc.strata[k].coverage_floor == cc.strata[k].coverage_floor
            end
            # Predictions identical after round-trip.
            @test conformal_halfwidth(rc, 5.0, -80.0) == conformal_halfwidth(cc, 5.0, -80.0)
        end
    end

    @testset "input validation and degenerate cases" begin
        @test_throws ArgumentError fit_conformal([1.0], [1.0], [1.0], [0.0]; coverage=0.0)
        @test_throws DimensionMismatch fit_conformal([1.0, 2.0], [1.0], [1.0], [0.0])
        @test_throws ArgumentError SolarSINDy._conformal_quantile(Float64[], 0.9)
        # Single calibration point: half-width = that residual, floor = 1/2.
        cc = fit_conformal([0.0], [3.0], [1.0], [0.0]; coverage=0.9, min_stratum_n=1)
        @test conformal_halfwidth(cc, 1.0, 0.0) == 3.0
        @test cc.global_stratum.coverage_floor ≈ 0.5 atol = 1e-12
        # Interval is centered on the point.
        lo, hi = conformal_interval(cc, -50.0, 1.0, 0.0)
        @test lo == -53.0 && hi == -47.0
    end

end
