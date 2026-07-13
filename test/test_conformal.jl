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

    @testset "robustness: sparse disturbed stratum falls back monotone-safe" begin
        rng = MersenneTwister(99)
        # Short-lead quiet (narrow) + long-lead quiet (wide), both well-populated,
        # plus a 3-row long-lead DISTURBED stratum (< min_stratum_n). The pooled
        # global band sits BETWEEN the two quiet bands, so a naive global fallback
        # for the sparse disturbed cell would be NARROWER than the same-lead quiet
        # band — the inversion this fix removes (cf. shipped sidecar: global
        # 13.98 nT < 6 h quiet 16.51 nT). The fallback must instead be the widest of
        # {global, same-lead quiet, shorter-lead disturbed}.
        nq = 600
        pts = zeros(2nq + 3)
        obs = vcat(3.0 .* randn(rng, nq),        # short-lead quiet, narrow
                   18.0 .* randn(rng, nq),       # long-lead quiet, wide
                   [120.0, -110.0, 115.0])       # long-lead disturbed, sparse
        hor = vcat(fill(1.0, nq), fill(6.0, nq), fill(6.0, 3))
        dst = vcat(fill(0.0, nq), fill(0.0, nq), fill(-90.0, 3))
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=20,
                           horizon_edges=[0.0, 3.0, Inf])
        hw_q_long = conformal_halfwidth(cc, 6.0, 0.0)     # long-lead quiet (wide)
        hw_d_long = conformal_halfwidth(cc, 6.0, -90.0)   # sparse long-lead disturbed
        # Monotone-safe: disturbed is never narrower than same-lead quiet or global.
        @test hw_d_long >= hw_q_long
        @test hw_d_long >= cc.global_stratum.half_width
        # The pooled global band is genuinely narrower than the long-lead quiet band
        # here (the inversion), so the fallback must pick the quiet band, not global.
        @test cc.global_stratum.half_width < hw_q_long
        @test hw_d_long == hw_q_long
        # Populated quiet cell is still resolved to itself.
        @test conformal_stratum(cc, 6.0, 0.0).key == :h2_quiet
    end

    @testset "width monotonicity: disturbed ≥ quiet at every lead (sparse disturbed)" begin
        rng = MersenneTwister(1234)
        # Well-populated quiet strata across 4 leads with increasing residual scale;
        # every disturbed stratum is deliberately sparse (< min_stratum_n) so every
        # disturbed query takes the monotone-safe fallback. Assert the module's own
        # premise — disturbed intervals are never narrower than same-lead quiet.
        leads = [1.0, 2.0, 3.0, 6.0]
        scales = [4.0, 6.0, 9.0, 15.0]
        pts = Float64[]; obs = Float64[]; hor = Float64[]; dst = Float64[]
        for (h, σ) in zip(leads, scales)
            append!(pts, zeros(500)); append!(obs, σ .* randn(rng, 500))
            append!(hor, fill(h, 500)); append!(dst, zeros(500))
            # 3 disturbed rows per lead (sparse), residuals near the quiet scale.
            append!(pts, zeros(3)); append!(obs, σ .* randn(rng, 3))
            append!(hor, fill(h, 3)); append!(dst, fill(-90.0, 3))
        end
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=20)
        for h in leads
            @test conformal_halfwidth(cc, h, -90.0) >= conformal_halfwidth(cc, h, 0.0)
            @test conformal_halfwidth(cc, h, -90.0) >= cc.global_stratum.half_width
        end
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

    @testset "ACI: alpha-update mechanics (hand-checked)" begin
        ac = init_adaptive_conformal(; target_coverage=0.90, gamma=0.1, warmup=0)
        # Seed history so the first step is not the empty/Inf warmup case.
        append!(ac.history, fill(5.0, 30))
        a0 = ac.alpha_t
        @test a0 ≈ 0.10 atol = 1e-12
        # A miss (obs far outside) lowers α by γ(1-α*): 0.10 + 0.1(0.10-1) = 0.01.
        s_miss = adaptive_conformal_step!(ac, 0.0, 1000.0)
        @test s_miss.covered == false
        @test ac.alpha_t ≈ 0.01 atol = 1e-9
        # A hit raises α by γ·α*: 0.01 + 0.1(0.10) = 0.02.
        s_hit = adaptive_conformal_step!(ac, 0.0, 0.0)
        @test s_hit.covered == true
        @test ac.alpha_t ≈ 0.02 atol = 1e-9
    end

    @testset "ACI: stationary stream tracks nominal coverage" begin
        rng = MersenneTwister(31)
        n = 3000; σ = 7.0
        pts = zeros(n); obs = σ .* randn(rng, n)
        r = run_adaptive_conformal(pts, obs; target_coverage=0.90, gamma=0.02, warmup=50)
        @test r.coverage >= 0.87
        @test r.coverage <= 0.93
    end

    @testset "ACI closes the gap under distribution shift (the fix)" begin
        _in(iv, y) = min(iv[1], iv[2]) <= y <= max(iv[1], iv[2])
        rng = MersenneTwister(20260613)
        n = 1500
        # Residual scale TRIPLES at the midpoint (non-exchangeable stream).
        σ = [k < n ÷ 2 ? 4.0 : 12.0 for k in 1:n]
        pts = zeros(n)
        obs = [σ[k] * randn(rng) for k in 1:n]
        late = (n ÷ 2 + 1):n

        # Static split conformal calibrated on the early (low-σ) regime.
        cal_early = fit_conformal(pts[1:n÷4], obs[1:n÷4], fill(1.0, n÷4),
                                  fill(0.0, n÷4); coverage=0.90, min_stratum_n=10)
        static_late = mean(_in(conformal_interval(cal_early, pts[k], 1.0, 0.0), obs[k])
                           for k in late)

        # ACI over the full stream; coverage on the shifted (late) window.
        ac = init_adaptive_conformal(; target_coverage=0.90, gamma=0.03, warmup=50)
        aci_hits = 0; aci_tot = 0
        for k in 1:n
            s = adaptive_conformal_step!(ac, pts[k], obs[k])
            if k in late
                aci_tot += 1; s.covered && (aci_hits += 1)
            end
        end
        aci_late = aci_hits / aci_tot

        # Static under-covers badly on the shifted regime; ACI recovers ~nominal.
        @test static_late < 0.80
        @test aci_late >= 0.86
        @test aci_late > static_late + 0.05
    end

    @testset "ACI input validation" begin
        @test_throws ArgumentError init_adaptive_conformal(; target_coverage=1.0)
        @test_throws ArgumentError init_adaptive_conformal(; gamma=0.0)
        @test_throws DimensionMismatch run_adaptive_conformal([1.0, 2.0], [1.0])
    end

    @testset "CF-1: conformal quantile uses finite-sample (n+1) index exactly" begin
        # Discriminating operating point: residuals 1..10, coverage 0.90.
        # k = ceil((10+1)*0.90) = ceil(9.9) = 10 → 10th smallest = 10.0, floor 10/11.
        # A floor() index gives 9.0 and the uncorrected n·coverage index gives 9.0;
        # both fail this exact check, so the assertion pins the (n+1)·ceil scheme.
        hw, cf = SolarSINDy._conformal_quantile(collect(1.0:10.0), 0.90)
        @test hw == 10.0
        @test cf ≈ 10 / 11 atol = 1e-12
    end

    @testset "CF-2: non-finite latest_dst row is retained as :disturbed" begin
        # Doc contract: a NaN latest_dst is NOT dropped; it is assigned :disturbed.
        # Build a fit where every row has NaN latest_dst, so the only occupied
        # stratum is the disturbed cell — proving the rows survived the filter.
        n = 40
        pts = zeros(n); obs = 3.0 .* ones(n); hor = fill(1.0, n)
        dst = fill(NaN, n)
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=5)
        # h-bin for horizon 1.0 with default edges is bin 1 → key :h1_disturbed.
        @test haskey(cc.strata, :h1_disturbed)
        @test !haskey(cc.strata, :h1_quiet)
        @test cc.strata[:h1_disturbed].n == n   # all NaN-dst rows retained
    end

    @testset "CF-3: corrupt calibration artifact is rejected on read" begin
        rng = MersenneTwister(515)
        n = 100
        cc = fit_conformal(zeros(n), 5.0 .* randn(rng, n), fill(1.0, n), fill(0.0, n);
                           coverage=0.90, min_stratum_n=20)
        mktempdir() do tmp
            # Valid round-trip still succeeds (guard does not reject good files).
            good = joinpath(tmp, "good.csv")
            write_conformal_calibration(good, cc)
            @test read_conformal_calibration(good) isa ConformalCalibration

            # Corrupt coverage (meta half_width column holds coverage).
            bad_cov = joinpath(tmp, "bad_cov.csv")
            df = CSV.read(good, DataFrame)
            df.half_width[1] = 1.5                       # coverage > 1
            CSV.write(bad_cov, df)
            @test_throws ArgumentError read_conformal_calibration(bad_cov)

            # Corrupt (descending) horizon edges.
            bad_edges = joinpath(tmp, "bad_edges.csv")
            df2 = CSV.read(good, DataFrame)
            df2.horizon_edges[1] = "0.0;3.0;1.0;Inf"     # not ascending
            CSV.write(bad_edges, df2)
            @test_throws ArgumentError read_conformal_calibration(bad_edges)

            # Corrupt min_stratum_n (< 1).
            bad_min = joinpath(tmp, "bad_min.csv")
            df3 = CSV.read(good, DataFrame)
            df3.min_stratum_n .= 0
            CSV.write(bad_min, df3)
            @test_throws ArgumentError read_conformal_calibration(bad_min)
        end
    end

    @testset "F4: out-of-support horizon falls back to global, not top-bin reuse" begin
        rng = MersenneTwister(404)
        # Fit only on leads {1,2,3,6}; the top bin [4.5,Inf) is calibrated at 6 h
        # but with a much SMALLER residual scale than longer leads would have.
        ngrp = 600
        leads = [1.0, 2.0, 3.0, 6.0]
        scales = [2.0, 3.0, 4.0, 6.0]
        pts = Float64[]; obs = Float64[]; hor = Float64[]; dst = Float64[]
        for (h, σ) in zip(leads, scales)
            append!(pts, zeros(ngrp))
            append!(obs, σ .* randn(rng, ngrp))
            append!(hor, fill(h, ngrp))
            append!(dst, zeros(ngrp))
        end
        # Inflate the global pool's tail so the global band is strictly wider than
        # the 6 h top-bin half-width (operationally: longer leads are worse).
        append!(pts, zeros(40)); append!(obs, fill(200.0, 40))
        append!(hor, fill(6.0, 40)); append!(dst, zeros(40))
        cc = fit_conformal(pts, obs, hor, dst; coverage=0.90, min_stratum_n=50)

        @test cc.max_horizon == 6.0
        hw_topbin = conformal_halfwidth(cc, 6.0, 0.0)          # in-support, top bin
        hw_24h    = conformal_halfwidth(cc, 24.0, 0.0)         # out-of-support
        # The guard must NOT silently reuse the 6 h half-width for 24 h.
        @test hw_24h != hw_topbin
        @test conformal_stratum(cc, 24.0, 0.0).key == :global
        @test hw_24h == cc.global_stratum.half_width
        # In-support queries are unchanged (no number drift): horizon 6.0 with the
        # default edges [0,1.5,2.5,4.5,Inf] lands in bin 4 → :h4_quiet, which has
        # ≥ min_stratum_n points and is therefore the resolved stratum.
        @test conformal_stratum(cc, 6.0, 0.0).key == :h4_quiet
        @test conformal_halfwidth(cc, 6.0, 0.0) == cc.strata[:h4_quiet].half_width
    end

    @testset "NEW-ACI-NAN-1: mid-stream NaN does not poison the ACI stream" begin
        rng = MersenneTwister(20260619)
        n = 600; σ = 6.0
        pts = zeros(n)
        obs = σ .* randn(rng, n)
        # Inject a non-finite observation mid-stream (a data gap / dropout).
        gap = 300
        obs[gap] = NaN
        r = run_adaptive_conformal(pts, obs; target_coverage=0.90, gamma=0.03, warmup=50)
        # No interval, half-width, or alpha may be NaN despite the gap. (The very
        # first warm-up step legitimately uses an Inf band from empty history, so
        # the poison signature to exclude is NaN, not the intended Inf.)
        @test !any(isnan, r.lo)
        @test !any(isnan, r.hi)
        @test !any(isnan, r.alpha)
        @test !isnan(r.coverage)
        # Post-warmup intervals are finite — a single NaN residual would otherwise
        # propagate NaN half-widths for the entire remaining stream.
        @test all(isfinite, r.lo[(gap + 1):n])
        @test all(isfinite, r.hi[(gap + 1):n])
        # Post-gap coverage recovers near nominal (the poisoned stream would NaN).
        post = (gap + 50):n
        post_cov = mean(r.covered[k] for k in post)
        @test 0.80 <= post_cov <= 1.0

        # Direct contract check: a NaN-observation step must not push history or
        # move alpha_t (mirrors the split-path skip-non-finite contract).
        ac = init_adaptive_conformal(; target_coverage=0.90, gamma=0.1, warmup=0)
        append!(ac.history, fill(5.0, 20))
        hlen0 = length(ac.history); a0 = ac.alpha_t
        s = adaptive_conformal_step!(ac, 0.0, NaN)
        @test isfinite(s.half_width)
        @test length(ac.history) == hlen0     # residual NOT pushed
        @test ac.alpha_t == a0                # alpha NOT updated

        # Warm-up path guard: with a non-finite entry already in history, the
        # warm-up band is max(finite history), not a NaN from maximum() over a
        # poisoned vector. (Force the poison directly to also cover the case where
        # the guard is partially reverted.)
        ac2 = init_adaptive_conformal(; target_coverage=0.90, gamma=0.1, warmup=50)
        append!(ac2.history, [3.0, NaN, 7.0])     # mid-warmup, one poisoned entry
        s2 = adaptive_conformal_step!(ac2, 0.0, 1.0)
        @test isfinite(s2.half_width)
        @test s2.half_width == 7.0                # widest FINITE residual
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
