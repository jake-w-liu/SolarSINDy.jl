using Test
using SolarSINDy
using LinearAlgebra
using Statistics
using Random
using CSV
using DataFrames

@testset verbose=true "SolarSINDy.jl" begin

    @testset "Utils" begin
        # Test numerical derivative on known function: x = t² → dx/dt = 2t
        t = collect(0.0:0.01:1.0)
        x = t.^2
        dx = numerical_derivative(x, 0.01)
        dx_exact = 2.0 .* t
        # Interior points should be very accurate (central diff is O(h²))
        @test maximum(abs.(dx[5:end-4] .- dx_exact[5:end-4])) < 1e-4

        # Test smoothing
        x_noisy = sin.(t) .+ 0.01 .* randn(MersenneTwister(1), length(t))
        x_smooth = smooth_moving_average(x_noisy, 5)
        @test length(x_smooth) == length(x_noisy)

        # Test pressure correction
        dst = [-50.0, -100.0, -30.0]
        pdyn = [4.0, 9.0, 1.0]
        dst_star = pressure_correct_dst(dst, pdyn)
        @test length(dst_star) == 3
        @test dst_star[1] ≈ -50.0 - 7.26 * 2.0 + 11.0

        # Test halfwave rectify
        bz = [5.0, -3.0, 0.0, -10.0]
        bs = halfwave_rectify(bz)
        @test bs == [0.0, 3.0, 0.0, 10.0]
    end

    @testset "Library" begin
        lib = build_solar_wind_library()
        @test length(lib) > 10  # Should have many terms

        # Test with dummy data
        n_pts = 100
        data = Dict{String,Vector{Float64}}(
            "V" => 400.0 .* ones(n_pts),
            "Bs" => 5.0 .* ones(n_pts),
            "Bz" => -5.0 .* ones(n_pts),
            "By" => 2.0 .* ones(n_pts),
            "n" => 5.0 .* ones(n_pts),
            "Pdyn" => 3.0 .* ones(n_pts),
            "Dst_star" => -50.0 .* ones(n_pts),
            "theta_c" => π .* ones(n_pts),
            "BT" => sqrt(29.0) .* ones(n_pts)
        )
        Θ = evaluate_library(lib, data)
        @test size(Θ) == (n_pts, length(lib))
        @test all(isfinite, Θ)

        # Minimal library
        mlib = build_minimal_library()
        @test length(mlib) == 3
    end

    @testset "STLSQ" begin
        # Known sparse system: y = 2x₁ + 3x₃ (x₂ coefficient is 0)
        rng = MersenneTwister(42)
        n = 200
        X = randn(rng, n, 5)
        ξ_true = [2.0, 0.0, 3.0, 0.0, 0.0]
        y = X * ξ_true

        ξ = stlsq(X, y; λ=0.5)
        @test abs(ξ[1] - 2.0) < 0.01
        @test abs(ξ[2]) < 1e-10
        @test abs(ξ[3] - 3.0) < 0.01
        @test abs(ξ[4]) < 1e-10
        @test abs(ξ[5]) < 1e-10

        # With noise
        y_noisy = y .+ 0.1 .* randn(rng, n)
        ξ_noisy = stlsq(X, y_noisy; λ=0.3)
        @test abs(ξ_noisy[1] - 2.0) < 0.2
        @test abs(ξ_noisy[3] - 3.0) < 0.2
    end

    @testset "Baselines" begin
        n = 100
        V = 500.0 .* ones(n)
        Bs = 10.0 .* ones(n)
        Dst_star = -50.0 .* ones(n)

        # Burton
        dDdt = burton_model(V, Bs, Dst_star)
        @test length(dDdt) == n
        @test all(isfinite, dDdt)

        # Simulate Burton
        Dst_sim = simulate_burton(V, Bs, 1.0)
        @test length(Dst_sim) == n
        @test Dst_sim[1] == 0.0  # starts at 0
        @test Dst_sim[end] < 0   # should decrease under southward IMF

        # Newell coupling
        BT = 10.0 .* ones(n)
        theta_c = π .* ones(n)
        phi = newell_coupling(V, BT, theta_c)
        @test all(phi .> 0)

        # O'Brien-McPherron
        dDdt_ob = obrien_mcpherron_model(V, Bs, Dst_star)
        @test length(dDdt_ob) == n
    end

    @testset "Synthetic Data" begin
        swd, event = generate_synthetic_storm(seed=42)
        @test length(swd.t) > 100
        @test event.min_dst < -50  # Should produce a real storm
        @test all(isfinite, swd.V)
        @test all(isfinite, swd.Dst_star)

        # Prepare for SINDy
        data, dDst = prepare_sindy_data(swd, 1.0)
        @test haskey(data, "V")
        @test haskey(data, "Bs")
        @test haskey(data, "Dst_star")
        @test length(dDst) == length(swd.t)

        # Phase identification
        phases = identify_storm_phases(data["Dst_star"], dDst)
        @test all(p -> p ∈ [1, 2, 3], phases)
    end

    @testset "SINDy Discovery on Synthetic Data" begin
        # Generate data from known Burton model, recover with SINDy
        swd, _ = generate_synthetic_storm(seed=42, noise_level=0.02)
        data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=7)

        # Use minimal library (should recover Burton terms)
        lib = build_minimal_library()
        ξ, active, Θ = sindy_discover(data, lib, dDst; λ=0.01)

        # Should find Dst_star (decay) and V*Bs (injection) as active
        @test haskey(active, "Dst_star") || haskey(active, "V*Bs")
        @test length(active) >= 2  # At least decay + injection

        # Prediction quality
        pred = Θ * ξ
        r = cor(pred, dDst)
        @test r > 0.8  # Should fit well
    end

    @testset "SINDy Forward Simulation" begin
        # Generate synthetic storm, discover SINDy model, simulate forward
        swd, _ = generate_synthetic_storm(seed=42, noise_level=0.02)
        data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=7)
        lib = build_minimal_library()
        ξ, _, _ = sindy_discover(data, lib, dDst; λ=0.01)

        # Forward simulate using the exported simulate_sindy
        Dst_sim = simulate_sindy(ξ, lib, swd, 1.0)
        @test length(Dst_sim) == length(swd.t)
        @test Dst_sim[1] == swd.Dst_star[1]
        @test all(isfinite, Dst_sim)
        # State bounds should be respected
        @test all(-2000.0 .<= Dst_sim .<= 50.0)
        # Should produce reasonable reconstruction (PE > 0 = better than mean)
        pe = prediction_efficiency(Dst_sim, swd.Dst_star)
        @test pe > 0.0

        # Test with custom Dst0
        Dst_sim2 = simulate_sindy(ξ, lib, swd, 1.0; Dst0=-10.0)
        @test Dst_sim2[1] == -10.0
    end

    @testset "Metrics" begin
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = [1.1, 1.9, 3.2, 3.8, 5.1]
        ref = [2.0, 2.0, 2.0, 2.0, 2.0]  # mean prediction

        @test rmse(pred, obs) < 0.2
        @test correlation(pred, obs) > 0.99
        @test skill_score(pred, obs, ref) > 0.9
        @test prediction_efficiency(pred, obs) > 0.9

        ms = metrics_summary(pred, obs; name="test")
        @test ms.name == "test"
        @test ms.n_points == 5
    end

    @testset "Phase-0 bugfix regressions" begin
        # m3: numerical_derivative must reject length < 2 instead of BoundsError.
        @test_throws ArgumentError numerical_derivative([5.0], 1.0)
        d = numerical_derivative([1.0, 3.0], 1.0)
        @test d == [2.0, 2.0]

        # m1: correlation on a zero-variance input returns NaN (no error/warning).
        @test isnan(correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        @test correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) ≈ 1.0

        # N-PE-SS-NOGUARD: a zero-variance denominator must flag NaN, not return
        # huge finite garbage from the 1e-20 floor. Constant observed ⇒ ss_tot=0
        # for PE; reference==observed ⇒ mse_ref=0 for the skill score.
        @test isnan(prediction_efficiency([1.0, 2.0, 3.0], fill(2.0, 3)))
        @test isnan(skill_score([1.0, 2.0, 3.0], fill(2.0, 3), fill(2.0, 3)))

        # N-RMSE-NODIMCHECK: paired metrics must reject mismatched lengths rather
        # than silently broadcasting a length-1 input against a length-N input.
        @test_throws DimensionMismatch rmse([1.0, 2.0], [1.0])
        @test_throws DimensionMismatch prediction_efficiency([1.0, 2.0], [1.0])
        @test_throws DimensionMismatch skill_score([1.0, 2.0], [1.0], [1.0])

        # N-SMOOTH-NOGUARD: a window wider than the series silently collapses to a
        # constant; it must throw instead. (Valid odd window ≤ length still works.)
        @test_throws ArgumentError smooth_moving_average([1.0, 2.0, 3.0], 5)
        @test smooth_moving_average([1.0, 2.0, 3.0], 3) ≈ [1.5, 2.0, 2.5]

        # M1: with normalize=true the threshold acts on the NORMALIZED coefficient
        # (term contribution), so a tiny physical coefficient on a large-scale
        # column survives a λ far above its physical magnitude.
        rng = MersenneTwister(7)
        n = 400
        x_small = randn(rng, n)                 # O(1) column
        x_big   = 1.0e5 .* randn(rng, n)        # O(1e5) column
        Θ = hcat(x_small, x_big)
        ξtrue = [3.0, 2.0e-5]                    # big-column physical coef ≪ λ
        y = Θ * ξtrue
        ξ = stlsq(Θ, y; λ=10.0, normalize=true)
        @test abs(ξ[2] - 2.0e-5) < 1e-7          # survives despite |ξ_phys| ≪ λ
        @test abs(ξ[1] - 3.0) < 1e-6
        # With normalize=false the same λ would kill the tiny physical coefficient.
        ξphys = stlsq(Θ, y; λ=10.0, normalize=false)
        @test ξphys[2] == 0.0
    end

    @testset "Wilcoxon signed-rank significance" begin
        # Textbook case: d = 1..10 all positive → W = min(55,0) = 0, n = 10,
        # μ = 27.5, σ² = 96.25; z = (0 − 27.5 + 0.5)/√96.25; p two-sided.
        t = wilcoxon_signed_rank_p(collect(1.0:10.0))
        @test t.n == 10
        @test t.w == 0.0
        @test isapprox(t.z, -27.0 / sqrt(96.25); atol=1e-9)
        @test isapprox(t.p, 0.0059215; atol=1e-6)
        # Zeros dropped; a symmetric set is non-significant.
        @test wilcoxon_signed_rank_p([0.0, 0.0]).n == 0
        @test wilcoxon_signed_rank_p([1.0, -1.0, 2.0, -2.0]).p > 0.5

        # Reproducibility: the per-storm SINDy−O'Brien RMSE pairs reproduce the
        # manuscript's reported p-values exactly (M10 reproducibility gate).
        d = get_data_dir()
        cc = CSV.read(joinpath(d, "cross_cycle_metrics.csv"), DataFrame)
        ho = CSV.read(joinpath(d, "real_holdout_metrics.csv"), DataFrame)
        function _sindy_minus_obrien(df, e)
            sub = e === nothing ? df : df[df.experiment .== e, :]
            s = sort(sub[sub.model .== "SINDy", :], :storm_id)
            o = sort(sub[sub.model .== "OBrienMcP", :], :storm_id)
            @test s.storm_id == o.storm_id
            return Float64.(s.rmse) .- Float64.(o.rmse)
        end
        @test isapprox(wilcoxon_signed_rank_p(_sindy_minus_obrien(cc, "C20-22->C23")).p,
                       1.43e-5; rtol=0.02)
        @test isapprox(wilcoxon_signed_rank_p(_sindy_minus_obrien(cc, "even->odd")).p,
                       1.61e-12; rtol=0.05)
        @test isapprox(wilcoxon_signed_rank_p(_sindy_minus_obrien(cc, "C20-23->C25")).p,
                       0.78; atol=0.01)
        @test isapprox(wilcoxon_signed_rank_p(_sindy_minus_obrien(ho, nothing)).p,
                       0.046; atol=0.002)
    end

    include("test_conformal.jl")
    include("test_assimilation.jl")
    include("test_forecast_alarm.jl")
    include("test_data_pipeline_cleaning.jl")
    include("test_realtime_monitor.jl")
    include("test_v2_broad_replay.jl")

    # Bundled operational dashboard (app/): golden-vector forecaster<->export contract,
    # traversal guard, physical regimes. Network-free (local models + mocks).
    let app_models = joinpath(@__DIR__, "..", "app", "models", "forecaster_FRD.json")
        isfile(app_models) && include(joinpath(@__DIR__, "..", "app", "test", "runtests.jl"))
    end

    include("test_live_forecast_verify.jl")

end
