using Test
using SolarSINDy
using LinearAlgebra
using Statistics
using Random

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

    include("test_forecast_alarm.jl")
    include("test_data_pipeline_cleaning.jl")
    include("test_realtime_monitor.jl")

end
