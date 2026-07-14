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
        @test smooth_moving_average(fill(floatmax(Float64), 3), 3) ==
              fill(floatmax(Float64), 3)
        @test all(isnan, smooth_moving_average(fill(NaN, 3), 3))

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

        # Fixed-width integer arithmetic must be promoted before subtracting,
        # squaring, negating, or taking an absolute value.
        integer_span = Float64(BigInt(typemax(Int)) - BigInt(typemin(Int)))
        @test numerical_derivative([typemin(Int), typemax(Int)], 1) ==
              fill(integer_span, 2)
        @test dynamic_pressure(1, typemax(Int)) ==
              SolarSINDy.PROTON_PDYN_COEFF * Float64(typemax(Int))^2
        tiny_density = nextfloat(0.0)
        wide_pressure_oracle = Float64(
            BigFloat(SolarSINDy.PROTON_PDYN_COEFF) * BigFloat(tiny_density) *
            BigFloat(floatmax(Float64))^2,
        )
        @test dynamic_pressure(tiny_density, floatmax(Float64)) ==
              wide_pressure_oracle
        @test isnan(dynamic_pressure(tiny_density, 1.0))
        @test halfwave_rectify([typemin(Int)]) ==
              [Float64(-BigInt(typemin(Int)))]
        @test imf_clock_angle([typemin(Int)], [0])[1] == π / 2
        extreme_swd = SolarWindData(
            collect(0.0:4.0), fill(400.0, 5), fill(1.0, 5),
            fill(floatmax(Float64), 5), fill(5.0, 5), fill(2.0, 5),
            zeros(5), zeros(5),
        )
        extreme_data, extreme_derivative = prepare_sindy_data(
            extreme_swd, 1.0; smooth_window=5,
        )
        @test all(==(floatmax(Float64)), extreme_data["BT"])
        @test all(iszero, extreme_derivative)
        @test_throws ArgumentError numerical_derivative(
            [-floatmax(Float64), floatmax(Float64)], 1.0,
        )
        @test numerical_derivative(
            [0.0, 0.0, floatmax(Float64)], floatmax(Float64),
        )[2] == 0.5
        @test numerical_derivative(
            [-floatmax(Float64), 0.0, floatmax(Float64)], floatmax(Float64),
        )[2] == 1.0
        @test isequal(
            numerical_derivative([0.0, NaN, 2.0, 3.0], 1.0),
            [NaN, 1.0, NaN, 1.0],
        )
        @test_throws ArgumentError numerical_derivative([0.0, Inf], 1.0)
    end

    @testset "Library" begin
        lib = build_solar_wind_library()
        @test length(lib) > 10  # Should have many terms
        @test length(lib) == 20
        @test "Pdyn" in get_term_names(lib)
        @test !("n*V^2" in get_term_names(lib))
        copied_names = get_term_names(lib)
        copied_names[1] = "mutated"
        @test get_term_names(lib)[1] == "1"

        legacy_lib = build_solar_wind_library(include_redundant_n_v2=true)
        @test length(legacy_lib) == 21
        @test "n*V^2" in get_term_names(legacy_lib)

        collapsed_lib = build_solar_wind_library(clock_basis=:collapsed)
        collapsed_names = get_term_names(collapsed_lib)
        @test length(collapsed_lib) == 15
        @test "Newell_d_Φ" in collapsed_names
        @test all(!in(collapsed_names), (
            "sin(θ_c/2)", "sin²(θ_c/2)", "sin⁴(θ_c/2)",
            "sin^(8/3)(θ_c/2)", "V*sin²(θ_c/2)",
        ))
        @test_throws ArgumentError build_solar_wind_library(clock_basis=:unknown)

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

        # Under the package pressure convention the legacy proxy is exactly
        # proportional to Pdyn, so the canonical default must not expose both.
        physical = copy(data)
        physical["V"] = collect(range(300.0, 700.0; length=n_pts))
        physical["n"] = collect(range(2.0, 12.0; length=n_pts))
        physical["Pdyn"] = dynamic_pressure.(physical["n"], physical["V"])
        Θlegacy = evaluate_library(legacy_lib, physical)
        ip = findfirst(==("Pdyn"), get_term_names(legacy_lib))
        inv2 = findfirst(==("n*V^2"), get_term_names(legacy_lib))
        @test Θlegacy[:, ip] ≈ 1.6726e-6 .* Θlegacy[:, inv2]

        # Minimal library
        mlib = build_minimal_library()
        @test length(mlib) == 3

        @test_throws DimensionMismatch CandidateLibrary(["1"], Function[])
        @test_throws ArgumentError CandidateLibrary(String[], Function[])
        @test_throws ArgumentError CandidateLibrary(["x", "x"], Function[d -> d["V"], d -> d["V"]])
        @test_throws ArgumentError CandidateLibrary([""], Function[d -> d["V"]])
        @test_throws ArgumentError CandidateLibrary(["   "], Function[d -> d["V"]])
        @test_throws ArgumentError build_solar_wind_library(max_poly_order=3)
        bad_data = copy(data)
        bad_data["Bs"] = ones(n_pts - 1)
        @test_throws DimensionMismatch evaluate_library(mlib, bad_data)
        @test_throws ArgumentError evaluate_library(mlib, Dict{String,Vector{Float64}}())
        empty_data = Dict(key => Float64[] for key in keys(data))
        @test_throws ArgumentError evaluate_library(lib, empty_data)
        nonfinite_data = copy(data)
        nonfinite_data["V"] = copy(data["V"])
        nonfinite_data["V"][1] = NaN
        @test_throws ArgumentError evaluate_library(lib, nonfinite_data)
        scalar_term = CandidateLibrary(["scalar"], Function[d -> 1.0])
        @test_throws ArgumentError evaluate_library(scalar_term, data)
        short_term = CandidateLibrary(["short"], Function[d -> [1.0]])
        @test_throws DimensionMismatch evaluate_library(short_term, data)
        oversized_term = CandidateLibrary(
            ["oversized"], Function[d -> fill(big"1e400", length(d["V"]))],
        )
        @test_throws ArgumentError evaluate_library(oversized_term, data)
        boolean_term = CandidateLibrary(
            ["boolean"], Function[d -> fill(true, length(d["V"]))],
        )
        @test_throws ArgumentError evaluate_library(boolean_term, data)
        infinite_term = CandidateLibrary(
            ["infinite"], Function[d -> fill(Inf, length(d["V"]))],
        )
        @test_throws ArgumentError evaluate_library(infinite_term, data)
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

        big_x = reshape(BigFloat[1], 1, 1)
        big_y = BigFloat[big"1e-400"]
        big_ξ = stlsq(big_x, big_y; λ=0, normalize=false)
        @test eltype(big_ξ) == BigFloat
        @test big_ξ == big_y
        big_sweep = sweep_lambda(big_x, big_y, BigFloat[0]; normalize=false)
        @test eltype(big_sweep[1].ξ) == BigFloat
        @test big_sweep[1].ξ == big_y
        @test big_sweep[1].rmse == zero(BigFloat)
        stable_sweep = sweep_lambda(zeros(2, 1), [1.0e200, -1.0e200], [0.0])
        @test only(stable_sweep).rmse == 1.0e200

        # A finite column can have a nonrepresentable raw Euclidean norm. It
        # must still be normalized and fitted instead of becoming an all-zero
        # column after division by Inf.
        huge_column = fill(floatmax(Float64), 2, 1)
        huge_ξ = stlsq(huge_column, ones(2); λ=0.0)
        @test huge_ξ == [inv(floatmax(Float64))]
        @test_throws ArgumentError stlsq(
            huge_column, fill(nextfloat(0.0), 2); λ=0.0,
        )
    end

    @testset "STLSQ returns a thresholding fixed point (final-contract)" begin
        # Finding sindy.jl:73 — a single final threshold+resolve is NOT a
        # thresholding fixed point: on the max_iter-exhaustion path the closing
        # refit can leave an active NORMALIZED coefficient below λ. The corrected
        # loop guarantees every returned nonzero normalized |ξ_j·‖Θ_j‖| ≥ λ
        # regardless of how the loop exited. Sweep forced-exhaustion (max_iter=1)
        # instances on near-collinear columns — the surface where the pre-fix code
        # violated the contract on a fraction of instances.
        viol = 0
        for s in 1:2000
            rng = MersenneTwister(s)
            n, p = 60, 6
            b1 = randn(rng, n); b2 = randn(rng, n)
            Θ = hcat(b1, b1 .+ 0.05 .* randn(rng, n), b2,
                     b2 .+ 0.05 .* randn(rng, n), randn(rng, n), randn(rng, n))
            y = Θ * [2.0, -1.5, 3.0, -2.0, 1.0, 0.0] .+ 0.1 .* randn(rng, n)
            λ = 0.8
            col_norms = [norm(Θ[:, j]) for j in 1:p]
            ξ = stlsq(Θ, y; λ=λ, max_iter=1, normalize=true)  # forced exhaustion
            for j in 1:p
                (ξ[j] != 0.0 && abs(ξ[j]) * col_norms[j] < λ - 1e-9) && (viol += 1)
            end
        end
        @test viol == 0
        # The fixed-point loop must not change the converged (in-repo) result: with a
        # generous max_iter every returned nonzero coefficient already satisfies the
        # contract, so the loop exits without altering it.
        rng = MersenneTwister(7)
        n = 300; b1 = randn(rng, n); b2 = randn(rng, n)
        Θ = hcat(b1, b1 .+ 0.03 .* randn(rng, n), b2, randn(rng, n))
        y = Θ * [2.0, -1.0, 3.0, 0.0] .+ 0.05 .* randn(rng, n)
        ξ = stlsq(Θ, y; λ=0.6, max_iter=50, normalize=true)
        @test all(j -> ξ[j] == 0.0 || abs(ξ[j]) * norm(Θ[:, j]) >= 0.6 - 1e-9, 1:4)
    end

    @testset "collinearity_diagnostics flags a canceling collinear block" begin
        # Finding sindy.jl:47 — surface a near-collinear block whose large
        # opposite-sign coefficients cancel to a near-null net contribution.
        rng = MersenneTwister(11)
        n = 500
        base = randn(rng, n)
        c1 = base
        c2 = base .+ 1e-3 .* randn(rng, n)   # near-collinear with c1
        c3 = randn(rng, n)                    # independent, well-identified
        Θ = hcat(c1, c2, c3)
        ξ = [40.0, -40.0, 2.0]                # block (1,2) cancels; column 3 clean
        diag = collinearity_diagnostics(Θ, ξ; groups=[[1, 2], [3]], cond_warn=50.0)
        @test diag.active == [1, 2, 3]
        @test diag.block_cond > 50.0          # active block is ill-conditioned
        @test diag.warn == true
        g12 = diag.groups[1]                  # canceling block: gross ≫ net
        @test g12.cond > 100.0
        @test g12.cancellation > 20.0
        @test g12.net_absmax < 0.1 * g12.gross_absmax
        g3 = diag.groups[2]                   # clean single column: no cancellation
        @test g3.cond ≈ 1.0 atol = 1e-6
        @test g3.cancellation ≈ 1.0 atol = 1e-6
        @test_throws DimensionMismatch collinearity_diagnostics(Θ, [1.0, 2.0])
        @test_throws ArgumentError collinearity_diagnostics(zeros(0, 3), ξ)
        @test_throws ArgumentError collinearity_diagnostics(
            [1.0 NaN; 2.0 3.0], [1.0, 2.0])
        @test_throws ArgumentError collinearity_diagnostics(Θ, ξ; cond_warn=Inf)
        @test_throws ArgumentError collinearity_diagnostics(Θ, ξ; groups=[[1, 4]])
        @test_throws ArgumentError collinearity_diagnostics(Θ, ξ; groups=[[1, 1]])
        wide = [1.0 0.0 1.0; 0.0 1.0 1.0]
        wide_diag = collinearity_diagnostics(wide, ones(3); cond_warn=10.0)
        @test wide_diag.block_cond == Inf
        @test wide_diag.warn


        huge_diag = collinearity_diagnostics(
            fill(floatmax(Float64), 2, 1), [inv(floatmax(Float64))];
            cond_warn=10.0,
        )
        @test huge_diag.block_cond == 1.0
        @test !huge_diag.warn
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
        @test newell_coupling([400.0], [0.0], [0.0]) == [0.0]
        @test newell_coupling([floatmax(Float64)], [0.0], [π]) == [0.0]
        @test newell_coupling([floatmax(Float64)], [1.0], [0.0]) == [0.0]
        full_library = build_solar_wind_library()
        newell_index = only(findall(==("Newell_d_Φ"), get_term_names(full_library)))
        newell_only = SolarSINDy._fast_candidate_library(
            ["Newell_d_Φ"], Function[full_library.funcs[newell_index]],
        )
        zero_newell_data = Dict(
            "V" => [floatmax(Float64)], "By" => [0.0], "Bz" => [0.0],
            "theta_c" => [0.0],
        )
        @test evaluate_library(newell_only, zero_newell_data) ==
              SolarSINDy._evaluate_point(
                  newell_only, 0.0, floatmax(Float64), 0.0, 0.0, 1.0, 1.0,
              ) == zeros(1, 1)
        @test_throws ArgumentError newell_coupling([400.0], [1.0], [-0.1])
        @test_throws ArgumentError newell_coupling([400.0], [1.0], [π + 0.1])

        # O'Brien-McPherron
        dDdt_ob = obrien_mcpherron_model(V, Bs, Dst_star)
        @test length(dDdt_ob) == n

        # Promote integer drivers before V*Bs. The pre-fix integer product
        # wrapped to -2 for this case, disabling both thresholded injections.
        integer_v = [typemax(Int)]
        integer_bs = [2]
        zero_dst = [0]
        expected_vbs = Float64(typemax(Int)) * 2.0
        @test only(burton_model_full(integer_v, integer_bs, zero_dst)) ==
              -5.4e-3 * (expected_vbs - 500.0)
        expected_ec = expected_vbs / 1000.0
        @test only(obrien_mcpherron_model(integer_v, integer_bs, zero_dst)) ==
              -4.4 * (expected_ec - 0.49)
        @test simulate_burton_full(fill(typemax(Int), 2), fill(2, 2), 1.0) ==
              [0.0, -200.0]
        @test simulate_obrien(fill(typemax(Int), 2), fill(2, 2), 1.0) ==
              [0.0, -200.0]

        # Reordered overflow paths are allowed only when the final derivative
        # remains representable; otherwise every model and simulator fails closed.
        @test isfinite(only(burton_model_full(
            [floatmax(Float64)], [2.0], [0.0],
        )))
        @test isfinite(only(obrien_mcpherron_model(
            [floatmax(Float64)], [2.0], [0.0],
        )))
        @test_throws ArgumentError burton_model_full(
            [floatmax(Float64)], [floatmax(Float64)], [0.0],
        )
        @test_throws ArgumentError obrien_mcpherron_model(
            [floatmax(Float64)], [floatmax(Float64)], [0.0],
        )
        @test_throws ArgumentError simulate_burton_full(
            fill(floatmax(Float64), 2), fill(floatmax(Float64), 2), 1.0,
        )
        @test_throws ArgumentError simulate_obrien(
            fill(floatmax(Float64), 2), fill(floatmax(Float64), 2), 1.0,
        )
    end

    @testset "Synthetic Data" begin
        swd, event = generate_synthetic_storm(seed=42)
        @test length(swd.t) > 100
        @test event.min_dst_star < -50  # Should produce a real storm
        @test event.min_dst == event.min_dst_star  # legacy compatibility alias
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

        # With measurement noise disabled, the generator must use the same
        # published Burton injection slope as the baseline implementation.
        exact_swd, _ = generate_synthetic_storm(seed=42, noise_level=0.0)
        exact_bs = halfwave_rectify(exact_swd.Bz)
        @test exact_swd.Dst_star ≈ simulate_burton(exact_swd.V, exact_bs, 1.0;
                                                   α=5.4e-3, τ=7.7, Dst0=0.0)
        @test_throws ArgumentError generate_synthetic_storm(dt=0.0)
        @test_throws ArgumentError generate_synthetic_storm(seed=-1)
        @test_throws ArgumentError generate_synthetic_storm(duration=10.0, dt=3.0)
        @test_throws ArgumentError generate_synthetic_storm(noise_level=-0.1)
        @test_throws ArgumentError generate_multistorm_dataset(n_storms=0)
        @test_throws ArgumentError generate_multistorm_dataset(seed=-1)
        @test_throws DimensionMismatch identify_storm_phases([0.0], [0.0, 1.0])
        @test_throws ArgumentError identify_storm_phases([0.0, NaN], [0.0, 1.0])
        @test_throws ArgumentError identify_storm_phases([0.0, 1.0], [0.0, Inf])
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

        # A user-defined library term must retain generic-function semantics on
        # the optimized point evaluator used by simulation.
        custom = CandidateLibrary(["custom_dst_sq"], Function[d -> d["Dst_star"].^2])
        tiny = SolarWindData([0.0, 1.0, 2.0], fill(400.0, 3), zeros(3), zeros(3),
                             fill(5.0, 3), fill(2.0, 3), zeros(3), zeros(3))
        @test simulate_sindy([0.01], custom, tiny, 1.0; Dst0=2.0) ≈ [2.0, 2.04, 2.081616]
        custom_builtin_name = CandidateLibrary(["V"], Function[d -> 2 .* d["V"]])
        @test evaluate_library(custom_builtin_name, Dict("V" => [400.0]))[1] == 800.0
        @test simulate_sindy([0.001], custom_builtin_name, tiny, 1.0; Dst0=0.0) ==
              [0.0, 0.8, 1.6]
        direct_trusted = CandidateLibrary(
            ["V"], Function[d -> 2 .* d["V"]],
            [SolarSINDy.TERM_V], Val(:trusted_codes),
        )
        @test evaluate_library(direct_trusted, Dict("V" => [400.0]))[1] == 400.0
        @test SolarSINDy._evaluate_point_vector!(
            zeros(1), direct_trusted, 0.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        ) == [400.0]

        # CandidateLibrary's outer struct is immutable, but its parallel vectors
        # remain publicly mutable.  A resized or semantically corrupted vector
        # must fail before the point kernel reaches parallel-vector indexing.
        resized_library = build_minimal_library()
        push!(resized_library.term_codes, SolarSINDy.TERM_ONE)
        @test_throws DimensionMismatch simulate_sindy(
            zeros(length(resized_library)), resized_library, tiny, 1.0;
            Dst0=0.0,
        )
        θ_resized = zeros(length(resized_library))
        @test_throws DimensionMismatch SolarSINDy._evaluate_point_vector!(
            θ_resized, resized_library, 0.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        )
        recoded_library = build_minimal_library()
        recoded_library.term_codes[1] = SolarSINDy.TERM_V
        @test_throws ArgumentError simulate_sindy(
            zeros(length(recoded_library)), recoded_library, tiny, 1.0;
            Dst0=0.0,
        )
        recoded_custom = CandidateLibrary(["V"], Function[d -> 2 .* d["V"]])
        recoded_custom.term_codes[1] = SolarSINDy.TERM_V
        @test_throws ArgumentError evaluate_library(
            recoded_custom, Dict("V" => [400.0]),
        )
        @test_throws ArgumentError SolarSINDy._evaluate_point_vector!(
            zeros(1), recoded_custom, 0.0, 400.0, -5.0, 0.0, 5.0, 2.0,
        )
        renamed_library = build_minimal_library()
        renamed_library.names[1] = "V"
        renamed_library.term_codes[1] = SolarSINDy.TERM_V
        @test_throws ArgumentError simulate_sindy(
            zeros(length(renamed_library)), renamed_library, tiny, 1.0;
            Dst0=0.0,
        )
        redefined_library = build_minimal_library()
        redefined_library.funcs[1] = d -> fill(2.0, length(d["V"]))
        @test_throws ArgumentError evaluate_library(
            redefined_library,
            Dict("V" => [400.0], "Bs" => [5.0], "Dst_star" => [0.0]),
        )

        # The standard library hot loop must not recreate Dicts, one-element
        # arrays, and a design matrix on every step (the prior path allocated
        # about 3 MB for 1,000 rows).
        nperf = 1_000
        perf = SolarWindData(collect(0.0:nperf-1), fill(500.0, nperf),
                             fill(-5.0, nperf), zeros(nperf), fill(5.0, nperf),
                             fill(2.0, nperf), zeros(nperf), zeros(nperf))
        full_lib = build_solar_wind_library()
        zero_ξ = zeros(length(full_lib))
        simulate_sindy(zero_ξ, full_lib, perf, 1.0) # compile before measuring
        @test (@allocated simulate_sindy(zero_ξ, full_lib, perf, 1.0)) < 100_000

        @test_throws DimensionMismatch simulate_sindy(zeros(length(full_lib) - 1), full_lib, perf, 1.0)
        @test_throws ArgumentError simulate_sindy(zero_ξ, full_lib, perf, 0.0)
        @test_throws ArgumentError simulate_sindy(fill(NaN, length(full_lib)), full_lib, perf, 1.0)
        bad_perf = SolarWindData(perf.t, copy(perf.V), perf.Bz, perf.By, perf.n,
                                 perf.Pdyn, perf.Dst, perf.Dst_star)
        bad_perf.V[2] = Inf
        @test_throws ArgumentError simulate_sindy(zero_ξ, full_lib, bad_perf, 1.0)

        # Observations are not integration inputs when an explicit finite anchor
        # is supplied, so missing observation arrays must not reject valid drivers.
        missing_obs = SolarWindData(perf.t, perf.V, perf.Bz, perf.By, perf.n,
                                    perf.Pdyn, fill(NaN, nperf), fill(NaN, nperf))
        @test simulate_sindy(zero_ξ, full_lib, missing_obs, 1.0; Dst0=0.0) ==
              zeros(nperf)
        overflowing = CandidateLibrary(
            ["overflow"], Function[d -> fill(floatmax(Float64), length(d["V"]))],
        )
        @test_throws ArgumentError simulate_sindy([2.0], overflowing, tiny, 1.0;
                                                   Dst0=0.0)
        singleton = SolarWindData(
            [0.0], [400.0], [0.0], [0.0], [5.0], [2.0], [0.0], [0.0],
        )
        @test_throws ArgumentError simulate_sindy(
            [0.0], overflowing, singleton, 1.0; Dst0=big"1e400",
        )
        @test_throws ArgumentError simulate_sindy(
            [big"1e400"], overflowing, tiny, 1.0; Dst0=0.0,
        )
        constant_two = CandidateLibrary(
            ["two"], Function[d -> fill(2.0, length(d["V"]))],
        )
        @test_throws ArgumentError simulate_sindy(
            [1.0], constant_two, tiny, floatmax(Float64); Dst0=0.0,
        )
        for invalid_custom in (
            CandidateLibrary(["scalar"], Function[d -> 1.0]),
            CandidateLibrary(["boolean"], Function[d -> [true]]),
            CandidateLibrary(["complex"], Function[d -> [1.0 + 1.0im]]),
            CandidateLibrary(["wide"], Function[d -> [big"1e400"]]),
        )
            @test_throws ArgumentError simulate_sindy(
                [1.0], invalid_custom, tiny, 1.0; Dst0=0.0,
            )
        end
    end

    @testset "Metrics" begin
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = [1.1, 1.9, 3.2, 3.8, 5.1]
        ref = [2.0, 2.0, 2.0, 2.0, 2.0]  # mean prediction

        @test rmse(pred, obs) < 0.2
        @test mae(pred, obs) ≈ mean(abs.(pred .- obs))
        @test correlation(pred, obs) > 0.99
        @test skill_score(pred, obs, ref) > 0.9
        @test prediction_efficiency(pred, obs) > 0.9

        ms = metrics_summary(pred, obs; name="test")
        @test ms.name == "test"
        @test ms.mae == mae(pred, obs)
        @test ms.n_points == 5

        # Integer arithmetic and naïve sums of squares must not overflow before
        # conversion. Scaled accumulation also keeps a representable Float64
        # RMS finite when the unscaled sum of squares would overflow.
        @test rmse([typemax(Int)], [0]) == Float64(typemax(Int))
        @test mae([typemax(Int)], [0]) == Float64(typemax(Int))
        @test rmse([typemax(Int)], [typemax(Int) - 1]) == 1.0
        @test mae([typemax(Int)], [typemax(Int) - 1]) == 1.0
        @test skill_score(
            [typemax(Int)], [typemax(Int) - 1], [typemax(Int) - 3],
        ) == 0.75
        adjacent = [typemax(Int) - 2, typemax(Int) - 1, typemax(Int)]
        @test correlation(adjacent, adjacent) == 1.0
        @test prediction_efficiency(adjacent, adjacent) == 1.0
        @test rmse([BigFloat(typemax(Int))], [BigFloat(typemax(Int)) - 1]) == 1.0
        @test mae([typemax(Int) // 1], [(typemax(Int) - 1) // 1]) == 1.0
        @test rmse(fill(floatmax(Float64), 2), zeros(2)) == floatmax(Float64)
        @test skill_score([typemax(Int)], [0], [1]) < -1e30
        @test_throws ArgumentError rmse([floatmax(Float64)], [-floatmax(Float64)])
    end

    @testset "Phase-0 bugfix regressions" begin
        # m3: numerical_derivative must reject length < 2 instead of BoundsError.
        @test_throws ArgumentError numerical_derivative([5.0], 1.0)
        @test_throws ArgumentError numerical_derivative([1.0, 2.0], 0.0)
        @test_throws ArgumentError numerical_derivative([1.0, 2.0], -1.0)
        @test_throws ArgumentError numerical_derivative([1.0, 2.0], Inf)
        d = numerical_derivative([1.0, 3.0], 1.0)
        @test d == [2.0, 2.0]
        @test numerical_derivative([1, 3], 2) == [1.0, 1.0]
        @test eltype(numerical_derivative([1, 3], 2)) === Float64

        # m1: correlation on a zero-variance input returns NaN (no error/warning).
        @test isnan(correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        @test correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) ≈ 1.0
        @test_throws DimensionMismatch correlation([1.0, 2.0], [1.0])
        @test_throws ArgumentError correlation(Float64[], Float64[])

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
        @test_throws ArgumentError rmse(Float64[], Float64[])
        @test_throws ArgumentError rmse([1.0, NaN], [1.0, 2.0])
        @test_throws ArgumentError correlation([1.0, Inf], [1.0, 2.0])
        @test_throws ArgumentError skill_score([1.0], [1.0], [NaN])

        # N-SMOOTH-NOGUARD: a window wider than the series silently collapses to a
        # constant; it must throw instead. (Valid odd window ≤ length still works.)
        @test_throws ArgumentError smooth_moving_average([1.0, 2.0, 3.0], 5)
        @test_throws ArgumentError smooth_moving_average([1.0, 2.0, 3.0], 0)
        @test_throws ArgumentError smooth_moving_average([1.0, 2.0, 3.0], 2)
        @test smooth_moving_average([1.0, 2.0, 3.0], 3) ≈ [1.5, 2.0, 2.5]
        @test smooth_moving_average([1, 2, 3], 3) == [1.5, 2.0, 2.5]

        # Physical helper inputs fail closed instead of returning plausible but
        # nonphysical negative pressure or propagating Inf into a model state.
        @test isnan(dynamic_pressure(-1.0, 400.0))
        @test isnan(dynamic_pressure(5.0, -400.0))
        @test isnan(dynamic_pressure(5.0, Inf))
        @test_throws ArgumentError resolve_pdyn(NaN, 2.0, -1)
        @test_throws ArgumentError resolve_pdyn(NaN, 2.0, 1; quiet=-1.0)
        @test_throws ArgumentError resolve_pdyn(big"1e400", 2.0, 0)
        @test_throws ArgumentError resolve_pdyn(NaN, 2.0, 10; quiet=big"1e400")

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

        @test_throws ArgumentError stlsq(Θ, y; λ=-0.1)
        @test_throws ArgumentError stlsq(Θ, y; λ=Inf)
        @test_throws ArgumentError stlsq(Θ, y; max_iter=0)
        @test_throws ArgumentError stlsq(reshape([1.0, Inf], 2, 1), [1.0, 2.0])
        @test_throws DimensionMismatch stlsq(Θ, y[1:end-1])

        lib = build_minimal_library()
        data_small = Dict(
            "V" => [400.0, 410.0], "Bs" => [1.0, 2.0],
            "Dst_star" => [-10.0, -11.0],
        )
        target_small = [0.0, 0.0]
        @test_throws ArgumentError ensemble_sindy(data_small, lib, target_small; n_models=0)
        @test_throws ArgumentError ensemble_sindy(data_small, lib, target_small; λ=-1.0)
        @test_throws ArgumentError ensemble_sindy(data_small, lib, target_small; subsample_frac=0.0)
        @test_throws ArgumentError ensemble_sindy(data_small, lib, [0.0, Inf]; n_models=1)
        @test_throws ArgumentError ensemble_sindy(data_small, lib, target_small;
                                                   n_models=1, seed=-1)
        @test_throws DimensionMismatch ensemble_sindy(data_small, lib, [0.0]; n_models=1)
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
        @test_throws ArgumentError wilcoxon_signed_rank_p([1.0, NaN])
        @test_throws ArgumentError wilcoxon_signed_rank_p([true, false])
        # Stable extreme tail: `1 + erf(z/√2)` cancels to exactly zero here,
        # while the independently evaluated normal-tail oracle is positive.
        extreme = wilcoxon_signed_rank_p(collect(1.0:100.0))
        @test extreme.p > 0.0
        @test isapprox(extreme.p, 3.955911608899558e-18; rtol=1e-12)
        large_n = 1_700_000
        large_moments = SolarSINDy._wilcoxon_null_moments(large_n, (large_n,))
        n_big = BigFloat(large_n)
        variance_oracle = n_big * (n_big + 1) * (2n_big + 1) / 24 -
                          (n_big^3 - n_big) / 48
        @test isfinite(last(large_moments))
        @test last(large_moments) ≈ Float64(variance_oracle) rtol=2e-16
        @test_throws ArgumentError SolarSINDy._wilcoxon_null_moments(3, (1, 1))
        @test_throws ArgumentError SolarSINDy._wilcoxon_null_moments(2, (1.0, 1.0))

        # Legacy diagnostic regression only: these package-local CSVs preserve
        # the submitted global-λ calculation. Canonical revision inference is
        # regenerated by validation/significance_tests.jl and is not asserted
        # against these historical values.
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

    include("test_discovery_provenance.jl")
    include("test_storm_lambda_selection.jl")
    include("test_synthetic_equation_recovery.jl")
    include("test_synthetic_robustness.jl")
    include("test_performance_statistics.jl")
    include("test_conformal.jl")
    include("test_assimilation.jl")
    include("test_forecast_alarm.jl")
    include("test_v2_features_and_ensemble.jl")
    include("test_data_pipeline_cleaning.jl")
    include("test_validation_paths.jl")
    include("test_canonical_provenance.jl")
    include("test_canonical_figure_generation.jl")
    include("test_significance_workflow.jl")
    include("test_real_discovery_helpers.jl")
    include("test_generate_ensemble_draws.jl")
    include("test_phase_discovery.jl")
    include("test_coupled_discovery.jl")
    include("test_realtime_monitor.jl")
    include("test_v2_broad_replay.jl")
    include("test_v2_gscale_replay.jl")
    include("test_noaa_kp_forecast_replay.jl")
    include("test_temerin_dst_archive_replay.jl")
    include("test_external_dst_snapshot_collector.jl")

    # Bundled operational dashboard (app/): golden-vector forecaster<->export contract,
    # traversal guard, physical regimes. Network-free (local models + mocks).
    # app/models/forecaster_FRD.json is a committed part of the package, so a missing artifact means
    # the dashboard contract + traversal-guard security tests would be silently dropped while
    # Pkg.test() still reports success — fail loudly instead of skipping without a record.
    let app_models = joinpath(@__DIR__, "..", "app", "models", "forecaster_FRD.json")
        @testset "Bundled dashboard app suite" begin
            @test isfile(app_models)
            isfile(app_models) && include(joinpath(@__DIR__, "..", "app", "test", "runtests.jl"))
        end
    end

    include("test_live_forecast_verify.jl")

end
