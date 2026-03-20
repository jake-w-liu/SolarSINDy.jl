# Tests for forecast.jl, alarm.jl, and untested baselines/sindy functions

using Dates

@testset "Forecast Module" begin

    @testset "A: Analytical — step_forecast! reproduces simulate_sindy" begin
        # Ground truth: step_forecast! with known ξ should produce the same
        # result as manually computing Θ·ξ and doing Forward Euler.
        # Use Burton-like coefficients: ξ = [0, 0, 0, 0, -1/7.7, ...] for Dst_star decay only.
        lib = build_solar_wind_library()
        n_terms = length(lib)
        ξ = zeros(n_terms)
        # Set only the Dst_star term (index 5: "Dst_star") to -1/7.7 ≈ -0.1299
        dst_idx = findfirst(==("Dst_star"), get_term_names(lib))
        ξ[dst_idx] = -1.0 / 7.7

        # Ensemble: all 500 members have same coefficients (no spread)
        ξ_ens = repeat(ξ', 500)

        state = ForecastState(
            DateTime(2026, 1, 1, 0, 0, 0),
            -100.0,  # Dst* = -100 nT
            lib, ξ, ξ_ens, 1.0,
            ForecastResult[],
        )

        # Step with quiet solar wind (V=400, Bz=+5 northward, no injection)
        result = step_forecast!(state, DateTime(2026, 1, 1, 1, 0, 0),
                                400.0, 5.0, 0.0, 5.0, 2.0)

        # Hand calculation: dDst/dt = -100 × (-1/7.7) = +12.99 nT/hr
        # Dst_next = -100 + 1 × 12.99 = -87.01 nT
        # atol=0.5 because library also has constant term (=0 coeff) and other
        # terms that are zero but get evaluated — exact match depends on all
        # library terms being zero. With only Dst_star active, should be close.
        @test isapprox(result.dst_predicted, -100.0 + (-100.0 * (-1.0/7.7)),
                       atol=0.01)  # Forward Euler with only decay term

        # Since all ensemble members are identical, CI should be very tight
        @test isapprox(result.dst_ci_05, result.dst_predicted, atol=0.1)
        @test isapprox(result.dst_ci_95, result.dst_predicted, atol=0.1)
    end

    @testset "B: Properties — forecast state anchoring" begin
        lib = build_solar_wind_library()
        ξ = zeros(length(lib))
        ξ_ens = zeros(500, length(lib))

        # With zero coefficients, dDst/dt = 0 always → Dst stays constant
        state = ForecastState(DateTime(2026,1,1), -50.0, lib, ξ, ξ_ens, 1.0, ForecastResult[])

        # Without observation anchoring: prediction should persist
        r1 = step_forecast!(state, DateTime(2026,1,1,1,0,0),
                            400.0, 0.0, 0.0, 5.0, 2.0)
        @test isapprox(r1.dst_predicted, -50.0, atol=1e-10)  # zero coefs → no change
        @test isapprox(state.dst_current, -50.0, atol=1e-10)

        # With observation anchoring: state should jump to observed value
        r2 = step_forecast!(state, DateTime(2026,1,1,2,0,0),
                            400.0, 0.0, 0.0, 5.0, 2.0;
                            dst_observed=-80.0)
        @test state.dst_current == -80.0  # anchored to observation

        # History should accumulate
        @test length(state.history) == 2
    end

    @testset "B: Properties — forecast_ahead does not modify state" begin
        lib = build_solar_wind_library()
        ξ = zeros(length(lib))
        ξ_ens = zeros(500, length(lib))
        state = ForecastState(DateTime(2026,1,1), -30.0, lib, ξ, ξ_ens, 1.0, ForecastResult[])

        dst_before = state.dst_current
        t_before = state.t_current
        hist_len_before = length(state.history)

        fc = forecast_ahead(state, 400.0, 0.0, 0.0, 5.0, 2.0, 6)

        # State must be unchanged
        @test state.dst_current == dst_before
        @test state.t_current == t_before
        @test length(state.history) == hist_len_before
        @test length(fc) == 6
    end

    @testset "B: Properties — ensemble CI monotonically widens" begin
        # With nonzero ensemble spread, multi-hour forecast CI should widen
        lib = build_solar_wind_library()
        n_t = length(lib)
        ξ = zeros(n_t)
        dst_idx = findfirst(==("Dst_star"), get_term_names(lib))
        ξ[dst_idx] = -0.05  # mild decay

        # Ensemble with spread on decay coefficient
        rng = MersenneTwister(42)
        ξ_ens = zeros(500, n_t)
        for i in 1:500
            ξ_ens[i, dst_idx] = -0.05 + 0.02 * randn(rng)
        end

        state = ForecastState(DateTime(2026,1,1), -50.0, lib, ξ, ξ_ens, 1.0, ForecastResult[])

        fc = forecast_ahead(state, 400.0, 5.0, 0.0, 5.0, 2.0, 12)
        # CI width = ci_95 - ci_05 should generally increase over time
        widths = [r.dst_ci_95 - r.dst_ci_05 for r in fc]
        # Last width should exceed first width (uncertainty grows)
        @test widths[end] > widths[1]
    end

    @testset "D: Edge cases — clamping bounds" begin
        lib = build_solar_wind_library()
        n_t = length(lib)
        ξ = zeros(n_t)
        # Huge negative coefficient to force extreme derivative
        bs_idx = findfirst(==("Bs"), get_term_names(lib))
        ξ[bs_idx] = -1000.0  # will produce dDst/dt = -1000 × Bs

        ξ_ens = repeat(ξ', 500)
        state = ForecastState(DateTime(2026,1,1), 0.0, lib, ξ, ξ_ens, 1.0, ForecastResult[])

        # Strong southward IMF → extreme injection
        r = step_forecast!(state, DateTime(2026,1,1,1,0,0),
                           500.0, -20.0, 0.0, 10.0, 5.0)
        # Derivative should be clamped to -200 nT/hr
        # So Dst_next = 0 + 1 × (-200) = -200, not -20000
        @test r.dst_predicted >= -200.0  # clamped
        @test r.dst_predicted <= 50.0    # within state bounds
    end

end

@testset "Alarm Module" begin

    @testset "A: Analytical — severity classification" begin
        config = default_alarm_config()
        th = config.thresholds
        # Exact threshold values from NOAA convention
        @test classify_severity(-30.0, th) == QUIET       # above -50
        @test classify_severity(-50.0, th) == MODERATE     # at -50
        @test classify_severity(-51.0, th) == MODERATE     # below -50
        @test classify_severity(-100.0, th) == INTENSE     # at -100
        @test classify_severity(-150.0, th) == INTENSE     # between -100 and -200
        @test classify_severity(-200.0, th) == SUPERINTENSE # at -200
        @test classify_severity(-500.0, th) == SUPERINTENSE # well below
        @test classify_severity(10.0, th) == QUIET         # positive Dst
    end

    @testset "B: Properties — alarm cooldown" begin
        alarms_fired = Alarm[]
        config = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            false,  # use predicted, not worst-case
            a -> push!(alarms_fired, a),  # collect alarms
            2,  # 2-hour cooldown
        )

        # First alarm should fire
        r1 = ForecastResult(DateTime(2026,1,1,0,0,0), -80.0, -80.0, -90.0, -70.0, NaN)
        a1, t1 = check_alarm(config, r1, DateTime(1970))
        @test a1 !== nothing
        @test a1.severity == MODERATE

        # Within cooldown: should NOT fire
        r2 = ForecastResult(DateTime(2026,1,1,1,0,0), -80.0, -80.0, -90.0, -70.0, NaN)
        a2, _ = check_alarm(config, r2, t1)
        @test a2 === nothing  # suppressed by cooldown

        # After cooldown: should fire again
        r3 = ForecastResult(DateTime(2026,1,1,3,0,0), -80.0, -80.0, -90.0, -70.0, NaN)
        a3, _ = check_alarm(config, r3, t1)
        @test a3 !== nothing  # cooldown expired (3 hr > 2 hr)
    end

    @testset "B: Properties — worst-case vs predicted" begin
        config_wc = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            true,   # use worst-case (ci_05)
            x -> nothing,
            0,
        )
        config_pred = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            false,  # use predicted
            x -> nothing,
            0,
        )

        # Predicted = -40 (QUIET), but worst case = -60 (MODERATE)
        r = ForecastResult(DateTime(2026,1,1), -40.0, -45.0, -60.0, -30.0, NaN)

        a_wc, _ = check_alarm(config_wc, r, DateTime(1970))
        a_pred, _ = check_alarm(config_pred, r, DateTime(1970))

        @test a_wc !== nothing    # worst-case triggers (ci_05 = -60 < -50)
        @test a_pred === nothing  # predicted doesn't trigger (-40 > -50)
    end

    @testset "D: Edge — quiet conditions produce no alarm" begin
        config = default_alarm_config(; callback=x->nothing)
        r = ForecastResult(DateTime(2026,1,1), 5.0, 5.0, 3.0, 7.0, NaN)
        a, _ = check_alarm(config, r, DateTime(1970))
        @test a === nothing
    end
end

@testset "Baselines — Full Models" begin

    @testset "A: Burton full vs simplified — threshold behavior" begin
        n = 3
        V = 300.0 .* ones(n)
        Bs = 1.0 .* ones(n)
        Dst = -10.0 .* ones(n)

        # Ey = V·Bs/1000 = 0.3 mV/m < 0.5 threshold → full model suppresses injection
        dDdt_full = burton_model_full(V, Bs, Dst)
        dDdt_simple = burton_model(V, Bs, Dst)

        # Full model: Q = 0 (suppressed), dDst/dt = -Dst/τ = +1.30 (recovery only)
        # Simple model: Q = -α·V·Bs, dDst/dt = Q - Dst/τ = -0.069 (injection + recovery)
        # Key test: full model has no injection → dDst/dt is MORE POSITIVE (less injection)
        for k in 1:n
            @test dDdt_full[k] > dDdt_simple[k]  # full lacks injection, so more positive
        end

        # Hand-calculated: full = -(-10)/7.7 = +1.2987, simple = -0.004559·300·1 + 1.2987 = -0.069
        @test isapprox(dDdt_full[1], 10.0 / 7.7, atol=1e-10)  # exact: only decay
        @test isapprox(dDdt_simple[1], -4.559e-3 * 300.0 * 1.0 + 10.0 / 7.7, atol=1e-10)

        # Above threshold: both should agree
        V_high = 600.0 .* ones(n)
        Bs_high = 10.0 .* ones(n)  # Ey = 6 mV/m >> 0.5
        dDdt_full2 = burton_model_full(V_high, Bs_high, Dst)
        dDdt_simple2 = burton_model(V_high, Bs_high, Dst)
        @test isapprox(dDdt_full2, dDdt_simple2, rtol=0.01)
    end

    @testset "A: O'Brien-McPherron — known analytical values" begin
        # At Ec = 5 mV/m: Q = -4.4(5 - 0.49) = -19.844, τ = 2.40·exp(9.74/9.69) ≈ 6.53 hr
        V = [5000.0]  # V·Bs/1000 = 5 mV/m when Bs = 1
        Bs = [1.0]
        Dst = [-50.0]
        dDdt = obrien_mcpherron_model(V, Bs, Dst)

        Ec = 5.0
        Q_expected = -4.4 * (Ec - 0.49)     # = -19.844
        τ_expected = 2.40 * exp(9.74 / (4.69 + Ec))  # ≈ 6.53
        dDdt_expected = Q_expected - (-50.0) / τ_expected

        @test isapprox(dDdt[1], dDdt_expected, rtol=1e-10)  # exact arithmetic
    end

    @testset "A: simulate_obrien — decay from Dst=-100 with no injection" begin
        n = 50
        V = 300.0 .* ones(n)
        Bs = zeros(n)  # no southward IMF → no injection

        Dst_sim = simulate_obrien(V, Bs, 1.0; Dst0=-100.0)
        @test Dst_sim[1] == -100.0
        # Should recover toward 0 (exponential decay)
        @test Dst_sim[end] > Dst_sim[1]  # recovering
        @test Dst_sim[end] < 0.0         # not yet fully recovered in 50 hr
    end

    @testset "B: simulate_burton_full — state bounds respected" begin
        n = 200
        V = 800.0 .* ones(n)
        Bs = 30.0 .* ones(n)  # extreme storm
        Dst_sim = simulate_burton_full(V, Bs, 1.0; Dst0=0.0)
        @test all(-2000.0 .<= Dst_sim .<= 50.0)  # bounds enforced
    end
end

@testset "SINDy — Additional Functions" begin

    @testset "A: sindy_predict matches Θ·ξ" begin
        lib = build_minimal_library()
        data = Dict{String,Vector{Float64}}(
            "V" => [400.0, 500.0, 600.0],
            "Bs" => [5.0, 10.0, 0.0],
            "Bz" => [-5.0, -10.0, 0.0],
            "By" => [0.0, 0.0, 0.0],
            "n" => [5.0, 5.0, 5.0],
            "Pdyn" => [2.0, 2.0, 2.0],
            "Dst_star" => [-30.0, -60.0, -10.0],
            "theta_c" => [π, π, 0.0],
            "BT" => [5.0, 10.0, 0.0],
        )
        ξ = [1.0, -0.1, -0.005]  # constant + decay + injection
        Θ = evaluate_library(lib, data)
        pred = sindy_predict(ξ, lib, data)
        @test pred ≈ Θ * ξ  # must be identical (exact arithmetic, atol=0)
    end

    @testset "B: sweep_lambda — monotonic sparsity" begin
        rng = MersenneTwister(99)
        n = 100
        X = randn(rng, n, 8)
        ξ_true = [3.0, 0.0, -2.0, 0.0, 0.0, 1.5, 0.0, 0.0]
        y = X * ξ_true + 0.05 .* randn(rng, n)

        lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        results = sweep_lambda(X, y, lambdas)

        # Higher λ → fewer terms (monotonically non-increasing)
        n_terms_seq = [r.n_terms for r in results]
        for i in 2:length(n_terms_seq)
            @test n_terms_seq[i] <= n_terms_seq[i-1]
        end

        # At low λ, should recover all 3 true terms
        @test results[1].n_terms >= 3
    end

    @testset "E: Error handling — stlsq dimension mismatch" begin
        X = randn(10, 3)
        y = randn(5)  # wrong length
        @test_throws AssertionError stlsq(X, y; λ=0.1)
    end

    @testset "E: Error handling — smooth_moving_average even window" begin
        @test_throws AssertionError smooth_moving_average([1.0, 2.0, 3.0], 4)
    end
end
