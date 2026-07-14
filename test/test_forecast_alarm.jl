# Tests for forecast.jl, alarm.jl, and untested baselines/sindy functions

using CSV
using DataFrames
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
        @test isapprox(result.dst_predicted, -100.0 + (-100.0 * (-1.0/7.7)),
                       atol=1e-12)  # Forward Euler with only decay term

        # Since all ensemble members are identical, CI should be very tight
        @test isapprox(result.dst_ci_05, result.dst_predicted, atol=1e-12)
        @test isapprox(result.dst_ci_95, result.dst_predicted, atol=1e-12)
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

    @testset "D: Forecast state and driver validation" begin
        lib = build_minimal_library()
        ξ = zeros(length(lib))
        ens = zeros(2, length(lib))
        t0 = DateTime(2026, 1, 1)
        @test_throws ArgumentError ForecastState(t0, Inf, lib, ξ, ens, 1.0)
        @test_throws ArgumentError ForecastState(t0, 0.0, lib, ξ, ens, 0.0)
        @test_throws ArgumentError ForecastState(t0, 0.0, lib, ξ, ens, 1e-10)
        @test_throws DimensionMismatch ForecastState(t0, 0.0, lib, ξ[1:2], ens, 1.0)
        @test_throws DimensionMismatch ForecastState(t0, 0.0, lib, ξ, zeros(2, 2), 1.0)
        @test_throws ArgumentError ForecastState(t0, 0.0, lib, ξ, zeros(0, length(lib)), 1.0)

        state = ForecastState(t0, -20.0, lib, ξ, ens, 1.0)
        @test_throws ArgumentError step_forecast!(state, t0 + Hour(1), Inf, 0.0, 0.0, 5.0, 2.0)
        @test_throws ArgumentError step_forecast!(state, t0 + Hour(1), 400.0, 0.0, 0.0, -1.0, 2.0)
        @test_throws ArgumentError step_forecast!(state, t0 + Hour(1), 400.0, 0.0, 0.0, 5.0, 2.0;
                                                  dst_observed=Inf)
        @test_throws ArgumentError step_forecast!(state, t0, 400.0, 0.0, 0.0, 5.0, 2.0)
        @test_throws ArgumentError step_forecast!(state, t0 + Hour(2), 400.0, 0.0, 0.0, 5.0, 2.0)
        corrupted = ForecastState(
            t0, -20.0, lib, zeros(length(lib)), zeros(2, length(lib)), 1.0,
        )
        corrupted.ξ_ensemble = zeros(1, 0)
        @test_throws DimensionMismatch step_forecast!(
            corrupted, t0 + Hour(1), 400.0, 0.0, 0.0, 5.0, 2.0,
        )
        @test_throws DimensionMismatch forecast_ahead(
            corrupted, 400.0, 0.0, 0.0, 5.0, 2.0, 1,
        )
        @test corrupted.t_current == t0
        @test corrupted.dst_current == -20.0
        @test_throws ArgumentError forecast_ahead(state, 400.0, 0.0, 0.0, 5.0, 2.0, -1)

        half_hour = ForecastState(t0, -20.0, lib, ξ, ens, 0.5)
        half_result = step_forecast!(half_hour, t0 + Minute(30),
                                     400.0, 0.0, 0.0, 5.0, 2.0)
        @test half_result.t == t0 + Minute(30)
        half_ahead = forecast_ahead(half_hour, 400.0, 0.0, 0.0, 5.0, 2.0, 2)
        @test getfield.(half_ahead, :t) == [t0 + Hour(1), t0 + Minute(90)]
    end

    @testset "A/D: V2 calibration" begin
        cal0 = default_operational_v2_calibration(;
            feature_names=[:latest_dst_nt],
            interval_scale=2.0,
        )
        pred0 = operational_v2_predict(
            cal0,
            -10.0,
            -12.0,
            -8.0,
            (; latest_dst_nt=-10.0),
        )
        @test pred0.pred_dst == -10.0
        @test pred0.ci05_dst == -14.0
        @test pred0.ci95_dst == -6.0
        @test pred0.correction == 0.0

        # Independent oracle: observations are exactly pred + 3 + 2*Bz.
        # Ridge-free residual calibration with a single feature should recover
        # that affine law and make scored residuals numerically zero.
        bz = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        base_pred = [-20.0, -18.0, -16.0, -14.0, -12.0, -10.0]
        observed = base_pred .+ 3.0 .+ 2.0 .* bz
        df = DataFrame(
            pred_dst_nt=base_pred,
            pred_dst_ci05_nt=base_pred .- 5.0,
            pred_dst_ci95_nt=base_pred .+ 5.0,
            observation_dst_nt=observed,
            Bz_nt=bz,
        )
        cal = fit_operational_v2_calibration(
            df;
            feature_names=[:Bz_nt],
            ridge=0.0,
            interval_coverage=0.90,
            label="unit_test_v2",
        )
        pred = operational_v2_predict(cal, -8.0, -13.0, -3.0, (; Bz_nt=4.0))
        @test isapprox(pred.pred_dst, -8.0 + 3.0 + 8.0, atol=1e-10)
        @test pred.interval_scale >= 1.0

        scored = score_operational_v2(df, cal)
        @test :v2_pred_dst_nt in propertynames(scored)
        @test maximum(abs.(scored.v2_residual_dst_nt)) < 1e-10
        @test all(scored.v2_observed_in_90ci)
        @test all(scored.v2_selected_component .== "v2")

        mktempdir() do tmp
            path = joinpath(tmp, "v2_calibration.csv")
            write_operational_v2_calibration(path, cal)
            reread = read_operational_v2_calibration(path)
            @test reread.feature_names == cal.feature_names
            @test reread.feature_mean ≈ cal.feature_mean
            @test reread.feature_scale ≈ cal.feature_scale
            @test reread.coefficients ≈ cal.coefficients
            @test reread.interval_scale == cal.interval_scale
            @test reread.label == cal.label
            @test reread.selector_names == cal.selector_names
            @test reread.selector_rmse ≈ cal.selector_rmse
            @test reread.selector_mae ≈ cal.selector_mae
            @test reread.selector_half_width ≈ cal.selector_half_width
            @test reread.selector_weights ≈ cal.selector_weights
            @test reread.selected_component == cal.selected_component
            @test reread.guard_margin_nt == cal.guard_margin_nt
        end
    end

    @testset "D: V2 center clamped to physical Dst range" begin
        # Mutation-sensitive guard for the operational-v2 center clamp. A
        # pathological positive residual correction must not push the issued
        # Dst* above the physical ceiling (+50 nT) used by every other forecast
        # path. Build a calibration whose residual is a constant +140 nT so the
        # fitted intercept is ~+140 and the :v2 component is selected (zero
        # residual), then issue a forecast from a moderately negative prediction
        # whose uncorrected center (-10 + 140 = +130 nT) is unphysical.
        bz = collect(-2.0:1.0:3.0)
        base_pred = collect(-25.0:1.0:-20.0)
        observed = base_pred .+ 140.0  # constant residual → intercept-only law
        df_hot = DataFrame(
            pred_dst_nt=base_pred,
            pred_dst_ci05_nt=base_pred .- 5.0,
            pred_dst_ci95_nt=base_pred .+ 5.0,
            observation_dst_nt=observed,
            Bz_nt=bz,
        )
        cal_hot = fit_operational_v2_calibration(
            df_hot;
            feature_names=[:Bz_nt],
            ridge=0.0,
            interval_coverage=0.90,
            label="clamp_unit_test",
        )
        @test cal_hot.selected_component == :v2  # perfect-fit v2 wins selector
        pred_hot = operational_v2_predict(cal_hot, -10.0, -15.0, -5.0, (; Bz_nt=0.0))
        # Without the clamp this would be ~ -10 + 140 = +130 nT (unphysical).
        @test pred_hot.pred_dst <= 50.0
        @test pred_hot.pred_dst >= -2000.0
        @test pred_hot.selected_component_pred <= 50.0
        @test pred_hot.corrected_sindy_pred <= 50.0
        # The interval is centered on the clamped value, so its center matches.
        @test (pred_hot.ci05_dst + pred_hot.ci95_dst) / 2 ≈ pred_hot.pred_dst atol=1e-9
    end

    @testset "A/D: V2 guarded baseline selector" begin
        n = 16
        v1_pred = fill(-40.0, n)
        observed = collect(-48.0:-1.0:-63.0)
        obrien = observed .+ 0.2
        df = DataFrame(
            pred_dst_nt=v1_pred,
            pred_dst_ci05_nt=v1_pred .- 3.0,
            pred_dst_ci95_nt=v1_pred .+ 3.0,
            observation_dst_nt=observed,
            latest_dst_nt=fill(-42.0, n),
            V_kms=fill(420.0, n),
            Bz_nt=collect(-8.0:1.0:7.0),
            By_nt=fill(1.0, n),
            n_cm3=fill(5.0, n),
            Pdyn_npa=fill(1.5, n),
            persistence_dst_nt=fill(-42.0, n),
            burton_dst_nt=fill(-43.0, n),
            burton_full_dst_nt=fill(-44.0, n),
            obrien_dst_nt=obrien,
        )
        cal = fit_operational_v2_calibration(
            df;
            ridge=1_000.0,
            guard_margin_nt=0.5,
            interval_coverage=0.90,
            label="guarded_unit_test",
        )
        @test cal.selected_component == :obrien
        @test :obrien in cal.selector_names
        @test cal.selector_mae[findfirst(==(:obrien), cal.selector_names)] < cal.selector_mae[1]

        pred = operational_v2_predict(
            cal,
            -40.0,
            -43.0,
            -37.0,
            operational_v2_feature_tuple(-42.0, 420.0, -1.0, 1.0, 5.0, 1.5);
            baselines=(; persistence=-42.0, burton=-43.0, burton_full=-44.0, obrien=-52.5),
        )
        @test pred.pred_dst == -52.5
        @test pred.selected_component == "obrien"
        @test pred.ci05_dst < pred.pred_dst < pred.ci95_dst

        @test_throws ArgumentError operational_v2_predict(
            cal,
            -40.0,
            -43.0,
            -37.0,
            operational_v2_feature_tuple(-42.0, 420.0, -1.0, 1.0, 5.0, 1.5),
        )

        scored = score_operational_v2(df, cal)
        @test all(scored.v2_selected_component .== "obrien")
        @test scored.v2_pred_dst_nt == obrien
        @test maximum(abs.(scored.v2_residual_dst_nt)) ≈ 0.2 atol=1e-12
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
        @test_throws ArgumentError classify_severity(NaN, th)
        @test_throws ArgumentError classify_severity(Inf, th)
        @test_throws ArgumentError classify_severity(-Inf, th)
        @test_throws ArgumentError AlarmConfig(
            Dict(MODERATE => NaN), false, identity, 1,
        )
        @test_throws ArgumentError AlarmConfig(
            Dict(MODERATE => -50.0), false, identity, -1,
        )
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

    @testset "D: future-dated alarm clock does not suppress a present-time alarm" begin
        # Regression: a forecast-horizon alarm can set last_alarm_time into the future;
        # the cooldown elapsed then goes negative. The guard must not treat a negative
        # elapsed as "within cooldown" and silently drop a genuine present-time alarm.
        config = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            false, x -> nothing, 2,
        )
        r_now = ForecastResult(DateTime(2026, 1, 1, 12), -120.0, -120.0, -130.0, -110.0, NaN)
        future_clock = DateTime(2026, 1, 1, 18)   # 6 h ahead of the present observation
        a, _ = check_alarm(config, r_now, future_clock)
        @test a !== nothing   # fires; under the bug (elapsed = -6h < cooldown) it was suppressed
    end

    @testset "D: cooldown boundary is closed at exactly cooldown_hours" begin
        # A target sitting precisely `cooldown_hours` after the last alarm must be
        # suppressed (closed upper bound), otherwise a fixed forecast-horizon offset
        # equal to the cooldown re-fires on every poll cycle.
        config = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            false, x -> nothing, 6,
        )
        r = ForecastResult(DateTime(2026, 1, 1, 6), -80.0, -80.0, -90.0, -70.0, NaN)
        a, _ = check_alarm(config, r, DateTime(2026, 1, 1, 0))  # elapsed == 6 h == cooldown
        @test a === nothing
        # Strictly beyond the cooldown still fires.
        r2 = ForecastResult(DateTime(2026, 1, 1, 7), -80.0, -80.0, -90.0, -70.0, NaN)
        a2, _ = check_alarm(config, r2, DateTime(2026, 1, 1, 0))  # elapsed == 7 h > cooldown
        @test a2 !== nothing
    end

    @testset "D: maybe_fire_horizon_alarm! dedups per target hour and escalates" begin
        fired = Alarm[]
        config = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            true, a -> push!(fired, a), 6,
        )
        seen = Dict{DateTime, StormSeverity}()
        t = DateTime(2026, 1, 1, 6)
        fr_mod = ForecastResult(t, -60.0, -60.0, -70.0, -50.0, NaN)

        # First evaluation fires once.
        a1 = maybe_fire_horizon_alarm!(config, fr_mod, seen)
        @test a1 !== nothing && a1.severity == MODERATE

        # Re-evaluating the SAME target at the SAME severity across poll cycles is
        # suppressed (this is the alarm-storm the finding reported).
        for _ in 1:5
            @test maybe_fire_horizon_alarm!(config, fr_mod, seen) === nothing
        end
        @test length(fired) == 1

        # Escalation to a more severe tier for the same target re-announces once.
        fr_int = ForecastResult(t, -120.0, -120.0, -130.0, -110.0, NaN)
        a2 = maybe_fire_horizon_alarm!(config, fr_int, seen)
        @test a2 !== nothing && a2.severity == INTENSE
        @test maybe_fire_horizon_alarm!(config, fr_int, seen) === nothing
        @test length(fired) == 2

        # A quiet forecast never fires and never records a target.
        fr_quiet = ForecastResult(DateTime(2026, 1, 1, 9), -10.0, -10.0, -20.0, 0.0, NaN)
        @test maybe_fire_horizon_alarm!(config, fr_quiet, seen) === nothing
        @test !haskey(seen, fr_quiet.t)
    end


    @testset "D: failed horizon delivery remains retryable" begin
        calls = Ref(0)
        callback = a -> begin
            calls[] += 1
            calls[] == 1 && error("transient delivery failure")
        end
        config = AlarmConfig(
            Dict(MODERATE => -50.0, INTENSE => -100.0, SUPERINTENSE => -200.0),
            false, callback, 6,
        )
        seen = Dict{DateTime,StormSeverity}()
        fr = ForecastResult(DateTime(2026, 1, 1, 6), -60.0, -60.0, -70.0, -50.0, NaN)
        @test_throws ErrorException maybe_fire_horizon_alarm!(config, fr, seen)
        @test !haskey(seen, fr.t)
        @test maybe_fire_horizon_alarm!(config, fr, seen) isa Alarm
        @test calls[] == 2
        @test seen[fr.t] == MODERATE
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

        # Hand-calculated: full = -(-10)/7.7 = +1.2987 (Q=0 below threshold),
        # simple = -5.4e-3·300·1 + 1.2987 = -0.321 (threshold-free injection).
        @test isapprox(dDdt_full[1], 10.0 / 7.7, atol=1e-10)  # exact: only decay
        @test isapprox(dDdt_simple[1], -5.4e-3 * 300.0 * 1.0 + 10.0 / 7.7, atol=1e-10)

        # Above threshold the published full model subtracts the injection threshold
        # (Q = -d·(V·Bs − 500)), so it injects LESS than the threshold-free simple
        # model by exactly d·500 = 2.7 nT/h — they must NOT coincide.
        V_high = 600.0 .* ones(n)
        Bs_high = 10.0 .* ones(n)  # V·Bs = 6000, Ey = 6 mV/m >> 0.5
        dDdt_full2 = burton_model_full(V_high, Bs_high, Dst)
        dDdt_simple2 = burton_model(V_high, Bs_high, Dst)
        for k in 1:n
            @test dDdt_full2[k] > dDdt_simple2[k]   # threshold offset ⇒ less injection
        end
        @test isapprox(dDdt_full2[1] - dDdt_simple2[1], 5.4e-3 * 500.0, atol=1e-10)
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
        @test_throws DimensionMismatch sindy_predict(ξ[1:2], lib, data)
        @test_throws ArgumentError sindy_predict([1.0, NaN, 0.0], lib, data)
        overflowing = CandidateLibrary(
            ["two"], Function[d -> fill(2.0, length(d["V"]))],
        )
        @test_throws ArgumentError sindy_predict(
            [floatmax(Float64)], overflowing, Dict("V" => [1.0]),
        )
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
        @test_throws ArgumentError sweep_lambda(X, y, Float64[])
        @test_throws ArgumentError sweep_lambda(X, y, [0.1, Inf])
    end

    @testset "E: Error handling — stlsq dimension mismatch" begin
        X = randn(10, 3)
        y = randn(5)  # wrong length
        @test_throws DimensionMismatch stlsq(X, y; λ=0.1)
    end

    @testset "E: Error handling — smooth_moving_average even window" begin
        @test_throws ArgumentError smooth_moving_average([1.0, 2.0, 3.0], 4)
    end

    @testset "Baseline threshold boundaries and degenerate IMF (mutation guards)" begin
        # Burton-full injection is threshold-continuous with the -500 (V·Bs) offset. At Ey = 0.5 mV/m
        # exactly (V·Bs = 500) the injection is zero, so the tendency is recovery-only; a regression
        # dropping the -500 offset (the old -α·V·Bs form) would inject a spurious step here instead.
        @test burton_model_full([500.0], [1.0], [-50.0])[1] ≈ 50.0 / 7.7 atol = 1e-12   # Q == 0
        @test burton_model_full([500.0], [1.001], [-50.0])[1] < 50.0 / 7.7              # Ey just > 0.5 → Q < 0

        # O'Brien injection is continuous at Ec_crit = 0.49 (so >/>= are identical there); bracket the
        # threshold rather than probing exact equality. With Dst* = 0 the tendency equals Q exactly.
        @test obrien_mcpherron_model([489.0], [1.0], [0.0])[1] == 0.0                     # Ec = 0.489 < crit
        @test obrien_mcpherron_model([491.0], [1.0], [0.0])[1] ≈ -4.4 * 0.001 atol = 1e-9 # Ec = 0.491 > crit

        # Degenerate IMF: BT = 0 (quiet solar wind) must not NaN — the max(BT, 1e-10) guard holds —
        # and with θ_c = 0 the Newell coupling is exactly zero.
        nc = newell_coupling([500.0], [0.0], [0.0])
        @test nc[1] == 0.0
        @test !isnan(nc[1])

        # Clock-angle quadrant convention θ_c = atan(|By|, Bz): the exported utility feeds the SINDy
        # θ_c / B_T features, so pin all four quadrants against a sign/argument-order flip.
        @test imf_clock_angle([0.0], [5.0])[1] ≈ 0.0 atol = 1e-12       # due north
        @test imf_clock_angle([0.0], [-5.0])[1] ≈ π atol = 1e-12        # due south
        @test imf_clock_angle([5.0], [5.0])[1] ≈ π / 4 atol = 1e-12     # north-east
        @test imf_clock_angle([5.0], [-5.0])[1] ≈ 3π / 4 atol = 1e-12   # south-east

        # Mismatched or nonphysical baseline inputs must fail before broadcast
        # truncation/shape errors or NaN/Inf propagation can masquerade as output.
        @test_throws DimensionMismatch burton_model([400.0, 500.0], [2.0], [-10.0, -20.0])
        @test_throws DimensionMismatch newell_coupling([400.0], [2.0, 3.0], [π])
        @test_throws ArgumentError simulate_burton([400.0], [2.0], 0.0)
        @test_throws ArgumentError simulate_burton([400.0], [2.0], 1.0; τ=Inf)
        @test_throws ArgumentError simulate_obrien([400.0], [2.0], 1.0; Ec_crit=-1.0)
        @test_throws ArgumentError burton_model([NaN], [2.0], [-10.0])
        @test_throws ArgumentError burton_model([-400.0], [2.0], [-10.0])
        @test_throws ArgumentError burton_model([400.0], [-2.0], [-10.0])
        @test_throws ArgumentError newell_coupling([400.0], [-2.0], [π])
        @test_throws ArgumentError simulate_obrien([400.0], [Inf], 1.0)
    end
end
