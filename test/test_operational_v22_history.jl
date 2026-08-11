using Test
using SolarSINDy
using Dates
using CSV
using DataFrames

const V22H = SolarSINDy

function _v22h_driver(; V=500.0, Bz=-4.0, By=1.0, n=5.0,
                      Pdyn=dynamic_pressure(n, V))
    return OperationalV22HistoryDriver(V, Bz, By, n, Pdyn)
end

function _v22h_core_tuple(driver::OperationalV22HistoryDriver)
    return (
        V=driver.speed_km_s,
        Bz=driver.bz_nt,
        By=driver.by_nt,
        n=driver.density_cm3,
        Pdyn=driver.pdyn_npa,
    )
end

@testset verbose=true "Operational V2.2-M1 causal sparse history" begin
    core = load_operational_core()
    zero_artifact = OperationalV22HistoryArtifact(
        core, (0.0, 0.0, 0.0);
        tau_memory_hours=6.0,
        support_mask=(false, false, false),
        coupling_bound_mvm=50.0,
        fit_rows=24,
        label="v2.2-m1-zero-oracle",
    )

    @testset "A: coupling, feature, and constant-memory oracles" begin
        southward = _v22h_driver(V=500.0, Bz=-4.0)
        northward = _v22h_driver(V=500.0, Bz=4.0)
        # Independent unit conversion: 10^-3 * 500 km/s * 4 nT = 2 mV/m.
        @test operational_v22_history_coupling(southward) == 2.0
        # Northward IMF is half-wave rectified, so its causal injection is zero.
        @test operational_v22_history_coupling(northward) == 0.0
        @test operational_v22_history_coupling(500.0, -4.0) == 2.0
        # Frozen features are exactly (m, E-m, x*m); swapped terms or signs fail.
        @test operational_v22_history_features(-20.0, 2.0, 5.0) ==
              (2.0, 3.0, -40.0)

        tau = 5.0
        rho = exp(-1.0 / tau)
        memory0 = 1.0
        constant_coupling = 4.0
        memory = memory0
        for _ in 1:7
            memory = operational_v22_history_memory(
                memory, constant_coupling, tau,
            )
        end
        # Closed form of m[k+1] = rho*m[k] + (1-rho)*E for constant E.
        expected = constant_coupling +
                   (memory0 - constant_coupling) * rho^7
        @test isapprox(memory, expected; rtol=0.0, atol=4e-15)
        @test operational_v22_history_rho(zero_artifact) == exp(-1.0 / 6.0)

        @test_throws ArgumentError operational_v22_history_coupling(-1.0, -4.0)
        @test_throws ArgumentError operational_v22_history_coupling(NaN, -4.0)
        @test_throws ArgumentError operational_v22_history_memory(-1.0, 2.0, tau)
        @test_throws ArgumentError operational_v22_history_memory(1.0, Inf, tau)
        @test_throws ArgumentError operational_v22_history_memory(1.0, 2.0, 0.0)
        @test_throws ArgumentError OperationalV22HistoryDriver(-1.0, 0.0, 0.0, 5.0, 1.0)
        @test_throws ArgumentError OperationalV22HistoryDriver(400.0, 0.0, 0.0, -1.0, 1.0)
        @test_throws ArgumentError OperationalV22HistoryDriver(400.0, 0.0, 0.0, 5.0, Inf)
    end

    @testset "A: zero augmentation is exactly the V2.1 trajectory" begin
        t0 = DateTime(2021, 3, 1, 0)
        state = OperationalV22HistoryState(t0, -60.0, 1.5)
        drivers = OperationalV22HistoryDriver[
            _v22h_driver(V=420.0, Bz=-3.0, By=1.0, n=4.0),
            _v22h_driver(V=510.0, Bz=-6.0, By=-2.0, n=7.0),
            _v22h_driver(V=380.0, Bz=2.0, By=3.0, n=3.0),
            _v22h_driver(V=650.0, Bz=-8.0, By=0.5, n=9.0),
        ]

        one = operational_v22_history_step(core, zero_artifact, state, drivers[1])
        one_oracle = only(operational_core_forecast(
            core, state.dst_star_nt, _v22h_core_tuple(drivers[1]), 1,
        ))
        # With eta=(0,0,0), the new state must be bit-identical to frozen V2.1.
        @test one.dst_star_nt == one_oracle
        @test one.augmentation_derivative_nt_per_h == 0.0
        @test one.next_state.t_current == t0 + Hour(1)

        expected = Float64[]
        dst = state.dst_star_nt
        for driver in drivers
            dst = only(operational_core_forecast(
                core, dst, _v22h_core_tuple(driver), 1,
            ))
            push!(expected, dst)
        end
        rollout = operational_v22_history_rollout(
            core, zero_artifact, state, drivers,
        )
        # A hand-composed sequence of independent V2.1 one-step calls is exact.
        @test rollout.dst_star_nt == expected
        @test all(iszero, rollout.augmentation_derivative_nt_per_h)
        @test rollout.forecast_times == [t0 + Hour(k) for k in 1:4]
    end

    @testset "A/C: hand one-step and multi-step sparse augmentation" begin
        eta = (-0.2, 0.3, -0.001)
        artifact = OperationalV22HistoryArtifact(
            core, eta;
            tau_memory_hours=4.0,
            coupling_bound_mvm=20.0,
            fit_rows=40,
            label="v2.2-m1-hand-oracle",
        )
        state = OperationalV22HistoryState(DateTime(2021, 4, 1), -40.0, 2.0)
        driver = _v22h_driver(V=600.0, Bz=-5.0, By=1.0, n=6.0)
        coupling = 3.0
        hand_features = (2.0, 1.0, -80.0)
        hand_augmentation = eta[1] * hand_features[1] +
                            eta[2] * hand_features[2] +
                            eta[3] * hand_features[3]
        base_next = only(operational_core_forecast(
            core, state.dst_star_nt, _v22h_core_tuple(driver), 1,
        ))
        derivative = operational_v22_history_derivative(
            core, artifact, state, driver,
        )
        result = operational_v22_history_step(core, artifact, state, driver)
        # The sparse term is added to the frozen derivative before state projection.
        @test derivative.augmentation_derivative_nt_per_h == hand_augmentation
        @test derivative.derivative_nt_per_h == result.derivative_nt_per_h
        @test result.features == hand_features
        @test result.augmentation_derivative_nt_per_h == hand_augmentation
        @test isapprox(
            result.dst_star_nt,
            base_next + hand_augmentation;
            rtol=0.0,
            atol=2e-14,
        )
        hand_memory = exp(-1 / 4) * 2.0 + (1 - exp(-1 / 4)) * coupling
        @test isapprox(result.next_state.memory_mvm, hand_memory; rtol=0.0, atol=2e-15)

        drivers = OperationalV22HistoryDriver[
            driver,
            _v22h_driver(V=450.0, Bz=-2.0, By=-1.0, n=5.0),
            _v22h_driver(V=700.0, Bz=-7.0, By=2.0, n=8.0),
            _v22h_driver(V=400.0, Bz=1.0, By=1.0, n=4.0),
            _v22h_driver(V=550.0, Bz=-4.0, By=-3.0, n=6.0),
        ]
        hand_dst = Float64[]
        x = state.dst_star_nt
        m = state.memory_mvm
        rho = exp(-1 / artifact.tau_memory_hours)
        for row in drivers
            e = 1e-3 * row.speed_km_s * max(-row.bz_nt, 0.0)
            g = eta[1] * m + eta[2] * (e - m) + eta[3] * x * m
            base_x = only(operational_core_forecast(
                core, x, _v22h_core_tuple(row), 1,
            ))
            x = clamp(base_x + g, -2000.0, 50.0)
            m = rho * m + (1 - rho) * e
            push!(hand_dst, x)
        end
        rollout = operational_v22_history_rollout(core, artifact, state, drivers)
        # Manual recurrence uses the frozen V2.1 oracle plus the stated three terms.
        @test isapprox(rollout.dst_star_nt, hand_dst; rtol=0.0, atol=8e-14)

        short = operational_v22_history_rollout(
            core, artifact, state, drivers[1:3],
        )
        # Longer execution cannot revise an already issued trajectory prefix.
        @test short.dst_star_nt == rollout.dst_star_nt[1:3]
        @test short.memory_mvm == rollout.memory_mvm[1:3]
        @test short.derivative_nt_per_h == rollout.derivative_nt_per_h[1:3]

        lagged = operational_v22_history_rollout(
            core, artifact, state, drivers;
            anchor_lag_hours=1,
        )
        catchup = operational_v22_history_step(core, artifact, state, drivers[1])
        lag_oracle = operational_v22_history_rollout(
            core, artifact, catchup.next_state, drivers[2:end],
        )
        # Lag one is exactly one catch-up step, followed by the shared kernel.
        @test lagged.dst_star_nt == lag_oracle.dst_star_nt
        @test lagged.memory_mvm == lag_oracle.memory_mvm
        @test lagged.forecast_times == lag_oracle.forecast_times
        @test_throws ArgumentError operational_v22_history_rollout(
            core, artifact, state, drivers; anchor_lag_hours=2,
        )
        @test_throws ArgumentError operational_v22_history_rollout(
            core, artifact, state, drivers; anchor_lag_hours=-1,
        )
        @test_throws ArgumentError operational_v22_history_rollout(
            core, artifact, state, drivers; anchor_lag_hours=true,
        )
        @test_throws ArgumentError operational_v22_history_rollout(
            core, artifact, state, drivers[1:1]; anchor_lag_hours=1,
        )
    end

    @testset "D/E: causal history reconstruction and gaps fail closed" begin
        start = DateTime(2021, 6, 1)
        timestamps = [start + Hour(k) for k in 0:5]
        constant_driver = _v22h_driver(V=500.0, Bz=-4.0)
        drivers = fill(constant_driver, length(timestamps))
        state = init_operational_v22_history_state(
            zero_artifact, timestamps, drivers, -35.0,
        )
        # Seeding and every update at constant E leave m exactly at E.
        @test state.memory_mvm == 2.0
        @test state.t_current == last(timestamps)
        @test state.dst_star_nt == -35.0

        gap = copy(timestamps)
        gap[4] += Hour(1)
        duplicate = copy(timestamps)
        duplicate[4] = duplicate[3]
        @test_throws ArgumentError init_operational_v22_history_state(
            zero_artifact, gap, drivers, -35.0,
        )
        @test_throws ArgumentError init_operational_v22_history_state(
            zero_artifact, duplicate, drivers, -35.0,
        )
        @test_throws ArgumentError init_operational_v22_history_state(
            zero_artifact, timestamps[1:1], drivers[1:1], -35.0,
        )
        @test_throws DimensionMismatch init_operational_v22_history_state(
            zero_artifact, timestamps, drivers[1:end-1], -35.0,
        )
        extreme = copy(drivers)
        extreme[end] = _v22h_driver(V=1000.0, Bz=-100.0)
        @test_throws ArgumentError init_operational_v22_history_state(
            zero_artifact, timestamps, extreme, -35.0,
        )
    end

    @testset "A/G: synthetic coefficient and all-support recovery" begin
        n_rows = 48
        x = [-12.0 - 0.73 * row + 0.11 * cos(row) for row in 1:n_rows]
        m = [0.8 + 0.09 * row + 0.23 * sin(0.7 * row) for row in 1:n_rows]
        e = [m[row] + 0.35 * sin(0.31 * row) + 0.2 for row in 1:n_rows]
        eta_true = (-0.21, 0.17, -0.003)

        fitted_hash = nothing
        for bits in 0:7
            mask = ntuple(index -> !iszero(bits & (1 << (index - 1))), 3)
            target = [
                (mask[1] ? eta_true[1] * m[row] : 0.0) +
                (mask[2] ? eta_true[2] * (e[row] - m[row]) : 0.0) +
                (mask[3] ? eta_true[3] * x[row] * m[row] : 0.0)
                for row in 1:n_rows
            ]
            fitted = fit_operational_v22_history(
                core, x, m, e, target;
                tau_memory_hours=8.0,
                support_mask=mask,
                coupling_bound_mvm=20.0,
                label="v2.2-m1-support-$bits",
            )
            # Each of the 2^3 frozen supports is fit directly, never post-hoc clipped.
            @test fitted.support_mask == mask
            @test all(index -> mask[index] || fitted.coefficients[index] == 0.0, 1:3)
            for index in findall(mask)
                @test isapprox(
                    fitted.coefficients[index], eta_true[index];
                    rtol=2e-12,
                    atol=2e-12,
                )
            end
            if bits == 7
                fitted_hash = operational_v22_history_sha256(fitted)
                repeated = fit_operational_v22_history(
                    core, x, m, e, target;
                    tau_memory_hours=8.0,
                    support_mask=mask,
                    coupling_bound_mvm=20.0,
                    label="v2.2-m1-support-$bits",
                )
                @test operational_v22_history_sha256(repeated) == fitted_hash
            end
        end

        all_terms_zero = OperationalV22HistoryArtifact(
            core, (0.0, 0.0, 0.0);
            tau_memory_hours=8.0,
            support_mask=(true, true, true),
            fit_rows=n_rows,
        )
        no_terms_zero = OperationalV22HistoryArtifact(
            core, (0.0, 0.0, 0.0);
            tau_memory_hours=8.0,
            support_mask=(false, false, false),
            fit_rows=n_rows,
        )
        # Support is byte-significant even when all numerical coefficients are zero.
        @test operational_v22_history_sha256(all_terms_zero) !=
              operational_v22_history_sha256(no_terms_zero)
        @test_throws ArgumentError OperationalV22HistoryArtifact(
            core, (-0.1, 0.0, 0.0);
            tau_memory_hours=8.0,
            support_mask=(false, true, true),
            fit_rows=n_rows,
        )
        @test_throws ArgumentError fit_operational_v22_history(
            core, fill(-20.0, n_rows), fill(2.0, n_rows),
            fill(3.0, n_rows), fill(0.0, n_rows);
            tau_memory_hours=8.0,
            support_mask=(true, true, true),
        )
        @test fitted_hash isa String
    end

    @testset "D/E: stability envelope, derivative cap, and state projection" begin
        @test_throws ArgumentError OperationalV22HistoryArtifact(
            core, (-0.1, 0.0, -1.0);
            tau_memory_hours=6.0,
            coupling_bound_mvm=50.0,
            fit_rows=24,
        )
        @test_throws ArgumentError OperationalV22HistoryArtifact(
            core, (0.1, 0.0, 0.0);
            tau_memory_hours=6.0,
            fit_rows=24,
        )
        @test_throws ArgumentError OperationalV22HistoryArtifact(
            core, (0.0, 0.0, 0.001);
            tau_memory_hours=6.0,
            fit_rows=24,
        )

        positive = OperationalV22HistoryArtifact(
            core, (0.0, 1000.0, 0.0);
            tau_memory_hours=6.0,
            support_mask=(false, true, false),
            fit_rows=24,
        )
        negative = OperationalV22HistoryArtifact(
            core, (0.0, -1000.0, 0.0);
            tau_memory_hours=6.0,
            support_mask=(false, true, false),
            fit_rows=24,
        )
        driven = _v22h_driver(V=500.0, Bz=-2.0)
        upper = operational_v22_history_step(
            core, positive,
            OperationalV22HistoryState(DateTime(2021, 8, 1), 49.0, 0.0),
            driven,
        )
        lower = operational_v22_history_step(
            core, negative,
            OperationalV22HistoryState(DateTime(2021, 8, 1), -1990.0, 0.0),
            driven,
        )
        # Derivative clipping and physical state projection are separate guards.
        @test upper.derivative_nt_per_h == 200.0
        @test upper.derivative_was_capped
        @test upper.dst_star_nt == 50.0
        @test upper.state_was_projected
        @test lower.derivative_nt_per_h == -200.0
        @test lower.derivative_was_capped
        @test lower.dst_star_nt == -2000.0
        @test lower.state_was_projected

        @test_throws ArgumentError OperationalV22HistoryState(
            DateTime(2021, 8, 1), 51.0, 0.0,
        )
        @test_throws ArgumentError operational_v22_history_step(
            core, zero_artifact,
            OperationalV22HistoryState(DateTime(2021, 8, 1), -20.0, 50.1),
            driven,
        )
        @test_throws ArgumentError operational_v22_history_step(
            core, zero_artifact,
            OperationalV22HistoryState(DateTime(2021, 8, 1), -20.0, 1.0),
            _v22h_driver(V=1000.0, Bz=-100.0),
        )
    end

    @testset "E/G: checksummed round trip, support mutation, and core identity" begin
        artifact = OperationalV22HistoryArtifact(
            core, (-0.2, 0.3, -0.001);
            tau_memory_hours=7.0,
            support_mask=(true, true, true),
            coupling_bound_mvm=20.0,
            fit_rows=96,
            label="v2.2-m1-roundtrip",
        )
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "history.csv")
            @test write_operational_v22_history(path, artifact) == path
            restored = read_operational_v22_history(path, core)
            # Exact SHA equality covers every serving-significant artifact field.
            @test operational_v22_history_sha256(restored) ==
                  operational_v22_history_sha256(artifact)
            @test restored.coefficients == artifact.coefficients
            @test restored.support_mask == artifact.support_mask

            coefficient_mutation = CSV.read(path, DataFrame)
            coefficient_mutation.coefficient[2] += 0.125
            CSV.write(path, coefficient_mutation)
            @test_throws ArgumentError read_operational_v22_history(path, core)

            write_operational_v22_history(path, artifact)
            support_mutation = CSV.read(path, DataFrame)
            support_mutation.selected[1] = false
            CSV.write(path, support_mutation)
            @test_throws ArgumentError read_operational_v22_history(path, core)

            write_operational_v22_history(path, artifact)
            altered_coefficients = copy(core.coefficients)
            dst_index = findfirst(==("Dst_star"), get_term_names(core.library))
            altered_coefficients[dst_index] += 1e-6
            altered_core = OperationalCore(
                core.artifacts, core.library, altered_coefficients,
            )
            @test_throws ArgumentError read_operational_v22_history(path, altered_core)
            @test_throws ArgumentError operational_v22_history_step(
                altered_core, artifact,
                OperationalV22HistoryState(DateTime(2021, 9, 1), -40.0, 2.0),
                _v22h_driver(),
            )
        end
    end
end
