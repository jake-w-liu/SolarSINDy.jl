using Test
using SolarSINDy
using Random
using DataFrames
using CSV


function _static_stack_manual_stack()
    pooled = StaticStackCell(
        1, :pooled, 60, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
        objective_mse=2.0, iterations=10,
    )
    quiet = StaticStackCell(
        1, :quiet, 60, [0.35, 0.25, 0.15, 0.10, 0.10, 0.05];
        objective_mse=1.0, iterations=11,
    )
    return StaticStack(
        [pooled, quiet]; label="static-stack-test", minimum_cell_rows=48,
    )
end

function _static_stack_centers(values::AbstractVector{<:Real})
    return NamedTuple{STATIC_STACK_COMPONENTS}(Tuple(Float64.(values)))
end

@testset verbose=true "Operational static stack constrained stack" begin
    @testset "Analytical hand blend and causal regimes" begin
        stack = _static_stack_manual_stack()
        centers = _static_stack_centers([10, 20, 30, 40, 50, 60])
        result = static_stack_predict(stack, 1, -10.0, 0.0, 0.0, centers)

        # Independent hand sum: 3.5 + 5 + 4.5 + 4 + 5 + 3 = 25 nT.
        @test result.pred_dst == 25.0
        @test result.regime == :quiet
        @test result.cell_regime == :quiet
        @test !result.used_pooled_fallback
        @test result.sindy_mass == 0.60
        @test result.sindy_family_center_dst == (3.5 + 5.0) / 0.60
        @test result.correction_dst == result.pred_dst - result.sindy_family_center_dst
        # Each logged contribution must retain the corresponding center and weight.
        @test Tuple(result.component_contributions) == (3.5, 5.0, 4.5, 4.0, 5.0, 3.0)
        @test sum(Tuple(result.component_contributions)) == result.pred_dst

        # The three branches use only issue-time Dst, its one-hour rate, and coupling.
        @test static_stack_regime(-10.0, 0.0, 0.0) == :quiet
        @test static_stack_regime(-40.0, 1.0, 0.0) == :recovery
        @test static_stack_regime(-40.0, -1.0, 0.0) == :active_deepening
        @test static_stack_regime(-10.0, -5.0, 0.1) == :active_deepening
        @test static_stack_regime(-10.0, -5.0, 0.0) == :quiet
        @test_throws ArgumentError static_stack_regime(-10.0, 0.0, -0.1)
        @test_throws ArgumentError static_stack_regime(NaN, 0.0, 0.0)
    end

    @testset "Exact projection, immutable constraints, and fail-closed PGD" begin
        # This vector's ordinary simplex projection assigns all mass to the physical
        # group; the exact capped projection must split 0.60/0.40 by group symmetry.
        projected = SolarSINDy._static_stack_project_weights([0, 0, 1, 1, 1, 1], 0.60)
        @test isapprox(projected, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
                       rtol=0.0, atol=2e-16) # direct projection, roundoff only
        @test sum(projected) == 1.0
        @test sum(projected[1:2]) >= 0.60
        @test all(>=(0.0), projected)

        @test !ismutabletype(StaticStackCell)
        @test !ismutabletype(StaticStack)
        @test_throws ArgumentError StaticStackCell(
            1, :pooled, 48, [0.4, 0.2, 0.1, 0.1, 0.1, 0.2],
        )
        @test_throws ArgumentError StaticStackCell(
            1, :pooled, 48, [0.5, 0.3, 0.1, 0.1, 0.1, -0.1],
        )
        weak_sindy = StaticStackCell(
            1, :pooled, 48, [0.25, 0.25, 0.20, 0.10, 0.10, 0.10],
        )
        @test_throws ArgumentError StaticStack(
            [weak_sindy]; minimum_cell_rows=48, sindy_mass_floor=0.60,
        )

        # One PGD step cannot solve this nonstationary full-rank system at 1e-14.
        rng = MersenneTwister(1201)
        centers = randn(rng, 40, 6)
        observations = centers * [0.35, 0.30, 0.10, 0.10, 0.10, 0.05]
        @test_throws ErrorException SolarSINDy._static_stack_fit_weights(
            centers, observations, 0.60; tolerance=1e-14, max_iterations=1,
        )
    end

    @testset "Synthetic known-weight recovery and cell threshold" begin
        rng = MersenneTwister(2202)
        n = 360
        centers = -25.0 .+ 15.0 .* randn(rng, n, 6)
        true_weights = [0.34, 0.31, 0.13, 0.09, 0.08, 0.05]
        observations = centers * true_weights
        frame = DataFrame(
            served_inertial_sindy_dst_nt=centers[:, 1],
            frozen_inertial_sindy_dst_nt=centers[:, 2],
            persistence_dst_nt=centers[:, 3],
            burton_dst_nt=centers[:, 4],
            burton_full_dst_nt=centers[:, 5],
            obrien_dst_nt=centers[:, 6],
            observation_dst_nt=observations,
            model_step_hours=fill(1, n),
            latest_dst_nt=fill(-10.0, n),
            dst_delta_1h_nt=zeros(n),
            coupling_active_mvm=zeros(n),
        )
        stack = fit_static_stack(
            frame; minimum_cell_rows=48, tolerance=1e-11,
        )
        pooled = only(c for c in stack.cells if c.regime == :pooled)
        quiet = only(c for c in stack.cells if c.regime == :quiet)
        # Exact noiseless affine data identify the six simplex weights; 2e-8 is
        # twenty times the requested PGD tolerance after conditioning.
        @test isapprox(collect(pooled.weights), true_weights; rtol=0.0, atol=2e-8)
        @test isapprox(collect(quiet.weights), true_weights; rtol=0.0, atol=2e-8)
        @test pooled.objective_mse < 1e-12
        @test quiet.objective_mse < 1e-12
        @test all(c -> c.weights[1] + c.weights[2] >= 0.60, stack.cells)

        # Only quiet has enough rows; absent active/recovery cells must not be invented.
        @test Set(c.regime for c in stack.cells) == Set((:pooled, :quiet))
        @test_throws ArgumentError fit_static_stack(
            first(frame, 47); minimum_cell_rows=48,
        )
    end

    @testset "Pooled fallback, unsupported lead, and post-issue invariance" begin
        stack = _static_stack_manual_stack()
        centers = _static_stack_centers([10, 20, 30, 40, 50, 60])
        active = static_stack_predict(stack, 1, -40.0, -1.0, 0.0, centers)
        # No active-specific cell exists: 3 + 6 + 3 + 4 + 5 + 6 = 27 nT.
        @test active.pred_dst == 27.0
        @test active.regime == :active_deepening
        @test active.cell_regime == :pooled
        @test active.used_pooled_fallback
        @test_throws ArgumentError static_stack_predict(
            stack, 2, -10.0, 0.0, 0.0, centers,
        )

        base = DataFrame(
            served_inertial_sindy_dst_nt=[10.0, 11.0],
            frozen_inertial_sindy_dst_nt=[20.0, 21.0],
            persistence_dst_nt=[30.0, 31.0],
            burton_dst_nt=[40.0, 41.0],
            burton_full_dst_nt=[50.0, 51.0],
            obrien_dst_nt=[60.0, 61.0],
            model_step_hours=[1, 1], latest_dst_nt=[-10.0, -40.0],
            dst_delta_1h_nt=[0.0, -1.0], coupling_active_mvm=[0.0, 0.0],
            observation_dst_nt=[25.0, -99.0], future_driver=[1.0, 2.0],
        )
        mutated = copy(base)
        mutated.observation_dst_nt .= [999.0, -999.0]
        mutated.future_driver .= [-1.0e9, 1.0e9]
        scored = score_static_stack(base, stack)
        rescored = score_static_stack(mutated, stack)
        causal_outputs = [
            :static_stack_pred_dst_nt, :static_stack_regime, :static_stack_cell_regime,
            :static_stack_used_pooled_fallback, :static_stack_sindy_mass, :static_stack_label,
            :static_stack_sindy_family_center_dst_nt, :static_stack_correction_dst_nt,
            (Symbol("static_stack_weight_", c) for c in STATIC_STACK_COMPONENTS)...,
            (Symbol("static_stack_contribution_", c, "_nt") for c in
                STATIC_STACK_COMPONENTS)...,
        ]
        # Mutation of post-issue columns cannot alter any forecast or routing output.
        @test scored[:, causal_outputs] == rescored[:, causal_outputs]
        @test scored.static_stack_residual_dst_nt != rescored.static_stack_residual_dst_nt
    end

    @testset "Atomic CSV round trip and corruption rejection" begin
        stack = _static_stack_manual_stack()
        centers = _static_stack_centers([10, 20, 30, 40, 50, 60])
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "static_stack.csv")
            @test write_static_stack(path, stack) == path
            restored = read_static_stack(path)
            @test restored.label == stack.label
            @test restored.supported_model_steps == stack.supported_model_steps
            @test restored.cells == stack.cells
            @test static_stack_predict(
                restored, 1, -10.0, 0.0, 0.0, centers,
            ) == static_stack_predict(stack, 1, -10.0, 0.0, 0.0, centers)

            valid = CSV.read(path, DataFrame)
            bad_sum = copy(valid)
            bad_sum.weight_obrien[1] += 0.01
            bad_sum_path = joinpath(tmp, "bad_sum.csv")
            CSV.write(bad_sum_path, bad_sum)
            @test_throws ArgumentError read_static_stack(bad_sum_path)

            duplicate = vcat(valid, valid[1:1, :])
            duplicate_path = joinpath(tmp, "duplicate.csv")
            CSV.write(duplicate_path, duplicate)
            @test_throws ArgumentError read_static_stack(duplicate_path)

            bad_steps = copy(valid)
            bad_steps.supported_model_steps .= "1;1"
            bad_steps_path = joinpath(tmp, "bad_steps.csv")
            CSV.write(bad_steps_path, bad_steps)
            @test_throws ArgumentError read_static_stack(bad_steps_path)

            wrong_schema = select(valid, Not(:iterations))
            wrong_schema_path = joinpath(tmp, "wrong_schema.csv")
            CSV.write(wrong_schema_path, wrong_schema)
            @test_throws ArgumentError read_static_stack(wrong_schema_path)

            symlink_path = joinpath(tmp, "stack-link.csv")
            # Windows requires Administrator/Developer Mode to create symlinks; when the OS
            # denies creation the symlink-rejection assertions cannot run here and are skipped.
            can_symlink = try
                symlink(path, symlink_path)
                true
            catch e
                e isa Base.IOError || rethrow()
                false
            end
            if can_symlink
                @test_throws ArgumentError read_static_stack(symlink_path)
                @test_throws ArgumentError write_static_stack(symlink_path, stack)
            else
                @warn "symlink creation denied by the OS; skipping symlink evidence checks"
            end
        end
    end

    @testset "S2: numeric-looking text columns round-trip as text" begin
        # The reader used to let CSV infer the text columns. A zero-padded or purely numeric label
        # then parsed as a number, and `String(cell)` either threw a bare `MethodError` or silently
        # dropped the leading zeros — a stack served under a label the file does not carry.
        centers = _static_stack_centers([10, 20, 30, 40, 50, 60])
        for label in ("007", "2026", "0.5", "-30")
            stack = StaticStack(
                [StaticStackCell(1, :pooled, 60, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
                                    objective_mse=2.0, iterations=10)];
                label=label, minimum_cell_rows=48,
            )
            mktempdir() do tmp
                path = joinpath(tmp, "numeric_label.csv")
                write_static_stack(path, stack)
                restored = read_static_stack(path)
                @test restored.label == label
                @test restored.label isa String
                @test restored.supported_model_steps == stack.supported_model_steps
                @test restored.cells == stack.cells
                @test static_stack_predict(restored, 1, -10.0, 0.0, 0.0, centers) ==
                      static_stack_predict(stack, 1, -10.0, 0.0, 0.0, centers)
            end
        end
    end

    @testset "S6: the shared lead set and correction cap are one definition" begin
        @test SolarSINDy.STATIC_STACK_MODEL_STEPS == (1, 2, 3, 4, 6, 7)
        # Every static stack layer that declares a lead set declares this one.
        @test STATIC_STACK_BOOST_SUPPORTED_MODEL_STEPS === SolarSINDy.STATIC_STACK_MODEL_STEPS
        @test SolarSINDy.STATIC_STACK_ERROR_SUPPORTED_MODEL_STEPS ===
              SolarSINDy.STATIC_STACK_MODEL_STEPS
        @test SolarSINDy.STATIC_STACK_ERROR_EXOGENOUS_SUPPORTED_MODEL_STEPS ===
              SolarSINDy.STATIC_STACK_MODEL_STEPS
        @test SolarSINDy.STATIC_STACK_CORE_PATH_SUPPORTED_MODEL_STEPS ===
              SolarSINDy.STATIC_STACK_MODEL_STEPS
        @test SolarSINDy.STATIC_STACK_SHADOW_SUPPORTED_HORIZONS_H ===
              SolarSINDy.STATIC_STACK_MODEL_STEPS
        # The analog ensemble and super-learner served products declare their own lead sets; they are not aliased to the
        # static stack one, so drift between the offline chain and the served product is asserted here.
        @test ANALOG_SERVING_MODEL_STEPS == SolarSINDy.STATIC_STACK_MODEL_STEPS
        @test SUPERLEARNER_SERVING_MODEL_STEPS == SolarSINDy.STATIC_STACK_MODEL_STEPS
        # The bounded correction cap is 5 nT plus 5 nT per lead hour, shared by all three layers.
        for step in SolarSINDy.STATIC_STACK_MODEL_STEPS
            @test SolarSINDy.static_stack_correction_cap_nt(step) == 5.0 + 5.0 * step
            @test SolarSINDy._static_stack_residual_cap(step) ==
                  SolarSINDy.static_stack_correction_cap_nt(step)
            @test SolarSINDy._static_stack_boost_cap(step) ==
                  SolarSINDy.static_stack_correction_cap_nt(step)
            @test SolarSINDy._static_stack_error_cap(step) ==
                  SolarSINDy.static_stack_correction_cap_nt(step)
        end
        # The exogenous lagged-issue positions are derived from the two schemas, not written out.
        @test SolarSINDy._STATIC_STACK_ERROR_EXOGENOUS_ISSUE_INDICES == (1, 5, 7, 11)
        residual_names = collect(SolarSINDy.STATIC_STACK_RESIDUAL_FEATURES)
        @test [residual_names[j] for j in SolarSINDy._STATIC_STACK_ERROR_EXOGENOUS_ISSUE_INDICES] ==
              [v for v in SolarSINDy.STATIC_STACK_ERROR_EXOGENOUS_TEMPORAL_VARIABLES
               if v != :h1_innovation_nt]
    end

    @testset "S9: the regime split is closed at both deployed edges" begin
        # Kyoto Dst is integer-valued and the one-hour rate is a difference of integers, so both
        # regime edges are hit constantly. Each is asserted at equality, one side and the other.
        thr = SolarSINDy.STATIC_STACK_DEFAULT_DISTURBED_DST_NT
        rate = SolarSINDy.STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H
        cpl = SolarSINDy.STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM
        @test (thr, rate, cpl) == (-30.0, -5.0, 0.0)

        # Depth edge, evaluated with a non-negative rate so only the depth test can fire: at exactly
        # the threshold the row is already disturbed, one ulp shallower it is quiet.
        @test static_stack_regime(thr, 1.0, 0.0) == :recovery
        @test static_stack_regime(nextfloat(thr), 1.0, 0.0) == :quiet
        @test static_stack_regime(prevfloat(thr), 1.0, 0.0) == :recovery

        # Deepening branch inside a disturbed state is a strict `rate < 0`.
        @test static_stack_regime(-40.0, 0.0, 0.0) == :recovery
        @test static_stack_regime(-40.0, prevfloat(0.0), 0.0) == :active_deepening

        # Rate and coupling edges, isolated with a shallow ring current so the depth branch cannot
        # fire: `rate <= threshold` is inclusive, `coupling > threshold` is strict.
        @test static_stack_regime(-10.0, rate, nextfloat(cpl)) == :active_deepening
        @test static_stack_regime(-10.0, nextfloat(rate), nextfloat(cpl)) == :quiet
        @test static_stack_regime(-10.0, prevfloat(rate), nextfloat(cpl)) == :active_deepening
        @test static_stack_regime(-10.0, rate, cpl) == :quiet
        @test static_stack_regime(-10.0, 0.0, nextfloat(cpl)) == :quiet
    end
end
