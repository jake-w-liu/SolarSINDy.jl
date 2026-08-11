using Test
using SolarSINDy
using Random
using DataFrames
using CSV

const V22 = SolarSINDy

function _v22_manual_stack()
    pooled = OperationalV22Cell(
        1, :pooled, 60, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
        objective_mse=2.0, iterations=10,
    )
    quiet = OperationalV22Cell(
        1, :quiet, 60, [0.35, 0.25, 0.15, 0.10, 0.10, 0.05];
        objective_mse=1.0, iterations=11,
    )
    return OperationalV22Stack(
        [pooled, quiet]; label="v2.2-test", minimum_cell_rows=48,
    )
end

function _v22_centers(values::AbstractVector{<:Real})
    return NamedTuple{OPERATIONAL_V22_COMPONENTS}(Tuple(Float64.(values)))
end

@testset verbose=true "Operational V2.2 constrained stack" begin
    @testset "Analytical hand blend and causal regimes" begin
        stack = _v22_manual_stack()
        centers = _v22_centers([10, 20, 30, 40, 50, 60])
        result = operational_v22_predict(stack, 1, -10.0, 0.0, 0.0, centers)

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
        @test operational_v22_regime(-10.0, 0.0, 0.0) == :quiet
        @test operational_v22_regime(-40.0, 1.0, 0.0) == :recovery
        @test operational_v22_regime(-40.0, -1.0, 0.0) == :active_deepening
        @test operational_v22_regime(-10.0, -5.0, 0.1) == :active_deepening
        @test operational_v22_regime(-10.0, -5.0, 0.0) == :quiet
        @test_throws ArgumentError operational_v22_regime(-10.0, 0.0, -0.1)
        @test_throws ArgumentError operational_v22_regime(NaN, 0.0, 0.0)
    end

    @testset "Exact projection, immutable constraints, and fail-closed PGD" begin
        # This vector's ordinary simplex projection assigns all mass to the physical
        # group; the exact capped projection must split 0.60/0.40 by group symmetry.
        projected = V22._operational_v22_project_weights([0, 0, 1, 1, 1, 1], 0.60)
        @test isapprox(projected, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
                       rtol=0.0, atol=2e-16) # direct projection, roundoff only
        @test sum(projected) == 1.0
        @test sum(projected[1:2]) >= 0.60
        @test all(>=(0.0), projected)

        @test !ismutabletype(OperationalV22Cell)
        @test !ismutabletype(OperationalV22Stack)
        @test_throws ArgumentError OperationalV22Cell(
            1, :pooled, 48, [0.4, 0.2, 0.1, 0.1, 0.1, 0.2],
        )
        @test_throws ArgumentError OperationalV22Cell(
            1, :pooled, 48, [0.5, 0.3, 0.1, 0.1, 0.1, -0.1],
        )
        weak_sindy = OperationalV22Cell(
            1, :pooled, 48, [0.25, 0.25, 0.20, 0.10, 0.10, 0.10],
        )
        @test_throws ArgumentError OperationalV22Stack(
            [weak_sindy]; minimum_cell_rows=48, sindy_mass_floor=0.60,
        )

        # One PGD step cannot solve this nonstationary full-rank system at 1e-14.
        rng = MersenneTwister(1201)
        centers = randn(rng, 40, 6)
        observations = centers * [0.35, 0.30, 0.10, 0.10, 0.10, 0.05]
        @test_throws ErrorException V22._operational_v22_fit_weights(
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
            served_v2_1_dst_nt=centers[:, 1],
            frozen_v2_1_dst_nt=centers[:, 2],
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
        stack = fit_operational_v22_stack(
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
        @test_throws ArgumentError fit_operational_v22_stack(
            first(frame, 47); minimum_cell_rows=48,
        )
    end

    @testset "Pooled fallback, unsupported lead, and post-issue invariance" begin
        stack = _v22_manual_stack()
        centers = _v22_centers([10, 20, 30, 40, 50, 60])
        active = operational_v22_predict(stack, 1, -40.0, -1.0, 0.0, centers)
        # No active-specific cell exists: 3 + 6 + 3 + 4 + 5 + 6 = 27 nT.
        @test active.pred_dst == 27.0
        @test active.regime == :active_deepening
        @test active.cell_regime == :pooled
        @test active.used_pooled_fallback
        @test_throws ArgumentError operational_v22_predict(
            stack, 2, -10.0, 0.0, 0.0, centers,
        )

        base = DataFrame(
            served_v2_1_dst_nt=[10.0, 11.0],
            frozen_v2_1_dst_nt=[20.0, 21.0],
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
        scored = score_operational_v22(base, stack)
        rescored = score_operational_v22(mutated, stack)
        causal_outputs = [
            :v2_2_pred_dst_nt, :v2_2_regime, :v2_2_cell_regime,
            :v2_2_used_pooled_fallback, :v2_2_sindy_mass, :v2_2_stack_label,
            :v2_2_sindy_family_center_dst_nt, :v2_2_correction_dst_nt,
            (Symbol("v2_2_weight_", c) for c in OPERATIONAL_V22_COMPONENTS)...,
            (Symbol("v2_2_contribution_", c, "_nt") for c in
                OPERATIONAL_V22_COMPONENTS)...,
        ]
        # Mutation of post-issue columns cannot alter any forecast or routing output.
        @test scored[:, causal_outputs] == rescored[:, causal_outputs]
        @test scored.v2_2_residual_dst_nt != rescored.v2_2_residual_dst_nt
    end

    @testset "Atomic CSV round trip and corruption rejection" begin
        stack = _v22_manual_stack()
        centers = _v22_centers([10, 20, 30, 40, 50, 60])
        mktempdir() do tmp
            path = joinpath(tmp, "nested", "operational_v22.csv")
            @test write_operational_v22_stack(path, stack) == path
            restored = read_operational_v22_stack(path)
            @test restored.label == stack.label
            @test restored.supported_model_steps == stack.supported_model_steps
            @test restored.cells == stack.cells
            @test operational_v22_predict(
                restored, 1, -10.0, 0.0, 0.0, centers,
            ) == operational_v22_predict(stack, 1, -10.0, 0.0, 0.0, centers)

            valid = CSV.read(path, DataFrame)
            bad_sum = copy(valid)
            bad_sum.weight_obrien[1] += 0.01
            bad_sum_path = joinpath(tmp, "bad_sum.csv")
            CSV.write(bad_sum_path, bad_sum)
            @test_throws ArgumentError read_operational_v22_stack(bad_sum_path)

            duplicate = vcat(valid, valid[1:1, :])
            duplicate_path = joinpath(tmp, "duplicate.csv")
            CSV.write(duplicate_path, duplicate)
            @test_throws ArgumentError read_operational_v22_stack(duplicate_path)

            bad_steps = copy(valid)
            bad_steps.supported_model_steps .= "1;1"
            bad_steps_path = joinpath(tmp, "bad_steps.csv")
            CSV.write(bad_steps_path, bad_steps)
            @test_throws ArgumentError read_operational_v22_stack(bad_steps_path)

            wrong_schema = select(valid, Not(:iterations))
            wrong_schema_path = joinpath(tmp, "wrong_schema.csv")
            CSV.write(wrong_schema_path, wrong_schema)
            @test_throws ArgumentError read_operational_v22_stack(wrong_schema_path)

            symlink_path = joinpath(tmp, "stack-link.csv")
            symlink(path, symlink_path)
            @test_throws ArgumentError read_operational_v22_stack(symlink_path)
            @test_throws ArgumentError write_operational_v22_stack(symlink_path, stack)
        end
    end
end
