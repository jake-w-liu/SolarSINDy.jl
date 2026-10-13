module ServingIdentityOracleTests

# Unit oracles for the two offline identity scripts
# (`validation/operational/static_stack_served_identity.jl`,
#  `validation/operational/superlearner_serving_identity.jl`).
#
# The scripts themselves are full-scale runs against the 400 MB archived base table, which is not a
# unit-test cost. What is checked here is everything about them that can be wrong without the archive:
# the base-table column contracts they declare, the driver-history construction they feed the serving
# functions, and the determinism and coverage of their anchor sample. The scripts' published results
# are then read back from their output artifacts and re-asserted, so a stale or failing run cannot sit
# unnoticed beside a green suite.

using Test
using CSV
using DataFrames
using Dates
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational",
                          "static_stack_served_identity.jl")))
include(normpath(joinpath(@__DIR__, "..", "validation", "operational",
                          "superlearner_serving_identity.jl")))

isdefined(Main, :LocalArtifactGate) ||
    Base.include(Main, normpath(joinpath(@__DIR__, "local_artifact_gate.jl")))
using Main.LocalArtifactGate: local_artifact_available

const STATIC_STACK_IDENTITY_OUT = joinpath(OPERATIONAL_OUTPUT_DIR, "static_stack_served_identity.csv")
const SUPERLEARNER_IDENTITY_OUT = joinpath(OPERATIONAL_OUTPUT_DIR, "superlearner_serving_identity.csv")

@testset "Serving identity oracles" begin
    @testset "declared base-table column contracts are complete" begin
        static_stack = static_stack_identity_table_columns()
        # Every component the stack multiplies, plus both regime inputs and the archived answer.
        for column in ("served_inertial_sindy_dst_nt", "frozen_inertial_sindy_dst_nt", "persistence_dst_nt",
                       "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt",
                       "latest_dst_nt", "dst_delta_1h_nt", "VBsouth_mvm",
                       "coupling_active_mvm", "static_stack_dst_nt", "model_step_hours",
                       "partition")
            @test column in static_stack
        end
        @test allunique(static_stack)

        superlearner = superlearner_identity_table_columns()
        # The super-learner oracle needs everything the analog ensemble correction needs, plus the two regime inputs of the
        # stack cell and the static-stack center, which Amendment A3 made expert ten of the stack.
        for column in ("latest_dst_nt", "V_kms", "Bz_nt", "By_nt", "n_cm3", "Pdyn_npa",
                       "dst_delta_1h_nt", "dst_delta_3h_nt", "Bz_delta_1h_nt",
                       "VBsouth_delta_1h_mvm", "VBsouth_mean_3h_mvm", "Bsouth_mean_3h_nt",
                       "VBsouth_mvm", "coupling_active_mvm", "persistence_dst_nt",
                       "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt",
                       "frozen_inertial_sindy_dst_nt", "served_inertial_sindy_dst_nt", "static_stack_dst_nt")
            @test column in superlearner
        end
        @test allunique(superlearner)
        # Every expert the served stack multiplies has a reported column, and so does every stage
        # between the experts and the published interval; a missing entry would hide a regression.
        for expert in SUPERLEARNER_SERVING_EXPERTS
            @test String(expert) in SUPERLEARNER_IDENTITY_COLUMNS
        end
        for column in ("l1_center", "superlearner_e", "superlearner_e_lo_nt", "superlearner_e_hi_nt", "static_stack")
            @test column in SUPERLEARNER_IDENTITY_COLUMNS
        end
        @test SUPERLEARNER_IDENTITY_SERVED_VARIANT == SUPERLEARNER_SERVED_VARIANT
        @test !any(occursin("superlearner_d", c) for c in SUPERLEARNER_IDENTITY_COLUMNS)
        @test STATIC_STACK_IDENTITY_TOL_NT == 1e-9
        @test SUPERLEARNER_IDENTITY_TOL_NT == 1e-9
        @test STATIC_STACK_IDENTITY_MIN_ROWS >= 5_000
        @test SUPERLEARNER_IDENTITY_HISTORY_LAGS == ANALOG_SOUTH_RUN_CAP_H
    end

    @testset "super-learner oracle history passes records through as the archive holds them" begin
        times = [DateTime(2021, 3, 1, 0) + Hour(k - 1) for k in 1:40]
        frame = DataFrame(
            time_utc = times,
            V = [400.0 + k for k in eachindex(times)],
            Bz = [-2.0 - 0.1 * k for k in eachindex(times)],
            By = [1.0 for _ in times],
            n = [5.0 + 0.01 * k for k in eachindex(times)],
            Pdyn = [dynamic_pressure(5.0 + 0.01 * k, 400.0 + k) for k in eachindex(times)],
            Dst = [-20.0 - k for k in eachindex(times)],
        )
        index = Dict(frame.time_utc[i] => i for i in 1:nrow(frame))
        anchor = times[30]
        history = superlearner_identity_driver_history(index, frame, anchor)
        @test length(history) == SUPERLEARNER_IDENTITY_HISTORY_LAGS
        for lag in 1:SUPERLEARNER_IDENTITY_HISTORY_LAGS
            row = index[anchor - Hour(lag)]
            @test history[lag].V == frame.V[row]
            @test history[lag].Bz == frame.Bz[row]
        end
        # An hour the frame has no row for is the only `nothing`. A record whose density is
        # non-positive is passed through unchanged, because the study's run-length and coupling-lag
        # features read such a record's other fields from the whole frame; filtering it here would
        # truncate a feature the study computed and break the identity it is checked against.
        early = superlearner_identity_driver_history(index, frame, times[3])
        @test early[3] === nothing
        holed = copy(frame)
        holed.n[index[anchor - Hour(2)]] = 0.0
        passed_through = superlearner_identity_driver_history(index, holed, anchor)[2]
        @test passed_through !== nothing
        @test passed_through.n == 0.0
        # The Dst ladder reaches back exactly as far as the design's deepest lag, and a non-finite
        # observation is omitted rather than imputed.
        dst = superlearner_identity_dst_history(index, frame, anchor)
        @test length(dst) == SUPERLEARNER_SERVING_DST_LAG_MAX_H + 1
        @test dst[anchor] == frame.Dst[index[anchor]]
        @test dst[anchor - Hour(SUPERLEARNER_SERVING_DST_LAG_MAX_H)] ==
              frame.Dst[index[anchor - Hour(SUPERLEARNER_SERVING_DST_LAG_MAX_H)]]
        @test !haskey(dst, anchor - Hour(SUPERLEARNER_SERVING_DST_LAG_MAX_H + 1))
        gapped = copy(frame)
        gapped.Dst[index[anchor - Hour(4)]] = NaN
        @test !haskey(superlearner_identity_dst_history(index, gapped, anchor), anchor - Hour(4))
    end

    @testset "super-learner anchor sample is deterministic and covers the storm branch" begin
        anchors = [DateTime(2025, 1, 8, 0) + Hour(k - 1) for k in 1:4000]
        complete = Set(anchors)
        latest = Dict(t => -10.0 for t in anchors)
        deep = anchors[3001:3100]
        for t in deep
            latest[t] = -250.0
        end
        first_call = superlearner_identity_anchor_selection(anchors, complete, latest, 500)
        @test first_call == superlearner_identity_anchor_selection(anchors, complete, latest, 500)
        @test issorted(first_call) && allunique(first_call)
        @test length(first_call) >= 500
        @test all(t -> t in first_call, deep)
    end

    @testset "super-learner identity reduction fails closed on non-finite deviations (V06-01)" begin
        deltas = DataFrame([Symbol("d_", c) => [0.0, 1e-12] for c in SUPERLEARNER_IDENTITY_COLUMNS])
        _, worst, worst_column = superlearner_identity_column_worst(deltas)
        @test worst == 1e-12 && worst <= SUPERLEARNER_IDENTITY_TOL_NT
        bad = SUPERLEARNER_IDENTITY_COLUMNS[2]
        deltas[2, Symbol("d_", bad)] = NaN
        per_column, worst, worst_column = superlearner_identity_column_worst(deltas)
        @test !isfinite(worst)
        @test worst_column == bad
        @test !(isfinite(worst) && worst <= SUPERLEARNER_IDENTITY_TOL_NT)
        @test !isfinite(only(r.max_abs_delta_nt for r in per_column if r.column == bad))
    end

    @testset "published oracle results" begin
        if !local_artifact_available("static stack served-identity table", STATIC_STACK_IDENTITY_OUT)
            @test_skip "run validation/operational/static_stack_served_identity.jl"
        else
            summary = CSV.read(STATIC_STACK_IDENTITY_OUT, DataFrame)
            pooled = summary[(summary.scope .== "all") .& (summary.key .== "pooled"), :]
            gate = summary[(summary.scope .== "all") .& (summary.key .== "coupling_gate"), :]
            @test nrow(pooled) == 1 && nrow(gate) == 1
            @test Int(pooled.n[1]) >= STATIC_STACK_IDENTITY_MIN_ROWS
            @test Float64(pooled.max_abs_delta_nt[1]) <= STATIC_STACK_IDENTITY_TOL_NT
            @test Float64(gate.max_abs_delta_nt[1]) <= STATIC_STACK_IDENTITY_TOL_NT
            steps = summary[summary.scope .== "model_step_hours", :]
            @test sort(parse.(Int, String.(steps.key))) == collect(ANALOG_SERVING_MODEL_STEPS)
            @test all(Float64.(steps.max_abs_delta_nt) .<= STATIC_STACK_IDENTITY_TOL_NT)
        end
        if !local_artifact_available("super-learner serving-identity table", SUPERLEARNER_IDENTITY_OUT)
            @test_skip "run validation/operational/superlearner_serving_identity.jl"
        else
            rows = CSV.read(SUPERLEARNER_IDENTITY_OUT, DataFrame)
            @test length(unique(rows.issue_time_utc)) >= 500
            @test sort(unique(Int.(rows.model_step_hours))) == collect(SUPERLEARNER_SERVING_MODEL_STEPS)
            # Every stage, not only the final center: an expert that stopped reproducing would
            # otherwise hide behind a stack weight that happens to be small.
            for column in SUPERLEARNER_IDENTITY_COLUMNS
                @test maximum(rows[!, Symbol("d_", column)]) <= SUPERLEARNER_IDENTITY_TOL_NT
            end
            # The deep cells and the deepening state must be exercised, or the identity is vacuous
            # for them.
            @test any(rows.deepening_cell)
            @test any(String.(rows.depth_bin) .== "deep")
            @test sort(unique(String.(rows.regime))) ==
                  sort(String.(collect(SUPERLEARNER_SERVING_REGIMES)))
            # The served bundle carries no point-forecast guard, so the published center is the stack
            # center on every row, including the deepening ones where a guard would have acted.
            @test !any(rows.guard_applied)
            @test all(rows.serving_superlearner_e_nt .== rows.serving_l1_center_nt)
            @test all(rows.half_width_nt .> 0)
        end
    end
end

end # module
