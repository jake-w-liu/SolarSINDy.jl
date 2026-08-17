module ServingIdentityOracleTests

# Unit oracles for the two offline identity scripts
# (`validation/operational/v2_2_served_identity.jl`,
#  `validation/operational/v2_3_serving_identity.jl`).
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
                          "v2_2_served_identity.jl")))
include(normpath(joinpath(@__DIR__, "..", "validation", "operational",
                          "v2_3_serving_identity.jl")))

const V22_IDENTITY_OUT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_2_served_identity.csv")
const V23_IDENTITY_OUT = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_3_serving_identity.csv")

@testset "Serving identity oracles" begin
    @testset "declared base-table column contracts are complete" begin
        v22 = v22_identity_table_columns()
        # Every component the stack multiplies, plus both regime inputs and the archived answer.
        for column in ("served_v2_1_dst_nt", "frozen_v2_1_dst_nt", "persistence_dst_nt",
                       "burton_dst_nt", "burton_full_dst_nt", "obrien_dst_nt",
                       "latest_dst_nt", "dst_delta_1h_nt", "VBsouth_mvm",
                       "coupling_active_mvm", "static_v2_2_dst_nt", "model_step_hours",
                       "partition")
            @test column in v22
        end
        @test allunique(v22)

        v23 = v23_identity_table_columns()
        # The analog correction needs the issue-time state, the six memory lags, the baseline panel
        # and the frozen center; a missing column would silently change the correction.
        for column in ("latest_dst_nt", "V_kms", "Bz_nt", "By_nt", "n_cm3", "Pdyn_npa",
                       "dst_delta_1h_nt", "dst_delta_3h_nt", "Bz_delta_1h_nt",
                       "VBsouth_delta_1h_mvm", "VBsouth_mean_3h_mvm", "Bsouth_mean_3h_nt",
                       "persistence_dst_nt", "burton_dst_nt", "burton_full_dst_nt",
                       "obrien_dst_nt", "frozen_v2_1_dst_nt", "served_v2_1_dst_nt")
            @test column in v23
        end
        @test allunique(v23)
        @test V22_IDENTITY_TOL_NT == 1e-9
        @test V23_IDENTITY_TOL_NT == 1e-9
        @test V22_IDENTITY_MIN_ROWS >= 5_000
        @test V23_IDENTITY_HISTORY_LAGS == V23_SOUTH_RUN_CAP_H
    end

    @testset "driver history is built in lag order and reports gaps" begin
        times = [DateTime(2020, 5, 1, 0) + Hour(k - 1) for k in 1:40]
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
        history = v23_identity_driver_history(index, frame, anchor)
        @test length(history) == V23_IDENTITY_HISTORY_LAGS
        # Lag j must be the record tagged anchor - j hours, in that order.
        for lag in 1:V23_IDENTITY_HISTORY_LAGS
            row = index[anchor - Hour(lag)]
            @test history[lag].V == frame.V[row]
            @test history[lag].Bz == frame.Bz[row]
            @test history[lag].Pdyn == frame.Pdyn[row]
        end
        # An absent record and a non-positive density are both reported as unavailable rather than
        # substituted, because an imputed key would change which archive origins are retrieved.
        early = v23_identity_driver_history(index, frame, times[3])
        @test early[1] !== nothing
        @test early[3] === nothing
        holed = copy(frame)
        holed.n[index[anchor - Hour(2)]] = 0.0
        @test v23_identity_driver_history(index, holed, anchor)[2] === nothing
        nanned = copy(frame)
        nanned.Bz[index[anchor - Hour(5)]] = NaN
        @test v23_identity_driver_history(index, nanned, anchor)[5] === nothing
    end

    @testset "anchor sample is deterministic and covers the storm branch" begin
        anchors = [DateTime(2020, 1, 8, 0) + Hour(k - 1) for k in 1:4000]
        complete = Set(anchors)
        # A deep tail well away from the strided positions, so the storm union is observable.
        latest = Dict(t => -10.0 for t in anchors)
        deep = anchors[3001:3050]
        for t in deep
            latest[t] = -250.0
        end
        first_call = v23_identity_anchor_selection(anchors, complete, latest, 500)
        second_call = v23_identity_anchor_selection(anchors, complete, latest, 500)
        @test first_call == second_call
        @test issorted(first_call)
        @test allunique(first_call)
        @test length(first_call) >= 500
        @test all(t -> t in complete, first_call)
        # Every deep anchor must be sampled; a stride alone would have missed most of them.
        @test all(t -> t in first_call, deep)
        # Anchors without all six steps are never sampled.
        partial = Set(anchors[1:2000])
        restricted = v23_identity_anchor_selection(anchors, partial, latest, 100)
        @test all(t -> t in partial, restricted)
    end

    @testset "published oracle results" begin
        if !isfile(V22_IDENTITY_OUT)
            @test_skip "run validation/operational/v2_2_served_identity.jl"
        else
            summary = CSV.read(V22_IDENTITY_OUT, DataFrame)
            pooled = summary[(summary.scope .== "all") .& (summary.key .== "pooled"), :]
            gate = summary[(summary.scope .== "all") .& (summary.key .== "coupling_gate"), :]
            @test nrow(pooled) == 1 && nrow(gate) == 1
            @test Int(pooled.n[1]) >= V22_IDENTITY_MIN_ROWS
            @test Float64(pooled.max_abs_delta_nt[1]) <= V22_IDENTITY_TOL_NT
            @test Float64(gate.max_abs_delta_nt[1]) <= V22_IDENTITY_TOL_NT
            steps = summary[summary.scope .== "model_step_hours", :]
            @test sort(parse.(Int, String.(steps.key))) == collect(V23_SERVING_MODEL_STEPS)
            @test all(Float64.(steps.max_abs_delta_nt) .<= V22_IDENTITY_TOL_NT)
        end
        if !isfile(V23_IDENTITY_OUT)
            @test_skip "run validation/operational/v2_3_serving_identity.jl"
        else
            rows = CSV.read(V23_IDENTITY_OUT, DataFrame)
            @test length(unique(rows.issue_time_utc)) >= 500
            @test sort(unique(Int.(rows.model_step_hours))) == collect(V23_SERVING_MODEL_STEPS)
            @test maximum(rows.abs_delta_nt) <= V23_IDENTITY_TOL_NT
            @test maximum(rows.abs_delta_lat_nt) <= V23_IDENTITY_TOL_NT
            @test maximum(rows.abs_delta_frozen_nt) <= V23_IDENTITY_TOL_NT
            # The error layer must have acted somewhere, or the identity would be vacuous for it.
            @test any(rows.e_layer_applied)
            layered = rows[rows.e_layer_applied, :]
            @test all(abs.(layered.e_delta_nt) .<=
                      [v23_serving_e_cap(h) for h in layered.model_step_hours])
        end
    end
end

end # module
