module V22CrossfitTests

using Test
using SolarSINDy
using DataFrames
using Dates
using LinearAlgebra

const CROSSFIT_IMPORT_TEXT = mktemp() do _, io
    redirect_stdout(io) do
        include(joinpath(
            @__DIR__, "..", "validation", "operational", "v2_2_crossfit.jl",
        ))
    end
    flush(io)
    seekstart(io)
    read(io, String)
end

function _v22_crossfit_synthetic()
    issues = DateTime[]
    for year in 2010:2012, day in 1:20
        push!(issues, DateTime(year, 6, day, 12))
    end
    for year in 2013:2016
        append!(issues, (DateTime(year, 1, 8), DateTime(year, 12, 24, 23)))
    end
    append!(issues, (DateTime(2017, 1, 8), V22_CROSSFIT_POINT_FIT_END))
    for block in V22_CROSSFIT_BLOCKS
        # With h=7, these anchors place the last target exactly 168 h and
        # exactly 167 h before the retained block, respectively.
        append!(issues, (block.start - Hour(175), block.start - Hour(174)))
    end
    sort!(issues)

    rows = NamedTuple[]
    truth = [0.34, 0.31, 0.12, 0.09, 0.08, 0.06]
    for (anchor_index, issue) in enumerate(issues)
        for lead in V22_CROSSFIT_MODEL_STEPS
            phase = 0.13 * anchor_index + 0.11 * lead
            centers = [
                -25.0 + 7.0sin(phase),
                -20.0 + 6.0cos(1.7phase),
                -18.0 + 5.0sin(2.3phase + 0.2lead),
                -22.0 + 4.0cos(2.9phase + lead),
                -24.0 + 3.0sin(3.7phase - 0.2lead),
                -21.0 + 2.0cos(4.3phase) - 0.1lead,
            ]
            push!(rows, (
                issue_time_utc=issue,
                target_time_utc=issue + Hour(lead),
                model_step_hours=lead,
                split_label="fit",
                served_v2_1_dst_nt=centers[1],
                frozen_v2_1_dst_nt=centers[2],
                persistence_dst_nt=centers[3],
                burton_dst_nt=centers[4],
                burton_full_dst_nt=centers[5],
                obrien_dst_nt=centers[6],
                observation_dst_nt=dot(truth, centers),
                latest_dst_nt=-10.0,
                dst_delta_1h_nt=0.0,
                coupling_active_mvm=0.0,
            ))
        end
    end
    table = DataFrame(rows)
    n = nrow(table)
    for (j, feature) in enumerate(OPERATIONAL_V22_RESIDUAL_FEATURES)
        String(feature) in names(table) && continue
        table[!, feature] = 0.01j .+ (1:n) ./ (n + j)
    end
    return table
end

@testset verbose=true "V2.2 leakage-safe primary cross-fit" begin
    @testset "silent import, pinned lineage, and calendar blocks" begin
        @test isempty(CROSSFIT_IMPORT_TEXT)
        @test _v22_crossfit_feature_schema_sha256() ==
              V22_CROSSFIT_INPUT_HASHES.feature_schema_sha256
        @test V22_CROSSFIT_INPUT_HASHES.residual_replay_sha256 ==
              "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f"
        @test first(V22_CROSSFIT_BLOCKS).start == DateTime(2013, 1, 8)
        @test first(V22_CROSSFIT_BLOCKS).stop == DateTime(2013, 12, 24, 23)
        @test last(V22_CROSSFIT_BLOCKS).start == DateTime(2017, 1, 8)
        @test last(V22_CROSSFIT_BLOCKS).stop == V22_CROSSFIT_POINT_FIT_END
        @test all(
            _v22_crossfit_elapsed_hours(
                V22_CROSSFIT_BLOCKS[i].start,
                V22_CROSSFIT_BLOCKS[i - 1].stop,
            ) >= 168 for i in 2:length(V22_CROSSFIT_BLOCKS)
        )
        @test isnothing(_v22_crossfit_validate_blocks(V22_CROSSFIT_BLOCKS))
        too_close = (
            (label="a", start=DateTime(2013, 1, 8),
             stop=DateTime(2013, 12, 24, 23)),
            (label="b", start=DateTime(2013, 12, 31),
             stop=DateTime(2013, 12, 31, 23)),
        )
        @test_throws ArgumentError _v22_crossfit_validate_blocks(too_close)
    end

    @testset "whole anchors and split-selection mutation invariance" begin
        table = _v22_crossfit_synthetic()
        block = first(V22_CROSSFIT_BLOCKS)
        original = v2_2_crossfit_partition(table, block)
        mutated = copy(table)
        purged_or_later = mutated.target_time_utc .>
                          block.start - Hour(V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS)
        mutated.observation_dst_nt[purged_or_later] .= 9.0e8
        for column in values(DEFAULT_OPERATIONAL_V22_COMPONENT_COLUMNS)
            mutated[purged_or_later, column] .= -8.0e8
        end
        for feature in OPERATIONAL_V22_RESIDUAL_FEATURES
            mutated[purged_or_later, feature] .= 7.0e8
        end
        changed = v2_2_crossfit_partition(mutated, block)
        @test original.fit_keys == changed.fit_keys
        @test original.oof_keys == changed.oof_keys
        @test maximum(original.fit_keys.target_time_utc) ==
              block.start - Hour(V22_CROSSFIT_MINIMUM_BLOCK_GAP_HOURS)
        @test minimum(original.oof_keys.issue_time_utc) == block.start
        exact_boundary_issue = block.start - Hour(175)
        purged_boundary_issue = block.start - Hour(174)
        @test all(step -> (
            exact_boundary_issue,
            exact_boundary_issue + Hour(step),
            step,
        ) in _v22_crossfit_key_set(original.fit_keys), V22_CROSSFIT_MODEL_STEPS)
        @test all(original.fit_keys.issue_time_utc .!= purged_boundary_issue)

        removed = (table.issue_time_utc .== block.start) .&
                  (table.model_step_hours .== maximum(V22_CROSSFIT_MODEL_STEPS))
        incomplete = table[.!removed, :]
        @test_throws ArgumentError v2_2_crossfit_partition(incomplete, block)
        duplicate = vcat(table, table[1:1, :])
        @test_throws ArgumentError v2_2_crossfit_partition(duplicate, block)
    end

    @testset "expanding OOF fit chronology and exact key coverage" begin
        table = _v22_crossfit_synthetic()
        result = build_v2_2_primary_crossfit(table; minimum_cell_rows=12)
        expected_keys = Tuple{DateTime,DateTime,Int}[]
        for block in V22_CROSSFIT_BLOCKS
            rows = table[
                (table.issue_time_utc .>= block.start) .&
                (table.issue_time_utc .<= block.stop),
                :,
            ]
            append!(expected_keys, collect(zip(
                rows.issue_time_utc, rows.target_time_utc, rows.model_step_hours,
            )))
        end
        actual_keys = collect(zip(
            result.oof.issue_time_utc,
            result.oof.target_time_utc,
            result.oof.model_step_hours,
        ))
        @test length(actual_keys) == length(unique(actual_keys))
        @test Set(actual_keys) == Set(expected_keys)
        @test Set(result.oof.v2_2_crossfit_fold) ==
              Set(block.label for block in V22_CROSSFIT_BLOCKS)
        @test nrow(result.fold_audit) == length(V22_CROSSFIT_BLOCKS)
        @test all(result.fold_audit.fit_issue_max_utc .<
                  result.fold_audit.block_start_utc)
        @test all(result.fold_audit.fit_target_max_utc .<
                  result.fold_audit.block_start_utc)
        @test all(result.fold_audit.fit_target_gap_hours .== 168)
        @test issorted(result.fold_audit.fit_rows)
        @test all(result.fold_audit.post_2022_rows_read .== 0)

        for block in V22_CROSSFIT_BLOCKS
            fold = result.oof[result.oof.v2_2_crossfit_fold .== block.label, :]
            audit = only(eachrow(result.fold_audit[
                result.fold_audit.fold .== block.label, :,
            ]))
            @test all(fold.v2_2_crossfit_fit_issue_max_utc .==
                      audit.fit_issue_max_utc)
            @test all(fold.v2_2_crossfit_fit_target_max_utc .==
                      audit.fit_target_max_utc)
            @test all(fold.v2_2_crossfit_fit_target_gap_hours .==
                      audit.fit_target_gap_hours)
            @test audit.fit_issue_max_utc < block.start
            @test audit.fit_target_gap_hours >= 168
            @test result.stacks[block.label].label ==
                  "v2_2_primary_crossfit_$(block.label)_purge168h"
        end

        @test result.oof.primary_minus_served_v2_1_nt ==
              result.oof.v2_2_pred_dst_nt .- result.oof.served_v2_1_dst_nt
        @test result.oof.primary_minus_frozen_v2_1_nt ==
              result.oof.v2_2_pred_dst_nt .- result.oof.frozen_v2_1_dst_nt
        @test result.oof.primary_minus_persistence_nt ==
              result.oof.v2_2_pred_dst_nt .- result.oof.persistence_dst_nt
        @test result.oof.primary_minus_burton_full_nt ==
              result.oof.v2_2_pred_dst_nt .- result.oof.burton_full_dst_nt
        @test result.oof.primary_minus_obrien_nt ==
              result.oof.v2_2_pred_dst_nt .- result.oof.obrien_dst_nt
        @test all(isfinite, result.oof.v2_2_pred_dst_nt)
    end
end

end # module
