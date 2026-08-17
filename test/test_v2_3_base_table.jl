module V23BaseTableTests

# Oracles for the Operational V2.3 base table. The heavyweight identity check
# joins the rebuilt 2010--2025 table against the archived V2.2 development
# replay and runs only when both artifacts are present (both are generated,
# gitignored outputs); the labelling, causal-frame and static-stack oracles are
# self-contained and always run.

using Test
using SolarSINDy
using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "validation", "operational", "v2_3_base_table.jl"))

function _v23_tiny_table(; issue::DateTime=DateTime(2013, 5, 1, 0),
                         partition::String="DEV", block::Int=2013)
    rows = NamedTuple[]
    for h in V23_BASE_HORIZONS
        push!(rows, (
            issue_time_utc=issue,
            target_time_utc=issue + Hour(h),
            model_step_hours=h,
            partition=partition,
            cv_block=block,
            served_v2_1_dst_nt=-40.0 - h,
        ))
    end
    return DataFrame(rows)
end

@testset verbose=true "Operational V2.3 base table" begin

    @testset "Partition and cross-validation block labels" begin
        # Plan section 3 boundaries, hand-checked against the stated windows.
        @test v2_3_partition(DateTime(2010, 1, 1, 0)) == ("embargo", 0)
        @test v2_3_partition(DateTime(2010, 1, 1, 1)) == ("DEV", 0)
        @test v2_3_partition(DateTime(2012, 6, 15, 3)) == ("DEV", 0)
        @test v2_3_partition(DateTime(2013, 1, 1, 0)) == ("DEV", 2013)
        @test v2_3_partition(DateTime(2014, 8, 1, 0)) == ("DEV", 2014)
        @test v2_3_partition(DateTime(2015, 8, 1, 0)) == ("DEV", 2015)
        @test v2_3_partition(DateTime(2016, 8, 1, 0)) == ("DEV", 2016)
        @test v2_3_partition(DateTime(2017, 8, 1, 0)) == ("DEV", 2017)
        @test v2_3_partition(DateTime(2018, 8, 1, 0)) == ("DEV", 2018)
        @test v2_3_partition(DateTime(2019, 8, 1, 0)) == ("DEV", 2018)
        @test v2_3_partition(DateTime(2019, 12, 24, 16)) == ("DEV", 2018)
        @test v2_3_partition(DateTime(2019, 12, 24, 17)) == ("embargo", 0)
        @test v2_3_partition(DateTime(2020, 1, 7, 23)) == ("embargo", 0)
        @test v2_3_partition(DateTime(2020, 1, 8, 0)) == ("TEST", 0)
        @test v2_3_partition(DateTime(2025, 12, 24, 16)) == ("TEST", 0)
        @test v2_3_partition(DateTime(2025, 12, 24, 17)) == ("embargo", 0)

        # 2018 and 2019 share one block; every other year outside 2013--2017 is
        # archive-only.
        @test v2_3_cv_block(DateTime(2012, 12, 31, 23)) == 0
        @test v2_3_cv_block(DateTime(2017, 12, 31, 23)) == 2017
        @test v2_3_cv_block(DateTime(2019, 1, 1, 0)) == 2018
        @test v2_3_cv_block(DateTime(2020, 1, 1, 0)) == 0

        # Each partition's last issue sits exactly one 168 h embargo plus the
        # longest model step before its last target, and TEST opens one hour
        # after the DEV embargo closes.
        @test V23_DEV_LAST_ISSUE + Hour(V23_BASE_EMBARGO_HOURS + V23_BASE_MAX_STEP_HOURS) ==
              V23_DEV_LAST_TARGET
        @test V23_TEST_LAST_ISSUE + Hour(V23_BASE_EMBARGO_HOURS + V23_BASE_MAX_STEP_HOURS) ==
              V23_TEST_LAST_TARGET
        @test V23_TEST_FIRST_ISSUE - V23_DEV_LAST_TARGET ==
              Hour(V23_BASE_EMBARGO_HOURS + 1)

        # The script's own `--self-test` path exercises the same contract.
        @test _v23_selftest_partitions()
    end

    @testset "Causal hourly driver frame" begin
        times = DateTime(2020, 1, 1) .+ Hour.(0:5)
        cleaned = DataFrame(
            datetime=times,
            V=[NaN, 400.0, NaN, 500.0, NaN, 600.0],
            Bz=[-5.0, -6.0, NaN, -8.0, -9.0, -10.0],
            By=[1.0, NaN, 3.0, 4.0, NaN, 6.0],
            n=[5.0, 6.0, NaN, 8.0, 9.0, NaN],
            Dst=[-10.0, NaN, -30.0, -40.0, -50.0, -60.0],
        )
        frame = v2_3_hourly_frame(cleaned)

        @test names(frame) == ["time_utc", "V", "Bz", "By", "n", "Pdyn", "Dst"]
        @test frame.time_utc == times
        # Last observation carried forward, never backward.
        @test isnan(frame.V[1])
        @test frame.V[2:end] == [400.0, 400.0, 500.0, 500.0, 600.0]
        @test frame.Bz == [-5.0, -6.0, -6.0, -8.0, -9.0, -10.0]
        @test frame.By == [1.0, 1.0, 3.0, 4.0, 4.0, 6.0]
        @test frame.n == [5.0, 6.0, 6.0, 8.0, 9.0, 9.0]
        # Independent proton-only pressure identity 1.6726e-6 n V^2 [nPa];
        # the tolerance is the rounding budget of one double-precision product.
        @test isnan(frame.Pdyn[1])
        for i in 2:6
            @test isapprox(frame.Pdyn[i], 1.6726e-6 * frame.n[i] * frame.V[i]^2;
                           rtol=1e-12, atol=0.0)
        end
        # Dst is an observation, so a gap must stay a gap.
        @test isnan(frame.Dst[2])
        @test frame.Dst[[1, 3, 4, 5, 6]] == [-10.0, -30.0, -40.0, -50.0, -60.0]

        lookup = v2_3_frame_driver_lookup(frame)
        @test length(lookup) == 5
        @test !haskey(lookup, times[1])
        @test lookup[times[4]] == (V=500.0, Bz=-8.0, By=4.0, n=8.0, Pdyn=frame.Pdyn[4])

        # Appending a later record cannot change an earlier row: the fill is
        # strictly causal.
        extended = vcat(cleaned, DataFrame(
            datetime=[times[6] + Hour(1)], V=[900.0], Bz=[-1.0], By=[0.0],
            n=[1.0], Dst=[-70.0],
        ))
        extended_frame = v2_3_hourly_frame(extended)
        # `isequal` so the deliberate NaN gaps compare as identical.
        @test isequal(extended_frame[1:6, :], frame)

        # Malformed inputs fail closed instead of silently mis-filling.
        @test_throws ArgumentError v2_3_hourly_frame(cleaned[[2, 1, 3, 4, 5, 6], :])
        @test_throws ArgumentError v2_3_hourly_frame(vcat(cleaned, cleaned[6:6, :]))
        @test_throws ArgumentError v2_3_hourly_frame(select(cleaned, Not(:Dst)))

        @test _v23_selftest_hourly_frame()
    end

    @testset "Static V2.2 stack column" begin
        pooled = OperationalV22Cell(
            6, :pooled, 60, [0.30, 0.30, 0.10, 0.10, 0.10, 0.10];
            objective_mse=2.0, iterations=10,
        )
        recovery = OperationalV22Cell(
            6, :recovery, 60, [0.40, 0.20, 0.20, 0.10, 0.05, 0.05];
            objective_mse=1.0, iterations=9,
        )
        stack = OperationalV22Stack(
            [pooled, recovery]; label="v2.3-base-test", minimum_cell_rows=48,
        )
        centers = (served_v2_1=-100.0, frozen_v2_1=-90.0, persistence=-80.0,
                   burton=-70.0, burton_full=-60.0, obrien=-50.0)

        # Disturbed and no longer falling selects the recovery cell:
        # -40 - 18 - 16 - 7 - 3 - 2.5 = -86.5 nT.
        @test isapprox(v2_3_static_v22_center(stack, 6, -50.0, 1.0, 0.0, centers),
                       -86.5; atol=1e-12, rtol=0.0)
        # Still falling selects active deepening, which has no cell at this lead,
        # so the pooled fallback applies: -30 - 27 - 8 - 7 - 6 - 5 = -83.0 nT.
        @test isapprox(v2_3_static_v22_center(stack, 6, -50.0, -1.0, 0.0, centers),
                       -83.0; atol=1e-12, rtol=0.0)
        # Quiet issue-time state also falls back to pooled at this lead.
        @test isapprox(v2_3_static_v22_center(stack, 6, -10.0, 0.0, 0.0, centers),
                       -83.0; atol=1e-12, rtol=0.0)
        # The blend must read the components, not a fixed order-independent sum.
        swapped = (served_v2_1=-90.0, frozen_v2_1=-100.0, persistence=-80.0,
                   burton=-70.0, burton_full=-60.0, obrien=-50.0)
        @test !isapprox(v2_3_static_v22_center(stack, 6, -50.0, 1.0, 0.0, swapped),
                        -86.5; atol=1e-9, rtol=0.0)
        @test_throws ArgumentError v2_3_static_v22_center(
            stack, 3, -50.0, 1.0, 0.0, centers,
        )

        @test _v23_selftest_static_v22()
    end

    @testset "Base-table invariants" begin
        table = _v23_tiny_table()
        @test v2_3_base_table_invariants(table; year_end=2013) === table

        duplicated = vcat(table, table[1:1, :])
        @test_throws ErrorException v2_3_base_table_invariants(
            duplicated; year_end=2013,
        )

        shifted = copy(table)
        shifted[1, :target_time_utc] = shifted[1, :issue_time_utc] + Hour(2)
        @test_throws ErrorException v2_3_base_table_invariants(shifted; year_end=2013)

        mislabelled = copy(table)
        mislabelled[1, :partition] = "TEST"
        @test_throws ErrorException v2_3_base_table_invariants(
            mislabelled; year_end=2013,
        )

        wrong_block = copy(table)
        wrong_block[!, :cv_block] = fill(2014, nrow(wrong_block))
        @test_throws ErrorException v2_3_base_table_invariants(
            wrong_block; year_end=2013,
        )

        non_finite = copy(table)
        non_finite[1, :served_v2_1_dst_nt] = NaN
        @test_throws ErrorException v2_3_base_table_invariants(
            non_finite; year_end=2013,
        )

        short_steps = table[table.model_step_hours .!= 4, :]
        @test_throws ErrorException v2_3_base_table_invariants(
            short_steps; year_end=2013,
        )

        # A target beyond the declared final year is rejected.
        @test_throws ErrorException v2_3_base_table_invariants(
            _v23_tiny_table(; issue=DateTime(2013, 12, 31, 23)), year_end=2013,
        )
    end

    @testset "2010--2022 identity against the V2.2 development replay" begin
        if !(isfile(V23_BASE_TABLE) && isfile(V23_V22_ORACLE_TABLE))
            @info "V2.3 base identity oracle skipped: generated artifact missing" table =
                V23_BASE_TABLE oracle = V23_V22_ORACLE_TABLE
        else
            oracle = v2_3_base_oracle_join()
            @test oracle.oracle_rows == V23_ORACLE_EXPECTED_ROWS
            @test oracle.joined_rows == V23_ORACLE_EXPECTED_ROWS
            @test oracle.rebuilt_rows ==
                  V23_ORACLE_EXPECTED_ROWS + V23_ORACLE_EXPECTED_EXTRA_ROWS
            # The rebuilt superset decomposes exactly as the archived table's own
            # construction predicts: 23 rows whose target crosses into 2023 (the
            # archive stopped at 2022), and 2 x 7 anchors x 6 model steps that
            # the purged V2.1 calibration split discarded at its two boundaries.
            # The 2022-12-31T23 anchor is in both groups (all six of its targets
            # cross into 2023), so 84 + 6 = 90 rows sit on an unarchived issue.
            @test oracle.extra_rows == V23_ORACLE_EXPECTED_EXTRA_ROWS
            @test oracle.extra_beyond_archived_horizon == 23
            @test oracle.extra_unarchived_issue == 90
            @test oracle.extra_explained == oracle.extra_rows
            @test all(<=(V23_ORACLE_ATOL_NT), oracle.differences.max_abs_difference)
            @test oracle.max_abs_difference <= V23_ORACLE_ATOL_NT
            @test Set(oracle.differences.column) ==
                  Set(String.(V23_ORACLE_SHARED_COLUMNS))
            @test oracle.agrees

            rebuilt = CSV.read(
                V23_BASE_TABLE, DataFrame;
                types=Dict("issue_time_utc" => DateTime,
                           "target_time_utc" => DateTime),
            )
            @test v2_3_base_table_invariants(rebuilt) === rebuilt
            @test maximum(rebuilt.target_time_utc) <= DateTime(2025, 12, 31, 23)
            @test minimum(rebuilt.issue_time_utc) >= DateTime(2010, 1, 1, 0)
            @test count(==("DEV"), rebuilt.partition) > 0
            @test count(==("TEST"), rebuilt.partition) > 0
            @test all(rebuilt.cv_block[rebuilt.partition .!= "DEV"] .== 0)
            @test Set(rebuilt.cv_block[rebuilt.partition .== "DEV"]) ⊆
                  Set([0, V23_CV_BLOCK_LABELS...])
            for column in String.(V23_ORACLE_SHARED_COLUMNS)
                @test column in names(rebuilt)
            end
            @test "static_v2_2_dst_nt" in names(rebuilt)
            @test "pred_dst_nt" in names(rebuilt)

            if isfile(V23_BASE_HOURLY_FRAME)
                frame = CSV.read(
                    V23_BASE_HOURLY_FRAME, DataFrame;
                    types=Dict("time_utc" => DateTime),
                )
                @test names(frame) ==
                      ["time_utc", "V", "Bz", "By", "n", "Pdyn", "Dst"]
                @test issorted(frame.time_utc)
                @test allunique(frame.time_utc)
                @test all(diff(frame.time_utc) .== Hour(1))
                # Every issue of the base table is an hourly frame record with a
                # finite Dst, and the frame drivers are the replay drivers.
                stamps = Set(frame.time_utc)
                @test all(in(stamps), unique(rebuilt.issue_time_utc))
                dst_at = Dict(zip(frame.time_utc, frame.Dst))
                sample = rebuilt[1:max(1, nrow(rebuilt) ÷ 5000):end, :]
                @test all(
                    dst_at[sample.issue_time_utc[i]] == sample.latest_dst_nt[i]
                    for i in 1:nrow(sample)
                )
                # The issue-time driver of row i is the complete hour [t-1, t),
                # so it must equal the hourly-frame record at t - 1 h exactly.
                # This is the join Task B's feature builder relies on.
                index_at = Dict(zip(frame.time_utc, 1:nrow(frame)))
                @test all(
                    let j = index_at[sample.issue_time_utc[i] - Hour(1)]
                        frame.V[j] == sample.V_kms[i] &&
                        frame.Bz[j] == sample.Bz_nt[i] &&
                        frame.By[j] == sample.By_nt[i] &&
                        frame.n[j] == sample.n_cm3[i] &&
                        frame.Pdyn[j] == sample.Pdyn_npa[i]
                    end
                    for i in 1:nrow(sample)
                )
            end
        end
    end
end

@testset "the DEV/TEST embargo is nominal 168 h and realised 337 h" begin
    # The partition constants embargo 168 h of targets nominally; the realised
    # gap is longer because the last DEV issue stops a week before the nominal
    # last DEV target, and the manifest records the realised number so a reader
    # does not have to re-derive it.
    @test V23_TEST_FIRST_ISSUE - V23_DEV_LAST_TARGET == Hour(V23_BASE_EMBARGO_HOURS + 1)
    @test v23_realised_embargo_hours() == 337
    @test v23_realised_embargo_hours() >= V23_BASE_EMBARGO_HOURS
    @test V23_DEV_LAST_ISSUE + Hour(V23_BASE_MAX_STEP_HOURS) == DateTime(2019, 12, 24, 23)
    @test _v23_selftest_partitions()
end

end # module
