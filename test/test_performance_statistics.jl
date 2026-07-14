using Statistics

@testset "Paired whole-storm performance statistics" begin
    model = [(storm_id=2, rmse=6.0), (storm_id=1, rmse=1.0)]
    reference = [(storm_id=1, rmse=2.0), (storm_id=2, rmse=3.0)]
    result = paired_storm_statistics(model, reference; seed=17)

    @test result.draws == 10_000
    @test result.seed == 17
    @test [row.storm_id for row in result.pair_records] == ["1", "2"]
    @test [row.rmse_difference_nt for row in result.pair_records] == [-1.0, 3.0]
    @test [row.relative_difference_fraction for row in result.pair_records] == [-0.5, 1.0]
    @test result.mean_rmse_difference == 1.0
    @test result.mean_relative_difference == 0.25
    frame_result = paired_storm_statistics(DataFrame(model), DataFrame(reference);
                                             draws=20, seed=17)
    @test frame_result.pair_records == result.pair_records

    # Independent exhaustive oracle: with two storms, the four ordered size-two
    # resamples have absolute-effect means {-1, 1, 1, 3} and relative-effect
    # means {-0.5, 0.25, 0.25, 1}. The 95% percentile limits are the extrema.
    enumerated_rmse = [-1.0, 1.0, 1.0, 3.0]
    enumerated_relative = [-0.5, 0.25, 0.25, 1.0]
    @test Set(row.mean_rmse_difference_nt for row in result.bootstrap_records) ==
          Set(enumerated_rmse)
    @test Set(row.mean_relative_difference_fraction for row in result.bootstrap_records) ==
          Set(enumerated_relative)
    @test Set((row.mean_rmse_difference_nt,
               row.mean_relative_difference_fraction)
              for row in result.bootstrap_records) ==
          Set(((-1.0, -0.5), (1.0, 0.25), (3.0, 1.0)))
    @test result.rmse_difference_interval == (-1.0, 3.0)
    @test result.relative_difference_interval == (-0.5, 1.0)
    @test mean(row.mean_rmse_difference_nt for row in result.bootstrap_records) ≈
          mean(enumerated_rmse) atol=0.04
    @test mean(row.mean_relative_difference_fraction for row in result.bootstrap_records) ≈
          mean(enumerated_relative) atol=0.02
    @test result.wilcoxon == wilcoxon_signed_rank_p([-1.0, 3.0])

    # Two individually finite effects have an overflowing direct sum even
    # though their mean is exactly representable. Both the estimate and every
    # paired-bootstrap draw must retain the finite oracle value.
    extreme_model = [(storm_id=1, rmse=0.0), (storm_id=2, rmse=0.0)]
    extreme_reference = [
        (storm_id=1, rmse=floatmax(Float64)),
        (storm_id=2, rmse=floatmax(Float64)),
    ]
    extreme = paired_storm_statistics(
        extreme_model, extreme_reference; draws=32, seed=17,
    )
    @test extreme.mean_rmse_difference == -floatmax(Float64)
    @test extreme.rmse_difference_interval ==
          (-floatmax(Float64), -floatmax(Float64))
    @test all(
        row.mean_rmse_difference_nt == -floatmax(Float64)
        for row in extreme.bootstrap_records
    )
    @test extreme.mean_relative_difference == -1.0
    @test extreme.relative_difference_interval == (-1.0, -1.0)

    # Pairing is keyed by storm id: row order is irrelevant, while changing a
    # value's id changes the paired effect and changing the id set fails closed.
    reordered = paired_storm_statistics(reverse(model), reverse(reference); seed=17)
    @test reordered.pair_records == result.pair_records
    @test reordered.bootstrap_records == result.bootstrap_records
    reassigned = [(storm_id=1, rmse=4.0), (storm_id=2, rmse=2.0)]
    @test paired_storm_statistics(model, reassigned; draws=20, seed=17).
          mean_rmse_difference != result.mean_rmse_difference
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=2.0), (storm_id=3, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=2.0), (storm_id=1, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(
        [(storm_id=1, rmse=1.0), (storm_id=1, rmse=2.0)], reference; draws=20)

    repeated = paired_storm_statistics(model, reference; seed=17)
    different_seed = paired_storm_statistics(model, reference; draws=100, seed=18)
    @test repeated.bootstrap_records == result.bootstrap_records
    @test repeated.summary_records == result.summary_records
    @test different_seed.mean_rmse_difference == result.mean_rmse_difference
    @test different_seed.bootstrap_records != result.bootstrap_records[1:100]

    @test_throws ArgumentError paired_storm_statistics(model, reference; draws=0)
    @test_throws ArgumentError paired_storm_statistics(model, reference; draws=1.5)
    @test_throws ArgumentError paired_storm_statistics(model, reference; coverage=1.0)
    @test_throws ArgumentError paired_storm_statistics(model, reference; coverage=NaN)
    @test_throws ArgumentError paired_storm_statistics(model, reference; seed=-1)
    @test_throws ArgumentError paired_storm_statistics(model, reference; seed=1.5)
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=0.0), (storm_id=2, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=-1.0), (storm_id=2, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=missing), (storm_id=2, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(model,
        [(storm_id=1, rmse=NaN), (storm_id=2, rmse=3.0)]; draws=20)
    @test_throws ArgumentError paired_storm_statistics(
        [(storm_id=1, rmse=false), (storm_id=2, rmse=6.0)], reference; draws=20)
    @test_throws ArgumentError paired_storm_statistics(
        [(storm_id=1, rmse=-1.0), (storm_id=2, rmse=6.0)], reference; draws=20)

    holm = holm_adjust([0.01, 0.04, 0.03, 0.002]; labels=["a", "b", "c", "d"])
    @test [row.holm_p_value for row in holm] ≈ [0.03, 0.06, 0.06, 0.008]
    @test [row.holm_rank for row in holm] == [2, 4, 3, 1]
    tied = holm_adjust([0.01, 0.01, 0.2]; labels=["first", "second", "third"])
    @test [row.holm_rank for row in tied] == [1, 2, 3]
    @test [row.holm_p_value for row in tied] ≈ [0.03, 0.03, 0.2]
    @test_throws ArgumentError holm_adjust(Float64[])
    @test_throws ArgumentError holm_adjust([0.1, NaN])
    @test_throws ArgumentError holm_adjust([0.1, 1.1])
    @test_throws ArgumentError holm_adjust([false, true])
    @test_throws DimensionMismatch holm_adjust([0.1, 0.2]; labels=["a"])
    @test_throws ArgumentError holm_adjust([0.1, 0.2]; labels=["a", "a"])

    mktempdir() do first_root
        mktempdir() do second_root
            first_paths = write_paired_storm_statistics(result, first_root; prefix="oracle")
            second_paths = write_paired_storm_statistics(repeated, second_root; prefix="oracle")
            for field in propertynames(first_paths)
                first_path = getproperty(first_paths, field)
                second_path = getproperty(second_paths, field)
                @test read(first_path) == read(second_path)
            end
            first_holm = write_holm_adjustment(holm, first_root; prefix="headline")
            second_holm = write_holm_adjustment(holm, second_root; prefix="headline")
            @test read(first_holm) == read(second_holm)
            summary = CSV.read(first_paths.summary, DataFrame)
            @test summary.bootstrap_draws == fill(10_000, 2)
            @test summary.seed == fill(17, 2)
            @test Set(summary.unit) == Set(("nT", "fraction"))
            pairs = CSV.read(first_paths.pairs, DataFrame)
            @test Set((:model_rmse_nt, :reference_rmse_nt,
                       :rmse_difference_nt, :relative_difference_fraction)) <=
                  Set(propertynames(pairs))
            bootstrap = CSV.read(first_paths.bootstrap, DataFrame)
            @test Set((:mean_rmse_difference_nt,
                       :mean_relative_difference_fraction)) <=
                  Set(propertynames(bootstrap))
        end
    end

    mktempdir() do root
        prefix = "rollback"
        pairs = joinpath(root, "$(prefix)_pairs.csv")
        collision = joinpath(root, "$(prefix)_bootstrap.csv")
        write(pairs, "sentinel\n")
        mkdir(collision)
        keep = joinpath(collision, "keep")
        write(keep, "preserve")
        @test_throws ArgumentError write_paired_storm_statistics(
            result, root; prefix,
        )
        @test read(pairs, String) == "sentinel\n"
        @test isdir(collision)
        @test read(keep, String) == "preserve"
        @test !ispath(joinpath(root, "$(prefix)_summary.csv"))
        @test !ispath(joinpath(root, "$(prefix)_wilcoxon.csv"))
    end
end
