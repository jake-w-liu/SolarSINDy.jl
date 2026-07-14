include(joinpath(@__DIR__, "..", "validation", "significance_tests.jl"))
using Dates

struct _InterruptingMetric <: Real end
(::Type{Float64})(::_InterruptingMetric) = throw(InterruptException())

function _significance_fixture()
    holdout_rows = NamedTuple[]
    for (storm_id, reference_rmse, difference) in
        zip(("h1", "h2", "h3"), (3.0, 4.0, 6.5), (-1.0, 0.5, -1.5))
        for (model, value) in (
            ("SINDy", reference_rmse + difference),
            ("OBrienMcP", reference_rmse),
            ("Burton", reference_rmse + 1.0),
            ("BurtonFull", reference_rmse + 0.5),
        )
            push!(holdout_rows, (;
                storm_id, model, rmse_nt=value, mae_nt=0.8 * value,
            ))
        end
    end
    holdout = DataFrame(holdout_rows)

    cross_rows = NamedTuple[]
    experiments = ("C20-22->C23", "even->odd", "C20-23->C25")
    differences = ((-1.0, -0.5, 0.25), (-0.8, 0.4, -0.2), (0.7, -0.3, 0.1))
    for (experiment_index, experiment) in enumerate(experiments)
        for storm_index in 1:3
            storm_id = "x$(experiment_index)_$(storm_index)"
            reference_rmse = 3.0 + storm_index + experiment_index / 10
            push!(cross_rows, (
                experiment=experiment,
                storm_id=storm_id,
                model="SINDy",
                rmse_nt=reference_rmse + differences[experiment_index][storm_index],
                mae_nt=0.8 * (reference_rmse + differences[experiment_index][storm_index]),
            ))
            push!(cross_rows, (
                experiment=experiment,
                storm_id=storm_id,
                model="OBrienMcP",
                rmse_nt=reference_rmse,
                mae_nt=0.8 * reference_rmse,
            ))
            push!(cross_rows, (
                experiment=experiment,
                storm_id=storm_id,
                model="Burton",
                rmse_nt=reference_rmse + 1.5,
                mae_nt=0.8 * (reference_rmse + 1.5),
            ))
            push!(cross_rows, (
                experiment=experiment,
                storm_id=storm_id,
                model="BurtonFull",
                rmse_nt=reference_rmse + 1.0,
                mae_nt=0.8 * (reference_rmse + 1.0),
            ))
        end
    end
    return holdout, DataFrame(cross_rows)
end

@testset "Significance metric conversion propagates interrupts" begin
    @test_throws InterruptException _validate_metric_value(
        _InterruptingMetric(), "interrupt fixture", :rmse_nt,
    )
end

function _manifested_significance_fixture()
    specifications = (
        (DateTime(1977, 1, 1), 21, "train"),
        (DateTime(1980, 1, 1), 21, "train"),
        (DateTime(1997, 1, 1), 23, "train"),
        (DateTime(2000, 1, 1), 23, "train"),
        (DateTime(2005, 1, 1), 23, "train"),
        (DateTime(2009, 1, 1), 24, "val"),
        (DateTime(2012, 1, 1), 24, "val"),
        (DateTime(2016, 1, 1), 24, "val"),
        (DateTime(2020, 1, 1), 25, "test"),
        (DateTime(2022, 1, 1), 25, "test"),
        (DateTime(2024, 1, 1), 25, "test"),
    )
    catalog = [begin
        onset, cycle, split = specification
        first_row = 1 + (storm_id - 1) * 37
        StormCatalogEntry(
            storm_id, onset, -80.0, onset + Hour(2), onset + Hour(36),
            36.0, cycle, split, first_row, first_row + 36,
        )
    end for (storm_id, specification) in enumerate(specifications)]

    function metric_rows(ids; experiment=nothing)
        rows = NamedTuple[]
        for (position, storm_id) in enumerate(ids)
            reference = 4.0 + position
            difference = (-0.8, 0.4, -0.2)[mod1(position, 3)]
            anchor = 1 + (storm_id - 1) * 37
            scored = join((anchor + 1):(anchor + 36), ";")
            for (model, value) in (
                ("SINDy", reference + difference),
                ("OBrienMcP", reference),
                ("Burton", reference + 1.5),
                ("BurtonFull", reference + 1.0),
            )
                row = (storm_id=storm_id, model=model, rmse_nt=value,
                       mae_nt=0.8 * value,
                       anchor_catalog_index=anchor,
                       driver_start_catalog_index=anchor,
                       driver_end_catalog_index=anchor + 36,
                       scored_catalog_indices=scored,
                       cohort_signature_sha256="fixture-$storm_id")
                push!(rows, experiment === nothing ? row :
                      merge((experiment=experiment,), row))
            end
        end
        return rows
    end
    holdout = DataFrame(metric_rows((6, 7, 8)))
    cross = DataFrame(vcat(
        metric_rows((3, 4, 5); experiment="C20-22->C23"),
        metric_rows((1, 2, 3, 4, 5); experiment="even->odd"),
        metric_rows((9, 10, 11); experiment="C20-23->C25"),
    ))
    eligibility = DataFrame([
        (storm_id=entry.storm_id, solar_cycle=entry.solar_cycle,
         catalog_split=entry.split, eligible=true) for entry in catalog
    ])
    return (; catalog, holdout, cross, eligibility)
end

@testset "Predeclared headline significance workflow" begin
    holdout, cross = _significance_fixture()
    expected_labels = [spec.label for spec in HEADLINE_SIGNIFICANCE_SPECS]

    mktempdir() do first_root
        mktempdir() do second_root
            first_result = run_headline_significance(holdout, cross, first_root)
            second_result = run_headline_significance(holdout, cross, second_root)

            @test [item.spec.label for item in first_result.comparisons] == expected_labels
            @test [row.label for row in first_result.holm] == expected_labels
            @test all(row.family_size == 4 for row in first_result.holm)
            @test basename(first_result.holm_path) ==
                  "$(HEADLINE_HOLM_PREFIX)_adjusted.csv"
            @test basename(first_result.claim_source_path) == HEADLINE_CLAIM_SOURCE_FILE

            first_files = sort(readdir(first_root))
            @test length(first_files) == 44
            @test first_files == sort(readdir(second_root))
            for file in first_files
                @test read(joinpath(first_root, file)) == read(joinpath(second_root, file))
            end

            claims = CSV.read(first_result.claim_source_path, DataFrame)
            @test claims.experiment == expected_labels
            @test claims.comparison == fill("SINDy_vs_OBrienMcP", 4)
            @test claims.source_file == [
                "real_holdout_metrics.csv",
                "cross_cycle_metrics.csv",
                "cross_cycle_metrics.csv",
                "cross_cycle_metrics.csv",
            ]
            @test claims.bootstrap_draws == fill(10_000, 4)
            @test claims.seed == fill(42, 4)
            @test claims.holm_family_size == fill(4, 4)
            @test all(isfinite, claims.wilcoxon_p_value)
            @test claims.holm_p_value == [row.holm_p_value for row in first_result.holm]
            all_baseline_claims = CSV.read(
                first_result.all_baseline_claim_source_path, DataFrame,
            )
            @test nrow(all_baseline_claims) == 12
            @test Set(all_baseline_claims.reference_model) ==
                  Set(("Burton", "BurtonFull", "OBrienMcP"))
            @test count(all_baseline_claims.in_predeclared_holm_family) == 4
            aggregate = CSV.read(first_result.aggregate_path, DataFrame)
            @test nrow(aggregate) == 16
            @test all(aggregate.n_storms .== 3)
            @test Set(propertynames(aggregate)) >=
                  Set((:mean_rmse_nt, :mean_mae_nt,
                       :standard_error_rmse_nt, :standard_error_mae_nt))
            @test all(isfinite, aggregate.mean_rmse_nt)
            @test all(isfinite, aggregate.mean_mae_nt)
            @test all(isfinite, aggregate.standard_deviation_rmse_nt)
            @test all(isfinite, aggregate.standard_deviation_mae_nt)
            @test all(aggregate.standard_deviation_definition .==
                      "sample_standard_deviation_n_minus_1")

            for spec in HEADLINE_SIGNIFICANCE_SPECS
                pairs = CSV.read(joinpath(first_root, "$(spec.prefix)_pairs.csv"), DataFrame)
                bootstrap = CSV.read(
                    joinpath(first_root, "$(spec.prefix)_bootstrap.csv"), DataFrame
                )
                @test nrow(pairs) == 3
                @test nrow(bootstrap) == 10_000
                @test bootstrap.draw == collect(1:10_000)
            end
        end
    end

    function throws_without_artifacts(held_out, cross_cycle)
        mktempdir() do root
            @test_throws ArgumentError run_headline_significance(held_out, cross_cycle, root)
            @test isempty(readdir(root))
        end
    end

    missing_experiment = cross[cross.experiment .!= "even->odd", :]
    throws_without_artifacts(holdout, missing_experiment)

    extra_experiment = vcat(cross, DataFrame([
        (experiment="unexpected", storm_id="extra", model="SINDy",
         rmse_nt=2.0, mae_nt=1.5),
    ]))
    throws_without_artifacts(holdout, extra_experiment)

    missing_pair = cross[.!((cross.experiment .== "C20-22->C23") .&
                            (cross.model .== "OBrienMcP") .&
                            (cross.storm_id .== "x1_1")), :]
    throws_without_artifacts(holdout, missing_pair)

    duplicate_pair = vcat(cross, cross[1:1, :])
    throws_without_artifacts(holdout, duplicate_pair)

    invalid_mae = copy(holdout)
    invalid_mae.mae_nt[1] = NaN
    throws_without_artifacts(invalid_mae, cross)

    extra_model = vcat(holdout, DataFrame([(
        storm_id="h1", model="Mystery", rmse_nt=1.0, mae_nt=1.0,
    )]))
    throws_without_artifacts(extra_model, cross)

    singleton_holdout = holdout[holdout.storm_id .== "h1", :]
    throws_without_artifacts(singleton_holdout, cross)

    mktempdir() do root
        collision = joinpath(root, "headline_validation_c24_bootstrap.csv")
        mkpath(collision)
        @test_throws ArgumentError run_headline_significance(holdout, cross, root)
        @test isdir(collision)
        @test readdir(root) == [basename(collision)]
        @test isempty(readdir(collision))
    end

    mktempdir() do root
        sentinel = joinpath(root, "headline_validation_c24_pairs.csv")
        write(sentinel, "sentinel\n")
        late_collision = joinpath(root, PRIMARY_MODEL_SUMMARY_FILE)
        mkdir(late_collision)
        keep = joinpath(late_collision, "keep")
        write(keep, "preserve")
        @test_throws ArgumentError run_headline_significance(holdout, cross, root)
        @test read(sentinel, String) == "sentinel\n"
        @test isdir(late_collision)
        @test read(keep, String) == "preserve"
        @test sort(readdir(root)) == sort([basename(sentinel), basename(late_collision)])
    end

    undefined_wilcoxon = copy(holdout)
    for storm_id in unique(undefined_wilcoxon.storm_id)
        reference = only(undefined_wilcoxon[
            (undefined_wilcoxon.storm_id .== storm_id) .&
            (undefined_wilcoxon.model .== "OBrienMcP"), :rmse_nt
        ])
        undefined_wilcoxon[
            (undefined_wilcoxon.storm_id .== storm_id) .&
            (undefined_wilcoxon.model .== "SINDy"), :rmse_nt
        ] .= reference
    end
    throws_without_artifacts(undefined_wilcoxon, cross)

    @test_throws ArgumentError run_headline_significance(holdout, cross, "")
    @test_throws ArgumentError run_headline_significance(
        select(holdout, Not(:rmse_nt)), cross, mktempdir()
    )
    withenv("SOLARSINDY_OUTPUT_ROOT" => nothing) do
        @test_throws ArgumentError _significance_main()
    end

    mktempdir() do root
        data_root = joinpath(root, "data")
        source_root = joinpath(data_root, "source")
        mkpath(source_root)
        fixture = _manifested_significance_fixture()
        omni_path = joinpath(source_root, "omni_extracted.csv")
        write(omni_path, "test frozen OMNI input\n")
        catalog_path = joinpath(data_root, "storm_catalog.csv")
        write_verified_storm_catalog(fixture.catalog, catalog_path;
            omni_path,
            producer_script=joinpath(@__DIR__, "..", "validation",
                                     "download_omni.jl"),
            parameters=storm_catalog_parameters(),
            mode=:test,
        )
        producer = joinpath(@__DIR__, "..", "validation", "real_data_discovery.jl")
        eligibility_path = joinpath(data_root, "real_storm_eligibility.csv")
        CSV.write(eligibility_path, fixture.eligibility)
        write_output_manifest(eligibility_path;
            producer_script=producer,
            input_paths=(omni_extracted=omni_path, storm_catalog=catalog_path),
            selection_record=(kind="predeclared_observation_policy",),
            deterministic=true, mode=:test)

        function manifested_split(filename, ids)
            path = joinpath(data_root, filename)
            CSV.write(path, DataFrame(
                storm_id=string.(collect(ids)),
                inner_split=[index == length(ids) ? "validation" : "train"
                             for index in eachindex(ids)],
            ))
            write_output_manifest(path;
                producer_script=producer,
                input_paths=(omni_extracted=omni_path,
                             storm_catalog=catalog_path,
                             storm_observation_audit=eligibility_path),
                selection_record=(kind="fixed_grid_whole_storm_forward_selection",
                                  artifact="split"),
                deterministic=true, mode=:test)
            return path
        end
        primary_split = manifested_split("primary_lambda_inner_split.csv", 1:5)
        cross_splits = (
            manifested_split("cross_c20_22_to_c23_lambda_inner_split.csv", 1:2),
            manifested_split("cross_even_to_odd_lambda_inner_split.csv", 6:8),
            manifested_split("cross_c20_23_to_c25_lambda_inner_split.csv", 1:5),
        )
        holdout_path = joinpath(data_root, "real_holdout_metrics.csv")
        cross_path = joinpath(data_root, "cross_cycle_metrics.csv")
        CSV.write(holdout_path, fixture.holdout)
        CSV.write(cross_path, fixture.cross)
        selection_split_paths = (
            holdout=Dict("Validation C24" => primary_split),
            cross=Dict(
                "C20-22->C23" => cross_splits[1],
                "even->odd" => cross_splits[2],
                "C20-23->C25" => cross_splits[3],
            ),
        )
        @test isnothing(_assert_manifested_metric_cohorts(
            fixture.holdout, fixture.cross, catalog_path, eligibility_path,
            selection_split_paths,
        ))
        omitted_outer = fixture.cross[.!(
            (fixture.cross.experiment .== "even->odd") .&
            (fixture.cross.storm_id .== 5)
        ), :]
        @test_throws ErrorException _assert_manifested_metric_cohorts(
            fixture.holdout, omitted_outer, catalog_path, eligibility_path,
            selection_split_paths,
        )
        overlapping_split = joinpath(data_root, "overlapping_inner_split.csv")
        CSV.write(overlapping_split,
                  DataFrame(storm_id=[6], inner_split=["validation"]))
        @test_throws ErrorException _assert_manifested_metric_cohorts(
            fixture.holdout, fixture.cross, catalog_path, eligibility_path,
            (holdout=Dict("Validation C24" => overlapping_split),
             cross=selection_split_paths.cross),
        )
        rm(overlapping_split)
        decision = (selection_rule=_SIGNIFICANCE_SELECTION_RULE,)
        holdout_selection = (
            kind="fixed_grid_whole_storm_forward_selection",
            basis="full", scope="primary", decision,
            outer_split="cycle_24",
        )
        cross_selection = (
            kind="independently_selected_cross_cycle_outer_metrics",
            basis="full",
            experiments=[(
                kind="fixed_grid_whole_storm_forward_selection",
                basis="full", scope=scope, decision,
            ) for scope in ("C20-22->C23", "even->odd", "C20-23->C25")],
        )
        function manifest_metric(path, selection_record, selection_inputs)
            write_output_manifest(path;
                producer_script=producer,
                input_paths=merge(Dict(
                    "omni_extracted" => omni_path,
                    "storm_catalog" => catalog_path,
                    "storm_observation_audit" => eligibility_path,
                ), selection_inputs),
                selection_record,
                deterministic=true,
                mode=:test,
            )
        end
        cross_inputs = Dict(
            "cross_1_full_split" => cross_splits[1],
            "cross_2_full_split" => cross_splits[2],
            "cross_3_full_split" => cross_splits[3],
        )
        holdout_inputs = Dict("full_primary_split" => primary_split)
        manifest_metric(cross_path, cross_selection, cross_inputs)
        manifest_metric(holdout_path, (kind="arbitrary",), holdout_inputs)
        @test_throws ErrorException run_manifested_headline_significance(
            holdout_path, cross_path, data_root;
            mode=:test, omni_path, catalog_path,
        )
        @test isempty(filter(name -> startswith(name, "headline_"),
                             readdir(data_root)))
        manifest_metric(holdout_path, holdout_selection, holdout_inputs)
        catalog_bytes = read(catalog_path)
        before_mutation_run = sort(readdir(data_root))
        hook_calls = Ref(0)
        @test_throws ErrorException run_manifested_headline_significance(
            holdout_path, cross_path, data_root;
            mode=:test, omni_path, catalog_path,
            _after_compute_hook=() -> begin
                hook_calls[] += 1
                open(catalog_path, "a") do io
                    write(io, '\n')
                end
            end,
        )
        @test hook_calls[] == 1
        @test sort(readdir(data_root)) == before_mutation_run
        @test_throws ErrorException verify_storm_catalog(
            catalog_path; omni_path, parameters=storm_catalog_parameters(),
            mode=:test, verify_source=true,
        )
        write(catalog_path, catalog_bytes)
        @test verify_storm_catalog(
            catalog_path; omni_path, parameters=storm_catalog_parameters(),
            mode=:test, verify_source=true,
        ) !== nothing
        primary_split_bytes = read(primary_split)
        before_split_mutation_run = sort(readdir(data_root))
        @test_throws ErrorException run_manifested_headline_significance(
            holdout_path, cross_path, data_root;
            mode=:test, omni_path, catalog_path,
            _after_compute_hook=() -> open(primary_split, "a") do io
                write(io, '\n')
            end,
        )
        @test sort(readdir(data_root)) == before_split_mutation_run
        write(primary_split, primary_split_bytes)
        @test verify_output_manifest(primary_split;
            require_canonical=false, verify_source=true) !== nothing
        withenv("SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_RUN_MODE" => "test") do
            canonical_shape_result = _significance_main()
            @test length(canonical_shape_result.artifacts) == 4
            output_names = canonical_shape_result.all_output_paths
            @test length(output_names) == 44
            for path in output_names
                @test isfile(path * ".manifest.json")
                record = verify_output_manifest(path;
                    require_canonical=false,
                    verify_source=true,
                )
                @test record !== nothing
            end
            headline_record = verify_output_manifest(
                canonical_shape_result.artifacts[1].paths.pairs;
                require_canonical=false, verify_source=true,
            )
            headline_selection = _json_get(headline_record, "selection_record")
            @test String(_json_get(headline_selection, "secondary_test")) ==
                  "two_sided_paired_Wilcoxon_normal_approximation"
            @test length(_json_get(headline_selection, "multiplicity_family")) == 4
            interval_record = verify_output_manifest(
                canonical_shape_result.baseline_artifacts[1].paths.pairs;
                require_canonical=false, verify_source=true,
            )
            interval_selection = _json_get(interval_record, "selection_record")
            @test String(_json_get(interval_selection, "secondary_test")) == "none"
            @test isempty(_json_get(interval_selection, "multiplicity_family"))
            @test String(_json_get(interval_selection, "multiplicity_method")) == "none"
            aggregate_record = verify_output_manifest(
                canonical_shape_result.aggregate_path;
                require_canonical=false, verify_source=true,
            )
            aggregate_selection = _json_get(aggregate_record, "selection_record")
            @test String(_json_get(aggregate_selection, "standard_deviation")) ==
                  "sample_standard_deviation_n_minus_1"

            transaction_paths = vcat(
                output_names,
                [path * ".manifest.json" for path in output_names],
            )
            @test length(transaction_paths) == 88
            before_late_failure = Dict(
                path => read(path) for path in transaction_paths
            )
            late_manifest_hook_calls = Ref(0)
            @test_throws ErrorException run_manifested_headline_significance(
                holdout_path, cross_path, data_root;
                mode=:test, omni_path, catalog_path,
                _after_manifest_hook=() -> begin
                    late_manifest_hook_calls[] += 1
                    error("injected post-manifest significance failure")
                end,
            )
            @test late_manifest_hook_calls[] == 1
            @test all(read(path) == before_late_failure[path]
                      for path in transaction_paths)
            @test all(verify_output_manifest(path;
                require_canonical=false, verify_source=true) !== nothing
                for path in output_names)
        end
    end
end
