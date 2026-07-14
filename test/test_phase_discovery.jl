@testset "Leakage-free phase discovery" begin
    include(joinpath(@__DIR__, "..", "validation", "phase_dependent_discovery.jl"))
    include(joinpath(@__DIR__, "..", "validation", "phase_sensitivity.jl"))

    lib = build_minimal_library()
    function toy_phase_storm(id::Int; split::String="train", gap::Bool=false)
        # Keep every phase above the production minimum of 100 pooled rows so
        # the coefficient-subsampling path exercises its real eligibility gate.
        n = 96
        t = collect(0.0:(n - 1))
        dst_star = vcat(
            collect(range(5.0, -65.0; length=32)),
            collect(range(-65.0, 5.0; length=32)),
            collect(range(5.0, -65.0; length=32)),
        )
        v = fill(420.0 + id, n)
        bz = fill(-4.0, n)
        by = fill(1.0, n)
        density = fill(5.0, n)
        pdyn = dynamic_pressure.(density, v)
        swd = SolarWindData(t, v, bz, by, density, pdyn,
                            copy(dst_star), copy(dst_star))
        data, target = prepare_sindy_data(swd, 1.0; smooth_window=5)
        theta = evaluate_library(lib, data)
        scoring = copy(dst_star)
        gap && (scoring[[8, 9, 31, 52]] .= NaN)
        entry = (split=split, solar_cycle=split == "train" ? 22 : 24)
        observation_record = (
            scoring_start_idx=1000 * id,
            scoring_end_idx=1000 * id + n - 1,
        )
        return (
            storm_id=id,
            onset_time=DateTime(2000, 1, 1) + Hour(100 * id),
            entry,
            swd,
            window_swd=swd,
            data,
            theta,
            target,
            regression_mask=trues(n),
            scoring_observations=scoring,
            observation_record,
            library_terms=Tuple(get_term_names(lib)),
        )
    end

    training = _with_phase_labels([toy_phase_storm(id; gap=id == 5)
                                   for id in 1:5])
    outer = _with_phase_labels([toy_phase_storm(99; split="val", gap=true)])
    @test _causal_phase_labels([0.0, -30.0, -33.0, -30.0], 1.0) == [1, 2, 2, 3]
    @test all(isnan, _phase_conditional_quantiles(Float64[]))
    phase_selection = _select_phase_lambda(
        training, lib; min_phase_rows=1, min_phase_storms=1,
    )
    @test length(phase_selection.candidate_records) == length(storm_lambda_grid())
    @test Set(row.storm_id for row in phase_selection.split_records) ==
          Set(string.(1:5))
    @test_throws ArgumentError _select_phase_lambda(
        vcat(training, outer), lib; min_phase_rows=1, min_phase_storms=1,
    )
    outer[1].scoring_observations[2:end] .= 1.0e9
    repeated = _select_phase_lambda(
        training, lib; min_phase_rows=1, min_phase_storms=1,
    )
    @test repeated.decision_record == phase_selection.decision_record
    @test repeated.model.coefficients == phase_selection.model.coefficients

    coefficient_rows = _phase_subsample_coefficients(
        training, phase_selection, lib; draws=2,
    )
    @test PHASE_ENSEMBLE_SEEDS == (quiet=42, main=43, recovery=44)
    @test Dict(row.phase => row.seed for row in coefficient_rows) ==
          Dict("quiet" => 42, "main" => 43, "recovery" => 44)

    diagnostics = _phase_design_diagnostic_records(
        training, phase_selection, lib; min_phase_rows=1, min_phase_storms=1,
    )
    @test length(diagnostics) == 3 * 4
    @test Set(row.block for row in diagnostics) == Set((
        "full_design", "selected_active_block", "clock_candidate_block",
        "selected_clock_block",
    ))
    @test all(row -> all(value -> value !== missing, values(row)), diagnostics)
    @test all(row -> row.row_scope == "phase_training_regression_rows", diagnostics)

    single_selection = _select_discovery_lambda(training, lib, Dict{Any,Any}())
    @test isnothing(_require_identical_inner_cohort(
        phase_selection, single_selection,
    ))
    bad_single = merge(single_selection, (
        split_records=reverse(single_selection.split_records),
    ))
    @test_throws ErrorException _require_identical_inner_cohort(
        phase_selection, bad_single,
    )
    mktempdir() do root
        omni = joinpath(root, "omni.csv")
        catalog = joinpath(root, "catalog.csv")
        write(omni, "fixture\n")
        write(catalog, "fixture\n")
        output_paths = (data=root, mode=:test)
        base_inputs = Dict("omni_extracted" => omni, "storm_catalog" => catalog)
        persisted = _phase_manifest_selection(
            phase_selection, "focused_phase";
            output_paths, producer_script=@__FILE__, inputs=base_inputs,
            kind="focused_phase_selection",
        )
        downstream_inputs = copy(base_inputs)
        for field in propertynames(persisted)
            downstream_inputs["phase_selection_$(field)"] = getproperty(persisted, field)
        end
        downstream = _phase_write(
            joinpath(root, "phase_downstream.csv"), [(value=1.0,)];
            output_paths, producer_script=@__FILE__, inputs=downstream_inputs,
            selection_record=(kind="focused_downstream",), deterministic=true,
        )
        record = verify_output_manifest(downstream; require_canonical=false)
        @test Set(String(input["name"]) for input in record["inputs"]) ==
              Set(keys(downstream_inputs))

        selection_bytes = Dict(
            path => (read(path), read(path * ".manifest.json"))
            for path in values(persisted)
        )
        replacement = merge(phase_selection, (
            split_records=reverse(phase_selection.split_records),
        ))
        @test replacement.split_records != phase_selection.split_records
        manifested_fields = Symbol[]
        @test_throws ErrorException _phase_manifest_selection(
            replacement, "focused_phase";
            output_paths, producer_script=@__FILE__, inputs=base_inputs,
            kind="failed_phase_selection",
            _after_manifest_hook=(field, _) -> begin
                push!(manifested_fields, field)
                field == :candidates && error("injected phase manifest failure")
            end,
        )
        @test manifested_fields == [:split, :candidates]
        for path in values(persisted)
            @test (read(path), read(path * ".manifest.json")) == selection_bytes[path]
            @test verify_output_manifest(path; require_canonical=false) isa AbstractDict
        end

        downstream_bytes = (read(downstream), read(downstream * ".manifest.json"))
        missing_producer = joinpath(root, "missing_producer.jl")
        @test_throws ArgumentError _phase_write(
            downstream, [(value=2.0,)];
            output_paths, producer_script=missing_producer, inputs=downstream_inputs,
            selection_record=(kind="failed_downstream",), deterministic=true,
        )
        @test (read(downstream), read(downstream * ".manifest.json")) == downstream_bytes
        @test verify_output_manifest(downstream; require_canonical=false) isa AbstractDict
    end
    identical_phase_model = ntuple(_ -> single_selection.model, 3)
    switching_oracle = _simulate_phase_switching(
        identical_phase_model, lib, training[1].swd, 1.0,
    )
    @test switching_oracle ≈ simulate_sindy(
        single_selection.model, lib, training[1].swd, 1.0,
    )
    _simulate_phase_switching(identical_phase_model, lib, training[1].swd, 1.0)
    @test (@allocated _simulate_phase_switching(
        identical_phase_model, lib, training[1].swd, 1.0,
    )) < 200_000

    scored_storm = training[end]
    score = _score_phase_storm(
        scored_storm, phase_selection.model, single_selection.model, lib,
    )
    expected = findall(isfinite, scored_storm.scoring_observations[2:end]) .+ 1
    @test score.scored_indices == expected
    @test length(score.rows) == 5
    @test Set(row.model for row in score.rows) == Set((
        "Switching-SINDy", "Single-SINDy", "Burton", "BurtonFull",
        "OBrien-McPherron",
    ))
    @test all(row -> row.n_points == length(expected), score.rows)
    @test length(unique(row.scored_absolute_rows for row in score.rows)) == 1
    @test length(unique(row.cohort_signature_sha256 for row in score.rows)) == 1
    @test score.predictions.BurtonFull == simulate_burton_full(
        score.swd.V, max.(-score.swd.Bz, 0.0), 1.0;
        Dst0=score.observations[1],
    )
    paired = _paired_phase_metrics(score.rows)
    @test length(paired) == 1
    @test paired[1].n_points == length(expected)
    @test paired[1].burton_published_rmse_nt == only(
        row.rmse_nt for row in score.rows if row.model == "BurtonFull"
    )

    excluded = _with_phase_labels([
        merge(toy_phase_storm(100; split="exclude"),
              (entry=(split="exclude", solar_cycle=19),)),
    ])
    cohort_rows = _phase_cohort_records(
        vcat(training[1:1], outer, excluded),
        Dict(string(training[1].storm_id) => "train"),
    )
    excluded_row = only(filter(row -> row.catalog_split == "exclude", cohort_rows))
    @test excluded_row.inner_split == "excluded"
    @test !excluded_row.used_by_switching && !excluded_row.used_by_single
    @test !excluded_row.used_by_burton &&
          !excluded_row.used_by_burton_published && !excluded_row.used_by_obrien
    outer_row = only(filter(row -> row.catalog_split == "val", cohort_rows))
    @test outer_row.inner_split == "outer"
    @test outer_row.used_by_switching && outer_row.used_by_single
    @test outer_row.used_by_burton && outer_row.used_by_burton_published &&
          outer_row.used_by_obrien

    sensitivity = _phase_threshold_selections(
        [toy_phase_storm(id) for id in 1:5], lib;
        quiet_thresholds=(-30.0, -15.0),
        deriv_thresholds=(-2.0,),
        min_phase_rows=1,
        min_phase_storms=1,
    )
    @test length(sensitivity) == 2
    @test all(result ->
        length(result.selection.candidate_records) == length(storm_lambda_grid()),
        sensitivity,
    )
    @test [(result.quiet_thresh, result.deriv_thresh) for result in sensitivity] ==
          [(-30.0, -2.0), (-15.0, -2.0)]
    threshold_rows = _threshold_result_rows(sensitivity, lib)
    @test length(threshold_rows.decisions) == 2
    @test all(row -> row.n_training_storms == 5, threshold_rows.decisions)
    threshold_outer = _phase_threshold_outer_rows(
        sensitivity, [toy_phase_storm(199; split="val", gap=true)],
        single_selection.model, lib,
    )
    @test length(threshold_outer.metrics) == 2 * 5
    @test length(threshold_outer.trajectories) == 2 * 96
    @test Set(row.model for row in threshold_outer.metrics) == Set((
        "Switching-SINDy", "Single-SINDy", "Burton", "BurtonFull",
        "OBrien-McPherron",
    ))
    @test all(group ->
        length(unique(group.cohort_signature_sha256)) == 1,
        groupby(DataFrame(threshold_outer.metrics),
                [:quiet_thresh, :deriv_thresh, :storm_id]),
    )
    @test !(:cohort_signature_sha256 in
            propertynames(DataFrame(threshold_outer.trajectories)))
    threshold_design = _phase_threshold_design_rows(sensitivity, lib)
    @test length(threshold_design) == 2 * 3 * 4
    @test Set((row.quiet_thresh, row.deriv_thresh) for row in threshold_design) ==
          Set(((-30.0, -2.0), (-15.0, -2.0)))
    @test !occursin("148.0", read(
        joinpath(@__DIR__, "..", "validation", "phase_sensitivity.jl"), String,
    ))
end
