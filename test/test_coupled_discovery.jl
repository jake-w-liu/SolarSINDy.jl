using Dates

include(joinpath(@__DIR__, "..", "validation", "coupled_discovery.jl"))

function _independent_coupled_stlsq(theta, dst_target, ae_target, lambda)
    return (
        dst=stlsq(theta, dst_target; λ=lambda, normalize=true),
        ae=stlsq(theta, ae_target; λ=lambda, normalize=true),
    )
end

function _synthetic_coupled_storm(id::Int; split="train")
    rng = MersenneTwister(10_000 + id)
    base = build_solar_wind_library()
    coupled = build_coupled_library(base)
    n_fit = 40
    theta = randn(rng, n_fit, length(coupled))
    dst_target = theta[:, 2] .* 0.01 .- theta[:, end] .* 0.005
    ae_target = theta[:, 3] .* 0.02 .+ theta[:, end - 1] .* 0.004
    n_score = 8
    observed_dst = [0.0, -2.0, -5.0, -8.0, -10.0, -9.0, -7.0, -5.0]
    observed_ae = [100.0, 200.0, 400.0, 700.0, 900.0, 800.0, 600.0, 400.0]
    swd = SolarWindData(
        collect(0.0:(n_score - 1)),
        fill(420.0 + id, n_score),
        fill(-5.0, n_score),
        fill(2.0, n_score),
        fill(5.0, n_score),
        fill(1.5, n_score),
        observed_dst,
        observed_dst,
    )
    start = 1000 + 20id
    return (
        storm_id = id,
        onset_time = DateTime(2000, 1, 1) + Day(id),
        entry = (split=split,),
        swd = swd,
        data = Dict{String,Vector{Float64}}(),
        theta = theta[:, 1:length(base)],
        target = dst_target,
        theta_single = theta[:, 1:length(base)],
        theta_coupled = theta,
        target_dst = dst_target,
        target_ae = ae_target,
        scoring_observations = observed_dst,
        scoring_dst = observed_dst,
        scoring_ae = observed_ae,
        score_start_idx = start,
        score_end_idx = start + n_score - 1,
        common_anchor_idx = start,
        library_terms = Tuple(get_term_names(base)),
        coupled_library_terms = Tuple(get_term_names(coupled)),
    )
end

@testset "Coupled discovery protocol" begin
    base = build_solar_wind_library()
    coupled = build_coupled_library(base)
    @test length(base) == 20
    @test length(coupled) == 25
    @test get_term_names(coupled)[1:20] == get_term_names(base)
    @test Tuple(get_term_names(coupled)[21:end]) == COUPLED_AE_TERMS
    @test !("n*V^2" in get_term_names(coupled))

    swd = SolarWindData(
        [0.0, 1.0], [400.0, 400.0], [-5.0, -5.0], [2.0, 2.0],
        [5.0, 5.0], [1.5, 1.5], [-10.0, -10.0], [-10.0, -10.0],
    )
    ξ_dst = zeros(length(coupled))
    ξ_ae = zeros(length(coupled))
    ξ_dst[21] = 0.01
    ξ_ae[24] = 0.001
    dst, ae = simulate_coupled(ξ_dst, ξ_ae, base, swd, 100.0, 1.0; dst0=-10.0)
    @test dst == [-10.0, -9.0]
    @test ae == [100.0, 140.0]
    point = Dict(
        "V" => [400.0], "Bs" => [5.0], "Bz" => [-5.0], "By" => [2.0],
        "n" => [5.0], "Pdyn" => [1.5], "Dst_star" => [-10.0],
        "theta_c" => [atan(2.0, -5.0)], "BT" => [hypot(2.0, -5.0)],
        "AE" => [100.0],
    )
    θ = evaluate_library(coupled, point)
    @test dst[2] - dst[1] == dot(θ[1, :], ξ_dst)
    @test ae[2] - ae[1] == dot(θ[1, :], ξ_ae)
    long_swd = SolarWindData(
        collect(0.0:999.0), fill(400.0, 1000), fill(-5.0, 1000),
        fill(2.0, 1000), fill(5.0, 1000), fill(1.5, 1000),
        fill(-10.0, 1000), fill(-10.0, 1000),
    )
    simulate_coupled(ξ_dst, ξ_ae, base, long_swd, 100.0, 1.0; dst0=-10.0)
    @test (@allocated simulate_coupled(
        ξ_dst, ξ_ae, base, long_swd, 100.0, 1.0; dst0=-10.0,
    )) < 100_000
    @test_throws DimensionMismatch simulate_coupled(
        ξ_dst[1:end-1], ξ_ae, base, swd, 100.0, 1.0,
    )
    @test_throws ArgumentError simulate_coupled(
        ξ_dst, ξ_ae, base, swd, -1.0, 1.0,
    )

    rng = MersenneTwister(77)
    ensemble_theta = randn(rng, 60, length(coupled))
    ensemble_dst = randn(rng, 60)
    ensemble_design = (
        theta=ensemble_theta,
        dst=ensemble_dst,
        ae=2 .* ensemble_dst,
    )
    for lambda in (0.0, 0.05, 0.5)
        shared_fit = _paired_coupled_stlsq(
            ensemble_theta, ensemble_dst, 2 .* ensemble_dst; lambda,
        )
        independent_fit = _independent_coupled_stlsq(
            ensemble_theta, ensemble_dst, 2 .* ensemble_dst, lambda,
        )
        @test shared_fit.dst ≈ independent_fit.dst rtol=1e-12 atol=1e-12
        @test shared_fit.ae ≈ independent_fit.ae rtol=1e-12 atol=1e-12
    end
    overflow_theta = fill(floatmax(Float64), 2, 1)
    overflow_dst = ones(2)
    overflow_ae = fill(2.0, 2)
    overflow_fit = _paired_coupled_stlsq(
        overflow_theta, overflow_dst, overflow_ae; lambda=0.0,
    )
    overflow_oracle = _independent_coupled_stlsq(
        overflow_theta, overflow_dst, overflow_ae, 0.0,
    )
    @test overflow_fit.dst == overflow_oracle.dst
    @test overflow_fit.ae == overflow_oracle.ae
    @test all(!iszero, overflow_fit.dst)
    @test all(!iszero, overflow_fit.ae)
    _paired_coupled_stlsq(
        ensemble_theta, ensemble_dst, 2 .* ensemble_dst; lambda=0.05,
    )
    _independent_coupled_stlsq(
        ensemble_theta, ensemble_dst, 2 .* ensemble_dst, 0.05,
    )
    GC.gc()
    shared_allocations = @allocated _paired_coupled_stlsq(
        ensemble_theta, ensemble_dst, 2 .* ensemble_dst; lambda=0.05,
    )
    GC.gc()
    independent_allocations = @allocated _independent_coupled_stlsq(
        ensemble_theta, ensemble_dst, 2 .* ensemble_dst, 0.05,
    )
    @test shared_allocations < independent_allocations
    paired = _paired_coupled_ensemble(
        ensemble_design, 0.0; n_models=3, subsample_fraction=0.8, seed=7,
    )
    repeated = _paired_coupled_ensemble(
        ensemble_design, 0.0; n_models=3, subsample_fraction=0.8, seed=7,
    )
    @test paired == repeated
    @test size(paired.dst) == (length(coupled), 3)
    @test paired.ae ≈ 2 .* paired.dst
    empty_quantiles = _conditional_quantiles(zeros(5))
    @test all(isnan, empty_quantiles)
    draw_rows = _coupled_draw_records(paired, get_term_names(coupled), 0.0)
    @test length(draw_rows) == 2 * length(coupled) * 3
    @test all(row.structural_zeros_retained for row in draw_rows)

    diagnostic_model = (
        dst=zeros(length(coupled)), ae=zeros(length(coupled)),
    )
    diagnostic_rows = _coupled_design_diagnostic_records(
        ensemble_design, diagnostic_model, get_term_names(coupled), 0.0,
    )
    @test length(diagnostic_rows) == 8
    @test Set(row.equation for row in diagnostic_rows) ==
          Set(("dDst_star/dt", "dAE/dt"))
    @test all(count(row -> row.equation == equation, diagnostic_rows) == 4
              for equation in ("dDst_star/dt", "dAE/dt"))
    @test Set(row.block for row in diagnostic_rows) == Set((
        "full_design", "selected_active_block", "cross_index_candidate_block",
        "selected_cross_index_block",
    ))
    @test all(row.contribution_units == "nT_per_hour" &&
              row.normalization == "training_column_l2_norm" &&
              row.n_rows == size(ensemble_theta, 1) &&
              !ismissing(row.normalized_design_condition_number)
              for row in diagnostic_rows)
    cancelling = _coupled_design_diagnostic_record(
        [1.0 1.0; 2.0 2.0; 3.0 3.0], [1.0, -1.0], ["AE", "AE^2"], [1, 2];
        equation="dDst_star/dt", block="cross_index_candidate_block",
        selected_lambda=0.1,
    )
    @test cancelling.normalized_design_rank == 1
    @test isinf(cancelling.normalized_design_condition_number)
    @test cancelling.net_contribution_absmax_nt_per_hour == 0.0
    @test cancelling.gross_contribution_absmax_nt_per_hour == 6.0
    @test cancelling.cancellation_ratio > 1e15
    @test cancelling.interpretation_status == "rank_deficient_grouped_net_only"

    mktempdir() do directory
        omni = joinpath(directory, "omni_fixture.txt")
        catalog = joinpath(directory, "catalog_fixture.csv")
        write(omni, "fixture")
        write(catalog, "fixture")
        context = (omni=omni, catalog=catalog, mode=:test, outputs=String[])
        output = joinpath(directory, "coupled_design_diagnostics.csv")
        _write_manifested_csv(
            output, diagnostic_rows, context;
            selection_record=(kind="focused_coupled_diagnostic_test",),
        )
        @test output in context.outputs
        @test isfile(output * ".manifest.json")
        @test verify_output_manifest(output; require_canonical=false) !== nothing
        persisted = CSV.read(output, DataFrame)
        @test all(column in names(persisted) for column in (
            "net_contribution_min_nt_per_hour",
            "net_contribution_max_nt_per_hour",
            "net_contribution_absmax_nt_per_hour",
            "gross_contribution_absmax_nt_per_hour",
        ))
        @test all(!ismissing, persisted.normalized_design_condition_number)
        @test occursin("NaN", read(output, String))

        selection_input = joinpath(directory, "selection_dependency.csv")
        write(selection_input, "storm_id,inner_split\n1,train\n")
        bound_context = merge(context, (
            observation_inputs=Dict("coupled_observation_audit" => output),
            selection_inputs=Dict("coupled_selection_split" => selection_input),
        ))
        bound_output = joinpath(directory, "bound_downstream.csv")
        _write_manifested_csv(
            bound_output, [(value=1.0,)], bound_context;
            selection_record=(kind="bound_dependency_test",),
        )
        bound_record = verify_output_manifest(bound_output;
            require_canonical=false)
        @test Set(String(input["name"]) for input in bound_record["inputs"]) ==
              Set(("omni_extracted", "storm_catalog",
                   "coupled_observation_audit", "coupled_selection_split"))

        output_bytes = read(output)
        manifest_bytes = read(output * ".manifest.json")
        @test_throws ArgumentError _write_manifested_csv(
            output, [(replacement=1.0,)], context; selection_record=nothing,
        )
        @test read(output) == output_bytes
        @test read(output * ".manifest.json") == manifest_bytes
        @test verify_output_manifest(output; require_canonical=false) !== nothing

        failed = joinpath(directory, "failed.csv")
        @test_throws ArgumentError _write_manifested_csv(
            failed, diagnostic_rows, context; selection_record=nothing,
        )
        @test !isfile(failed)
        @test !isfile(failed * ".manifest.json")

        first_selection = joinpath(directory, "selection_first.csv")
        second_selection = joinpath(directory, "selection_second.csv")
        write(first_selection, "x\n1\n")
        write(second_selection, "x\n2\n")
        selection_paths = (first=first_selection, second=second_selection)
        _manifest_selection_outputs!(
            selection_paths, context, (kind="initial_transaction_test",),
        )
        transaction_paths = vcat(
            collect(values(selection_paths)),
            [path * ".manifest.json" for path in values(selection_paths)],
        )
        before = Dict(path => read(path) for path in transaction_paths)
        before_outputs = copy(context.outputs)
        manifested_fields = Symbol[]
        @test_throws ErrorException _manifest_selection_outputs!(
            selection_paths, context, (kind="replacement_transaction_test",);
            _after_manifest_hook=(field, _) -> begin
                push!(manifested_fields, field)
                error("injected coupled manifest failure")
            end,
        )
        @test manifested_fields == [:first]
        @test all(read(path) == before[path] for path in transaction_paths)
        @test all(verify_output_manifest(path; require_canonical=false) !== nothing
                  for path in values(selection_paths))
        @test context.outputs == before_outputs
    end

    # Long AE outages stay unobserved after cleaning and fail the shared
    # require_ae regression-support gate; they are never nearest-filled.
    n = 42
    times = collect(DateTime(2001):Hour(1):DateTime(2001) + Hour(n - 1))
    frame = DataFrame(
        datetime = times,
        V = fill(450.0, n), Bz = fill(-4.0, n), By = fill(2.0, n),
        n = fill(5.0, n), Pdyn = fill(1.7, n), T = fill(1.0e5, n),
        Dst = collect(range(0.0, -80.0; length=n)), AE = fill(200.0, n),
        AL = fill(-150.0, n), AU = fill(50.0, n),
    )
    frame.AE[5:38] .= NaN
    add_original_observation_flags!(frame)
    clean_omni_data!(frame)
    entry = StormCatalogEntry(1, times[1], -80.0, times[30], times[end],
                              n - 1.0, 23, "train", 1, n)
    policy = DiscoveryObservationPolicy(require_ae=true)
    audit = _audit_coupled_observations(frame, [entry], policy)
    @test isempty(audit.eligible_entries)
    @test audit.storm_records[1].require_ae
    @test !audit.storm_records[1].eligible
    @test !audit.storm_records[1].coupled_eligible
    @test audit.storm_records[1].exclusion_reason ==
          "insufficient_original_support_regression_rows"
    @test all(isnan, frame.AE[5:38])

    # A localized AE outage may be excluded from the audited regression stencil
    # without invalidating an otherwise well-observed coupled storm.
    n_gap = 80
    gap_times = collect(DateTime(2001, 6):Hour(1):
                        DateTime(2001, 6) + Hour(n_gap - 1))
    gap_frame = DataFrame(
        datetime = gap_times,
        V = fill(450.0, n_gap), Bz = fill(-4.0, n_gap),
        By = fill(2.0, n_gap), n = fill(5.0, n_gap),
        Pdyn = fill(1.7, n_gap), T = fill(1.0e5, n_gap),
        Dst = collect(range(0.0, -90.0; length=n_gap)),
        AE = fill(200.0, n_gap), AL = fill(-150.0, n_gap),
        AU = fill(50.0, n_gap),
    )
    gap_frame.AE[35:39] .= NaN
    add_original_observation_flags!(gap_frame)
    clean_omni_data!(gap_frame)
    gap_entry = StormCatalogEntry(
        3, gap_times[1], -90.0, gap_times[55], gap_times[end],
        n_gap - 1.0, 23, "train", 1, n_gap,
    )
    gap_audit = _audit_coupled_observations(gap_frame, [gap_entry], policy)
    @test only(gap_audit.eligible_entries) == gap_entry
    prepared_gap = _prepare_coupled_storm(
        gap_frame, gap_entry, base, coupled, policy,
    )
    @test all(isfinite, prepared_gap.target_ae)
    @test all(isfinite, prepared_gap.theta_coupled)
    @test all(isnan, gap_frame.AE[35:39])

    # Enough early AE rows for regression do not make an almost-unobserved
    # outer AE trajectory eligible for coupled scoring.
    n_sparse = 60
    sparse_times = collect(DateTime(2002):Hour(1):DateTime(2002) + Hour(n_sparse - 1))
    sparse = DataFrame(
        datetime = sparse_times,
        V = fill(450.0, n_sparse), Bz = fill(-4.0, n_sparse),
        By = fill(2.0, n_sparse), n = fill(5.0, n_sparse),
        Pdyn = fill(1.7, n_sparse), T = fill(1.0e5, n_sparse),
        Dst = collect(range(0.0, -80.0; length=n_sparse)),
        AE = fill(200.0, n_sparse), AL = fill(-150.0, n_sparse),
        AU = fill(50.0, n_sparse),
    )
    sparse.AE[31:end] .= NaN
    add_original_observation_flags!(sparse)
    clean_omni_data!(sparse)
    sparse_entry = StormCatalogEntry(
        2, sparse_times[1], -80.0, sparse_times[45], sparse_times[end],
        n_sparse - 1.0, 23, "train", 1, n_sparse,
    )
    sparse_audit = _audit_coupled_observations(sparse, [sparse_entry], policy)
    @test sparse_audit.storm_records[1].eligible
    @test !sparse_audit.storm_records[1].coupled_eligible
    @test sparse_audit.storm_records[1].post_anchor_ae_rows == 29
    @test sparse_audit.storm_records[1].post_anchor_ae_fraction <
          policy.min_scoring_fraction
    @test sparse_audit.storm_records[1].coupled_exclusion_reason ==
          "insufficient_post_anchor_original_ae_fraction"

    storms = [_synthetic_coupled_storm(id) for id in 1:10]
    outer = _synthetic_coupled_storm(11; split="val")
    coupled_selection = _select_coupled_lambda(
        storms, base, coupled, Dict{Any,Any}(),
    )
    @test coupled_selection.selected_lambda in storm_lambda_grid()
    @test length(coupled_selection.model.dst) == length(coupled)
    @test length(coupled_selection.model.ae) == length(coupled)
    @test Set(row.storm_id for row in coupled_selection.split_records) ==
          Set(string.(1:10))
    @test all(row.storm_id != string(outer.storm_id)
              for row in coupled_selection.error_records)
    full_refits = filter(row -> row.stage == "full_refit",
                         coupled_selection.equation_fit_records)
    @test length(full_refits) == 2
    @test Set(row.equation for row in full_refits) ==
          Set(("dDst_star/dt", "dAE/dt"))
    @test all(row.shared_lambda && row.lambda == coupled_selection.selected_lambda &&
              row.n_fit_storms == 10 for row in full_refits)

    single_selection = _select_discovery_lambda(storms, base, Dict{Any,Any}())
    split_audit = _assert_matched_inner_split(coupled_selection, single_selection)
    @test length(split_audit) == 10
    @test all(row.exact_match for row in split_audit)

    mktempdir() do directory
        omni = joinpath(directory, "omni_fixture.txt")
        catalog = joinpath(directory, "catalog_fixture.csv")
        write(omni, "fixture")
        write(catalog, "fixture")
        context = (omni=omni, catalog=catalog, mode=:test, outputs=String[])
        paths = _write_manifested_selection_outputs!(
            coupled_selection, directory, "transactional", context,
            (kind="focused_selection_transaction_test",),
        )
        output_paths = [getproperty(paths, field) for field in propertynames(paths)]
        transaction_paths = vcat(output_paths,
            [path * ".manifest.json" for path in output_paths])
        before = Dict(path => read(path) for path in transaction_paths)
        before_outputs = copy(context.outputs)
        replacement = merge(coupled_selection, (
            split_records=reverse(coupled_selection.split_records),
        ))
        @test_throws MethodError _write_manifested_selection_outputs!(
            replacement, directory, "transactional", context, nothing,
        )
        @test all(read(path) == before[path] for path in transaction_paths)
        @test all(verify_output_manifest(path; require_canonical=false) !== nothing
                  for path in output_paths)
        @test context.outputs == before_outputs
    end

    scored = _score_coupled_storm(
        outer, coupled_selection.model, single_selection.model, base,
    )
    @test length(scored.cohorts) == 5
    @test _assert_exact_outer_cohorts(scored.cohorts)
    @test length(unique(row.cohort_signature_sha256 for row in scored.cohorts)) == 1
    @test Set(row.model for row in scored.cohorts) == Set((
        "Coupled-SINDy", "Single-SINDy", "Burton", "BurtonFull",
        "OBrien-McPherron",
    ))
    @test scored.predictions.BurtonFull == simulate_burton_full(
        outer.swd.V, max.(-outer.swd.Bz, 0.0), 1.0;
        Dst0=first(outer.scoring_dst),
    )
    @test count(row -> row.target == "Dst_star", scored.metrics) == 5
    @test count(row -> row.target == "AE", scored.metrics) == 1
    @test all(row.original_target_only for row in scored.metrics)
    ae_metric = only(filter(row -> row.target == "AE", scored.metrics))
    @test ae_metric.scored_catalog_indices == join(
        (outer.score_start_idx + index - 1 for index in scored.ae_indices), ";",
    )
end
