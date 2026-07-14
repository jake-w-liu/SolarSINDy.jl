using Dates

include(joinpath(@__DIR__, "..", "validation", "real_discovery_helpers.jl"))
include(joinpath(@__DIR__, "..", "validation", "real_data_discovery.jl"))

@testset "Canonical real-discovery helpers" begin
    @testset "canonical ensemble fallback integrates with forecast initialization" begin
        library = build_solar_wind_library()
        terms = get_term_names(library)
        draws = zeros(length(terms), 4)
        draws[1, :] = [-1.0, -0.8, -1.2, -0.9]
        inclusion = vec(sum(draws .!= 0.0; dims=2)) ./ size(draws, 2)
        records = _real_empirical_subsample_records(
            terms, draws, inclusion;
            lambda=0.1, seed=42, subsample_fraction=0.8,
        )
        mktempdir() do directory
            coefficients = joinpath(directory, "coefficients.csv")
            ensemble = joinpath(directory, "real_ensemble_inclusion.csv")
            CSV.write(coefficients, DataFrame(term=[terms[1]], coefficient=[-0.5]))
            CSV.write(ensemble, DataFrame(records))
            state = init_forecast(
                coefficients_csv=coefficients,
                ensemble_csv=ensemble,
                draws_csv=joinpath(directory, "missing_joint_draws.csv"),
                t0=DateTime(2024, 1, 1), dst0=-20.0,
            )
            @test std(state.ξ_ensemble[:, 1]) > 0.0
            @test all(state.ξ_ensemble[:, 2:end] .== 0.0)
        end
    end

    n = 40
    total = 3n
    datetime = collect(DateTime(2000):Hour(1):DateTime(2000) + Hour(total - 1))
    velocity = fill(450.0, total)
    short_gap = (n + 8):(n + 9)
    long_gap = (2n + 18):(2n + 21)
    velocity[short_gap] .= NaN
    velocity[long_gap] .= NaN
    frame = DataFrame(
        datetime = datetime,
        V = velocity,
        Bz = fill(-4.0, total),
        By = fill(2.0, total),
        n = fill(5.0, total),
        Pdyn = dynamic_pressure.(fill(5.0, total), velocity),
        T = fill(1.0e5, total),
        Dst = collect(range(0.0, -90.0; length=total)),
        AE = fill(100.0, total),
        AL = fill(-80.0, total),
        AU = fill(20.0, total),
    )
    add_original_observation_flags!(frame)
    clean_omni_data!(frame)
    @test all(.!frame.V_observed[short_gap])
    @test all(isfinite, frame.V[short_gap])
    @test all(.!frame.V_observed[long_gap])
    @test all(isnan, frame.V[long_gap])

    entries = [
        StormCatalogEntry(1, datetime[1], -30.0, datetime[10], datetime[n],
                          n - 1.0, 23, "train", 1, n),
        StormCatalogEntry(2, datetime[n + 1], -60.0, datetime[n + 10], datetime[2n],
                          n - 1.0, 23, "train", n + 1, 2n),
        StormCatalogEntry(3, datetime[2n + 1], -90.0, datetime[2n + 10],
                          datetime[3n], n - 1.0, 24, "val", 2n + 1, 3n),
    ]

    policy = DiscoveryObservationPolicy()
    audit = _audit_discovery_observations(frame, entries; policy=policy)
    @test [entry.storm_id for entry in audit.eligible_entries] == [1, 2]
    @test audit.policy === policy
    records = audit.storm_records
    @test records[1].eligible
    @test records[1].regression_rows == n
    @test records[1].scoring_rows == n - 1
    @test records[2].eligible
    @test records[2].admissible_cleaned_driver_rows == n
    @test records[2].admissible_filled_driver_rows == 2
    @test records[2].admissible_filled_driver_fraction == 2 / n
    @test records[2].regression_rows == n - 8 # radius-three target stencil
    @test records[2].scoring_rows == n - 3    # anchor excluded; two targets absent
    @test records[2].scoring_fraction == (n - 3) / (n - 1)
    @test records[2].scoring_admissible_filled_driver_rows == 2
    @test records[2].exclusion_reason == ""
    @test !records[3].eligible
    @test records[3].admissible_cleaned_driver_rows == n - 4
    @test records[3].scoring_rows == 18
    @test records[3].exclusion_reason ==
          "insufficient_original_dst_star_scoring_rows"

    cycles = audit.cycle_records
    @test getproperty.(cycles, :solar_cycle) == [23, 24]
    @test cycles[1].catalog_storms == 2
    @test cycles[1].eligible_storms == 2
    @test cycles[1].catalog_rows == 2n
    @test cycles[1].admissible_cleaned_driver_rows == 2n
    @test cycles[1].admissible_filled_driver_rows == 2
    @test cycles[1].regression_rows == 2n - 8
    @test cycles[1].scoring_rows == 2n - 4
    @test cycles[1].exclusion_reason == ""
    @test cycles[2].excluded_storms == 1
    @test occursin("insufficient_original_dst_star_scoring_rows=1",
                   cycles[2].exclusion_reason)
    explicit_records = _real_explicit_exclusion_records(records)
    @test explicit_records[1].exclusion_reason == "none"
    @test explicit_records[3].exclusion_reason ==
          "insufficient_original_dst_star_scoring_rows"

    # A finite value is not automatically an admissible cleaned value: a
    # manually filled four-hour raw gap remains excluded by the <=3 h policy.
    tampered = deepcopy(frame)
    tampered.V[long_gap] .= 450.0
    tampered.Pdyn[long_gap] .= dynamic_pressure(5.0, 450.0)
    tampered_state = _discovery_window_policy(tampered, entries[3], policy)
    @test count(tampered_state.driver_mask) == n - 4
    @test all(.!tampered_state.driver_mask[18:21])
    bad_bound = deepcopy(frame)
    bad_bound.V[n + 7] = NaN # flag says original, but cleaned bound is corrupted
    bad_bound_state = _discovery_window_policy(bad_bound, entries[2], policy)
    @test all(.!bad_bound_state.driver_mask[7:9])
    ae_frame = deepcopy(frame)
    ae_frame.AE_observed[(n + 25):(n + 26)] .= false
    coupled_policy = DiscoveryObservationPolicy(require_ae=true)
    coupled_state = _discovery_window_policy(ae_frame, entries[2], coupled_policy)
    @test coupled_state.record.require_ae
    @test coupled_state.record.regression_rows == records[2].regression_rows - 8

    # Availability thresholds are prospective and exact at their boundaries.
    state2 = _discovery_window_policy(frame, entries[2], policy)
    exact = DiscoveryObservationPolicy(
        min_regression_rows=state2.record.regression_rows,
        min_scoring_rows=state2.record.scoring_rows,
        min_scoring_fraction=state2.record.scoring_fraction,
    )
    @test _discovery_window_policy(frame, entries[2], exact).record.eligible
    too_few_regression = DiscoveryObservationPolicy(
        min_regression_rows=state2.record.regression_rows + 1,
        min_scoring_rows=1,
        min_scoring_fraction=0.01,
    )
    @test _discovery_window_policy(frame, entries[2], too_few_regression).
          record.exclusion_reason == "insufficient_original_support_regression_rows"
    too_few_scores = DiscoveryObservationPolicy(
        min_regression_rows=1,
        min_scoring_rows=state2.record.scoring_rows + 1,
        min_scoring_fraction=0.01,
    )
    @test _discovery_window_policy(frame, entries[2], too_few_scores).
          record.exclusion_reason == "insufficient_original_dst_star_scoring_rows"
    too_sparse = DiscoveryObservationPolicy(
        min_regression_rows=1,
        min_scoring_rows=1,
        min_scoring_fraction=nextfloat(state2.record.scoring_fraction),
    )
    @test _discovery_window_policy(frame, entries[2], too_sparse).
          record.exclusion_reason == "insufficient_original_dst_star_scoring_fraction"

    @test_throws ArgumentError DiscoveryObservationPolicy(smooth_window=4)
    @test_throws ArgumentError DiscoveryObservationPolicy(min_regression_rows=0)
    @test_throws ArgumentError DiscoveryObservationPolicy(min_scoring_rows=0)
    @test_throws ArgumentError DiscoveryObservationPolicy(min_scoring_fraction=0.0)
    @test_throws ArgumentError DiscoveryObservationPolicy(min_scoring_fraction=NaN)
    @test_throws ArgumentError _predeclare_discovery_storms(
        frame, entries; policy=policy, smooth_window=3,
    )
    missing_flags = select(frame, Not(:Dst_observed))
    @test_throws ArgumentError _audit_discovery_observations(missing_flags, entries)
    invalid_flags = deepcopy(frame)
    invalid_flags[!, :Dst_observed] = Any[frame.Dst_observed...]
    invalid_flags.Dst_observed[1] = "not-a-flag"
    @test_throws ArgumentError _audit_discovery_observations(invalid_flags, entries)
    @test_throws ArgumentError _audit_discovery_observations(
        select(frame, Not(:AE_observed)), entries; policy=coupled_policy,
    )
    @test_throws ArgumentError _audit_discovery_observations(
        frame, [entries[1], entries[1]],
    )
    invalid_entry = StormCatalogEntry(4, datetime[1], -30.0, datetime[1],
        datetime[1], 1.0, 23, "train", 0, n)
    @test_throws ArgumentError _audit_discovery_observations(frame, [invalid_entry])
    empty_audit = _audit_discovery_observations(frame, StormCatalogEntry[])
    @test isempty(empty_audit.eligible_entries)
    @test isempty(empty_audit.storm_records)
    @test isempty(empty_audit.cycle_records)

    eligible, wrapper_records = _predeclare_discovery_storms(frame, entries)
    @test getproperty.(eligible, :storm_id) == [1, 2]
    @test wrapper_records == records

    lib = build_solar_wind_library()
    @test length(lib) == 20
    prepared = _prepare_discovery_storm(frame, entries[2], lib; policy=policy)
    @test size(prepared.theta) == (n - 8, 20)
    @test length(prepared.target) == n - 8
    @test prepared.observation_record.admissible_filled_driver_rows == 2
    @test all(isnan, prepared.scoring_observations[8:9])
    @test count(isfinite, prepared.scoring_observations) == n - 2
    @test_throws ArgumentError _prepare_discovery_storm(frame, entries[3], lib)

    # Metrics use only original targets; the cleaned values at 8:9 never enter
    # the observed vector or n_points count.
    scored_real = _score_discovery_storm(prepared, zeros(length(lib)), lib)
    @test scored_real.anchor_index == 1
    @test scored_real.scored_indices == vcat(collect(2:7), collect(10:n))
    @test all(isnan, scored_real.observations[8:9])
    @test all(row -> row.n_points == n - 3, scored_real.metrics)
    @test length(unique(row.scored_catalog_indices for row in scored_real.metrics)) == 1
    @test length(unique(row.cohort_signature_sha256 for row in scored_real.metrics)) == 1
    @test _real_assert_exact_metric_cohorts(scored_real.metrics)
    bad_cohort = copy(scored_real.metrics)
    bad_cohort[1] = merge(bad_cohort[1], (cohort_signature_sha256="tampered",))
    @test_throws ErrorException _real_assert_exact_metric_cohorts(bad_cohort)
    for row in scored_real.metrics
        prediction = getproperty(scored_real.predictions, Symbol(row.model))
        @test row.rmse_nt ≈ rmse(prediction[scored_real.scored_indices],
                                 scored_real.observations[scored_real.scored_indices])
    end
    reconstruction = _real_may2024_frame(frame, prepared, scored_real)
    @test all(reconstruction.storm_id .== prepared.storm_id)
    @test all(isfinite, reconstruction.dst_observed_nt[8:9])
    @test all(isnan, reconstruction.dst_star_observed_nt[8:9])
    @test all(.!reconstruction.dst_star_original_target_flag[8:9])
    @test reconstruction.dst_cleaned_nt != reconstruction.dst_star_cleaned_nt
    invalid_observations = copy(prepared.scoring_observations)
    invalid_observations[2] = Inf
    @test_throws ArgumentError _score_discovery_storm(
        merge(prepared, (scoring_observations=invalid_observations,)),
        zeros(length(lib)), lib,
    )

    # Direct segment adversaries pin deterministic gap splitting and sparse-score
    # fractions independently of DataFrame preprocessing.
    segmented = _best_forward_scoring_segment(
        Bool[trues(12)..., falses(4)..., trues(20)...], trues(36),
    )
    @test segmented.anchor == 17
    @test segmented.endpoint == 36
    @test segmented.scoring_rows == 19
    sparse = _best_forward_scoring_segment(trues(10),
        Bool[true, false, false, true, false, false, true, false, false, true])
    @test sparse.scoring_rows == 3
    @test sparse.scoring_fraction == 1 / 3
    @test _best_forward_scoring_segment(falses(5), trues(5)).scoring_rows == 0

    selection_lib = build_minimal_library()
    synthetic = NamedTuple[]
    for storm_id in 1:2
        swd, _ = generate_synthetic_storm(seed=storm_id, duration=8.0,
                                          noise_level=0.0)
        data, target = prepare_sindy_data(swd, 1.0; smooth_window=5)
        theta = evaluate_library(selection_lib, data)
        push!(synthetic, (
            storm_id = storm_id,
            onset_time = DateTime(1990 + storm_id),
            entry = nothing,
            swd = swd,
            data = data,
            theta = theta,
            target = target,
            observation_record = (
                scoring_start_idx=1000 * storm_id,
                scoring_end_idx=1000 * storm_id + length(swd.t) - 1,
            ),
            library_terms = Tuple(get_term_names(selection_lib)),
        ))
    end
    cache = Dict{Any,Any}()
    selection = _select_discovery_lambda(synthetic, selection_lib, cache)
    @test selection.selected_lambda in storm_lambda_grid()
    @test length(selection.model) == 3
    @test length(cache) == 2
    @test all(row -> row.anchor_index == 1 && row.n_scored == 8,
              selection.error_records)
    full_design = _cached_subset_design!(cache, reverse(synthetic))
    @test full_design === _cached_subset_design!(cache, synthetic)
    @test length(cache) == 2
    concatenated = _concat_discovery_data(reverse(synthetic))
    @test evaluate_library(selection_lib, concatenated) ≈ full_design.theta

    scored = _score_discovery_storm(synthetic[end], selection.model, selection_lib)
    @test scored.anchor_index == 1
    @test length(scored.metrics) == 4
    @test all(row -> row.n_points == 8, scored.metrics)
    @test all(first(getproperty(scored.predictions, model)) ==
              first(scored.swd.Dst_star) for model in propertynames(scored.predictions))
    for row in scored.metrics
        prediction = getproperty(scored.predictions, Symbol(row.model))
        @test row.rmse_nt ≈ rmse(@view(prediction[2:end]),
                                 @view(scored.swd.Dst_star[2:end]))
    end

    mktempdir() do root
        storm_path = _write_discovery_csv(joinpath(root, "storm_eligibility.csv"),
                                          records)
        cycle_path = _write_discovery_csv(joinpath(root, "cycle_eligibility.csv"),
                                          cycles)
        @test isfile(storm_path)
        @test isfile(cycle_path)
        persisted = CSV.read(storm_path, DataFrame)
        @test persisted.storm_id == [1, 2, 3]
        @test persisted.eligible == [true, true, false]
        @test persisted.admissible_filled_driver_rows == [0, 2, 0]
        persisted_cycles = CSV.read(cycle_path, DataFrame)
        @test persisted_cycles.solar_cycle == [23, 24]
    end

    @testset "Actual-design conditioning and grouped cancellation" begin
        x = collect(range(-1.0, 1.0; length=101))
        theta = hcat(ones(length(x)), x, x)
        coefficients = [0.0, 40.0, -40.0]
        terms = ["1", "sin(θ_c/2)", "sin²(θ_c/2)"]
        diagnostics = _real_conditioning_records(
            theta, coefficients, terms; basis="full",
        )
        full = only(filter(row -> row.block == "full_design", diagnostics))
        active = only(filter(row -> row.block == "selected_active_block", diagnostics))
        clock = only(filter(
            row -> row.block == "clock_proxy_candidate_block", diagnostics,
        ))
        @test full.n_rows == 101
        @test full.n_columns == 3
        @test full.rank == 2
        @test isinf(full.condition_number)
        @test active.rank == 1
        @test isinf(active.condition_number)
        @test clock.rank == 1
        @test clock.rank_tolerance > 0
        @test isinf(clock.condition_number)
        empty_block = _real_normalized_block(theta, Int[])
        @test empty_block.rank == 0
        @test isnan(empty_block.rank_tolerance)
        @test isnan(empty_block.condition_number)
        @test isnan(empty_block.largest_singular_value)
        @test isnan(empty_block.smallest_singular_value)

        contributions = _real_contribution_records(
            theta, coefficients, terms; basis="full",
        )
        grouped = only(filter(row -> row.name == "clock_proxy_net", contributions))
        @test grouped.net_contribution_min == 0.0
        @test grouped.net_contribution_max == 0.0
        @test grouped.gross_contribution_absmax == 80.0
        @test grouped.cancellation_ratio > 1e15
        @test grouped.interpretation_status == "rank_deficient_grouped_net_only"
        term = only(filter(row -> row.name == "sin(θ_c/2)" &&
                                 row.record_kind == "term", contributions))
        @test term.net_contribution_min == -40.0
        @test term.net_contribution_max == 40.0
        @test term.cancellation_ratio == 1.0
    end

    @testset "Raw subsample draws retain structural zeros" begin
        draws = [1.0 0.0 3.0 0.0; 0.0 0.0 0.0 0.0]
        frozen = copy(draws)
        records = _real_empirical_subsample_records(
            ["active", "inactive"], draws, [0.5, 0.0];
            lambda=2.0, seed=42, subsample_fraction=0.8,
        )
        @test draws == frozen
        @test records[1].nonzero_draws == 2
        @test records[1].structural_zero_draws == 2
        @test records[1].conditional_nonzero_median == 2.0
        @test records[1].conditional_nonzero_empirical_q025 ≈ 1.05
        @test records[1].conditional_nonzero_empirical_q975 ≈ 2.95
        @test !records[1].confidence_interval
        @test records[2].structural_zero_draws == 4
        @test isnan(records[2].conditional_nonzero_median)
        @test isnan(records[2].conditional_nonzero_empirical_q025)
        @test isnan(records[2].conditional_nonzero_empirical_q975)
        buffer = IOBuffer()
        CSV.write(buffer, DataFrame(records))
        serialized = String(take!(buffer))
        @test occursin("NaN", serialized)
        roundtrip = CSV.read(IOBuffer(serialized), DataFrame)
        @test isnan(roundtrip.conditional_nonzero_median[2])
        @test isnan(roundtrip.conditional_nonzero_empirical_q025[2])
        @test isnan(roundtrip.conditional_nonzero_empirical_q975[2])
        persisted_shape = Matrix(DataFrame(permutedims(draws), [:active, :inactive]))
        @test persisted_shape == permutedims(frozen)
    end

    @testset "Full and collapsed selectors are independent on one cohort" begin
        full_library = build_solar_wind_library(clock_basis=:full)
        collapsed_library = build_solar_wind_library(clock_basis=:collapsed)
        full_storms = NamedTuple[]
        collapsed_storms = NamedTuple[]
        for storm_id in 1:2
            swd, _ = generate_synthetic_storm(
                seed=100 + storm_id, duration=8.0, noise_level=0.0,
            )
            data, target = prepare_sindy_data(swd, 1.0; smooth_window=5)
            common = (
                storm_id=storm_id,
                onset_time=DateTime(1990 + storm_id),
                entry=(split="train",),
                swd=swd,
                window_swd=swd,
                data=data,
                target=target,
                regression_mask=trues(length(target)),
                scoring_observations=swd.Dst_star,
                observation_record=(scoring_start_idx=1,
                                    scoring_end_idx=length(swd.t)),
            )
            push!(full_storms, merge(common, (
                theta=evaluate_library(full_library, data),
                library_terms=Tuple(get_term_names(full_library)),
            )))
            push!(collapsed_storms, merge(common, (
                theta=evaluate_library(collapsed_library, data),
                library_terms=Tuple(get_term_names(collapsed_library)),
            )))
        end
        _real_assert_matching_cohort(full_storms, collapsed_storms, "focused test")
        mismatched_target = copy(collapsed_storms[1].target)
        mismatched_target[1] = nextfloat(mismatched_target[1])
        @test_throws ErrorException _real_assert_matching_cohort(
            full_storms,
            [merge(collapsed_storms[1], (
                 target=mismatched_target,
             )), collapsed_storms[2]],
            "adversarial targets",
        )
        mismatched_mask = copy(collapsed_storms[1].regression_mask)
        mismatched_mask[1] = false
        @test_throws ErrorException _real_assert_matching_cohort(
            full_storms,
            [merge(collapsed_storms[1], (regression_mask=mismatched_mask,)),
             collapsed_storms[2]],
            "adversarial masks",
        )
        full_cache = Dict{Any,Any}()
        collapsed_cache = Dict{Any,Any}()
        selected = _real_select_basis_pair(
            full_storms, collapsed_storms, full_library, collapsed_library,
            full_cache, collapsed_cache; label="focused test",
        )
        collapsed_again = _select_discovery_lambda(
            collapsed_storms, collapsed_library, Dict{Any,Any}(),
        )
        @test length(selected.full.candidate_records) == 60
        @test length(selected.collapsed.candidate_records) == 60
        @test selected.collapsed.selected_lambda == collapsed_again.selected_lambda
        @test selected.collapsed.model == collapsed_again.model
        @test selected.full.split_records == selected.collapsed.split_records
        @test all(key -> first(key) == Tuple(get_term_names(full_library)),
                  keys(full_cache))
        @test all(key -> first(key) == Tuple(get_term_names(collapsed_library)),
                  keys(collapsed_cache))
    end

    @testset "Canonical CSV wrapper always writes and verifies a manifest" begin
        mktempdir() do root
            omni = joinpath(root, "omni.csv")
            catalog = joinpath(root, "catalog.csv")
            write(omni, "frozen input\n")
            write(catalog, "verified catalog\n")
            context = (data=root, omni=omni, catalog=catalog, mode=:test)
            path = _real_manifested_csv(
                context, "result.csv", [(value=1.0,)];
                selection_record=(kind="focused_test",),
            )
            @test isfile(path * ".manifest.json")
            record = verify_output_manifest(path;
                package_root=normpath(joinpath(@__DIR__, "..")),
            )
            @test Set(String(input["name"]) for input in record["inputs"]) ==
                  Set(("omni_extracted", "storm_catalog"))
        end
        mktempdir() do root
            context = (data=root,
                       omni=joinpath(root, "missing_omni.csv"),
                       catalog=joinpath(root, "missing_catalog.csv"),
                       mode=:test)
            result_path = joinpath(root, "orphan.csv")
            @test_throws ArgumentError _real_manifested_csv(
                context, "orphan.csv", [(value=1.0,)];
                selection_record=(kind="focused_test",),
            )
            @test !isfile(result_path)
            @test !isfile(result_path * ".manifest.json")
        end
    end
end
