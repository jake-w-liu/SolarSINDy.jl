# Locked canonical output inventory. Each basename is bound to its only accepted producer.

function _inventory_add!(inventory, producer, names)
    path = abspath(joinpath(@__DIR__, producer))
    for name in names
        haskey(inventory, name) && error("duplicate canonical artifact basename: $name")
        inventory[name] = path
    end
    return inventory
end

function _selection_artifacts(prefixes)
    suffixes = ("inner_split", "candidates", "validation_errors", "support", "decision")
    return ["$(prefix)_$(suffix).csv" for prefix in prefixes for suffix in suffixes]
end

const CANONICAL_DATA_ARTIFACT_INVENTORY = let inventory = Dict{String,String}()
    _inventory_add!(inventory, "download_omni.jl", (
        "observation_provenance.csv", "storm_catalog.csv",
    ))
    _inventory_add!(inventory, "real_data_discovery.jl", vcat(
        _selection_artifacts((
            "primary_lambda", "primary_collapsed_lambda",
            "cross_c20_22_to_c23_lambda",
            "cross_c20_22_to_c23_lambda_collapsed",
            "cross_even_to_odd_lambda", "cross_even_to_odd_lambda_collapsed",
            "cross_c20_23_to_c25_lambda",
            "cross_c20_23_to_c25_lambda_collapsed",
        )),
        [
            "real_storm_eligibility.csv", "real_cycle_observation_audit.csv",
            "real_lambda_sweep.csv", "real_design_conditioning.csv",
            "real_design_column_norms.csv", "real_contribution_diagnostics.csv",
            "real_sindy_discovery_coefficients.csv",
            "real_sindy_collapsed_coefficients.csv",
            "real_sindy_discovery_provenance.csv", "real_holdout_metrics.csv",
            "real_holdout_collapsed_metrics.csv", "may2024_reconstruction.csv",
            "cross_cycle_metrics.csv", "cross_cycle_collapsed_metrics.csv",
        ],
    ))
    _inventory_add!(inventory, "generate_ensemble_draws.jl", (
        "real_sindy_ensemble_draws.csv", "real_ensemble_inclusion.csv",
        "real_sindy_coefficients.csv", "real_primary_refit_audit.csv",
    ))
    _inventory_add!(inventory, "phase_dependent_discovery.jl", vcat(
        _selection_artifacts(("phase_switching_lambda", "phase_single_lambda")),
        [
            "phase_observation_audit.csv", "phase_observation_cycle_audit.csv",
            "phase_cohort_audit.csv", "phase_dependent_real_coefficients.csv",
            "phase_design_diagnostics.csv", "phase_single_control_coefficients.csv",
            "switching_model_metrics.csv", "switching_model_paired_metrics.csv",
        ],
    ))
    _inventory_add!(inventory, "phase_sensitivity.jl", vcat(
        _selection_artifacts(("phase_threshold_single_lambda",)),
        [
            "phase_threshold_sensitivity.csv",
            "phase_threshold_selection_decisions.csv",
            "phase_threshold_selection_candidates.csv",
            "phase_threshold_selection_errors.csv",
            "phase_threshold_selection_support.csv",
            "phase_threshold_selection_inner_split.csv",
            "phase_threshold_cohort_counts.csv",
            "phase_threshold_observation_audit.csv",
            "phase_threshold_outer_metrics.csv",
            "phase_threshold_outer_trajectories.csv",
            "phase_threshold_design_diagnostics.csv",
            "phase_threshold_single_control_coefficients.csv",
        ],
    ))
    _inventory_add!(inventory, "coupled_discovery.jl", vcat(
        _selection_artifacts(("coupled_lambda", "coupled_single_lambda")),
        [
            "coupled_observation_storm_audit.csv",
            "coupled_observation_cycle_audit.csv", "coupled_exclusion_audit.csv",
            "coupled_ae_secondary_candidates.csv", "coupled_ae_secondary_errors.csv",
            "coupled_equation_fit_audit.csv", "coupled_single_inner_split_audit.csv",
            "coupled_single_coefficients.csv", "coupled_design_diagnostics.csv",
            "coupled_coefficients.csv", "coupled_ensemble_draws.csv",
            "coupled_metrics.csv", "coupled_cohort_audit.csv",
        ],
    ))
    _inventory_add!(inventory, "synthetic_equation_recovery.jl", (
        "synthetic_equation_recovery_summary.csv",
        "synthetic_equation_recovery_coefficients.csv",
        "synthetic_equation_recovery_lambda_selection.csv",
        "synthetic_equation_recovery_trajectories.csv",
    ))
    _inventory_add!(inventory, "synthetic_robustness.jl", (
        "synthetic_noise_sweep.csv", "synthetic_noise_lambda_selection.csv",
        "synthetic_ablation.csv", "synthetic_ablation_lambda_selection.csv",
        "synthetic_scalability.csv", "synthetic_scalability_lambda_selection.csv",
    ))

    headline_prefixes = (
        "headline_validation_c24", "headline_c20_22_to_c23",
        "headline_even_to_odd", "headline_c20_23_to_c25",
    )
    headline_outputs = ["$(prefix)_$(suffix).csv" for prefix in headline_prefixes
                        for suffix in ("pairs", "bootstrap", "summary", "wilcoxon")]
    paired_outputs = [
        "paired_$(experiment)_vs_$(baseline)_$(suffix).csv"
        for experiment in ("validation_c24", "c20_22_to_c23", "even_to_odd",
                           "c20_23_to_c25")
        for baseline in ("burton_simplified", "burton_published")
        for suffix in ("pairs", "bootstrap", "summary")
    ]
    _inventory_add!(inventory, "significance_tests.jl", vcat(
        headline_outputs, paired_outputs,
        [
            "headline_sindy_vs_obrienmcp_holm_adjusted.csv",
            "headline_sindy_vs_obrienmcp_claim_sources.csv",
            "paired_sindy_vs_all_baselines_claim_sources.csv",
            "primary_model_metric_summary.csv",
        ],
    ))
    inventory
end

const CANONICAL_FIGURE_ARTIFACT_INVENTORY = let
    producer = abspath(joinpath(@__DIR__, "canonical_figure_generation.jl"))
    Dict(name => producer for name in (
        "fig_discovery_validation.pdf",
        "fig_lambda_selection.pdf",
        "fig_coefficient_stability.pdf",
        "fig_synthetic_recovery.pdf",
        "fig_paired_performance.pdf",
    ))
end
