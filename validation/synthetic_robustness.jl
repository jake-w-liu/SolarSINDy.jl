#!/usr/bin/env julia

module SyntheticRobustnessValidation

using SolarSINDy
using CSV
using DataFrames
using Statistics

include(joinpath(@__DIR__, "synthetic_equation_recovery.jl"))
using .SyntheticEquationRecoveryValidation

export DEFAULT_ROBUSTNESS_SEED, DEFAULT_NOISE_LEVELS,
       DEFAULT_NOISE_REPLICATES, DEFAULT_SCALABILITY_TRAINING_ROWS,
       run_synthetic_robustness

const DEFAULT_ROBUSTNESS_SEED = 20_000
const DEFAULT_NOISE_LEVELS = Float64[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
const DEFAULT_NOISE_REPLICATES = 10
const DEFAULT_SCALABILITY_TRAINING_ROWS = Int[80, 160, 240, 320, 400, 480, 560]
const _CANONICAL_NOISE_LEVELS = Tuple(DEFAULT_NOISE_LEVELS)
const _CANONICAL_SCALABILITY_TRAINING_ROWS =
    Tuple(DEFAULT_SCALABILITY_TRAINING_ROWS)
const _CANONICAL_LAMBDAS =
    Tuple(SyntheticEquationRecoveryValidation.DEFAULT_LAMBDAS)
const ABLATION_SEED_OFFSET = 10_000
const SCALABILITY_SEED_OFFSET = 20_000
const STANDARD_FIT_ROWS = 1:420
const STANDARD_VALIDATION_ROWS = 421:560
const STANDARD_REFIT_ROWS = 1:560
const STANDARD_HOLDOUT_ROWS = 561:801
const SCALABILITY_VALIDATION_ROWS = 561:640
const SCALABILITY_HOLDOUT_ROWS = 641:801

const ROBUSTNESS_LAMBDA_PROTOCOL =
    "fit on predeclared training rows; select among the fixed lambda grid on " *
    "later disjoint validation rows; retain lambdas within max(2% validation-" *
    "target SD, 1e-10) of minimum validation RMSE; choose fewest terms, then " *
    "lowest RMSE, then largest lambda; refit only on the predeclared refit " *
    "rows; evaluate once on fixed later oracle holdout rows"

function _canonical_ablation_variants()
    return [
        "minimal_identifiable" => build_minimal_library(),
        "no_clock_angle_terms" => build_solar_wind_library(
            max_poly_order=2, include_trig=false, include_cross=true,
            include_known=false,
        ),
        "no_polynomial_cross_terms" => build_solar_wind_library(
            max_poly_order=2, include_trig=true, include_cross=false,
            include_known=true,
        ),
        "full_canonical" => build_solar_wind_library(
            max_poly_order=2, include_trig=true, include_cross=true,
            include_known=true,
        ),
    ]
end

function _ablation_signatures(variants)
    return [(
        name=String(first(pair)),
        terms=Tuple(get_term_names(last(pair))),
        decay_term_available="Dst_star" in get_term_names(last(pair)),
        injection_term_available="V*Bs" in get_term_names(last(pair)),
    ) for pair in variants]
end

function _validate_rows(rows, n::Int, label::AbstractString)
    indices = collect(Int, rows)
    isempty(indices) && throw(ArgumentError("$label must not be empty"))
    issorted(indices) && length(unique(indices)) == length(indices) ||
        throw(ArgumentError("$label must be strictly increasing"))
    all(index -> 1 <= index <= n, indices) ||
        throw(ArgumentError("$label contains an out-of-range row"))
    return indices
end

function _select_lambda_rows(theta::Matrix{Float64}, target::Vector{Float64};
                             lambdas::Vector{Float64}, fit_rows,
                             validation_rows)
    size(theta, 1) == length(target) ||
        throw(DimensionMismatch("design rows and target length must match"))
    all(isfinite, theta) || throw(ArgumentError("design matrix must be finite"))
    all(isfinite, target) || throw(ArgumentError("target must be finite"))
    isempty(lambdas) && throw(ArgumentError("lambda grid must not be empty"))
    all(lambda -> isfinite(lambda) && lambda >= 0, lambdas) ||
        throw(ArgumentError("lambdas must be finite and nonnegative"))
    length(unique(lambdas)) == length(lambdas) ||
        throw(ArgumentError("lambda grid must not contain duplicates"))
    fit = _validate_rows(fit_rows, size(theta, 1), "fit_rows")
    validation = _validate_rows(
        validation_rows, size(theta, 1), "validation_rows",
    )
    maximum(fit) < minimum(validation) || throw(ArgumentError(
        "fit rows must precede the disjoint validation rows",
    ))
    length(validation) >= 2 ||
        throw(ArgumentError("validation_rows must contain at least two rows"))

    sweep = NamedTuple[]
    for lambda in lambdas
        coefficients = stlsq(theta[fit, :], target[fit];
                             λ=lambda, normalize=true)
        residual = theta[validation, :] * coefficients .- target[validation]
        score = sqrt(mean(abs2, residual))
        isfinite(score) || error("lambda validation RMSE is non-finite")
        push!(sweep, (
            lambda=lambda,
            validation_rmse_nt_per_hour=score,
            n_terms=count(!iszero, coefficients),
        ))
    end
    best_rmse = minimum(row.validation_rmse_nt_per_hour for row in sweep)
    tolerance = max(0.02 * std(target[validation]), 1e-10)
    isfinite(tolerance) || error("lambda validation tolerance is non-finite")
    eligible = filter(
        row -> row.validation_rmse_nt_per_hour <= best_rmse + tolerance,
        sweep,
    )
    sort!(eligible;
          by=row -> (row.n_terms, row.validation_rmse_nt_per_hour, -row.lambda))
    selected = first(eligible)
    annotated = [(
        row...,
        eligible=row.validation_rmse_nt_per_hour <= best_rmse + tolerance,
        selected=row.lambda == selected.lambda,
    ) for row in sweep]
    return (; lambda=selected.lambda, sweep=annotated, best_rmse, tolerance,
            fit_rows=fit, validation_rows=validation)
end

function _fit_fixed_holdout(bundle, library::CandidateLibrary;
                            lambdas::Vector{Float64}, fit_rows,
                            validation_rows, refit_rows, holdout_rows)
    theta = evaluate_library(library, bundle.data)
    selection = _select_lambda_rows(
        theta, bundle.observed_derivative;
        lambdas, fit_rows, validation_rows,
    )
    refit = _validate_rows(refit_rows, size(theta, 1), "refit_rows")
    holdout = _validate_rows(holdout_rows, size(theta, 1), "holdout_rows")
    maximum(refit) < minimum(holdout) || throw(ArgumentError(
        "refit rows must precede the fixed holdout rows",
    ))
    isempty(intersect(selection.validation_rows, holdout)) ||
        throw(ArgumentError("validation and holdout rows must be disjoint"))
    coefficients = stlsq(
        theta[refit, :], bundle.observed_derivative[refit];
        λ=selection.lambda, normalize=true,
    )
    all(isfinite, coefficients) || error("synthetic coefficients are non-finite")

    names = get_term_names(library)
    decay_index = findfirst(==("Dst_star"), names)
    injection_index = findfirst(==("V*Bs"), names)
    active = findall(!iszero, coefficients)
    decay_available = decay_index !== nothing
    injection_available = injection_index !== nothing
    true_positives = Int(decay_available && decay_index in active) +
                     Int(injection_available && injection_index in active)
    support_precision = isempty(active) ? 0.0 : true_positives / length(active)
    support_recall = true_positives / 2
    true_decay = -1.0 / CANONICAL_TAU
    true_injection = -CANONICAL_ALPHA
    decay = decay_available ? coefficients[decay_index] : missing
    injection = injection_available ? coefficients[injection_index] : missing
    decay_sign_ok = decay_available && !iszero(decay) &&
                    signbit(decay) == signbit(true_decay)
    injection_sign_ok = injection_available && !iszero(injection) &&
                        signbit(injection) == signbit(true_injection)
    decay_relative_error = decay_available ?
        abs((decay - true_decay) / true_decay) : missing
    injection_relative_error = injection_available ?
        abs((injection - true_injection) / true_injection) : missing

    derivative_prediction = theta[holdout, :] * coefficients
    derivative_rmse = sqrt(mean(abs2,
        derivative_prediction .- bundle.true_derivative[holdout],
    ))
    heldout = SyntheticEquationRecoveryValidation._heldout_solar_wind(
        bundle.swd, holdout,
    )
    simulation = simulate_sindy(
        coefficients, library, heldout, bundle.dt;
        Dst0=first(heldout.Dst_star),
    )
    forward_rmse = sqrt(mean(abs2, simulation .- heldout.Dst_star))
    all(isfinite, (support_precision, support_recall,
                   derivative_rmse, forward_rmse)) ||
        error("synthetic fixed-holdout metrics are non-finite")
    all(value -> ismissing(value) || isfinite(value),
        (decay, injection, decay_relative_error, injection_relative_error)) ||
        error("synthetic available-term metrics are non-finite")

    metrics = (
        seed=Int(bundle.seed),
        derivative_noise_std_nt_per_hour=
            Float64(bundle.derivative_noise_std_nt_per_hour),
        library_terms=length(names),
        active_terms=length(active),
        active_term_names=join(names[active], ";"),
        false_discoveries=length(active) - true_positives,
        support_precision=support_precision,
        support_recall=support_recall,
        decay_term_available=decay_available,
        injection_term_available=injection_available,
        decay_coefficient=decay,
        injection_coefficient=injection,
        decay_sign_ok,
        injection_sign_ok,
        decay_relative_error,
        injection_relative_error,
        holdout_derivative_rmse_nt_per_hour=derivative_rmse,
        heldout_forward_rmse_nt=forward_rmse,
        selected_lambda=selection.lambda,
        lambda_normalize=true,
        lambda_fit_first_row=first(selection.fit_rows),
        lambda_fit_last_row=last(selection.fit_rows),
        lambda_fit_rows=length(selection.fit_rows),
        lambda_validation_first_row=first(selection.validation_rows),
        lambda_validation_last_row=last(selection.validation_rows),
        lambda_validation_rows=length(selection.validation_rows),
        refit_first_row=first(refit),
        refit_last_row=last(refit),
        refit_rows=length(refit),
        holdout_first_row=first(holdout),
        holdout_last_row=last(holdout),
        holdout_rows=length(holdout),
        validation_rmse_tolerance_nt_per_hour=selection.tolerance,
    )
    return (; metrics, selection, coefficients, names)
end

function _lambda_records(result; analysis::AbstractString,
                         variant::AbstractString, replicate::Int,
                         seed::Int, noise_level::Real, training_rows::Int)
    return [(
        analysis=String(analysis),
        variant=String(variant),
        replicate,
        seed,
        derivative_noise_std_nt_per_hour=Float64(noise_level),
        training_rows,
        protocol=ROBUSTNESS_LAMBDA_PROTOCOL,
        lambda=row.lambda,
        validation_rmse_nt_per_hour=row.validation_rmse_nt_per_hour,
        n_terms=row.n_terms,
        eligible=row.eligible,
        selected=row.selected,
    ) for row in result.selection.sweep]
end

function _robustness_selection_record(; root_seed, noise_levels,
                                      noise_replicates, training_rows,
                                      lambdas, ablation_signatures)
    return (
        kind="seeded_synthetic_noise_ablation_and_training_data_scalability",
        equation="dDst_star_dt=-alpha*V*Bs-Dst_star/tau",
        alpha=CANONICAL_ALPHA,
        tau_hours=CANONICAL_TAU,
        derivative_noise_placement="regression_target_only",
        state_trajectory="noise_free_forward_Euler_oracle",
        root_seed,
        lambda_grid=Tuple(lambdas),
        lambda_protocol=ROBUSTNESS_LAMBDA_PROTOCOL,
        noise_sweep=(
            levels_nt_per_hour=Tuple(noise_levels),
            replicates=noise_replicates,
            seed_rule="root_seed + replicate; paired across noise levels",
            seeds=Tuple(root_seed .+ (1:noise_replicates)),
            libraries=("minimal_identifiable", "full_canonical"),
            fit_rows="1:420",
            validation_rows="421:560",
            refit_rows="1:560",
            holdout_rows="561:801",
        ),
        ablation=(
            seed=root_seed + ABLATION_SEED_OFFSET,
            derivative_noise_std_nt_per_hour=0.25,
            variants=Tuple(ablation_signatures),
            separately_selected_lambda=true,
            fit_rows="1:420",
            validation_rows="421:560",
            refit_rows="1:560",
            holdout_rows="561:801",
        ),
        scalability=(
            seed=root_seed + SCALABILITY_SEED_OFFSET,
            derivative_noise_std_nt_per_hour=0.25,
            training_rows=Tuple(training_rows),
            library="full_canonical",
            validation_rows="561:640",
            holdout_rows="641:801",
            evaluation_holdout_fixed_across_training_sizes=true,
        ),
    )
end

"""Run manifested synthetic noise, library-ablation, and data-scale studies."""
function run_synthetic_robustness(;
        root_seed::Int=DEFAULT_ROBUSTNESS_SEED,
        noise_levels::Vector{Float64}=copy(DEFAULT_NOISE_LEVELS),
        noise_replicates::Int=DEFAULT_NOISE_REPLICATES,
        scalability_training_rows::Vector{Int}=
            copy(DEFAULT_SCALABILITY_TRAINING_ROWS),
        lambdas::Vector{Float64}=
            copy(SyntheticEquationRecoveryValidation.DEFAULT_LAMBDAS),
        ablation_variants=_canonical_ablation_variants(),
        output_paths=SyntheticEquationRecoveryValidation.validation_output_paths(),
        persist::Bool=true)
    root_seed >= 0 || throw(ArgumentError("root_seed must be nonnegative"))
    isempty(noise_levels) && throw(ArgumentError("noise_levels must not be empty"))
    all(level -> isfinite(level) && level >= 0, noise_levels) ||
        throw(ArgumentError("noise levels must be finite and nonnegative"))
    length(unique(noise_levels)) == length(noise_levels) ||
        throw(ArgumentError("noise levels must not contain duplicates"))
    noise_replicates >= 1 ||
        throw(ArgumentError("noise_replicates must be positive"))
    isempty(scalability_training_rows) && throw(ArgumentError(
        "scalability_training_rows must not be empty",
    ))
    issorted(scalability_training_rows) &&
        length(unique(scalability_training_rows)) ==
            length(scalability_training_rows) || throw(ArgumentError(
                "scalability training rows must be sorted and unique",
            ))
    all(count -> 2 <= count < first(SCALABILITY_VALIDATION_ROWS),
        scalability_training_rows) || throw(ArgumentError(
            "scalability training-row counts are outside the pre-validation range",
        ))
    isempty(ablation_variants) &&
        throw(ArgumentError("ablation_variants must not be empty"))
    all(pair -> pair isa Pair && last(pair) isa CandidateLibrary,
        ablation_variants) || throw(ArgumentError(
            "ablation_variants must pair names with candidate libraries",
        ))
    ablation_names = String[first(pair) for pair in ablation_variants]
    length(unique(ablation_names)) == length(ablation_names) ||
        throw(ArgumentError("ablation variant names must be unique"))
    ablation_signatures = _ablation_signatures(ablation_variants)

    for field in (:data, :mode, :explicit)
        hasproperty(output_paths, field) || throw(ArgumentError(
            "output_paths is missing required field $field",
        ))
    end
    mode = SyntheticEquationRecoveryValidation._provenance_mode(output_paths.mode)
    if mode == :canonical
        output_paths.explicit || throw(ArgumentError(
            "canonical synthetic robustness requires an explicit output root",
        ))
        root_seed == DEFAULT_ROBUSTNESS_SEED || throw(ArgumentError(
            "canonical synthetic robustness requires root seed $DEFAULT_ROBUSTNESS_SEED",
        ))
        Tuple(noise_levels) == _CANONICAL_NOISE_LEVELS || throw(ArgumentError(
            "canonical synthetic robustness requires the predeclared noise grid",
        ))
        noise_replicates == DEFAULT_NOISE_REPLICATES || throw(ArgumentError(
            "canonical synthetic robustness requires $DEFAULT_NOISE_REPLICATES noise replicates",
        ))
        Tuple(scalability_training_rows) ==
            _CANONICAL_SCALABILITY_TRAINING_ROWS ||
            throw(ArgumentError(
                "canonical synthetic robustness requires the predeclared training-row grid",
            ))
        Tuple(lambdas) == _CANONICAL_LAMBDAS ||
            throw(ArgumentError(
                "canonical synthetic robustness requires the predeclared lambda grid",
            ))
        ablation_signatures ==
            _ablation_signatures(_canonical_ablation_variants()) ||
            throw(ArgumentError(
                "canonical synthetic robustness requires the predeclared ablations",
            ))
    end

    minimal_library = build_minimal_library()
    full_library = build_solar_wind_library()
    full_names = get_term_names(full_library)
    length(full_names) == 20 ||
        error("synthetic robustness requires the 20-term canonical library")
    "n*V^2" in full_names &&
        error("synthetic robustness must exclude duplicate n*V^2")
    noise_variants = (
        "minimal_identifiable" => minimal_library,
        "full_canonical" => full_library,
    )
    noise_metrics = NamedTuple[]
    noise_lambda = NamedTuple[]
    for noise_level in noise_levels, replicate in 1:noise_replicates
        seed = root_seed + replicate
        bundle = synthetic_recovery_dataset(
            seed=seed, derivative_noise_std=noise_level,
        )
        for pair in noise_variants
            variant = first(pair)
            result = _fit_fixed_holdout(
                bundle, last(pair); lambdas,
                fit_rows=STANDARD_FIT_ROWS,
                validation_rows=STANDARD_VALIDATION_ROWS,
                refit_rows=STANDARD_REFIT_ROWS,
                holdout_rows=STANDARD_HOLDOUT_ROWS,
            )
            push!(noise_metrics, merge((
                analysis="noise_sweep",
                variant,
                replicate,
                paired_noise_seed=true,
            ), result.metrics))
            append!(noise_lambda, _lambda_records(
                result; analysis="noise_sweep", variant,
                replicate, seed, noise_level,
                training_rows=length(STANDARD_REFIT_ROWS),
            ))
        end
    end

    ablation_seed = root_seed + ABLATION_SEED_OFFSET
    ablation_bundle = synthetic_recovery_dataset(
        seed=ablation_seed, derivative_noise_std=0.25,
    )
    ablation_metrics = NamedTuple[]
    ablation_lambda = NamedTuple[]
    for pair in ablation_variants
        variant = String(first(pair))
        library = last(pair)
        result = _fit_fixed_holdout(
            ablation_bundle, library; lambdas,
            fit_rows=STANDARD_FIT_ROWS,
            validation_rows=STANDARD_VALIDATION_ROWS,
            refit_rows=STANDARD_REFIT_ROWS,
            holdout_rows=STANDARD_HOLDOUT_ROWS,
        )
        push!(ablation_metrics, merge((
            analysis="library_ablation",
            variant,
            replicate=1,
        ), result.metrics))
        append!(ablation_lambda, _lambda_records(
            result; analysis="library_ablation", variant, replicate=1,
            seed=ablation_seed, noise_level=0.25,
            training_rows=length(STANDARD_REFIT_ROWS),
        ))
    end

    scalability_seed = root_seed + SCALABILITY_SEED_OFFSET
    scalability_bundle = synthetic_recovery_dataset(
        seed=scalability_seed, derivative_noise_std=0.25,
    )
    scalability_metrics = NamedTuple[]
    scalability_lambda = NamedTuple[]
    for training_rows in scalability_training_rows
        result = _fit_fixed_holdout(
            scalability_bundle, full_library; lambdas,
            fit_rows=1:training_rows,
            validation_rows=SCALABILITY_VALIDATION_ROWS,
            refit_rows=1:training_rows,
            holdout_rows=SCALABILITY_HOLDOUT_ROWS,
        )
        push!(scalability_metrics, merge((
            analysis="training_data_scalability",
            variant="full_canonical",
            replicate=1,
            requested_training_rows=training_rows,
            fixed_holdout=true,
        ), result.metrics))
        append!(scalability_lambda, _lambda_records(
            result; analysis="training_data_scalability",
            variant="full_canonical", replicate=1, seed=scalability_seed,
            noise_level=0.25, training_rows,
        ))
    end

    paths = (
        noise_metrics=joinpath(output_paths.data, "synthetic_noise_sweep.csv"),
        noise_lambda=joinpath(output_paths.data,
                              "synthetic_noise_lambda_selection.csv"),
        ablation_metrics=joinpath(output_paths.data, "synthetic_ablation.csv"),
        ablation_lambda=joinpath(output_paths.data,
                                 "synthetic_ablation_lambda_selection.csv"),
        scalability_metrics=joinpath(output_paths.data,
                                     "synthetic_scalability.csv"),
        scalability_lambda=joinpath(output_paths.data,
                                    "synthetic_scalability_lambda_selection.csv"),
    )
    frames = (
        noise_metrics=DataFrame(noise_metrics),
        noise_lambda=DataFrame(noise_lambda),
        ablation_metrics=DataFrame(ablation_metrics),
        ablation_lambda=DataFrame(ablation_lambda),
        scalability_metrics=DataFrame(scalability_metrics),
        scalability_lambda=DataFrame(scalability_lambda),
    )
    selection_record = _robustness_selection_record(
        ; root_seed, noise_levels, noise_replicates,
        training_rows=scalability_training_rows, lambdas, ablation_signatures,
    )
    if persist
        SyntheticEquationRecoveryValidation._persist_synthetic_outputs!(
            paths, frames, selection_record;
            seed=root_seed, mode,
            producer_script=@__FILE__,
        )
    end
    return (; frames, paths, selection_record)
end

function main()
    output_paths = SyntheticEquationRecoveryValidation.validation_output_paths()
    output_paths.explicit || error(
        "set SOLARSINDY_OUTPUT_ROOT to an explicit revision output directory",
    )
    result = run_synthetic_robustness(; output_paths)
    println("Synthetic noise rows: $(nrow(result.frames.noise_metrics))")
    println("Synthetic ablation rows: $(nrow(result.frames.ablation_metrics))")
    println("Synthetic scalability rows: $(nrow(result.frames.scalability_metrics))")
    return result
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SyntheticRobustnessValidation.main()
end
