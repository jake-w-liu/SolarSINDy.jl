#!/usr/bin/env julia

module SyntheticEquationRecoveryValidation

using SolarSINDy
using CSV
using DataFrames
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "output_paths.jl"))
include(joinpath(@__DIR__, "canonical_provenance.jl"))

const _SYNTHETIC_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

export CANONICAL_ALPHA, CANONICAL_TAU, DEFAULT_RECOVERY_SEED,
       synthetic_recovery_dataset, run_recovery_experiment,
       run_synthetic_equation_recovery

const CANONICAL_ALPHA = 5.4e-3
const CANONICAL_TAU = 7.7
const DEFAULT_RECOVERY_SEED = 31_415
const DEFAULT_LAMBDAS = Float64[
    0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0,
    20.0, 30.0, 50.0, 70.0, 100.0, 130.0, 160.0, 200.0,
    250.0, 300.0,
]
const LAMBDA_PROTOCOL =
    "chronological internal validation; retain lambdas within max(2% validation-target SD, 1e-10) of minimum validation RMSE; choose fewest terms, then lowest RMSE, then largest lambda; refit on all pre-holdout rows"
const CLOCK_RESPONSE_TERMS = (
    "sin(θ_c/2)", "sin²(θ_c/2)", "sin⁴(θ_c/2)",
    "sin^(8/3)(θ_c/2)", "V*sin²(θ_c/2)", "Newell_d_Φ",
)

"""
    synthetic_recovery_dataset(; seed=31415, derivative_noise_std=0.25,
                                 n_points=801, dt=0.25)

Create a persistently excited Burton-system trajectory. The state and derivative
are generated directly from `dDst*/dt = -5.4e-3 V Bs - Dst*/7.7`, independently
of the package baseline implementation. Gaussian noise is added only to the
regression target; the held-out state trajectory remains an exact forward oracle.
"""
function synthetic_recovery_dataset(; seed::Int=DEFAULT_RECOVERY_SEED,
                                      derivative_noise_std::Real=0.25,
                                      n_points::Int=801,
                                      dt::Real=0.25)
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    n_points >= 20 || throw(ArgumentError("n_points must be at least 20"))
    dt isa Real && !(dt isa Bool) && isfinite(dt) && dt > 0 ||
        throw(ArgumentError("dt must be a finite positive real number"))
    derivative_noise_std isa Real && !(derivative_noise_std isa Bool) &&
        isfinite(derivative_noise_std) && derivative_noise_std >= 0 ||
        throw(ArgumentError("derivative_noise_std must be finite and nonnegative"))

    rng = MersenneTwister(seed)
    step = Float64(dt)
    t = collect(0.0:step:(step * (n_points - 1)))
    V = 450.0 .+ 65.0 .* sin.(2pi .* t ./ 17.0) .+
        35.0 .* cos.(2pi .* t ./ 6.7) .+ 12.0 .* randn(rng, n_points)
    Bs = max.(0.2, 6.0 .+ 3.5 .* sin.(2pi .* t ./ 11.0 .+ 0.3) .+
                    2.0 .* cos.(2pi .* t ./ 4.3) .+
                    0.8 .* randn(rng, n_points))
    Bz = .-Bs
    By = 3.0 .* sin.(2pi .* t ./ 8.3) .+ 0.7 .* randn(rng, n_points)
    density = max.(1.0, 6.0 .+ 2.0 .* sin.(2pi .* t ./ 13.0) .+
                         0.5 .* randn(rng, n_points))
    Pdyn = dynamic_pressure.(density, V)

    dst_star = zeros(n_points)
    true_derivative = zeros(n_points)
    for k in eachindex(t)
        true_derivative[k] = -CANONICAL_ALPHA * V[k] * Bs[k] -
                             dst_star[k] / CANONICAL_TAU
        k < n_points &&
            (dst_star[k + 1] = dst_star[k] + step * true_derivative[k])
    end
    observed_derivative = true_derivative .+
        Float64(derivative_noise_std) .* randn(rng, n_points)
    theta_c = imf_clock_angle(By, Bz)
    data = Dict{String,Vector{Float64}}(
        "V" => V,
        "Bs" => Bs,
        "Bz" => Bz,
        "By" => By,
        "n" => density,
        "Pdyn" => Pdyn,
        "Dst_star" => dst_star,
        "theta_c" => theta_c,
        "BT" => hypot.(By, Bz),
    )
    swd = SolarWindData(t, V, Bz, By, density, Pdyn, copy(dst_star), dst_star)
    return (; data, swd, true_derivative, observed_derivative,
            seed,
            derivative_noise_std_nt_per_hour=Float64(derivative_noise_std),
            dt=step)
end

function _select_lambda(theta::Matrix{Float64}, target::Vector{Float64};
                        lambdas::Vector{Float64}, fit_stop::Int,
                        validation_stop::Int)
    size(theta, 1) == length(target) ||
        throw(DimensionMismatch("design rows and target length must match"))
    size(theta, 2) >= 1 || throw(ArgumentError("design matrix must have a column"))
    all(isfinite, theta) || throw(ArgumentError("design matrix must be finite"))
    all(isfinite, target) || throw(ArgumentError("target must be finite"))
    isempty(lambdas) && throw(ArgumentError("lambda grid must not be empty"))
    all(lambda -> isfinite(lambda) && lambda >= 0, lambdas) ||
        throw(ArgumentError("lambdas must be finite and nonnegative"))
    length(unique(lambdas)) == length(lambdas) ||
        throw(ArgumentError("lambda grid must not contain duplicates"))
    1 <= fit_stop < validation_stop <= size(theta, 1) ||
        throw(ArgumentError("invalid chronological lambda-selection split"))
    fit_rows = 1:fit_stop
    validation_rows = (fit_stop + 1):validation_stop
    sweep = NamedTuple[]
    for lambda in lambdas
        coefficients = stlsq(theta[fit_rows, :], target[fit_rows];
                             λ=lambda, normalize=true)
        residual = theta[validation_rows, :] * coefficients .- target[validation_rows]
        push!(sweep, (
            lambda=lambda,
            validation_rmse_nt_per_hour=sqrt(mean(abs2, residual)),
            n_terms=count(!iszero, coefficients),
        ))
    end
    best_rmse = minimum(row.validation_rmse_nt_per_hour for row in sweep)
    tolerance = max(0.02 * std(target[validation_rows]), 1e-10)
    eligible = filter(
        row -> row.validation_rmse_nt_per_hour <= best_rmse + tolerance, sweep,
    )
    sort!(eligible; by=row -> (row.n_terms, row.validation_rmse_nt_per_hour, -row.lambda))
    selected = first(eligible)
    annotated = [(
        row...,
        eligible=row.validation_rmse_nt_per_hour <= best_rmse + tolerance,
        selected=row.lambda == selected.lambda,
    ) for row in sweep]
    return (; lambda=selected.lambda, sweep=annotated, best_rmse, tolerance,
            fit_stop, validation_stop)
end

function _maximum_pair_correlation(theta::Matrix{Float64})
    variable = [j for j in axes(theta, 2) if std(@view(theta[:, j])) > sqrt(eps())]
    length(variable) < 2 && return 0.0
    correlations = cor(theta[:, variable])
    result = 0.0
    for i in 2:size(correlations, 1), j in 1:(i - 1)
        result = max(result, abs(correlations[i, j]))
    end
    return result
end

function _heldout_solar_wind(swd::SolarWindData, rows)
    return SolarWindData(
        swd.t[rows], swd.V[rows], swd.Bz[rows], swd.By[rows], swd.n[rows],
        swd.Pdyn[rows], swd.Dst[rows], swd.Dst_star[rows],
    )
end

"""
    run_recovery_experiment(bundle, library; experiment, objective, ...)

Select lambda without holdout contact, refit, and compute structural coefficient
recovery plus held-out forward-simulation diagnostics. Inclusion by itself is not
used as a recovery criterion.
"""
function run_recovery_experiment(bundle, library::CandidateLibrary;
                                 experiment::AbstractString,
                                 objective::AbstractString,
                                 lambdas::Vector{Float64}=copy(DEFAULT_LAMBDAS),
                                 lambda_fit_stop::Int=420,
                                 train_stop::Int=560,
                                 coefficient_relative_error_max::Real=0.05,
                                 forward_rmse_max::Real=0.5,
                                 stress_condition_min::Real=100.0)
    isempty(strip(experiment)) && throw(ArgumentError("experiment must not be blank"))
    objectives = ("identifiable_recovery", "false_discovery_collinearity_stress")
    objective in objectives || throw(ArgumentError(
        "objective must be one of $(join(objectives, ", "))"
    ))
    all(value -> isfinite(value) && value >= 0,
        (coefficient_relative_error_max, forward_rmse_max, stress_condition_min)) ||
        throw(ArgumentError("recovery thresholds must be finite and nonnegative"))
    n_points = length(bundle.observed_derivative)
    length(bundle.true_derivative) == n_points ||
        throw(DimensionMismatch("true and observed derivative lengths must match"))
    all(isfinite, bundle.true_derivative) ||
        throw(ArgumentError("true derivative must be finite"))
    all(isfinite, bundle.observed_derivative) ||
        throw(ArgumentError("observed derivative must be finite"))
    isfinite(bundle.dt) && bundle.dt > 0 ||
        throw(ArgumentError("bundle dt must be finite and positive"))
    lambda_fit_stop < train_stop < n_points ||
        throw(ArgumentError("recovery splits must leave internal validation and holdout rows"))
    theta = evaluate_library(library, bundle.data)
    selection = _select_lambda(
        theta,
        bundle.observed_derivative;
        lambdas=lambdas,
        fit_stop=lambda_fit_stop,
        validation_stop=train_stop,
    )
    coefficients = stlsq(
        theta[1:train_stop, :],
        bundle.observed_derivative[1:train_stop];
        λ=selection.lambda,
        normalize=true,
    )
    selection_column_norms = [norm(@view(theta[1:lambda_fit_stop, index]))
                              for index in axes(theta, 2)]
    column_norms = [norm(@view(theta[1:train_stop, index]))
                    for index in axes(theta, 2)]

    names = get_term_names(library)
    decay_index = findfirst(==("Dst_star"), names)
    injection_index = findfirst(==("V*Bs"), names)
    decay_index === nothing && error("recovery library lacks Dst_star")
    injection_index === nothing && error("recovery library lacks V*Bs")
    truth_indices = [decay_index, injection_index]
    active = findall(!iszero, coefficients)
    true_positives = count(index -> index in truth_indices, active)
    support_precision = isempty(active) ? 0.0 : true_positives / length(active)
    support_recall = true_positives / length(truth_indices)
    false_discoveries = length(active) - true_positives

    true_decay = -1.0 / CANONICAL_TAU
    true_injection = -CANONICAL_ALPHA
    decay = coefficients[decay_index]
    injection = coefficients[injection_index]
    decay_relative_error = abs((decay - true_decay) / true_decay)
    injection_relative_error = abs((injection - true_injection) / true_injection)
    decay_sign_ok = signbit(decay) == signbit(true_decay) && !iszero(decay)
    injection_sign_ok = signbit(injection) == signbit(true_injection) &&
                        !iszero(injection)

    holdout_rows = (train_stop + 1):n_points
    heldout = _heldout_solar_wind(bundle.swd, holdout_rows)
    simulated = simulate_sindy(
        coefficients,
        library,
        heldout,
        bundle.dt;
        Dst0=first(heldout.Dst_star),
    )
    forward_rmse = sqrt(mean(abs2, simulated .- heldout.Dst_star))

    clock_indices = findall(name -> name in CLOCK_RESPONSE_TERMS, names)
    clock_false_discoveries = count(index -> index in clock_indices, active)
    diagnostics = collinearity_diagnostics(
        theta[1:train_stop, :],
        coefficients;
        groups=[clock_indices],
    )
    clock_diagnostics = first(diagnostics.groups)
    maximum_pair_correlation = _maximum_pair_correlation(theta[1:train_stop, :])

    support_pass = support_precision >= 1.0 && support_recall >= 1.0
    coefficient_pass = decay_sign_ok && injection_sign_ok &&
        decay_relative_error <= coefficient_relative_error_max &&
        injection_relative_error <= coefficient_relative_error_max
    forecast_pass = forward_rmse <= forward_rmse_max
    recovery_pass = support_pass && coefficient_pass && forecast_pass
    stress_detected = clock_false_discoveries >= 1 &&
        clock_diagnostics.cond >= stress_condition_min &&
        maximum_pair_correlation >= 0.95
    # The full-library run is an outcome-neutral stress experiment.  Its
    # diagnostics must be persisted whether the redundant clock block is
    # selected cleanly, produces false discoveries, or loses true support.
    # `experiment_pass` records whether the declared stress signature was
    # observed; it is deliberately not a canonical persistence gate.
    experiment_pass = objective == "identifiable_recovery" ? recovery_pass : stress_detected
    canonical_gate_applied = objective == "identifiable_recovery"
    outcome_label = objective == "identifiable_recovery" ?
        (recovery_pass ? "recovery_pass" : "recovery_fail") :
        (stress_detected ? "stress_detected" : "stress_not_detected")

    summary = (
        experiment=String(experiment),
        objective=String(objective),
        seed=bundle.seed,
        n_points=n_points,
        dt_hours=bundle.dt,
        derivative_noise_std_nt_per_hour=bundle.derivative_noise_std_nt_per_hour,
        alpha_true=CANONICAL_ALPHA,
        tau_true=CANONICAL_TAU,
        true_decay_coefficient=true_decay,
        true_injection_coefficient=true_injection,
        library_terms=length(names),
        active_terms=length(active),
        active_term_names=join(names[active], ";"),
        false_discoveries=false_discoveries,
        clock_false_discoveries=clock_false_discoveries,
        support_precision=support_precision,
        support_recall=support_recall,
        decay_coefficient=decay,
        injection_coefficient=injection,
        decay_sign_ok=decay_sign_ok,
        injection_sign_ok=injection_sign_ok,
        decay_relative_error=decay_relative_error,
        injection_relative_error=injection_relative_error,
        heldout_forward_rmse_nt=forward_rmse,
        selected_lambda=selection.lambda,
        lambda_normalize=true,
        lambda_protocol=LAMBDA_PROTOCOL,
        lambda_fit_rows=lambda_fit_stop,
        lambda_validation_rows=train_stop - lambda_fit_stop,
        refit_rows=train_stop,
        heldout_rows=length(holdout_rows),
        validation_rmse_tolerance_nt_per_hour=selection.tolerance,
        active_condition_number=diagnostics.block_cond,
        active_cancellation_ratio=diagnostics.cancellation,
        clock_block_condition_number=clock_diagnostics.cond,
        clock_block_cancellation_ratio=clock_diagnostics.cancellation,
        maximum_pair_correlation=maximum_pair_correlation,
        support_precision_min=1.0,
        support_recall_min=1.0,
        coefficient_relative_error_max=Float64(coefficient_relative_error_max),
        heldout_forward_rmse_max_nt=Float64(forward_rmse_max),
        stress_condition_min=Float64(stress_condition_min),
        stress_pair_correlation_min=0.95,
        stress_clock_false_discoveries_min=1,
        support_pass=support_pass,
        coefficient_pass=coefficient_pass,
        forecast_pass=forecast_pass,
        recovery_pass=recovery_pass,
        stress_detected=stress_detected,
        experiment_pass=experiment_pass,
        canonical_gate_applied=canonical_gate_applied,
        outcome_label=outcome_label,
    )
    return (; summary, coefficients, selection_column_norms, column_norms,
            names, active, selection, simulated)
end

function _coefficient_rows(result)
    true_coefficients = Dict(
        "Dst_star" => -1.0 / CANONICAL_TAU,
        "V*Bs" => -CANONICAL_ALPHA,
    )
    return [(
        experiment=result.summary.experiment,
        term=name,
        discovered_coefficient=result.coefficients[index],
        selected=!iszero(result.coefficients[index]),
        true_support=haskey(true_coefficients, name),
        true_coefficient=get(true_coefficients, name, missing),
        lambda_fit_column_norm=result.selection_column_norms[index],
        refit_column_norm=result.column_norms[index],
    ) for (index, name) in enumerate(result.names)]
end

function _lambda_rows(result)
    return [(
        experiment=result.summary.experiment,
        seed=result.summary.seed,
        protocol=LAMBDA_PROTOCOL,
        lambda=row.lambda,
        validation_rmse_nt_per_hour=row.validation_rmse_nt_per_hour,
        n_terms=row.n_terms,
        eligible=row.eligible,
        selected=row.selected,
    ) for row in result.selection.sweep]
end

function _write_csv_atomic(path::AbstractString, frame::DataFrame)
    mkpath(dirname(path))
    SolarSINDy._require_regular_output_target(path)
    temporary, io = mktemp(dirname(path); cleanup=false)
    close(io)
    try
        CSV.write(temporary, frame)
        SolarSINDy._atomic_replace_regular(temporary, path)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function _trajectory_rows(bundle, result)
    first_global_row = result.summary.refit_rows + 1
    last_global_row = first_global_row + length(result.simulated) - 1
    last_global_row <= length(bundle.swd.t) || error(
        "synthetic held-out trajectory exceeds the generated dataset",
    )
    return [(
        experiment=result.summary.experiment,
        objective=result.summary.objective,
        seed=result.summary.seed,
        global_row=global_row,
        holdout_row=holdout_row,
        time_hours=bundle.swd.t[global_row],
        dst_star_oracle_nt=bundle.swd.Dst_star[global_row],
        dst_star_simulated_nt=result.simulated[holdout_row],
        error_nt=result.simulated[holdout_row] - bundle.swd.Dst_star[global_row],
        is_anchor=holdout_row == 1,
    ) for (holdout_row, global_row) in enumerate(first_global_row:last_global_row)]
end

function _synthetic_selection_record(bundle, minimal, full, validation_pass)
    return (
        kind="seeded_synthetic_equation_recovery_and_collinearity_stress_test",
        equation="dDst_star_dt=-alpha*V*Bs-Dst_star/tau",
        alpha=CANONICAL_ALPHA,
        alpha_units="nT_per_hour_per_km_per_second_per_nT",
        tau_hours=CANONICAL_TAU,
        generator=(
            n_points=length(bundle.swd.t),
            dt_hours=bundle.dt,
            derivative_noise_std_nt_per_hour=bundle.derivative_noise_std_nt_per_hour,
            derivative_noise_placement="regression_target_only",
            state_trajectory="noise_free_forward_Euler_oracle",
        ),
        libraries=(
            minimal=(
                objective=minimal.summary.objective,
                terms=Tuple(minimal.names),
                selected_lambda=minimal.selection.lambda,
            ),
            full=(
                objective=full.summary.objective,
                terms=Tuple(full.names),
                selected_lambda=full.selection.lambda,
            ),
        ),
        lambda_grid=Tuple(DEFAULT_LAMBDAS),
        normalize=true,
        lambda_protocol=LAMBDA_PROTOCOL,
        split=(
            lambda_fit_rows="1:420",
            lambda_validation_rows="421:560",
            refit_rows="1:560",
            holdout_rows="561:801",
        ),
        forward_anchor="first_heldout_oracle_Dst_star",
        acceptance_thresholds=(
            support_precision_min=minimal.summary.support_precision_min,
            support_recall_min=minimal.summary.support_recall_min,
            coefficient_relative_error_max=minimal.summary.coefficient_relative_error_max,
            heldout_forward_rmse_max_nt=minimal.summary.heldout_forward_rmse_max_nt,
            stress_condition_min=full.summary.stress_condition_min,
            stress_pair_correlation_min=full.summary.stress_pair_correlation_min,
            stress_clock_false_discoveries_min=
                full.summary.stress_clock_false_discoveries_min,
        ),
        minimal_experiment_pass=minimal.summary.experiment_pass,
        full_experiment_pass=full.summary.experiment_pass,
        full_stress_detected=full.summary.stress_detected,
        full_stress_outcome_neutral=true,
        canonical_gate_scope="minimal_identifiable_recovery_only",
        overall_validation_pass=validation_pass,
    )
end

function _persist_synthetic_outputs!(paths, frames, base_selection_record;
                                     seed::Int,
                                     mode,
                                     producer_script::AbstractString=@__FILE__,
                                     package_root::AbstractString=_SYNTHETIC_PACKAGE_ROOT,
                                     _after_artifact_hook::Function=(name, path) -> nothing)
    propertynames(paths) == propertynames(frames) || throw(ArgumentError(
        "synthetic output paths and frames must have identical fields",
    ))
    run_mode = _provenance_mode(mode)
    output_paths = collect(String, values(paths))
    transaction_paths = vcat(
        output_paths, [path * ".manifest.json" for path in output_paths],
    )
    snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
    try
        for name in propertynames(paths)
            path = getproperty(paths, name)
            frame = getproperty(frames, name)
            _write_csv_atomic(path, frame)
            write_output_manifest(path;
                producer_script,
                input_paths=(;),
                selection_record=merge(base_selection_record, (artifact=String(name),)),
                seed,
                metadata=(rows=nrow(frame), columns=Tuple(String.(names(frame)))),
                package_root,
                mode=run_mode,
            )
            verify_output_manifest(path;
                package_root,
                require_canonical=run_mode == :canonical,
                verify_source=true,
            )
            _after_artifact_hook(name, path)
        end
        all(path -> isfile(path) && isfile(path * ".manifest.json"), values(paths)) ||
            error("synthetic recovery persistence did not produce every artifact")
        for path in values(paths)
            verify_output_manifest(path;
                package_root,
                require_canonical=run_mode == :canonical,
                verify_source=true,
            )
        end
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return paths
end

"""
    run_synthetic_equation_recovery(; kwargs...)

Run the identifiable minimal-library recovery and the separate full canonical
library false-discovery/collinearity stress test. CSV outputs are written only
under `validation_output_paths().data` (or the supplied compatible path tuple).
"""
function run_synthetic_equation_recovery(;
        seed::Int=DEFAULT_RECOVERY_SEED,
        derivative_noise_std::Real=0.25,
        output_paths=validation_output_paths(),
        persist::Bool=true)
    for field in (:data, :mode, :explicit)
        hasproperty(output_paths, field) || throw(ArgumentError(
            "output_paths is missing required field $field",
        ))
    end
    mode = _provenance_mode(output_paths.mode)
    if mode == :canonical
        output_paths.explicit || throw(ArgumentError(
            "canonical synthetic recovery requires an explicit output root",
        ))
        seed == DEFAULT_RECOVERY_SEED || throw(ArgumentError(
            "canonical synthetic recovery requires seed $DEFAULT_RECOVERY_SEED",
        ))
        Float64(derivative_noise_std) == 0.25 || throw(ArgumentError(
            "canonical synthetic recovery requires derivative_noise_std=0.25",
        ))
    end
    bundle = synthetic_recovery_dataset(
        seed=seed,
        derivative_noise_std=derivative_noise_std,
    )
    minimal_library = build_minimal_library()
    full_library = build_solar_wind_library()
    full_names = get_term_names(full_library)
    if mode == :canonical
        length(full_names) == 20 || error(
            "canonical synthetic stress test requires the 20-term identifiable library",
        )
        "n*V^2" in full_names && error(
            "canonical synthetic stress test must exclude duplicate n*V^2",
        )
    end
    minimal = run_recovery_experiment(
        bundle,
        minimal_library;
        experiment="minimal_identifiable",
        objective="identifiable_recovery",
    )
    full = run_recovery_experiment(
        bundle,
        full_library;
        experiment="full_canonical",
        objective="false_discovery_collinearity_stress",
    )
    # Only the identifiable minimal-library oracle is a correctness gate.
    # The full-library stress outcome is evidence to report, not a required
    # result that the implementation is allowed to manufacture or suppress.
    validation_pass = minimal.summary.experiment_pass

    summary_path = joinpath(output_paths.data, "synthetic_equation_recovery_summary.csv")
    coefficients_path = joinpath(output_paths.data,
                                 "synthetic_equation_recovery_coefficients.csv")
    lambda_path = joinpath(output_paths.data,
                           "synthetic_equation_recovery_lambda_selection.csv")
    trajectories_path = joinpath(
        output_paths.data, "synthetic_equation_recovery_trajectories.csv",
    )
    paths = (; summary=summary_path, coefficients=coefficients_path,
             lambda_selection=lambda_path, trajectories=trajectories_path)
    if persist
        mode == :canonical && !validation_pass && error(
            "canonical minimal-library equation-recovery validation did not pass",
        )
        mkpath(output_paths.data)
        summary = DataFrame([minimal.summary, full.summary])
        summary[!, :overall_validation_pass] = fill(validation_pass, nrow(summary))
        summary[!, :minimal_recovery_validation_pass] =
            fill(validation_pass, nrow(summary))
        summary[!, :full_stress_outcome_neutral] = fill(true, nrow(summary))
        frames = (
            summary=summary,
            coefficients=DataFrame(vcat(
                _coefficient_rows(minimal), _coefficient_rows(full),
            )),
            lambda_selection=DataFrame(vcat(
                _lambda_rows(minimal), _lambda_rows(full),
            )),
            trajectories=DataFrame(vcat(
                _trajectory_rows(bundle, minimal), _trajectory_rows(bundle, full),
            )),
        )
        selection_record = _synthetic_selection_record(
            bundle, minimal, full, validation_pass,
        )
        _persist_synthetic_outputs!(
            paths, frames, selection_record; seed, mode,
        )
    end
    return (; bundle, minimal, full, validation_pass, paths)
end

function main()
    output_paths = validation_output_paths()
    output_paths.explicit || error(
        "set SOLARSINDY_OUTPUT_ROOT to an explicit revision output directory"
    )
    result = run_synthetic_equation_recovery(; output_paths)
    for experiment in (result.minimal, result.full)
        summary = experiment.summary
        println(
            "$(summary.experiment): lambda=$(summary.selected_lambda), " *
            "precision=$(round(summary.support_precision; digits=3)), " *
            "recall=$(round(summary.support_recall; digits=3)), " *
            "decay_relerr=$(round(summary.decay_relative_error; sigdigits=4)), " *
            "injection_relerr=$(round(summary.injection_relative_error; sigdigits=4)), " *
            "forward_rmse_nt=$(round(summary.heldout_forward_rmse_nt; sigdigits=4)), " *
            "recovery_pass=$(summary.recovery_pass), " *
            "experiment_pass=$(summary.experiment_pass)",
        )
    end
    result.validation_pass || error(
        "minimal-library synthetic equation-recovery validation failed",
    )
    return result
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SyntheticEquationRecoveryValidation.main()
end
