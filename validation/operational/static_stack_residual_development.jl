#!/usr/bin/env julia

# Evaluate the single predeclared static stack secondary candidate on development rows.
# The exposed 2023--2025 benchmark is neither located nor read by this script.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA
using Statistics

include(joinpath(@__DIR__, "static_stack_development.jl"))

const STATIC_STACK_RESIDUAL_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "static_stack_residual_development")
const STATIC_STACK_RESIDUAL_TABLE = joinpath(STATIC_STACK_RESIDUAL_DIR, "static_stack_residual_replay.csv")
const STATIC_STACK_RESIDUAL_CORE = joinpath(STATIC_STACK_RESIDUAL_DIR, "static_stack_residual.csv")
const STATIC_STACK_RESIDUAL_SUMMARY = joinpath(STATIC_STACK_RESIDUAL_DIR, "static_stack_residual_validation_summary.csv")
const STATIC_STACK_RESIDUAL_LEAD_AUDIT = joinpath(STATIC_STACK_RESIDUAL_DIR, "static_stack_residual_lead_audit.csv")
const STATIC_STACK_RESIDUAL_AUDIT = joinpath(STATIC_STACK_RESIDUAL_DIR, "static_stack_residual_development_audit.csv")

# The frozen primary stack may exist on disk under either its release-era serialization or its
# canonical serialization; both digests identify the same pinned artifact.
const STATIC_STACK_RESIDUAL_PRIMARY_STACK_SHA256 = (
    "66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d",
    "06fb5be2f4160d05a586f1da9928e20e278ca5026ca673ffaf50307a85e9db32",
)
const STATIC_STACK_RESIDUAL_SOURCE_SHA256 =
    "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a"
const STATIC_STACK_RESIDUAL_PRIMARY_REPLAY_SHA256 =
    "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43"
const STATIC_STACK_RESIDUAL_REPLAY_SHA256 =
    "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f"
const STATIC_STACK_RESIDUAL_SOURCE_FEATURES = (
    :latest_dst_nt,
    :dst_delta_1h_nt,
    :dst_delta_3h_nt,
    :dst_delta_6h_nt,
    :Bz_nt,
    :Bz_delta_1h_nt,
    :VBsouth_mvm,
    :VBsouth_delta_1h_mvm,
    :VBsouth_mean_3h_mvm,
    :VBsouth_mean_6h_mvm,
    :sqrt_Pdyn_npa,
    :main_phase_pressure_nt,
    :main_phase_pressure_6h_nt,
    :recovery_pressure_nt,
    :main_phase_recovery_pressure,
    :baseline_spread_nt,
)

function _static_stack_residual_source(path::AbstractString)
    columns = (
        :issue_time_utc, :target_time_utc, :model_step_hours,
        STATIC_STACK_RESIDUAL_SOURCE_FEATURES...,
    )
    source = SolarSINDy.canonicalize_operational_frozen_table!(CSV.read(path, DataFrame))
    select!(source, collect(String.(columns)))
    Set(Symbol.(names(source))) == Set(columns) || error(
        "static stack residual source schema does not match the predeclared selection",
    )
    source.issue_time_utc = DateTime.(_inertial_parse_datetime.(source.issue_time_utc))
    source.target_time_utc = DateTime.(_inertial_parse_datetime.(source.target_time_utc))
    source.model_step_hours = Int.(source.model_step_hours)
    return source
end

function _static_stack_residual_assert_same_keys(base::DataFrame, source::DataFrame)
    nrow(base) == nrow(source) || error("static stack residual source row count changed")
    for column in (:issue_time_utc, :target_time_utc, :model_step_hours)
        all(base[!, column] .== source[!, column]) || error(
            "static stack residual source key/order mismatch in $column",
        )
    end
    return nothing
end

function _static_stack_residual_add_features!(table::DataFrame, source::DataFrame)
    for column in STATIC_STACK_RESIDUAL_SOURCE_FEATURES
        values = Float64.(source[!, column])
        all(isfinite, values) || error("non-finite residual source feature $column")
        if String(column) in names(table)
            all(Float64.(table[!, column]) .== values) || error(
                "static stack replay/source mismatch in $column",
            )
        else
            table[!, column] = values
        end
    end
    table[!, :served_minus_frozen_inertial_sindy_nt] =
        table.served_inertial_sindy_dst_nt .- table.frozen_inertial_sindy_dst_nt
    table[!, :primary_minus_served_inertial_sindy_nt] =
        table.static_stack_pred_dst_nt .- table.served_inertial_sindy_dst_nt
    table[!, :primary_minus_frozen_inertial_sindy_nt] =
        table.static_stack_pred_dst_nt .- table.frozen_inertial_sindy_dst_nt
    table[!, :primary_minus_persistence_nt] =
        table.static_stack_pred_dst_nt .- table.persistence_dst_nt
    table[!, :primary_minus_burton_full_nt] =
        table.static_stack_pred_dst_nt .- table.burton_full_dst_nt
    table[!, :primary_minus_obrien_nt] =
        table.static_stack_pred_dst_nt .- table.obrien_dst_nt
    all(feature -> String(feature) in names(table), STATIC_STACK_RESIDUAL_FEATURES) ||
        error("static stack residual table omits a predeclared feature")
    all(feature -> all(isfinite, Float64.(table[!, feature])),
        STATIC_STACK_RESIDUAL_FEATURES) || error(
        "static stack residual table contains a non-finite predeclared feature",
    )
    return table
end

function build_static_stack_residual_table(source_path::AbstractString=
                                   _static_stack_source_scored_path())
    isfile(STATIC_STACK_DEVELOPMENT_TABLE) || error(
        "Run static_stack_development.jl before the residual development script",
    )
    _static_stack_file_sha256(source_path) == STATIC_STACK_RESIDUAL_SOURCE_SHA256 ||
        error("audited inertial-SINDy residual source SHA-256 changed")
    _static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_TABLE) == STATIC_STACK_RESIDUAL_PRIMARY_REPLAY_SHA256 ||
        error("frozen static stack primary replay SHA-256 changed")
    _static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_STACK) in
        STATIC_STACK_RESIDUAL_PRIMARY_STACK_SHA256 ||
        error("the frozen static stack primary-stack SHA-256 changed")
    base = SolarSINDy.canonicalize_static_stack_frozen_table!(CSV.read(
        STATIC_STACK_DEVELOPMENT_TABLE, DataFrame; types=Dict(
            :issue_time_utc => DateTime, :target_time_utc => DateTime,
        ),
    ))
    stack = read_static_stack(STATIC_STACK_DEVELOPMENT_STACK)
    table = score_static_stack(base, stack)
    source = _static_stack_residual_source(source_path)
    _static_stack_residual_assert_same_keys(table, source)
    _static_stack_residual_add_features!(table, source)
    return table
end

function _static_stack_residual_summary(scored::DataFrame)
    rows = NamedTuple[]
    for lead in STATIC_STACK_PRIMARY_LEADS
        lead_rows = scored[scored.model_step_hours .== lead, :]
        observed = Float64.(lead_rows.observation_dst_nt)
        primary = _static_stack_rmse(observed, Float64.(lead_rows.static_stack_pred_dst_nt))
        secondary = _static_stack_rmse(
            observed, Float64.(lead_rows.static_stack_secondary_pred_dst_nt),
        )
        comparator_metrics = Dict(
            name => _static_stack_rmse(observed, Float64.(lead_rows[!, column]))
            for (name, column) in pairs(STATIC_STACK_DEVELOPMENT_COMPONENTS)
            if name != :static_stack
        )
        best_name = first(sort!(collect(keys(comparator_metrics));
                               by=name -> comparator_metrics[name]))
        best_rmse = comparator_metrics[best_name]
        push!(rows, (
            lead_h=lead,
            n_rows=nrow(lead_rows),
            primary_rmse_nt=primary,
            secondary_rmse_nt=secondary,
            gain_vs_primary_nt=primary - secondary,
            best_comparator=String(best_name),
            best_comparator_rmse_nt=best_rmse,
            gain_vs_best_nt=best_rmse - secondary,
            beats_every_comparator=secondary < best_rmse,
            passes_general_effect_gate=(best_rmse - secondary) >=
                max(0.25, 0.02 * best_rmse),
        ))
    end
    return DataFrame(rows)
end

function _static_stack_residual_lead_audit(fit_rows::DataFrame,
                                  validation_rows::DataFrame)
    rows = NamedTuple[]
    cells = StaticStackResidualCell[]
    for lead in sort!(unique(Int.(fit_rows.model_step_hours)))
        fit_lead = fit_rows[fit_rows.model_step_hours .== lead, :]
        validation_lead = validation_rows[validation_rows.model_step_hours .== lead, :]
        accepted = false
        reason = ""
        cell = nothing
        try
            lead_core = fit_static_stack_residual(fit_lead, validation_lead)
            cell = only(lead_core.cells)
            push!(cells, cell)
            accepted = true
        catch err
            err isa InterruptException && rethrow()
            reason = sprint(showerror, err)
        end
        push!(rows, (
            model_step_hours=lead,
            fit_rows=nrow(fit_lead),
            validation_rows=nrow(validation_lead),
            accepted=accepted,
            rejection_reason=reason,
            ridge=accepted ? cell.ridge : missing,
            top_k=accepted ? cell.top_k : missing,
            selected_features=accepted ? join(String.(cell.feature_names), ";") : missing,
            validation_base_rmse_nt=accepted ? cell.validation_base_rmse_nt : missing,
            validation_rmse_nt=accepted ? cell.validation_rmse_nt : missing,
            validation_active_base_rmse_nt=
                accepted ? cell.validation_active_base_rmse_nt : missing,
            validation_active_rmse_nt=accepted ? cell.validation_active_rmse_nt : missing,
            validation_recovery_base_rmse_nt=
                accepted ? cell.validation_recovery_base_rmse_nt : missing,
            validation_recovery_rmse_nt=
                accepted ? cell.validation_recovery_rmse_nt : missing,
        ))
    end
    return DataFrame(rows), cells
end

function run_static_stack_residual_development(; rebuild::Bool=false)
    mkpath(STATIC_STACK_RESIDUAL_DIR)
    source_path = _static_stack_source_scored_path()
    _static_stack_file_sha256(source_path) == STATIC_STACK_RESIDUAL_SOURCE_SHA256 ||
        error("audited inertial-SINDy residual source SHA-256 changed")
    table = if rebuild || !isfile(STATIC_STACK_RESIDUAL_TABLE)
        built = build_static_stack_residual_table(source_path)
        if isfile(STATIC_STACK_RESIDUAL_TABLE)
            # The frozen replay may still be serialized under the release-era column spellings,
            # so byte digests cannot compare across the rename; the rebuilt table must reproduce
            # the frozen content exactly after canonicalization.
            frozen = SolarSINDy.canonicalize_static_stack_frozen_table!(CSV.read(
                STATIC_STACK_RESIDUAL_TABLE, DataFrame; types=Dict(
                    :issue_time_utc => DateTime, :target_time_utc => DateTime,
                ),
            ))
            isequal(built, frozen) ||
                error("rebuilt static stack residual replay content changed")
        end
        CSV.write(STATIC_STACK_RESIDUAL_TABLE, built)
        built
    else
        _static_stack_file_sha256(STATIC_STACK_RESIDUAL_TABLE) == STATIC_STACK_RESIDUAL_REPLAY_SHA256 ||
            error("cached static stack residual replay SHA-256 changed")
        loaded = SolarSINDy.canonicalize_static_stack_frozen_table!(CSV.read(
            STATIC_STACK_RESIDUAL_TABLE, DataFrame; types=Dict(
                :issue_time_utc => DateTime, :target_time_utc => DateTime,
            ),
        ))
        _static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_STACK) in
            STATIC_STACK_RESIDUAL_PRIMARY_STACK_SHA256 ||
            error("the frozen static stack primary-stack SHA-256 changed")
        loaded
    end
    fit_rows = table[table.split_label .== "fit", :]
    validation_rows = table[table.split_label .== "validation", :]
    maximum(fit_rows.target_time_utc) < minimum(validation_rows.issue_time_utc) ||
        error("residual fit targets overlap validation issues")

    lead_audit, cells = _static_stack_residual_lead_audit(fit_rows, validation_rows)
    CSV.write(STATIC_STACK_RESIDUAL_LEAD_AUDIT, lead_audit)
    accepted = nrow(lead_audit) == length(cells) && all(lead_audit.accepted)
    summary = DataFrame()
    residual_sha = ""
    if accepted
        core = StaticStackResidualCore(cells)
        write_static_stack_residual(STATIC_STACK_RESIDUAL_CORE, core)
        restored = read_static_stack_residual(STATIC_STACK_RESIDUAL_CORE)
        scored = score_static_stack_residual(validation_rows, restored)
        summary = _static_stack_residual_summary(scored)
        CSV.write(STATIC_STACK_RESIDUAL_SUMMARY, summary)
        residual_sha = _static_stack_file_sha256(STATIC_STACK_RESIDUAL_CORE)
    elseif isfile(STATIC_STACK_RESIDUAL_CORE)
        rm(STATIC_STACK_RESIDUAL_CORE)
    end
    audit = DataFrame(
        source_path=[abspath(source_path)],
        source_sha256=[_static_stack_file_sha256(source_path)],
        replay_path=[abspath(STATIC_STACK_RESIDUAL_TABLE)],
        replay_sha256=[_static_stack_file_sha256(STATIC_STACK_RESIDUAL_TABLE)],
        primary_stack_path=[abspath(STATIC_STACK_DEVELOPMENT_STACK)],
        primary_stack_sha256=[_static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_STACK)],
        secondary_accepted=[accepted],
        residual_core_path=[accepted ? abspath(STATIC_STACK_RESIDUAL_CORE) : ""],
        residual_core_sha256=[residual_sha],
        fit_rows=[nrow(fit_rows)],
        validation_rows=[nrow(validation_rows)],
        calibration_rows=[count(==("calibration"), table.split_label)],
        exposed_benchmark_rows_read=[0],
    )
    CSV.write(STATIC_STACK_RESIDUAL_AUDIT, audit)
    return (; accepted, table, lead_audit, summary, audit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_stack_residual_development(; rebuild="--rebuild" in ARGS)
    println("static stack secondary accepted = ", result.accepted)
end
