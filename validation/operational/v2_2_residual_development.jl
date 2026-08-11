#!/usr/bin/env julia

# Evaluate the single predeclared V2.2 secondary candidate on development rows.
# The exposed 2023--2025 benchmark is neither located nor read by this script.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA
using Statistics

include(joinpath(@__DIR__, "v2_2_development.jl"))

const V22_RESIDUAL_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_2_residual_development")
const V22_RESIDUAL_TABLE = joinpath(V22_RESIDUAL_DIR, "v2_2_residual_replay.csv")
const V22_RESIDUAL_CORE = joinpath(V22_RESIDUAL_DIR, "operational_v2_2_residual.csv")
const V22_RESIDUAL_SUMMARY = joinpath(V22_RESIDUAL_DIR, "v2_2_residual_validation_summary.csv")
const V22_RESIDUAL_LEAD_AUDIT = joinpath(V22_RESIDUAL_DIR, "v2_2_residual_lead_audit.csv")
const V22_RESIDUAL_AUDIT = joinpath(V22_RESIDUAL_DIR, "v2_2_residual_development_audit.csv")

const V22_RESIDUAL_PRIMARY_STACK_SHA256 =
    "66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d"
const V22_RESIDUAL_SOURCE_SHA256 =
    "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a"
const V22_RESIDUAL_PRIMARY_REPLAY_SHA256 =
    "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43"
const V22_RESIDUAL_REPLAY_SHA256 =
    "0af14a736871d17583563a2f0d994abe6cfd6a8193497fff13353d923b39ed5f"
const V22_RESIDUAL_SOURCE_FEATURES = (
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

function _v22_residual_source(path::AbstractString)
    columns = (
        :issue_time_utc, :target_time_utc, :model_step_hours,
        V22_RESIDUAL_SOURCE_FEATURES...,
    )
    source = CSV.read(path, DataFrame; select=collect(String.(columns)))
    Set(Symbol.(names(source))) == Set(columns) || error(
        "V2.2 residual source schema does not match the predeclared selection",
    )
    source.issue_time_utc = DateTime.(_v21_parse_datetime.(source.issue_time_utc))
    source.target_time_utc = DateTime.(_v21_parse_datetime.(source.target_time_utc))
    source.model_step_hours = Int.(source.model_step_hours)
    return source
end

function _v22_residual_assert_same_keys(base::DataFrame, source::DataFrame)
    nrow(base) == nrow(source) || error("V2.2 residual source row count changed")
    for column in (:issue_time_utc, :target_time_utc, :model_step_hours)
        all(base[!, column] .== source[!, column]) || error(
            "V2.2 residual source key/order mismatch in $column",
        )
    end
    return nothing
end

function _v22_residual_add_features!(table::DataFrame, source::DataFrame)
    for column in V22_RESIDUAL_SOURCE_FEATURES
        values = Float64.(source[!, column])
        all(isfinite, values) || error("non-finite residual source feature $column")
        if String(column) in names(table)
            all(Float64.(table[!, column]) .== values) || error(
                "V2.2 replay/source mismatch in $column",
            )
        else
            table[!, column] = values
        end
    end
    table[!, :served_minus_frozen_v2_1_nt] =
        table.served_v2_1_dst_nt .- table.frozen_v2_1_dst_nt
    table[!, :primary_minus_served_v2_1_nt] =
        table.v2_2_pred_dst_nt .- table.served_v2_1_dst_nt
    table[!, :primary_minus_frozen_v2_1_nt] =
        table.v2_2_pred_dst_nt .- table.frozen_v2_1_dst_nt
    table[!, :primary_minus_persistence_nt] =
        table.v2_2_pred_dst_nt .- table.persistence_dst_nt
    table[!, :primary_minus_burton_full_nt] =
        table.v2_2_pred_dst_nt .- table.burton_full_dst_nt
    table[!, :primary_minus_obrien_nt] =
        table.v2_2_pred_dst_nt .- table.obrien_dst_nt
    all(feature -> String(feature) in names(table), OPERATIONAL_V22_RESIDUAL_FEATURES) ||
        error("V2.2 residual table omits a predeclared feature")
    all(feature -> all(isfinite, Float64.(table[!, feature])),
        OPERATIONAL_V22_RESIDUAL_FEATURES) || error(
        "V2.2 residual table contains a non-finite predeclared feature",
    )
    return table
end

function build_v2_2_residual_table(source_path::AbstractString=
                                   _v22_source_scored_path())
    isfile(V22_DEVELOPMENT_TABLE) || error(
        "Run v2_2_development.jl before the residual development script",
    )
    _v22_file_sha256(source_path) == V22_RESIDUAL_SOURCE_SHA256 ||
        error("audited V2.1 residual source SHA-256 changed")
    _v22_file_sha256(V22_DEVELOPMENT_TABLE) == V22_RESIDUAL_PRIMARY_REPLAY_SHA256 ||
        error("frozen V2.2 primary replay SHA-256 changed")
    _v22_file_sha256(V22_DEVELOPMENT_STACK) == V22_RESIDUAL_PRIMARY_STACK_SHA256 ||
        error("the frozen V2.2 primary-stack SHA-256 changed")
    base = CSV.read(V22_DEVELOPMENT_TABLE, DataFrame; types=Dict(
        :issue_time_utc => DateTime, :target_time_utc => DateTime,
    ))
    stack = read_operational_v22_stack(V22_DEVELOPMENT_STACK)
    table = score_operational_v22(base, stack)
    source = _v22_residual_source(source_path)
    _v22_residual_assert_same_keys(table, source)
    _v22_residual_add_features!(table, source)
    return table
end

function _v22_residual_summary(scored::DataFrame)
    rows = NamedTuple[]
    for lead in V22_PRIMARY_LEADS
        lead_rows = scored[scored.model_step_hours .== lead, :]
        observed = Float64.(lead_rows.observation_dst_nt)
        primary = _v22_rmse(observed, Float64.(lead_rows.v2_2_pred_dst_nt))
        secondary = _v22_rmse(
            observed, Float64.(lead_rows.v2_2_secondary_pred_dst_nt),
        )
        comparator_metrics = Dict(
            name => _v22_rmse(observed, Float64.(lead_rows[!, column]))
            for (name, column) in pairs(V22_DEVELOPMENT_COMPONENTS)
            if name != :v2_2
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

function _v22_residual_lead_audit(fit_rows::DataFrame,
                                  validation_rows::DataFrame)
    rows = NamedTuple[]
    cells = OperationalV22ResidualCell[]
    for lead in sort!(unique(Int.(fit_rows.model_step_hours)))
        fit_lead = fit_rows[fit_rows.model_step_hours .== lead, :]
        validation_lead = validation_rows[validation_rows.model_step_hours .== lead, :]
        accepted = false
        reason = ""
        cell = nothing
        try
            lead_core = fit_operational_v22_residual(fit_lead, validation_lead)
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

function run_v2_2_residual_development(; rebuild::Bool=false)
    mkpath(V22_RESIDUAL_DIR)
    source_path = _v22_source_scored_path()
    _v22_file_sha256(source_path) == V22_RESIDUAL_SOURCE_SHA256 ||
        error("audited V2.1 residual source SHA-256 changed")
    table = if rebuild || !isfile(V22_RESIDUAL_TABLE)
        built = build_v2_2_residual_table(source_path)
        CSV.write(V22_RESIDUAL_TABLE, built)
        _v22_file_sha256(V22_RESIDUAL_TABLE) == V22_RESIDUAL_REPLAY_SHA256 ||
            error("rebuilt V2.2 residual replay SHA-256 changed")
        built
    else
        _v22_file_sha256(V22_RESIDUAL_TABLE) == V22_RESIDUAL_REPLAY_SHA256 ||
            error("cached V2.2 residual replay SHA-256 changed")
        loaded = CSV.read(V22_RESIDUAL_TABLE, DataFrame; types=Dict(
            :issue_time_utc => DateTime, :target_time_utc => DateTime,
        ))
        _v22_file_sha256(V22_DEVELOPMENT_STACK) == V22_RESIDUAL_PRIMARY_STACK_SHA256 ||
            error("the frozen V2.2 primary-stack SHA-256 changed")
        loaded
    end
    fit_rows = table[table.split_label .== "fit", :]
    validation_rows = table[table.split_label .== "validation", :]
    maximum(fit_rows.target_time_utc) < minimum(validation_rows.issue_time_utc) ||
        error("residual fit targets overlap validation issues")

    lead_audit, cells = _v22_residual_lead_audit(fit_rows, validation_rows)
    CSV.write(V22_RESIDUAL_LEAD_AUDIT, lead_audit)
    accepted = nrow(lead_audit) == length(cells) && all(lead_audit.accepted)
    summary = DataFrame()
    residual_sha = ""
    if accepted
        core = OperationalV22ResidualCore(cells)
        write_operational_v22_residual(V22_RESIDUAL_CORE, core)
        restored = read_operational_v22_residual(V22_RESIDUAL_CORE)
        scored = score_operational_v22_residual(validation_rows, restored)
        summary = _v22_residual_summary(scored)
        CSV.write(V22_RESIDUAL_SUMMARY, summary)
        residual_sha = _v22_file_sha256(V22_RESIDUAL_CORE)
    elseif isfile(V22_RESIDUAL_CORE)
        rm(V22_RESIDUAL_CORE)
    end
    audit = DataFrame(
        source_path=[abspath(source_path)],
        source_sha256=[_v22_file_sha256(source_path)],
        replay_path=[abspath(V22_RESIDUAL_TABLE)],
        replay_sha256=[_v22_file_sha256(V22_RESIDUAL_TABLE)],
        primary_stack_path=[abspath(V22_DEVELOPMENT_STACK)],
        primary_stack_sha256=[_v22_file_sha256(V22_DEVELOPMENT_STACK)],
        secondary_accepted=[accepted],
        residual_core_path=[accepted ? abspath(V22_RESIDUAL_CORE) : ""],
        residual_core_sha256=[residual_sha],
        fit_rows=[nrow(fit_rows)],
        validation_rows=[nrow(validation_rows)],
        calibration_rows=[count(==("calibration"), table.split_label)],
        exposed_benchmark_rows_read=[0],
    )
    CSV.write(V22_RESIDUAL_AUDIT, audit)
    return (; accepted, table, lead_audit, summary, audit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_v2_2_residual_development(; rebuild="--rebuild" in ARGS)
    println("V2.2 secondary accepted = ", result.accepted)
end
