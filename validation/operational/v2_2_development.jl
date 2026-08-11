#!/usr/bin/env julia

# Fit and validate the predeclared Operational V2.2 stack without reading the
# exposed 2023--2025 benchmark. The source is the audited V2.1 calibration replay;
# only the actual served-tail center is reconstructed here.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA
using Statistics

include(joinpath(@__DIR__, "v2_1_served_holdout.jl"))

const V22_DEVELOPMENT_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_2_development")
const V22_DEVELOPMENT_TABLE = joinpath(V22_DEVELOPMENT_DIR, "v2_2_development_replay.csv")
const V22_DEVELOPMENT_STACK = joinpath(V22_DEVELOPMENT_DIR, "operational_v2_2_stack.csv")
const V22_DEVELOPMENT_SUMMARY = joinpath(V22_DEVELOPMENT_DIR, "v2_2_validation_summary.csv")
const V22_DEVELOPMENT_WEIGHTS = joinpath(V22_DEVELOPMENT_DIR, "v2_2_weight_audit.csv")
const V22_DEVELOPMENT_AUDIT = joinpath(V22_DEVELOPMENT_DIR, "v2_2_development_audit.csv")
const V22_DEVELOPMENT_SOURCE_SHA256 =
    "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a"
const V22_DEVELOPMENT_REPLAY_SHA256 =
    "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43"
const V22_DEVELOPMENT_STACK_SHA256 =
    "66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d"

const V22_PRIMARY_LEADS = Int[1, 2, 3, 6]
const V22_DEVELOPMENT_COMPONENTS = (
    v2_2=:v2_2_pred_dst_nt,
    served_v2_1=:served_v2_1_dst_nt,
    frozen_v2_1=:frozen_v2_1_dst_nt,
    raw_sindy=:raw_sindy_dst_nt,
    persistence=:persistence_dst_nt,
    burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,
    obrien=:obrien_dst_nt,
)

function _v22_source_scored_path()
    path = strip(get(ENV, "SOLARSINDY_V21_CALIBRATION_SCORED", ""))
    isempty(path) && error(
        "Set SOLARSINDY_V21_CALIBRATION_SCORED to the audited V2.1 calibration-scored CSV",
    )
    path = abspath(path)
    isfile(path) && !islink(path) || error(
        "V2.1 calibration source must be a regular non-symlink file: $path",
    )
    return path
end

_v22_file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _v22_required_source_columns(calibration::OperationalV2Calibration)
    columns = Symbol[
        :issue_time_utc, :target_time_utc, :model_step_hours, :v2_split,
        :latest_dst_nt, :observation_dst_nt, :pred_dst_nt, :v2_pred_dst_nt,
        :persistence_dst_nt, :burton_dst_nt, :burton_full_dst_nt, :obrien_dst_nt,
        :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa, :dst_delta_1h_nt,
        :coupling_active_mvm,
    ]
    append!(columns, calibration.feature_names)
    return unique(columns)
end

function _v22_development_source(path::AbstractString,
                                 calibration::OperationalV2Calibration)
    required = _v22_required_source_columns(calibration)
    table = CSV.read(path, DataFrame; select=String.(required))
    missing_columns = setdiff(required, Symbol.(names(table)))
    isempty(missing_columns) || error(
        "V2.1 source omits required V2.2 columns: $(join(missing_columns, ", "))",
    )
    allowed = Set(("fit", "validation", "holdout"))
    all(in(allowed), String.(table.v2_split)) || error(
        "V2.1 source contains an unexpected split label",
    )
    return table
end

function _v22_hourly_served_centers(row, core::OperationalCore,
                                    calibration::OperationalV2Calibration,
                                    driver_lookup)
    issue = _v21_parse_datetime(row.issue_time_utc)
    h = Int(row.model_step_hours)
    drivers = (
        V=Float64(row.V_kms), Bz=Float64(row.Bz_nt), By=Float64(row.By_nt),
        n=Float64(row.n_cm3), Pdyn=Float64(row.Pdyn_npa),
    )
    latest = Float64(row.latest_dst_nt)
    anchor = pressure_correct_dst([latest], [drivers.Pdyn])[1]
    rate = Float64(row.dst_delta_1h_nt)
    future = k -> get(driver_lookup, issue + Hour(k - 1), nothing)
    raw, served = _v2_forecast(
        core.library, core.coefficients, anchor, drivers, future, latest,
        calibration, h, rate; calibration_features=row,
    )
    all(isfinite, (raw, served)) || error(
        "non-finite served reconstruction at issue=$issue, step=$h",
    )
    return raw, served
end

function build_v2_2_development_table(source_path::AbstractString=
                                      _v22_source_scored_path())
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    calibration = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
    calibration.selected_component == :v2 || error(
        "V2.2 development requires the deployed corrected-SINDy V2.1 component",
    )
    source = _v22_development_source(source_path, calibration)
    lookup = _driver_lookup_range(2010, 2022)
    raw_sindy = Vector{Float64}(undef, nrow(source))
    served = similar(raw_sindy)
    for row in 1:nrow(source)
        raw_sindy[row], served[row] = _v22_hourly_served_centers(
            source[row, :], core, calibration, lookup,
        )
    end
    out = DataFrame(
        issue_time_utc=DateTime.(_v21_parse_datetime.(source.issue_time_utc)),
        target_time_utc=DateTime.(_v21_parse_datetime.(source.target_time_utc)),
        model_step_hours=Int.(source.model_step_hours),
        split_label=replace.(String.(source.v2_split), "holdout" => "calibration"),
        served_v2_1_dst_nt=served,
        frozen_v2_1_dst_nt=Float64.(source.v2_pred_dst_nt),
        raw_sindy_dst_nt=raw_sindy,
        persistence_dst_nt=Float64.(source.persistence_dst_nt),
        burton_dst_nt=Float64.(source.burton_dst_nt),
        burton_full_dst_nt=Float64.(source.burton_full_dst_nt),
        obrien_dst_nt=Float64.(source.obrien_dst_nt),
        observation_dst_nt=Float64.(source.observation_dst_nt),
        latest_dst_nt=Float64.(source.latest_dst_nt),
        dst_delta_1h_nt=Float64.(source.dst_delta_1h_nt),
        coupling_active_mvm=Float64.(source.coupling_active_mvm),
    )
    expected_steps = OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS
    sort!(unique(out.model_step_hours)) == expected_steps || error(
        "development table does not cover every supported model step",
    )
    all(out.target_time_utc .== out.issue_time_utc .+ Hour.(out.model_step_hours)) ||
        error("development target-time invariant failed")
    keys = Tuple.(eachrow(select(
        out, :issue_time_utc, :target_time_utc, :model_step_hours,
    )))
    length(unique(keys)) == nrow(out) || error("development table contains duplicate keys")
    numeric = names(out, Real)
    all(column -> all(isfinite, out[!, column]), numeric) || error(
        "development table contains a non-finite numeric value",
    )
    return out
end

_v22_rmse(observed, predicted) = sqrt(mean(abs2, observed .- predicted))

function _v22_validation_summary(scored::DataFrame)
    rows = NamedTuple[]
    for lead in V22_PRIMARY_LEADS
        lead_rows = scored[scored.model_step_hours .== lead, :]
        nrow(lead_rows) > 0 || error("validation has no rows at lead $lead")
        observed = Float64.(lead_rows.observation_dst_nt)
        metrics = Dict{Symbol,Float64}()
        for (name, column) in pairs(V22_DEVELOPMENT_COMPONENTS)
            metrics[name] = _v22_rmse(observed, Float64.(lead_rows[!, column]))
        end
        references = [name for name in propertynames(V22_DEVELOPMENT_COMPONENTS)
                      if name != :v2_2]
        best_name = first(sort(references; by=name -> metrics[name]))
        best_rmse = metrics[best_name]
        gain = best_rmse - metrics[:v2_2]
        push!(rows, (
            lead_h=lead,
            n_rows=nrow(lead_rows),
            v2_2_rmse_nt=metrics[:v2_2],
            best_comparator=String(best_name),
            best_comparator_rmse_nt=best_rmse,
            gain_vs_best_nt=gain,
            beats_every_comparator=gain > 0.0,
            served_v2_1_rmse_nt=metrics[:served_v2_1],
            frozen_v2_1_rmse_nt=metrics[:frozen_v2_1],
            raw_sindy_rmse_nt=metrics[:raw_sindy],
            persistence_rmse_nt=metrics[:persistence],
            burton_rmse_nt=metrics[:burton],
            burton_full_rmse_nt=metrics[:burton_full],
            obrien_rmse_nt=metrics[:obrien],
        ))
    end
    return DataFrame(rows)
end

function run_v2_2_development(; rebuild::Bool=false)
    mkpath(V22_DEVELOPMENT_DIR)
    source_path = _v22_source_scored_path()
    _v22_file_sha256(source_path) == V22_DEVELOPMENT_SOURCE_SHA256 || error(
        "audited V2.1 calibration source SHA-256 changed",
    )
    table = if rebuild || !isfile(V22_DEVELOPMENT_TABLE)
        built = build_v2_2_development_table(source_path)
        CSV.write(V22_DEVELOPMENT_TABLE, built)
        _v22_file_sha256(V22_DEVELOPMENT_TABLE) == V22_DEVELOPMENT_REPLAY_SHA256 ||
            error("rebuilt V2.2 development replay SHA-256 changed")
        built
    else
        _v22_file_sha256(V22_DEVELOPMENT_TABLE) == V22_DEVELOPMENT_REPLAY_SHA256 ||
            error("cached V2.2 development replay SHA-256 changed")
        CSV.read(V22_DEVELOPMENT_TABLE, DataFrame; types=Dict(
            :issue_time_utc => DateTime, :target_time_utc => DateTime,
        ))
    end
    fit_rows = table[table.split_label .== "fit", :]
    validation_rows = table[table.split_label .== "validation", :]
    nrow(fit_rows) > 0 && nrow(validation_rows) > 0 || error(
        "development fit or validation split is empty",
    )
    maximum(fit_rows.target_time_utc) < minimum(validation_rows.issue_time_utc) ||
        error("fit targets overlap validation issues")
    stack = fit_operational_v22_stack(
        fit_rows;
        label="operational_v2_2_primary_sindy60_fit$(nrow(fit_rows))",
    )
    write_operational_v22_stack(V22_DEVELOPMENT_STACK, stack)
    _v22_file_sha256(V22_DEVELOPMENT_STACK) == V22_DEVELOPMENT_STACK_SHA256 ||
        error("frozen V2.2 development stack SHA-256 changed")
    scored = score_operational_v22(validation_rows, stack)
    summary = _v22_validation_summary(scored)
    CSV.write(V22_DEVELOPMENT_SUMMARY, summary)
    weights = DataFrame(
        model_step_hours=[cell.model_step_hours for cell in stack.cells],
        regime=String.([cell.regime for cell in stack.cells]),
        n_rows=[cell.n_rows for cell in stack.cells],
        objective_rmse_nt=sqrt.([cell.objective_mse for cell in stack.cells]),
        iterations=[cell.iterations for cell in stack.cells],
        weight_served_v2_1=[cell.weights[1] for cell in stack.cells],
        weight_frozen_v2_1=[cell.weights[2] for cell in stack.cells],
        weight_persistence=[cell.weights[3] for cell in stack.cells],
        weight_burton=[cell.weights[4] for cell in stack.cells],
        weight_burton_full=[cell.weights[5] for cell in stack.cells],
        weight_obrien=[cell.weights[6] for cell in stack.cells],
    )
    CSV.write(V22_DEVELOPMENT_WEIGHTS, weights)
    audit = DataFrame(
        source_path=[abspath(source_path)],
        source_sha256=[_v22_file_sha256(source_path)],
        replay_path=[abspath(V22_DEVELOPMENT_TABLE)],
        replay_sha256=[_v22_file_sha256(V22_DEVELOPMENT_TABLE)],
        stack_path=[abspath(V22_DEVELOPMENT_STACK)],
        stack_sha256=[_v22_file_sha256(V22_DEVELOPMENT_STACK)],
        fit_rows=[nrow(fit_rows)],
        validation_rows=[nrow(validation_rows)],
        calibration_rows=[count(==("calibration"), table.split_label)],
        validation_beats_every_comparator=[all(summary.beats_every_comparator)],
        exposed_benchmark_rows_read=[0],
    )
    CSV.write(V22_DEVELOPMENT_AUDIT, audit)
    return (; table, stack, scored, summary, audit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rebuild = "--rebuild" in ARGS
    result = run_v2_2_development(; rebuild)
    println("V2.2 development complete; validation beats every comparator = ",
            all(result.summary.beats_every_comparator))
end
