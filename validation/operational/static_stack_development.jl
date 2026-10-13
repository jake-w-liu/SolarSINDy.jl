#!/usr/bin/env julia

# Fit and validate the predeclared Operational static stack without reading the
# exposed 2023--2025 benchmark. The source is the audited inertial-SINDy calibration replay;
# only the actual served-tail center is reconstructed here.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA
using Statistics

include(joinpath(@__DIR__, "inertial_sindy_served_holdout.jl"))

const STATIC_STACK_DEVELOPMENT_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "static_stack_development")
const STATIC_STACK_DEVELOPMENT_TABLE = joinpath(STATIC_STACK_DEVELOPMENT_DIR, "static_stack_development_replay.csv")
const STATIC_STACK_DEVELOPMENT_STACK = joinpath(STATIC_STACK_DEVELOPMENT_DIR, "static_stack.csv")
const STATIC_STACK_DEVELOPMENT_SUMMARY = joinpath(STATIC_STACK_DEVELOPMENT_DIR, "static_stack_validation_summary.csv")
const STATIC_STACK_DEVELOPMENT_WEIGHTS = joinpath(STATIC_STACK_DEVELOPMENT_DIR, "static_stack_weight_audit.csv")
const STATIC_STACK_DEVELOPMENT_AUDIT = joinpath(STATIC_STACK_DEVELOPMENT_DIR, "static_stack_development_audit.csv")
const STATIC_STACK_DEVELOPMENT_SOURCE_SHA256 =
    "52405451659e35e7ea2307ce06987fe030407e6fb5fa81044da288249e7aad4a"
const STATIC_STACK_DEVELOPMENT_REPLAY_SHA256 =
    "41f76e4cc7f935aef67a16d526a2a0b3f91bede6608e04aef5050fdeb5888f43"
# Canonical serialization of the frozen stack; the release-era serialization of the same
# artifact digests to 66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d.
const STATIC_STACK_DEVELOPMENT_STACK_SHA256 =
    "06fb5be2f4160d05a586f1da9928e20e278ca5026ca673ffaf50307a85e9db32"

const STATIC_STACK_PRIMARY_LEADS = Int[1, 2, 3, 6]
const STATIC_STACK_DEVELOPMENT_COMPONENTS = (
    static_stack=:static_stack_pred_dst_nt,
    served_inertial_sindy=:served_inertial_sindy_dst_nt,
    frozen_inertial_sindy=:frozen_inertial_sindy_dst_nt,
    raw_sindy=:raw_sindy_dst_nt,
    persistence=:persistence_dst_nt,
    burton=:burton_dst_nt,
    burton_full=:burton_full_dst_nt,
    obrien=:obrien_dst_nt,
)

function _static_stack_source_scored_path()
    path = strip(get(ENV, "SOLARSINDY_INERTIAL_SINDY_CALIBRATION_SCORED", ""))
    isempty(path) && error(
        "Set SOLARSINDY_INERTIAL_SINDY_CALIBRATION_SCORED to the audited inertial-SINDy calibration-scored CSV",
    )
    path = abspath(path)
    isfile(path) && !islink(path) || error(
        "inertial-SINDy calibration source must be a regular non-symlink file: $path",
    )
    return path
end

_static_stack_file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

function _static_stack_required_source_columns(calibration::InertialSindyCalibration)
    columns = Symbol[
        :issue_time_utc, :target_time_utc, :model_step_hours, :operational_split,
        :latest_dst_nt, :observation_dst_nt, :pred_dst_nt, :inertial_sindy_pred_dst_nt,
        :persistence_dst_nt, :burton_dst_nt, :burton_full_dst_nt, :obrien_dst_nt,
        :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa, :dst_delta_1h_nt,
        :coupling_active_mvm,
    ]
    append!(columns, calibration.feature_names)
    return unique(columns)
end

function _static_stack_development_source(path::AbstractString,
                                 calibration::InertialSindyCalibration)
    required = _static_stack_required_source_columns(calibration)
    table = SolarSINDy.canonicalize_operational_frozen_table!(CSV.read(path, DataFrame))
    missing_columns = setdiff(required, Symbol.(names(table)))
    isempty(missing_columns) || error(
        "inertial-SINDy source omits required static stack columns: $(join(missing_columns, ", "))",
    )
    select!(table, String.(required))
    allowed = Set(("fit", "validation", "holdout"))
    all(in(allowed), String.(table.operational_split)) || error(
        "inertial-SINDy source contains an unexpected split label",
    )
    return table
end

function _static_stack_hourly_served_centers(row, core::OperationalCore,
                                    calibration::InertialSindyCalibration,
                                    driver_lookup)
    issue = _inertial_parse_datetime(row.issue_time_utc)
    h = Int(row.model_step_hours)
    drivers = (
        V=Float64(row.V_kms), Bz=Float64(row.Bz_nt), By=Float64(row.By_nt),
        n=Float64(row.n_cm3), Pdyn=Float64(row.Pdyn_npa),
    )
    latest = Float64(row.latest_dst_nt)
    anchor = pressure_correct_dst([latest], [drivers.Pdyn])[1]
    rate = Float64(row.dst_delta_1h_nt)
    future = k -> get(driver_lookup, issue + Hour(k - 1), nothing)
    raw, served = _operational_forecast(
        core.library, core.coefficients, anchor, drivers, future, latest,
        calibration, h, rate; calibration_features=row,
    )
    all(isfinite, (raw, served)) || error(
        "non-finite served reconstruction at issue=$issue, step=$h",
    )
    return raw, served
end

function build_static_stack_development_table(source_path::AbstractString=
                                      _static_stack_source_scored_path())
    core = load_operational_core(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)
    calibration = read_inertial_sindy_calibration(
        operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv,
    )
    calibration.selected_component == :inertial_sindy || error(
        "static stack development requires the deployed corrected-SINDy inertial-SINDy component",
    )
    source = _static_stack_development_source(source_path, calibration)
    lookup = _driver_lookup_range(2010, 2022)
    raw_sindy = Vector{Float64}(undef, nrow(source))
    served = similar(raw_sindy)
    for row in 1:nrow(source)
        raw_sindy[row], served[row] = _static_stack_hourly_served_centers(
            source[row, :], core, calibration, lookup,
        )
    end
    out = DataFrame(
        issue_time_utc=DateTime.(_inertial_parse_datetime.(source.issue_time_utc)),
        target_time_utc=DateTime.(_inertial_parse_datetime.(source.target_time_utc)),
        model_step_hours=Int.(source.model_step_hours),
        split_label=replace.(String.(source.operational_split), "holdout" => "calibration"),
        served_inertial_sindy_dst_nt=served,
        frozen_inertial_sindy_dst_nt=Float64.(source.inertial_sindy_pred_dst_nt),
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
    expected_steps = OPERATIONAL_INERTIAL_SINDY_SUPPORTED_MODEL_STEPS
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

_static_stack_rmse(observed, predicted) = sqrt(mean(abs2, observed .- predicted))

function _static_stack_validation_summary(scored::DataFrame)
    rows = NamedTuple[]
    for lead in STATIC_STACK_PRIMARY_LEADS
        lead_rows = scored[scored.model_step_hours .== lead, :]
        nrow(lead_rows) > 0 || error("validation has no rows at lead $lead")
        observed = Float64.(lead_rows.observation_dst_nt)
        metrics = Dict{Symbol,Float64}()
        for (name, column) in pairs(STATIC_STACK_DEVELOPMENT_COMPONENTS)
            metrics[name] = _static_stack_rmse(observed, Float64.(lead_rows[!, column]))
        end
        references = [name for name in propertynames(STATIC_STACK_DEVELOPMENT_COMPONENTS)
                      if name != :static_stack]
        best_name = first(sort(references; by=name -> metrics[name]))
        best_rmse = metrics[best_name]
        gain = best_rmse - metrics[:static_stack]
        push!(rows, (
            lead_h=lead,
            n_rows=nrow(lead_rows),
            static_stack_rmse_nt=metrics[:static_stack],
            best_comparator=String(best_name),
            best_comparator_rmse_nt=best_rmse,
            gain_vs_best_nt=gain,
            beats_every_comparator=gain > 0.0,
            served_inertial_sindy_rmse_nt=metrics[:served_inertial_sindy],
            frozen_inertial_sindy_rmse_nt=metrics[:frozen_inertial_sindy],
            raw_sindy_rmse_nt=metrics[:raw_sindy],
            persistence_rmse_nt=metrics[:persistence],
            burton_rmse_nt=metrics[:burton],
            burton_full_rmse_nt=metrics[:burton_full],
            obrien_rmse_nt=metrics[:obrien],
        ))
    end
    return DataFrame(rows)
end

function run_static_stack_development(; rebuild::Bool=false)
    mkpath(STATIC_STACK_DEVELOPMENT_DIR)
    source_path = _static_stack_source_scored_path()
    _static_stack_file_sha256(source_path) == STATIC_STACK_DEVELOPMENT_SOURCE_SHA256 || error(
        "audited inertial-SINDy calibration source SHA-256 changed",
    )
    table = if rebuild || !isfile(STATIC_STACK_DEVELOPMENT_TABLE)
        built = build_static_stack_development_table(source_path)
        if isfile(STATIC_STACK_DEVELOPMENT_TABLE)
            # The frozen replay may still be serialized under the release-era column spellings,
            # so byte digests cannot compare across the rename; the rebuilt table must reproduce
            # the frozen content exactly after canonicalization.
            frozen = SolarSINDy.canonicalize_static_stack_frozen_table!(CSV.read(
                STATIC_STACK_DEVELOPMENT_TABLE, DataFrame; types=Dict(
                    :issue_time_utc => DateTime, :target_time_utc => DateTime,
                ),
            ))
            isequal(built, frozen) ||
                error("rebuilt static stack development replay content changed")
        end
        CSV.write(STATIC_STACK_DEVELOPMENT_TABLE, built)
        built
    else
        _static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_TABLE) == STATIC_STACK_DEVELOPMENT_REPLAY_SHA256 ||
            error("cached static stack development replay SHA-256 changed")
        SolarSINDy.canonicalize_static_stack_frozen_table!(CSV.read(
            STATIC_STACK_DEVELOPMENT_TABLE, DataFrame; types=Dict(
                :issue_time_utc => DateTime, :target_time_utc => DateTime,
            ),
        ))
    end
    fit_rows = table[table.split_label .== "fit", :]
    validation_rows = table[table.split_label .== "validation", :]
    nrow(fit_rows) > 0 && nrow(validation_rows) > 0 || error(
        "development fit or validation split is empty",
    )
    maximum(fit_rows.target_time_utc) < minimum(validation_rows.issue_time_utc) ||
        error("fit targets overlap validation issues")
    stack = fit_static_stack(
        fit_rows;
        label="static_stack_primary_sindy60_fit$(nrow(fit_rows))",
    )
    write_static_stack(STATIC_STACK_DEVELOPMENT_STACK, stack)
    _static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_STACK) == STATIC_STACK_DEVELOPMENT_STACK_SHA256 ||
        error("frozen static stack development stack SHA-256 changed")
    scored = score_static_stack(validation_rows, stack)
    summary = _static_stack_validation_summary(scored)
    CSV.write(STATIC_STACK_DEVELOPMENT_SUMMARY, summary)
    weights = DataFrame(
        model_step_hours=[cell.model_step_hours for cell in stack.cells],
        regime=String.([cell.regime for cell in stack.cells]),
        n_rows=[cell.n_rows for cell in stack.cells],
        objective_rmse_nt=sqrt.([cell.objective_mse for cell in stack.cells]),
        iterations=[cell.iterations for cell in stack.cells],
        weight_served_inertial_sindy=[cell.weights[1] for cell in stack.cells],
        weight_frozen_inertial_sindy=[cell.weights[2] for cell in stack.cells],
        weight_persistence=[cell.weights[3] for cell in stack.cells],
        weight_burton=[cell.weights[4] for cell in stack.cells],
        weight_burton_full=[cell.weights[5] for cell in stack.cells],
        weight_obrien=[cell.weights[6] for cell in stack.cells],
    )
    CSV.write(STATIC_STACK_DEVELOPMENT_WEIGHTS, weights)
    audit = DataFrame(
        source_path=[abspath(source_path)],
        source_sha256=[_static_stack_file_sha256(source_path)],
        replay_path=[abspath(STATIC_STACK_DEVELOPMENT_TABLE)],
        replay_sha256=[_static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_TABLE)],
        stack_path=[abspath(STATIC_STACK_DEVELOPMENT_STACK)],
        stack_sha256=[_static_stack_file_sha256(STATIC_STACK_DEVELOPMENT_STACK)],
        fit_rows=[nrow(fit_rows)],
        validation_rows=[nrow(validation_rows)],
        calibration_rows=[count(==("calibration"), table.split_label)],
        validation_beats_every_comparator=[all(summary.beats_every_comparator)],
        exposed_benchmark_rows_read=[0],
    )
    CSV.write(STATIC_STACK_DEVELOPMENT_AUDIT, audit)
    return (; table, stack, scored, summary, audit)
end

if abspath(PROGRAM_FILE) == @__FILE__
    rebuild = "--rebuild" in ARGS
    result = run_static_stack_development(; rebuild)
    println("static stack development complete; validation beats every comparator = ",
            all(result.summary.beats_every_comparator))
end
