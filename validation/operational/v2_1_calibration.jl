#!/usr/bin/env julia

# Build and fit the Operational V2.1 point/conformal calibration on a
# chronological pre-evaluation archive. Point forecasts use the validated 20/11
# core and a deterministic point-only rollout that is independently pinned to
# forecast_ahead by package tests. The canonical deploy files are replaced only
# after the unchanged selection and held-out coverage gates pass.

using SolarSINDy
using CSV
using DataFrames
using Dates
using SHA
using Statistics

include(joinpath(@__DIR__, "paths.jl"))
include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))

const V21_CALIBRATION_YEAR_START = 2010
const V21_CALIBRATION_YEAR_END = 2022
const V21_CALIBRATION_HORIZONS = Int[1, 2, 3, 6]
const V21_CALIBRATION_TRAIN_FRACTION = 0.60
const V21_CALIBRATION_VALIDATION_FRACTION = 0.20
const V21_CALIBRATION_COVERAGE = 0.90
const V21_CALIBRATION_COVERAGE_FLOOR = 0.85
const V21_SEED_HALF_WIDTH_NT = 1.0

const V21_CALIBRATION_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_1_calibration")
const V21_CALIBRATION_TABLE = joinpath(V21_CALIBRATION_DIR, "v2_1_calibration_replay.csv")
const V21_CALIBRATION_POINT = joinpath(V21_CALIBRATION_DIR, "operational_v2_calibration.csv")
const V21_CALIBRATION_SPLIT = joinpath(V21_CALIBRATION_DIR, "operational_v2_calibration_split.csv")
const V21_CALIBRATION_SOURCE_AUDIT = joinpath(V21_CALIBRATION_DIR, "v2_1_calibration_source_audit.csv")

function _v21_driver_lookup(plasma::DataFrame, mag::DataFrame)
    nrow(plasma) == nrow(mag) || throw(DimensionMismatch(
        "plasma and magnetic replay frames must have equal rows",
    ))
    plasma.time_tag == mag.time_tag || throw(ArgumentError(
        "plasma and magnetic replay frames must have identical timestamps",
    ))
    lookup = Dict{DateTime,NamedTuple}()
    for i in 1:nrow(plasma)
        driver = (
            V=Float64(plasma.speed[i]),
            Bz=Float64(mag.bz_gsm[i]),
            By=Float64(mag.by_gsm[i]),
            n=Float64(plasma.density[i]),
            Pdyn=dynamic_pressure(Float64(plasma.density[i]), Float64(plasma.speed[i])),
        )
        all(isfinite, values(driver)) || continue
        lookup[DateTime(plasma.time_tag[i])] = driver
    end
    return lookup
end

function _v21_baseline_trajectories(anchor_dst_star::Float64, drivers, n_steps::Int)
    burton = anchor_dst_star
    burton_full = anchor_dst_star
    obrien = anchor_dst_star
    b = Vector{Float64}(undef, n_steps)
    bf = similar(b)
    o = similar(b)
    for h in 1:n_steps
        burton = _advance_baselines(burton, drivers).burton
        burton_full = _advance_baselines(burton_full, drivers).burton_full
        obrien = _advance_baselines(obrien, drivers).obrien
        b[h] = burton
        bf[h] = burton_full
        o[h] = obrien
    end
    return (; burton=b, burton_full=bf, obrien=o)
end

"""
    build_v2_1_calibration_table(plasma, mag, dst_times, dst_values; horizons)

Create the causal multi-horizon table used to fit V2.1 calibration. The source
driver for anchor `t` is the complete archived hour `[t-1h,t)`. Forecast drivers
remain fixed within this calibration experiment so the residual layer is fitted
to the declared core forecast independently of the operational tail.
"""
function build_v2_1_calibration_table(
    plasma::DataFrame,
    mag::DataFrame,
    dst_times,
    dst_values;
    horizons::Vector{Int}=copy(V21_CALIBRATION_HORIZONS),
    core::OperationalCore=load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION),
)
    isempty(horizons) && throw(ArgumentError("calibration horizons must not be empty"))
    any(<=(0), horizons) && throw(ArgumentError("calibration horizons must be positive"))
    length(dst_times) == length(dst_values) || throw(DimensionMismatch(
        "Dst time/value vectors must have equal length",
    ))
    canonical_operational_version(core.artifacts.version) == OPERATIONAL_V2_1_MODEL_VERSION ||
        throw(ArgumentError("V2.1 calibration requires the current V2.1 core"))

    driver_at = _v21_driver_lookup(plasma, mag)
    dst_at = Dict{DateTime,Float64}()
    for (time, value) in zip(dst_times, dst_values)
        v = Float64(value)
        isfinite(v) && (dst_at[DateTime(time)] = v)
    end
    max_h = maximum(horizons)
    rows = NamedTuple[]
    for anchor_time in sort!(collect(keys(dst_at)))
        source_time = anchor_time - Hour(1)
        drivers = get(driver_at, source_time, nothing)
        drivers === nothing && continue
        latest_dst = dst_at[anchor_time]
        anchor_dst_star = dst_to_dst_star(latest_dst, drivers.Pdyn)
        core_star = operational_core_forecast(core, anchor_dst_star, drivers, max_h)
        baseline_star = _v21_baseline_trajectories(anchor_dst_star, drivers, max_h)
        for h in horizons
            target_time = anchor_time + Hour(h)
            observed = get(dst_at, target_time, NaN)
            isfinite(observed) || continue
            pred = dst_star_to_dst(core_star[h], drivers.Pdyn)
            burton = dst_star_to_dst(baseline_star.burton[h], drivers.Pdyn)
            burton_full = dst_star_to_dst(baseline_star.burton_full[h], drivers.Pdyn)
            obrien = dst_star_to_dst(baseline_star.obrien[h], drivers.Pdyn)
            push!(rows, (
                issue_time_utc=anchor_time,
                source_driver_start_utc=source_time,
                source_driver_end_utc=anchor_time,
                latest_dst_time_utc=anchor_time,
                target_time_utc=target_time,
                model_step_hours=h,
                model_version="v2.1_core_uncalibrated",
                operational_core_version=core.artifacts.version,
                operational_candidate_count=core.artifacts.candidate_count,
                operational_active_count=core.artifacts.active_count,
                latest_dst_nt=latest_dst,
                observation_dst_nt=observed,
                pred_dst_nt=pred,
                residual_dst_nt=observed - pred,
                pred_dst_ci05_nt=pred - V21_SEED_HALF_WIDTH_NT,
                pred_dst_ci95_nt=pred + V21_SEED_HALF_WIDTH_NT,
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred - V21_SEED_HALF_WIDTH_NT,
                v1_pred_dst_ci95_nt=pred + V21_SEED_HALF_WIDTH_NT,
                persistence_dst_nt=latest_dst,
                burton_dst_nt=burton,
                burton_full_dst_nt=burton_full,
                obrien_dst_nt=obrien,
                V_kms=drivers.V,
                Bz_nt=drivers.Bz,
                By_nt=drivers.By,
                n_cm3=drivers.n,
                Pdyn_npa=drivers.Pdyn,
                replay_note="causal_previous_complete_hour_fixed_driver_v2_1_core",
            ))
        end
    end
    isempty(rows) && error("No V2.1 calibration rows could be generated")
    out = DataFrame(rows)
    sort!(out, [:issue_time_utc, :model_step_hours])
    return out
end

function build_v2_1_calibration_table(
    omni_path::AbstractString=OPERATIONAL_OMNI;
    year_start::Int=V21_CALIBRATION_YEAR_START,
    year_end::Int=V21_CALIBRATION_YEAR_END,
    horizons::Vector{Int}=copy(V21_CALIBRATION_HORIZONS),
)
    year_start <= year_end || throw(ArgumentError("year_start must not exceed year_end"))
    plasma, mag, dst_times, dst_values =
        _omni_replay_inputs(String(omni_path), year_start, year_end)
    return build_v2_1_calibration_table(
        plasma, mag, dst_times, dst_values; horizons=horizons,
    )
end

function _v21_split_audit(scored::DataFrame)
    required = ("v2_split", "issue_time_utc", "target_time_utc", "model_step_hours")
    all(in(names(scored)), required) || throw(ArgumentError(
        "scored calibration table lacks split-audit columns",
    ))
    expected = ("fit", "validation", "holdout")
    sets = Dict{String,Set{DateTime}}()
    rows = NamedTuple[]
    for split in expected
        sub = scored[string.(scored.v2_split) .== split, :]
        nrow(sub) > 0 || error("V2.1 calibration split $split is empty")
        issues = DateTime.(_parse_dt.(sub.issue_time_utc))
        targets = DateTime.(_parse_dt.(sub.target_time_utc))
        sets[split] = Set(issues)
        push!(rows, (
            split=split,
            rows=nrow(sub),
            anchors=length(unique(issues)),
            minimum_issue_utc=minimum(issues),
            maximum_issue_utc=maximum(issues),
            minimum_target_utc=minimum(targets),
            maximum_target_utc=maximum(targets),
            minimum_horizon_hours=minimum(Int.(sub.model_step_hours)),
            maximum_horizon_hours=maximum(Int.(sub.model_step_hours)),
        ))
    end
    isempty(intersect(sets["fit"], sets["validation"])) || error(
        "fit and validation issue-time sets overlap",
    )
    isempty(intersect(sets["fit"], sets["holdout"])) || error(
        "fit and holdout issue-time sets overlap",
    )
    isempty(intersect(sets["validation"], sets["holdout"])) || error(
        "validation and holdout issue-time sets overlap",
    )
    rows[1].maximum_target_utc < rows[2].minimum_issue_utc || error(
        "validation issuance begins before all fit targets are observable",
    )
    rows[2].maximum_target_utc < rows[3].minimum_issue_utc || error(
        "holdout issuance begins before all validation targets are observable",
    )
    return DataFrame(rows)
end

function _v21_atomic_promote(source::AbstractString, target::AbstractString)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "promotion source must be a regular non-symlink file: $source",
    ))
    mkpath(dirname(target))
    tmp, io = mktemp(dirname(target); cleanup=false)
    close(io)
    try
        cp(source, tmp; force=true)
        _sync_file(tmp)
        _live_atomic_replace(tmp, target)
        _sync_parent_directory(target)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return target
end

function fit_and_promote_v2_1_calibration!(table_path::AbstractString=V21_CALIBRATION_TABLE)
    mkpath(V21_CALIBRATION_DIR)
    cfg = LiveVerifyConfig(
        mode=:fit_v2_calibration,
        model=:v2,
        table_path=String(table_path),
        v2_calibration_path=V21_CALIBRATION_POINT,
        v2_train_fraction=V21_CALIBRATION_TRAIN_FRACTION,
        v2_validation_fraction=V21_CALIBRATION_VALIDATION_FRACTION,
        v2_ridge_grid=Float64[0.0, 1.0, 10.0, 100.0, 1000.0],
        v2_interval_coverage=V21_CALIBRATION_COVERAGE,
        v2_coverage_floor=V21_CALIBRATION_COVERAGE_FLOOR,
    )
    cal = fit_v2_calibration!(cfg)
    startswith(cal.label, "operational_v2_1_") || error(
        "calibration label does not identify V2.1: $(cal.label)",
    )
    occursin("fallback", cal.label) && error(
        "V2.1 candidate did not clear the unchanged deployment gates",
    )

    selection_path = _selection_path(V21_CALIBRATION_POINT)
    selection = CSV.read(selection_path, DataFrame)
    any(Bool.(selection.deployed)) || error(
        "V2.1 selection audit contains no deployed candidate",
    )
    selected = selection[Bool.(selection.selected_by_validation), :]
    nrow(selected) == 1 || error("V2.1 selection audit must have exactly one selected candidate")
    Float64(selected.conformal_holdout_coverage[1]) >= V21_CALIBRATION_COVERAGE_FLOOR ||
        error("V2.1 conformal holdout coverage is below the unchanged floor")

    scored_path = replace(V21_CALIBRATION_POINT, r"\.csv$" => "_scored.csv")
    scored = CSV.read(scored_path, DataFrame)
    split_audit = _v21_split_audit(scored)
    split_audit[!, :point_calibration_sha256] =
        fill(bytes2hex(sha256(read(V21_CALIBRATION_POINT))), nrow(split_audit))
    split_audit[!, :source_table_sha256] =
        fill(bytes2hex(sha256(read(String(table_path)))), nrow(split_audit))
    split_audit[!, :conformal_holdout_coverage] =
        fill(Float64(selected.conformal_holdout_coverage[1]), nrow(split_audit))
    CSV.write(V21_CALIBRATION_SPLIT, split_audit)

    current = operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION)
    _v21_atomic_promote(V21_CALIBRATION_POINT, current.point_csv)
    _v21_atomic_promote(_conformal_path(V21_CALIBRATION_POINT), current.conformal_csv)
    _v21_atomic_promote(
        selection_path,
        joinpath(dirname(current.point_csv), "operational_v2_calibration_selection.csv"),
    )
    _v21_atomic_promote(
        V21_CALIBRATION_SPLIT,
        joinpath(dirname(current.point_csv), "operational_v2_calibration_split.csv"),
    )
    return cal, split_audit
end

function main_v2_1_calibration()
    mkpath(V21_CALIBRATION_DIR)
    started = time_ns()
    table = build_v2_1_calibration_table()
    CSV.write(V21_CALIBRATION_TABLE, table)
    source_audit = DataFrame(
        model_version=[OPERATIONAL_V2_1_MODEL_VERSION],
        candidate_count=[20],
        active_count=[11],
        year_start=[V21_CALIBRATION_YEAR_START],
        year_end=[V21_CALIBRATION_YEAR_END],
        horizons=[join(V21_CALIBRATION_HORIZONS, ";")],
        rows=[nrow(table)],
        anchors=[length(unique(table.issue_time_utc))],
        minimum_issue_utc=[minimum(table.issue_time_utc)],
        maximum_issue_utc=[maximum(table.issue_time_utc)],
        omni_sha256=[bytes2hex(sha256(read(OPERATIONAL_OMNI)))],
        coefficient_sha256=[bytes2hex(sha256(read(operational_core_artifacts().coefficients_csv)))],
        seed_half_width_nt=[V21_SEED_HALF_WIDTH_NT],
        generation_seconds=[(time_ns() - started) / 1e9],
    )
    CSV.write(V21_CALIBRATION_SOURCE_AUDIT, source_audit)
    cal, split = fit_and_promote_v2_1_calibration!()
    println("Operational V2.1 calibration promoted: $(cal.label)")
    println(split)
    return cal
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_1_calibration()
end
